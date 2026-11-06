"""Opt-in fixed-batch scheduling on the existing two-slot progressive runtime.

No CUDA work runs at import time. Constructing the runtime is an explicit GPU
operation and must follow the experiment launcher's exclusivity check. All Pool,
Lease, generation and Event operations remain in ProgressiveRuntime on its sole
Host. The batch has one frontier per Euler round: a layer becomes ready for all
lanes together, and every lane keeps its own old/new K/V within the two slots.
"""
from functools import partial

import numpy as np
import torch

from .progressive_kv import ProgressiveGraphs, ProgressiveRuntime
from .runtime_graph import input_tensors, layout


MODES = ("serial", "pipeline", "green")
SCHEDULES = ("serial", "pipeline")
MAX_ABS_ERROR = 1e-5
MAE_ERROR = 1e-6


class LabeledGraph:
    """Instance-local replay scope; never replace global torch methods.

    Collect with --cuda-graph-trace=node:host-only:nvtx-live to retain individual
    kernels and project this complete runtime invocation onto the GPU timeline.
    CPU NVTX duration describes launch submission, not GPU execution duration.
    """

    def __init__(self, graph, label, *, range_factory=None):
        if label not in ("encoder", "vlm", "AE"):
            raise ValueError("Expected encoder, vlm or AE graph label")
        self.graph, self.label = graph, label
        self.range_factory = range_factory or torch.cuda.nvtx.range

    def replay(self):
        with self.range_factory(self.label):
            return self.graph.replay()

    def __getattr__(self, name):
        return getattr(self.graph, name)


class BatchProgressiveGraphs(ProgressiveGraphs):
    """Same captures and buffers, with runtime-only outer replay labels."""

    def __init__(self, backend, prepared, noise, *, profile=False, streams=None):
        # No detailed capture-time hooks: the three replay envelopes are the
        # intended application ranges. No global NVTX/CUDAGraph monkeypatch.
        super().__init__(backend, prepared, noise, profile=False, streams=streams)
        for slot in self.slots:
            slot.vision_graph = LabeledGraph(slot.vision_graph, "encoder")
            slot.prefix_graph = LabeledGraph(slot.prefix_graph, "vlm")
            slot.ae_graph = LabeledGraph(slot.ae_graph, "AE")


def _batch_comparison(backend, prepared, reference, actual):
    """Require every lane, including decoded gripper and action prefix, to pass."""
    from .layer_bench import difference, within_tolerance

    reference, actual = np.asarray(reference), np.asarray(actual)
    expected = (backend.batch_size, backend.model.config.action_horizon,
                backend.model.config.action_dim)
    if reference.shape != expected or actual.shape != expected:
        raise ValueError(f"Normalized batch outputs must both have shape {expected}")
    decoded_reference = np.asarray(backend.output_batch(prepared, reference))
    decoded_actual = np.asarray(backend.output_batch(prepared, actual))
    if (decoded_reference.ndim != 3 or decoded_reference.shape != decoded_actual.shape
            or decoded_reference.shape[:2] != expected[:2]):
        raise ValueError("Decoded batch outputs lost batch or action dimensions")
    normalized = difference(reference, actual)
    decoded = difference(decoded_reference, decoded_actual)
    lanes = []
    for index, lane_id in enumerate(backend.contract.lane_ids):
        raw = difference(reference[index], actual[index])
        action = difference(decoded_reference[index], decoded_actual[index])
        lanes.append(dict(lane=index, lane_id=lane_id, normalized=raw, decoded=action,
            passed=within_tolerance(raw, MAX_ABS_ERROR, MAE_ERROR)
                   and within_tolerance(action, MAX_ABS_ERROR, MAE_ERROR)))
    return dict(**normalized, normalized=normalized, decoded=decoded, per_lane=lanes,
                passed=all(lane["passed"] for lane in lanes),
                tolerance=dict(max_abs=MAX_ABS_ERROR, mae=MAE_ERROR))


class BatchProgressiveRuntime(ProgressiveRuntime):
    """One fixed batch, two full KV slots, and original Host-owned Pool.

    ``mode`` selects resources at construction: serial/pipeline use ordinary
    full-GPU streams; green binds AE to the first Green Context group and both
    Encoder/VLM to the disjoint remainder. Request ``schedule`` can alternate
    serial/pipeline on the same captures without recapture. A serial request
    waits for all current K/V and disables next-Encoder overlap, while preserving
    the same CPU input lookahead if supplied. Green serial is only a restricted
    resource diagnostic, never the full-GPU serial baseline.

    All public action arrays retain [B,H,D], including B=1. Changing B, shapes,
    dtype, device or lane order requires a separate runtime. No per-lane reset.
    close() drains/destroys graphs and returns the owned GreenContextPair, if
    any. The caller must free its input/noise/model tensors and CUDA workspaces
    before closing that returned resource. Their allocator record_stream marks
    can outlive graph destruction and must never refer to destroyed streams.
    """

    def __init__(self, backend, prepared, noise, *, mode="pipeline", ae_sm_min=16,
                 green_library=None, profile=False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        backend.validate_prepared(prepared)
        backend.validate_noise(noise)
        self.batch_backend = backend
        self.mode = mode
        self.batch_size = backend.batch_size
        self.lane_ids = tuple(backend.contract.lane_ids)
        self.signature = layout(prepared, noise)
        self._noise_template = noise
        self._runtime_batch_identity = backend.contract.identity
        self.green_pair = None
        self._closed = False
        self.resource_layout = dict(policy="full_gpu", ae="ordinary_stream",
                                    vision="ordinary_stream", vlm="ordinary_stream")
        streams = None
        if mode == "green":
            if green_library is None:
                raise ValueError("green mode requires an explicitly prepared green_library")
            if isinstance(ae_sm_min, bool) or not isinstance(ae_sm_min, int) or ae_sm_min <= 0:
                raise ValueError("ae_sm_min must be a positive integer")
            from .green_context import GreenContextPair
            self.green_pair = GreenContextPair(green_library, first_sm_min=ae_sm_min,
                                               device=backend.device)
            ae, prefix = self.green_pair.streams
            streams = (prefix, prefix, ae)
            self.resource_layout = dict(policy="green_disjoint", branch_order=["ae", "prefix"],
                actual_ae_sms=self.green_pair.info["sm_counts"][0],
                actual_prefix_sms=self.green_pair.info["sm_counts"][1],
                vision="prefix_partition", vlm="prefix_partition", green=dict(self.green_pair.info))
        try:
            factory = partial(BatchProgressiveGraphs, streams=streams)
            super().__init__(backend, prepared, noise, profile=profile, graphs_factory=factory)
        except BaseException:
            # A failed capture traceback can still own CUDA tensors. Destroying
            # their Green streams here would make later allocator cleanup abort.
            # Failed setup is terminal for this experiment process; its CUDA
            # resources stay alive until process teardown.
            raise

    def _validate_input(self, prepared, noise=None, *, permit_new_epoch=False):
        self.backend.validate_prepared(prepared)
        if tuple(self.backend.contract.lane_ids) != self.lane_ids:
            raise ValueError("Runtime lane order changed")
        if not permit_new_epoch and self.backend.contract.identity != self._runtime_batch_identity:
            raise ValueError("Whole-batch epoch changed; explicitly reset the runtime first")
        noise = self._noise_template if noise is None else noise
        self.backend.validate_noise(noise)
        if layout(prepared, noise) != self.signature:
            raise ValueError("Batch graph input shape/dtype/device changed; construct another runtime")

    def bootstrap(self, prepared):
        self._validate_input(prepared)
        return super().bootstrap(prepared)

    def reset(self, prepared):
        self._validate_input(prepared, permit_new_epoch=True)
        # Base reset drains pending vision tickets and all readers before epoch
        # advancement and bootstrap. Both lanes and both slots reset together.
        previous = self._runtime_batch_identity
        self._runtime_batch_identity = self.backend.contract.identity
        try:
            return super().reset(prepared)
        except BaseException:
            self._runtime_batch_identity = previous
            raise

    def reset_batch(self, raws, *, lane_ids=None):
        """Convenience entry: drain -> whole-batch epoch -> prepare -> reset."""
        self.backend.contract.validate(lane_ids)
        raws = tuple(raws)
        if len(raws) != self.batch_size:
            raise ValueError("Whole-batch reset requires one observation per lane")
        self.drain()
        self.backend.contract.reset(lane_ids)
        prepared = self.backend.prepare_batch(raws, lane_ids=lane_ids)
        self.reset(prepared)
        return prepared

    def download_actions(self, actions):
        expected = (self.batch_size, self.backend.model.config.action_horizon,
                    self.backend.model.config.action_dim)
        if tuple(actions.shape) != expected:
            raise ValueError(f"Graph output must preserve full batch shape {expected}")
        return actions.detach().cpu().numpy().copy()

    def verification_summary(self, prepared, references, output, actual_steps):
        result = {name: _batch_comparison(self.backend, prepared, reference, output)
                  for name, reference in references.items()}
        result.update(actual_steps=actual_steps,
                      passed=result["matched_schedule"]["passed"],
                      scope="Same observed per-round KV frontier, every lane normalized and decoded; no additional KV allocation")
        return result

    def request(self, prepared, noise, *, lead=3, force_old=False, lookahead=None,
                next_encoder=False, verify=False, collect_events=True, schedule=None):
        self._validate_input(prepared, noise)
        schedule = schedule or ("serial" if self.mode == "serial" else "pipeline")
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}")
        requested_lead = lead
        if schedule == "serial":
            if force_old:
                raise ValueError("Fresh serial control cannot force old KV")
            lead, next_encoder = self.graphs.num_layers, False
        def checked_lookahead():
            value = lookahead()
            if value is not None:
                self._validate_input(value)
            return value
        # Release/poll in the base request may clear a retired observation.
        # Snapshot only scalar Host metadata, including a previous producer
        # still BUILDING; never mistake its ahead-vision ticket for published KV.
        prior_observations = {
            index: (slot.generation, slot.observation.observation_id)
            for index, slot in enumerate(self.pool.slots) if slot.observation is not None
        }
        output, row, future = super().request(prepared, noise, lead=lead, force_old=force_old,
            lookahead=checked_lookahead if lookahead is not None else None,
            next_encoder=next_encoder, verify=verify, collect_events=collect_events)
        old_slot, old_generation = row["old_slot"], row["old_generation"]
        previous = prior_observations.get(old_slot)
        if (previous is None or previous[0] != old_generation
                or self.pool.slots[old_slot].generation != old_generation):
            raise RuntimeError("Old KV observation metadata does not match the consumed slot generation")
        old_observation = previous[1]
        age = row["request"] - old_observation
        if age < 0:
            raise RuntimeError("Old KV observation follows the current request")
        row.update(batch_size=self.batch_size, lane_ids=list(self.lane_ids),
                   batch_epoch=self.backend.contract.epoch, schedule=schedule,
                   requested_lead=requested_lead, resource_layout=self.resource_layout,
                   frontier_scope="whole_batch_per_round", graph_recaptures=0, full_kv_slots=2,
                   num_vlm_layers=self.graphs.num_layers,
                   old_kv_observation_id=old_observation, old_kv_age_requests=age,
                   old_kv_age_scope="request_generations_before_lease_release")
        return output, row, future

    @torch.no_grad()
    def update_inputs_inplace(self, target, source):
        """Validation entry for updated contents at identical Tensor addresses.

        The scheduler first drains all work, then updates caller-owned input
        tensors. Slot K/V is never copied here. No inference result is cached by
        input address. The next request stages the new values into its reserved
        slot using the original encode/generate protocol.
        """
        self._validate_input(target)
        self._validate_input(source)
        targets, sources = input_tensors(target), input_tensors(source)
        if len(targets) != len(sources):
            raise ValueError("Prepared input structure changed")
        self.drain()
        addresses = tuple(value.data_ptr() for value in targets)
        for destination, value in zip(targets, sources, strict=True):
            if value.device.type == "cuda":
                value.record_stream(torch.cuda.current_stream(value.device))
            destination.copy_(value, non_blocking=True)
        if addresses != tuple(value.data_ptr() for value in targets):
            raise RuntimeError("In-place validation unexpectedly replaced input addresses")
        return target

    def finish_record(self, record):
        events = record.get("_events")
        if events is None:
            return record
        # Preserve handles before base consumes them. These are GPU timeline
        # spans from Encoder begin, which can precede this request's Host
        # admission when its Encoder was submitted by the preceding request.
        super().finish_record(record)
        origin = events["vision_begin"]
        prefill = origin.elapsed_time(events["prefix_done"])
        ae = origin.elapsed_time(events["ae_done"])
        record.update(prefill_completion_event_ms=prefill, ae_completion_event_ms=ae,
            both_completion_event_ms=max(prefill, ae),
            completion_event_origin="current_encoder_gpu_begin; may precede current Host admission; spans include waits and contention")
        end = max(prefill, ae)
        if "next_vision_done" in events:
            ahead = origin.elapsed_time(events["next_vision_done"])
            record["next_encoder_completion_event_ms"] = ahead
            end = max(end, ahead)
        record["all_submitted_work_completion_event_ms"] = end
        return record

    def close(self):
        if self._closed:
            return None
        # If the drain fails, keep contexts alive for explicit caller recovery.
        # Never destroy their streams while a graph or reader may still be live.
        super().close()
        resources, self.green_pair = self.green_pair, None
        self._closed = True
        return resources
