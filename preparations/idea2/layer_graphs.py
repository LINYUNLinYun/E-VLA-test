"""Experimental same-frame KV-ready graphs; original Gemma math is unchanged.

All Python hooks exist only during cold capture on the scheduler host. Each
slot owns its KV, three private graph pools, and distinct AE writable inputs.
Producer graphs MUST be launched before consumers for that generation.
"""
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import dataclass
import gc
import time

import torch

from .cewp_segmented import embeddings, language_prefix
from .external_events import ExternalEvents, initialized_events
from .openpi_backend import Prefix, Prepared, cache_tensors
from .runtime_graph import DynamicCore, input_tensors, layout, stage_tensor


@contextmanager
def capture_labels(model, stage):
    """Node labels captured by nsys node:nvtx-precapture; no new GPU work."""
    from .cewp_profile_labels import labeled_method
    pair = model.paligemma_with_expert
    with ExitStack() as stack:
        if stage == "vision":
            stack.enter_context(labeled_method(pair.paligemma.model.vision_tower,
                                               "forward", "Prefill/VisionEncoder"))
            stack.enter_context(labeled_method(pair.paligemma.model.multi_modal_projector,
                                               "forward", "Prefill/VisionProjector"))
        else:
            layers = (pair.paligemma.language_model.layers if stage == "vlm"
                      else pair.gemma_expert.model.layers)
            branch = "Prefill/VLM" if stage == "vlm" else "AE"
            for index, layer in enumerate(layers):
                name = f"{branch}/layer_{index:02d}"
                stack.enter_context(labeled_method(layer, "forward", name))
                stack.enter_context(labeled_method(layer.self_attn, "forward", f"{name}/Attention"))
                stack.enter_context(labeled_method(layer.mlp, "forward", f"{name}/FFN"))
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    stack.enter_context(labeled_method(getattr(layer.self_attn, projection),
                                                       "forward", f"{name}/{projection}"))
        yield


class WaitingCache:
    """Read-only cache view used only while capturing first-iteration attention."""
    def __init__(self, cache, ready, external, stream):
        self.cache, self.ready = cache, ready
        self.external, self.stream = external, stream
        self.waited = set()

    def __getattr__(self, name):
        return getattr(self.cache, name)

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, index):
        if index not in self.waited:
            self.external.wait(self.ready[index], self.stream)
            self.waited.add(index)
        return self.cache[index]

    def update(self, *args, **kwargs):
        raise RuntimeError("AE must not mutate its prefix KV")


class TwoIterationGraphs:
    """Cold-capture split after AE1; graph two contains all remaining steps.

No Python denoising logic runs during replay. Separate graphs preserve each
step's prepared condition addresses; blindly replaying step 1 as step 2 would
instead use the wrong timestep/modulation. Each graph has its own pool.
"""
    def __init__(self, stream, tail_stream=None):
        self.stream = stream
        self.tail_stream = stream if tail_stream is None else tail_stream
        self.graphs = (torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph())
        self.handoff = torch.cuda.Event(enable_timing=True)
        self.index = 0
        self.context = None

    def __enter__(self):
        self.context = torch.cuda.graph(self.graphs[0], stream=self.stream)
        self.context.__enter__()
        return self

    def next_iteration(self):
        if self.index != 0:
            raise RuntimeError("Exactly two iteration graphs are allowed")
        context, self.context = self.context, None
        context.__exit__(None, None, None)
        # Cold setup only: initialize first-step outputs used by second capture.
        self.graphs[0].replay()
        self.stream.synchronize()
        self.index = 1
        self.context = torch.cuda.graph(self.graphs[1], stream=self.tail_stream)
        self.context.__enter__()

    def __exit__(self, *exc):
        if self.context is not None:
            context, self.context = self.context, None
            context.__exit__(*exc)
        if exc[0] is None and self.index != 1:
            raise RuntimeError("Second iteration was not captured")

    def replay(self, *, prefix_done=None):
        # These waits are GPU operations, not Host polling. Keep the same
        # handoff operations for full->full and partition->full controls.
        with torch.cuda.stream(self.stream):
            self.graphs[0].replay()
            self.handoff.record(self.stream)
        with torch.cuda.stream(self.tail_stream):
            self.tail_stream.wait_event(self.handoff)
            if prefix_done is not None:
                self.tail_stream.wait_event(prefix_done)
            self.graphs[1].replay()


@contextmanager
def record_layer_ready(model, events, external, stream, boundary, group_size):
    """Record after original cache.update, or after the entire decoder block."""
    from transformers.cache_utils import DynamicCache
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    calls, hooks = [], []
    original = DynamicCache.update

    def mark(index):
        calls.append(index)
        if (index + 1) % group_size == 0 or index == len(events) - 1:
            first = index - (index % group_size)
            for member in range(first, index + 1):
                external.mark(events[member], stream)

    def update(cache, key, value, layer_idx, cache_kwargs=None):
        result = original(cache, key, value, layer_idx, cache_kwargs)
        mark(layer_idx)
        return result

    try:
        if boundary == "kv_ready":
            DynamicCache.update = update
        elif boundary == "block_ready":
            for index, layer in enumerate(layers):
                hooks.append(layer.register_forward_hook(
                    lambda module, inputs, output, index=index: mark(index)))
        else:
            raise ValueError("Unknown readiness boundary")
        yield calls
        if calls != list(range(len(events))):
            raise RuntimeError(f"Unexpected producer layer order: {calls}")
    finally:
        DynamicCache.update = original
        for hook in hooks:
            hook.remove()


@dataclass
class LayerGraphSlot:
    prepared: Prepared
    ae_prepared: Prepared
    noise: object
    core: object
    vision_graph: object
    embedding_values: object
    prefix_graph: object
    prefix: Prefix
    ae_graph: object
    actions: object
    metadata_ready: object
    layer_ready: tuple
    ae_marks: tuple
    first_actions: object
    encoded_identity: object = None
    producer_identity: object = None


class LayerGraphs:
    def __init__(self, backend, prepared, noise, prefix_stream, ae_stream,
                 *, boundary="kv_ready", group_size=1, profile_labels=False, ae_layout="whole",
                 tail_stream=None, tail_gate=False, capture_intermediates=False, vision_stream=None):
        if backend.num_steps not in (2, 10):
            raise ValueError("This experiment requires two or ten AE steps")
        if group_size not in (1, 2, 4):
            raise ValueError("Layer grouping must be 1, 2, or 4")
        if ae_layout not in ("whole", "per_step"):
            raise ValueError("AE graph layout must be whole or per_step")
        if (tail_stream is not None or tail_gate) and ae_layout != "per_step":
            raise ValueError("Tail resource switching requires split AE graphs")
        if tail_stream is not None and tail_stream != ae_stream and not tail_gate:
            raise ValueError("A new tail resource context must wait for full Prefix completion")
        self.ae_layout = ae_layout
        self.backend, self.prefix_stream, self.ae_stream = backend, prefix_stream, ae_stream
        self.vision_stream = prefix_stream if vision_stream is None else vision_stream
        self.completion_stream = ae_stream if tail_stream is None else tail_stream
        self.tail_gate = tail_gate
        # Diagnostic graphs only: time clones introduce extra nodes, so callers
        # must never mix these captures into production performance results.
        self.capture_intermediates = capture_intermediates
        self.signature = layout(prepared, noise)
        self.boundary, self.group_size = boundary, group_size
        self.profile_labels = profile_labels
        self.external, self.slots = ExternalEvents(), []
        begin = time.perf_counter()
        initial = torch.cuda.Event()
        initial.record()
        prefix_stream.wait_event(initial)
        if self.vision_stream is not prefix_stream:
            self.vision_stream.wait_event(initial)
        ae_stream.wait_event(initial)
        self.completion_stream.wait_event(initial)
        try:
            for _ in range(2):
                self.slots.append(self._capture(prepared, noise))
            self.addresses = [tuple(x.data_ptr() for x in cache_tensors(s.prefix.cache))
                              for s in self.slots]
            if set(self.addresses[0]) & set(self.addresses[1]):
                raise RuntimeError("KV slots alias")
            pools = [graph.pool() for s in self.slots
                     for graph in (s.vision_graph, s.prefix_graph,
                                   *getattr(s.ae_graph, "graphs", (s.ae_graph,)))]
            if len(set(pools)) != len(pools):
                raise RuntimeError("Concurrent graphs require private memory pools")
            self.setup_ms = (time.perf_counter() - begin) * 1000
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def _capture(self, prepared, noise):
        pstream, astream = self.prefix_stream, self.ae_stream
        vstream = self.vision_stream
        with torch.cuda.stream(vstream):
            static = Prepared(None, [x.clone() for x in prepared.images],
                [x.clone() for x in prepared.image_masks], prepared.tokens.clone(),
                prepared.token_masks.clone(), prepared.state.clone())
            # Only AE writes this state/noise. Image/token tensors are not read by AE.
            ae_static = Prepared(None, static.images, static.image_masks, static.tokens,
                                 static.token_masks, prepared.state.clone())
            static_noise = noise.clone()
            core = getattr(self.backend, "graph_core_class", DynamicCore)(self.backend, ae_static)
            count = len(core.model.paligemma_with_expert.paligemma.language_model.layers)
            ready = tuple(initialized_events(vstream, count))
            metadata, = initialized_events(vstream, 1)
            marks = tuple(initialized_events(vstream, 3))
            for _ in range(2):
                temporary = embeddings(core, static)
                vstream.synchronize()
                del temporary
            vision_graph = torch.cuda.CUDAGraph()
            labels = capture_labels(core.model, "vision") if self.profile_labels else nullcontext()
            with labels, torch.cuda.graph(vision_graph, stream=vstream):
                with torch.cuda.nvtx.range("Prefill/VisionEncoder+Projector+Metadata"):
                    values = embeddings(core, static)
            vision_graph.replay()
            if vstream is not pstream:
                # Cold capture of full-resource VLM must see initialized
                # Encoder outputs produced on its separate partition.
                captured_vision_ready = torch.cuda.Event()
                captured_vision_ready.record(vstream)
                pstream.wait_event(captured_vision_ready)
        with torch.cuda.stream(pstream):
            for _ in range(2):
                temporary = language_prefix(core, values)
                pstream.synchronize()
                del temporary
            gc.collect()
            prefix_graph = torch.cuda.CUDAGraph()
            labels = capture_labels(core.model, "vlm") if self.profile_labels else nullcontext()
            with labels, record_layer_ready(core.model, ready, self.external, pstream,
                                    self.boundary, self.group_size):
                with torch.cuda.graph(prefix_graph, stream=pstream):
                    with torch.cuda.nvtx.range("Prefill/VLM"):
                        prefix = language_prefix(core, values)
            prefix_graph.replay()
        pstream.synchronize()
        with torch.cuda.stream(astream):
            for _ in range(2):
                # Cold-only schedule evidence; no diagnostic clone kernels in
                # captured production graphs or timed replay.
                schedule_nodes = []
                temporary = core.generate(prefix, ae_static, static_noise,
                                          time_nodes=schedule_nodes)
                astream.synchronize()
                del temporary
            schedule_times = [float(t.reshape(-1)[0].cpu()) for t in schedule_nodes]
            condition_times = ([float(t.reshape(-1)[0].cpu()) for t in core.schedule_values]
                               if hasattr(core, "schedule_values") else None)
            del schedule_nodes
            waiting = WaitingCache(prefix.cache, ready, self.external, astream)
            gated_prefix = Prefix(waiting, prefix.pad_masks, prefix.position_ids)
            original_velocity = core.velocity
            iterations = []
            intermediates = []
            iteration_inputs, iteration_velocities, captured_times = [], [], []
            ae_graph = (TwoIterationGraphs(astream, self.completion_stream) if self.ae_layout == "per_step"
                        else torch.cuda.CUDAGraph())

            def velocity(cache, current, actions, timestep):
                step = len(iterations)
                if step == 1:
                    # Previous Euler update and time update precede this marker.
                    self.external.mark(marks[1], astream)
                    intermediates.append(actions)
                    if isinstance(ae_graph, TwoIterationGraphs):
                        ae_graph.next_iteration()
                iterations.append(step)
                if self.capture_intermediates:
                    iteration_inputs.append(actions)
                with torch.cuda.nvtx.range(f"AE/iteration_{step + 1}"):
                    value = original_velocity(gated_prefix if step == 0 else cache,
                                              current, actions, timestep)
                if self.capture_intermediates:
                    iteration_velocities.append(value)
                return value

            try:
                core.velocity = velocity
                labels = capture_labels(core.model, "ae") if self.profile_labels else nullcontext()
                capture = (ae_graph if isinstance(ae_graph, TwoIterationGraphs)
                           else torch.cuda.graph(ae_graph, stream=astream))
                with labels, capture:
                    with torch.cuda.nvtx.range("AE/Denoising[same_frame_KV]"):
                        self.external.mark(marks[0], astream)
                        actions = core.generate(prefix, ae_static, static_noise,
                            time_nodes=captured_times if self.capture_intermediates else None)
                        self.external.mark(marks[2], self.completion_stream)
            finally:
                core.velocity = original_velocity
            if iterations != list(range(self.backend.num_steps)) or waiting.waited != set(range(count)):
                raise RuntimeError("AE capture did not consume every layer exactly in first step")
            ae_graph.replay()
        astream.synchronize()
        self.completion_stream.synchronize()
        slot = LayerGraphSlot(static, ae_static, static_noise, core, vision_graph, values,
                              prefix_graph, prefix, ae_graph, actions, metadata, ready, marks,
                              intermediates[0])
        slot.schedule_times = schedule_times
        slot.condition_times = condition_times
        slot.actual_nfe = len(iterations)
        slot.iteration_actions = iteration_inputs[1:] + [actions] if self.capture_intermediates else []
        slot.iteration_velocities = iteration_velocities
        slot.captured_times = captured_times
        return slot

    def compatible(self, prepared, noise):
        return self.signature == layout(prepared, noise)

    def _exclusive(self, pool, identity):
        owner = pool._slot(identity)
        if owner.state != "BUILDING" or owner.ready is not None or owner.leases or owner.readers:
            raise RuntimeError("Producer requires an exclusive BUILDING reservation")
        return self.slots[identity.slot]

    @staticmethod
    def _submit_label(kind, pool, identity):
        frame = pool._slot(identity).observation.observation_id
        return (f"submit/{kind}/frame={frame}/epoch={identity.epoch}"
                f"/slot={identity.slot}/gen={identity.generation}")

    @torch.no_grad()
    def encode(self, pool, identity, prepared, *, dependency=None):
        slot = self._exclusive(pool, identity)
        if slot.encoded_identity == identity:
            raise RuntimeError("Encoder already submitted for this generation")
        with torch.cuda.stream(self.vision_stream):
            if dependency is not None:
                dependencies = dependency if isinstance(dependency, (tuple, list)) else (dependency,)
                for event in dependencies:
                    self.vision_stream.wait_event(event)
            for target, source in zip(input_tensors(slot.prepared), input_tensors(prepared), strict=True):
                stage_tensor(target, source)
            with torch.cuda.nvtx.range(self._submit_label("vision", pool, identity)):
                slot.vision_graph.replay()
            slot.metadata_ready.record(self.vision_stream)
        slot.encoded_identity = identity
        return slot.metadata_ready

    @torch.no_grad()
    def prefill(self, pool, identity, *, dependency=None):
        slot = self._exclusive(pool, identity)
        if slot.encoded_identity != identity or slot.producer_identity == identity:
            raise RuntimeError("VLM requires exactly one encoder submission for this generation")
        with torch.cuda.stream(self.prefix_stream):
            if self.vision_stream is not self.prefix_stream:
                self.prefix_stream.wait_event(slot.metadata_ready)
            if dependency is not None:
                self.prefix_stream.wait_event(dependency)
            with torch.cuda.nvtx.range(self._submit_label("vlm", pool, identity)):
                slot.prefix_graph.replay()
            done = torch.cuda.Event(enable_timing=True)
            done.record(self.prefix_stream)
        slot.producer_identity = identity
        return slot.prefix, done, slot.layer_ready, slot.metadata_ready

    @torch.no_grad()
    def generate(self, pool, lease, prepared, noise, *, input_ready, serial=False):
        slot = self.slots[lease.identity.slot]
        submission = pool.validate_replay(lease, layer_events=slot.layer_ready,
                                          metadata_event=slot.metadata_ready)
        if slot.producer_identity != lease.identity or lease.value is not slot.prefix:
            raise RuntimeError("AE graph does not match the submitted KV generation")
        try:
            with torch.cuda.stream(self.ae_stream):
                # Next-frame vision may have run before this noise/state was made.
                # record_stream() protects storage lifetime, not producer ordering.
                self.ae_stream.wait_event(input_ready)
                self.ae_stream.wait_event(slot.metadata_ready)
                if serial:
                    self.ae_stream.wait_event(submission.producer_done)
                stage_tensor(slot.ae_prepared.state, prepared.state)
                stage_tensor(slot.noise, noise)
                with torch.cuda.nvtx.range(self._submit_label("ae", pool, lease.identity)):
                    if isinstance(slot.ae_graph, TwoIterationGraphs):
                        slot.ae_graph.replay(prefix_done=submission.producer_done if self.tail_gate else None)
                    else:
                        slot.ae_graph.replay()
            with torch.cuda.stream(self.completion_stream):
                actions = slot.actions.clone()
                done = torch.cuda.Event(enable_timing=True)
                done.record(self.completion_stream)
        except BaseException:
            # Only on submission failure: an exception before enqueueing the
            # cross-stream handoff must not release a Slot while AE1 still reads.
            self.ae_stream.synchronize()
            self.completion_stream.synchronize()
            raise
        return actions, done, slot.ae_marks

    def close(self):
        if self.vision_stream is not self.prefix_stream:
            self.vision_stream.synchronize()
        self.prefix_stream.synchronize()
        self.ae_stream.synchronize()
        self.completion_stream.synchronize()
        self.slots.clear()
