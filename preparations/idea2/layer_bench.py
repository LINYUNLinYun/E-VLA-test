"""Two/ten-step same-frame layer pipeline: matched serial/overlap experiments.

GPU exclusivity is checked before importing torch or initializing CUDA. Model
oracle outputs are prepared first and retained only on CPU; the subsequent
runtime owns exactly two complete KV allocations. nsys and performance runs
must be separate. --iteration-timing is diagnostic: reading reused graph
markers occurs after request timing but contributes to trajectory wall time.
CUDA Event spans include stream waits, submission gaps, and resource contention;
they are not measurements of kernel busy time. In particular AE span includes
metadata readiness and, in serial mode, the full-prefix completion wait.
"""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import subprocess
import sys
import time
import traceback

import numpy as np


def smi_rows(query):
    result = subprocess.run(["nvidia-smi", query, "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    return [[field.strip() for field in row] for row in csv.reader(result.stdout.splitlines()) if row]


def gpu_snapshot(uuid):
    from .gpu_guard import graphics_processes

    gpus = smi_rows("--query-gpu=uuid,name,memory.used,utilization.gpu")
    matches = [row for row in gpus if row[0] == uuid]
    if len(matches) != 1:
        raise RuntimeError("--gpu-uuid must exactly identify one visible physical GPU")
    apps = smi_rows("--query-compute-apps=pid,gpu_uuid,used_gpu_memory")
    return dict(uuid=uuid, name=matches[0][1], memory_used_mib=float(matches[0][2]),
                utilization_percent=float(matches[0][3]),
                processes=[dict(pid=int(row[0]), memory_mib=row[2]) for row in apps if row[1] == uuid],
                graphics_processes=graphics_processes(uuid),
                host_time=time.time())


def require_idle_gpu(uuid):
    snapshot = gpu_snapshot(uuid)
    if (snapshot["processes"] or snapshot["graphics_processes"]
            or snapshot["memory_used_mib"] > 256 or snapshot["utilization_percent"] > 1):
        raise RuntimeError("GPU is not exclusive/idle; refusing experiment: " + json.dumps(snapshot))
    # Set before importing CUDA libraries; never silently map a different GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return snapshot


def require_same_processes(uuid, expected):
    snapshot = gpu_snapshot(uuid)
    if snapshot["graphics_processes"] or {row["pid"] for row in snapshot["processes"]} != set(expected):
        raise RuntimeError("GPU process occupancy changed during experiment: " + json.dumps(snapshot))
    return snapshot


def statistics(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return dict(n=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))


def difference(reference, actual):
    left, right = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    if left.shape != right.shape:
        raise RuntimeError(f"Output shape differs: {left.shape} vs {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return dict(mae=None, max_abs=None, finite=False)
    delta = np.abs(left - right)
    return dict(mae=float(delta.mean()), max_abs=float(delta.max()),
                finite=bool(np.isfinite(left).all() and np.isfinite(right).all()))


def within_tolerance(error, max_error, mae_error):
    return error["finite"] and error["max_abs"] <= max_error and error["mae"] <= mae_error


def input_schedule(index, record_count):
    """Each changing observation has a stable noise seed shared by all arms."""
    frame = index % record_count
    return frame, 1701 + frame


def iteration_timing_fields(num_steps, marks=None):
    """The three marks delimit AE1 and its remaining iterations, not AE2 in D10."""
    if num_steps not in (2, 10):
        raise ValueError("Layer experiments support exactly 2 or 10 AE steps")
    tail = "ae_iteration2_event_ms" if num_steps == 2 else "ae_tail2to10_event_ms"
    fields = {"ae_iteration1_event_ms": None, tail: None}
    if marks is not None:
        if len(marks) != 3:
            raise RuntimeError("AE timing requires start, AE1-done, and final-done marks")
        fields["ae_iteration1_event_ms"] = marks[0].elapsed_time(marks[1])
        fields[tail] = marks[1].elapsed_time(marks[2])
    return fields


def validate_tail_policy(args):
    if args.tail_policy != "native" and args.ae_layout != "per_step":
        raise ValueError("wait_prefix/full tail policy requires --ae-layout per_step")
    if args.tail_policy == "full" and args.ae_sm_min <= 0:
        raise ValueError("full tail policy requires a disjoint Green Context pair via --ae-sm-min > 0")


def ae_completion_stream(graphs):
    return getattr(graphs, "completion_stream", graphs.ae_stream)


def record_reader_tail(torch, graphs):
    """Cover partial AE submission even if failure preceded the tail handoff."""
    submitted = torch.cuda.Event()
    submitted.record(graphs.ae_stream)
    completion = ae_completion_stream(graphs)
    if completion is graphs.ae_stream:
        return submitted
    completion.wait_event(submitted)
    tail = torch.cuda.Event()
    tail.record(completion)
    return tail


def native_schedule_reference(num_steps, device):
    """Independently materialize the native FP32 recurrence outside timing.

    In particular D10 nodes are accumulated CUDA FP32 values, not decimal
    linspace values or a Python FP64 calculation rounded afterward.
    """
    import torch

    if num_steps not in (2, 10):
        raise ValueError("Layer experiments support exactly 2 or 10 AE steps")
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    current = torch.tensor(1.0, dtype=torch.float32, device=device)
    nodes = []
    for _ in range(num_steps):
        nodes.append(current.clone())
        current += dt
    times = torch.stack(nodes).cpu().tolist()
    final_time, step_size, stop_threshold = (float(value.cpu()) for value in (current, dt, -dt / 2))
    if not all(value >= stop_threshold for value in times) or final_time >= stop_threshold:
        raise RuntimeError("Requested NFE disagrees with native while-loop termination")
    return dict(times=times, dt=step_size, final_time=final_time,
                stop_threshold=stop_threshold, actual_nfe=num_steps,
                dtype="torch.float32", device=str(device))


def validate_schedule(times, actual_nfe, expected, *, label):
    """Reject wrong NFE or any changed time value without a tolerance escape."""
    values = np.asarray(times, dtype=np.float64)
    reference = np.asarray(expected, dtype=np.float64)
    if (values.ndim != 1 or reference.ndim != 1 or not np.isfinite(values).all()
            or actual_nfe != len(reference) or len(values) != actual_nfe
            or not np.array_equal(values, reference)):
        raise RuntimeError(f"{label} changed native FP32 schedule/NFE: {times}, NFE={actual_nfe}; expected {expected}")


def audit_graph_schedules(graphs, oracle):
    """Compare cold-loop nodes and captured NFE; no replay diagnostic kernels."""
    expected = oracle[0]["times"]
    for row in oracle:
        validate_schedule(row["times"], row["actual_nfe"], expected, label="Oracle")
    slots = []
    for index, slot in enumerate(graphs.slots):
        validate_schedule(slot.schedule_times, slot.actual_nfe, expected, label=f"Graph slot {index}")
        if slot.condition_times is not None:
            validate_schedule(slot.condition_times, len(slot.condition_times), expected,
                              label=f"Prepared conditions slot {index}")
        slots.append(dict(slot=index, times=slot.schedule_times, captured_nfe=slot.actual_nfe,
                          condition_times=slot.condition_times))
    return dict(passed=True, expected_nfe=len(expected), slots=slots,
                scope="Time values from cold original-loop execution; NFE counted during capture. No diagnostic clones added to replay.")


def write_result(dest, result):
    temporary = dest / "result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(dest / "result.json")
    rows = result.get("rows", [])
    if rows:
        with (dest / "measurements.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(set().union(*(row.keys() for row in rows))))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()} for row in rows)


def capture_sources(dest):
    source = dest / "source"
    source.mkdir()
    hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        data = path.read_bytes()
        (source / path.name).write_bytes(data)
        hashes[str(path)] = hashlib.sha256(data).hexdigest()
    for path in (Path("policies/openpi/src/openpi/models_pytorch/pi0_pytorch.py"),
                 Path("policies/openpi/src/openpi/models_pytorch/gemma_pytorch.py"),
                 Path("policies/openpi/.venv/lib/python3.11/site-packages/transformers/models/gemma/modeling_gemma.py")):
        if path.exists():
            data = path.read_bytes()
            (source / (path.parent.name + "_" + path.name)).write_bytes(data)
            hashes[str(path)] = hashlib.sha256(data).hexdigest()
    return hashes


def eager_oracle(backend, records):
    """Before graph allocation, retain only CPU actions, first step, and hashes."""
    import torch
    from .experiments import native_raw
    from .graph_bench import cache_digest

    expected = native_schedule_reference(backend.num_steps, backend.device)
    outputs = []
    for index, record in enumerate(records):
        _, seed = input_schedule(index, len(records))
        prepared = backend.prepare(native_raw(record))
        noise = backend.noise(seed)
        prefix = backend.build_prefix(prepared)
        original = backend.velocity
        steps, first = [], []

        def velocity(selected, current, actions, timestep):
            if len(steps) == 1:
                first.append(actions.detach().clone())
            steps.append(timestep.detach().clone())
            return original(selected, current, actions, timestep)

        try:
            backend.velocity = velocity
            raw = backend.generate(prefix, prepared, noise)
        finally:
            backend.velocity = original
        times = [float(step.reshape(-1)[0].cpu()) for step in steps]
        validate_schedule(times, len(steps), expected["times"], label="Eager oracle")
        if any(step.dtype != torch.float32 for step in steps) or len(first) != 1:
            raise RuntimeError("Oracle changed the FP32 timestep dtype or first-update boundary")
        outputs.append(dict(index=index, seed=seed, normalized=raw[0].detach().cpu().numpy().copy(),
                            first_actions=first[0][0].detach().cpu().numpy().copy(),
                            decoded=np.asarray(backend.output(prepared, raw)).copy(),
                            kv_digest=cache_digest(prefix), times=times, actual_nfe=len(steps),
                            schedule_reference=expected))
        torch.cuda.current_stream(backend.device).synchronize()
        del prefix, prepared, noise, raw, first, steps
        print(f"Oracle {index + 1}/{len(records)}", flush=True)
    gc.collect()
    return outputs


class LayerRunner:
    def __init__(self, backend, graphs, records, *, iteration_timing=False):
        import torch
        from .layer_scheduler import LayerPool

        self.torch, self.backend, self.graphs, self.records = torch, backend, graphs, records
        self.pool = LayerPool(len(graphs.slots[0].layer_ready), task="layer_bench")
        self.iteration_timing = iteration_timing
        self.request = 0
        self.pending = []
        self.outputs = []
        self.reset_rows = []

    def reset(self, *, phase, mode, index):
        begin = time.perf_counter()
        self.pool.drain_reset(self.pool.epoch + 1, "layer_bench")
        row = dict(phase=phase, mode=mode, index=index, epoch=self.pool.epoch,
                   reset_ms=(time.perf_counter() - begin) * 1000)
        self.reset_rows.append(row)
        return row

    def _reserve(self, observation):
        self.pool.offer(observation)
        identity = self.pool.reserve()
        while identity is None:
            # Wait only on owned blocking work, never on the whole device.
            slots = sorted(self.pool.slots, key=lambda slot: slot.observation.observation_id
                           if slot.observation is not None else -1)
            unfinished = [event for slot in slots for event in
                          ([slot.ready] if slot.ready is not None else []) + slot.readers
                          if not event.query()]
            if not unfinished:
                raise RuntimeError("No free slot and no submitted completion fence can release one")
            unfinished[0].synchronize()
            self.pool.poll()
            identity = self.pool.reserve()
        return identity[0]

    def predict(self, index, mode, *, phase, pair=None, arm_order=None, verify=False):
        from .experiments import native_raw
        from .graph_bench import cache_digest
        from .openpi_backend import cache_tensors
        from .scheduler import Observation

        torch = self.torch
        frame, seed = input_schedule(index, len(self.records))
        serial = mode == "serial"
        if mode not in ("serial", "overlap"):
            raise ValueError("Unknown layer benchmark mode")
        start = time.perf_counter()
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        prepare_begin = time.perf_counter()
        prepared = self.backend.prepare(native_raw(self.records[frame]))
        prepare_end = time.perf_counter()
        noise = self.backend.noise(seed)
        noise_end = time.perf_counter()
        input_ready = torch.cuda.Event()
        input_ready.record()
        observation = Observation(self.request, self.pool.epoch, "layer_bench", prepared,
                                  host_receipt_time=start)
        self.request += 1
        reserve_begin = time.perf_counter()
        identity = self._reserve(observation)
        reserve_end = time.perf_counter()
        prefix_begin_event = torch.cuda.Event(enable_timing=True)
        vision_stream = getattr(self.graphs, "vision_stream", self.graphs.prefix_stream)
        with torch.cuda.nvtx.range(f"layer/{phase}/{mode}/frame_{frame}/slot_{identity.slot}/gen_{identity.generation}"):
            submit_begin = time.perf_counter()
            with torch.cuda.stream(vision_stream):
                vision_stream.wait_event(input_ready)
                prefix_begin_event.record(vision_stream)
            metadata = self.graphs.encode(self.pool, identity, prepared, dependency=input_ready)
            # A per-request marker, unlike the slot's reusable metadata event,
            # is safe to retain for deferred timing after many generations.
            vision_done = torch.cuda.Event(enable_timing=True)
            vision_done.record(vision_stream)
            self.pool.track_preparation(identity, metadata)
            prefix, prefix_done, layer_ready, metadata = self.graphs.prefill(self.pool, identity)
            self.pool.submitted_layers(identity, prefix, prefix_done, layer_ready, metadata_ready=metadata)
            submit_end = time.perf_counter()
            prefix_done_at_submit = prefix_done.query()
            lease = self.pool.acquire_layers(identity)
            ae_begin = time.perf_counter()
            try:
                ae_begin_event = torch.cuda.Event(enable_timing=True)
                ae_begin_event.record(self.graphs.ae_stream)
                raw, ae_done, marks = self.graphs.generate(self.pool, lease, prepared, noise,
                                                         input_ready=input_ready, serial=serial)
                ae_submit_end = time.perf_counter()
                ae_done_at_submit = ae_done.query()
                wait_begin = time.perf_counter()
                ae_done.synchronize()
                wait_end = time.perf_counter()
                # Same two D2H transfers and output transform as backend.output;
                # retain the raw CPU array for checks after the measured phase.
                state = prepared.state[0].detach().cpu().numpy()
                normalized = raw[0].detach().cpu().numpy()
                normalized_saved = normalized.copy()
                decoded = self.backend.policy._output_transform(dict(state=state, actions=normalized))["actions"]
                output_end = time.perf_counter()
                self.pool.release(lease, ae_done)
            except BaseException:
                if not lease.released:
                    reader_tail = record_reader_tail(torch, self.graphs)
                    self.pool.release(lease, reader_tail)
                raise
        end = time.perf_counter()
        row = dict(phase=phase, mode=mode, pair=pair, arm_order=arm_order, index=index,
                   frame=frame, seed=seed, request=observation.observation_id, epoch=identity.epoch,
                   slot=identity.slot, generation=identity.generation, observation_age=0,
                   cpu_cpu_ms=(end - start) * 1000, host_begin=start, host_end=end,
                   prepare_cpu_ms=(prepare_end - prepare_begin) * 1000,
                   noise_cpu_ms=(noise_end - prepare_end) * 1000,
                   output_cpu_ms=(output_end - wait_end) * 1000,
                   action_output_host_ms=(output_end - start) * 1000,
                   slot_wait_ms=(reserve_end - reserve_begin) * 1000,
                   prefill_cpu_submit_ms=(submit_end - submit_begin) * 1000,
                   ae_cpu_submit_ms=(ae_submit_end - ae_begin) * 1000,
                   ae_host_wait_ms=(wait_end - wait_begin) * 1000,
                   prefill_done_at_submit=prefix_done_at_submit, ae_done_at_submit=ae_done_at_submit,
                   prefill_done_at_response=prefix_done.query(),
                   live_snapshots=self.pool.live_snapshots, graph_recaptures=0,
                   num_steps=self.backend.num_steps,
                   **iteration_timing_fields(self.backend.num_steps))
        if self.iteration_timing:
            # These captured markers are reused at the next replay of this slot.
            # Never retain their references for deferred cross-generation reads.
            row.update(iteration_timing_fields(self.backend.num_steps, marks))
        self.pending.append((row, origin, prefix_begin_event, vision_done, prefix_done, ae_begin_event, ae_done))
        self.outputs.append((row, normalized_saved, np.asarray(decoded).copy()))
        if verify:
            # Explicit diagnostic only, outside request timing and formal runs.
            prefix_done.synchronize()
            slot = self.graphs.slots[identity.slot]
            row["first_actions"] = slot.first_actions[0].detach().cpu().numpy().tolist()
            row["kv_digest"] = cache_digest(prefix)
            row["kv_addresses"] = tuple(tensor.data_ptr() for tensor in cache_tensors(prefix.cache))
        return row

    def finish_events(self):
        for row, origin, prefix_begin, vision_done, prefix_done, ae_begin, ae_done in self.pending:
            prefix_done.synchronize()
            ae_done.synchronize()
            row["prefill_gpu_event_ms"] = prefix_begin.elapsed_time(prefix_done)
            row["vision_embedding_event_ms"] = prefix_begin.elapsed_time(vision_done)
            row["vlm_event_ms"] = vision_done.elapsed_time(prefix_done)
            row["ae_gpu_event_ms"] = ae_begin.elapsed_time(ae_done)
            row["prefill_completion_event_ms"] = origin.elapsed_time(prefix_done)
            row["ae_completion_event_ms"] = origin.elapsed_time(ae_done)
            row["both_completion_event_ms"] = max(row["prefill_completion_event_ms"], row["ae_completion_event_ms"])
            row["model_completion_event_ms"] = (row["both_completion_event_ms"]
                - row["prefill_completion_event_ms"] + row["prefill_gpu_event_ms"])
        self.pending.clear()

    def verify_outputs(self, oracle, max_error, mae_error):
        passed = True
        for row, normalized, decoded in self.outputs:
            reference = oracle[row["frame"]]
            row["normalized_error"] = difference(reference["normalized"], normalized)
            row["decoded_error"] = difference(reference["decoded"], decoded)
            correct = all(within_tolerance(row[key], max_error, mae_error)
                          for key in ("normalized_error", "decoded_error"))
            if "first_actions" in row:
                row["first_actions_error"] = difference(reference["first_actions"], row.pop("first_actions"))
                row["kv_equal"] = row["kv_digest"] == reference["kv_digest"]
                correct = correct and row["kv_equal"] and within_tolerance(row["first_actions_error"], max_error, mae_error)
            row["correct"] = bool(correct)
            passed = passed and correct
        self.outputs.clear()
        return bool(passed)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    result.add_argument("--num-steps", type=int, choices=(2, 10), default=2)
    result.add_argument("--boundary", choices=("kv_ready", "block_ready"), default="kv_ready")
    result.add_argument("--group-size", type=int, choices=(1, 2, 4), default=1)
    result.add_argument("--ae-layout", choices=("whole", "per_step"), default="whole")
    result.add_argument("--tail-policy", choices=("native", "wait_prefix", "full"), default="native",
                        help="native keeps existing execution; wait_prefix gates remaining steps on Prefill; full also moves them to an unrestricted stream")
    result.add_argument("--lifecycle-audit", action="store_true")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--count", type=int, default=100)
    result.add_argument("--continuous-count", type=int, default=100)
    result.add_argument("--reverse-order", action="store_true",
                        help="Reverse warmup, paired starting order and continuous arms for an independent repeat")
    result.add_argument("--reset-every", type=int, default=50)
    result.add_argument("--hoist-modulation", action="store_true")
    result.add_argument("--ae-priority", type=int, choices=(0, -1), default=0)
    result.add_argument("--ae-sm-min", type=int, default=0,
                        help="0 uses ordinary streams; otherwise reserve this minimum SM count for AE")
    result.add_argument("--prefix-sm-cap", type=int, default=0,
                        help="Restrict only Prefill to a Green Context; AE keeps an unrestricted primary-context stream. Actual rounded SM count is recorded.")
    result.add_argument("--iteration-timing", action="store_true")
    result.add_argument("--profile", action="store_true")
    result.add_argument("--profile-mode", choices=("serial", "overlap"), default="overlap")
    result.add_argument("--max-error", type=float, default=1e-5)
    result.add_argument("--mae-error", type=float, default=1e-6)
    return result


def main():
    args = parser().parse_args()
    validate_tail_policy(args)
    if args.prefix_sm_cap < 0 or (args.prefix_sm_cap and args.ae_sm_min):
        raise ValueError("prefix-sm-cap must be nonnegative and cannot be combined with ae-sm-min")
    if args.warmup < 2 or args.count < 0 or args.continuous_count < 1 or args.reset_every < 0 or args.ae_sm_min < 0:
        raise ValueError("Require warmup>=2, count>=0, continuous-count>=1, reset-every>=0, ae-sm-min>=0")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(status="preparing", args=vars(args), rows=[], smoke=[], groups=[],
                  environment={key: os.environ.get(key) for key in
                               ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  steps=args.num_steps, tolerance=dict(max_abs=args.max_error, mae=args.mae_error),
                  scope="Same-frame fresh KV; serial and overlap share identical graph structure and preparation.",
                  timing_semantics={
                      "vision_embedding_event_ms": "Encoder, projector, text embedding, metadata, and stream/submission gaps.",
                      "vlm_event_ms": "From vision completion to VLM full completion, including submission gaps.",
                      "ae_gpu_event_ms": "AE branch span including metadata and layer waits; serial also waits full Prefill. Not kernel busy time.",
                      "ae_iteration1_event_ms": "Diagnostic first iteration span including KV-ready waits.",
                      "ae_iteration2_event_ms": "Diagnostic second iteration span including resource contention.",
                      "ae_tail2to10_event_ms": "D10 diagnostic combined iterations 2 through 10 span, not one AE iteration.",
                      "quota_comparison": "With SM restriction, both paired arms use the same restricted streams; compare separately against unrestricted serial for global speedup."},
                  profile_run=args.profile, iteration_timing_diagnostic=args.iteration_timing,
                  performance_eligible=False,
                  command=shlex.join([sys.executable, "-m", "preparations.idea2.layer_bench", *sys.argv[1:]]),
                  source_sha256=capture_sources(dest),
                  bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest())
    write_result(dest, result)
    runner = graphs = green_pair = prefix_stream = ae_stream = tail_stream = None
    failure = None
    try:
        result["gpu_before_init"] = require_idle_gpu(args.gpu_uuid)
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import torch
        from .experiments import load_backend, native_raw
        from .external_events import cross_graph_self_test
        from .layer_graphs import LayerGraphs

        if torch.cuda.device_count() != 1:
            raise RuntimeError("The experiment must see exactly one CUDA device")
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if actual_uuid.lower().removeprefix("gpu-") != args.gpu_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError(f"CUDA visible device UUID mismatch: {actual_uuid} vs {args.gpu_uuid}")
        result["actual_cuda_uuid"] = actual_uuid
        result["external_events"] = cross_graph_self_test()
        occupancy = gpu_snapshot(args.gpu_uuid)
        own_pids = {row["pid"] for row in occupancy["processes"]}
        if len(own_pids) != 1 or occupancy["graphics_processes"]:
            raise RuntimeError("Expected exactly one CUDA process and no graphics processes after self-test")
        result["own_gpu_processes"] = sorted(own_pids)
        backend = load_backend(args.checkpoint, model_config="pi05_libero", num_steps=args.num_steps)
        result["identity"] = backend.identity
        original_records = pickle.loads(Path(args.bank).read_bytes())
        instruction = original_records[0]["instruction"]
        indices = [index for index, record in enumerate(original_records) if record["instruction"] == instruction]
        records = [original_records[index] for index in indices]
        if len(records) < 2:
            raise RuntimeError("Need changing observations of the same task")
        result["selected_bank_indices"] = indices
        result["oracle_status"] = "running"
        write_result(dest, result)
        oracle = eager_oracle(backend, records)
        result["oracle"] = [dict(index=row["index"], seed=row["seed"], kv_digest=row["kv_digest"],
                                 times=row["times"], actual_nfe=row["actual_nfe"],
                                 schedule_reference=row["schedule_reference"])
                            for row in oracle]
        result["oracle_status"] = "passed"
        if args.hoist_modulation:
            from .cewp_preparation import PreparedModulationCore
            backend.graph_core_class = PreparedModulationCore
        prepared = backend.prepare(native_raw(records[0]))
        noise = backend.noise(input_schedule(0, len(records))[1])
        if args.prefix_sm_cap:
            from .green_context import GreenContextPair, build_library
            library = build_library(dest / "green_build")
            green_pair = GreenContextPair(library, first_sm_min=args.prefix_sm_cap,
                                          device=backend.device, priorities=(0, 0))
            prefix_stream = green_pair.streams[0]
            ae_stream = torch.cuda.Stream(device=backend.device, priority=args.ae_priority)
            result["green_context"] = dict(green_pair.info, branch_order=["prefix", "unused"],
                                           ae_resource_policy="unrestricted_primary_context",
                                           prefix_actual_sm_count=green_pair.info["sm_counts"][0],
                                           library=str(library))
        elif args.ae_sm_min:
            from .green_context import GreenContextPair, build_library
            library = build_library(dest / "green_build")
            green_pair = GreenContextPair(library, first_sm_min=args.ae_sm_min,
                                          device=backend.device, priorities=(args.ae_priority, 0))
            ae_stream, prefix_stream = green_pair.streams
            result["green_context"] = dict(green_pair.info, branch_order=["ae", "prefix"],
                                           library=str(library))
        else:
            prefix_stream = torch.cuda.Stream(device=backend.device)
            ae_stream = torch.cuda.Stream(device=backend.device, priority=args.ae_priority)
            result["green_context"] = None
        if args.tail_policy == "full":
            tail_stream = torch.cuda.Stream(device=backend.device, priority=args.ae_priority)
        graphs = LayerGraphs(backend, prepared, noise, prefix_stream, ae_stream,
                             boundary=args.boundary, group_size=args.group_size, ae_layout=args.ae_layout,
                             profile_labels=args.profile, tail_stream=tail_stream,
                             tail_gate=args.tail_policy != "native")
        del prepared, noise
        result["graph_setup_ms"] = graphs.setup_ms
        result["kv_slot_addresses"] = graphs.addresses
        result["graph_schedule_audit"] = audit_graph_schedules(graphs, oracle)
        runner = LayerRunner(backend, graphs, records, iteration_timing=args.iteration_timing)
        # Complete cycles force both slots to see several different contents;
        # resetting between modes also exercises epoch changes with reused graphs.
        mode_order = ("overlap", "serial") if args.reverse_order else ("serial", "overlap")
        for mode in mode_order:
            runner.reset(phase="smoke", mode=mode, index=0)
            for index in range(2 * len(records)):
                result["smoke"].append(runner.predict(index, mode, phase="smoke", verify=True))
            runner.finish_events()
            if not runner.verify_outputs(oracle, args.max_error, args.mae_error):
                raise RuntimeError("Numerical/KV smoke failed before formal timing")
        for row in result["smoke"]:
            if tuple(row["kv_addresses"]) != tuple(graphs.addresses[row["slot"]]):
                raise RuntimeError("KV addresses changed during smoke")
        result["smoke_status"] = "passed"
        if args.lifecycle_audit:
            from .layer_lifecycle import audit
            result["lifecycle_audit"] = audit(backend, graphs, runner.pool, records, oracle,
                                               max_error=args.max_error, mae_error=args.mae_error)
        write_result(dest, result)
        for mode in mode_order:
            runner.reset(phase="warmup", mode=mode, index=0)
            for index in range(args.warmup):
                runner.predict(index, mode, phase="warmup")
            runner.finish_events()
            if not runner.verify_outputs(oracle, args.max_error, args.mae_error):
                raise RuntimeError("Warmup output differs from same-frame oracle")
        result["status"] = "measuring"
        write_result(dest, result)
        if args.count and not args.profile:
            group = dict(phase="paired", gpu_before=require_same_processes(args.gpu_uuid, own_pids))
            torch.cuda.reset_peak_memory_stats()
            for pair in range(args.count):
                names = mode_order if pair % 2 == 0 else tuple(reversed(mode_order))
                for ordinal, mode in enumerate(names):
                    runner.reset(phase="paired", mode=mode, index=pair)
                    result["rows"].append(runner.predict(pair, mode, phase="paired", pair=pair, arm_order=ordinal))
                if pair % 10 == 9:
                    print(f"Pairs {pair + 1}/{args.count}", flush=True)
            runner.finish_events()
            group["correct"] = runner.verify_outputs(oracle, args.max_error, args.mae_error)
            group.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         gpu_after=require_same_processes(args.gpu_uuid, own_pids))
            result["groups"].append(group)
            write_result(dest, result)
            if not group["correct"]:
                raise RuntimeError("Paired outputs failed numerical gate")
        for mode in ((args.profile_mode,) if args.profile else mode_order):
            runner.reset(phase="continuous", mode=mode, index=0)
            group = dict(phase="continuous", mode=mode,
                         gpu_before=require_same_processes(args.gpu_uuid, own_pids))
            torch.cuda.reset_peak_memory_stats()
            publications_before = runner.pool.publications
            phase_rows = []
            if args.profile:
                torch.cuda.profiler.start()
            begin = time.perf_counter()
            try:
                for index in range(args.continuous_count):
                    if index and args.reset_every and index % args.reset_every == 0:
                        runner.reset(phase="continuous", mode=mode, index=index)
                    row = runner.predict(index, mode, phase="continuous")
                    result["rows"].append(row)
                    phase_rows.append(row)
                responses_end = time.perf_counter()
                # Drain only at trajectory end, including remaining VLM tail
                # in the all-work throughput denominator.
                graphs.prefix_stream.synchronize()
                ae_completion_stream(graphs).synchronize()
                runner.pool.poll()
                all_work_end = time.perf_counter()
            finally:
                if args.profile:
                    torch.cuda.profiler.stop()
            runner.finish_events()
            group["correct"] = runner.verify_outputs(oracle, args.max_error, args.mae_error)
            group.update(response_trajectory_ms=(responses_end - begin) * 1000,
                         all_work_trajectory_ms=(all_work_end - begin) * 1000,
                         response_hz=len(phase_rows) / (responses_end - begin),
                         all_work_hz=len(phase_rows) / (all_work_end - begin),
                         completed_prefills=len(phase_rows), consumed_snapshots=len(phase_rows),
                         completed_prefill_hz=len(phase_rows) / (all_work_end - begin),
                         observed_publications=runner.pool.publications - publications_before,
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                         gpu_after=require_same_processes(args.gpu_uuid, own_pids))
            group["output_interval_ms"] = statistics([(b["host_end"] - a["host_end"]) * 1000
                                                      for a, b in zip(phase_rows, phase_rows[1:])])
            result["groups"].append(group)
            write_result(dest, result)
            if not group["correct"]:
                raise RuntimeError("Continuous outputs failed numerical gate")
        result["summary"] = {}
        for phase in ("paired", "continuous"):
            for mode in ("serial", "overlap"):
                selected = [row for row in result["rows"] if (row["phase"], row["mode"]) == (phase, mode)]
                if selected:
                    result["summary"][phase + "/" + mode] = {
                        key: statistics([row[key] for row in selected])
                        for key in ("cpu_cpu_ms", "prefill_gpu_event_ms", "vision_embedding_event_ms", "vlm_event_ms",
                                    "ae_gpu_event_ms", "ae_completion_event_ms",
                                    "both_completion_event_ms", "model_completion_event_ms", "prefill_cpu_submit_ms", "ae_cpu_submit_ms",
                                    "prepare_cpu_ms", "noise_cpu_ms", "output_cpu_ms", "action_output_host_ms",
                                    *iteration_timing_fields(args.num_steps))}
        result["resets"] = runner.reset_rows
        result["status"] = "passed"
        result["performance_eligible"] = (not args.profile and not args.iteration_timing
                                          and args.count >= 100 and args.continuous_count >= 100)
    except BaseException as error:
        # Keep text, not the original traceback: failed capture/replay frames
        # can hold Graph references whose contexts must outlive those objects.
        failure = repr(error)
        result["status"] = "failed"
        result["error"] = failure
        result["traceback"] = traceback.format_exc()
    finally:
        graph_cleanup_ok = True
        if graphs is not None:
            try:
                graphs.close()
            except BaseException:
                result["cleanup_error"] = traceback.format_exc()
                result["status"] = "failed"
                graph_cleanup_ok = False
        # Graph owners and all stream-user references are dropped before their
        # external streams/Green Contexts are destroyed. Source traceback above
        # is text only; gc also collects cold-capture Python hook cycles.
        runner = graphs = prefix_stream = ae_stream = tail_stream = None
        gc.collect()
        if green_pair is not None and graph_cleanup_ok:
            try:
                green_pair.close()
                result["green_context_closed"] = True
            except BaseException:
                result["green_cleanup_error"] = traceback.format_exc()
                result["status"] = "failed"
        elif green_pair is not None:
            result["green_context_closed"] = False
            result["green_cleanup_note"] = "Graph stream drain failed; do not destroy possibly referenced contexts. Process exit owns final cleanup."
        if result["status"] != "passed":
            result["performance_eligible"] = False
        write_result(dest, result)
    if result["status"] != "passed":
        raise RuntimeError(failure or "Experiment cleanup failed; inspect result.json")


if __name__ == "__main__":
    main()
