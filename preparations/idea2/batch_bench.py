"""Opt-in fixed-batch pi05 D10/H10 replay benchmark; never starts from a plan.

``--help`` and ``--dry-run`` perform no CUDA imports, GPU queries or admission.
Execution needs an explicit physical UUID and a fresh output directory. This
replays independent recorded lanes, not future observations from one robot.
"""
import argparse
from contextlib import contextmanager, nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import sys
import time
import traceback

import numpy as np

from .layer_bench import difference, statistics, within_tolerance, write_result


MAX_ERROR = 1e-5
MAE_ERROR = 1e-6
MODES = ("serial", "pipeline", "green", "paired")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--batch-size", type=int, choices=(1, 2, 4), default=1)
    result.add_argument("--mode", choices=MODES, default="paired")
    result.add_argument("--input-mode", choices=("prepared", "online"), default="prepared")
    result.add_argument("--prepare-stream", choices=("caller", "prefix", "dedicated"), default="caller",
                        help="Online replay only: place unchanged prepare on caller, Prefix, or a dedicated input stream")
    result.add_argument("--disable-lookahead", action="store_true",
                        help="Prepare each current input inside its request; disable future CPU preparation and next Encoder")
    result.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    result.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    result.add_argument("--gpu-uuid")
    result.add_argument("--output")
    result.add_argument("--pairs", type=int, default=100)
    result.add_argument("--length", type=int, default=8)
    result.add_argument("--warmup", type=int, default=20)
    result.add_argument("--verify-count", type=int, default=4)
    result.add_argument("--lead", type=int, choices=(0, 1, 3, 6, 18), default=3)
    result.add_argument("--ae-sm-min", type=int, default=16)
    result.add_argument("--seed", type=int, default=1701)
    result.add_argument("--reverse", action="store_true")
    result.add_argument("--profile", action="store_true")
    result.add_argument("--diagnostic-batch-drift", action="store_true",
                        help="Measure same-B scheduling only; retain failed B1 equivalence as diagnostic, never promotion eligible")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_args(args):
    if args.pairs < 1 or args.length < 2 or args.warmup < 2 or args.verify_count < 4:
        raise ValueError("Require pairs>=1, length>=2, warmup>=2, verify-count>=4")
    if args.ae_sm_min <= 0:
        raise ValueError("ae-sm-min must be positive")
    if not args.dry_run and (not args.output or not args.gpu_uuid or not args.gpu_uuid.startswith("GPU-")):
        raise ValueError("Execution requires --output and an explicit physical --gpu-uuid")
    if args.profile and args.mode == "paired":
        raise ValueError("Profile one explicit mode; use paired only for unprofiled performance")
    if args.diagnostic_batch_drift and args.batch_size == 1:
        raise ValueError("Batch-drift diagnostic is only meaningful for B>1")
    if args.prepare_stream != "caller" and args.input_mode != "online":
        raise ValueError("Non-caller prepare-stream requires online input mode")


def seed_for(record_index, lane, base=1701):
    """A record keeps its noise when regrouped into another B or lane."""
    del lane  # Lane identity is recorded separately and cannot change noise.
    data = f"idea2-batch-v1:{base}:{record_index}".encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "little")


def bank_schedule(records, batch_size, *, seed=1701):
    """Disjoint samples within each frame; fixed lanes and one task/mask family."""
    if not records:
        raise ValueError("Input bank is empty")
    instruction = records[0]["instruction"]
    selected = [i for i, row in enumerate(records) if row["instruction"] == instruction]
    # B1/B2/B4 use the same record set, with the same per-record noise.
    usable = len(selected) // 4 * 4
    if usable < 2 * batch_size:
        raise ValueError("Need at least two changing batch frames of the same task")
    frames = []
    for frame, offset in enumerate(range(0, usable, batch_size)):
        indices = selected[offset:offset + batch_size]
        frames.append(dict(frame=frame, bank_indices=indices,
                           lane_ids=[f"lane{lane}" for lane in range(batch_size)],
                           seeds=[seed_for(index, lane, seed) for lane, index in enumerate(indices)]))
    return dict(instruction=instruction, batch_size=batch_size, frames=frames,
                selected_bank_indices=selected[:usable],
                excluded_same_task_remainder=selected[usable:],
                common_sample_set_for_batch_sizes=[1, 2, 4],
                independent_lanes=True, future_single_robot_frames=False)


def arm_order(mode, pair, reverse=False):
    arms = ["serial", "pipeline"] if mode == "paired" else [mode]
    return arms[::-1] if bool(pair % 2) ^ bool(reverse) else arms


def compare_lanes(reference, actual, lane_ids):
    """A bad lane cannot be diluted by a large batch's aggregate MAE."""
    if len(reference) != len(lane_ids) or len(actual) != len(lane_ids):
        raise ValueError("Output lane count changed")
    lanes = []
    for lane, lane_id in enumerate(lane_ids):
        error = difference(reference[lane], actual[lane])
        lanes.append(dict(lane=lane, lane_id=lane_id, **error,
                          passed=within_tolerance(error, MAX_ERROR, MAE_ERROR)))
    return dict(aggregate=difference(reference, actual), per_lane=lanes,
                passed=all(row["passed"] for row in lanes))


def plan_description(args):
    return dict(status="dry_run", gpu_work_started=False, args=vars(args),
                model="pi05_libero", num_steps=10, action_horizon=10,
                resource_mode="green_disjoint" if args.mode == "green" else "full_gpu",
                input_preparation_stream=args.prepare_stream,
                lookahead_enabled=not args.disable_lookahead,
                schedules=arm_order(args.mode, 0, args.reverse),
                timed_requests=args.length * (1 if args.profile else args.pairs)
                               * len(arm_order(args.mode, 0)),
                tolerance=dict(max_abs=MAX_ERROR, mae=MAE_ERROR),
                semantics="Fixed independent lanes; two batched KV slots; per-round frozen current/old frontier.",
                input_boundary=("Preprepared recorded bank; excludes preparing the bank, no camera-age claim."
                                if args.input_mode == "prepared" else
                                "Recorded CPU raw bank; prepare_batch and CPU lookahead included, no camera-age claim."),
                throughput="B * completed requests / group all-work wall time, including drain; never sum stage spans.",
                numerical_coverage="B1 eager fresh oracle before captures; fresh-serial all samples and separate mixed-frontier verification; timed mixed samples retain outputs but do not run eager oracles.")


def eager_lane_oracle(base, raw_frames, schedule, dest):
    """Keep no GPU KV when the two-slot graph runtime is subsequently built."""
    import torch

    references = []
    for raw_frame, frame in zip(raw_frames, schedule["frames"], strict=True):
        normalized, decoded = [], []
        for raw, seed in zip(raw_frame, frame["seeds"], strict=True):
            prepared = base.prepare(raw)
            noise = base.noise(seed)
            prefix = base.build_prefix(prepared)
            actions = base.generate(prefix, prepared, noise)
            normalized.append(actions[0].detach().cpu().numpy().copy())
            decoded.append(np.asarray(base.output(prepared, actions)).copy())
            torch.cuda.current_stream(base.device).synchronize()
            del prefix, prepared, noise, actions
        references.append(dict(normalized=np.stack(normalized), decoded=np.stack(decoded)))
    gc.collect()
    np.savez_compressed(dest / "b1_eager_oracle.npz",
                        normalized=np.stack([row["normalized"] for row in references]),
                        decoded=np.stack([row["decoded"] for row in references]))
    return references


def eager_batch_oracle(backend, raw_frames, schedule, dest):
    """Independent original same-B math; release its sole KV before captures."""
    import torch
    references = []
    for raws, frame in zip(raw_frames, schedule["frames"], strict=True):
        prepared = backend.prepare_batch(raws)
        noise = backend.noise_batch(frame["seeds"])
        prefix = backend.build_prefix(prepared)
        actions = backend.generate(prefix, prepared, noise)
        normalized = actions.detach().cpu().numpy().copy()
        references.append(dict(normalized=normalized,
                               decoded=backend.output_batch(prepared, normalized)))
        torch.cuda.current_stream(backend.device).synchronize()
        del prepared, noise, prefix, actions
    gc.collect()
    np.savez_compressed(dest / "same_batch_eager_oracle.npz",
                        normalized=np.stack([row["normalized"] for row in references]),
                        decoded=np.stack([row["decoded"] for row in references]))
    return references


def _scope(nvtx, label):
    return nvtx(label) if nvtx is not None else nullcontext()


@contextmanager
def _cleanup_before_lease_release(cleanup):
    """The GPU lease outlives pending kernels, graphs and external streams."""
    try:
        yield
    finally:
        cleanup()


def release_green_resources(resources, torch_module):
    """Caller tensors must already be gone; all operations are outside timing."""
    gc.collect()
    if resources:
        # cuBLAS workspaces may carry allocator marks for the external streams.
        torch_module._C._cuda_clearCublasWorkspaces()
        for pair in resources:
            for stream in pair.streams:
                stream.synchronize()
        torch_module.cuda.synchronize()
    torch_module.cuda.empty_cache()
    gc.collect()
    for pair in resources:
        pair.close()
    resources.clear()


def prepare_in_stream(backend, raws, *, stream=None, cuda=None):
    """Preserve original prepare math and its producer dependency.

    The caller-stream path is unchanged. Explicit placement uses the benchmark's
    owned stream, then joins its ready Event into the caller stream so existing
    Graph input-ready Events still cover every prepared tensor. Prepared owns
    the Event until its normal request/slot lifetime ends; no Event is reused.
    """
    if stream is None:
        return backend.prepare_batch(raws)
    if cuda is None:
        import torch
        cuda = torch.cuda
    with cuda.stream(stream):
        prepared = backend.prepare_batch(raws)
        ready = cuda.Event()
        ready.record(stream)
        prepared._batch_prepare_ready = ready
    cuda.current_stream(backend.device).wait_event(ready)
    return prepared


def sequence(engine, backend, raw_frames, noises, schedule, *, mode, input_mode,
             start, length, phase, pair, lead, verify=False, nvtx=None,
             clock=time.perf_counter, profile_control=None, prepare_stream=None, cuda=None,
             prepare_stream_name=None, lookahead_enabled=True):
    """Only the caller's group drain resolves producer timing Events."""
    engine.drain()
    backend.contract.reset()
    frame_count = len(raw_frames)
    prepared_bank = None
    input_started = {}
    prepare_setup_begin = clock()
    if input_mode == "prepared":
        prepared_bank = [backend.prepare_batch(raws) for raws in raw_frames]
    prepare_setup_ms = (clock() - prepare_setup_begin) * 1000

    def prepare(index):
        if prepared_bank is not None:
            return prepared_bank[index % frame_count]
        started = clock()
        with _scope(nvtx, "preprocess"):
            value = prepare_in_stream(backend, raw_frames[index % frame_count],
                                      stream=prepare_stream, cuda=cuda)
        input_started[id(value)] = started
        return value

    bootstrap_begin = clock()
    engine.reset(prepare(start - 1))
    engine.drain()
    bootstrap_ms = (clock() - bootstrap_begin) * 1000
    rows, outputs = [], []
    current = None
    if profile_control is not None:
        profile_control(True)
    begin = clock()
    try:
        for index in range(length):
            request_begin = clock()
            frame_index = (start + index) % frame_count
            frame = schedule["frames"][frame_index]
            prepare_begin = clock()
            if current is None:
                current = prepare(start + index)
            prepare_ms = (clock() - prepare_begin) * 1000
            lookahead = ((lambda source=start + index + 1: prepare(source))
                         if lookahead_enabled and index + 1 < length else None)
            normalized, row, future = engine.request(
                current, noises[frame_index], schedule="serial" if mode == "serial" else "pipeline",
                lead=lead, lookahead=lookahead, next_encoder=mode != "serial" and lookahead_enabled,
                verify=verify, collect_events=False)
            decode_begin = clock()
            decoded = np.asarray(backend.output_batch(current, normalized))
            response = clock()
            own_prepare_started = input_started.pop(id(current), None)
            row.update(runtime_cpu_cpu_ms=row.get("cpu_cpu_ms"),
                       cpu_cpu_ms=(response - request_begin) * 1000,
                       phase=phase, mode=mode, input_mode=input_mode, pair=pair,
                       index=index, frame=frame_index, source_index=start + index,
                       prepare_stream=prepare_stream_name or ("prefix" if prepare_stream is not None else "caller"),
                       lookahead_enabled=lookahead_enabled,
                       input_prepare_to_cpu_ms=((response - own_prepare_started) * 1000
                                                if own_prepare_started is not None else None),
                       input_prepare_age_at_admission_ms=(max(0.0, (request_begin - own_prepare_started) * 1000)
                                                          if own_prepare_started is not None else None),
                       seeds=frame["seeds"], bank_indices=frame["bank_indices"],
                       input_prepare_ms=prepare_ms, output_decode_ms=(response - decode_begin) * 1000,
                       observation_age_control_steps=None,
                       camera_age_applicable=False, numerical_timed=False)
            rows.append(row)
            outputs.append((np.array(normalized, copy=True), np.array(decoded, copy=True)))
            current = future
        engine.drain()
        end = clock()
    finally:
        if profile_control is not None:
            profile_control(False)
    # Read Events after the completed group, outside its performance boundary.
    for row in rows:
        engine.finish_record(row)
    wall_ms = (end - begin) * 1000
    group = dict(phase=phase, mode=mode, input_mode=input_mode, pair=pair, start=start,
                 count=length, batch_size=backend.batch_size, all_work_ms=wall_ms,
                 lookahead_enabled=lookahead_enabled,
                 request_hz=length * 1000 / wall_ms,
                 sample_hz=backend.batch_size * length * 1000 / wall_ms,
                 prepare_bank_setup_ms=prepare_setup_ms, bootstrap_ms=bootstrap_ms,
                 bootstrap_and_prepare_setup_in_timing=False)
    return rows, group, outputs


def attach_checks(rows, outputs, oracle, lane_ids, *, same_batch_oracle=None,
                  diagnostic_batch_drift=False):
    """CPU comparisons run after group timing, including all decoded action lanes."""
    passed = True
    for row, (normalized, decoded) in zip(rows, outputs, strict=True):
        reference = oracle[row["frame"]]
        checks = dict(fresh_normalized=compare_lanes(reference["normalized"], normalized, lane_ids),
                      fresh_decoded=compare_lanes(reference["decoded"], decoded, lane_ids))
        b1_passed = checks["fresh_normalized"]["passed"] and checks["fresh_decoded"]["passed"]
        if same_batch_oracle is not None:
            batch_reference = same_batch_oracle[row["frame"]]
            checks.update(same_batch_normalized=compare_lanes(batch_reference["normalized"], normalized, lane_ids),
                          same_batch_decoded=compare_lanes(batch_reference["decoded"], decoded, lane_ids))
        if row["mode"] == "serial":
            if diagnostic_batch_drift:
                if same_batch_oracle is None:
                    raise ValueError("Diagnostic timing requires an independent same-B oracle")
                checks.update(scope="Strict same-B eager oracle; B1 equivalence remains separately recorded",
                              passed=checks["same_batch_normalized"]["passed"] and checks["same_batch_decoded"]["passed"])
            else:
                checks.update(scope="Strict fresh B1 eager oracle", passed=b1_passed)
            checks["b1_equivalence_passed"] = b1_passed
        elif "numerical" in row:
            checks.update(scope="Strict batched serial oracle using the observed same KV frontier; fresh differences are diagnostic",
                          passed=bool(row["numerical"]["matched_schedule"]["passed"]))
        else:
            checks.update(scope="Timed mixed output retained; fresh differences are diagnostic; matched schedule validated in separate phase",
                          passed=None)
        row["checks"] = checks
        if checks["passed"] is False:
            passed = False
    return passed


def summaries(rows, groups, batch_size):
    result = {}
    for mode in sorted({row["mode"] for row in rows}):
        selected = [row for row in rows if row["mode"] == mode and row["phase"] == "paired"]
        runs = [row for row in groups if row["mode"] == mode and row["phase"] == "paired"]
        if not runs:
            continue
        count = sum(row["count"] for row in runs)
        wall_ms = sum(row["all_work_ms"] for row in runs)
        fields = ("cpu_cpu_ms", "runtime_cpu_cpu_ms", "input_prepare_ms", "lookahead_prepare_ms",
                  "input_prepare_to_cpu_ms", "input_prepare_age_at_admission_ms",
                  "output_decode_ms", "vision_ms", "vlm_ms", "ae_ms", "ae_submit_ms",
                  "exposed_ae_after_vlm_ms", "model_after_encoder_ms",
                  "ae_finish_minus_next_encoder_ms", "prefill_completion_event_ms",
                  "ae_completion_event_ms", "both_completion_event_ms",
                  "all_submitted_work_completion_event_ms", "old_kv_age_requests")
        entry = dict(samples=count * batch_size, requests=count,
                     request_hz=count * 1000 / wall_ms,
                     sample_hz=count * batch_size * 1000 / wall_ms,
                     group_sample_hz=statistics([row["sample_hz"] for row in runs]),
                     kv_update_batches_hz=count * 1000 / wall_ms,
                     kv_update_samples_hz=count * batch_size * 1000 / wall_ms,
                     all_work_ms=statistics([row["all_work_ms"] for row in runs]),
                     steady_graph_recaptures=sum(row.get("graph_recaptures", 0) for row in selected))
        entry.update({field: statistics([row.get(field) for row in selected]) for field in fields})
        if selected and all("frontiers" in row for row in selected):
            entry["mean_new_layers_by_ae_step"] = np.mean([row["frontiers"] for row in selected], axis=0).tolist()
            layers = selected[0].get("num_vlm_layers")
            if layers is not None:
                entry["mean_old_layers_by_ae_step"] = (layers - np.mean(
                    [row["frontiers"] for row in selected], axis=0)).tolist()
        result[mode] = entry
    if "serial" in result and "pipeline" in result:
        result["paired_comparison"] = dict(
            throughput_speedup=result["pipeline"]["sample_hz"] / result["serial"]["sample_hz"],
            cpu_cpu_mean_speedup=result["serial"]["cpu_cpu_ms"]["mean"] / result["pipeline"]["cpu_cpu_ms"]["mean"],
            same_process_same_captures=True,
            semantics="Mixed-frontier pipeline versus fresh serial; quality equivalence is not implied.")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    if args.dry_run:
        print(json.dumps(plan_description(args), indent=2))
        return 0
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    from .batch_resources import exclusive_gpu
    from .layer_bench import capture_sources, gpu_snapshot, require_same_processes

    result = plan_description(args)
    result.update(status="preparing", rows=[], groups=[], warmup_groups=[], verification_rows=[],
                  performance_eligible=False, profile_run=args.profile,
                  scheduling_performance_eligible=False, numerical_gates_passed=False,
                  same_batch_numerical_gates_passed=False, b1_equivalence_passed=None,
                  diagnostic_only=bool(args.diagnostic_batch_drift), promotion_eligible=False,
                  source_before=capture_sources(dest),
                  command=shlex.join([sys.executable, "-m", "preparations.idea2.batch_bench",
                                      *(sys.argv[1:] if argv is None else argv)]),
                  timing_semantics={
                      "cpu_cpu_ms": "Recorded input admission/current preparation through normalized CPU download and decoded actions. CPU lookahead inside the request is included.",
                      "input_prepare_to_cpu_ms": "This specific input's own Host prepare start through decoded CPU actions, including waiting when prepared in a prior request; no camera timestamp.",
                      "lookahead": "Disabled; each current input is prepared in its measured request." if args.disable_lookahead else "Current input may have been prepared in the preceding request; next-input preparation is counted in the current admission segment.",
                      "event_spans": "Include stream waits, submission gaps and contention; not kernel busy time.",
                      "all_work_ms": "Group wall time through all stream completion. Bootstrap, bank setup, checks, output archival and Event reads are excluded.",
                      "freshness": "Offline recorded input replay, no robot observation-age or closed-loop-success claim.",
                      "green": "Separate process/resources; comparisons to full-GPU runs are not same-process paired experiments."})
    archives = []
    engine = None
    input_stream = None
    pending_green_resources = []
    base = backend = noises = initial = probe = target = replacement = None
    failure = None
    def close_engine():
        nonlocal engine, input_stream
        # An exception may occur while prepare has queued work, before encode
        # can consume its ready Event. Drain that producer before freeing inputs.
        if input_stream is not None:
            input_stream.synchronize()
        owned = engine
        if owned is not None:
            resources = owned.close()
            if resources is not None:
                pending_green_resources.append(resources)
            engine = None
        input_stream = None
    def close_cuda():
        nonlocal base, backend, noises, initial, probe, target, replacement
        try:
            close_engine()
        finally:
            base = backend = noises = initial = probe = target = replacement = None
            gc.collect()
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_initialized():
                release_green_resources(pending_green_resources, torch_module)
    try:
        data = Path(args.bank).read_bytes()
        result["bank_sha256"] = hashlib.sha256(data).hexdigest()
        records = pickle.loads(data)
        schedule = bank_schedule(records, args.batch_size, seed=args.seed)
        result["input_manifest"] = schedule
        write_result(dest, result)
        with exclusive_gpu(args.gpu_uuid, dest) as lease, _cleanup_before_lease_release(close_cuda):
            result["admission"] = dict(lease)
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
            os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
            import torch
            from .batch_backend import BatchOpenPIBackend
            from .batch_runtime import BatchProgressiveRuntime
            from .experiments import load_backend, native_raw

            if torch.cuda.device_count() != 1:
                raise RuntimeError("Exactly one admitted GPU must be visible")
            result["gpu_work_started"] = True
            actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
            if actual_uuid.lower().removeprefix("gpu-") != args.gpu_uuid.lower().removeprefix("gpu-"):
                raise RuntimeError("Visible CUDA GPU UUID differs from the admitted physical GPU")
            load_begin = time.perf_counter()
            base = load_backend(args.checkpoint, model_config="pi05_libero", num_steps=10)
            if base.model.config.action_horizon != 10:
                raise RuntimeError("This experiment requires the existing H10 checkpoint/config")
            backend = BatchOpenPIBackend(base.policy, num_steps=10,
                                          lane_ids=schedule["frames"][0]["lane_ids"])
            raw_frames = [[native_raw(records[index]) for index in frame["bank_indices"]]
                          for frame in schedule["frames"]]
            result.update(model_identity=base.identity, model_load_ms=(time.perf_counter() - load_begin) * 1000)
            oracle_begin = time.perf_counter()
            oracle = eager_lane_oracle(base, raw_frames, schedule, dest)
            result["b1_oracle_ms"] = (time.perf_counter() - oracle_begin) * 1000
            same_batch_oracle = None
            if args.diagnostic_batch_drift:
                oracle_begin = time.perf_counter()
                same_batch_oracle = eager_batch_oracle(backend, raw_frames, schedule, dest)
                result["same_batch_oracle_ms"] = (time.perf_counter() - oracle_begin) * 1000
            noises = [backend.noise_batch(frame["seeds"]) for frame in schedule["frames"]]
            initial = backend.prepare_batch(raw_frames[0])
            # Reject differing old/new prefix masks before graph capture.
            mask_signature = None
            for raw_frame in raw_frames:
                probe = backend.prepare_batch(raw_frame)
                signature = tuple((tuple(value.shape), value.detach().cpu().numpy().tobytes())
                                  for value in (*probe.image_masks, probe.token_masks))
                if mask_signature is not None and signature != mask_signature:
                    raise RuntimeError("Input bank changes masks; the old/new-KV experiment requires one fixed mask family")
                mask_signature = signature
                del probe
            green_library = None
            if args.mode == "green":
                from .green_context import build_library
                green_library = build_library(dest / "green_build")
            setup_begin = time.perf_counter()
            engine = BatchProgressiveRuntime(backend, initial, noises[0],
                mode="green" if args.mode == "green" else "pipeline",
                ae_sm_min=args.ae_sm_min, green_library=green_library, profile=args.profile)
            if args.prepare_stream == "prefix":
                input_stream = engine.graphs.prefix_stream
            elif args.prepare_stream == "dedicated":
                input_stream = torch.cuda.Stream(device=backend.device)
            result.update(graph_setup_ms=(time.perf_counter() - setup_begin) * 1000,
                          resource_layout=engine.resource_layout,
                          kv_addresses=engine.graphs.kv_addresses,
                          full_kv_slots=2, hidden_third_full_kv=False)
            own = {row["pid"] for row in gpu_snapshot(args.gpu_uuid)["processes"]}
            if len(own) != 1:
                raise RuntimeError("Expected exactly one GPU process after model loading")
            result["own_gpu_processes"] = sorted(own)
            del initial

            def run(mode, phase, pair, length, start=0, verify=False):
                def profile_control(started):
                    operation = "cudaProfilerStart" if started else "cudaProfilerStop"
                    getattr(torch.cuda.cudart(), operation)()
                rows, group, values = sequence(engine, backend, raw_frames, noises, schedule,
                    mode=mode, input_mode=args.input_mode, start=start, length=length,
                    phase=phase, pair=pair, lead=args.lead, verify=verify,
                    nvtx=torch.cuda.nvtx.range,
                    prepare_stream=input_stream, prepare_stream_name=args.prepare_stream, cuda=torch.cuda,
                    lookahead_enabled=not args.disable_lookahead,
                    profile_control=profile_control if phase == "profile" else None)
                good = attach_checks(rows, values, oracle, backend.contract.lane_ids,
                    same_batch_oracle=same_batch_oracle,
                    diagnostic_batch_drift=args.diagnostic_batch_drift)
                if phase == "fresh_b1_gate":
                    result["b1_equivalence_passed"] = all(row["checks"]["b1_equivalence_passed"] for row in rows)
                if phase != "warmup":
                    for row, value in zip(rows, values, strict=True):
                        row["output_index"] = len(archives)
                        archives.append(value)
                if phase == "paired" or phase == "profile":
                    result["rows"].extend(rows)
                    result["groups"].append(group)
                elif phase != "warmup":
                    result["verification_rows"].extend(rows)
                else:
                    result["warmup_groups"].append(group)
                if not good:
                    raise RuntimeError("Numerical gate failed; diagnostics retained and tolerance unchanged")
                return group

            result["status"] = "verifying"
            # Full-new B>1 is compared to independently computed B1 eager results,
            # even when the requested performance condition is mixed/green.
            run("serial", "fresh_b1_gate", -1, len(raw_frames), verify=True)
            conditions = arm_order(args.mode, 0, args.reverse)
            for mode in conditions:
                run(mode, "mixed_schedule_gate", -1, args.verify_count, verify=True)
                run(mode, "warmup", -1, args.warmup)
            # Same addresses, changing contents, explicit whole-batch reset.
            engine.drain()
            backend.contract.reset()
            target = backend.prepare_batch(raw_frames[0])
            replacement = backend.prepare_batch(raw_frames[1])
            engine.reset(target)
            from .runtime_graph import input_tensors
            addresses = [value.data_ptr() for value in input_tensors(target)]
            engine.update_inputs_inplace(target, replacement)
            normalized, row, _ = engine.request(target, noises[1], schedule="serial",
                                                verify=True, collect_events=False)
            decoded = backend.output_batch(target, normalized)
            engine.drain()
            engine.finish_record(row)
            row.update(mode="serial", frame=1, phase="same_address_updated_contents",
                       seeds=schedule["frames"][1]["seeds"],
                       bank_indices=schedule["frames"][1]["bank_indices"])
            good = attach_checks([row], [(normalized, decoded)], oracle, backend.contract.lane_ids,
                same_batch_oracle=same_batch_oracle,
                diagnostic_batch_drift=args.diagnostic_batch_drift)
            row["addresses_unchanged"] = addresses == [value.data_ptr() for value in input_tensors(target)]
            row["output_index"] = len(archives)
            archives.append((normalized, decoded))
            result["verification_rows"].append(row)
            del target, replacement
            if not good or not row["addresses_unchanged"]:
                raise RuntimeError("Same-address updated-input gate failed")
            result["same_batch_numerical_gates_passed"] = True
            result["numerical_gates_passed"] = bool(result["b1_equivalence_passed"])
            result["diagnostic_only"] = bool(args.diagnostic_batch_drift and not result["b1_equivalence_passed"])
            require_same_processes(args.gpu_uuid, own)
            result.update(setup_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          setup_peak_reserved_bytes=torch.cuda.max_memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            result["status"] = "profiling" if args.profile else "measuring"
            write_result(dest, result)
            if args.profile:
                # Capture excludes setup, oracles and warmup; never compare these
                # profile timings against unprofiled performance measurements.
                run(args.mode, "profile", 0, args.length)
            else:
                for pair in range(args.pairs):
                    for mode in arm_order(args.mode, pair, args.reverse):
                        group = run(mode, "paired", pair, args.length, start=pair * args.length)
                        print(f"pair={pair} mode={mode} B={args.batch_size} sample_hz={group['sample_hz']:.3f}", flush=True)
                        require_same_processes(args.gpu_uuid, own)
                    write_result(dest, result)
            result.update(gpu_after=require_same_processes(args.gpu_uuid, own),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            close_engine()
        result["summary"] = summaries(result["rows"], result["groups"], args.batch_size)
        result["source_after"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  for path in result["source_before"]}
        result["source_unchanged"] = result["source_before"] == result["source_after"]
        if not result["source_unchanged"]:
            raise RuntimeError("Runtime source changed during the experiment")
        result.update(status="diagnostic_passed" if result["diagnostic_only"] else "passed",
                      performance_eligible=not args.profile and args.pairs >= 100 and result["numerical_gates_passed"],
                      scheduling_performance_eligible=not args.profile and args.pairs >= 100
                                                      and result["same_batch_numerical_gates_passed"])
    except BaseException:
        result.update(status="failed", performance_eligible=False, scheduling_performance_eligible=False,
                      error=traceback.format_exc())
        failure = sys.exc_info()
    finally:
        # CUDA cleanup already ran inside the active lease, including failures.
        # Do not enqueue GPU work after releasing resource ownership.
        if (dest / "gpu_lease.json").is_file():
            result["admission"] = json.loads((dest / "gpu_lease.json").read_text())
        result["source_after"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                  if Path(path).is_file() else None
                                  for path in result["source_before"]}
        result["source_unchanged"] = result["source_before"] == result["source_after"]
        if archives:
            np.savez_compressed(dest / "actions.npz",
                                normalized=np.stack([value[0] for value in archives]),
                                decoded=np.stack([value[1] for value in archives]))
            result["action_archive"] = dict(file="actions.npz", entries=len(archives),
                                            indexing="rows/verification_rows.output_index")
        write_result(dest, result)
    if failure is not None:
        raise failure[1].with_traceback(failure[2])
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
