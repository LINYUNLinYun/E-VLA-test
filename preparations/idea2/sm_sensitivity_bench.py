"""Fixed-input isolated module SM curves, not an end-to-end speedup benchmark.

One model is reused sequentially across full/16/32/64/full AE SM allocations.
Each allocation owns exactly one two-slot LayerGraphs instance; it is destroyed
before the next allocation is created. Inputs/KV are ready before each measured
phase. The optional encoder-pair experiment overlaps different slots only.
AE-all retains both graph launches
and the original GPU Event handoff. CUDA Event spans also include Host gaps.
"""
import argparse
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
import weakref

from .layer_bench import (LayerRunner, audit_graph_schedules, capture_sources,
                          difference, eager_oracle, gpu_snapshot, input_schedule,
                          require_idle_gpu, require_same_processes, smi_rows,
                          statistics, within_tolerance, write_result)


PHASES = ("vision", "vlm", "ae1", "ae_all")
MAX_ERROR, MAE_ERROR = 1e-5, 1e-6


class NumericalGateError(RuntimeError):
    pass


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    result.add_argument("--num-steps", type=int, choices=(2, 10), default=10)
    result.add_argument("--ae-sm-mins", type=int, nargs="+", default=[16, 32, 64],
                        help="Distinct positive AE SM minima, each smaller than the physical SM count")
    result.add_argument("--warmup", type=int, default=20)
    result.add_argument("--count", type=int, default=30)
    result.add_argument("--hoist-modulation", action="store_true")
    result.add_argument("--encoder-pair", action="store_true",
                        help="Also compare serial/overlapping AE10 slot0 and Vision slot1; fixed-input microbenchmark only")
    return result


def memory_snapshot(torch):
    return dict(allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def summarize(rows):
    return {key: statistics([row[key] for row in rows])
            for key in ("gpu_event_ms", "cpu_submit_ms", "cpu_wait_ms", "cpu_total_ms")}


def measure_once(torch, stream, operation):
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    host_start = time.perf_counter()
    with torch.cuda.stream(stream):
        begin.record(stream)
        submit_start = time.perf_counter()
        operation()
        submit_end = time.perf_counter()
        end.record(stream)
    wait_start = time.perf_counter()
    end.synchronize()
    host_end = time.perf_counter()
    return dict(gpu_event_ms=begin.elapsed_time(end),
                cpu_submit_ms=(submit_end - submit_start) * 1000,
                cpu_wait_ms=(host_end - wait_start) * 1000,
                cpu_total_ms=(host_end - host_start) * 1000)


def prime_inputs(torch, graphs, slot, phase):
    """Complete every prerequisite outside phase timing, without new KV."""
    with torch.cuda.stream(graphs.prefix_stream):
        slot.vision_graph.replay()
        if phase in ("ae1", "ae_all"):
            slot.prefix_graph.replay()
        ready = torch.cuda.Event()
        ready.record(graphs.prefix_stream)
    ready.synchronize()
    if phase in ("ae1", "ae_all"):
        graphs.ae_stream.wait_event(ready)
        graphs.ae_stream.synchronize()


def validate_slot(backend, graphs, slot, reference, *, phase):
    """Output checks outside timing; no additional full KV snapshot is made."""
    from .graph_bench import cache_digest

    checks = dict(kv_equal=cache_digest(slot.prefix) == reference["kv_digest"])
    if phase in ("ae1", "ae_all"):
        checks["first_actions_error"] = difference(reference["first_actions"],
                                                    slot.first_actions[0].detach().cpu().numpy())
    if phase == "ae_all":
        checks["normalized_error"] = difference(reference["normalized"],
                                                 slot.actions[0].detach().cpu().numpy())
        checks["decoded_error"] = difference(reference["decoded"], backend.output(slot.ae_prepared, slot.actions))
    checks["passed"] = checks["kv_equal"] and all(
        within_tolerance(value, MAX_ERROR, MAE_ERROR)
        for key, value in checks.items() if key.endswith("_error"))
    return checks


def run_phase(torch, backend, graphs, phase, args, quota_name, actual_sms, reference, own_pids):
    """Private static-buffer microbench after the owner Pool has drained."""
    slot = graphs.slots[0]
    prime_inputs(torch, graphs, slot, phase)
    if phase == "vision":
        operation, stream = slot.vision_graph.replay, graphs.prefix_stream
    elif phase == "vlm":
        operation, stream = slot.prefix_graph.replay, graphs.prefix_stream
    elif phase == "ae1":
        operation, stream = slot.ae_graph.graphs[0].replay, graphs.ae_stream
    else:
        operation, stream = slot.ae_graph.replay, graphs.ae_stream
    before = require_same_processes(args.gpu_uuid, own_pids)
    warmup_start = time.perf_counter()
    for _ in range(args.warmup):
        measure_once(torch, stream, operation)
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index in range(args.count):
        row = measure_once(torch, stream, operation)
        row.update(quota=quota_name, phase=phase, index=index, num_steps=args.num_steps,
                   actual_sm_count=actual_sms["ae" if phase.startswith("ae") else "prefix"],
                   eligible=False)
        rows.append(row)
    after = require_same_processes(args.gpu_uuid, own_pids)
    peak = memory_snapshot(torch)
    # Vision has no cached oracle for its embeddings: propagate its final output
    # through VLM once outside timing and compare the resulting KV digest.
    if phase == "vision":
        with torch.cuda.stream(graphs.prefix_stream):
            slot.prefix_graph.replay()
        graphs.prefix_stream.synchronize()
    checks = validate_slot(backend, graphs, slot, reference, phase=phase)
    return rows, dict(status="passed" if checks["passed"] else "failed", phase=phase, samples=args.count, warmup=args.warmup,
                      warmup_ms=warmup_ms, summary=summarize(rows), checks=checks,
                      gpu_before=before, gpu_after=after, memory=peak)


def measure_encoder_pair(torch, graphs, mode, vision_first):
    """Join both private-slot branches; no CPU wait between submissions."""
    primary = torch.cuda.current_stream()
    begin, end, ae_begin, ae_done, vision_begin, vision_done = (
        torch.cuda.Event(enable_timing=True) for _ in range(6))
    ae, vision = graphs.slots
    host_start = time.perf_counter()
    begin.record(primary)
    graphs.ae_stream.wait_event(begin)
    graphs.prefix_stream.wait_event(begin)

    def launch_ae():
        with torch.cuda.stream(graphs.ae_stream):
            ae_begin.record(graphs.ae_stream)
            ae.ae_graph.replay()
            ae_done.record(graphs.ae_stream)

    def launch_vision():
        with torch.cuda.stream(graphs.prefix_stream):
            if mode == "serial":
                graphs.prefix_stream.wait_event(ae_done)
            vision_begin.record(graphs.prefix_stream)
            vision.vision_graph.replay()
            vision_done.record(graphs.prefix_stream)

    submit_start = time.perf_counter()
    if mode == "overlap" and vision_first:
        launch_vision()
        launch_ae()
    else:
        launch_ae()
        launch_vision()
    submit_end = time.perf_counter()
    primary.wait_event(ae_done)
    primary.wait_event(vision_done)
    end.record(primary)
    wait_start = time.perf_counter()
    end.synchronize()
    host_end = time.perf_counter()
    return dict(gpu_event_ms=begin.elapsed_time(end),
                ae_event_ms=ae_begin.elapsed_time(ae_done),
                vision_event_ms=vision_begin.elapsed_time(vision_done),
                cpu_submit_ms=(submit_end-submit_start)*1000,
                cpu_wait_ms=(host_end-wait_start)*1000,
                cpu_total_ms=(host_end-host_start)*1000,
                submission_order="vision_first" if mode == "overlap" and vision_first else "ae_first")


def run_encoder_pair(torch, backend, graphs, args, name, actual_sms, reference, own_pids):
    # Slot0's complete KV stays immutable while Slot1 writes its own Encoder
    # outputs/metadata. Replaying Vision in Slot0 would race AE's pad masks.
    prime_inputs(torch, graphs, graphs.slots[0], "ae_all")
    prime_inputs(torch, graphs, graphs.slots[1], "vision")
    before = require_same_processes(args.gpu_uuid, own_pids)
    smoke = []
    for mode in ("serial", "overlap"):
        measure_encoder_pair(torch, graphs, mode, True)
        ae_check = validate_slot(backend, graphs, graphs.slots[0], reference, phase="ae_all")
        with torch.cuda.stream(graphs.prefix_stream):
            graphs.slots[1].prefix_graph.replay()
        graphs.prefix_stream.synchronize()
        vision_check = validate_slot(backend, graphs, graphs.slots[1], reference, phase="vision")
        smoke.append(dict(mode=mode, ae=ae_check, vision=vision_check))
        if not ae_check["passed"] or not vision_check["passed"]:
            raise NumericalGateError("Encoder-pair pre-timing numerical gate failed")
    for index in range(args.warmup):
        for mode in ("serial", "overlap"):
            measure_encoder_pair(torch, graphs, mode, bool(index % 2))
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index in range(args.count):
        modes = ("serial", "overlap") if index % 2 == 0 else ("overlap", "serial")
        for order, mode in enumerate(modes):
            row = measure_encoder_pair(torch, graphs, mode, bool(index % 2))
            row.update(quota=name, phase="encoder_pair_"+mode, index=index, pair=index,
                       arm_order=order, num_steps=args.num_steps, actual_sm_counts=actual_sms,
                       ae_slot=0, vision_slot=1, eligible=False)
            rows.append(row)
    peak = memory_snapshot(torch)
    ae_checks = validate_slot(backend, graphs, graphs.slots[0], reference, phase="ae_all")
    # Propagate the other Encoder output outside timing to check its values.
    with torch.cuda.stream(graphs.prefix_stream):
        graphs.slots[1].prefix_graph.replay()
    graphs.prefix_stream.synchronize()
    vision_checks = validate_slot(backend, graphs, graphs.slots[1], reference, phase="vision")
    checks = dict(passed=ae_checks["passed"] and vision_checks["passed"], ae=ae_checks, vision=vision_checks)
    after = require_same_processes(args.gpu_uuid, own_pids)
    outcomes = []
    for mode in ("serial", "overlap"):
        phase = "encoder_pair_"+mode
        selected = [row for row in rows if row["phase"] == phase]
        summary = summarize(selected)
        summary.update({key: statistics([row[key] for row in selected])
                        for key in ("ae_event_ms", "vision_event_ms")})
        outcomes.append(dict(status="passed" if checks["passed"] else "failed", phase=phase,
                             samples=args.count, warmup=args.warmup, summary=summary, checks=checks,
                             smoke=smoke,
                             gpu_before=before, gpu_after=after, memory=peak,
                             scope="Fixed inputs ready; AE slot0 and Vision slot1, no VLM in timing, both completions included. Not request latency."))
    return rows, outcomes


def run_quota(torch, backend, prepared, noise, record, oracle, args, name, minimum,
              library, own_pids, total_sms):
    from .green_context import GreenContextPair
    from .layer_graphs import LayerGraphs

    graphs = runner = pair = prefix_stream = ae_stream = None
    owner_refs = []
    rows = []
    entry = dict(name=name, requested_ae_sm_min=minimum, status="preparing", phases=[],
                 graph_recaptures=0, actual_sm_counts=None, eligible=False,
                 cleanup_passed=False, fatal=False)
    try:
        entry["gpu_before"] = require_same_processes(args.gpu_uuid, own_pids)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        entry["memory_before_capture"] = memory_snapshot(torch)
        if minimum:
            pair = GreenContextPair(library, first_sm_min=minimum, device=backend.device)
            ae_stream, prefix_stream = pair.streams
            entry["green_context"] = dict(pair.info, branch_order=["ae", "prefix"])
            entry["actual_sm_counts"] = dict(ae=pair.info["sm_counts"][0], prefix=pair.info["sm_counts"][1])
        else:
            ae_stream = torch.cuda.Stream(device=backend.device)
            prefix_stream = torch.cuda.Stream(device=backend.device)
            entry["green_context"] = None
            entry["actual_sm_counts"] = dict(ae=total_sms, prefix=total_sms)
        graphs = LayerGraphs(backend, prepared, noise, prefix_stream, ae_stream, ae_layout="per_step")
        owner_refs = [weakref.ref(graphs), *(weakref.ref(slot) for slot in graphs.slots)]
        entry["capture_ms"] = graphs.setup_ms
        entry["memory_after_capture"] = memory_snapshot(torch)
        entry["kv_slot_addresses"] = graphs.addresses
        entry["schedule_audit"] = audit_graph_schedules(graphs, oracle)
        entry["gpu_after_capture"] = require_same_processes(args.gpu_uuid, own_pids)
        runner = LayerRunner(backend, graphs, [record])
        runner.reset(phase="smoke", mode="serial", index=0)
        entry["smoke"] = [runner.predict(index, "serial", phase="smoke", verify=True) for index in range(2)]
        runner.finish_events()
        if not runner.verify_outputs(oracle, MAX_ERROR, MAE_ERROR):
            raise NumericalGateError("Raw/decoded/AE1/KV smoke differs from eager oracle")
        if {row["slot"] for row in entry["smoke"]} != {0, 1}:
            raise RuntimeError("The two-slot smoke did not exercise both captured KV addresses")
        runner.reset(phase="isolated", mode="serial", index=0)
        entry["scope"] = "Owner Pool drained; sole Host replays static slots. Isolated phases use slot0; optional Encoder pairing uses AE slot0 and Vision slot1 with private scratch. No in-flight leases."
        for phase in PHASES:
            phase_rows, outcome = run_phase(torch, backend, graphs, phase, args, name,
                entry["actual_sm_counts"], oracle[0], own_pids)
            rows.extend(phase_rows)
            entry["phases"].append(outcome)
            if not outcome["checks"]["passed"]:
                raise NumericalGateError(f"Isolated {phase} output/KV gate failed: {outcome['checks']}")
            print(f"{name}/{phase}: GPU mean {outcome['summary']['gpu_event_ms']['mean']:.3f} ms", flush=True)
        if args.encoder_pair:
            pair_rows, outcomes = run_encoder_pair(torch, backend, graphs, args, name,
                                                  entry["actual_sm_counts"], oracle[0], own_pids)
            rows.extend(pair_rows)
            entry["phases"].extend(outcomes)
            if not all(item["checks"]["passed"] for item in outcomes):
                raise NumericalGateError("Encoder-pair output or immutable KV changed")
            for item in outcomes:
                print(f"{name}/{item['phase']}: GPU mean {item['summary']['gpu_event_ms']['mean']:.3f} ms", flush=True)
        entry["status"] = "passed"
    except BaseException as error:
        entry.update(status="failed", error=repr(error), traceback=traceback.format_exc(),
                     fatal=not isinstance(error, NumericalGateError))
    finally:
        drained = True
        if graphs is not None:
            try:
                graphs.close()
            except BaseException:
                entry.update(status="failed", fatal=True, cleanup_error=traceback.format_exc())
                drained = False
        # Neither result dictionaries nor exception tracebacks retain tensors,
        # bound replay methods or owners when the external contexts are closed.
        runner = graphs = prefix_stream = ae_stream = None
        gc.collect()
        released = all(reference() is None for reference in owner_refs)
        entry["graph_and_slot_owners_released"] = released
        if not released:
            entry.update(status="failed", fatal=True, cleanup_error="Graph/Slot owner remains alive")
            drained = False
        if pair is not None and drained:
            try:
                pair.close()
                entry["green_context_closed"] = True
            except BaseException:
                entry.update(status="failed", fatal=True, cleanup_error=traceback.format_exc())
                drained = False
        elif pair is not None:
            entry["green_context_closed"] = False
        pair = None
        gc.collect()
        if drained:
            torch.cuda.empty_cache()
            entry["memory_after_cleanup"] = memory_snapshot(torch)
        entry["cleanup_passed"] = drained
    entry["eligible"] = entry["status"] == "passed" and entry["cleanup_passed"]
    for row in rows:
        row["eligible"] = entry["eligible"]
    return entry, rows


def comparisons(entries):
    valid = {entry["name"]: entry for entry in entries if entry["eligible"]}
    if "full_start" not in valid:
        return dict(available=False, reason="Initial full-SM baseline did not pass")
    base = {item["phase"]: item["summary"]["gpu_event_ms"]["mean"] for item in valid["full_start"]["phases"]}
    values = {name: {item["phase"]: dict(
        gpu_event_mean_speedup=base[item["phase"]] / item["summary"]["gpu_event_ms"]["mean"],
        gpu_event_mean_ms=item["summary"]["gpu_event_ms"]["mean"])
        for item in entry["phases"]} for name, entry in valid.items()}
    drift = ({phase: values["full_repeat"][phase]["gpu_event_mean_ms"] / base[phase] - 1
              for phase in base} if "full_repeat" in valid else None)
    return dict(available=True, relative_to_initial_full=values, full_repeat_fractional_drift=drift,
                scope="Exploratory module ratios across sequential quotas; optional Encoder serial/overlap samples are paired within each quota. Not end-to-end speedups.")


def main():
    args = parser().parse_args()
    if args.warmup < 2 or args.count < 1:
        raise ValueError("Require warmup>=2 and count>=1")
    if any(value <= 0 for value in args.ae_sm_mins) or len(set(args.ae_sm_mins)) != len(args.ae_sm_mins):
        raise ValueError("AE SM minima must be distinct positive integers")
    quotas = [("full_start", 0), *((f"ae{value}_prefix_rest", value) for value in args.ae_sm_mins),
              ("full_repeat", 0)]
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(status="preparing", args=vars(args), steps=args.num_steps, rows=[], quotas=[],
                  performance_eligible=False, paired_end_to_end=False,
                  tolerance=dict(max_abs=MAX_ERROR, mae=MAE_ERROR),
                  command=shlex.join([sys.executable, "-m", "preparations.idea2.sm_sensitivity_bench", *sys.argv[1:]]),
                  source_sha256=capture_sources(dest), bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                  environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS",
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  timing_semantics="Fixed input/KV ready beforehand. Isolated phases run individually; optional encoder_pair uses AE slot0 plus Vision slot1 and joins both completions. Event spans include Host gaps and AE-all graph/Event handoff. No preprocessing, decoding or request/trajectory latency in these timings.",
                  quota_order=[dict(name=name, requested_ae_sm_min=count) for name, count in quotas])
    write_result(dest, result)
    try:
        result["gpu_before_init"] = require_idle_gpu(args.gpu_uuid)
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import torch
        from .experiments import load_backend, native_raw
        from .external_events import cross_graph_self_test
        from .green_context import build_library

        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible")
        properties = torch.cuda.get_device_properties(0)
        actual_uuid = str(properties.uuid)
        if actual_uuid.lower().removeprefix("gpu-") != args.gpu_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError("CUDA device UUID differs from requested GPU")
        result.update(actual_cuda_uuid=actual_uuid, physical_sm_count=properties.multi_processor_count)
        if any(value >= properties.multi_processor_count for value in args.ae_sm_mins):
            raise ValueError("Each AE SM minimum must leave a nonempty physical-SM remainder")
        versions = [row for row in smi_rows("--query-gpu=uuid,driver_version") if row[0] == args.gpu_uuid]
        if len(versions) != 1:
            raise RuntimeError("Driver version query did not match the selected UUID")
        result["driver_version"] = versions[0][1]
        result["external_events"] = cross_graph_self_test()
        occupancy = gpu_snapshot(args.gpu_uuid)
        own_pids = {row["pid"] for row in occupancy["processes"]}
        if len(own_pids) != 1 or occupancy["graphics_processes"]:
            raise RuntimeError("Expected exactly one CUDA process and no graphics processes")
        result["own_gpu_processes"] = sorted(own_pids)
        backend = load_backend(args.checkpoint, model_config="pi05_libero", num_steps=args.num_steps)
        result["identity"] = backend.identity
        bank = pickle.loads(Path(args.bank).read_bytes())
        if not bank:
            raise RuntimeError("Input bank is empty")
        record = bank[0]
        result["selected_bank_indices"] = [0]
        oracle = eager_oracle(backend, [record])
        result["oracle"] = [dict(index=row["index"], seed=row["seed"], kv_digest=row["kv_digest"],
                                 times=row["times"], actual_nfe=row["actual_nfe"],
                                 schedule_reference=row["schedule_reference"]) for row in oracle]
        if args.hoist_modulation:
            from .cewp_preparation import PreparedModulationCore
            backend.graph_core_class = PreparedModulationCore
        prepared = backend.prepare(native_raw(record))
        noise = backend.noise(input_schedule(0, 1)[1])
        torch.cuda.current_stream(backend.device).synchronize()
        library = build_library(dest / "green_build")
        result["green_library"] = dict(path=str(library), sha256=hashlib.sha256(library.read_bytes()).hexdigest())
        result["status"] = "measuring"
        write_result(dest, result)
        for name, minimum in quotas:
            entry, rows = run_quota(torch, backend, prepared, noise, record, oracle, args,
                                   name, minimum, library, own_pids, properties.multi_processor_count)
            result["quotas"].append(entry)
            result["rows"].extend(rows)
            result["comparisons"] = comparisons(result["quotas"])
            write_result(dest, result)
            if entry["fatal"] or (name == "full_start" and not entry["eligible"]):
                raise RuntimeError(f"Cannot continue after {name}: {entry.get('error', entry.get('cleanup_error'))}")
        result["gpu_after"] = require_same_processes(args.gpu_uuid, own_pids)
        result["status"] = "passed" if all(entry["eligible"] for entry in result["quotas"]) else "completed_with_rejections"
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        write_result(dest, result)
    if result["status"] == "failed":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
