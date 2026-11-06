"""Diagnostic-only D10 per-iteration audit of partitioned AE1 -> full-SM tail.

Capture retains all velocity/action outputs and cloned time nodes. Its extra
memory and kernels invalidate performance comparisons. CPU oracle arrays are
saved before allocating the only two complete graph KV slots. This script does
not modify production captures or create a second runtime/third full KV cache.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import os
from pathlib import Path
import pickle
import shlex
import sys
import time
import traceback

import numpy as np

from .layer_bench import (LayerRunner, audit_graph_schedules, capture_sources,
                          difference, gpu_snapshot, input_schedule,
                          native_schedule_reference, require_idle_gpu,
                          require_same_processes, smi_rows, validate_schedule,
                          within_tolerance, write_result)


MAX_ERROR, MAE_ERROR = 1e-5, 1e-6
NUM_STEPS = 10


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    result.add_argument("--record-count", type=int, default=13,
                        help="First N matching-task real observations; use 2 for a bounded initial gate")
    result.add_argument("--hoist-modulation", action="store_true")
    return result


def tensor_packet(value):
    """Preserve original dtype bytes as well as numeric values, including BF16."""
    import torch
    cpu = value.detach().cpu()
    return dict(values=cpu.float().numpy().copy(),
                bits=cpu.contiguous().reshape(-1).view(torch.uint8).numpy().copy(),
                dtype=str(cpu.dtype))


def sequence_packet(values):
    packets = [tensor_packet(value) for value in values]
    return dict(values=np.stack([packet["values"] for packet in packets]),
                bits=np.stack([packet["bits"] for packet in packets]),
                dtypes=[packet["dtype"] for packet in packets])


def packet_at(packet, index):
    return dict(values=packet["values"][index], bits=packet["bits"][index], dtype=packet["dtypes"][index])


def compare_packet(reference, actual, *, exact=False):
    if reference["values"].shape != actual["values"].shape:
        return dict(passed=False, bitwise=False, finite=False, mae=None, max_abs=None,
                    reference_shape=list(reference["values"].shape), actual_shape=list(actual["values"].shape))
    error = difference(reference["values"], actual["values"])
    same_dtype = reference["dtype"] == actual["dtype"]
    bitwise = same_dtype and np.array_equal(reference["bits"], actual["bits"])
    return dict(error, dtype_equal=same_dtype, bitwise=bool(bitwise),
                passed=bool(same_dtype and within_tolerance(error, MAX_ERROR, MAE_ERROR)
                            and (bitwise or not exact)))


def numpy_packet(value):
    value = np.asarray(value)
    return dict(values=value.copy(), bits=np.ascontiguousarray(value).reshape(-1).view(np.uint8).copy(),
                dtype=str(value.dtype))


def compare_iterations(reference, actual, captured_nfe):
    keys = ("inputs", "velocities", "times", "actions")
    counts = {key: len(actual[key]["values"]) for key in keys}
    if captured_nfe != NUM_STEPS or any(count != NUM_STEPS for count in counts.values()):
        return dict(passed=False, captured_nfe=captured_nfe, counts=counts,
                    error="Expected all ten iteration inputs/velocities/times/actions")
    steps = []
    for index in range(NUM_STEPS):
        fields = {key: compare_packet(packet_at(reference[key], index), packet_at(actual[key], index),
                                      exact=key == "times") for key in keys}
        steps.append(dict(iteration=index + 1, passed=all(item["passed"] for item in fields.values()),
                          fields=fields))
    final = compare_packet(reference["final"], actual["final"])
    decoded = compare_packet(numpy_packet(reference["decoded"]), numpy_packet(actual["decoded"]))
    return dict(passed=all(step["passed"] for step in steps) and final["passed"] and decoded["passed"],
                captured_nfe=captured_nfe, counts=counts, iterations=steps,
                final=final, decoded=decoded,
                all_bitwise=all(item["bitwise"] for step in steps for item in step["fields"].values())
                            and final["bitwise"] and decoded["bitwise"])


def save_arrays(dest, name, data):
    arrays, dtypes = {}, {}
    for key in ("inputs", "velocities", "times", "actions", "final"):
        arrays[key + "_values"] = data[key]["values"]
        arrays[key + "_bits"] = data[key]["bits"]
        dtypes[key] = data[key].get("dtypes", data[key].get("dtype"))
    arrays["decoded"] = data["decoded"]
    path = dest / (name + ".npz")
    np.savez(path, **arrays)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shapes={key: list(value.shape) for key, value in arrays.items()}, dtypes=dtypes)


def eager_iteration_oracle(backend, records, dest):
    import torch
    from .experiments import native_raw
    from .graph_bench import cache_digest

    expected = native_schedule_reference(NUM_STEPS, backend.device)
    oracle = []
    for index, record in enumerate(records):
        _, seed = input_schedule(index, len(records))
        prepared, noise = backend.prepare(native_raw(record)), backend.noise(seed)
        prefix = backend.build_prefix(prepared)
        original = backend.velocity
        inputs, velocities, times = [], [], []

        def velocity(selected, current, actions, timestep):
            inputs.append(actions.detach().clone())
            if timestep.numel() != 1:
                raise RuntimeError("This fixed-input diagnostic requires batch size one")
            # Native hook sees an expanded batch-one value; Graph time_nodes
            # records its scalar source. Preserve the same FP32 bytes/shape.
            times.append(timestep.detach().reshape(-1)[0].clone())
            value = original(selected, current, actions, timestep)
            velocities.append(value.detach().clone())
            return value

        try:
            backend.velocity = velocity
            final = backend.generate(prefix, prepared, noise)
        finally:
            backend.velocity = original
        actual_times = [float(value.reshape(-1)[0].cpu()) for value in times]
        validate_schedule(actual_times, len(velocities), expected["times"], label="D10 iteration oracle")
        # The next call's actual input is the preceding original Euler result;
        # do not recompute updates using Python/NumPy arithmetic.
        after = inputs[1:] + [final]
        arrays = dict(inputs=sequence_packet(inputs), velocities=sequence_packet(velocities),
                      times=sequence_packet(times), actions=sequence_packet(after),
                      final=tensor_packet(final), decoded=np.asarray(backend.output(prepared, final)).copy())
        oracle.append(dict(index=index, seed=seed, times=actual_times, actual_nfe=len(velocities),
                           schedule_reference=expected, kv_digest=cache_digest(prefix),
                           normalized=arrays["final"]["values"][0].copy(),
                           first_actions=arrays["actions"]["values"][0, 0].copy(),
                           decoded=arrays["decoded"], arrays=arrays,
                           artifact=save_arrays(dest, f"oracle_{index:03d}", arrays)))
        torch.cuda.current_stream(backend.device).synchronize()
        del prefix, prepared, noise, final, inputs, velocities, times, after, arrays
        print(f"Iteration oracle {index + 1}/{len(records)} saved to CPU", flush=True)
    gc.collect()
    return oracle


@contextmanager
def reuse_prepared_addresses(backend, raw):
    """Keep one external Prepared address set while alternating real contents."""
    from .runtime_graph import input_tensors

    original = backend.prepare
    shared = original(raw)
    target = input_tensors(shared)
    addresses = [value.data_ptr() for value in target]
    receipt = dict(addresses=addresses, preparations_reused=0)

    def prepare(changed):
        replacement = original(changed)
        source = input_tensors(replacement)
        if len(source) != len(target):
            raise RuntimeError("In-place audit input layout changed")
        for left, right in zip(target, source, strict=True):
            if (left.shape, left.dtype, left.device) != (right.shape, right.dtype, right.device):
                raise RuntimeError("In-place audit input shape/dtype/device changed")
            left.copy_(right)
        if [value.data_ptr() for value in input_tensors(shared)] != addresses:
            raise RuntimeError("In-place audit changed an external input address")
        receipt["preparations_reused"] += 1
        return shared

    try:
        backend.prepare = prepare
        yield receipt
    finally:
        backend.prepare = original


def graph_iteration_arrays(backend, slot):
    # Inputs follow the same generate() recurrence: input_1=noise and every
    # later input is the preceding retained Euler output. This is reported as
    # derived input evidence, not as an independent instrumented input tensor.
    inputs = [slot.noise, *slot.iteration_actions[:-1]]
    return dict(inputs=sequence_packet(inputs), velocities=sequence_packet(slot.iteration_velocities),
                times=sequence_packet(slot.captured_times), actions=sequence_packet(slot.iteration_actions),
                final=tensor_packet(slot.actions), decoded=np.asarray(backend.output(slot.ae_prepared, slot.actions)).copy())


def main():
    args = parser().parse_args()
    if not 2 <= args.record_count <= 13:
        raise ValueError("Require 2 through 13 real observations")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(status="preparing", args=vars(args), steps=NUM_STEPS, rows=[],
                  diagnostic_only=True, performance_eligible=False,
                  tolerance=dict(max_abs=MAX_ERROR, mae=MAE_ERROR, time_nodes="exact FP32 bytes"),
                  scope="Partitioned AE1 (requested 16 SM) + full-SM AE2..10, same-frame KV. Extra captured diagnostic outputs/time clones exclude all timings from performance results.",
                  graph_input_evidence="input_1 from slot.noise; input_i from retained Euler output i-1; velocities, times and Euler outputs read directly.",
                  command=shlex.join([sys.executable, "-m", "preparations.idea2.d10_iteration_audit", *sys.argv[1:]]),
                  source_sha256=capture_sources(dest), bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                  environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS",
                               "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    write_result(dest, result)
    graphs = runner = pair = prefix_stream = ae_stream = tail_stream = slot = None
    try:
        result["gpu_before_init"] = require_idle_gpu(args.gpu_uuid)
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import torch
        from .experiments import load_backend, native_raw
        from .external_events import cross_graph_self_test
        from .green_context import GreenContextPair, build_library
        from .layer_graphs import LayerGraphs
        from .layer_lifecycle import audit

        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible")
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        if actual_uuid.lower().removeprefix("gpu-") != args.gpu_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError("Visible CUDA UUID differs from requested GPU")
        result["actual_cuda_uuid"] = actual_uuid
        versions = [row for row in smi_rows("--query-gpu=uuid,driver_version") if row[0] == args.gpu_uuid]
        if len(versions) != 1:
            raise RuntimeError("Driver version query did not match selected GPU")
        result["driver_version"] = versions[0][1]
        result["external_events"] = cross_graph_self_test()
        occupancy = gpu_snapshot(args.gpu_uuid)
        own_pids = {row["pid"] for row in occupancy["processes"]}
        if len(own_pids) != 1 or occupancy["graphics_processes"]:
            raise RuntimeError("Expected one CUDA process and no graphics processes")
        result["own_gpu_processes"] = sorted(own_pids)
        backend = load_backend(args.checkpoint, model_config="pi05_libero", num_steps=NUM_STEPS)
        result["identity"] = backend.identity
        bank = pickle.loads(Path(args.bank).read_bytes())
        indices = [index for index, record in enumerate(bank) if record["instruction"] == bank[0]["instruction"]]
        if len(indices) < args.record_count:
            raise RuntimeError("Not enough matching-task real observations; no synthetic fallback")
        indices = indices[:args.record_count]
        records = [bank[index] for index in indices]
        result["selected_bank_indices"] = indices
        oracle = eager_iteration_oracle(backend, records, dest)
        result["oracle"] = [{key: row[key] for key in ("index", "seed", "times", "actual_nfe",
                            "schedule_reference", "kv_digest", "artifact")} for row in oracle]
        write_result(dest, result)
        # The preceding routine has dropped all eager Prefix/Cache tensors;
        # only its CPU arrays survive when the two graph slots are allocated.
        if args.hoist_modulation:
            from .cewp_preparation import PreparedModulationCore
            backend.graph_core_class = PreparedModulationCore
        prepared = backend.prepare(native_raw(records[0]))
        noise = backend.noise(input_schedule(0, len(records))[1])
        library = build_library(dest / "green_build")
        pair = GreenContextPair(library, first_sm_min=16, device=backend.device)
        ae_stream, prefix_stream = pair.streams
        tail_stream = torch.cuda.Stream(device=backend.device)
        result["green_context"] = dict(pair.info, branch_order=["ae1", "prefix"], tail_policy="full")
        graphs = LayerGraphs(backend, prepared, noise, prefix_stream, ae_stream,
                             ae_layout="per_step", tail_stream=tail_stream, tail_gate=True,
                             capture_intermediates=True)
        del prepared, noise
        result["kv_slot_addresses"] = graphs.addresses
        result["graph_schedule_audit"] = audit_graph_schedules(graphs, oracle)
        result["gpu_after_capture"] = require_same_processes(args.gpu_uuid, own_pids)
        runner = LayerRunner(backend, graphs, records)
        runner.reset(phase="iteration_audit", mode="alternating", index=0)
        result["status"] = "auditing"
        with reuse_prepared_addresses(backend, native_raw(records[0])) as receipt:
            for index in range(2 * len(records)):
                if index and index % len(records) == 0:
                    runner.reset(phase="iteration_audit", mode="alternating", index=index)
                modes = ("serial", "overlap") if index % 2 == 0 else ("overlap", "serial")
                for order, mode in enumerate(modes):
                    row = runner.predict(index, mode, phase="iteration_audit", pair=index,
                                         arm_order=order, verify=True)
                    slot = graphs.slots[row["slot"]]
                    actual = graph_iteration_arrays(backend, slot)
                    row["iteration_artifact"] = save_arrays(dest, f"graph_{len(result['rows']):03d}_{mode}", actual)
                    row["iteration_checks"] = compare_iterations(oracle[row["frame"]]["arrays"], actual, slot.actual_nfe)
                    row["diagnostic_only"] = True
                    row["external_input_addresses_unchanged"] = True
                    runner.finish_events()
                    row["base_checks_passed"] = runner.verify_outputs(oracle, MAX_ERROR, MAE_ERROR)
                    row["all_checks_passed"] = row["iteration_checks"]["passed"] and row["base_checks_passed"]
                    result["rows"].append(row)
                    slot = actual = None
                result["gpu_latest"] = require_same_processes(args.gpu_uuid, own_pids)
                write_result(dest, result)
            result["inplace_inputs"] = receipt
        if {row["slot"] for row in result["rows"]} != {0, 1}:
            raise RuntimeError("Iteration audit did not cover both KV slots")
        if not all(row["all_checks_passed"] for row in result["rows"]):
            raise RuntimeError("One or more D10 iteration checks failed; saved arrays are retained")
        result["lifecycle_audit"] = audit(backend, graphs, runner.pool, records, oracle,
                                           max_error=MAX_ERROR, mae_error=MAE_ERROR)
        result["resets"] = runner.reset_rows
        result["gpu_after"] = require_same_processes(args.gpu_uuid, own_pids)
        result["all_bitwise"] = all(row["iteration_checks"]["all_bitwise"] for row in result["rows"])
        result["status"] = "passed"
    except BaseException as error:
        result.update(status="failed", error=repr(error), traceback=traceback.format_exc())
    finally:
        drained = True
        if graphs is not None:
            try:
                graphs.close()
            except BaseException:
                result.update(status="failed", cleanup_error=traceback.format_exc())
                drained = False
        slot = runner = graphs = prefix_stream = ae_stream = tail_stream = None
        gc.collect()
        if pair is not None and drained:
            try:
                pair.close()
                result["green_context_closed"] = True
            except BaseException:
                result.update(status="failed", cleanup_error=traceback.format_exc())
        elif pair is not None:
            result["green_context_closed"] = False
        write_result(dest, result)
    if result["status"] != "passed":
        raise RuntimeError(result.get("error", "Diagnostic cleanup failed"))


if __name__ == "__main__":
    main()
