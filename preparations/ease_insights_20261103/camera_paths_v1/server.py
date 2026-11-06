"""Within-suite service; finite old-bank numerical checks have explicit scope."""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback
from .common import atomic_json, read_manifest, validate_condition, planned_config, validate_output
from .common import GATE_SUITES, SUITES, runtime_revision
from .rollout import SETTLING_ACTIONS
from .common import LIMITS

GATE_BANK_SUITES = tuple(GATE_SUITES)


def finite_bank_scope(manifest):
    from preparations.ease_insights_20261103.vlm_local_search_v1.common import SUITES as FROZEN_BANK_SUITES
    if GATE_BANK_SUITES != tuple(FROZEN_BANK_SUITES) or tuple(manifest["suites"]) != tuple(SUITES):
        raise ValueError("Finite gate bank and new evaluation suites must retain their separate identities")
    return dict(gate_bank_suites=list(GATE_BANK_SUITES), evaluation_suites=list(SUITES),
                gate_covers_new_case_universe=False, gate_covers_all_new_prompts=False,
                scope="Four fixed-bank observations and all four fixed camera configurations; not all closed-loop inputs",
                case_gate="Actual official state, BDDL, prompt, first observation and noise are checked and paired per new case before prediction")


def finite_gate(backend, manifest, output):
    from .gate import run
    scope = finite_bank_scope(manifest)
    result = run(backend, manifest, output)
    result.update(within_suite_scope=scope)
    atomic_json(Path(output)/"gate.json", result)
    return result


def finite_benchmark(backend, manifest, output):
    from .benchmark import run
    scope = finite_bank_scope(manifest)
    result = run(backend, manifest, output)
    result.update(within_suite_scope=scope)
    atomic_json(Path(output)/"fixed_input_cost.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not args.gpu_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise ValueError("Server CUDA_VISIBLE_DEVICES must equal its explicitly assigned GPU UUID")
    manifest = read_manifest(args.manifest, verify_sources=True)
    validate_output(Path(args.manifest).resolve().parent, manifest)
    cfg = planned_config(manifest, args.condition)
    validate_condition(cfg)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(out / "qk_receipt.json"))
    import torch
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    from .backend import Backend
    from .policy import Policy
    from preparations.idea1_v2_1.runtime_client import recv_exact
    from preparations.idea1_v3_1.benchmark import load_banks
    backend = Backend()
    if backend.protocol["checkpoint_sha256"] != manifest["checkpoints"]["pi0"]["sha256"]:
        raise ValueError("Loaded checkpoint differs from frozen experiment")
    load_s = time.perf_counter() - begin
    tick = time.perf_counter(); gate = finite_gate(backend, manifest, out); gate_s = time.perf_counter() - tick
    qk = json.loads((out / "qk_receipt.json").read_text())
    if (qk.get("passed") is not True or qk.get("unsafe_legacy_kernel_executed") is not False or
            qk.get("atol") != .0009765625 or qk.get("rtol") != .0078125):
        raise ValueError("Actual QK call failed the unchanged predefined reference tolerance")
    tick = time.perf_counter(); cost = finite_benchmark(backend, manifest, out); benchmark_s = time.perf_counter() - tick
    policy = Policy(backend)
    policy.configure(cfg)
    records, _ = load_banks([manifest["bank"]["path"]])
    warmups = []
    for index in range(20):
        row = records[index % len(records)]
        # Per-prompt cache history must never cross task boundaries.
        if index == 0 or row["instruction"] != records[(index-1) % len(records)]["instruction"]:
            policy.configure(cfg)
        tick = time.perf_counter()
        reply = policy.predict(row["obs"], row["instruction"], row["suite"], int(row["seed"]))
        policy.acknowledge(reply["actions"][:5])
        warmups.append(dict(index=index, wall_ms=(time.perf_counter()-tick)*1000, route=reply["route"]["path"]))
    policy.configure(cfg)
    protocol = dict(backend.protocol, experiment="ease_camera_paths_v1", condition_id=cfg["id"], config=cfg,
        manifest_hash=manifest["manifest_hash"], gpu_uuid=args.gpu_uuid, server_pid=os.getpid(),
        real_input_parity_passed=gate["passed"], qk_reference_passed=qk["passed"], selector_calls=0, gradients_enabled=False,
        D=cfg["D"], refresh_D=10, fixed_skip_layers=cfg["skip"],
        policy_action_limits=dict(LIMITS), settling_actions=SETTLING_ACTIONS,
        evaluation_suites=list(SUITES), gate_bank_scope=finite_bank_scope(manifest),
        technical_runtime_revision=runtime_revision(manifest), ae_full_layers=18,
        internal_actual_NFE_exported=True, AE_numerical_gate_passed=gate["passed"], fixed_ae_skip_layers=cfg["ae_skip"],
        camera_numerical_gate_passed=gate["passed"], actual_vision_input_batch_exported=True, fixed_fresh_camera=cfg["fresh_camera"],
        service_load_s=load_s, gate_s=gate_s, fixed_input_cost_s=benchmark_s,
        fixed_input_cost_record=str(out / "fixed_input_cost.json"), total_preparation_s=time.perf_counter()-begin)
    atomic_json(out / "protocol.json", protocol)
    atomic_json(out / "preparation.json", dict(service_load_s=load_s, gate_s=gate_s, fixed_input_cost_s=benchmark_s,
                fixed_input_cost_warmup_requests=cost["total_warmup_requests"],
                fixed_input_cost_measured_requests=cost["total_measured_requests"], warmup=warmups,
                total_preparation_s=time.perf_counter()-begin, cold_costs_excluded_from_rollout=True))
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", args.port)); server.listen(4)
        print("READY", flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    request = pickle.loads(recv_exact(connection, struct.unpack("!I", recv_exact(connection, 4))[0]))
                    op = request.pop("op")
                    if op == "health":
                        result = dict(protocol=protocol)
                    elif op == "configure":
                        if request["config"] != cfg:
                            raise ValueError("A service is bound to one frozen condition")
                        result = policy.configure(**request)
                    elif op == "predict":
                        result = policy.predict(**request)
                    elif op == "ack":
                        result = policy.acknowledge(**request)
                    else:
                        raise ValueError(op)
                    result["ok"] = True
                except Exception:
                    result = dict(ok=False, error=traceback.format_exc())
                payload = pickle.dumps(result, protocol=5)
                connection.sendall(struct.pack("!I", len(payload)) + payload)


if __name__ == "__main__":
    main()
