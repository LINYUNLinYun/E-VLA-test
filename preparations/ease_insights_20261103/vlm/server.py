"""Single-rollout stateful server on one explicitly assigned GPU UUID and port."""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback
from .common import atomic_json, read_manifest, validate_condition


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
    cfg = next(c for c in manifest["conditions"] if c["id"] == args.condition)
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
    from .gate import run
    from preparations.idea1_v2_1.runtime_client import recv_exact
    from preparations.idea1_v3_1.benchmark import load_banks
    backend = Backend()
    if backend.protocol["checkpoint_sha256"] != manifest["checkpoints"]["pi0"]["sha256"]:
        raise ValueError("Loaded checkpoint differs from frozen experiment")
    load_s = time.perf_counter() - begin
    tick = time.perf_counter(); gate = run(backend, manifest, out); gate_s = time.perf_counter() - tick
    from .benchmark import run as benchmark
    tick = time.perf_counter(); cost = benchmark(backend, manifest, out); benchmark_s = time.perf_counter() - tick
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
    protocol = dict(backend.protocol, experiment="ease_fixed_vlm_masks", condition_id=cfg["id"], config=cfg,
        manifest_hash=manifest["manifest_hash"], gpu_uuid=args.gpu_uuid, server_pid=os.getpid(),
        real_input_parity_passed=gate["passed"], selector_calls=0, gradients_enabled=False,
        D=cfg["D"], refresh_D=10, fixed_skip_layers=cfg["skip"],
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
