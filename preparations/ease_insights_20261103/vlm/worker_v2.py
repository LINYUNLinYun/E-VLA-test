"""Technical receipt repair: preserve completed science and audit exported schedule nodes."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import uuid
import numpy as np
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v2_2.common import seed
from .common import atomic_json, read_manifest, sha, cases
from .recovery import runtime_revision, scheduled_nfe, run_with_retries, recover_case


def episode_dir(output, condition, scene):
    return Path(output)/condition/"episodes"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"


def completed(output, condition, scene, manifest_hash):
    directory = episode_dir(output, condition, scene)
    path = directory / "episode_receipt.json"
    if not path.exists():
        return False
    receipt = json.loads(path.read_text())
    if receipt["manifest_hash"] != manifest_hash or receipt["condition"] != condition or receipt["case"] != scene:
        raise ValueError("Mixed episode identity")
    for name, expected in receipt["artifacts"].items():
        if sha(directory/name) != expected:
            raise ValueError("Episode artifact changed")
    return bool(receipt["valid"])


def initial_gate(output, manifest_hash, condition, scene, input_hash, noise_seed):
    path = Path(output)/"initial_observations"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}.json"
    payload = dict(manifest_hash=manifest_hash, case=scene, input_hash=input_hash, seed=noise_seed)
    with lock(str(path)+".lock"):
        if path.exists():
            ref = json.loads(path.read_text())
            if any(ref.get(k) != v for k, v in payload.items()):
                raise ValueError("Paired initial observation/noise mismatch")
        else:
            atomic_json(path, dict(payload, condition=condition))
    return dict(**payload, passed=True, reference=str(path))


def finalize(output, condition, scene, manifest, directory=None):
    directory = Path(directory) if directory is not None else episode_dir(output, condition, scene)
    result = json.loads((directory/"result.json").read_text())
    done = json.loads((directory/"complete.json").read_text())
    identity = json.loads((directory/"identity.json").read_text())
    cfg = next(c for c in manifest["conditions"] if c["id"] == condition)
    if not result["valid"] or not done["valid"] or any(x["manifest_hash"] != manifest["manifest_hash"] for x in (result, done, identity)):
        raise ValueError("Invalid or mixed-manifest episode")
    if result["case"] != scene or identity["case"] != scene or result["condition"] != condition or identity["config"] != cfg:
        raise ValueError("Case or condition identity differs")
    expected_init = next(r for r in manifest["official_init_availability"]["cases"] if r["suite"] == scene["suite"] and r["task"] == scene["task"])
    if identity["init_vector_sha256"] != expected_init["init_sha256"][str(scene["init"])]:
        raise ValueError("Official initial state changed")
    protocol = identity["protocol"]
    if protocol["manifest_hash"] != manifest["manifest_hash"] or protocol["condition_id"] != condition or not protocol["real_input_parity_passed"]:
        raise ValueError("Incorrect or ungated service")
    rows = [json.loads(s) for s in (directory/"infer.jsonl").read_text().splitlines()]
    if len(rows) != result["predictions"] or not rows:
        raise ValueError("Incomplete inference rows")
    step = 0
    for index, row in enumerate(rows):
        expected_seed = seed(scene["suite"], scene["task"], scene["init"], index)
        expected_noise = hashlib.sha256(np.random.default_rng(expected_seed).standard_normal((50,32), dtype=np.float32).tobytes()).hexdigest()
        full = condition == "full_d10" or index % 3 == 0
        skip = [] if full else cfg["skip"]
        audit = row["route"]["execution_audit"]
        if (row["round"] != index or row["step"] != step or row["seed"] != expected_seed or row["model_noise_sha256"] != expected_noise or
            row["route"]["refresh"] != full or row["route"]["skip_layers"] != skip or audit["kv_only_layers"] != skip or
            audit["qkv_layers"] != [l for l in range(1,19) if l not in skip] or
            audit["hidden_update_layers"] != [l for l in range(1,18) if l not in skip] or
            audit["captured_layers"] != (cfg["skip"] if full else []) or
            audit["patched_layers"] != skip or audit["grad_enabled"] or audit["selector_calls"] != 0 or
            scheduled_nfe(row["timing"]["nodes"]) != (10 if full else 2)):
            raise ValueError("Round, execution, or paired noise audit mismatch")
        step += row["executed_length"]
    if step != result["actions"]:
        raise ValueError("Execution length mismatch")
    gate = initial_gate(output, manifest["manifest_hash"], condition, scene, rows[0]["input_hash"], rows[0]["seed"])
    receipt = dict(valid=True, manifest_hash=manifest["manifest_hash"], condition=condition, case=scene, initial_gate=gate,
        success=result["success"], executed_steps=step, predictions=len(rows), counts=result["counts"],
        technical_runtime_revision=runtime_revision(manifest),
        ae_audit=dict(scope="exported schedule nodes", scheduled_NFE_counts=dict(Counter(scheduled_nfe(r["timing"]["nodes"]) for r in rows)),
                      actual_NFE_exported=False, note="Frozen backend exports time nodes only, not the internal actual_NFE counter"),
        wall_s=result["wall_s"], model_request_ms_total=sum(r["timing"]["policy_wall_ms"] for r in rows),
        rpc_ms_total=sum(r["rpc_ms"] for r in rows), gpu_uuid=protocol["gpu_uuid"], worker_pid=os.getpid(),
        artifacts={name: sha(directory/name) for name in ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz")})
    atomic_json(directory/"episode_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True); parser.add_argument("--condition", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stage", default="discovery", choices=("discovery", "confirm"))
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(); output = Path(args.output).resolve()
    manifest = read_manifest(output/"manifest.json", verify_sources=True)
    cfg = next(c for c in manifest["conditions"] if c["id"] == args.condition)
    scenes = cases(manifest, args.stage, args.shard_index, args.shards)
    from preparations.idea1_v3_4 import rollout
    from preparations.idea1_v3_1.rollout import Environments
    from preparations.idea1_v2_1.runtime_client import input_digest
    class SeededEnvironments(Environments):
        def acquire(self, suite, task):
            random.seed(7); np.random.seed(7)
            return super().acquire(suite, task)
    pool = SeededEnvironments(reuse=False)
    original_rpc, current = rollout.rpc, {}
    def gated_rpc(port, request):
        if request.get("op") == "predict" and not current.get("checked"):
            initial_gate(output, manifest["manifest_hash"], cfg["id"], current["scene"],
                input_digest(request["obs"], request["instruction"]), request["seed"])
            current["checked"] = True
        response = original_rpc(port, request)
        if request.get("op") == "health":
            p = response["protocol"]
            if p["manifest_hash"] != manifest["manifest_hash"] or p["condition_id"] != cfg["id"] or not p["real_input_parity_passed"]:
                raise ValueError("Incorrect or ungated service")
        return response
    rollout.rpc = gated_rpc
    try:
        for scene in scenes:
            directory = episode_dir(output, cfg["id"], scene)
            with lock(str(directory)+".lock"):
                if completed(output, cfg["id"], scene, manifest["manifest_hash"]):
                    continue
                recovered = recover_case(output, cfg["id"], scene, manifest, finalize, already_locked=True)
                if recovered is not None:
                    print(json.dumps(recovered), flush=True); continue
                current.clear(); current.update(scene=scene)
                result = run_with_retries(directory, manifest["technical_max_attempts"],
                    lambda: rollout.episode(directory, args.port, scene, cfg, pool, manifest["manifest_hash"]),
                    lambda: finalize(output, cfg["id"], scene, manifest), pool.close,
                    before_attempt=lambda: (current.clear(), current.update(scene=scene)))
                print(json.dumps(result), flush=True)
    finally:
        rollout.rpc = original_rpc; pool.close()


if __name__ == "__main__":
    main()
