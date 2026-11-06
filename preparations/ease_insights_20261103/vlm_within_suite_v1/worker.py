"""Strict LIBERO90 receipts and case-local recovery for the new experiment."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v2_2.common import seed
from .common import atomic_json, read_manifest, sha
from .common import cases, planned_config, runtime_revision, validate_output, stage_for
from .rollout import MAX_POLICY_ACTIONS, SETTLING_ACTIONS
from preparations.ease_insights_20261103.vlm.recovery import scheduled_nfe, run_with_retries, recover_case


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


def initial_gate(output, manifest_hash, condition, scene, input_hash, noise_seed, *, assets=None):
    path = Path(output)/"initial_observations"/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}.json"
    payload = dict(manifest_hash=manifest_hash, case=scene, input_hash=input_hash, seed=noise_seed)
    if assets is not None:
        payload["physical_assets"] = assets
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
    cfg = planned_config(manifest, condition)
    if not result["valid"] or not done["valid"] or any(x["manifest_hash"] != manifest["manifest_hash"] for x in (result, done, identity)):
        raise ValueError("Invalid or mixed-manifest episode")
    if result["case"] != scene or identity["case"] != scene or result["condition"] != condition or identity["config"] != cfg or result["config"] != cfg:
        raise ValueError("Case or condition identity differs")
    assets = verify_physical_identity(manifest, scene, identity)
    protocol = identity["protocol"]
    if (protocol["manifest_hash"] != manifest["manifest_hash"] or protocol["condition_id"] != condition or not protocol["real_input_parity_passed"] or
        protocol.get("qk_reference_passed") is not True or protocol.get("config") != cfg or
        protocol["checkpoint_sha256"] != manifest["checkpoints"]["pi0"]["sha256"] or
        protocol.get("policy_action_limit") != MAX_POLICY_ACTIONS or protocol.get("settling_actions") != SETTLING_ACTIONS):
        raise ValueError("Incorrect or ungated service")
    rows = [json.loads(s) for s in (directory/"infer.jsonl").read_text().splitlines()]
    if len(rows) != result["predictions"] or not rows:
        raise ValueError("Incomplete inference rows")
    validate_action_artifacts(directory, rows, result)
    step = 0
    history = []
    for index, row in enumerate(rows):
        expected_seed = seed(scene["suite"], scene["task"], scene["init"], index)
        expected_noise = hashlib.sha256(np.random.default_rng(expected_seed).standard_normal((50,32), dtype=np.float32).tobytes()).hexdigest()
        full = condition == "full_d10" or index % 3 == 0
        gap = None if len(history) < 2 else history[-1]-history[-2]
        ratio = 0. if gap is None else (step-history[-1])/gap
        if full and cfg["skip"]:
            history = [*history, step][-2:]
        route = row["route"]
        if (route["history_steps"] != history or route["has_second_history"] != (gap is not None) or
            route["extrapolation_ratio"] != ratio or route["compensation"] != cfg["compensation"] or
            route.get("path") != ("full" if full else "extrapolate") or
            route.get("selector_calls") != 0 or route.get("capture_layers") != (cfg["skip"] if full else [])):
            raise ValueError("Own-path residual history or compensation differs")
        full_count = index+1 if condition == "full_d10" else index//3+1
        if row["counts"] != dict(full=full_count,draft=index+1-full_count,verify=0):
            raise ValueError("Exported policy route counters differ")
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
        if not isinstance(row["executed_length"], int) or not 0 < row["executed_length"] <= 5:
            raise ValueError("Invalid executed action count")
        step += row["executed_length"]
    from preparations.idea1_v2_1.runtime_client import input_digest
    first_input = directory / "input_0000.npz"
    if sha(first_input) != rows[0]["input_file_sha256"]:
        raise ValueError("Initial observation artifact changed")
    with np.load(first_input, allow_pickle=False) as saved:
        actual_input = input_digest({k: saved[k] for k in saved.files if k not in ("instruction", "seed")}, str(saved["instruction"].item()))
        if (actual_input != rows[0]["input_hash"] or int(saved["seed"]) != rows[0]["seed"] or
            str(saved["instruction"].item()) != identity["prompt"]):
            raise ValueError("Initial observation content/noise differs from its request")
    if step != result["actions"]:
        raise ValueError("Execution length mismatch")
    if result["counts"] != rows[-1]["counts"]:
        raise ValueError("Final route counters differ")
    gate = initial_gate(output, manifest["manifest_hash"], condition, scene, rows[0]["input_hash"], rows[0]["seed"], assets=assets)
    receipt = dict(valid=True, manifest_hash=manifest["manifest_hash"], condition=condition, case=scene, initial_gate=gate,
        policy_action_limit=MAX_POLICY_ACTIONS, settling_actions=SETTLING_ACTIONS,
        success=result["success"], executed_steps=step, predictions=len(rows), counts=result["counts"],
        technical_runtime_revision=runtime_revision(manifest),
        ae_audit=dict(scope="exported schedule nodes", scheduled_NFE_counts=dict(Counter(scheduled_nfe(r["timing"]["nodes"]) for r in rows)),
                      actual_NFE_exported=False, note="Frozen backend exports time nodes only, not the internal actual_NFE counter"),
        wall_s=result["wall_s"], model_request_ms_total=sum(r["timing"]["policy_wall_ms"] for r in rows),
        rpc_ms_total=sum(r["rpc_ms"] for r in rows), gpu_uuid=protocol["gpu_uuid"], worker_pid=os.getpid(),
        artifacts={name: sha(directory/name) for name in ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz", "input_0000.npz")})
    atomic_json(directory/"episode_receipt.json", receipt)
    return receipt


def validate_action_artifacts(directory, rows, result):
    actions = result["actions"]
    if (type(actions) is not int or not 0 < actions <= MAX_POLICY_ACTIONS or
        result.get("policy_action_limit") != MAX_POLICY_ACTIONS or result.get("settling_actions") != SETTLING_ACTIONS or
        result.get("termination") != ("success" if result["success"] else "timeout") or
        (not result["success"] and actions != MAX_POLICY_ACTIONS)):
        raise ValueError("Episode termination or uniform policy action budget differs")
    telemetry = [json.loads(line) for line in (directory/"telemetry.jsonl").read_text().splitlines()]
    if len(telemetry) != actions+1 or telemetry[0]["step"] != 0 or telemetry[0]["action"] is not None:
        raise ValueError("Incomplete policy action telemetry")
    step = 0
    with np.load(directory/"arrays.npz", allow_pickle=False) as saved:
        for index, row in enumerate(rows):
            decoded, raw = saved[f"actions_{index}"], saved[f"raw_{index}"]
            if (decoded.shape != (50,7) or raw.shape != (50,32) or
                not np.isfinite(decoded).all() or not np.isfinite(raw).all() or row.get("execute_length") != 5):
                raise ValueError("Raw/decoded output shape or execution prefix differs")
            count = row["executed_length"]
            if type(count) is not int or not 0 < count <= 5 or (index < len(rows)-1 and count != 5):
                raise ValueError("Invalid executed prefix length")
            for command in decoded[:count]:
                step += 1
                observed = telemetry[step]
                if observed["step"] != step or not np.array_equal(np.asarray(observed["action"],np.float32),command):
                    raise ValueError("Executed action differs from acknowledged model prefix")
    if step != actions:
        raise ValueError("Action artifact count differs")


def physical_assets(identity):
    return {k: identity[k] for k in ("init_vector_sha256", "bddl_sha256", "prompt", "env_seed", "settling", "policy_action_limit", "control_hz")}


def verify_physical_identity(manifest, scene, identity):
    expected = next(r for r in manifest["official_init_availability"]["cases"]
                    if r["suite"] == scene["suite"] and r["task"] == scene["task"])
    if (identity["case"] != scene or identity["manifest_hash"] != manifest["manifest_hash"] or
        identity["init_vector_sha256"] != expected["init_sha256"][str(scene["init"])] or
        identity["bddl_sha256"] != expected["bddl_sha256"] or identity["prompt"] != expected["prompt"] or
        identity["env_seed"] != 7 or identity["settling"] != SETTLING_ACTIONS or
        identity["policy_action_limit"] != MAX_POLICY_ACTIONS or identity["control_hz"] != 20.):
        raise ValueError("Official state, BDDL, prompt, seed, settling or action limit changed")
    return physical_assets(identity)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", required=True); parser.add_argument("--condition", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--stage", default=None, choices=("discovery", "confirm"))
    parser.add_argument("--shard-index", type=int, default=0); parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args(); output = Path(args.output).resolve()
    manifest = read_manifest(output/"manifest.json", verify_sources=True)
    validate_output(output, manifest)
    cfg = planned_config(manifest, args.condition)
    scenes = cases(manifest, args.stage, args.shard_index, args.shards)
    from .rollout import episode
    from preparations.idea1_v3_1.rollout import Environments
    from preparations.idea1_v2_1.runtime_client import input_digest, rpc
    class SeededEnvironments(Environments):
        def acquire(self, suite, task):
            random.seed(7); np.random.seed(7)
            return super().acquire(suite, task)
    pool = SeededEnvironments(reuse=False)
    def gated_rpc(port, request):
        response = rpc(port, request)
        if request.get("op") == "health":
            p = response["protocol"]
            if (p["manifest_hash"] != manifest["manifest_hash"] or p["condition_id"] != cfg["id"] or
                p.get("config") != cfg or not p["real_input_parity_passed"] or
                p.get("qk_reference_passed") is not True or p.get("policy_action_limit") != MAX_POLICY_ACTIONS):
                raise ValueError("Incorrect or ungated within-suite service")
        return response
    try:
        for scene in scenes:
            directory = episode_dir(output, cfg["id"], scene)
            with lock(str(directory)+".lock"):
                if completed(output, cfg["id"], scene, manifest["manifest_hash"]):
                    continue
                recovered = recover_case(output, cfg["id"], scene, manifest, finalize, already_locked=True)
                if recovered is not None:
                    print(json.dumps(recovered), flush=True); continue
                def gate(identity, request):
                    assets = verify_physical_identity(manifest, scene, identity)
                    initial_gate(output, manifest["manifest_hash"], cfg["id"], scene,
                        input_digest(request["obs"], request["instruction"]), request["seed"], assets=assets)
                result = run_with_retries(directory, manifest["technical_max_attempts"],
                    lambda: episode(directory, args.port, scene, cfg, pool, manifest["manifest_hash"],
                                    max_policy_actions=manifest["science"]["policy_action_limit"],
                                    settling_actions=manifest["science"]["settling_actions"],
                                    rpc_call=gated_rpc, initial_input_gate=gate),
                    lambda: finalize(output, cfg["id"], scene, manifest), pool.close)
                print(json.dumps(result), flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
