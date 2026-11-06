"""Plan-first, eight-case fresh-KV pi0.5 quality baseline on separately assigned GPU0."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

WRAPPER = Path(__file__).resolve()
ROOT = WRAPPER.parent.parents[2]
sys.path.insert(0, str(ROOT))
base = importlib.import_module("results.idea2.d10_dynamic_sm_20261024.closedloop_smoke")
BASE_PATH = Path(base.__file__).resolve()
BASE_PARSER, BASE_PLAN, BASE_SOURCES, BASE_WORKER = base.parser, base.make_plan, base.source_hashes, base.run_worker
GPU_UUID = "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1"
COMPARISON = "Compare success/steps on the same eight cases against old-KV runs; no cross-GPU absolute latency ranking."


def episode_cases(reverse=False):
    rows = [dict(case=index, arm=0, label="F0", mode="fresh_serial", suite=suite, task=task, init=0,
                 relative_directory=f"episodes/F0/{suite}/task{task:02d}/init00")
            for index, (suite, task) in enumerate((suite, task) for suite in base.SUITES for task in (0, 5))]
    return rows[::-1] if reverse else rows


def source_hashes():
    values = BASE_SOURCES()
    for path in (BASE_PATH, WRAPPER):
        values[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def parser():
    result = BASE_PARSER()
    result.description = __doc__
    result.set_defaults(output=str(WRAPPER.parent / "closedloop_d10_fresh_gpu0"), gpu_uuid=GPU_UUID,
                        egl_device=3, port=19083)
    return result


def make_plan(args):
    plan = BASE_PLAN(args)
    plan.update(protocol=base.expected_protocol("fresh_serial"), conditioning="current_frame_complete_KV",
        expected_observation_age=0, total_episodes=8, comparison_scope=COMPARISON,
        schedule="One fresh_serial mode, four suites x tasks0/5 x init0 = eight real episodes; one persistent service.",
        semantics="Every request completes its own current-observation Prefix before AE. No old-KV reuse or future observation input.")
    plan["limitations"] = ["Eight-case exploratory quality smoke, not official full LIBERO evaluation.", COMPARISON,
        "Default EGL index3 maps to assigned GPU0 in this host's current mapping; check at execution, not a portable index rule.",
        "Client CUDA is disabled; rendering still uses the assigned GPU. Host/container PIDs are recorded separately.",
        "First request can include Graph capture. LIBERO pauses during inference; no robot SLO claim."]
    snapshots = Path(args.output).resolve() / "source"
    snapshots.mkdir(parents=True, exist_ok=True)
    for path in (BASE_PATH, WRAPPER):
        snapshot = snapshots / path.name
        data = path.read_bytes()
        if snapshot.exists() and snapshot.read_bytes() != data:
            raise RuntimeError("Refusing to overwrite an existing source snapshot")
        if not snapshot.exists():
            snapshot.write_bytes(data)
    plan["wrapper"] = dict(path=str(WRAPPER), sha256=source_hashes()[str(WRAPPER.relative_to(ROOT))],
                           base_path=str(BASE_PATH), base_sha256=source_hashes()[str(BASE_PATH.relative_to(ROOT))],
                           source_snapshot=str(snapshots / WRAPPER.name), worker_entrypoint=str(WRAPPER))
    return plan


def validate_artifacts(directory, client, case):
    import numpy as np
    result = json.loads((directory / "result.json").read_text())
    base.check_protocol(json.loads((directory / "native_protocol.json").read_text()), "fresh_serial")
    infers = [json.loads(line) for line in (directory / "infer.jsonl").read_text().splitlines()]
    telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    if len(infers) != result["predictions"] or len(telemetry) != result["actions"] + 1:
        raise RuntimeError("Incomplete action/inference telemetry")
    for index, row in enumerate(infers):
        if row["seed"] != client.model_seed(case["suite"], case["task"], case["init"], index):
            raise RuntimeError("Original model noise seed changed")
        if (row["mode"] != "fresh_serial" or row["observation_request_age"] != 0
                or row["prefix_observation_id"] != row["current_observation_id"]):
            raise RuntimeError("Fresh current-frame KV contract failed")
        if row["graph_kv_buffers"] != 2 or row["pool"]["live_snapshots"] > 2:
            raise RuntimeError("Two-slot KV bound failed")
        actions = np.asarray(row["chunk_actions"])
        if actions.shape != (10, 7) or not np.isfinite(actions).all() or (index and row["graph_setup_ms"] != 0):
            raise RuntimeError("Invalid H10 actions or within-episode recapture")
    if any("eef_pos" not in row or "eef_quat" not in row for row in telemetry):
        raise RuntimeError("Missing measured EEF state")
    def stats(values):
        return (dict(n=len(values), mean=float(np.mean(values)), p50=float(np.percentile(values, 50)),
                     p95=float(np.percentile(values, 95))) if values else None)
    steady = [row for index, row in enumerate(infers) if index and row["graph_setup_ms"] == 0]
    return dict(valid=True, success=result["success"], steps=result["actions"], predictions=len(infers),
        conditioning="current_frame_complete_KV", expected_observation_age=0, comparison_scope=COMPARISON,
        cross_gpu_latency_comparable=False, cold_capture_requests=sum(row["graph_setup_ms"] > 0 for row in infers),
        first_input_hash=infers[0]["input_hash"] if infers else None,
        all_request_roundtrip_ms=stats([row["client_roundtrip_ms"] for row in infers]),
        steady_fresh_roundtrip_ms=stats([row["client_roundtrip_ms"] for row in steady]),
        steady_fresh_action_ready_ms=stats([row["timing"]["action_ready_host_ms"] for row in steady]),
        runtime_result=result, noise_identity="Original seeds/construction identity, not noise tensor byte hashes")


def run_worker(case_path):
    from preparations.idea1_v2_1 import runtime_client as client
    from unittest.mock import patch
    original = client.run_episode
    def fresh_episode(*args, **kwargs):
        kwargs["kind"] = "idea2_pi05_d10_fresh_smoke"  # Metadata only; original env/sampling/actions unchanged.
        return original(*args, **kwargs)
    with patch.object(client, "run_episode", fresh_episode):
        return BASE_WORKER(case_path)


def install():
    base.MODES = {"F0": "fresh_serial"}
    base.episode_cases, base.source_hashes, base.parser = episode_cases, source_hashes, parser
    base.make_plan, base.validate_artifacts, base.run_worker = make_plan, validate_artifacts, run_worker
    base.__file__ = str(WRAPPER)  # Dispatch each child through these same hooks; base file is not edited.


if __name__ == "__main__":
    install()
    base.main()
