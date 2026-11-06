#!/usr/bin/env python3
"""Fail-closed CPU audit for the clean Experiment 7 D3/D10 controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
ACTION_HORIZON = 50
ACTION_DIM = 32
FULL_LAYERS = list(range(1, 19))
NOISE_RULE = "SeedSequence(global_noise_seed, episode_idx, replan_idx)"
SCOPE = {
    "unit": "task",
    "task_count": 100,
    "rollouts_per_task": 1,
    "init_source": "one frozen HDF5 demonstration state per task",
    "environment_seed": 7,
    "interpretation": (
        "Paired causal audit on 100 selected tasks, one frozen HDF5 state and one seeded rollout per task; "
        "not a comprehensive or official multi-initial-state LIBERO benchmark."
    ),
}


class AuditError(ValueError):
    """Raised when an audit invariant is violated."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read valid JSON from {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AuditError(f"cannot read {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AuditError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise AuditError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _resolved_inside(root: Path, relative: Any, *, label: str) -> Path:
    candidate = Path(str(relative))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise AuditError(f"{label}: artifact path must be relative and contained in run directory: {candidate}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError(f"{label}: artifact escapes run directory: {candidate}") from exc
    if not resolved.is_file():
        raise AuditError(f"{label}: missing artifact: {resolved}")
    return resolved


def _require_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) != expected:
        raise AuditError(f"{label}: expected {expected}, got {value!r}")


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise AuditError(f"{label}: expected {expected!r}, got {value!r}")


def _finite_vector(value: Any, length: int | None, label: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AuditError(f"{label}: not a numeric vector") from exc
    if array.ndim != 1 or (length is not None and array.shape != (length,)) or array.size == 0:
        expected = "a non-empty vector" if length is None else f"shape ({length},)"
        raise AuditError(f"{label}: expected {expected}, got {array.shape}")
    if not np.isfinite(array).all():
        raise AuditError(f"{label}: contains non-finite values")
    return array


def expected_noise_sha256(global_seed: int, episode_idx: int, infer_id: int) -> str:
    """Reproduce the historical Experiment 5 float32 noise identity exactly."""
    generator = np.random.default_rng(np.random.SeedSequence([global_seed, episode_idx, infer_id]))
    noise = generator.standard_normal((ACTION_HORIZON, ACTION_DIM)).astype(np.float32)
    return hashlib.sha256(noise.tobytes(order="C")).hexdigest()


def _load_testset(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = _read_json(path)
    samples = payload.get("samples", payload.get("tasks")) if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples:
        raise AuditError(f"{path}: expected a non-empty samples array")
    normalized = []
    for index, raw in enumerate(samples):
        if not isinstance(raw, dict):
            raise AuditError(f"{path}: sample {index} is not an object")
        sample = dict(raw)
        sample["sample_index"] = index
        for key in ("sample_id", "suite_name", "task_id", "init_state_path", "init_state_sha256"):
            if key not in sample:
                raise AuditError(f"{path}: sample {index} lacks {key}")
        state_path = Path(str(sample["init_state_path"]))
        if not state_path.is_absolute():
            state_path = path.parent / state_path
        state_path = state_path.resolve()
        if not state_path.is_file():
            raise AuditError(f"{path}: sample {index} init state is missing: {state_path}")
        expected_hash = str(sample["init_state_sha256"]).lower()
        file_hash = _sha256(state_path)
        try:
            state = np.load(state_path, allow_pickle=False)
            array_hash = hashlib.sha256(np.ascontiguousarray(state).tobytes()).hexdigest()
            finite = bool(np.isfinite(np.asarray(state)).all())
        except Exception as exc:
            raise AuditError(f"{state_path}: cannot load init state: {exc}") from exc
        if not finite:
            raise AuditError(f"{state_path}: init state contains non-finite values")
        if expected_hash not in {file_hash, array_hash}:
            raise AuditError(
                f"{path}: sample {index} init_state_sha256 does not match file or array bytes"
            )
        sample["resolved_init_state_path"] = str(state_path)
        normalized.append(sample)
    return normalized, _sha256(path)


def _assert_identity(actual: dict[str, Any], sample: dict[str, Any], label: str) -> None:
    expected = {
        "sample_index": int(sample["sample_index"]),
        "sample_id": str(sample["sample_id"]),
        "suite_name": str(sample["suite_name"]),
        "task_id": int(sample["task_id"]),
    }
    for key, value in expected.items():
        actual_key = "task_suite_name" if key == "suite_name" and "suite_name" not in actual else key
        _require_equal(actual.get(actual_key), value, f"{label}.{actual_key}")


def _audit_episode(
    *,
    run_dir: Path,
    episode: dict[str, Any],
    sample: dict[str, Any],
    denoise_steps: int,
    global_seed: int,
) -> dict[str, Any]:
    label = f"sample[{sample['sample_index']}]"
    _assert_identity(episode, sample, label)
    _require_int(episode.get("episode_idx"), int(sample["sample_index"]), f"{label}.episode_idx")
    _require_int(episode.get("noise_episode_idx"), int(sample["sample_index"]), f"{label}.noise_episode_idx")
    _require_equal(
        str(episode.get("init_state_sha256", "")).lower(),
        str(sample["init_state_sha256"]).lower(),
        f"{label}.init_state_sha256",
    )
    if episode.get("failure_reason") is not None:
        raise AuditError(f"{label}: rollout has failure_reason={episode['failure_reason']!r}")

    infer_path = _resolved_inside(run_dir, episode.get("infer_path"), label=f"{label}.infer_path")
    trace_path = _resolved_inside(run_dir, episode.get("trace_path"), label=f"{label}.trace_path")
    inferences = _read_jsonl(infer_path)
    traces = _read_jsonl(trace_path)
    if not inferences or not traces:
        raise AuditError(f"{label}: infer and trace files must both be non-empty")
    _require_int(episode.get("infer_calls"), len(inferences), f"{label}.infer_calls")
    _require_int(episode.get("infer_record_count"), len(inferences), f"{label}.infer_record_count")
    _require_int(episode.get("trace_record_count"), len(traces), f"{label}.trace_record_count")
    _require_int(episode.get("executed_action_count"), len(traces), f"{label}.executed_action_count")
    _require_int(episode.get("trajectory_length"), len(traces), f"{label}.trajectory_length")

    inference_by_id: dict[int, dict[str, Any]] = {}
    noise_hashes: dict[int, str] = {}
    for expected_infer_id, infer in enumerate(inferences):
        infer_label = f"{label}.infer[{expected_infer_id}]"
        _assert_identity(infer, sample, infer_label)
        _require_int(infer.get("episode_idx"), int(sample["sample_index"]), f"{infer_label}.episode_idx")
        _require_int(infer.get("infer_id"), expected_infer_id, f"{infer_label}.infer_id")
        _require_equal(infer.get("route_type"), "full", f"{infer_label}.route_type")
        metadata = infer.get("server_experiment")
        if not isinstance(metadata, dict):
            raise AuditError(f"{infer_label}: missing server_experiment metadata")
        _require_equal(metadata.get("vlm_layer_pattern"), "vlm_full18", f"{infer_label}.vlm_layer_pattern")
        _require_equal(metadata.get("action_layer_pattern"), "full18", f"{infer_label}.action_layer_pattern")
        _require_equal(metadata.get("vlm_layer_indices_1based"), FULL_LAYERS, f"{infer_label}.vlm layers")
        _require_equal(metadata.get("action_layer_indices_1based"), FULL_LAYERS, f"{infer_label}.action layers")
        _require_int(metadata.get("denoise_steps"), denoise_steps, f"{infer_label}.denoise_steps")
        _require_int(metadata.get("ablation_noise_seed"), global_seed, f"{infer_label}.global_noise_seed")
        _require_equal(metadata.get("noise_seed_mode"), "episode_infer", f"{infer_label}.noise_seed_mode")
        _require_equal(metadata.get("noise_seed_rule"), NOISE_RULE, f"{infer_label}.noise_seed_rule")
        _require_int(metadata.get("trace_episode_idx"), int(sample["sample_index"]), f"{infer_label}.trace_episode_idx")
        _require_int(metadata.get("trace_infer_id"), expected_infer_id, f"{infer_label}.trace_infer_id")
        expected_hash = expected_noise_sha256(global_seed, int(sample["sample_index"]), expected_infer_id)
        _require_equal(metadata.get("noise_sha256"), expected_hash, f"{infer_label}.noise_sha256")
        actions = np.asarray(infer.get("chunk_actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise AuditError(f"{infer_label}.chunk_actions: expected finite Nx7 actions, got {actions.shape}")
        chunk_exec_len = infer.get("chunk_exec_len")
        if isinstance(chunk_exec_len, bool) or not isinstance(chunk_exec_len, int):
            raise AuditError(f"{infer_label}.chunk_exec_len is not an integer")
        if chunk_exec_len <= 0 or chunk_exec_len > len(actions):
            raise AuditError(f"{infer_label}.chunk_exec_len is outside available chunk actions")
        inference_by_id[expected_infer_id] = infer
        noise_hashes[expected_infer_id] = expected_hash

    offsets: dict[int, list[int]] = {infer_id: [] for infer_id in inference_by_id}
    for trace_index, trace in enumerate(traces):
        trace_label = f"{label}.trace[{trace_index}]"
        _assert_identity(trace, sample, trace_label)
        _require_int(trace.get("episode_idx"), int(sample["sample_index"]), f"{trace_label}.episode_idx")
        infer_id = trace.get("infer_id")
        if isinstance(infer_id, bool) or not isinstance(infer_id, int) or infer_id not in inference_by_id:
            raise AuditError(f"{trace_label}.infer_id does not identify an inference")
        offset = trace.get("action_offset_in_chunk")
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise AuditError(f"{trace_label}.action_offset_in_chunk is not an integer")
        infer = inference_by_id[infer_id]
        if offset < 0 or offset >= int(infer["chunk_exec_len"]):
            raise AuditError(f"{trace_label}.action_offset_in_chunk is out of executed chunk range")
        actual_action = _finite_vector(trace.get("executed_action"), 7, f"{trace_label}.executed_action").astype(np.float32)
        expected_action = np.asarray(infer["chunk_actions"][offset], dtype=np.float32)
        if not np.array_equal(actual_action, expected_action):
            raise AuditError(f"{trace_label}.executed_action differs from infer chunk action {infer_id}:{offset}")
        _finite_vector(trace.get("measured_eef_pos"), 3, f"{trace_label}.measured_eef_pos")
        _finite_vector(trace.get("measured_eef_quat"), 4, f"{trace_label}.measured_eef_quat")
        _finite_vector(trace.get("measured_gripper_qpos"), None, f"{trace_label}.measured_gripper_qpos")
        _require_equal(trace.get("measured_state_when"), "post_action", f"{trace_label}.measured_state_when")
        offsets[infer_id].append(offset)

    for infer_id, used_offsets in offsets.items():
        if not used_offsets:
            raise AuditError(f"{label}.infer[{infer_id}] has no executed trace action")
        expected_offsets = list(range(len(used_offsets)))
        if used_offsets != expected_offsets:
            raise AuditError(
                f"{label}.infer[{infer_id}] trace offsets are not consecutive from zero: {used_offsets}"
            )
        if infer_id != max(inference_by_id) and len(used_offsets) != int(inference_by_id[infer_id]["chunk_exec_len"]):
            raise AuditError(f"{label}.infer[{infer_id}] is not final but has an incomplete executed chunk")

    return {
        "sample_index": int(sample["sample_index"]),
        "sample_id": str(sample["sample_id"]),
        "success": bool(episode.get("success")),
        "infer_count": len(inferences),
        "action_count": len(traces),
        "noise_sha256_by_infer": {str(key): value for key, value in noise_hashes.items()},
    }


def audit_run(
    run_dir: Path,
    testset_manifest: Path,
    *,
    expected_count: int,
    expected_denoise_steps: int | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    testset_manifest = testset_manifest.resolve()
    if expected_count <= 0:
        raise AuditError("expected_count must be positive")
    manifest = _read_json(run_dir / "manifest.json")
    episodes = _read_json(run_dir / "episode_log.json")
    if not isinstance(manifest, dict) or not isinstance(episodes, list):
        raise AuditError("run manifest must be an object and episode_log.json must be an array")
    samples, testset_hash = _load_testset(testset_manifest)
    if expected_count > len(samples):
        raise AuditError(f"expected_count={expected_count} exceeds testset size {len(samples)}")
    if len(episodes) != expected_count:
        raise AuditError(f"expected exactly {expected_count} completed episode records, found {len(episodes)}")
    indices = [row.get("sample_index") for row in episodes if isinstance(row, dict)]
    if indices != list(range(expected_count)):
        raise AuditError(f"episode coverage must be ordered unique indices 0..{expected_count - 1}, got {indices}")

    _require_equal(manifest.get("testset_sha256"), testset_hash, "manifest.testset_sha256")
    _require_int(manifest.get("sample_count"), len(samples), "manifest.sample_count")
    declared = manifest.get("samples")
    if not isinstance(declared, list) or len(declared) != len(samples):
        raise AuditError("manifest.samples must declare every testset sample exactly once")
    for index, (actual, sample) in enumerate(zip(declared, samples)):
        if not isinstance(actual, dict):
            raise AuditError(f"manifest.samples[{index}] is not an object")
        _assert_identity(actual, sample, f"manifest.samples[{index}]")

    resume_markers = ("resume", "resumed", "resumed_from", "resume_count")
    present_resume_markers = {key: manifest[key] for key in resume_markers if key in manifest}
    if any(value not in (False, None, 0, "") for value in present_resume_markers.values()):
        raise AuditError(f"run manifest contains an active resume marker: {present_resume_markers}")

    denoise_steps = manifest.get("denoise_steps") if expected_denoise_steps is None else expected_denoise_steps
    if denoise_steps not in (3, 10):
        raise AuditError(f"clean control denoise_steps must be D3 or D10, got {denoise_steps!r}")
    _require_int(manifest.get("denoise_steps"), int(denoise_steps), "manifest.denoise_steps")
    _require_equal(manifest.get("vlm_layer_pattern"), "vlm_full18", "manifest.vlm_layer_pattern")
    _require_equal(manifest.get("action_layer_pattern"), "full18", "manifest.action_layer_pattern")
    global_seed = manifest.get("global_noise_seed")
    if isinstance(global_seed, bool) or not isinstance(global_seed, int):
        raise AuditError("manifest.global_noise_seed must be an integer")
    _require_int(manifest.get("seed"), 7, "manifest.seed")

    strict_final = expected_count == 100
    if strict_final:
        if len(samples) != 100:
            raise AuditError(f"strict final audit requires a 100-sample testset, found {len(samples)}")
        _require_equal(manifest.get("status"), "completed", "manifest.status")
        _require_int(manifest.get("completed_sample_count"), 100, "manifest.completed_sample_count")
    elif manifest.get("status") not in {"running", "completed"}:
        raise AuditError(f"incremental audit requires running/completed status, got {manifest.get('status')!r}")

    episode_reports = [
        _audit_episode(
            run_dir=run_dir,
            episode=episode,
            sample=samples[index],
            denoise_steps=int(denoise_steps),
            global_seed=int(global_seed),
        )
        for index, episode in enumerate(episodes)
    ]
    successes = sum(row["success"] for row in episode_reports)
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_type": "clean_control_single_run",
        "status": "passed",
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id"),
        "denoise_steps": int(denoise_steps),
        "expected_count": expected_count,
        "strict_final_100": strict_final,
        "episode_count": len(episode_reports),
        "successes": successes,
        "success_rate": successes / len(episode_reports),
        "inference_count": sum(row["infer_count"] for row in episode_reports),
        "executed_action_count": sum(row["action_count"] for row in episode_reports),
        "init_state_hashes_verified": len(episode_reports),
        "noise_rule": NOISE_RULE,
        "resume_audit": {
            "active_resume_marker": False,
            "manifest_markers": present_resume_markers,
            "evidence_limit": "Artifact audit cannot reconstruct an omitted historical command line.",
        },
        "scope": {**SCOPE, "task_count": expected_count},
        "episodes": episode_reports,
    }


def mcnemar_exact_two_sided(d3_only_success: int, d10_only_success: int) -> float:
    discordant = d3_only_success + d10_only_success
    if discordant == 0:
        return 1.0
    tail = min(d3_only_success, d10_only_success)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1)) / (2**discordant)
    return min(1.0, 2.0 * probability)


def audit_pair(
    d3_run_dir: Path,
    d10_run_dir: Path,
    testset_manifest: Path,
    *,
    expected_count: int,
) -> dict[str, Any]:
    d3 = audit_run(
        d3_run_dir, testset_manifest, expected_count=expected_count, expected_denoise_steps=3
    )
    d10 = audit_run(
        d10_run_dir, testset_manifest, expected_count=expected_count, expected_denoise_steps=10
    )
    d3_by_id = {row["sample_id"]: row for row in d3["episodes"]}
    d10_by_id = {row["sample_id"]: row for row in d10["episodes"]}
    if set(d3_by_id) != set(d10_by_id):
        raise AuditError("D3 and D10 do not cover the same sample IDs")

    common_inferences = d3_only_inferences = d10_only_inferences = 0
    pairs = []
    d3_only_success = d10_only_success = both_success = both_failure = 0
    for sample_id in (row["sample_id"] for row in d3["episodes"]):
        left, right = d3_by_id[sample_id], d10_by_id[sample_id]
        left_noise = left["noise_sha256_by_infer"]
        right_noise = right["noise_sha256_by_infer"]
        common = sorted(set(left_noise) & set(right_noise), key=int)
        for infer_id in common:
            if left_noise[infer_id] != right_noise[infer_id]:
                raise AuditError(f"{sample_id} infer {infer_id}: paired D3/D10 noise hashes differ")
        common_inferences += len(common)
        d3_only_inferences += len(set(left_noise) - set(right_noise))
        d10_only_inferences += len(set(right_noise) - set(left_noise))
        left_success, right_success = bool(left["success"]), bool(right["success"])
        if left_success and right_success:
            both_success += 1
        elif left_success:
            d3_only_success += 1
        elif right_success:
            d10_only_success += 1
        else:
            both_failure += 1
        pairs.append(
            {
                "sample_id": sample_id,
                "d3_success": left_success,
                "d10_success": right_success,
                "common_noise_paired_inferences": len(common),
                "d3_infer_count": left["infer_count"],
                "d10_infer_count": right["infer_count"],
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "audit_type": "clean_control_d3_d10_pair",
        "status": "passed",
        "expected_count": expected_count,
        "strict_final_100": expected_count == 100,
        "scope": {**SCOPE, "task_count": expected_count},
        "d3": {key: d3[key] for key in ("run_dir", "run_id", "successes", "success_rate", "inference_count")},
        "d10": {key: d10[key] for key in ("run_dir", "run_id", "successes", "success_rate", "inference_count")},
        "paired_noise": {
            "common_inferences": common_inferences,
            "d3_only_after_termination": d3_only_inferences,
            "d10_only_after_termination": d10_only_inferences,
            "all_common_hashes_equal": True,
        },
        "mcnemar_exact": {
            "both_success": both_success,
            "d3_only_success": d3_only_success,
            "d10_only_success": d10_only_success,
            "both_failure": both_failure,
            "discordant_pairs": d3_only_success + d10_only_success,
            "two_sided_p_value": mcnemar_exact_two_sided(d3_only_success, d10_only_success),
            "method": "two-sided exact binomial McNemar test, p=2*lower tail, capped at 1",
        },
        "pairs": pairs,
    }


def _write_success_report(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except FileExistsError as exc:
        raise AuditError(f"refusing to overwrite existing report: {path}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset-manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output", type=Path, help="Write only after every audit invariant passes; must not exist.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-dir", type=Path, help="Audit one D3 or D10 run.")
    mode.add_argument("--d3-run-dir", type=Path, help="Audit a paired D3/D10 comparison.")
    parser.add_argument("--d10-run-dir", type=Path)
    parser.add_argument("--expected-denoise-steps", type=int, choices=(3, 10))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.run_dir is not None:
            if args.d10_run_dir is not None:
                raise AuditError("--d10-run-dir is valid only with --d3-run-dir")
            report = audit_run(
                args.run_dir,
                args.testset_manifest,
                expected_count=args.expected_count,
                expected_denoise_steps=args.expected_denoise_steps,
            )
        else:
            if args.d10_run_dir is None:
                raise AuditError("paired mode requires --d10-run-dir")
            if args.expected_denoise_steps is not None:
                raise AuditError("paired mode fixes D3/D10; omit --expected-denoise-steps")
            report = audit_pair(
                args.d3_run_dir,
                args.d10_run_dir,
                args.testset_manifest,
                expected_count=args.expected_count,
            )
        if args.output is not None:
            _write_success_report(args.output, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except AuditError as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
