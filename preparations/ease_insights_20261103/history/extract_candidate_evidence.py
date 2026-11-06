#!/usr/bin/env python3
"""CPU-only, read-only extraction of the frozen vision-mask discovery evidence.

Run from any directory with Python 3. All outputs are written beside this script
unless --output is supplied. Historical data and implementation are never edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
CANDIDATES = {
    "A": {"condition": "pi0_keep24_2_front_middle", "skip": [8, 9, 10]},
    "B": {"condition": "pi0_keep24_1_uniform", "skip": [5, 14, 23]},
}
PROTOCOL_DIFFERENCE_ALLOWLIST = {
    "id", "condition_id", "layout", "skip", "keep", "server_pid"
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_records = {}

    def source(relative_or_absolute):
        path = Path(relative_or_absolute)
        if not path.is_absolute():
            path = root / path
        key = str(path)
        if key not in source_records:
            source_records[key] = {"path": key, "sha256": sha256(path), "bytes": path.stat().st_size}
        return path

    manifest_path = source("results/idea1_v3_1/manifest_v2.json")
    manifest = read_json(manifest_path)
    payload = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    assert digest(payload) == manifest["manifest_hash"], "Historical manifest identity mismatch"
    source("results/idea1_v3_1/final_artifacts_20261021_0524/layout_masks.csv")
    summary_path = source("results/idea1_v3_1/final_artifacts_20261021_0524/quality_screen_init0.csv")
    historical_summary = list(csv.DictReader(summary_path.open()))
    frozen_source_checks = []
    for relative, expected in manifest["source_files"].items():
        path = source(relative)
        frozen_source_checks.append({"path": str(path), "expected_sha256": expected,
                                     "actual_sha256": source_records[str(path)]["sha256"],
                                     "matches_frozen_manifest": source_records[str(path)]["sha256"] == expected})
    hash_helper = source("preparations/idea1_v2_1/runtime_client.py")
    source("preparations/idea1_v2_1/runtime_models.py")
    source(Path(__file__).resolve())

    protocols = {}
    cases = {}
    actual_request_counts = {}
    request_audit_failures = []
    episode_protocol_failures = []
    telemetry_sim_times = set()
    observed_environment_reuse = set()
    for label, candidate in CANDIDATES.items():
        cid = candidate["condition"]
        directory = root / "results/idea1_v3_1/run" / cid
        protocols[label] = read_json(source(directory / "loaded_protocol.json"))
        actual_request_counts[label] = 0
        cases[label] = {}
        expected_condition = next(c for c in manifest["conditions"] if c["id"] == cid)
        assert expected_condition["skip"] == candidate["skip"]
        for suite in SUITES:
            for task in range(10):
                episode_dir = directory / "episodes" / suite / f"task{task:02d}" / "init00"
                paths = {name: source(episode_dir / name) for name in
                         ("identity.json", "result.json", "complete.json", "infer.jsonl")}
                identity = read_json(paths["identity.json"])
                result = read_json(paths["result.json"])
                complete = read_json(paths["complete.json"])
                assert result["valid"] and complete["valid"]
                assert (result["suite"], result["task"], result["init"]) == (suite, task, 0)
                assert result["condition"] == cid
                assert result["manifest_hash"] == identity["manifest_hash"] == manifest["manifest_hash"]
                if identity["condition"] != expected_condition or identity["protocol"] != protocols[label]:
                    episode_protocol_failures.append({"label": label, "suite": suite, "task": task})
                observed_environment_reuse.add(identity["environment_reused"])
                with paths["infer.jsonl"].open() as stream:
                    first_line = next(stream)
                    first = json.loads(first_line)
                    requests = [first, *(json.loads(line) for line in stream if line.strip())]
                for request in requests:
                    request_index = request["request"]
                    expected_seed = int(digest(["idea1-v3.1", 1701, suite, task, 0, request_index])[:8], 16)
                    checks = {
                        "seed": request["seed"] == expected_seed,
                        "NFE": request["audit"]["NFE"] == 10,
                        "skip": request["audit"]["vision_skip"] == candidate["skip"],
                        "original_scanned_ae": request["audit"]["original_scanned_ae"] is True,
                        "execute_length": request["execute_length"] == 5,
                        "executed_length": 1 <= request["executed_length"] <= 5,
                        "horizon": len(request["actions"]) == 50,
                    }
                    if not all(checks.values()):
                        request_audit_failures.append({"label": label, "suite": suite, "task": task,
                                                       "request": request_index, "checks": checks})
                actual_request_counts[label] += len(requests)
                telemetry_path = episode_dir / "telemetry.jsonl"
                with telemetry_path.open() as stream:
                    telemetry_line = next(stream)
                    telemetry = json.loads(telemetry_line)
                telemetry_sim_times.add(telemetry["sim_time_s"])
                cases[label][(suite, task)] = {
                    "success": result["success"], "episode_directory": str(episode_dir),
                    "init_vector_sha256": identity["init_vector_sha256"],
                    "bddl_sha256": identity["bddl_sha256"], "env_seed": identity["env_seed"],
                    "control_hz": identity["control_hz"], "controller": identity["controller"],
                    "environment_reused": identity["environment_reused"],
                    "first_input_hash": first.get("input_hash"), "first_noise_hash": first.get("noise_hash"),
                    "first_noise_seed": first.get("seed"), "first_request_step": first["step"],
                    "infer_first_line_sha256": hashlib.sha256(first_line.encode()).hexdigest(),
                    "first_telemetry_sim_time_s": telemetry["sim_time_s"],
                    "first_telemetry_source": str(telemetry_path),
                    "first_telemetry_line_sha256": hashlib.sha256(telemetry_line.encode()).hexdigest(),
                    "initial_raw_observation_npz_present": any(episode_dir.glob("observation_0000.npz")),
                    "requests": len(requests), "actions": result["actions"],
                }

    protocol_differences = {key: {"A": protocols["A"].get(key), "B": protocols["B"].get(key)}
                            for key in sorted(set(protocols["A"]) | set(protocols["B"]))
                            if protocols["A"].get(key) != protocols["B"].get(key)}
    unexpected_protocol_differences = {k: v for k, v in protocol_differences.items()
                                       if k not in PROTOCOL_DIFFERENCE_ALLOWLIST}
    paired_rows = []
    discordant = []
    paired_fields = ("init_vector_sha256", "bddl_sha256", "env_seed", "control_hz", "controller",
                     "first_input_hash", "first_noise_hash", "first_noise_seed", "first_telemetry_sim_time_s")
    matching_counts = {field: 0 for field in paired_fields}
    for suite in SUITES:
        for task in range(10):
            a, b = cases["A"][(suite, task)], cases["B"][(suite, task)]
            row = {"suite": suite, "task": task, "init": 0}
            for label, values in (("A", a), ("B", b)):
                for key, value in values.items():
                    row[f"{label}_{key}"] = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for field in paired_fields:
                same = a[field] == b[field] and a[field] is not None
                row[field + "_equal"] = same
                matching_counts[field] += int(same)
            paired_rows.append(row)
            if a["success"] != b["success"]:
                discordant.append({"suite": suite, "task": task, "init": 0,
                                   "A_success": a["success"], "B_success": b["success"]})
    suite_successes = {}
    for label in CANDIDATES:
        suite_successes[label] = {suite: {"success": sum(cases[label][(suite, task)]["success"] for task in range(10)), "n": 10}
                                  for suite in SUITES}
        suite_successes[label]["all"] = {"success": sum(r["success"] for r in cases[label].values()), "n": 40}
        for suite, counts in suite_successes[label].items():
            rows = [r for r in historical_summary if r["condition"] == CANDIDATES[label]["condition"]
                    and r["suite"] == ("ALL" if suite == "all" else suite)]
            assert len(rows) == 1 and int(rows[0]["successes"]) == counts["success"]
            assert int(rows[0]["denominator"]) == counts["n"]

    csv_path = output / "paired_discovery_cases.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)
    sources_path = output / "source_manifest.json"
    sources_path.write_text(json.dumps({"sources": list(source_records.values()),
                                      "frozen_source_checks": frozen_source_checks}, indent=2) + "\n")
    evidence = {
        "schema": "ease-insights-history-v1", "status": "discovery_only_not_independent_confirmation",
        "source_manifest_hash": manifest["manifest_hash"],
        "candidates": {label: {**candidate, "keep": [i for i in range(1, 28) if i not in candidate["skip"]],
                                "directory": str(root / "results/idea1_v3_1/run" / candidate["condition"])}
                       for label, candidate in CANDIDATES.items()},
        "discovery_cases": {"suites": list(SUITES), "tasks": list(range(10)), "inits": [0], "paired_cases": 40},
        "successes": suite_successes, "discordant_cases": discordant,
        "frozen_confirmation_directions": {"libero_goal": "A > B", "libero_10": "B > A"},
        "numeric_protocol": {
            "model": "pi0", "backend": "native JAX OpenPI", "module": "vision",
            "vision_depth": 27, "kept_vision_blocks": 24, "D": 10, "H": 50, "R": 5,
            "vlm_depth": 18, "ae": "complete original scanned AE", "pure_fixed_submodel": True,
            "full_refresh": False, "compensation": None, "verifier": False,
            "checkpoint": protocols["A"]["checkpoint"], "env_seed": 7, "control_hz": 20,
            "settling_steps": 10, "settling_action": [0, 0, 0, 0, 0, 0, -1],
            "image_height_width": [256, 256], "limits": dict(zip(SUITES, (220, 280, 300, 520))),
            "initial_sim_times_s": sorted(telemetry_sim_times),
            "noise_seed_formula": "int(sha256(canonical_json(['idea1-v3.1',1701,suite,task,init,request]))[:8],16)",
            "environment_reused_values": sorted(observed_environment_reuse),
        },
        "protocol_comparison": {"all_differences": protocol_differences,
                                "unexpected_differences": unexpected_protocol_differences,
                                "episode_protocol_failures": episode_protocol_failures,
                                "frozen_sources_all_match": all(r["matches_frozen_manifest"] for r in frozen_source_checks)},
        "paired_audit": {"matching_counts_of_40": matching_counts, "requests_checked": actual_request_counts,
                         "request_protocol_failures": request_audit_failures,
                         "first_input_hash_present": sum(bool(r["A_first_input_hash"] and r["B_first_input_hash"]) for r in paired_rows),
                         "initial_raw_observation_npz_pairs_available": sum(r["A_initial_raw_observation_npz_present"] and r["B_initial_raw_observation_npz_present"] for r in paired_rows)},
        "hash_interpretation": {
            "input_hash": "Historical infer.jsonl digest over the transmitted ndarray observations and instruction; all 40 A/B pairs match.",
            "helper_source": str(hash_helper),
            "limitation": "No initial raw NPZ arrays are saved for these episodes; matching historical digests are verified, but this extraction cannot recompute them from raw arrays.",
            "telemetry": "Only the initial telemetry line was read; its exact line SHA256 is included per case, not a full telemetry-file integrity claim.",
        },
        "interpretation": [
            "A exceeds B by 2/10 on Goal; B exceeds A by 2/10 on Long. Both are usable on the discovery set.",
            "These paths were chosen after examining historical results; each suite has only 10 discovery cases and no independent replication yet.",
            "The pair uses the same frozen runner, checkpoint, D/H/R, environment seed, settling, controller, initial transmitted observation digest, and initial noise digest.",
            "Seed7 is set after environment construction in the historical runner. The matching first-input digests reduce this concern for the recorded A/B pairs; new replication should seed before construction and verify fresh initial inputs.",
            "No inference speedup is inferred from the historical per-request timings or from keeping 24 vision blocks.",
        ],
        "artifacts": {"paired_cases": {"path": str(csv_path), "sha256": sha256(csv_path)},
                      "sources": {"path": str(sources_path), "sha256": sha256(sources_path)}},
        "reproduce": f"python3 {Path(__file__).resolve()} --root {root} --output {output}",
    }
    path = output / "candidate_evidence.json"
    path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"evidence": str(path), "cases": 40, "successes": suite_successes,
                      "matching_counts": matching_counts, "unexpected_protocol_differences": unexpected_protocol_differences,
                      "request_failures": len(request_audit_failures), "sources": len(source_records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
