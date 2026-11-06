#!/usr/bin/env python3
"""CPU-only first-Full-request audit. Reads science outputs; writes only reports.

No project/model modules or third-party packages are imported. Completed failures
are included. The compared task identity always includes its target suite.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[3]
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
ROUTE = {"full": 1, "draft": 0, "verify": 0}


def utc(ns):
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def case_key(row):
    return row["suite"], int(row["task_id"]), int(row["init_id"])


def case_dict(key):
    return dict(target_suite=key[0], task_id=key[1], init_id=key[2])


def compare_prefix(candidate, baseline, atol, rtol):
    def valid(x):
        return (isinstance(x, list) and len(x) == 12 and
                all(isinstance(row, list) and len(row) == 7 for row in x) and
                all(isinstance(v, (float, int)) and not isinstance(v, bool) and
                    math.isfinite(v) for row in x for v in row))
    if not valid(candidate) or not valid(baseline):
        return dict(valid_12_by_7=False, exact=False, within_tolerance=False,
                    max_abs=None, mismatches=[dict(reason="invalid_shape_or_nonfinite")])
    differences = []
    mismatches = []
    for i in range(12):
        for j in range(7):
            actual, reference = candidate[i][j], baseline[i][j]
            delta = abs(actual - reference)
            tolerance = atol + rtol * abs(reference)
            differences.append(delta)
            if delta > tolerance:
                mismatches.append(dict(index=[i, j], baseline=reference,
                                       candidate=actual, abs_difference=delta,
                                       allowed_difference=tolerance))
    return dict(valid_12_by_7=True, exact=all(x == 0 for x in differences),
                within_tolerance=not mismatches, max_abs=max(differences),
                mismatches=mismatches)


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), dict(path=str(path), sha256=sha(data))


def audit(root, atol, rtol):
    cutoff = time.time_ns()
    errors, skipped, run_evidence, snapshots = [], [], [], []
    full, flash = {}, {}
    baseline_manifest = None
    # Freeze membership by episode result mtime before reading append-only requests.
    runs = []
    for branch in ("draft_smoke", "draft_discovery"):
        for run in sorted((root / branch).glob("*")):
            path = run / "manifest.json"
            if not path.exists():
                continue
            manifest, provenance = read_json(path)
            if manifest.get("mode") == "draft":
                skipped.append(dict(run=str(run), reason="pure_draft_not_comparable_to_Full"))
                continue
            if manifest.get("mode") not in ("full", "flash"):
                continue
            completed = {}
            planned = {case_key(c) for c in manifest["cases"]}
            for result_path in sorted(run.glob("episodes/*/result.json")):
                if result_path.stat().st_mtime_ns > cutoff:
                    continue
                result, evidence = read_json(result_path)
                if result.get("status") != "complete":
                    continue
                key = case_key(result)
                if (key not in planned or result.get("mode") != manifest["mode"] or
                        result.get("source_suite") != manifest["source_suite"]):
                    errors.append(dict(reason="completed_result_identity_mismatch", **evidence))
                    continue
                if key in completed:
                    errors.append(dict(reason="duplicate_completed_case_in_run", **evidence))
                    continue
                completed[key] = dict(result=result, evidence=evidence)
            runs.append((run, manifest, provenance, completed))
    for run, manifest, provenance, completed in runs:
        mode, source = manifest["mode"], manifest["source_suite"]
        if mode == "full":
            if baseline_manifest is not None:
                errors.append(dict(reason="multiple_Full_runs", run=str(run)))
            baseline_manifest = manifest
        path = run / "requests.jsonl"
        first = {}
        requests_info = dict(path=str(path), bytes_at_read_start=0, partial_tail_ignored=False)
        if path.exists():
            st = path.stat()
            requests_info.update(bytes_at_read_start=st.st_size, mtime_ns_at_read_start=st.st_mtime_ns)
            with path.open("rb") as stream:
                remaining = st.st_size
                line_number = 0
                while remaining:
                    line = stream.readline(remaining)
                    remaining -= len(line)
                    line_number += 1
                    if not line.endswith(b"\n"):
                        requests_info["partial_tail_ignored"] = True
                        break
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError) as exc:
                        errors.append(dict(reason="invalid_request_json", path=str(path),
                                           line=line_number, error=str(exc)))
                        continue
                    if row.get("replan_id") != 0:
                        continue
                    key = case_key(row)
                    if key not in completed:
                        continue
                    evidence = dict(path=str(path), line=line_number, sha256=sha(line))
                    if (row.get("mode") != mode or row.get("source_suite") != source):
                        errors.append(dict(reason="first_request_identity_mismatch", **evidence))
                        continue
                    if key in first:
                        errors.append(dict(reason="duplicate_replan_zero", **evidence))
                        continue
                    first[key] = dict(row=row, evidence=evidence)
        manifest_checks = dict(base_checkpoint=manifest.get("base_checkpoint"),
                               normalization=manifest["protocol"].get("normalization"),
                               protocol=manifest["protocol"])
        source_hashes = {}
        for name in ("common.py", "server.py"):
            matches = [(p, h) for p, h in manifest["code_sha256"].items() if Path(p).name == name]
            source_hashes[name] = [dict(path=p, recorded_sha256=h,
                                       current_sha256=sha(Path(p).read_bytes())) for p, h in matches]
        run_evidence.append(dict(run=str(run), mode=mode, source_suite=source,
                                 manifest=provenance, completed_cases=len(completed),
                                 first_requests=len(first), requests_snapshot=requests_info,
                                 target_metadata=manifest_checks, execution_sources=source_hashes))
        for key, item in completed.items():
            if key not in first:
                errors.append(dict(reason="completed_case_missing_replan_zero", run=str(run), **case_dict(key)))
                continue
            record = dict(**first[key], result=item["result"], result_evidence=item["evidence"],
                          manifest=manifest_checks, run=str(run))
            mapping, identity = (full, key) if mode == "full" else (flash, (source, *key))
            if identity in mapping:
                errors.append(dict(reason="duplicate_canonical_identity", identity=list(identity), run=str(run)))
                continue
            mapping[identity] = record
            snapshots.append(dict(mode=mode, source_suite=source, **case_dict(key),
                                  request=record["row"], request_evidence=record["evidence"],
                                  result_evidence=record["result_evidence"],
                                  success=record["result"]["success"]))
    if baseline_manifest is None:
        raise ValueError("No Full baseline manifest")
    norm_path = (Path(baseline_manifest["base_checkpoint"]).parent /
                 "assets/physical-intelligence/libero/norm_stats.json")
    norm_object, norm_evidence = read_json(norm_path)
    action_stats = norm_object["norm_stats"]["actions"]
    normalization_checks = []
    for snapshot in snapshots:
        request = snapshot["request"]
        reconstructed = [[value * (action_stats["std"][j] + 1e-6) + action_stats["mean"][j]
                          + (request["state8"][j] if j < 6 else 0)
                          for j, value in enumerate(row)] for row in request["raw_prefix"]]
        check = compare_prefix(request["decoded_prefix"], reconstructed, atol, rtol)
        snapshot["shared_normalization_reconstruction"] = check
        normalization_checks.append(dict(mode=snapshot["mode"], source_suite=snapshot["source_suite"],
                                         target_suite=snapshot["target_suite"], task_id=snapshot["task_id"],
                                         init_id=snapshot["init_id"], **check))
    expected_cases = sorted({case_key(c) for c in baseline_manifest["cases"]})
    expected = [(source, *case) for source in SUITES for case in expected_cases]
    comparisons, uncovered = [], []
    for identity in expected:
        source, *parts = identity
        case = tuple(parts)
        missing = [name for name, present in (("Full", case in full), ("FLASH", identity in flash)) if not present]
        if missing:
            uncovered.append(dict(source_suite=source, **case_dict(case), missing=missing))
            continue
        base, candidate = full[case], flash[identity]
        b, c = base["row"], candidate["row"]
        checks = {
            "Full_route_is_one_full_only": b.get("route_calls") == ROUTE,
            "FLASH_route_is_one_full_only": c.get("route_calls") == ROUTE,
            "input_sha256_equal": bool(b.get("input_sha256")) and c.get("input_sha256") == b["input_sha256"],
            "noise_sha256_equal": bool(b.get("noise_sha256")) and c.get("noise_sha256") == b["noise_sha256"],
            "noise_seed_equal": bool(b.get("noise_seed")) and c.get("noise_seed") == b["noise_seed"],
            "both_used_noise": b.get("noise_used") is True and c.get("noise_used") is True,
            "prompt_equal": bool(b.get("prompt")) and c.get("prompt") == b["prompt"],
            "state8_exact_equal": len(b.get("state8", [])) == 8 and c.get("state8") == b["state8"],
            "base_checkpoint_equal": bool(base["manifest"]["base_checkpoint"]) and candidate["manifest"]["base_checkpoint"] == base["manifest"]["base_checkpoint"],
            "normalization_metadata_equal": bool(base["manifest"]["normalization"]) and candidate["manifest"]["normalization"] == base["manifest"]["normalization"],
            "protocol_equal": candidate["manifest"]["protocol"] == base["manifest"]["protocol"],
        }
        raw = compare_prefix(c.get("raw_prefix"), b.get("raw_prefix"), atol, rtol)
        decoded = compare_prefix(c.get("decoded_prefix"), b.get("decoded_prefix"), atol, rtol)
        comparisons.append(dict(source_suite=source, **case_dict(case),
                                Full_success=base["result"]["success"], FLASH_success=candidate["result"]["success"],
                                Full_request=base["evidence"], FLASH_request=candidate["evidence"],
                                checks=checks, raw_prefix=raw, decoded_prefix=decoded,
                                passed=all(checks.values()) and raw["within_tolerance"] and decoded["within_tolerance"]))
    for run in run_evidence:
        for name, values in run["execution_sources"].items():
            if len(values) != 1 or any(v["current_sha256"] != v["recorded_sha256"] for v in values):
                errors.append(dict(reason="execution_source_hash_changed_or_missing", run=run["run"], source=name))
    per_source = {}
    for source in SUITES:
        rows = [x for x in comparisons if x["source_suite"] == source]
        per_source[source] = dict(paired=len(rows), expected=len(expected_cases),
                                  passed=sum(r["passed"] for r in rows),
                                  FLASH_successes=sum(r["FLASH_success"] for r in rows),
                                  FLASH_failures=sum(not r["FLASH_success"] for r in rows),
                                  raw_exact=sum(r["raw_prefix"]["exact"] for r in rows),
                                  decoded_exact=sum(r["decoded_prefix"]["exact"] for r in rows))
    summary = dict(expected_FLASH_pairs=len(expected), paired=len(comparisons),
                   covered_Full_cases=len(full), uncovered=len(uncovered),
                   passed_pairs=sum(r["passed"] for r in comparisons),
                   mismatch_pairs=sum(not r["passed"] for r in comparisons),
                   all_covered_pairs_pass=bool(comparisons) and all(r["passed"] for r in comparisons)
                       and all(r["within_tolerance"] for r in normalization_checks) and not errors,
                   full_matrix_covered=len(comparisons) == len(expected),
                   raw_exact_pairs=sum(r["raw_prefix"]["exact"] for r in comparisons),
                   decoded_exact_pairs=sum(r["decoded_prefix"]["exact"] for r in comparisons),
                   raw_max_abs=max((r["raw_prefix"]["max_abs"] or 0 for r in comparisons), default=None),
                   decoded_max_abs=max((r["decoded_prefix"]["max_abs"] or 0 for r in comparisons), default=None),
                   shared_normalization_records=len(normalization_checks),
                   shared_normalization_passed=sum(r["within_tolerance"] for r in normalization_checks),
                   shared_normalization_max_abs=max((r["max_abs"] or 0 for r in normalization_checks), default=None),
                   paired_outcomes=dict(Counter(f"Full={int(r['Full_success'])},FLASH={int(r['FLASH_success'])}" for r in comparisons)),
                   per_source=per_source)
    report = dict(revision="first-full-target-identity-v1", snapshot_cutoff_utc=utc(cutoff),
                  snapshot_cutoff_ns=cutoff, read_completed_utc=utc(time.time_ns()),
                  root=str(root), script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).read_bytes())),
                  protocol=dict(replan_id=0, pairing_key=["source_suite", "target_suite/task_id", "init_id"],
                                atol=atol, rtol=rtol, tolerance="abs(FLASH-Full) <= atol + rtol * abs(Full)",
                                exact="Numeric equality of JSON-decoded values; not a binary tensor bitwise assertion",
                                included="All canonical complete episode results before cutoff, successes and failures",
                                excluded="Pure draft, incomplete/newer episode results, later replans"),
                  summary=summary, errors=errors, uncovered=uncovered, comparisons=comparisons,
                  normalization_reconstruction=dict(asset=norm_evidence,
                      formula="raw[j] * (std[j] + 1e-6) + mean[j] + (state8[j] if j < 6 else 0)",
                      scope="Independent scalar reconstruction using fixed base action statistics and pi0_libero six-dimensional delta-action mask",
                      references=[str(ROOT / p) for p in [
                          "policies/realtime-vla-flash/scripts/spec/spec_serve_policy.py",
                          "policies/realtime-vla-flash/src/openpi/transforms.py",
                          "policies/realtime-vla-flash/src/openpi/training/config.py"]],
                      checks=normalization_checks),
                  run_evidence=run_evidence, excluded_runs=skipped,
                  limits=["Checks only the stored 12x7 action prefixes of first Full calls, not all H=50 outputs or later divergent observations.",
                          "Shared target metadata and raw/decoded checks assess covered inputs; checkpoint bytes are not rehashed. The small shared normalization asset is hashed and its decoder formula independently checked.",
                          "No success-rate, acceleration, draft quality, or independent replication claim follows from this identity check.",
                          "Request logs are read only through their size at read start; incomplete final JSONL lines are ignored and recorded. Membership uses complete-result mtime at cutoff."])
    return report, snapshots


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/ease_insights_20261103")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if not (math.isfinite(args.atol) and math.isfinite(args.rtol) and args.atol >= 0 and args.rtol >= 0):
        parser.error("Tolerances must be finite and nonnegative")
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the science results tree")
    report, snapshots = audit(root, args.atol, args.rtol)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(report["snapshot_cutoff_ns"] / 1e9, timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_path = output / f"first_requests_{stamp}.json"
    snapshot_data = (json.dumps(snapshots, indent=2, allow_nan=False) + "\n").encode()
    snapshot_path.write_bytes(snapshot_data)
    report["first_request_snapshot"] = dict(path=str(snapshot_path), sha256=sha(snapshot_data), records=len(snapshots))
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    ledger = output / f"audit_{stamp}.json"
    ledger.write_text(data)
    (output / "ledger.json").write_text(data)
    print(json.dumps(dict(ledger=str(ledger), snapshot=str(snapshot_path), summary=report["summary"], errors=report["errors"])))
    return 0 if report["summary"]["all_covered_pairs_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
