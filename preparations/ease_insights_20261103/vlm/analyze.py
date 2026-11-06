"""Read-only discovery analysis and explicit, complete-data-only pair freezing.

This module is deliberately outside the frozen execution manifest. It never
changes the service, runner, scientific manifest, or validation launch lock.
"""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import statistics

from preparations.idea1_v3_1.common import atomic_json, digest, lock, read_manifest
from .common import sha


def key(case):
    return case["suite"], case["task"], case["init"]


def exact_paired(a_only, b_only, one_sided=False):
    n = a_only + b_only
    if not n:
        return 1.0
    if one_sided:
        return sum(math.comb(n, k) for k in range(a_only, n+1)) / 2**n
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only)+1)) / 2**n)


def paired(a, b, records, cases):
    both = [c for c in cases if (a, key(c)) in records and (b, key(c)) in records]
    counts = Counter((records[a, key(c)]["success"], records[b, key(c)]["success"]) for c in both)
    n, a_only, b_only = len(both), counts[True, False], counts[False, True]
    return dict(a=a, b=b, n_paired=n, expected_n=len(cases), a_success=counts[True, True]+a_only,
        b_success=counts[True, True]+b_only, both_success=counts[True, True], both_failure=counts[False, False],
        a_only=a_only, b_only=b_only, success_difference_count=a_only-b_only,
        success_difference=None if not n else (a_only-b_only)/n,
        exact_mcnemar_two_sided=exact_paired(a_only, b_only),
        interpretation="Exploratory paired comparison; selection makes these p-values descriptive")


def expected_cases(manifest):
    return [dict(suite=s, task=t, init=i) for s in manifest["suites"] for t in manifest["tasks"] for i in manifest["discovery_inits"]]


def load_episodes(root, manifest):
    records, missing, invalid, evidence = {}, [], [], []
    for cfg in manifest["conditions"]:
        for case in expected_cases(manifest):
            directory = root/cfg["id"]/"episodes"/case["suite"]/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            receipt_path = directory/"episode_receipt.json"
            identity = dict(condition=cfg["id"], case=case)
            if not receipt_path.exists():
                missing.append(identity); continue
            try:
                receipt = json.loads(receipt_path.read_text())
                if (not receipt["valid"] or receipt["manifest_hash"] != manifest["manifest_hash"] or
                    receipt["condition"] != cfg["id"] or receipt["case"] != case):
                    raise ValueError("Receipt identity mismatch")
                required = {"identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz"}
                if not required.issubset(receipt["artifacts"]):
                    raise ValueError("Receipt does not cover necessary episode artifacts")
                for filename, expected in receipt["artifacts"].items():
                    if Path(filename).name != filename or sha(directory/filename) != expected:
                        raise ValueError("Receipt artifact changed: "+filename)
                result = json.loads((directory/"result.json").read_text())
                if (not result["valid"] or result["manifest_hash"] != manifest["manifest_hash"] or result["case"] != case or
                    result["condition"] != cfg["id"] or result["config"] != cfg or result["success"] != receipt["success"]):
                    raise ValueError("Result identity mismatch")
                rows = [json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
                if len(rows) != result["predictions"] or not rows:
                    raise ValueError("Missing inference rows")
                full = [r for r in rows if r["route"]["refresh"]]
                fast = [r for r in rows if not r["route"]["refresh"]]
                if (len(full) != result["counts"]["full"] or len(fast) != result["counts"]["draft"] or
                    sum(r["executed_length"] for r in rows) != result["actions"]):
                    raise ValueError("Episode execution accounting differs")
                record = dict(condition=cfg["id"], case=case, success=bool(result["success"]),
                    actions=result["actions"], predictions=len(rows), full_requests=len(full), fast_requests=len(fast),
                    policy_ms=[r["timing"]["policy_wall_ms"] for r in rows],
                    full_policy_ms=[r["timing"]["policy_wall_ms"] for r in full],
                    fast_policy_ms=[r["timing"]["policy_wall_ms"] for r in fast],
                    rpc_ms=[r["rpc_ms"] for r in rows], episode_wall_s=result["wall_s"],
                    full_executed_actions=sum(r["executed_length"] for r in full),
                    fast_executed_actions=sum(r["executed_length"] for r in fast),
                    gpu_uuid=receipt["gpu_uuid"], initial_gate=receipt["initial_gate"])
                if any(not math.isfinite(v) or v <= 0 for v in record["policy_ms"]):
                    raise ValueError("Invalid completed policy timing")
                records[cfg["id"], key(case)] = record
                evidence.append(dict(path=str(receipt_path), sha256=sha(receipt_path)))
            except (ValueError, KeyError, OSError, TypeError) as exc:
                invalid.append(dict(**identity, error=str(exc)))
    # Initial-observation equality is checked again at the comparison boundary.
    for case in expected_cases(manifest):
        rows = [records[c["id"], key(case)] for c in manifest["conditions"] if (c["id"], key(case)) in records]
        identities = {(r["initial_gate"]["input_hash"], r["initial_gate"]["seed"]) for r in rows}
        if len(identities) > 1:
            invalid.append(dict(case=case, error="Compared conditions have different initial observation/noise"))
    return records, missing, invalid, evidence


def cost_records(root, manifest):
    services, excluded = [], []
    ids = [c["id"] for c in manifest["conditions"]]
    for path in sorted(root.glob("*/runs/*/fixed_input_cost.json")):
        try:
            protocol = json.loads((path.parent/"protocol.json").read_text())
            gate = json.loads((path.parent/"gate.json").read_text())
            cross = json.loads((path.parent/"cross_service.json").read_text())
            obj = json.loads(path.read_text())
            if (obj["manifest_hash"] != manifest["manifest_hash"] or protocol["manifest_hash"] != manifest["manifest_hash"] or
                gate["manifest_hash"] != manifest["manifest_hash"] or not gate["passed"] or not cross["passed"] or
                not protocol["real_input_parity_passed"]):
                raise ValueError("Cost probe has no matching passed service gates")
            if [s["suite"] for s in obj["suites"]] != manifest["suites"]:
                raise ValueError("Cost suite coverage differs")
            cycle_by_condition = {cid: [] for cid in ids}
            stage_by_condition = {cid: dict(full=[], fast=[]) for cid in ids}
            for suite in obj["suites"]:
                if sorted(b["condition"] for b in suite["conditions"]) != sorted(ids):
                    raise ValueError("Cost condition coverage differs")
                noise = []
                for block in suite["conditions"]:
                    cid, rows = block["condition"], block["measured"]
                    if len(block["warmup"]) != 6 or len(rows) != 12 or [r["round"] for r in rows] != list(range(6,18)):
                        raise ValueError("Cost probe warmup/measurement protocol differs")
                    noise.append([(r["seed"], r["model_noise_sha256"]) for r in rows])
                    for row in rows:
                        full = cid == "full_d10" or row["round"] % 3 == 0
                        value = row["policy_wall_ms"]
                        if row["refresh"] != full or not math.isfinite(value) or value <= 0:
                            raise ValueError("Cost stage or latency differs")
                        stage_by_condition[cid]["full" if full else "fast"].append(value)
                    cycle_by_condition[cid].extend(sum(r["policy_wall_ms"] for r in rows[i:i+3]) for i in range(0,12,3))
                if any(n != noise[0] for n in noise):
                    raise ValueError("Cost condition noises differ")
            means = {cid: statistics.mean(v) for cid,v in cycle_by_condition.items()}
            services.append(dict(path=str(path), sha256=sha(path), gpu_uuid=protocol["gpu_uuid"],
                cycle_mean_ms=means, full_over_policy_cycle={cid: means["full_d10"]/v for cid,v in means.items()},
                stage_policy_mean_ms={cid: {stage: statistics.mean(v) if v else None for stage,v in stages.items()}
                    for cid,stages in stage_by_condition.items()}))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            excluded.append(dict(path=str(path), reason=str(exc)))
    mean = {cid: statistics.mean(s["cycle_mean_ms"][cid] for s in services) for cid in ids} if services else {}
    return dict(valid_services=len(services), services=services, excluded_or_pending=excluded,
        mean_cycle_ms=mean, faster_than_full={cid: value < mean["full_d10"] for cid,value in mean.items() if cid != "full_d10"},
        full_over_policy_cycle={cid: mean["full_d10"]/value for cid,value in mean.items()},
        pooling="Equal weight per complete, matching, gated service; each contributes four equally weighted suite inputs",
        scope="Fixed-input inference cost only; small differences remain uncertain; no acceptance/quality/closed-loop speed claim")


def summary(rows):
    def timing(field):
        values = [v for r in rows for v in r[field]]
        return dict(n=len(values), mean_ms=statistics.mean(values), p50_ms=statistics.median(values), total_ms=sum(values)) if values else dict(n=0, mean_ms=None, p50_ms=None, total_ms=0)
    n = len(rows)
    count = sum(r["success"] for r in rows)
    requests = sum(r["predictions"] for r in rows)
    actions = sum(r["actions"] for r in rows)
    full = sum(r["full_requests"] for r in rows)
    wall = sum(r["episode_wall_s"] for r in rows)
    return dict(n=n, successes=count, success_rate=count/n if n else None, actions=actions, predictions=requests,
        full_requests=full, fast_requests=requests-full, full_request_fraction=full/requests if requests else None,
        full_executed_actions=sum(r["full_executed_actions"] for r in rows),
        fast_executed_actions=sum(r["fast_executed_actions"] for r in rows),
        policy=timing("policy_ms"), full_policy=timing("full_policy_ms"), fast_policy=timing("fast_policy_ms"), rpc=timing("rpc_ms"),
        summed_episode_wall_s=wall, episode_per_summed_worker_hour=n*3600/wall if wall else None,
        actions_per_summed_worker_second=actions/wall if wall else None,
        wall_scope="Sum of completed episode wall times; excludes preparation and is not parallel makespan")


def select_pair(manifest, records, costs, complete):
    """Deterministic selection, only from complete discovery and measured costs."""
    if not complete:
        return dict(status="incomplete_discovery", selected=None, eligible_crossings=[])
    if not costs["valid_services"]:
        return dict(status="missing_valid_cost_probe", selected=None, eligible_crossings=[])
    candidates = [c["id"] for c in manifest["conditions"] if c["id"] != "full_d10"]
    all_cases = expected_cases(manifest)
    rates = {cid: sum(records[cid,key(c)]["success"] for c in all_cases)/len(all_cases) for cid in candidates}
    gates = {cid: dict(success_rate=rates[cid], success_rate_pass=rates[cid] >= .60,
        mean_cycle_ms=costs["mean_cycle_ms"][cid], full_mean_cycle_ms=costs["mean_cycle_ms"]["full_d10"],
        cost_pass=costs["faster_than_full"][cid]) for cid in candidates}
    eligible = []
    for a,b in itertools.combinations(candidates,2):
        if not all(gates[cid]["success_rate_pass"] and gates[cid]["cost_pass"] for cid in (a,b)):
            continue
        differences = {}
        for suite in manifest["suites"]:
            rows = [c for c in all_cases if c["suite"] == suite]
            differences[suite] = paired(a,b,records,rows)["success_difference"]
        for a_suite in manifest["suites"]:
            for b_suite in manifest["suites"]:
                if differences[a_suite] > 0 and differences[b_suite] < 0:
                    eligible.append(dict(pair=[a,b], directions=[dict(suite=a_suite, better=a, worse=b, discovery_difference=differences[a_suite]),
                        dict(suite=b_suite, better=b, worse=a, discovery_difference=-differences[b_suite])],
                        objective_min_opposite_suite_difference=min(differences[a_suite],-differences[b_suite]),
                        tie_key=[candidates.index(a),candidates.index(b),manifest["suites"].index(a_suite),manifest["suites"].index(b_suite)]))
    eligible.sort(key=lambda x: (-x["objective_min_opposite_suite_difference"], *x["tie_key"]))
    return dict(status="selected" if eligible else "none", selected=eligible[0] if eligible else None,
        condition_eligibility=gates, eligible_crossings=eligible,
        rule="Both success rates >=0.60 and both measured mean three-request costs < Full; maximize min(two opposite suite differences), then fixed condition/suite order",
        condition_order=candidates, suite_order=manifest["suites"],
        limitation="Discovery selects the pair and category labels; it is not independent evidence of reproducibility")


def confirmation_plan(manifest, selection, selection_frozen):
    chosen = selection["selected"]
    return dict(status="not_executed", executable=False, discovery_manifest_hash=manifest["manifest_hash"],
        selection_frozen=selection_frozen, selected_pair=None if chosen is None else chosen["pair"],
        preregistered_directions=[] if chosen is None else chosen["directions"],
        inits=list(range(31,36)), suites=manifest["suites"], tasks=manifest["tasks"],
        conditions=None if chosen is None else ["full_d10", *chosen["pair"]],
        episodes_per_condition=200, total_if_selected=600,
        protocol="Same fixed global masks, own residual extrapolation, Full-D10/Fast-D2/Fast-D2, H50/R5, matched official initial states and request noise",
        launch_requirement="New output directory, new confirmation manifest, independent wrapper; current discovery confirm lock remains untouched",
        analysis="Paired success difference, discordant counts and exact paired tests in the two frozen suites; other suite/task outcomes descriptive",
        proposed_interpretation="Report direction reproduction separately from statistical resolution; apply Holm correction to the two prespecified one-sided exact paired tests if used",
        boundary="Do not rename preferred suites or reselect masks from validation outcomes; universal dominance supports a fixed path, not complementarity")


def analyze(root):
    root = Path(root).resolve()
    manifest = read_manifest(root/"manifest.json")
    records, missing, invalid, episode_evidence = load_episodes(root, manifest)
    costs = cost_records(root, manifest)
    all_cases = expected_cases(manifest)
    expected = len(all_cases)*len(manifest["conditions"])
    complete = len(records) == expected and not missing and not invalid
    groups = [("all",None,None,all_cases)]
    groups += [("suite",s,None,[c for c in all_cases if c["suite"] == s]) for s in manifest["suites"]]
    groups += [("task",s,t,[c for c in all_cases if c["suite"] == s and c["task"] == t]) for s in manifest["suites"] for t in manifest["tasks"]]
    comparisons, condition_rows = [], []
    ids = [c["id"] for c in manifest["conditions"]]
    for level,suite,task,scenes in groups:
        for cid in ids:
            rows = [records[cid,key(c)] for c in scenes if (cid,key(c)) in records]
            condition_rows.append(dict(level=level,suite=suite,task=task,condition=cid,expected_n=len(scenes),**summary(rows)))
        for a,b in itertools.combinations(ids,2):
            comparisons.append(dict(level=level,suite=suite,task=task,**paired(a,b,records,scenes)))
    selection = select_pair(manifest, records, costs, complete)
    return dict(manifest_hash=manifest["manifest_hash"], manifest_path=str(root/"manifest.json"),
        expected_episodes=expected, valid_episodes=len(records), complete=complete, missing=missing, invalid=invalid,
        condition_summaries=condition_rows, paired_comparisons=comparisons, fixed_input_cost=costs,
        selection=selection, interpretation="Task rows have one discovery initial state and are case diagnostics, not task-category evidence; suite comparisons determine selection",
        evidence=dict(episodes=episode_evidence,cost_services=[dict(path=s["path"],sha256=s["sha256"]) for s in costs["services"]],
            analysis_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))))


def freeze(output, report):
    if (not report["complete"] or report["valid_episodes"] != 160 or report["expected_episodes"] != 160 or
        report["selection"]["status"] not in ("selected","none")):
        raise ValueError("Cannot freeze selection without all 160 valid outcomes and a matching valid cost probe")
    payload = dict(discovery_manifest_hash=report["manifest_hash"], complete_discovery_episodes=report["valid_episodes"],
        selection=report["selection"], evidence=report["evidence"], independent_validation_launched=False)
    payload["selection_hash"] = digest(payload)
    path = output/"selection.json"
    with lock(output/"selection.lock"):
        if path.exists():
            if json.loads(path.read_text()) != payload:
                raise ValueError("Existing frozen selection differs; refusing to overwrite")
        else:
            atomic_json(path,payload)
    return payload


def markdown(report):
    lines = [f"VLM discovery: {report['valid_episodes']}/{report['expected_episodes']} valid episodes.",
        f"Selection status: {report['selection']['status']}. Independent validation has not run.", "",
        "| Condition | Suite | n | Success | Full request fraction | Policy mean ms |", "|---|---|---:|---:|---:|---:|"]
    for row in report["condition_summaries"]:
        if row["level"] == "task": continue
        fmt = lambda x: "pending" if x is None else f"{x:.3f}"
        lines.append(f"| {row['condition']} | {row['suite'] or 'all'} | {row['n']} | {row['successes']} | {fmt(row['full_request_fraction'])} | {fmt(row['policy']['mean_ms'])} |")
    lines += ["", "All same-case comparisons, discordant counts, task diagnostics, timings, and cost probes are in analysis.json.",
        "A task row is one initial-state outcome, not evidence for a task category. Selected suite differences require independent init31–35 validation."]
    if report["selection"]["selected"]:
        chosen = report["selection"]["selected"]
        lines += ["", "Selected discovery pair: "+", ".join(chosen["pair"])+"."]
        lines += [f"- {d['suite']}: {d['better']} > {d['worse']}, discovery difference {d['discovery_difference']:+.3f}." for d in chosen["directions"]]
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root",required=True); parser.add_argument("--output",required=True)
    parser.add_argument("--freeze-selection",action="store_true")
    args = parser.parse_args(); output = Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=True)
    report = analyze(args.root)
    atomic_json(output/"analysis.json",report)
    (output/"analysis.md").write_text(markdown(report))
    frozen = freeze(output,report) if args.freeze_selection else None
    if frozen is None and (output/"selection.json").exists():
        frozen = json.loads((output/"selection.json").read_text())
        if frozen["discovery_manifest_hash"] != report["manifest_hash"]:
            raise ValueError("Output directory contains selection from another discovery manifest")
    manifest = read_manifest(report["manifest_path"])
    atomic_json(output/"confirmation_plan.json",confirmation_plan(manifest,
        report["selection"] if frozen is None else frozen["selection"],frozen is not None))
    print(json.dumps(dict(valid_episodes=report["valid_episodes"],expected_episodes=report["expected_episodes"],
        complete=report["complete"],selection_status=report["selection"]["status"],
        selection_hash=None if frozen is None else frozen["selection_hash"],output=str(output))))


if __name__ == "__main__": main()
