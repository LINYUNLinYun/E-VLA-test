"""Small pilot summary; no inference or trajectory re-execution."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent
LABELS = ("sync_fresh18", "sync_sm16", "async_unaligned", "async_offset")


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(n=len(a), mean=float(a.mean()), p50=float(np.median(a)), p95=float(np.percentile(a, 95)),
                minimum=float(a.min()), maximum=float(a.max())) if len(a) else None


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def csv_write(path, rows):
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else ["status"]
    with path.open("w") as f:
        out = csv.DictWriter(f, fieldnames=keys)
        out.writeheader()
        for r in rows:
            out.writerow({k:json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    runs = []
    for name in ("sync_gate", "async_gate0", "async_gate1"):
        f = R/name/"execution.json"
        if not f.exists():
            if args.partial: continue
            raise ValueError("Missing planned stage: "+name)
        d = json.loads(f.read_text())
        if not args.partial and (d["status"] != "passed" or not d.get("source_unchanged")):
            raise ValueError("Incomplete/changed stage: "+name)
        runs.append((name, d))
    episodes, applications, chunks = [], [], []
    data = defaultdict(lambda:defaultdict(list))
    bootstraps = defaultdict(dict)
    total_predictions = 0
    recaptures = 0
    for name, d in runs:
        for c in d["cases"]:
            if c["status"] != "valid": continue
            sync = name == "sync_gate"
            label = ("sync_fresh18" if c["label"] == "fresh18" else "sync_sm16") if sync else (
                "async_offset" if c["alignment"] == "control_steps" else "async_unaligned")
            leaf = R/name/c["relative_directory"] if sync else Path(c["directory"])
            result = c["result"]
            infers = read_rows(leaf/"infer.jsonl")
            total_predictions += len(infers)
            case = f"{c['suite']}/{c['task']}/{c['init']}"
            head = infers[0]
            if sync:
                normalized = np.asarray(head["normalized_actions"])
            else:
                with np.load(head["prediction_path"]) as f:
                    normalized = f["normalized_actions"].copy()
            bootstraps[case][label] = dict(input_hash=head["input_hash"], seed=head["seed"],
                actions=np.asarray(head["chunk_actions"]), normalized=normalized)
            work = result["all_work_wall_s"]
            termination = result["runtime_result"]["termination"] if sync else result["termination"]
            episode = dict(configuration=label, case=case, suite=c["suite"], task=c["task"], init=c["init"],
                success=result["success"], steps=result["steps"], predictions=len(infers),
                all_work_wall_s=work, termination=termination, directory=str(leaf))
            episodes.append(episode)
            data[label]["episode_work_s"].append(work)
            data[label]["steps"].append(result["steps"])
            if sync:
                expected_step = 0
                telemetry = read_rows(leaf/"telemetry.jsonl")
                for r in infers:
                    if r["step"] != expected_step:
                        raise ValueError("Synchronous prediction was not after the previous complete R5")
                    for j in range(r["executed_length"]):
                        a = telemetry[expected_step+j+1]
                        if not np.array_equal(np.asarray(a["action"], np.float32), np.asarray(r["chunk_actions"][j], np.float32)):
                            raise ValueError("Synchronous action prefix changed")
                    expected_step += r["executed_length"]
                    if not r["bootstrap"]:
                        data[label]["first_action_age_steps"].append(0)
                        data[label]["offset_steps"].append(0)
                        data[label]["first_action_residual_steps"].append(0)
                        chunks.append(dict(configuration=label, case=case, input_step=r["step"],
                            application_step=r["step"], offset_steps=0, first_action_age_steps=0,
                            first_action_residual_steps=0, first_action_age_ms=None))
                    if not r["bootstrap"] and not r["graph_setup_ms"]:
                        data[label]["rpc_ms"].append(r["client_roundtrip_ms"])
                        if r["progressive"].get("iteration_ms"):
                            data[label]["ae_ms"].append(sum(r["progressive"]["iteration_ms"]))
                    recaptures += r.get("graph_recaptures", r.get("progressive", {}).get("graph_recaptures", 0))
                if expected_step != result["steps"]:
                    raise ValueError("Synchronous step accounting mismatch")
            else:
                selected = read_rows(leaf/"consumed_chunks.jsonl")
                telemetry = read_rows(leaf/"telemetry.jsonl")[1:]
                for r in selected:
                    if r["target_chunk_id"] == 0: continue
                    age = r["execution_start_step"]-r["input_step"]
                    offset = r["skipped_action_prefix"]
                    age_ms = 1000*(r["execution_start_host"]-r["input_capture_host"])
                    for key, value in (("first_action_age_steps", age), ("offset_steps", offset),
                                       ("first_action_residual_steps", age-offset), ("first_action_age_ms", age_ms)):
                        data[label][key].append(value)
                    chunks.append(dict(configuration=label, case=case, input_step=r["input_step"],
                        application_step=r["execution_start_step"], offset_steps=offset,
                        first_action_age_steps=age, first_action_residual_steps=age-offset, first_action_age_ms=age_ms))
                for a in telemetry:
                    if a["chunk_id"]:
                        applications.append(dict(configuration=label, case=case, step=a["step"],
                            request=a["consumed_request_id"], action_index=a["action_index"],
                            offset_steps=a["skipped_action_prefix"], input_age_steps=a["observation_age_steps"],
                            input_age_ms=a["observation_age_ms"], residual_steps=a["alignment_residual_steps"]))
                for r in infers:
                    recaptures += r.get("graph_recaptures", 0)
                    if r["input_step"]>0:
                        data[label]["rpc_ms"].append(r["client_roundtrip_ms"])
                gpu = json.loads((leaf/"gpu_records.json").read_text())["records"]
                for r in gpu:
                    if r["input_step"]<=0: continue
                    for key in ("ae_ms", "vlm_ms", "vision_ms"):
                        data[label][key].append(r[key])
                    a0 = r["ae_start_after_vlm_start_ms"]
                    a1 = a0+r["ae_ms"]
                    data[label]["ae_vlm_event_intersection_ms"].append(max(0., min(a1, r["vlm_ms"])-max(a0, 0.)))
                    if r["next_encoder"]:
                        n1 = a1-r["ae_finish_minus_next_encoder_ms"]
                        n0 = n1-r["next_vision_ms"]
                        data[label]["ae_encoder_event_intersection_ms"].append(max(0., min(a1, n1)-max(a0, n0)))
    summaries = []
    for label in LABELS:
        es = [e for e in episodes if e["configuration"] == label]
        if not es: continue
        summaries.append(dict(configuration=label, episodes=len(es), successes=sum(e["success"] for e in es),
            horizon_exhausted=sum(e["termination"] == "prediction_horizon_exhausted" for e in es),
            total_steps=sum(e["steps"] for e in es), total_predictions=sum(e["predictions"] for e in es),
            offset_histogram=dict(Counter(data[label]["offset_steps"])),
            **{k:stats(v) for k, v in data[label].items()}))
    bootstrap_checks = []
    for case, variants in bootstraps.items():
        if "sync_fresh18" not in variants: continue
        ref = variants["sync_fresh18"]
        for label, value in variants.items():
            if label == "sync_fresh18": continue
            if (ref["seed"], ref["input_hash"]) != (value["seed"], value["input_hash"]):
                raise ValueError("Bootstrap input/noise identity mismatch")
            for field in ("actions", "normalized"):
                error = np.abs(ref[field].astype(float)-value[field].astype(float))
                bootstrap_checks.append(dict(case=case, configuration=label, field=field,
                    mae=float(error.mean()), max_abs=float(error.max()), passed=bool(error.mean()<=1e-6 and error.max()<=1e-5)))
    if any(not r["passed"] for r in bootstrap_checks):
        raise ValueError("Original bootstrap numerical tolerance failed")
    paired = []
    lookup = {(e["configuration"], e["case"]):e for e in episodes}
    for reference, candidate in (("sync_fresh18", "sync_sm16"), ("async_unaligned", "async_offset")):
        for case in bootstraps:
            a, b = lookup.get((reference, case)), lookup.get((candidate, case))
            if a and b:
                paired.append(dict(reference=reference, candidate=candidate, case=case,
                    reference_success=a["success"], candidate_success=b["success"],
                    gained=int(b["success"] and not a["success"]), lost=int(a["success"] and not b["success"]),
                    reference_steps=a["steps"], candidate_steps=b["steps"]))
    if not args.partial and (len(episodes) != 32 or any(r["episodes"] != 8 for r in summaries) or len(summaries) != 4):
        raise ValueError("Fixed 32-episode grid is incomplete")
    out = R/("partial_analysis" if args.partial else "analysis")
    out.mkdir(exist_ok=True)
    for name, rows in (("episodes", episodes), ("conditions", summaries), ("first_actions", chunks),
                       ("applications", applications), ("bootstrap", bootstrap_checks), ("paired", paired)):
        csv_write(out/(name+".csv"), rows)
    report = dict(status="partial" if args.partial else "passed", episodes=len(episodes),
        predictions=total_predictions, conditions=summaries, bootstrap_checks=len(bootstrap_checks),
        bootstrap_max_abs=max((r["max_abs"] for r in bootstrap_checks), default=None),
        steady_graph_recaptures=recaptures,
        scope="Fixed eight-case exploratory gate; Event envelope intersections are not Kernel busy overlap; sync and async clocks differ")
    (out/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
