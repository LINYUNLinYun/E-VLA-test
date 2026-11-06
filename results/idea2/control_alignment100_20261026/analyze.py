"""Paired fixed100 statistics and bounded CPU artifact checks."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from results.idea2.continuous_encoder_20261026.statistics import (
    case_grid, wilson, exact_mcnemar, task_cluster_bootstrap_ci, mean_bootstrap_ci, write_csv)
from results.idea2.control_alignment_20261026.summarize import stats
from .runner import R, save


def key(row):
    return row["suite"], row["task"], row["init"]


def lines(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def comparison(a, b, keys):
    differences = {k:int(b[k]["success"])-int(a[k]["success"]) for k in keys}
    tasks = defaultdict(list)
    for k, v in differences.items(): tasks[k[:2]].append(v)
    gained = sum(v == 1 for v in differences.values())
    lost = sum(v == -1 for v in differences.values())
    return dict(n=len(keys), candidate_only_success=gained, reference_only_success=lost,
        difference=(gained-lost)/len(keys), exact_mcnemar_two_sided_p=exact_mcnemar(gained, lost),
        task_cluster95=task_cluster_bootstrap_ci(tasks, draws=10000, seed=20261026),
        episode95_sensitivity=mean_bootstrap_ci(list(differences.values()), draws=10000, seed=20261027),
        task_clusters=len(tasks), task_cluster_sizes=sorted(map(len,tasks.values())),
        caution="Exploratory, includes observed pilot; no noninferiority margin; task clustering retained")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    state = json.loads((R/"dispatch_execution.json").read_text())
    rows = json.loads((R/"normalized_cases.json").read_text())["cases"]
    grid = case_grid()
    selected = {mode:{} for mode in ("off", "control_steps")}
    for r in rows:
        k = key(r)
        if r["configuration"] not in selected or k not in grid or k in selected[r["configuration"]]:
            raise ValueError("Unexpected or duplicate expanded cell")
        selected[r["configuration"]][k] = r
    complete = state["status"] == "passed" and len(rows) == 200 and all(r["status"] == "valid" for r in rows)
    if not args.partial and not complete:
        raise ValueError("Expanded execution incomplete")
    configurations = []
    for mode in selected:
        valid = [r for r in selected[mode].values() if r["status"] == "valid"]
        for scope in ("overall", "libero_object", "libero_spatial", "libero_goal", "libero_10"):
            subset = [r for r in valid if scope == "overall" or r["suite"] == scope]
            n = len(subset); successes = sum(r["success"] for r in subset)
            configurations.append(dict(configuration=mode, scope=scope, n=n, successes=successes,
                rate=successes/n if n else None, wilson95=wilson(successes,n),
                horizon_exhausted=sum(r["result"]["termination"] == "prediction_horizon_exhausted" for r in subset)))
    paired, numerical = [], []
    exception_path = R/"cross_gpu_exception.json"
    allowed_cross = {tuple(k) for k in json.loads(exception_path.read_text())["pairs"]} if exception_path.exists() else set()
    heads = {}
    for mode in selected:
        for k, r in selected[mode].items():
            if r["status"] != "valid": continue
            with (Path(r["directory"])/"infer.jsonl").open() as f:
                head = json.loads(next(f))
            with np.load(head["prediction_path"]) as f:
                heads[(mode,k)] = dict(input_hash=head["input_hash"], seed=head["seed"],
                                      actions=f["actions"].copy(), normalized=f["normalized_actions"].copy())
    for k in sorted(grid):
        a, b = selected["off"][k], selected["control_steps"][k]
        if a["status"] != "valid" or b["status"] != "valid": continue
        if a["gpu"] != b["gpu"] and k not in allowed_cross:
            raise ValueError("Paired cases used different GPUs")
        x, y = heads[("off", k)], heads[("control_steps", k)]
        if (x["input_hash"], x["seed"]) != (y["input_hash"], y["seed"]):
            raise ValueError("Paired bootstrap identity changed")
        for field in ("actions", "normalized"):
            e = np.abs(x[field].astype(float)-y[field].astype(float))
            if e.max()>1e-5 or e.mean()>1e-6:
                raise ValueError("Bootstrap tolerance failed")
            numerical.append(dict(suite=k[0],task=k[1],init=k[2],field=field,mae=float(e.mean()),max_abs=float(e.max())))
        paired.append(dict(suite=k[0],task=k[1],init=k[2],off_success=a["success"],offset_success=b["success"],
            gained=int(b["success"] and not a["success"]),lost=int(a["success"] and not b["success"]),
            off_steps=a["steps"],offset_steps=b["steps"],gpu=a["gpu"],offset_gpu=b["gpu"],same_gpu=a["gpu"]==b["gpu"],source_kind=a["source_kind"],
            off_directory=a["directory"],offset_directory=b["directory"]))
    comparisons = {}
    if complete:
        comparisons["primary"] = comparison(selected["off"],selected["control_steps"],sorted(grid))
        new = sorted(k for k in grid if selected["off"][k]["source_kind"] == "expanded")
        comparisons["excluding_pilot"] = comparison(selected["off"],selected["control_steps"],new)
        same_gpu = sorted(k for k in grid if selected["off"][k]["gpu"] == selected["control_steps"][k]["gpu"])
        if len(same_gpu) != len(grid):
            comparisons["same_gpu_pairs_only"] = comparison(selected["off"],selected["control_steps"],same_gpu)
    out = R/("partial_analysis" if args.partial else "analysis")
    out.mkdir(exist_ok=True)
    metrics = defaultdict(lambda:defaultdict(list))
    episodes, first_actions = [], []
    predictions = applications = recaptures = 0
    if not args.partial:
        for r in rows:
            mode = r["configuration"]
            leaf = Path(r["directory"])
            infers = lines(leaf/"infer.jsonl")
            chunks = lines(leaf/"consumed_chunks.jsonl")
            result = r["result"]
            predictions += len(infers)
            applications += result["steps"]
            metrics[mode]["episode_work_s"].append(result["all_work_wall_s"])
            metrics[mode]["steps"].append(result["steps"])
            episodes.append(dict(configuration=mode,suite=r["suite"],task=r["task"],init=r["init"],gpu=r["gpu"],
                source_kind=r["source_kind"],success=r["success"],steps=r["steps"],predictions=len(infers),
                all_work_wall_s=result["all_work_wall_s"],actor_wall_s=result["actor_wall_s"],termination=result["termination"],
                directory=r["directory"]))
            for a in infers:
                recaptures += a.get("graph_recaptures",0)
                if a["input_step"]>0: metrics[mode]["rpc_ms"].append(a["client_roundtrip_ms"])
            for a in chunks:
                if a["target_chunk_id"] == 0: continue
                age = a["execution_start_step"]-a["input_step"]
                offset = a["skipped_action_prefix"]
                residual = age-offset
                if mode == "control_steps" and residual != 0:
                    raise ValueError("Nonzero aligned residual")
                age_ms = 1000*(a["execution_start_host"]-a["input_capture_host"])
                for k,v in (("first_action_age_steps",age),("offset_steps",offset),("residual_steps",residual),("first_action_age_ms",age_ms)):
                    metrics[mode][k].append(v)
                first_actions.append(dict(configuration=mode,suite=r["suite"],task=r["task"],init=r["init"],
                    request_id=a["request_id"],input_step=a["input_step"],application_step=a["execution_start_step"],
                    age_steps=age,offset_steps=offset,residual_steps=residual,age_ms=age_ms))
            gpu = json.loads((leaf/"gpu_records.json").read_text())["records"]
            for a in gpu:
                if a["input_step"]<=0: continue
                for k in ("vision_ms","vlm_ms","ae_ms"): metrics[mode][k].append(a[k])
                a0 = a["ae_start_after_vlm_start_ms"];a1 = a0+a["ae_ms"]
                metrics[mode]["ae_vlm_event_intersection_ms"].append(max(0.,min(a1,a["vlm_ms"])-max(a0,0.)))
                if a["next_encoder"]:
                    n1 = a1-a["ae_finish_minus_next_encoder_ms"];n0 = n1-a["next_vision_ms"]
                    metrics[mode]["ae_encoder_event_intersection_ms"].append(max(0.,min(a1,n1)-max(a0,n0)))
        for name,values in (("episodes",episodes),("first_actions",first_actions)):
            write_csv(out/(name+".csv"),values)
    timing = {m:{k:stats(v) for k,v in values.items()} for m,values in metrics.items()}
    histogram = {m:dict(Counter(values["offset_steps"])) for m,values in metrics.items()}
    for name, values in (("configurations",configurations),("paired",paired),("bootstrap",numerical)):
        write_csv(out/(name+".csv"),values)
    historical = json.loads((R/"synchronous_reference.json").read_text())
    report = dict(status="complete" if complete else "partial",valid_cells=sum(r["status"] == "valid" for r in rows),
        configurations=configurations,comparisons=comparisons,matched_bootstrap_pairs=len(paired),
        bootstrap_max_abs=max((r["max_abs"] for r in numerical),default=None),
        predictions=predictions if not args.partial else None,applications=applications if not args.partial else None,
        graph_recaptures=recaptures if not args.partial else None,timing=timing,offset_histogram=histogram,
        synchronous_same100_historical=historical["successes"],
        cross_gpu_pairs=[list(k) for k in sorted(allowed_cross)],
        limits="No noninferiority claim; observed pilot retained, separate92-pair sensitivity; one resource-induced cross-GPU pair also excluded in sensitivity; no new nsys collection")
    save(out/"summary.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("timing","configurations")},indent=2))


if __name__ == "__main__":
    main()
