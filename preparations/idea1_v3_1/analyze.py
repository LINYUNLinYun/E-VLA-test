"""Exploratory screen/confirmation rates and measured throughput, never guessed floors."""
import argparse
from collections import defaultdict
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .common import atomic_json,read_manifest


def summarize(manifest,output):
    output = Path(output)
    rows,technical = defaultdict(list),[]
    for path in output.rglob("result.json"):
        if ".invalid." in str(path):
            continue
        row = json.loads(path.read_text())
        if row.get("valid") and (path.parent/"complete.json").exists():
            if row.get("manifest_hash")!=manifest["manifest_hash"]:
                raise ValueError("Mixed manifest results")
            row["_directory"] = str(path.parent)
            rows[(row["condition"],path.parents[3].name)].append(row)
    for path in output.rglob("technical_error.json"):
        technical.append(str(path))
    quality = []
    for condition in manifest["conditions"]:
        data = rows.get((condition["id"],"episodes"),[])
        if not data:
            continue
        baseline = rows.get((condition["model"]+"_full","episodes"),[])
        by_scene = {(r["suite"],r["task"],r["init"]):r for r in baseline}
        paired = [r for r in data if (r["suite"],r["task"],r["init"]) in by_scene]
        full_successes = sum(by_scene[(r["suite"],r["task"],r["init"])]["success"] for r in paired)
        variant_successes = sum(r["success"] for r in paired)
        retained = variant_successes/full_successes if full_successes else None
        quality.append(dict(condition=condition["id"],model=condition["model"],skip=condition["skip"],valid=len(data),
                            successes=sum(r["success"] for r in data),success_rate=float(np.mean([r["success"] for r in data])),
                            per_suite={suite:dict(successes=sum(r["success"] for r in data if r["suite"]==suite),
                                                 valid=sum(r["suite"]==suite for r in data)) for suite in manifest["suites"]},
                            paired_valid=len(paired),paired_variant_successes=variant_successes,paired_full_successes=full_successes,
                            relative_retention=retained,passes_70_percent_floor=retained is not None and retained>=.70,
                            failure_furthest_phase=dict(Counter(r.get("furthest_phase","unknown") for r in data if not r["success"])),
                            phase_caveat="Heuristic progress evidence, not a causal failure diagnosis",
                            data_use="Exploratory screen and candidate selection share observations"))
    throughput = []
    for path in output.rglob("throughput_*.json"):
        row = json.loads(path.read_text())
        variant = row.get("variant") or "episodes"
        episodes = rows.get((row["condition"],variant),[])
        measured = row.get("newly_completed_episodes",len(episodes))
        throughput.append(dict(row,completed_episodes=len(episodes),episodes_per_hour=measured*3600/row["wall_s"] if row["wall_s"] and measured else None,
                               warning="Aggregate throughput includes environment/RPC/I/O; not a model latency speedup"))
    performance = []
    measured = {}
    for path in output.glob("*/benchmark.json"):
        result = json.loads(path.read_text())
        if result["manifest_hash"]!=manifest["manifest_hash"]:
            raise ValueError("Mixed benchmark manifest")
        measured[result["condition"]["id"]] = result
    for cid,result in measured.items():
        model = result["condition"]["model"]
        native = measured.get(model+"_full")
        matched = measured.get(model+"_full_vision_scan_reference")
        performance.append(dict(condition=cid,model=model,summary=result["summary"],cold=result.get("cold"),
                                native_full_speedups={key:native["summary"][key]["p50"]/result["summary"][key]["p50"]
                                                      for key in ("vision_ms","request_roundtrip_ms")} if native else None,
                                matched_full_speedups={key:matched["summary"][key]["p50"]/result["summary"][key]["p50"]
                                                       for key in ("vision_ms","request_roundtrip_ms")} if matched else None))
    frontier = {}
    for model in ("xiaomi-robotic0","db-cogact"):
        passed = [row for row in quality if row["model"]==model and row["skip"] and row["passes_70_percent_floor"]]
        frontier[model] = dict(maximum_tested_skip_meeting_relative_floor=max((len(row["skip"]) for row in passed),default=None),
                              candidates=[dict(condition=row["condition"],valid=row["valid"],relative_retention=row["relative_retention"])
                                          for row in passed],interpretation="Largest usable tested candidate, not an optimum over all layer masks")
    optimizations = []
    for condition in manifest["conditions"]:
        if condition["skip"]:
            continue
        baseline = rows.get((condition["id"],"full_io"),[])
        if len(baseline)!=8:
            continue
        reference = {(row["suite"],row["task"],row["init"]):row for row in baseline}
        for variant in ("lean_io","reuse_env","two_workers"):
            data = rows.get((condition["id"],variant),[])
            paired = len(data)==8
            largest = 0.
            for row in data:
                original = reference.get((row["suite"],row["task"],row["init"]))
                if original is None or row["init_vector_sha256"]!=original["init_vector_sha256"] or row["success"]!=original["success"]:
                    paired = False
                    continue
                a = [json.loads(line) for line in (Path(original["_directory"])/"infer.jsonl").read_text().splitlines()]
                b = [json.loads(line) for line in (Path(row["_directory"])/"infer.jsonl").read_text().splitlines()]
                if len(a)!=len(b):
                    paired = False
                    continue
                for left,right in zip(a,b,strict=True):
                    if any(left[key]!=right[key] for key in ("seed","input_hash","noise_hash")):
                        paired = False
                    error = np.max(np.abs(np.asarray(left["actions"])-np.asarray(right["actions"])))
                    largest = max(largest,float(error))
                    if not np.allclose(left["actions"],right["actions"],atol=1e-5,rtol=1e-5):
                        paired = False
            timed = {row["variant"]:row for row in throughput if row["condition"]==condition["id"]
                     and row.get("newly_completed_episodes")==8}
            speedup = timed[variant]["episodes_per_hour"]/timed["full_io"]["episodes_per_hour"] if variant in timed and "full_io" in timed else None
            optimizations.append(dict(condition=condition["id"],variant=variant,paired_valid_episodes=len(data),
                                      same_initial_states_outcomes_inputs_noise_outputs=paired,max_action_abs_error=largest,
                                      aggregate_throughput_speedup=speedup,
                                      eligible_for_adoption=paired and speedup is not None and speedup>1,
                                      caveat="Exact scene budgets and steady services; rendering remains enabled; this is throughput, not exclusive model latency"))
    return dict(manifest_hash=manifest["manifest_hash"],quality=quality,throughput=throughput,
                performance=performance,frontier=frontier,evaluation_optimizations=optimizations,
                technical_errors=technical,completion_is_not_implied_by_code=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--root",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    report = summarize(read_manifest(args.manifest),args.root)
    atomic_json(args.output,report)
    print(json.dumps(dict(quality_conditions=len(report["quality"]),throughput_arms=len(report["throughput"]),
                          technical_errors=len(report["technical_errors"]))))


if __name__ == "__main__":
    main()
