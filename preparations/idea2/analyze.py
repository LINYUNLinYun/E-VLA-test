"""Summarize completed experiment artifacts without touching raw results."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path

import numpy as np


def percentiles(values):
    return dict(zip(("p50", "p95", "p99"), np.percentile(values, [50, 95, 99]).tolist())) if values else None


def analyze(root):
    root = Path(root)
    report = dict(e0=[], e1=[], e1b=[], e2=[], e2_age=[], e3=[], videos=[], simulation_ages=[])
    unique_replay = {}
    for path in root.rglob("e0.json"):
        row = json.loads(path.read_text())
        report["e0"].append(dict(path=str(path), passed=row["passed"], max_abs=row["normalized"]["max_abs"],
                                  kv_unchanged=row["kv_unchanged"], original_steps=row.get("original_denoise_calls")))
    for path in root.rglob("e1.json"):
        data = json.loads(path.read_text())
        serial, parallel = data["summary"]["serial"], data["summary"]["two_stream"]
        report["e1"].append(dict(path=str(path), summary=data["summary"], trace_summary=data.get("trace_summary"),
                                  action_ready_speedup=serial["action_ready_host_ms"]["p50"]/parallel["action_ready_host_ms"]["p50"],
                                  all_done_speedup=serial["all_done_host_ms"]["p50"]/parallel["all_done_host_ms"]["p50"],
                                  worst_paired_error=max(r["output_error"]["max_abs"] for r in data["rows"]),
                                  peak_allocated_by_mode={mode:max(r["peak_allocated"] for r in data["rows"] if r["mode"]==mode)
                                                          for mode in ("serial", "two_stream")}))
    for path in root.rglob("e2.json"):
        if "attempts" in path.parts:
            continue
        rows = json.loads(path.read_text())["rows"]
        for row in rows:
            key = (path.parent.name,row["request"],row["prefix_source"])
            if key not in unique_replay:
                unique_replay[key] = row
        for interval in (1, 2, 4):
            selected = [r for r in rows if r["interval"] == interval]
            for horizon in ("first5", "full50"):
                for component in ("position", "rotation", "gripper"):
                    report["e2"].append(dict(suite=path.parent.name, interval=interval, horizon=horizon, component=component,
                                              requests=len(selected), mean_request_rmse=float(np.mean([r["by_horizon"][horizon][component]["rmse"] for r in selected])),
                                              max_abs=max(r["by_horizon"][horizon][component]["max_abs"] for r in selected),
                                              worst_serial_async_error=max(r["serial_async_error"]["max_abs"] for r in selected)))
    for age in (0,1,2,3):
        selected = [row for (_,request,source),row in unique_replay.items() if request-source==age]
        for horizon,length in (("first5",5),("full50",50)):
            if not selected:
                continue
            report["e2_age"].append(dict(request_age=age,horizon=horizon,unique_request_source_pairs=len(selected),
                                         position_coordinate_denominator=len(selected)*length*3,
                                         pooled_position_rmse=float(np.sqrt(np.mean([row["by_horizon"][horizon]["position"]["rmse"]**2 for row in selected]))),
                                         gripper_sign_mismatches=sum(row["by_horizon"][horizon]["gripper_sign_mismatches"] for row in selected),
                                         gripper_action_denominator=len(selected)*length,
                                         gripper_sign_mismatch_rate=sum(row["by_horizon"][horizon]["gripper_sign_mismatches"] for row in selected)/(len(selected)*length),
                                         deduplication="One (suite,request,prefix_source) pair across reuse intervals",units="Decoded action coordinates; no physical distance conversion"))
    for path in root.rglob("e1b.json"):
        if "attempts" not in path.parts:
            result = json.loads(path.read_text())
            report["e1b"].append(dict(path=str(path),identity=result["identity"],summary=result["summary"],
                                      eager_summary=result.get("eager_summary"),original_eager_summary=result.get("original_eager_summary"),
                                      trace_summary=result.get("trace_summary"),scope=result["scope"]))
    conditions = {}
    for path in root.rglob("complete.json"):
        if not (path.parent/"result.json").exists() or "attempts" in path.parts:
            continue
        result = json.loads((path.parent/"result.json").read_text())
        report["videos"].append(dict(mode=result["mode"], interval=result["interval"], suite=result["suite"],
                                      task=result["task"], init=result["init"], success=result["success"],
                                      path=str((path.parent/"video.mp4").relative_to(root))))
        key = (result["mode"], result["interval"])
        group = conditions.setdefault(key, dict(results=[], infers=[]))
        group["results"].append(result)
        infer_rows = [json.loads(line) for line in (path.parent/"infer.jsonl").read_text().splitlines()]
        group["infers"].extend(infer_rows)
        telemetry = [json.loads(line) for line in (path.parent/"telemetry.jsonl").read_text().splitlines()]
        by_request = {row["request"]:row for row in infer_rows}
        for row in infer_rows:
            source_step = by_request[row["prefix_observation_id"]]["step"]
            origin = telemetry[source_step]["sim_time_s"]
            step,n = row["step"],row["executed_length"]
            report["simulation_ages"].append(dict(mode=result["mode"],interval=result["interval"],suite=result["suite"],
                                                   task=result["task"],init=result["init"],request=row["request"],
                                                   source_request=row["prefix_observation_id"],source_step=source_step,current_step=step,
                                                   generation_sim_age_ms=(telemetry[step]["sim_time_s"]-origin)*1000,
                                                   first_execution_sim_age_ms=(telemetry[step+1]["sim_time_s"]-origin)*1000,
                                                   last_execution_sim_age_ms=(telemetry[step+n]["sim_time_s"]-origin)*1000,
                                                   host_receipt_age_at_generation_ms=row["source_receipt_age_at_generation_ms"],
                                                   clock_note="Simulation paused during inference; simulation age measures environment advancement, host age includes the pause"))
    for (mode, interval), data in conditions.items():
        results, infers = data["results"], data["infers"]
        report["e3"].append(dict(mode=mode, interval=interval, episodes=len(results), successes=sum(r["success"] for r in results),
                                  success_rate=float(np.mean([r["success"] for r in results])),
                                  controller_roundtrip_ms=percentiles([r["client_roundtrip_ms"] for r in infers]),
                                  condition_request_age=percentiles([r["observation_request_age"] for r in infers]),
                                  execution_receipt_age_end_ms=percentiles([r["execution_source_receipt_age_ms_end"] for r in infers
                                                                           if r.get("execution_source_receipt_age_ms_end") is not None]),
                                  camera_capture_age_ms=None,
                                  failure_furthest_phase=dict(Counter(r.get("furthest_phase", "unknown") for r in results if not r["success"])),
                                  failure_phase_caveat="Heuristic telemetry phase, not a causal diagnosis",
                                  all_work_wall_s=sum(r["all_work_wall_s"] for r in results)))
    recorded = sum(row["episodes"] for row in report["e3"])
    report["status"] = dict(recorded_e3_episodes=recorded,expected_e3_episodes=112,
                            partial=recorded!=112 or any(row["episodes"]!=16 for row in report["e3"]) or len(report["e3"])!=7,
                            interpretation="Recorded completed-episode outcomes; final_audit is a separate raw-record gate")
    return report


def trajectories(root, destination, records=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, destination = Path(root), Path(destination)
    snapshot = {(root/row["path"]).parent/"telemetry.jsonl" for row in records} if records is not None else None
    outputs = []
    for baseline in root.rglob("fresh_serial_r1/*/task00/init00/telemetry.jsonl"):
        if not (baseline.parent/"complete.json").exists():
            continue
        if snapshot is not None and baseline not in snapshot:
            continue
        suite = baseline.parents[2].name
        container = baseline.parents[4]
        a = np.array([json.loads(line)["eef_pos"] for line in baseline.read_text().splitlines()])
        for mode in ("serial_reuse", "async_reuse"):
            for interval in (1, 2, 4):
                variant = container/f"{mode}_r{interval}"/suite/"task00/init00/telemetry.jsonl"
                if not variant.exists() or not (variant.parent/"complete.json").exists():
                    continue
                if snapshot is not None and variant not in snapshot:
                    continue
                b = np.array([json.loads(line)["eef_pos"] for line in variant.read_text().splitlines()])
                fig = plt.figure(figsize=(6, 5))
                axis = fig.add_subplot(projection="3d")
                axis.plot(*a.T, color="tab:blue", alpha=.65, marker="o", markevery=max(1,len(a)//20), markersize=3, label="fresh_serial")
                axis.plot(*b.T, color="tab:orange", alpha=.65, marker="^", markevery=max(1,len(b)//20), markersize=3, label=f"{mode} r={interval}")
                axis.set(xlabel="EEF X (m)", ylabel="EEF Y (m)", zlabel="EEF Z (m)", title=f"{suite}, task0 / init0")
                axis.legend()
                path = destination/f"eef_{suite}_{mode}_r{interval}.png"
                fig.savefig(path, dpi=170, bbox_inches="tight", pad_inches=.35)
                plt.close(fig)
                outputs.append(str(path))
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    report = analyze(args.root)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    (destination/"summary.json").write_text(json.dumps(report, indent=2))
    (destination/"video_index.json").write_text(json.dumps(report["videos"], indent=2))
    if args.plots:
        from .summary_plots import plot_summaries,plot_sources
        plots = trajectories(args.root, destination,report["videos"])+plot_summaries(report,destination)
        (destination/"plot_index.json").write_text(json.dumps(plots, indent=2))
        (destination/"plot_sources.json").write_text(json.dumps(plot_sources(report,args.root,plots),indent=2))
    for name in ("e0", "e2", "e2_age", "e3", "simulation_ages"):
        if not report[name]:
            continue
        with (destination/f"{name}.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report[name][0]))
            writer.writeheader()
            writer.writerows(report[name])
    print(json.dumps(dict(counts={key:len(value) for key,value in report.items() if isinstance(value,list)},status=report["status"])))


if __name__ == "__main__":
    main()
