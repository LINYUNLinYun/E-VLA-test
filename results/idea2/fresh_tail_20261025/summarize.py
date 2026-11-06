"""CPU-only summaries; separate paired/continuous and ignore non-passed cases."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
P = R.parents[2] / "reports/idea2_fresh_tail_20261025"
METRICS = ["cpu_cpu_ms", "vision_ms", "vlm_ms", "ae_ms", "fresh_tail_wait_ms",
           "exposed_ae_after_vlm_ms", "ae_finish_minus_next_encoder_ms", "ae_submit_ms",
           "lookahead_prepare_ms", "decode_host_ms"]


def write(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (P / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def main():
    P.mkdir(parents=True, exist_ok=True)
    values, evidence, requests, frontiers = [], [], [], []
    for path in sorted(R.glob("formal_*/result.json")):
        data = json.loads(path.read_text())
        if data["status"] != "passed":
            continue
        adapter = data["fresh_tail_adapter"]
        if not adapter["cleanup"]["passed"] or not data["source_unchanged"]:
            raise ValueError(f"Unsafe formal artifact {path}")
        name = path.parent.name
        verification = [x for x in data["rows"] if x["phase"] == "verification"]
        oracle = [x["numerical"]["matched_schedule"] for x in verification]
        evidence.append(dict(case=name, result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             oracle_count=len(oracle), oracle_max=max(x["max_abs"] for x in oracle),
                             oracle_mae=max(x["mae"] for x in oracle), source_unchanged=data["source_unchanged"],
                             cleanup_passed=adapter["cleanup"]["passed"]))
        for phase in ["paired", "continuous"]:
            for lead in [3, 18]:
                rows = [x for x in data["rows"] if x["phase"] == phase and x["lead"] == lead]
                if not rows: continue
                tail, sms = adapter["fresh_tail_steps"], adapter["ae_sm_min"]
                if any(tail and x["frontiers"][-tail:] != [18]*tail for x in rows):
                    raise ValueError(f"Tail freshness failure: {name}")
                groups = [g for g in data["groups"] if g["phase"] == phase and g["lead"] == lead]
                out = dict(case=name, ae_sms=sms, tail=tail, lead=lead, variant="encoder", phase=phase,
                           gpu=data["args"]["gpu_uuid"], requests=len(rows), groups=len(groups),
                           trajectory_ms_mean=float(np.mean([g["all_work_ms"] for g in groups])),
                           all_work_hz=float(sum(g["count"] for g in groups)*1000/sum(g["all_work_ms"] for g in groups)),
                           peak_allocated_bytes=data["peak_allocated_bytes"], peak_reserved_bytes=data["peak_reserved_bytes"],
                           graph_recaptures=max(x["graph_recaptures"] for x in rows))
                for metric in METRICS:
                    samples = [x[metric] for x in rows if x.get(metric) is not None]
                    for stat in ["mean", "p50", "p95"]:
                        out[metric+"_"+stat] = float(np.mean(samples) if stat == "mean" else np.percentile(samples, int(stat[1:]))) if samples else None
                out["mean_frontiers_by_step"] = json.dumps(np.mean([x["frontiers"] for x in rows], axis=0).tolist())
                values.append(out)
                for x in rows:
                    requests.append(dict(case=name, ae_sms=sms, tail=tail, phase=phase, lead=lead,
                                         **{k:x.get(k) for k in ["pair", "index", "source_id", "noise_seed", "input_sha256", "slot", "generation", "epoch", "fresh_tail_wait_ms"]}))
                for step in range(10):
                    samples = [x["iteration_ms"][step] for x in rows]
                    frontiers.append(dict(case=name, ae_sms=sms, tail=tail, phase=phase, lead=lead, step=step+1,
                                          frontier_mean=float(np.mean([x["frontiers"][step] for x in rows])),
                                          iteration_including_wait_ms_mean=float(np.mean(samples))))
    write("performance.csv", values); write("requests_identity.csv", requests); write("iterations.csv", frontiers)
    write("implementation_oracle.csv", evidence)
    comparisons=[]
    for candidate in values:
        if candidate["lead"]!=3 or candidate["tail"]==0: continue
        base_case=candidate["case"].replace(f"_tail{candidate['tail']}", "_tail0").replace("_retry", "")
        bases=[x for x in values if x["case"]==base_case and x["lead"]==3 and x["phase"]==candidate["phase"]]
        if not bases:continue
        base=bases[0]
        comparisons.append(dict(case=candidate["case"], baseline_case=base_case, phase=candidate["phase"], ae_sms=candidate["ae_sms"], tail=candidate["tail"],
                                latency_ratio=candidate["trajectory_ms_mean"]/base["trajectory_ms_mean"],
                                throughput_ratio=candidate["all_work_hz"]/base["all_work_hz"],
                                scope="same-card independent-process blocks, reversed tail order on repeat"))
    write("tail_comparisons.csv", comparisons)
    (P/"summary.json").write_text(json.dumps(dict(formal_arms=len(evidence), summaries=len(values), implementation_evidence=evidence), indent=2)+"\n")
    print(f"Summarized {len(evidence)} passed formal processes")


if __name__ == "__main__": main()
