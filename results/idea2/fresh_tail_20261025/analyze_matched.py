"""CPU-only aggregation of fixed-input approximation deltas and oracle gates."""
import argparse
import json
from pathlib import Path

import numpy as np

from results.idea2.fresh_tail_20261025.common import DIMENSIONS,save,write_csv


def analyze(directory):
    path=Path(directory)/"result.json"
    if not path.exists():return dict(status="in_progress",variants=[])
    source=json.loads(path.read_text());manifest=source["manifest"]
    count=manifest["args"]["count"];variants=[];checks=[]
    baseline=source["baseline"]
    for variant in source["variants"]:
        rows=variant["rows"]
        valid=len(rows)==count and variant.get("kv_python_references_after_close")==0
        valid=valid and variant["kv_slots"]==2 and variant["forced_reset_sample"] in variant["oracle_indices"]
        for index,row in enumerate(rows):
            valid=valid and row["sample"]==index
            valid=valid and all(row[key]==baseline[index][key] for key in ("input_hash","seed","noise_sha256"))
            k=variant["config"]["fresh_tail_steps"]
            valid=valid and (not k or row["frontiers"][-k:]==[18]*k)
        oracles=[r["matched_schedule_oracle"] for r in rows if r["matched_schedule_oracle"] is not None]
        valid=valid and len(oracles)>=8 and all(x["finite"] and x["max_abs"]<=1e-5 and x["mae"]<=1e-6 for x in oracles)
        summary=dict(configuration=variant["configuration"],**variant["config"],samples=len(rows),validation_passed=bool(valid),
            oracle_count=len(oracles),oracle_max_abs=max((r["max_abs"] for r in oracles),default=None),
            frontier_mean=np.mean([r["frontiers"] for r in rows],axis=0).tolist() if rows else [],
            fresh_tail_wait_ms_mean=float(np.mean([r["fresh_tail_wait_ms"] for r in rows])) if rows else None,
            freshness_full_round_fraction=float(np.mean(np.asarray([r["frontiers"] for r in rows])==18)) if rows else None)
        for horizon in ("R5","H10"):
            summary[horizon]=dict(mae=float(np.mean([r["approximation_delta"][horizon]["mae"] for r in rows])),
                max_abs=max(r["approximation_delta"][horizon]["max_abs"] for r in rows),
                per_dimension={d:dict(mae=float(np.mean([r["approximation_delta"][horizon]["per_dimension"][d]["mae"] for r in rows])),
                    max_abs=max(r["approximation_delta"][horizon]["per_dimension"][d]["max_abs"] for r in rows)) for d in DIMENSIONS})
        variants.append(summary);checks.append(valid)
    comparisons=[]
    by_name={v["configuration"]:v for v in source["variants"]}
    for sms in (16,32):
        original=by_name.get(f"sm{sms}_tail0")
        if original is None:continue
        for tail in (1,2):
            candidate=by_name.get(f"sm{sms}_tail{tail}")
            if candidate is None:continue
            for h in ("R5","H10"):
                left=np.array([r["approximation_delta"][h]["mae"] for r in original["rows"]])
                right=np.array([r["approximation_delta"][h]["mae"] for r in candidate["rows"]])
                if left.shape!=right.shape:continue
                comparisons.append(dict(ae_sm_min=sms,fresh_tail_steps=tail,horizon=h,samples=len(left),
                    tail0_mean_mae=float(left.mean()),tail_mean_mae=float(right.mean()),
                    paired_mae_change=float((right-left).mean()),improved_inputs=int(np.sum(right<left)),
                    unchanged_inputs=int(np.sum(right==left)),worsened_inputs=int(np.sum(right>left))))
    complete=source["status"]=="complete" and source.get("cleanup",{}).get("passed") and len(variants)==len(manifest["args"]["configs"]) and all(checks)
    return dict(status="complete" if complete else ("failed" if source["status"].startswith("failed") else "in_progress"),
        source=str(path.resolve()),gpu_uuid=source["gpu_uuid"],count=count,variants=variants,tail_comparisons=comparisons,
        interpretation="Deltas versus original full-current-KV D10, identical current/actual-previous observations and explicit noise. Approximation quality differs from the matched mixed-schedule implementation oracle. This is not a timing benchmark.")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("directory")
    args=parser.parse_args();result=analyze(args.directory)
    save(Path(args.directory)/"quality_summary.json",result)
    write_csv(Path(args.directory)/"quality_summary.csv",result["variants"])
    write_csv(Path(args.directory)/"tail_comparisons.csv",result.get("tail_comparisons",[]))
    print(json.dumps(dict(status=result["status"],variants=len(result["variants"]))))


if __name__=="__main__":main()
