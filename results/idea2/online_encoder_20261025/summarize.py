"""CPU-only pilot summary; CUDA Event spans are not kernel busy-time proofs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def stats(values):
    values = list(values)
    return dict(n=len(values), mean=float(np.mean(values)), p50=float(np.percentile(values,50)),
                p95=float(np.percentile(values,95))) if values else None


def main():
    p=argparse.ArgumentParser();p.add_argument("--inputs",nargs="+",required=True);p.add_argument("--output",required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cases=[];seen=set();rows=[]
    for source in args.inputs:
        run=json.loads((Path(source)/"execution.json").read_text())
        if run["status"]!="passed" or not run["source_unchanged"] or run["server_exit_code"]!=0:raise ValueError("Summary requires passed frozen runs")
        for case in run["cases"]:
            key=tuple(case[k] for k in ["arm","suite","task","init"])
            if key in seen:raise ValueError("Duplicate pilot case")
            seen.add(key);leaf=Path(case["directory"])
            value=json.loads((leaf/"complete.json").read_text())
            records=json.loads((leaf/"gpu_records.json").read_text())["records"]
            infers=[json.loads(x) for x in (leaf/"infer.jsonl").read_text().splitlines()]
            if len(infers)!=value["predictions"] or len(records)!=len(infers)-1:raise ValueError("Incomplete response/event accounting")
            cases.append(dict(arm=case["arm"],suite=case["suite"],task=case["task"],init=case["init"],
                              success=value["success"],steps=value["steps"],wall_s=value["wall_s"],
                              all_work_wall_s=value["all_work_wall_s"],predictions=value["predictions"],directory=str(leaf)))
            by_request={x["request_id"]:x for x in infers}
            for record in records:
                ae_start=record["ae_start_after_vlm_start_ms"];ae_end=ae_start+record["ae_ms"]
                online=record["online"];response=by_request[record["online_request_id"]]
                row=dict(arm=case["arm"],suite=case["suite"],task=case["task"],init=case["init"],
                         request=record["online_request_id"],next_encoder=record["next_encoder"],terminal=online["terminal"],
                         observation_wait_host_ms=online["observation_wait_host_ms"],
                         ae_done_when_server_received_next=online["ae_done_when_next_observation_received"],
                         ae_done_after_next_prepare=online.get("ae_done_after_next_preparation"),
                         next_prepare_host_ms=online.get("next_prepare_host_ms"),
                         ae_ms=record["ae_ms"],vlm_ms=record["vlm_ms"],vision_ms=record["vision_ms"],
                         response_roundtrip_ms=response["client_roundtrip_ms"],
                         frontiers=record["frontiers"],ae_next_encoder_event_envelope_intersection_ms=0.,
                         ae_finish_minus_next_encoder_ms=record.get("ae_finish_minus_next_encoder_ms"))
                if record["next_encoder"]:
                    vision_start=record["vlm_ms"]+record["next_encoder_start_after_vlm_ms"]
                    vision_end=vision_start+record["next_vision_ms"]
                    row["ae_next_encoder_event_envelope_intersection_ms"]=max(0.,min(ae_end,vision_end)-max(ae_start,vision_start))
                rows.append(row)
    summary=[]
    for arm in sorted({c["arm"] for c in cases}):
        group=[c for c in cases if c["arm"]==arm];events=[r for r in rows if r["arm"]==arm]
        steady=[r for r in events if not r["terminal"]];ahead=[r for r in steady if r["next_encoder"]]
        summary.append(dict(arm=arm,cases=len(group),successes=sum(c["success"] for c in group),steps=sum(c["steps"] for c in group),
             episode_wall_s=stats(c["wall_s"] for c in group),episode_work_s_sum=sum(c["all_work_wall_s"] for c in group),
             online_requests=len(events),nonterminal_requests=len(steady),next_encoder_requests=len(ahead),
             received_before_ae_completed=sum(not r["ae_done_when_server_received_next"] for r in steady),
             prepared_before_ae_completed=sum(r["ae_done_after_next_prepare"] is False for r in ahead),
             event_envelope_intersection_positive=sum(r["ae_next_encoder_event_envelope_intersection_ms"]>0.001 for r in ahead),
             event_envelope_intersection_ms=stats(r["ae_next_encoder_event_envelope_intersection_ms"] for r in ahead),
             ae_finish_minus_next_encoder_ms=stats(r["ae_finish_minus_next_encoder_ms"] for r in ahead),
             ae_ms=stats(r["ae_ms"] for r in steady),vlm_ms=stats(r["vlm_ms"] for r in steady),
             current_vision_ms=stats(r["vision_ms"] for r in steady),
             observation_wait_host_ms=stats(r["observation_wait_host_ms"] for r in steady),
             response_roundtrip_ms=stats(r["response_roundtrip_ms"] for r in steady)))
    result=dict(status="complete",cases=cases,arms=summary,requests=rows,
                scope="new causal one-chunk-lag pilot, not original synchronous protocol; no inference of noninferiority from8cases",
                timing="Event envelopes include waits/gaps. Positive envelope intersections require kernel traces for kernel overlap; disjoint envelopes preclude kernel overlap. RPC waits for actual R5 environment advance.")
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,group in [("cases.csv",cases),("requests.csv",rows),("arms.csv",summary)]:
        with (out/name).open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(group[0]));writer.writeheader();writer.writerows(group)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
