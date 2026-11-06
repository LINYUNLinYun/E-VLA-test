"""Analyze runtime graph node kernels through their replay API correlations."""
from pathlib import Path
import json
import sqlite3
import statistics

from preparations.idea2.scheduling_trace import analyze


root=Path(__file__).resolve().parent
workspace=root.parents[2]
modes=["original","threaded_prefill_first","graph_serial","graph_ae_first",
       "graph_prefill_first","graph_ae_first_high","graph_prefill_first_high"]
summary={}
for i,mode in enumerate(modes,1):
    path=workspace/f"nsys/idea2_experiment/runtime_graph_20261021.{i}.sqlite"
    result=analyze(path)
    db=sqlite3.connect(path)
    count=db.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    matched=db.execute("""SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
        AND k.globalPid=(r.globalTid & -16777216)""").fetchone()[0]
    assert count==matched,(count,matched)
    db.close()
    sections=[r for label,r in result["sections"].items() if int(label.rsplit("/",1)[1])>0]
    assert len(sections)==4
    total=sum(r["kernel_union_ms"] for r in sections)
    overlap=sum(r["overlap_ms"] for r in sections)
    summary[mode]=dict(kernel_count=count,matched_launches=matched,
        unassigned_input_kernels=result["unassigned_module_kernels"],
        samples=len(sections),mean_kernel_union_ms=total/len(sections),
        mean_overlap_ms=overlap/len(sections),overlap_fraction=overlap/total,
        modules={name:{field:statistics.mean(r["modules"][name][field] for r in sections)
                       for field in ("kernel_count","kernel_busy_ms","gpu_envelope_ms")}
                 for name in ("ae","prefill")},
        graph_launch_calls=sum(a["calls"] for a in result["cuda_api"] if "GraphLaunch" in a["name"]),
        capture_api_calls=sum(a["calls"] for a in result["cuda_api"] if "BeginCapture" in a["name"]),
        device_sync_calls=sum(a["calls"] for a in result["cuda_api"] if "DeviceSynchronize" in a["name"]))
    (root/f"trace_{mode}.json").write_text(json.dumps(result,indent=2))
(root/"trace_summary.json").write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
