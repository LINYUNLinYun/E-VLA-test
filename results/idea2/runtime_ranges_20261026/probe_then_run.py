"""Verify range projection with a tiny graph before capturing real policies."""
import collections
import csv
import json
import os
import subprocess
import sys
import time

from . import run
from preparations.idea2.layer_bench import require_idle_gpu, gpu_snapshot


def main():
    r=run.R
    if (r/"probe.json").exists():raise RuntimeError("Do not rerun this driver")
    os.sched_setaffinity(0,set(range(96,112)))
    run.lease(True,require_idle_gpu(run.GPU))
    status=dict(status="starting",gpu=3,scope="Projection smoke before model captures")
    def save():run.save(r/"probe.json",status)
    try:
        prior=json.loads((run.ROOT/"results/idea2/closedloop200_20261025/gpu2_chunk0/plan.json").read_text())
        env=dict(os.environ,**prior["server"]["environment"])
        env["CUDA_VISIBLE_DEVICES"]=run.GPU
        command=["nsys","profile","--trace=cuda,nvtx","--cuda-graph-trace=node:host-only:nvtx-live",
            "--sample=none","--cpuctxsw=none","--nvtx-domain-exclude=TSL,CCCL",
            "--capture-range=cudaProfilerApi","--capture-range-end=stop","--kill=none",
            "--output="+str(r/"micro_projection"),prior["server"]["python"],"-u","-m",
            "results.idea2.runtime_ranges_20261026.micro",str(r/"micro_result.json")]
        status["command"]=command;save()
        with (r/"micro.log").open("x") as log:
            subprocess.run(command,cwd=run.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        command=["nsys","stats","--report","nvtx_gpu_proj_trace","--format","csv",
                 "--output",str(r/"micro_projection_stats"),str(r/"micro_projection.nsys-rep")]
        with (r/"projection_stats.log").open("x") as log:
            subprocess.run(command,cwd=run.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
        files=list(r.glob('micro_projection_stats*.csv'))
        if len(files)!=1:raise RuntimeError("Projection CSV missing or ambiguous")
        rows=list(csv.DictReader(files[0].open()))
        status["projected_ranges"]=rows
        names=[row["Name"].rsplit(":",1)[-1] for row in rows]
        if collections.Counter(names)!=dict(encoder=3,vlm=3,AE=3):
            raise RuntimeError("Expected one projected GPU range per runtime stage; do not collect model on failure")
        status.update(status="passed",projected_range_count=len(rows),
            verification="Official nvtx_gpu_proj_trace: exactly3 full ranges per phase, individual graph nodes retained")
        save()
        time.sleep(2)
        sys.argv=[sys.argv[0],"--pipeline-alignment","off","--preclaimed"]
        run.main()
    except BaseException:
        status.update(status="failed");save()
        raise
    finally:
        current=json.loads(run.LEASES.read_text()).get('3',{})
        if current.get('owner')==run.OWNER and current.get('active'):
            run.lease(False,gpu_snapshot(run.GPU))


if __name__=="__main__":main()
