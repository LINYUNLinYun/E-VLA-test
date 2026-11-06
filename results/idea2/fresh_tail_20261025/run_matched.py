"""Run the formal fixed-input experiment after an independently passed gate.

This process owns only the GPU3 matched experiment; it never starts closed loop.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from results.idea2.fresh_tail_20261025.common import HERE,ROOT,save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    ready=HERE/"MATCHED_READY.json";receipt_path=output/"process_receipt.json"
    gpu="GPU-555cde36-5e41-bc44-0c69-ea47358fd703"
    receipt=dict(status="starting",supervisor_pid=os.getpid(),start_unix=time.time(),gpu_uuid=gpu)
    try:
        gate=json.loads((Path(args.gate)/"result.json").read_text())
        gate_process=json.loads((Path(args.gate)/"process_receipt.json").read_text())
        if gate["status"]!="complete" or not gate["cleanup"]["passed"] or gate_process["exit_code"]!=0:
            raise RuntimeError("12-input gate must finish with cleanup passed and process exit0")
        processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader"],text=True)
        if any(line.split(",")[0].strip()==gpu for line in processes.splitlines()):
            raise RuntimeError("GPU3 not idle immediately before formal launch")
        environment=dict(CUDA_VISIBLE_DEVICES=gpu,JAX_PLATFORMS="cpu",JAX_PLATFORM_NAME="cpu",XLA_PYTHON_CLIENT_PREALLOCATE="false",
            OMP_NUM_THREADS="2",MKL_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",CUDA_DEVICE_MAX_CONNECTIONS="32")
        command=[sys.executable,"-u","-m","results.idea2.fresh_tail_20261025.matched_quality","--execute","--count","100",
            "--output",str(output),"--gpu-uuid",gpu,"--cpus","96-111"]
        receipt.update(environment=environment,command=shlex.join(command),gate=str(Path(args.gate).resolve()),log=str(output/"run.log"))
        with (output/"run.log").open("w") as log:
            process=subprocess.Popen(command,env=dict(os.environ,**environment),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            receipt.update(status="running",pid=process.pid);save(receipt_path,receipt)
            code=process.wait()
        receipt.update(exit_code=code,end_unix=time.time())
        if code:raise RuntimeError(f"Formal driver exited {code}; preserve run.log")
        from results.idea2.fresh_tail_20261025.analyze_matched import analyze
        summary=analyze(output);save(output/"quality_summary.json",summary)
        from results.idea2.fresh_tail_20261025.common import write_csv
        write_csv(output/"quality_summary.csv",summary["variants"])
        write_csv(output/"tail_comparisons.csv",summary["tail_comparisons"])
        result=json.loads((output/"result.json").read_text())
        passed=summary["status"]=="complete" and result["cleanup"]["passed"] and len(summary["variants"])==6
        if not passed:raise RuntimeError("Formal quality/cleanup validation incomplete")
        receipt.update(status="passed",cleanup_passed=True);save(receipt_path,receipt)
        save(ready,dict(status="passed",result=str(output/"result.json"),receipt=str(receipt_path),exit_code=0,cleanup_passed=True))
    except BaseException:
        receipt.update(status="failed",traceback=traceback.format_exc(),end_unix=time.time());save(receipt_path,receipt)
        save(ready,dict(status="failed",result=str(output/"result.json"),receipt=str(receipt_path),exit_code=receipt.get("exit_code"),cleanup_passed=False))
        raise


if __name__=="__main__":main()
