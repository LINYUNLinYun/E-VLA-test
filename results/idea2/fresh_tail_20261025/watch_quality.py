"""CPU-only report watcher; never starts a model, simulator or GPU command."""
import argparse
import os
from pathlib import Path
import time
import traceback

from results.idea2.fresh_tail_20261025.analyze_quality import analyze
from results.idea2.fresh_tail_20261025.common import HERE,ROOT,save,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",default=str(HERE/"closedloop_gpu2"))
    parser.add_argument("--output",default=str(ROOT/"reports/idea2_fresh_tail_20261025"))
    parser.add_argument("--timeout",type=float,default=10800)
    args=parser.parse_args();output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    started=time.time();receipt=dict(status="running",pid=os.getpid(),start_unix=started,run=args.run)
    save(output/"watch_receipt.json",receipt)
    try:
        while time.time()-started<args.timeout:
            result=analyze(args.run);save(output/"closedloop_summary.json",result)
            for key in ("configurations","cases","frontiers","bootstrap_comparisons","approximation_delta"):
                write_csv(output/(key+".csv"),result[key])
            valid=sum(r["status"]=="valid" for r in result["cases"])
            print(time.strftime("%Y-%m-%d %H:%M:%S"),result["status"],"valid",valid,"bootstrap",result["bootstrap_gate_passed"],flush=True)
            receipt.update(last_update_unix=time.time(),valid_cases=valid,summary_status=result["status"])
            save(output/"watch_receipt.json",receipt)
            if result["status"]!="in_progress":
                receipt.update(status=result["status"],end_unix=time.time(),exit_code=0 if result["status"]=="complete" else 2)
                save(output/"watch_receipt.json",receipt)
                return receipt["exit_code"]
            time.sleep(30)
        raise TimeoutError("Quality report watcher deadline; artifacts remain explicitly incomplete")
    except BaseException:
        receipt.update(status="watcher_failed",traceback=traceback.format_exc(),end_unix=time.time(),exit_code=1)
        save(output/"watch_receipt.json",receipt)
        raise


if __name__=="__main__":raise SystemExit(main())
