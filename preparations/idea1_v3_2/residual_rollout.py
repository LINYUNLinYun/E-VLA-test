"""Use original native LIBERO rollout/telemetry without modifying that code."""
import argparse
import json
from pathlib import Path
import time
import traceback

from preparations.idea1_v3_1.common import SUITES,atomic_json
from preparations.idea1_v3_1.rollout import Environments,run_episode
from .residual_server import gate_receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate",required=True)
    parser.add_argument("--mode",choices=("native","reuse03","d2"),required=True)
    parser.add_argument("--port",required=True,type=int)
    parser.add_argument("--output",required=True)
    parser.add_argument("--tasks",type=int,nargs="+",default=list(range(10)))
    parser.add_argument("--deadline-unix",required=True,type=float)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,episodes=4*len(args.tasks),**vars(args))))
        return
    gate,identity = gate_receipt(args.gate)
    condition = dict(id=args.mode,model="pi0",horizon=50,execute_length=5,
                     experiment="v3.2 E2",skip=[],schedule="uniform D10" if args.mode!="d2" else "uniform D2")
    pool = Environments(reuse=False)
    report = dict(started_unix=time.time(),deadline_unix=args.deadline_unix,mode=args.mode,
        gate=args.gate,gate_hash=identity,episodes=[],failures=[],
        approximation="Transformer pre-final-norm residual reuse, no cached velocity",
        telemetry="original native per-step measured EEF XYZ, executed command, contact/grasp truth and heuristic stage; XYZ measured separately from commands")
    receipt = Path(args.output)/args.mode/"rollout_receipt.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    atomic_json(receipt,report)
    try:
        for task in args.tasks:
            for suite in SUITES:
                if time.time()>=args.deadline_unix:
                    report["failure_code"] = "DEADLINE"
                    return
                directory = Path(args.output)/args.mode/suite/f"task{task:02d}"/"init00"
                try:
                    result = run_episode(directory,args.port,condition,suite,task,0,pool,identity,
                                         video=task==0,save_observations=False)
                    report["episodes"].append(result)
                    print(json.dumps(result),flush=True)
                except Exception:
                    report["failures"].append(dict(suite=suite,task=task,failure_code="TECHNICAL",error=traceback.format_exc()))
                    print(json.dumps(report["failures"][-1]),flush=True)
                atomic_json(receipt,report)
    finally:
        pool.close()
        report["completed_unix"] = time.time()
        report["successes"] = sum(r["success"] for r in report["episodes"])
        atomic_json(receipt,report)


if __name__ == "__main__":
    main()
