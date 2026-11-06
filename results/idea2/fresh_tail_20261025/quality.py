"""Finite fresh-tail LIBERO study; default is a CPU-only 56-episode plan.

Six SM/tail conditions plus same-GPU unpartitioned fresh18, with the frozen
original run_episode, D10/H10/R5, seeds, observations and environment guards.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from results.idea2.fresh_tail_20261025.common import CONFIGS,sha

BASE=ROOT/"results/idea2/progressive_kv_20261025/closedloop.py"
spec=importlib.util.spec_from_file_location("_fresh_tail_closedloop_base",BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.legacy.__file__=str(Path(__file__).resolve())
CONDITIONS={**CONFIGS,"fresh18":dict(ae_sm_min=0,fresh_tail_steps=0)}
current=dict(ae_sm_min=16,fresh_tail_steps=0)
remaining=[]


def source_hashes():
    result=base.source_hashes()
    for path in (BASE,HERE/"adapter.py",HERE/"common.py",Path(__file__).resolve(),
                 ROOT/"results/idea2/progressive_kv_20261025/resource_adapter.py"):
        result[str(path.relative_to(ROOT))]=sha(path)
    return result


def check_resources(protocol,require_actual=False):
    resource=protocol.get("resource_adapter",{})
    requested=resource.get("requested",resource)
    if requested.get("ae_sm_min")!=current["ae_sm_min"] or requested.get("encoder_after_step")!=1:
        raise RuntimeError("Resource configuration mismatch")
    if protocol.get("fresh_tail_steps")!=current["fresh_tail_steps"]:
        raise RuntimeError("Fresh-tail configuration mismatch")
    layouts=resource.get("actual_resources")
    if not isinstance(layouts,list):raise RuntimeError("Missing actual resource list")
    if current["ae_sm_min"]:
        if require_actual and not layouts:raise RuntimeError("Actual resources absent after prediction")
        expected=[current["ae_sm_min"],128-current["ae_sm_min"]]
        if any(layout.get("sm_counts")!=expected for layout in layouts):
            raise RuntimeError("Actual SM allocation mismatch")
    elif layouts:
        raise RuntimeError("Unpartitioned fresh18 unexpectedly has a resource partition")
    return layouts


def check_protocol(protocol,mode=None):
    base.check_protocol(protocol,mode)
    check_resources(protocol)


def make_plan(args):
    global current
    if len(set(args.configs))!=len(args.configs) or len(set(args.cases))!=len(args.cases):
        raise ValueError("Duplicate condition/case")
    base.legacy.MODES={label:("fresh18" if label=="fresh18" else "lead3") for label in args.configs}
    def cases(reverse=False):
        labels=list(reversed(args.configs)) if reverse else list(args.configs)
        rows=[]
        for arm,label in enumerate(labels):
            for index,(suite,task) in enumerate((s,t) for s in base.legacy.SUITES for t in (0,5)):
                if index not in args.cases:continue
                mode="fresh18" if label=="fresh18" else "lead3"
                rows.append(dict(case=index,arm=arm,label=label,mode=mode,lead=base.LEADS[mode],
                    suite=suite,task=task,init=0,**CONDITIONS[label],
                    relative_directory=f"episodes/{label}/{suite}/task{task:02d}/init00"))
        return rows
    base.legacy.episode_cases=cases
    plan=base.original_plan(args)
    first=plan["cases"][0];current=dict(CONDITIONS[first["label"]])
    affinity=base.parse_cpus(args.cpus) if args.cpus else None
    if affinity and not affinity<=os.sched_getaffinity(0):raise ValueError("Invalid CPU affinity")
    plan["cpu_affinity"]=sorted(affinity) if affinity else None
    if affinity:
        for role in ("server","client"):
            plan[role]["environment"]["PROGRESSIVE_EXPECTED_CPUS"]=",".join(map(str,sorted(affinity)))
    plan["server"]["argv"]=[args.server_python,"-u","-m","results.idea2.fresh_tail_20261025.adapter",
        "--serve","--ae-sm-min",str(current["ae_sm_min"]),"--fresh-tail-steps",str(current["fresh_tail_steps"]),
        "--encoder-after-step","1","--checkpoint",plan["checkpoint"],"--port",str(args.port),"--mode",first["mode"]]
    plan.update(source_sha256=source_hashes(),protocol=base.expected_protocol(),selected_configs=args.configs,
        selected_case_indices=args.cases,next_encoder=False,
        full_matrix=dict(configs=list(CONDITIONS),case_indices=list(range(8)),episodes=56),
        schedule="Seven condition groups, each containing all eight fixed cases with original seeds/resets; persistent model, at most seven captures. Quality-only grouping avoids per-case configuration recapture; no paired timing claim.",
        semantics="SM16/32 x final 0/1/2 fresh-KV Euler rounds at lead3 are approximate algorithms; same-card unpartitioned fresh18 is the current-KV reference. Last-layer KV-ready is sufficient for the forced tail; Pool producer completion still covers VLM FFN. next_encoder=False.",
        limitations=["Eight fixed cases per configuration are exploratory quality evidence, not full LIBERO evaluation.",
                     "Free trajectories can diverge; action differences are valid only on matched inputs/noise.",
                     "Closed-loop RPC latency includes rendering/client/I/O conditions and is not isolated GPU latency.",
                     "First-request graph capture/bootstrap is a cold cost. No future observation is supplied.",
                     "Sources remain frozen through server shutdown. Context cleanup runs after service returns."])
    return plan


def rpc(port,message,timeout=600):
    global current
    if message.get("op")=="configure":
        if not remaining:raise RuntimeError("Configure outside planned case sequence")
        case=remaining.pop(0);current=dict(CONDITIONS[case["label"]])
        if case["mode"]!=message["mode"]:raise RuntimeError("Configure order mismatch")
        message=dict(op="configure",mode=case["mode"],interval=1,**current)
    response=base.original_rpc(port,message,timeout)
    if "protocol" in response:
        check_protocol(response["protocol"])
        check_resources(response["protocol"],message.get("op")=="predict")
    return response


def validate_artifacts(directory,client,case):
    result=base.validate_artifacts(directory,client,case)
    rows=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
    for row in rows:
        check_protocol(row["protocol"],case["mode"])
        check_resources(row["protocol"],True)
        k=case["fresh_tail_steps"]
        if k and row["frontiers"][-k:]!=[18]*k:raise RuntimeError("Fresh-tail frontier incomplete")
    result.update(condition=case["label"],ae_sm_min=case["ae_sm_min"],fresh_tail_steps=case["fresh_tail_steps"],
                  resource_protocols_checked=len(rows),all_fresh_tail_frontiers_verified=True)
    return result


def run_worker(path):
    global current
    from preparations.idea1_v2_1 import runtime_client as client
    from unittest.mock import patch
    case=json.loads(Path(path).read_text());current=dict(CONDITIONS[case["label"]])
    original=client.rpc
    def checked_rpc(port,message,*args,**kwargs):
        response=original(port,message,*args,**kwargs)
        if "protocol" in response:
            check_protocol(response["protocol"],case["mode"])
            check_resources(response["protocol"],message.get("op")=="predict")
        return response
    with patch.object(client,"rpc",checked_rpc):return base.run_worker(path)


def execute(args,plan,dest):
    remaining[:]=plan["cases"]
    args.modes=list(args.configs)
    base.execute(args,plan,dest)
    if remaining:raise RuntimeError("Planned cases were not all configured")


def parser():
    result=base.original_parser();result.description=__doc__
    result.add_argument("--configs",nargs="+",choices=list(CONDITIONS),default=list(CONDITIONS))
    result.add_argument("--cases",nargs="+",type=int,choices=list(range(8)),default=list(range(8)))
    result.add_argument("--cpus",default="96-111")
    result.set_defaults(output=str(HERE/"closedloop_gpu3"),port=19088,egl_device=0,
        gpu_uuid="GPU-555cde36-5e41-bc44-0c69-ea47358fd703")
    return result


base.legacy.source_hashes=source_hashes
base.legacy.check_protocol=check_protocol
base.legacy.make_plan=make_plan
base.legacy.rpc=rpc
base.legacy.validate_artifacts=validate_artifacts
base.legacy.run_worker=run_worker
base.legacy.execute=execute
base.legacy.parser=parser

if __name__=="__main__":base.legacy.main()
