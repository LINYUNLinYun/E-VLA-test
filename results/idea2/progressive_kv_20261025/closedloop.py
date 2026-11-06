"""Finite real LIBERO mixed-KV evaluation, wrapping the existing guarded runner.

Default plan: four suites x task 0/5 x init 0 x five conditions = 40 episodes.
No GPU/model/environment execution occurs without --execute.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
LEGACY_PATH=ROOT/"results/idea2/d10_dynamic_sm_20261024/closedloop_smoke.py"
spec=importlib.util.spec_from_file_location("_progressive_closedloop_legacy",LEGACY_PATH)
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
LEADS={"lead0":0,"lead1":1,"lead3":3,"lead6":6,"fresh18":18}
original_plan=legacy.make_plan
original_hashes=legacy.source_hashes
original_execute=legacy.execute
original_worker=legacy.run_worker
original_rpc=legacy.rpc
legacy.__file__=str(Path(__file__).resolve())


def source_hashes():
    result=original_hashes()
    result[str(LEGACY_PATH.relative_to(ROOT))]=hashlib.sha256(LEGACY_PATH.read_bytes()).hexdigest()
    return result


def expected_protocol(mode=None):
    result=dict(H=10,R=5,num_steps=10,cache_slots=2,execution="progressive_cuda_graph",interval=1,
                next_encoder=False,lookahead=False,host_workers=0,diagnostics=True,
                submission_order="prefill_then_ae_kv_ready",preparation_core="PreparedModulationCore")
    if mode is not None:
        result.update(mode=mode,lead=LEADS[mode],approximation=mode!="fresh18")
    return result


def check_protocol(protocol,mode=None):
    for key,value in expected_protocol(mode).items():
        if protocol.get(key)!=value:
            raise RuntimeError(f"Progressive protocol mismatch {key}: {protocol.get(key)!r} != {value!r}")
    if "pi05=True" not in protocol.get("config",""):
        raise RuntimeError("Expected original pi0.5 configuration")


def make_plan(args):
    if len(set(args.modes))!=len(args.modes) or len(set(args.cases))!=len(args.cases):
        raise ValueError("Duplicate conditions or case identities are not allowed")
    legacy.MODES={name:name for name in args.modes}
    def cases(reverse=False):
        rows=[]
        selected=list(args.modes)
        if reverse:
            selected.reverse()
        for index,(suite,task) in enumerate((s,t) for s in legacy.SUITES for t in (0,5)):
            if index not in args.cases:
                continue
            offset=index%len(selected)
            for arm,label in enumerate(selected[offset:]+selected[:offset]):
                rows.append(dict(case=index,arm=arm,label=label,mode=label,lead=LEADS[label],suite=suite,task=task,init=0,
                                 relative_directory=f"episodes/{label}/{suite}/task{task:02d}/init00"))
        return rows
    legacy.episode_cases=cases
    plan=original_plan(args)
    affinity=parse_cpus(args.cpus) if args.cpus else None
    if affinity is not None and not affinity<=os.sched_getaffinity(0):
        raise ValueError("Requested CPU mask is outside current sched_getaffinity")
    plan["cpu_affinity"]=sorted(affinity) if affinity else None
    if affinity:
        value=",".join(map(str,sorted(affinity)))
        for role in ("server","client"):
            plan[role]["environment"]["PROGRESSIVE_EXPECTED_CPUS"]=value
    plan["server"]["argv"]=[args.server_python,"-u","-m","preparations.idea2.progressive_kv_policy",
        "--checkpoint",plan["checkpoint"],"--port",str(args.port),"--mode",args.modes[0]]
    plan["protocol"]=expected_protocol()
    plan["selected_modes"]=list(args.modes)
    plan["selected_case_indices"]=list(args.cases)
    plan["full_matrix"]=dict(modes=list(LEADS),case_indices=list(range(8)),episodes=40)
    plan["schedule"]="Eight fixed suite/task/init identities; rotate whole-episode condition order by case; one persistent model and one environment client"
    plan["semantics"]="lead0/1/3/6 are approximate per-Euler mixed old/new layer KV algorithms; fresh18 uses fully current KV. Every episode starts with one current-frame full Prefill and fresh original AE. No future observation or next encoder is invented. Free closed-loop trajectories may diverge; exact equality versus fresh is not required."
    plan["next_encoder"]=False
    plan["bootstrap"]="Lazy first predict: retain captured graphs across resets/tasks, bootstrap current Prefix once and use original AE on it; bootstrap/capture latency reported separately"
    plan["measurements"]="Original run_episode seeds, env init, settling, H10/R5, actions, inputs/hashes and EEF telemetry; additionally retain normalized actions, per-step KV frontiers, Pool/generation metadata and request latency. Rendering/client/I/O costs remain in closed-loop wall time."
    plan["limitations"]=["Eight fixed cases per selected mode are an exploratory quality test, not official full LIBERO evaluation.",
        "This run validates mixed-KV quality without next-encoder lookahead; it cannot establish the throughput of a system with available next observations.",
        "CUDA-visible client devices are empty; EGL intentionally uses the explicitly assigned GPU and is checked by inherited C/G namespace-aware guards.",
        "First-request capture and fresh bootstrap are cold costs, excluded from steady latency summaries.",
        "Executed sources must remain frozen from plan creation to server shutdown."]
    return plan


def rpc(port,message,timeout=600):
    if message.get("op")=="configure":
        # The legacy executor supplies old-Pipeline knobs. This adapter only
        # maps the selected condition and never pretends those knobs apply.
        message=dict(op="configure",mode=message["mode"],interval=1)
    return original_rpc(port,message,timeout)


def validate_artifacts(directory,client,case):
    import numpy as np
    result=json.loads((directory/"result.json").read_text())
    protocol=json.loads((directory/"native_protocol.json").read_text())
    check_protocol(protocol,case["mode"])
    infers=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
    telemetry=[json.loads(line) for line in (directory/"telemetry.jsonl").read_text().splitlines()]
    if len(infers)!=result["predictions"] or len(telemetry)!=result["actions"]+1:
        raise RuntimeError("Incomplete original episode telemetry")
    for index,row in enumerate(infers):
        if row["seed"]!=client.model_seed(case["suite"],case["task"],case["init"],index):
            raise RuntimeError("Noise seed protocol changed")
        if row["bootstrap"]!=(index==0) or row["request"]!=index:
            raise RuntimeError("Episode bootstrap/request identity mismatch")
        if row["lead"]!=LEADS[case["mode"]] or row["next_encoder"] or row["lookahead"]:
            raise RuntimeError("Selected condition or no-lookahead contract changed")
        frontiers=row["frontiers"]
        if len(frontiers)!=10 or any(type(f) is not int or not 0<=f<=18 for f in frontiers) or frontiers!=sorted(frontiers):
            raise RuntimeError("Invalid per-iteration frontier history")
        if (index==0 or case["mode"]=="fresh18") and frontiers!=[18]*10:
            raise RuntimeError("Fresh bootstrap/control consumed stale layers")
        if min(frontiers)<LEADS[case["mode"]]:
            raise RuntimeError("AE ran before its requested KV-ready lead")
        if row["old_prefix_observation_id"]!=(0 if index==0 else index-1) or row["new_prefix_observation_id"]!=index:
            raise RuntimeError("Cross-observation identity mismatch")
        if row["graph_kv_buffers"]!=2 or row["pool"]["live_snapshots"]>2:
            raise RuntimeError("Two-slot bound violated")
        if index and row["graph_setup_ms"]:
            raise RuntimeError("Unexpected within-episode graph recapture")
        for name,shape in (("chunk_actions",(10,7)),("normalized_actions",(10,32))):
            actions=np.asarray(row[name])
            if actions.shape!=shape or not np.isfinite(actions).all():
                raise RuntimeError("Invalid "+name)
    if any("eef_pos" not in row or "eef_quat" not in row for row in telemetry):
        raise RuntimeError("Missing measured EEF pose")
    def stats(values):
        return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))) if values else None
    steady=[r for r in infers if not r["bootstrap"] and r["graph_setup_ms"]==0]
    return dict(valid=True,success=result["success"],steps=result["actions"],predictions=len(infers),
        lead=LEADS[case["mode"]],approximation=case["mode"]!="fresh18",next_encoder=False,
        cold_capture_requests=sum(r["graph_setup_ms"]>0 for r in infers),
        bootstrap_requests=sum(r["bootstrap"] for r in infers),first_input_hash=infers[0]["input_hash"] if infers else None,
        all_request_roundtrip_ms=stats([r["client_roundtrip_ms"] for r in infers]),
        bootstrap_roundtrip_ms=stats([r["client_roundtrip_ms"] for r in infers if r["bootstrap"]]),
        steady_progressive_roundtrip_ms=stats([r["client_roundtrip_ms"] for r in steady]),
        steady_progressive_action_ready_ms=stats([r["timing"]["action_ready_host_ms"] for r in steady]),
        frontier_histogram={str(f):sum(r["frontiers"].count(f) for r in steady) for f in range(19)},
        latency_scope="Actual closed-loop RPC/observation/renderer path; no isolated GPU performance claim",
        runtime_result=result,noise_identity="Original model_seed and CUDA FP32 noise generator, seeds recorded")


def run_worker(path):
    sys.path.insert(0,str(ROOT))
    from preparations.idea1_v2_1 import runtime_client as client
    from unittest.mock import patch
    expected=os.environ.get("PROGRESSIVE_EXPECTED_CPUS")
    if expected and set(map(int,expected.split(",")))!=os.sched_getaffinity(0):
        raise RuntimeError("Client did not inherit planned CPU affinity")
    original=client.run_episode
    def run(*args,**kwargs):
        kwargs["kind"]="idea2_pi05_d10_progressive_kv_smoke"
        return original(*args,**kwargs)
    with patch.object(client,"run_episode",run):
        return original_worker(path)


def execute(args,plan,dest):
    previous=os.sched_getaffinity(0)
    try:
        if plan.get("cpu_affinity"):
            os.sched_setaffinity(0,set(plan["cpu_affinity"]))
        original_execute(args,plan,dest)
    finally:
        os.sched_setaffinity(0,previous)
    state=json.loads((dest/"execution.json").read_text())
    state["summary"]={label:dict(episodes=sum(r["label"]==label for r in state["cases"]),
        successes=sum(r["result"]["success"] for r in state["cases"] if r["label"]==label),
        executed_steps=sum(r["result"]["steps"] for r in state["cases"] if r["label"]==label)) for label in args.modes}
    state["semantics"]=plan["semantics"]
    state["next_encoder"]=False
    legacy.write_json(dest/"execution.json",state)


original_parser=legacy.parser
def parse_cpus(value):
    cpus=set()
    for part in value.split(","):
        if "-" in part:
            low,high=map(int,part.split("-"))
            if low<0 or high<low or high>4095:
                raise ValueError("Invalid CPU range")
            cpus.update(range(low,high+1))
        else:
            cpus.add(int(part))
    if not cpus or min(cpus)<0:
        raise ValueError("Empty/negative CPU affinity")
    return cpus


def parser():
    result=original_parser()
    result.description=__doc__
    result.set_defaults(output=str(HERE/"closedloop_all"),port=19085,egl_device=2)
    result.add_argument("--modes",nargs="+",choices=list(LEADS),default=list(LEADS))
    result.add_argument("--cases",type=int,nargs="+",choices=list(range(8)),default=list(range(8)))
    result.add_argument("--cpus",help="Optional inherited CPU mask, e.g. 32-47; validated against sched_getaffinity")
    return result


legacy.source_hashes=source_hashes
legacy.expected_protocol=expected_protocol
legacy.check_protocol=check_protocol
legacy.make_plan=make_plan
legacy.rpc=rpc
legacy.validate_artifacts=validate_artifacts
legacy.run_worker=run_worker
legacy.execute=execute
legacy.parser=parser


if __name__=="__main__":
    legacy.main()
