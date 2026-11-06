"""Results-only SM32/96 lead3 quality check using the frozen closed-loop runner.

Default matrix is the original eight fixed cases, with no next-observation
lookahead. Actual resource allocation is required after every prediction.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE_PATH=HERE/"closedloop.py"
ADAPTER_PATH=HERE/"resource_adapter.py"
spec=importlib.util.spec_from_file_location("_resource_closedloop_base",BASE_PATH)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.legacy.__file__=str(Path(__file__).resolve())
original_check=base.check_protocol


def source_hashes():
    result=base.source_hashes()
    for path in (BASE_PATH,ADAPTER_PATH,Path(__file__).resolve()):
        result[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def check_resources(protocol,*,require_actual=False):
    resource=protocol.get("resource_adapter")
    if not isinstance(resource,dict):
        raise RuntimeError("Missing resource-adapter protocol")
    requested=resource.get("requested",resource)
    if requested.get("ae_sm_min")!=32 or requested.get("encoder_after_step")!=1:
        raise RuntimeError("SM32/96 experiment requires ae_sm_min=32 and encoder_after_step=1")
    actual=resource.get("actual_resources")
    if not isinstance(actual,list) or (require_actual and not actual):
        raise RuntimeError("Actual SM resources must be reported after prediction")
    for layout in actual:
        if layout.get("sm_counts")!=[32,96]:
            raise RuntimeError("Actual AE/Prefill resources differ from [32,96]")
        if layout.get("device_sm_count",128)!=128 or layout.get("requested_first_min",32)!=32:
            raise RuntimeError("Resource split metadata disagrees with SM32/96")
    return actual


def check_protocol(protocol,mode=None):
    original_check(protocol,mode)
    check_resources(protocol)


def make_plan(args):
    plan=base.make_plan(args)
    plan["source_sha256"]=source_hashes()
    plan["server"]["argv"]=[args.server_python,"-u","-m",
        "results.idea2.progressive_kv_20261025.resource_adapter","--serve","--ae-sm-min","32",
        "--encoder-after-step","1","--checkpoint",plan["checkpoint"],"--port",str(args.port),"--mode",args.modes[0]]
    plan["resource_adapter"]=dict(ae_sm_min=32,encoder_after_step=1,expected_actual_sm_counts=[32,96],
        initial_health_may_have_empty_actual_resources=True,actual_required="every prediction, including fresh bootstrap")
    plan["full_matrix"]=dict(modes=list(args.modes),case_indices=list(range(8)),episodes=8*len(args.modes))
    plan["semantics"]+=" SM allocation is a separate algorithm-quality condition because it changes the observed per-Euler new-KV frontiers; microbenchmark gains do not establish closed-loop quality."
    plan["limitations"].append("AE has 32 SMs; Vision/VLM share the disjoint 96-SM remainder. next_encoder=False remains fixed; this quality run cannot reproduce a benchmark that has a future observation available.")
    return plan


def rpc(port,message,timeout=600):
    result=base.rpc(port,message,timeout)
    if "protocol" in result:
        check_protocol(result["protocol"])
        check_resources(result["protocol"],require_actual=message.get("op")=="predict")
    return result


def validate_artifacts(directory,client,case):
    result=base.validate_artifacts(directory,client,case)
    rows=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
    layouts=[]
    for row in rows:
        check_protocol(row["protocol"],case["mode"])
        layouts.extend(check_resources(row["protocol"],require_actual=True))
    result["resource_adapter"]=dict(ae_sm_min=32,encoder_after_step=1,
        actual_sm_counts=sorted({tuple(layout["sm_counts"]) for layout in layouts}),
        verified_prediction_protocols=len(rows),next_encoder=False)
    return result


def run_worker(path):
    sys.path.insert(0,str(ROOT))
    from preparations.idea1_v2_1 import runtime_client as client
    from unittest.mock import patch
    case=json.loads(Path(path).read_text())
    original=client.rpc
    def checked_rpc(port,request,*args,**kwargs):
        response=original(port,request,*args,**kwargs)
        if "protocol" in response:
            check_protocol(response["protocol"],case["mode"])
        if request.get("op")=="predict":
            check_resources(response.get("protocol",{}),require_actual=True)
        return response
    with patch.object(client,"rpc",checked_rpc):
        return base.run_worker(path)


def parser():
    result=base.parser()
    result.description=__doc__
    result.set_defaults(modes=["lead3"],gpu_uuid="GPU-555cde36-5e41-bc44-0c69-ea47358fd703",
        egl_device=0,port=19086,cpus="96-111",output=str(HERE/"closedloop_sm32_gpu3"))
    return result


base.legacy.source_hashes=source_hashes
base.legacy.check_protocol=check_protocol
base.legacy.make_plan=make_plan
base.legacy.rpc=rpc
base.legacy.validate_artifacts=validate_artifacts
base.legacy.run_worker=run_worker
base.legacy.parser=parser


if __name__=="__main__":
    base.legacy.main()
