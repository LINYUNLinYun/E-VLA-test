"""Explicitly authorized CPU B endpoint over a shared, unchanged LIBERO service.

No model constructor, CUDA allocation, upstream source edit, or hidden category
remapping. Real B quality stays upstream quality. Its totals are PUBLICLY shared
Plus+B; this proxy separately attributes acknowledged calls to B. Timings are
shared-traffic, not isolated speed measurements. No retry after uncertain RPC.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback
import numpy as np

from preparations.idea1_v3_2.transfer_protocol import digest,sha,load_manifest
from preparations.idea1_v3_2.transfer_policy import MODES,validate_raw,json_safe
from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
from preparations.idea1_v3_1.common import atomic_json
from preparations.idea1_v3_1.server import receive
from .transfer_registration import load_registration
from .policy_B_server import validate_binding
from .policy_native_watch import rpc

ROOT=Path(__file__).resolve().parents[2]
SCOPE="corrected_B_transfer"
LATENCY="shared_upstream_traffic_including_queueing_not_isolated"
MIXED="upstream_actual_quality_totals_are_shared_Plus_and_corrected_B"


def check_upstream(health,manifest):
    protocol=health["protocol"];upstream=manifest["upstream"]
    for key,expected in (("manifest_hash",upstream["manifest_hash"]),("server_pid",upstream["server_pid"]),
                         ("gpu_uuid",upstream["gpu_uuid"]),("family","libero_transfer")):
        if protocol.get(key)!=expected:raise PermissionError("Shared upstream identity changed: "+key)
    if not protocol.get("legacy_gate_passed") or (protocol.get("H"),protocol.get("R"),protocol.get("D"))!=(50,5,10):
        raise PermissionError("Shared upstream real-input legacy gate/H50R5D10 missing")
    return protocol


def validate_response(response,manifest,raw,instruction,seed,mode,category):
    if response.get("manifest_hash")!=manifest["upstream"]["manifest_hash"] or response.get("mode")!=mode:
        raise ValueError("Shared source response identity/mode mismatch")
    actions=np.asarray(response.get("actions"));model_raw=np.asarray(response.get("raw_actions"))
    if actions.shape!=(50,7) or model_raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(model_raw).all():
        raise ValueError("Invalid shared source raw/decoded action chunk")
    noise=np.random.default_rng(int(seed)).standard_normal((50,32),dtype=np.float32)
    if response.get("input_hash")!=input_digest(raw,instruction) or response.get("noise_hash")!=value_digest(noise):
        raise ValueError("Shared source actual input/noise differs from B request")
    draft=mode.startswith("draft_")
    expected=dict(vision_encoder_calls=1,vlm_prefix_calls=0 if draft else 1,
        ae_layer_calls=0 if draft else 180,learned_head_calls=1 if draft else 0,verify_calls=0,
        ae_euler_updates=0 if draft else 10)
    observed=response.get("observed_calls",{})
    if any(observed.get(key)!=value for key,value in expected.items()):
        raise ValueError("Shared source actual route counters differ from B mode")
    route=response.get("route",{})
    if route.get("fresh_prefix") is not True or route.get("kv_cross_request") is not False or route.get("fallback") is not False or route.get("verify") is not False:
        raise ValueError("Shared source has hidden history/fallback/verification")
    if response.get("noise_used_in_inference") is not (not draft) or response.get("category")!=category:
        raise ValueError("B request category/noise use was relabelled upstream")
    return response


class BProxyPolicy:
    def __init__(self,registration,authorization,manifest,forward,upstream_health,*,ledger=None):
        validate_binding(registration,authorization,manifest)
        if registration.get("schema")!=2 or manifest.get("kind")!="independent_corrected_B_CPU_proxy":
            raise PermissionError("Final Bv2 registration and independent proxy identity required")
        upstream=manifest["upstream"]
        checks=dict(upstream_service_manifest_hash=upstream["manifest_hash"],upstream_server_pid=upstream["server_pid"],
            upstream_execution_extension_sha256=upstream["execution_extension"]["sha256"])
        if authorization.get("allow_shared_upstream_model") is not True:
            raise PermissionError("Explicit existing-model B sharing authorization required")
        for key,value in checks.items():
            if authorization.get(key)!=value:raise PermissionError("Parent shared-upstream B binding mismatch: "+key)
        if authorization["gpu_uuid"]!=upstream["gpu_uuid"] or authorization["gpu"]!=upstream["gpu"]:
            raise PermissionError("Parent B GPU differs existing upstream owner")
        check_upstream(upstream_health,manifest)
        self.registration,self.authorization,self.manifest=registration,authorization,manifest
        self.forward=forward;self.ledger=Path(ledger) if ledger else None
        self.cases={case["case_id"]:case for case in registration["cases"]}
        self.namespace=SCOPE+":"+registration["registration_hash"]
        self.counters={};self.bank_gate_passed=False
        self.protocol=dict(family="libero_transfer",scope=SCOPE,manifest_hash=manifest["manifest_hash"],
            registration_hash=registration["registration_hash"],authorized_scientific_registrations=[registration["registration_hash"]],
            model_reference_manifest_hash=registration["policy_reference"]["manifest_hash"],counter_namespace=self.namespace,
            server_pid=os.getpid(),proxy_pid=os.getpid(),backend_server_pid=upstream["server_pid"],
            upstream_manifest_hash=upstream["manifest_hash"],gpu_uuid=upstream["gpu_uuid"],gpu_owner="upstream_model_process_not_proxy",
            model_constructed=False,proxy_cuda_allocation=False,H=50,R=5,D=10,modes=list(MODES),
            source_legacy_gate_passed=True,legacy_gate_passed=False,proxy_bank_gate_passed=False,
            latency_context=LATENCY,upstream_counter_scope=MIXED,
            adapter_contract_hash=manifest["adapter_contract_hash"],shutdown_scope="proxy_only_never_upstream")

    def event(self,row):
        if self.ledger:
            with self.ledger.open("a") as stream:
                stream.write(json.dumps(json_safe(dict(row,time_unix=time.time())),allow_nan=False)+"\n");stream.flush()

    def reset(self,registration_hash):
        if registration_hash!=self.registration["registration_hash"]:raise PermissionError("B reset registration mismatch")
        return dict(request_local=True,upstream_reset_forwarded=False,
            reason="Frozen source has no trajectory cache; B pose/lift reset remains native client-owned",counter_namespace=self.namespace)

    def predict(self,*,registration_hash,case_id,environment,mode,raw,instruction,seed,category="quality"):
        validate_binding(self.registration,self.authorization,self.manifest)
        if registration_hash!=self.registration["registration_hash"]:raise PermissionError("B request registration mismatch")
        case=self.cases.get(case_id)
        if case is None or case["environment"]!=environment or mode not in MODES:
            raise PermissionError("Unregistered B case/environment/mode; Plus/native observations forbidden")
        if category not in ("quality","preflight","warmup"):raise ValueError("Unknown B request category")
        if category!="preflight" and not self.bank_gate_passed:raise RuntimeError("Explicit real proxy bank gate required before B quality")
        raw=validate_raw(raw);tick=time.perf_counter()
        count=self.counters.setdefault(category,{}).setdefault(mode,Counter())
        count["attempted_requests"]+=1
        identity=dict(registration_hash=registration_hash,case_id=case_id,case_hash=digest(case),environment=environment,
            mode=mode,category=category,seed=int(seed),input_hash=input_digest(raw,instruction))
        self.event(dict(identity,event="authorized_request"))
        forwarded=False
        try:
            check_upstream(self.forward({"op":"health"}),self.manifest)
            payload=dict(op="predict",family="libero_transfer",environment=environment,mode=mode,
                raw=raw,instruction=instruction,seed=int(seed),category=category)
            count["forward_attempts"]+=1;forwarded=True;forward_tick=time.perf_counter()
            source=self.forward(payload)  # unchanged real quality stays quality
            forward_ms=(time.perf_counter()-forward_tick)*1000
            validate_response(source,self.manifest,raw,instruction,seed,mode,category)
        except Exception:
            count["failed_requests"]+=1
            count["uncertain_model_completion"]+=int(forwarded)
            self.event(dict(identity,event="failed_request",forward_attempted=forwarded,
                model_completion_unknown=forwarded,error=traceback.format_exc()))
            raise
        count["acknowledged_model_requests"]+=1;count.update(source["observed_calls"])
        upstream_counters=source.get("counters",{})
        sequence=upstream_counters.get(category,{}).get(mode,{}).get("logical_requests")
        self.event(dict(identity,event="acknowledged_model_request",observed_calls=source["observed_calls"],
            upstream_category_mode_sequence=sequence,upstream_manifest_hash=self.manifest["upstream"]["manifest_hash"],
            noise_hash=source["noise_hash"],decoded_hash=value_digest(np.asarray(source["actions"],np.float32)),
            raw_hash=value_digest(np.asarray(source["raw_actions"],np.float32)),latency_context=LATENCY))
        return dict(source,manifest_hash=self.manifest["manifest_hash"],protocol_hash=self.manifest["manifest_hash"],
            registration_hash=registration_hash,counter_namespace=self.namespace,counters={self.namespace:json_safe(self.counters)},
            upstream_manifest_hash=self.manifest["upstream"]["manifest_hash"],upstream_server_pid=self.manifest["upstream"]["server_pid"],
            upstream_shared_counters_snapshot=upstream_counters,upstream_category_mode_sequence=sequence,
            upstream_counter_scope=MIXED,latency_context=LATENCY,
            proxy_timing=dict(forward_wall_ms=forward_ms,end_to_end_ms=(time.perf_counter()-tick)*1000),
            attribution=dict(identity,scope=SCOPE,service_manifest_hash=self.manifest["manifest_hash"],
                model_reference_manifest_hash=self.registration["policy_reference"]["manifest_hash"],
                adapter_contract_hash=self.manifest["adapter_contract_hash"],upstream_counter_scope=MIXED))

    def bank_gate(self,bank):
        if self.authorization.get("allow_proxy_bank_gate") is not True:
            raise PermissionError("Parent must separately authorize small real direct/proxy bank gate")
        from preparations.idea1_v2_1.runtime_models import native_observation
        record=pickle.loads(Path(bank).read_bytes())[0]
        images,state=native_observation(record["obs"])
        raw=dict(base_rgb=np.asarray(images[0]),wrist_rgb=np.asarray(images[1]),state8=np.asarray(state,np.float32))
        case=self.registration["cases"][0];checks=[]
        for mode in MODES:
            direct=self.forward(dict(op="predict",family="libero_transfer",environment=case["environment"],
                mode=mode,raw=raw,instruction=record["instruction"],seed=int(record["seed"]),category="preflight"))
            validate_response(direct,self.manifest,raw,record["instruction"],record["seed"],mode,"preflight")
            proxied=self.predict(registration_hash=self.registration["registration_hash"],case_id=case["case_id"],
                environment=case["environment"],mode=mode,raw=raw,instruction=record["instruction"],seed=int(record["seed"]),category="preflight")
            comparisons={}
            for key in ("raw_actions","actions"):
                a,b=np.asarray(direct[key],np.float32),np.asarray(proxied[key],np.float32)
                comparisons[key]=dict(bitwise_equal=bool(np.array_equal(a,b)),max_abs=float(np.max(np.abs(a-b))),
                    passed=bool(np.allclose(a,b,atol=1e-5,rtol=0)))
            checks.append(dict(mode=mode,comparisons=comparisons,direct_observed_calls=direct["observed_calls"],proxy_observed_calls=proxied["observed_calls"]))
        self.bank_gate_passed=all(value["passed"] for row in checks for value in row["comparisons"].values())
        self.protocol.update(proxy_bank_gate_passed=self.bank_gate_passed,legacy_gate_passed=self.bank_gate_passed)
        return dict(passed=self.bank_gate_passed,bank_path=str(bank),bank_sha256=sha(bank),checks=checks,
            source_requests=12,direct_reference_requests=6,proxy_path_requests=6,category="preflight",
            scope="Same actual native LIBERO bank input/noise, direct vs CPUproxy transport raw+decoded; NOT B environment policy parity or B quality samples")


def build_manifest(registration_path,extension_path,upstream_pid,upstream_gpu,upstream_uuid,upstream_port):
    registration=load_registration(registration_path,verify_sources=True)
    if registration.get("schema")!=2:raise ValueError("Final official-options Bv2 registration required")
    reference_path=registration["input_files"]["model_protocol_reference"]["path"]
    reference=load_manifest(reference_path);extension=json.loads(Path(extension_path).read_text())
    if reference["manifest_hash"]!=registration["policy_reference"]["manifest_hash"] or extension.get("manifest_hash")!=reference["manifest_hash"]:
        raise ValueError("Shared model/extension differs registered weights-only reference")
    sources=dict(reference["source_files"])
    sources.update(registration["source_files"])
    for path in (Path(__file__),Path(__file__).with_name("test_policy_B_proxy.py"),Path(__file__).with_name("policy_B_server.py"),Path(__file__).with_name("policy_native_watch.py")):
        sources[str(path.relative_to(ROOT))]=sha(path)
    adapter=digest({key:registration.get(key) for key in ("state","source_action","reset","source_files")})
    manifest=dict(schema=1,kind="independent_corrected_B_CPU_proxy",scope=SCOPE,family="libero_transfer",
        registration_hash=registration["registration_hash"],model_reference_manifest_hash=reference["manifest_hash"],
        adapter_contract_hash=adapter,H=50,R=5,modes=list(MODES),case_count=300,condition_count=1800,
        upstream=dict(manifest_hash=reference["manifest_hash"],server_pid=int(upstream_pid),gpu=int(upstream_gpu),
            gpu_uuid=upstream_uuid,port=int(upstream_port),execution_extension=dict(path=str(Path(extension_path).resolve()),sha256=sha(extension_path))),
        source_files=sources,receipt_files={str(Path(path).resolve()):sha(path) for path in (registration_path,extension_path,reference_path)},
        created_unix=time.time(),model_constructed=False,gpu_used=False,quality_launch_authorized=False,
        upstream_counter_scope=MIXED,latency_context=LATENCY,
        counters="Real B quality remains sourcequality; source totals explicitlyshared; independent B ack/attempt/uncertain ledger; no estimates for failed unknown calls",
        authorization="Requires new parent auth binding independentproxyhash/Bv2hash/upstreammanifest/PID/UUID/extension and explicit sharedmodel+smallbank permission",
        shutdown_scope="CPUproxy only, never shutdown existing Plus model")
    manifest["manifest_hash"]=digest(manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--registration",required=True);p.add_argument("--output",required=True)
    p.add_argument("--freeze-manifest",action="store_true");p.add_argument("--extension");p.add_argument("--upstream-pid",type=int)
    p.add_argument("--upstream-gpu",type=int);p.add_argument("--upstream-uuid");p.add_argument("--upstream-port",type=int,default=19143)
    p.add_argument("--authorization");p.add_argument("--protocol");p.add_argument("--port",type=int,default=19145)
    p.add_argument("--deadline-unix",type=float);p.add_argument("--legacy-bank",default="results/idea1_v3_1/input_bank.pkl");p.add_argument("--launch",action="store_true")
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    if a.freeze_manifest:
        if not all((a.extension,a.upstream_pid,a.upstream_uuid)) or a.upstream_gpu is None:raise ValueError("Explicit current upstream owner/extension required")
        result=build_manifest(a.registration,a.extension,a.upstream_pid,a.upstream_gpu,a.upstream_uuid,a.upstream_port)
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(manifest_hash=result["manifest_hash"],model_constructed=False,quality_launch_authorized=False)));return
    if not a.launch:
        print(json.dumps(dict(dry_run=True,model_constructed=False,gpu_used=False)));return
    if not all((a.authorization,a.protocol,a.deadline_unix)):raise PermissionError("Explicit B proxy parent authorization/protocol/deadline required")
    registration=load_registration(a.registration,verify_sources=True);manifest=load_manifest(a.protocol)
    authorization=json.loads(Path(a.authorization).read_text());validate_binding(registration,authorization,manifest)
    if a.deadline_unix>authorization["deadline_unix"]:raise PermissionError("B proxy deadline exceeds parent auth")
    forward=lambda payload:rpc(manifest["upstream"]["port"],payload,timeout=120)
    output.parent.mkdir(parents=True,exist_ok=True)
    policy=BProxyPolicy(registration,authorization,manifest,forward,forward({"op":"health"}),ledger=output.with_suffix(".ledger.jsonl"))
    receipt=dict(status="gating",proxy_pid=os.getpid(),manifest_hash=manifest["manifest_hash"],
        registration_hash=registration["registration_hash"],authorization_sha256=sha(a.authorization),
        deadline_unix=a.deadline_unix,model_constructed=False,gpu_used_by_proxy=False,upstream_counter_scope=MIXED,latency_context=LATENCY)
    atomic_json(output,receipt)
    try:
        receipt["bank_gate"]=policy.bank_gate(a.legacy_bank)
        if not receipt["bank_gate"]["passed"]:raise RuntimeError("B proxy direct/rawdecoded transport parity failed")
        receipt.update(status="ready",protocol=policy.protocol);atomic_json(output,receipt)
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(("127.0.0.1",a.port));server.listen(8);server.settimeout(3)
            while time.time()<a.deadline_unix:
                try:connection,_=server.accept()
                except socket.timeout:continue
                stop=False
                with connection:
                    try:
                        size=struct.unpack("!I",receive(connection,4))[0]
                        if size>128*1024*1024:raise ValueError("Oversized localhost request")
                        request=pickle.loads(receive(connection,size));op=request.pop("op")
                        if op=="health":result=dict(protocol=policy.protocol,authorized_scientific_registrations=policy.protocol["authorized_scientific_registrations"])
                        elif op=="predict":
                            if request.pop("family","libero_transfer")!="libero_transfer" or "observation" in request:raise PermissionError("B raw branch only")
                            result=policy.predict(**request)
                        elif op=="reset":result=policy.reset(**request)
                        elif op=="shutdown":result,stop={},True
                        else:raise ValueError("Unknown B proxy operation")
                        result=json_safe(dict(result,ok=True,manifest_hash=manifest["manifest_hash"],protocol_hash=manifest["manifest_hash"]))
                    except Exception:result=dict(ok=False,failure_code="B_PROXY_SCOPE_OR_TECHNICAL",error=traceback.format_exc())
                    data=pickle.dumps(result,protocol=4);connection.sendall(struct.pack("!I",len(data))+data)
                receipt["counters"]={policy.namespace:json_safe(policy.counters)};atomic_json(output,receipt)
                if stop:break
        receipt.update(status="complete",completed_unix=time.time())
    except Exception:receipt.update(status="failed",error=traceback.format_exc(),completed_unix=time.time());raise
    finally:atomic_json(output,receipt)


if __name__=="__main__":main()
