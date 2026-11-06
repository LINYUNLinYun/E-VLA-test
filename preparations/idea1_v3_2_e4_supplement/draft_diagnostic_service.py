"""Draft-only diagnostic service with four recorded GPU3 strict startup references."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback
import numpy as np
from .draft_diagnostic_audit import ROOT, HEADS, read


def strict_compare(actual, expected, shape):
    a,b=np.asarray(actual,np.float32),np.asarray(expected,np.float32)
    valid=a.shape==b.shape==shape and np.isfinite(a).all() and np.isfinite(b).all()
    return dict(passed=bool(valid and np.allclose(a,b,atol=1e-5,rtol=0)),
                bitwise_equal=bool(valid and np.array_equal(a,b)),
                max_abs=float(np.max(np.abs(a-b))) if valid else None,shape=list(a.shape),atol=1e-5,rtol=0)


def comparisons_accepted(comparisons):
    # User requires no cross-device numeric differences, even within strict1e-5.
    return all(c['passed'] and c['bitwise_equal'] for c in comparisons.values())


def reference_entries():
    ledger=read(ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json')
    result=[]
    for head in HEADS.values():
        entries=[e for e in ledger['selected_completed'] if e['mode']==head]
        entry=min(entries,key=lambda e:(e['group']!='perturbed350',e['source_index']))
        result.append(entry)
    return result


def source_identity():
    ledger=ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json'
    paths=[Path(__file__),ledger]+[ROOT/e['path'] for e in reference_entries()]
    for entry in reference_entries():
        directory=(ROOT/entry['path']).parent
        paths.extend([directory/'initial_native_observation.npz',directory/'infer.jsonl'])
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def startup_gate(policy):
    from .policy_server import native_libero_raw
    checks=[]
    for entry in reference_entries():
        result_path=ROOT/entry['path'];directory=result_path.parent;result=read(result_path)
        assert result['gpu_uuid']=='GPU-555cde36-5e41-bc44-0c69-ea47358fd703'
        with np.load(directory/'initial_native_observation.npz',allow_pickle=False) as archive:
            observation={k:archive[k].copy() for k in archive.files if k!='instruction'}
            instruction=str(archive['instruction'])
        with (directory/'infer.jsonl').open() as stream: recorded=json.loads(next(stream))
        for key,value in observation.items():
            digest=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            if digest!=recorded['input_hashes'][key]: raise RuntimeError('Reference first observation hash mismatch')
        response=policy.predict(entry['mode'],native_libero_raw(observation),instruction,int(recorded['seed']),category='preflight')
        comparisons=dict(actions=strict_compare(response['actions'],recorded['actions'],(50,7)),
                         raw_actions=strict_compare(response['raw_actions'],recorded['response']['raw_actions'],(50,32)))
        if response['input_hash']!=recorded['response']['input_hash']:
            raise RuntimeError('Reference model input digest mismatch')
        checks.append(dict(mode=entry['mode'],reference=str(result_path),reference_gpu_uuid=result['gpu_uuid'],
            reference_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest(),comparisons=comparisons))
        if not comparisons_accepted(comparisons):
            return dict(passed=False,checks=checks)
    return dict(passed=True,checks=checks,scope='Recorded GPU3 vs current device same real observations; raw+decoded bitwise equality mandatory; strict1e-5 also reported')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch: print(json.dumps(dict(dry_run=True,reference_entries=reference_entries())));return
    if a.output.exists(): raise FileExistsError(a.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=a.gpu_uuid: raise RuntimeError('Explicit UUID binding required')
    from preparations.idea1_v3_2.residual import require_idle_gpu
    from preparations.idea1_v3_2.transfer_policy import TransferPolicy,json_safe
    from preparations.idea1_v3_1.common import atomic_json
    from preparations.idea1_v3_1.server import receive
    from .policy_server import native_libero_raw
    gpu=require_idle_gpu()
    receipt=dict(status='constructing',gpu_uuid=a.gpu_uuid,gpu_preflight=gpu,server_pid=os.getpid(),draft_only=True,source_sha256=source_identity())
    atomic_json(a.output,receipt)
    try:
        identity=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        policy=TransferPolicy({},identity)
        from .policy_plus_mirror import audit_cpu_backend
        cpu=audit_cpu_backend(policy.backend)
        if cpu['cuda_initialized_before_guard']:raise RuntimeError('Unexpected constructor CUDA side effect')
        final=require_idle_gpu()
        receipt['final_preCUDA_guard']=dict(cpu_assembly_audit=cpu,gpu_preflight=final,checked=True)
        atomic_json(a.output,json_safe(receipt))
        gate=startup_gate(policy);receipt['gate']=gate;atomic_json(a.output,json_safe(receipt))
        if not gate['passed']: raise RuntimeError('STRICT_RECORDED_GPU3_DRAFT_GATE_FAILED; no quality permitted')
        policy.legacy_gate_passed=True
        policy.protocol.update(gpu_uuid=a.gpu_uuid,server_pid=os.getpid(),draft_only=True,
                               legacy_gate_passed=True,recorded_gpu3_gate=gate,manifest_hash=identity,
                               source_sha256=receipt['source_sha256'],final_preCUDA_guard=receipt['final_preCUDA_guard'])
        receipt.update(status='ready',protocol=policy.protocol);atomic_json(a.output,json_safe(receipt))
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('127.0.0.1',a.port));server.listen(8);server.settimeout(2)
            while time.time()<a.deadline_unix:
                try: connection,_=server.accept()
                except socket.timeout: continue
                shutdown=False
                with connection:
                    try:
                        size=struct.unpack('!I',receive(connection,4))[0]
                        if size>128*1024*1024:raise ValueError('Oversized request')
                        request=pickle.loads(receive(connection,size));op=request.pop('op')
                        if op=='health':response=dict(protocol=policy.protocol)
                        elif op=='shutdown':response={};shutdown=True
                        elif op=='reset':response=policy.reset()
                        elif op=='predict':
                            mode=request['mode'];suite=request['mother_suite']
                            if HEADS.get(suite)!=mode:raise ValueError('Exact matching mother-suite draft only')
                            response=policy.predict(mode,native_libero_raw(request['observation']),request['instruction'],int(request['seed']),category=request.get('category','quality'))
                            buffers=policy.backend.adapter.backend.runtime.buffers
                            def scale(value):
                                value=value.detach().float();return dict(shape=list(value.shape),rms=float(value.square().mean().sqrt().item()),l2=float(value.norm().item()))
                            response['feature_scales']=dict(visual_preencoder=scale(buffers['encoder_x'][:512]),normalized_state=scale(buffers['observation_state_normalized']))
                        else:raise ValueError('Unknown operation')
                        response=json_safe(dict(response,ok=True,manifest_hash=identity))
                    except Exception:response=dict(ok=False,error=traceback.format_exc())
                    payload=pickle.dumps(response,protocol=4);connection.sendall(struct.pack('!I',len(payload))+payload)
                if shutdown:break
        receipt.update(status='complete')
    except BaseException:
        receipt.update(status='failed',error=traceback.format_exc());raise
    finally:atomic_json(a.output,json_safe(receipt))


if __name__=='__main__':main()
