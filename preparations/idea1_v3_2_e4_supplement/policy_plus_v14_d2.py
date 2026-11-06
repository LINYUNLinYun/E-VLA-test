"""Independent V14/D2 route and serialized native Plus RPC server.

Never changes the frozen D10 mode table or its gate. Construction is explicit.
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
from preparations.idea1_v3.policy import TritonBackend, V14
from preparations.idea1_v3_2.transfer_policy import RawTritonBackend, validate_raw, route_observations, observed_counts, json_safe
from preparations.idea1_v3_1.server import receive
from .native_envs import write_json
from .evaluation import protocol_identity
from .policy_server import native_libero_raw


def array(value):
    if hasattr(value, 'detach'):
        return value.detach().float().cpu().numpy().copy()
    return np.asarray(value, np.float32)


def compare(a, b, limit):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid comparison arrays')
    delta = np.abs(a-b)
    return dict(max_abs=float(delta.max(initial=0)), bitwise_equal=bool(np.array_equal(a,b)),
                allclose_002=bool(np.allclose(a,b,atol=.002,rtol=.002)), passed=bool(delta.max(initial=0)<=limit))


class V14D2Policy:
    def __init__(self, manifest, backend=None):
        self.backend = backend if backend is not None else RawTritonBackend()
        self.manifest = manifest
        self.counters = {name: {} for name in ('preflight','preflight_reference','quality')}
        self.gate_passed = False
        self.protocol = dict(manifest_hash=manifest['manifest_hash'], family='libero_transfer', modes=['v14_d2'],
            H=50,R=5,D=2,ae_depth=18,vision_depth=27,skip_vlm=list(V14),nodes=[1.,.5,0.],
            fresh_prefix=True,verify=False,fallback=False,legacy_gate_passed=False)

    def reset(self):
        return dict(request_local=True,counters=self.counters)

    def predict(self, mode, raw, instruction, seed, category='quality'):
        if mode!='v14_d2' or category not in ('preflight','quality'):
            raise ValueError('Only independent v14_d2 route is registered')
        if category=='quality' and not self.gate_passed:
            raise RuntimeError('V14_D2_NATIVE_GATE_REQUIRED')
        if not isinstance(instruction,str) or not instruction.strip() or not isinstance(seed,(int,np.integer)) or seed<0:
            raise ValueError('Nonempty actual language and nonnegative integer seed required')
        raw=validate_raw(raw)
        from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
        tick=time.perf_counter();cumulative=self.counters[category].setdefault(mode,Counter())
        cumulative['attempted_requests']+=1
        with route_observations() as observed:
            context,timing=self.backend.prepare(raw,instruction,V14,encode=True)
            actions,model_raw,costs=self.backend.sample(context,int(seed),2)
        calls=observed_counts(observed);cumulative.update(calls)
        expected=dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,
                      learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0)
        if any(calls[k]!=v for k,v in expected.items()) or costs['actual_NFE']!=2 or not np.allclose(costs['nodes'],[1,.5,0],atol=1e-7,rtol=0):
            raise RuntimeError('Actual V14/D2 execution contract violated: '+str((calls,costs)))
        actions,model_raw=array(actions),array(model_raw)
        if actions.shape!=(50,7) or model_raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(model_raw).all():
            raise ValueError('Nonfinite/invalid H50 outputs')
        cumulative['logical_requests']+=1;calls['ae_euler_updates']=2
        noise=np.random.default_rng(int(seed)).standard_normal((50,32),dtype=np.float32)
        return json_safe(dict(actions=actions,raw_actions=model_raw,execute_length=5,mode=mode,seed=int(seed),
            input_hash=input_digest(raw,instruction),noise_hash=value_digest(noise),noise_used_in_inference=True,
            timing=dict(timing,**costs,request_wall_ms=(time.perf_counter()-tick)*1000),
            route=dict(mode=mode,fresh_prefix=True,full_vision=True,skip_vlm=list(V14),kv_cross_request=False,
                       fallback=False,verify=False),observed_calls=calls,category=category,counters=self.counters,
            manifest_hash=self.manifest['manifest_hash'],protocol_hash=self.manifest['manifest_hash']))

    def preflight(self):
        """Saved real Full/V14 D10 first requests plus strict D2 native parity."""
        checks=[]
        for saved in self.manifest['startup_inputs']:
            with np.load(saved['observation_path'],allow_pickle=False) as data:
                obs={k:data[k] for k in data.files if k!='instruction'};instruction=str(data['instruction'])
            row=json.loads(Path(saved['infer_path']).open().readline());seed=row['seed']
            self.backend.ensure_prompt(instruction)
            skip=V14 if saved['mode']=='v14_d10' else ()
            with route_observations() as observed:
                context,_=TritonBackend.prepare(self.backend,obs,instruction,skip,encode=True)
                actions,raw,_=TritonBackend.sample(self.backend,context,seed,10)
            historical=dict(raw=compare(array(raw),row['response']['raw_actions'],.02),
                            decoded=compare(array(actions),row['actions'],.02),
                            first5_gripper_sign_equal=bool(np.array_equal(np.sign(array(actions)[:5,6]),np.sign(np.asarray(row['actions'])[:5,6]))))
            self.counters['preflight_reference'][saved['mode']]=observed_counts(observed)
            with route_observations() as observed:
                context,_=TritonBackend.prepare(self.backend,obs,instruction,V14,encode=True)
                expected,expected_raw,_=TritonBackend.sample(self.backend,context,seed,2)
            reference_calls=observed_counts(observed)
            candidate=self.predict('v14_d2',native_libero_raw(obs),instruction,seed,category='preflight')
            parity={k:compare(a,b,1e-5) for k,a,b in (
                ('raw',array(expected_raw),candidate['raw_actions']),('decoded',array(expected),candidate['actions']))}
            parity['reference_calls']=reference_calls
            parity['passed']=all(parity[k]['passed'] for k in ('raw','decoded')) and reference_calls['ae_layer_calls']==36 and reference_calls['vision_encoder_calls']==1 and reference_calls['vlm_prefix_calls']==1
            checks.append(dict(saved=saved,historical=historical,d2_native_parity=parity))
        self.gate_passed=bool(checks) and all(c['historical']['raw']['passed'] and c['historical']['decoded']['passed'] and c['historical']['first5_gripper_sign_equal'] and c['d2_native_parity']['passed'] for c in checks)
        self.protocol['legacy_gate_passed']=self.gate_passed
        return dict(passed=self.gate_passed,checks=checks,quality_requests=0,
                    cross_gpu_limitations='Historical585 GPU3; D2 assigned idle device; numeric diagnostic only, no quality rerun')


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--port',required=True,type=int);p.add_argument('--deadline-unix',required=True,type=float);p.add_argument('--launch',action='store_true');a=p.parse_args()
    from .plus_v14_d2 import verify_manifest
    m=verify_manifest(a.protocol)
    if not a.launch: print(json.dumps(dict(dry_run=True,manifest_hash=m['manifest_hash'],D=2)));return
    if Path(a.output).exists(): raise FileExistsError(a.output)
    if not time.time()<a.deadline_unix<=m['deadline_unix']: raise ValueError('Fresh bounded deadline required')
    from preparations.idea1_v3_2.residual import require_idle_gpu
    gpu=require_idle_gpu();receipt=dict(status='constructing',server_pid=os.getpid(),gpu_preflight=gpu,manifest_hash=m['manifest_hash'],deadline_unix=a.deadline_unix)
    write_json(a.output,receipt)
    try:
        policy=V14D2Policy(m)
        from .policy_plus_mirror import audit_cpu_backend
        cpu=audit_cpu_backend(policy.backend)
        if cpu['cuda_initialized_before_guard']: raise RuntimeError('Unexpected CUDA before final idle guard')
        final_gpu=require_idle_gpu()
        policy.protocol['final_preCUDA_guard']=dict(checked=True,cpu_assembly_audit=cpu,gpu_preflight=final_gpu)
        receipt['final_preCUDA_guard']=policy.protocol['final_preCUDA_guard']
        gate=policy.preflight();receipt['gate']=gate;write_json(a.output,receipt)
        if not gate['passed']: raise RuntimeError('V14_D2_STARTUP_GATE_FAILED')
        policy.protocol.update(server_pid=os.getpid(),gpu_uuid=gpu['device'].split(',')[1].strip())
        receipt.update(status='ready',protocol=policy.protocol);write_json(a.output,receipt)
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('127.0.0.1',a.port));server.listen(4);server.settimeout(2)
            while time.time()<a.deadline_unix:
                try: connection,_=server.accept()
                except socket.timeout: continue
                stop=False
                with connection:
                    try:
                        size=struct.unpack('!I',receive(connection,4))[0]
                        if size>128*1024*1024: raise ValueError('Oversized RPC')
                        request=pickle.loads(receive(connection,size));op=request.pop('op')
                        if op=='health': result=dict(protocol=policy.protocol)
                        elif op=='reset': result=policy.reset()
                        elif op=='shutdown': result={};stop=True
                        elif op=='predict':
                            if request.pop('family',None)!='libero_transfer' or request.pop('environment',None) not in ('libero','libero_plus'): raise ValueError('Native Plus family required')
                            request.pop('mother_suite',None);request['raw']=native_libero_raw(request.pop('observation'))
                            result=policy.predict(**request)
                        else: raise ValueError('Unknown RPC operation')
                        result=json_safe(dict(result,ok=True,manifest_hash=m['manifest_hash'],protocol_hash=m['manifest_hash']))
                    except Exception: result=dict(ok=False,failure_code='TECHNICAL',error=traceback.format_exc())
                    payload=pickle.dumps(result,protocol=4);connection.sendall(struct.pack('!I',len(payload))+payload)
                receipt['counters']=json_safe(policy.counters);write_json(a.output,receipt)
                if stop: break
        receipt['status']='complete'
    except Exception: receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally: receipt['completed_unix']=time.time();write_json(a.output,json_safe(receipt))


if __name__=='__main__': main()
