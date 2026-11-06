"""Original OpenPI policy and semantic router behind the existing local RPC.

The original LIBERO rollout client is reused unchanged. Exactly one sequential
episode is configured at a time; configure resets the routing counter.
"""
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

import torch

from preparations.idea1_v2_1.runtime_client import input_digest
from preparations.idea1_v3_1.common import atomic_json, read_manifest
from preparations.idea1_v3_1.server import receive
from preparations.idea2.experiments import native_raw
from .native_probe import NativeProbe
from .runtime_router import SemanticSkipRuntime


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--runtime-check',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    manifest=read_manifest(args.manifest,verify_sources=True)
    weight=manifest['checkpoint_weight_metadata'];stat=Path(weight['path']).stat()
    if (stat.st_size,stat.st_mtime_ns)!=(weight['bytes'],weight['mtime_ns']):
        raise ValueError('Checkpoint weight metadata changed')
    candidate_path=Path(manifest['candidate'])
    candidate=json.loads(candidate_path.read_text())
    if hashlib.sha256(candidate_path.read_bytes()).hexdigest()!=manifest['candidate_sha256']:
        raise ValueError('Frozen candidate changed')
    check=json.loads(Path(args.runtime_check).read_text())
    if not (check['passed'] and check['device']==args.device
            and check.get('candidate_sha256')==manifest['candidate_sha256']
            and Path(check['checkpoint']).resolve()==Path(manifest['checkpoint']).resolve()):
        raise ValueError('Need a passing same-device, same-candidate action-path check')
    if not check.get('runtime_sources'):
        raise ValueError('Runtime check lacks source identity')
    for filename,expected in check['runtime_sources'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Runtime changed since action-path check: {filename}')
    probe=NativeProbe(manifest['checkpoint'],args.device,args.threads)
    if probe.backend.num_steps!=10 or probe.model.config.action_horizon!=10:
        raise ValueError('Keep original H10/D10 policy')
    runtime=SemanticSkipRuntime(probe,candidate,manifest['assets'])
    runtime.detach()
    conditions={c['id']:c for c in manifest['conditions']}
    state={}
    protocol=dict(manifest_hash=manifest['manifest_hash'],condition_id=None,H=10,R=5,num_steps=10,
                  device=args.device,checkpoint=manifest['checkpoint'],server_pid=os.getpid(),
                  real_input_parity_passed=True,runtime_check=str(Path(args.runtime_check).resolve()),
                  original_full='Original unhooked sample_actions; no segmentation or SAE',
                  concurrency='One sequential rollout client; configure resets each episode')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)

    def synchronize():
        if probe.backend.device.type=='cuda':torch.cuda.synchronize(probe.backend.device)

    def dispatch(request):
        operation=request.pop('op')
        if operation=='health':return dict(protocol=protocol.copy())
        if operation=='configure':
            condition=conditions[request['condition']]
            scene=request['scene']
            if scene not in manifest['episodes']:raise ValueError('Scene is not in frozen heldout batch')
            key=f"{condition['id']}/{scene['suite']}/{scene['task']}/{scene['init']}"
            runtime.detach()
            if condition['mode']!='full':runtime.attach()
            runtime.reset_episode(key)
            state.update(condition=condition,episode_key=key,scene=scene,request=0)
            protocol['condition_id']=condition['id']
            atomic_json(output/'protocol.json',protocol)
            return dict(configured=key)
        if operation=='predict':
            if not state:raise ValueError('Configure an episode before inference')
            scene=state['scene']
            if request['suite']!=scene['suite'] or request['instruction']!=scene['instruction']:
                raise ValueError('Request differs from configured scene')
            synchronize();started=time.perf_counter()
            prepared=probe.backend.prepare(native_raw(request))
            noise=probe.backend.noise(request['seed'])
            mode=state['condition']['mode']
            if mode=='full':
                raw=probe.backend.original(prepared,noise)
                actions=probe.backend.output(prepared,raw)
                trace=dict(mode='full',mask=[],reason='original_full',confidence={},active_concepts=[],
                           request=state['request'],policy_call=state['request']+1,sae_router_ms=0.)
            else:
                _,actions,trace=runtime.infer_prepared(prepared,request['instruction'],noise,state['episode_key'],mode)
            synchronize();elapsed=(time.perf_counter()-started)*1000
            state['request']+=1
            return dict(actions=actions,input_hash=input_digest(request['obs'],request['instruction']),
                        policy_ms=elapsed,routing=trace,model='pi05',H=10,R=5,num_steps=10)
        if operation=='shutdown':return dict(shutdown=True)
        raise ValueError(f'Unknown operation: {operation}')

    try:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',args.port));server.listen(2)
            atomic_json(output/'ready.json',dict(pid=os.getpid(),port=args.port,protocol=protocol))
            print(json.dumps(dict(event='ready',port=args.port,device=args.device)),flush=True)
            while True:
                connection,_=server.accept()
                with connection:
                    try:
                        size=struct.unpack('!I',receive(connection,4))[0]
                        if size>128*1024*1024:raise ValueError('Oversized localhost request')
                        result=dispatch(pickle.loads(receive(connection,size)))
                        result['ok']=True
                    except Exception:
                        result=dict(ok=False,error=traceback.format_exc())
                    payload=pickle.dumps(result,protocol=5)
                    connection.sendall(struct.pack('!I',len(payload))+payload)
                    if result.get('shutdown'):return
    finally:
        runtime.detach()


if __name__=='__main__':main()
