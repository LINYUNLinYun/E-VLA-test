"""Frozen v3.3 episode loop with common warmup and ack-inclusive cost."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import uuid
import numpy as np
from preparations.idea1_v3.io import atomic,write_rows,digest
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v3_1.common import LIMITS,lock
from preparations.idea1_v2_1.runtime_client import official_init_states,rpc
from preparations.idea1_v2_2.common import seed


def episode(directory,port,case,config,pool,manifest_hash,probe=False,block=None,binding=None):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter(); timing={}; tick=begin
    env,definition,bddl,selected,created=pool.acquire(case['suite'],case['task'])
    timing['environment_acquire_s']=time.perf_counter()-tick
    try:
        vector=np.asarray(official_init_states(selected,case['task'])[case['init']])
        random.seed(7); np.random.seed(7); tick=time.perf_counter(); env.seed(7); env.reset(); obs=env.set_init_state(vector.copy())
        for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
        timing['reset_s']=time.perf_counter()-tick
        if float(env.env.control_freq)!=20.: raise ValueError('Control frequency changed')
        controller=env.env.robots[0].controller
        scales={k:np.asarray(getattr(controller,k)).tolist() for k in ('input_min','input_max','output_min','output_max')}
        if not np.allclose(scales['output_max'],[.05,.05,.05,.5,.5,.5]): raise ValueError('Controller scale changed')
        model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
        initial_hash=digest({k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in model_obs.items()})
        generation_seed=seed(case['suite'],case['task'],case['init'],0)
        probe_s=0.
        if probe:
            tick=time.perf_counter();response=rpc(port,dict(op='probe',obs=model_obs,instruction=definition.language,suite=case['suite'],seed=generation_seed));probe_s=time.perf_counter()-tick
            atomic(Path(directory).parents[4]/f"native_probe_gpu{binding['model_gpu']}.json",response)
            if not response['passed']:raise ValueError('Native Full/V14 real-input gate failed')
        warm=rpc(port,dict(op='configure',config=dict(_control='common_warm',obs=model_obs,instruction=definition.language,suite=case['suite'],seed=generation_seed)))
        if block.get('initial_hash') not in (None,initial_hash):raise ValueError('Initial full RGB/state differs across conditions')
        if block.get('startup_signature') not in (None,warm['canonical']):raise ValueError('Canonical initial Full differs across conditions')
        rpc(port,dict(op='configure',config=config)); protocol=rpc(port,dict(op='health'))['protocol']
        atomic(directory/'identity.json',dict(case=case,config=config,protocol=protocol,manifest_hash=manifest_hash,engineering_binding=binding,
            init_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),controller=scales,env_seed=7,settling=10,
            initial_hash=initial_hash,common_warm=warm,native_probe_s=probe_s))
        ack_policy_ms=0.
        infers=[]; telemetry=[]; arrays={}; steps=0; done=False
        def capture(action=None):
            telemetry.append(dict(step=steps,eef_pos=np.asarray(obs['robot0_eef_pos']).tolist(),eef_quat=np.asarray(obs['robot0_eef_quat']).tolist(),
                action=None if action is None else action.tolist(),sim_time_s=float(env.env.sim.data.time)))
        capture(); timing['rpc_s']=timing['step_s']=timing['ack_s']=0.
        while steps<LIMITS[case['suite']] and not done:
            index=len(infers); generation_seed=seed(case['suite'],case['task'],case['init'],index)
            model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
            if index==0:
                np.savez_compressed(directory/'input0.npz',**model_obs,instruction=np.asarray(definition.language),seed=np.asarray(generation_seed))
            tick=time.perf_counter(); r=rpc(port,dict(op='predict',obs=model_obs,instruction=definition.language,suite=case['suite'],seed=generation_seed)); elapsed=time.perf_counter()-tick; timing['rpc_s']+=elapsed
            if index==0:
                signature=dict(raw_sha256=hashlib.sha256(np.asarray(r['raw']).tobytes()).hexdigest(),decoded_sha256=hashlib.sha256(np.asarray(r['actions']).tobytes()).hexdigest())
                if signature!=warm['canonical']:raise ValueError('Actual startup Full differs from warmed canonical reference')
                block.update(initial_hash=initial_hash,startup_signature=signature)
            actions=np.asarray(r.pop('actions'),np.float32); raw=r.pop('raw'); small=r.pop('arrays'); r.pop('ok',None)
            arrays[f'actions_{index}']=actions; arrays[f'raw_{index}']=raw
            for name,value in small.items(): arrays[f'{name}_{index}']=value
            record=dict(r,rpc_ms=elapsed*1000,replan=index,step=steps,executed_length=0,input_hash=digest({k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in model_obs.items()}))
            infers.append(record); executed=[]; poses=[]; tick=time.perf_counter()
            for action in actions[:min(r['execute_length'],LIMITS[case['suite']]-steps)]:
                obs,_,done,_=env.step(action.tolist()); steps+=1; executed.append(action); poses.append(np.asarray(obs['robot0_eef_quat']).copy()); record['executed_length']+=1; capture(action)
                if done: break
            timing['step_s']+=time.perf_counter()-tick; tick=time.perf_counter()
            ack=rpc(port,dict(op='ack',executed=np.asarray(executed),quaternions=np.asarray(poses))); timing['ack_s']+=time.perf_counter()-tick
            record['ack_policy_wall_ms']=ack['ack_policy_wall_ms'];ack_policy_ms+=ack['ack_policy_wall_ms']
        tick=time.perf_counter(); write_rows(directory/'infer.jsonl',infers); write_rows(directory/'telemetry.jsonl',telemetry); np.savez(directory/'arrays.npz',**arrays); timing['save_s']=time.perf_counter()-tick
        result=dict(valid=True,success=bool(done),case=case,condition=config['id'],config=config,manifest_hash=manifest_hash,
            actions=steps,predictions=len(infers),wall_s=time.perf_counter()-begin,timing=timing,policy_wall_s=(sum(r['timing']['policy_wall_ms'] for r in infers)+ack_policy_ms)/1000,
            predict_policy_wall_s=sum(r['timing']['policy_wall_ms'] for r in infers)/1000,ack_policy_wall_s=ack_policy_ms/1000,common_warm_s=warm['common_warm_ms']/1000,native_probe_s=probe_s,
            engineering_binding=binding,initial_hash=initial_hash,startup_signature=block['startup_signature'],environment_reused=not created,counts=infers[-1]['counts'],termination='success' if done else 'timeout')
        atomic(directory/'result.json',result); atomic(directory/'complete.json',dict(valid=True,manifest_hash=manifest_hash)); return result
    finally: pool.release(case['suite'],case['task'])

