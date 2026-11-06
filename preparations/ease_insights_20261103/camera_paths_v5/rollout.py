"""Full diversity search rollout with frozen suite-specific policy budgets."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from preparations.idea1_v3.io import atomic, write_rows
from preparations.idea1_v2_1.runtime_client import official_init_states,rpc,input_digest
from .noise import model_seed


from .common import LIMITS
MAX_POLICY_ACTIONS = None
SETTLING_ACTIONS = 10


def episode(directory, port, case, config, pool, manifest_hash, *,
            max_policy_actions=MAX_POLICY_ACTIONS, settling_actions=SETTLING_ACTIONS,
            rpc_call=None, initial_input_gate=None, noise_repeat=0):
    """Complete reset and paired noise; suite policy budgets exclude settling.

    RPC injection is local to this call. No frozen shared module or LIMITS dict
    is modified. A callback gates the physical initial input before prediction.
    """
    if case['suite'] not in LIMITS or max_policy_actions != LIMITS[case['suite']] or settling_actions != SETTLING_ACTIONS:
        raise ValueError('Diversity search requires the frozen suite-specific policy action budget and10 settling actions')
    from .common import validate_condition
    validate_condition(config)
    call = rpc if rpc_call is None else rpc_call
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter(); timing={}; tick=begin
    env,definition,bddl,selected,created=pool.acquire(case['suite'],case['task'])
    timing['environment_acquire_s']=time.perf_counter()-tick
    try:
        vector=np.asarray(official_init_states(selected,case['task'])[case['init']])
        random.seed(7); np.random.seed(7); tick=time.perf_counter(); env.seed(7); env.reset(); obs=env.set_init_state(vector.copy())
        for _ in range(settling_actions): obs,_,_,_=env.step([0.]*6+[-1.])
        timing['reset_s']=time.perf_counter()-tick
        if float(env.env.control_freq)!=20.: raise ValueError('Control frequency changed')
        ctl=env.env.robots[0].controller
        scales={k:np.asarray(getattr(ctl,k)).tolist() for k in ('input_min','input_max','output_min','output_max')}
        if not np.allclose(scales['output_max'],[.05,.05,.05,.5,.5,.5]): raise ValueError('Controller scale changed')
        call(port,dict(op='configure',config=config)); protocol=call(port,dict(op='health'))['protocol']
        identity = dict(case=case,config=config,protocol=protocol,manifest_hash=manifest_hash,
            init_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),controller=scales,env_seed=7,settling=settling_actions,
            policy_action_limit=max_policy_actions,control_hz=20.,prompt=definition.language,noise_repeat=noise_repeat)
        atomic(directory/'identity.json', identity)
        infers=[]; telemetry=[]; arrays={}; steps=0; done=False
        def capture(action=None):
            telemetry.append(dict(step=steps,eef_pos=np.asarray(obs['robot0_eef_pos']).tolist(),eef_quat=np.asarray(obs['robot0_eef_quat']).tolist(),
                action=None if action is None else action.tolist(),sim_time_s=float(env.env.sim.data.time)))
        capture(); timing.update(rpc_s=0.,step_s=0.,ack_s=0.,shadow_s=0.,input_save_s=0.)
        while steps<max_policy_actions and not done:
            index=len(infers); generation_seed=model_seed(noise_repeat,case['suite'],case['task'],case['init'],index)
            model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
            request = dict(op='predict',obs=model_obs,instruction=definition.language,suite=case['suite'],seed=generation_seed)
            if index == 0 and initial_input_gate is not None:
                initial_input_gate(identity, request)
            tick=time.perf_counter(); r=call(port,request)
            elapsed=time.perf_counter()-tick; timing['rpc_s']+=elapsed
            actions=np.asarray(r.pop('actions'),np.float32); raw=np.asarray(r.pop('raw'),np.float32); small=r.pop('arrays'); r.pop('ok',None)
            if actions.shape!=(50,7) or raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(raw).all(): raise ValueError('Invalid model output')
            if r.get('execute_length') != 5:
                raise ValueError('Frozen execution prefix must remain R5')
            arrays[f'actions_{index}']=actions; arrays[f'raw_{index}']=raw
            for name,value in small.items(): arrays[f'{name}_{index}']=value
            record=dict(r,rpc_ms=elapsed*1000,replan=index,step=steps,executed_length=0,input_hash=input_digest(model_obs,definition.language))
            infers.append(record)
            if index==0:
                tick=time.perf_counter(); path=directory/f'input_{index:04d}.npz'
                np.savez_compressed(path,**model_obs,instruction=np.asarray(definition.language),seed=np.asarray(generation_seed))
                record.update(input_path=str(path.resolve()),input_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                timing['input_save_s']+=time.perf_counter()-tick
            executed=[]; poses=[]; tick=time.perf_counter()
            for action in actions[:min(r['execute_length'],max_policy_actions-steps)]:
                obs,_,done,_=env.step(action.tolist()); steps+=1; executed.append(action); poses.append(np.asarray(obs['robot0_eef_quat']).copy()); record['executed_length']+=1; capture(action)
                if done: break
            timing['step_s']+=time.perf_counter()-tick; tick=time.perf_counter()
            call(port,dict(op='ack',executed=np.asarray(executed),quaternions=np.asarray(poses))); timing['ack_s']+=time.perf_counter()-tick
        tick=time.perf_counter(); write_rows(directory/'infer.jsonl',infers); write_rows(directory/'telemetry.jsonl',telemetry); np.savez(directory/'arrays.npz',**arrays); timing['save_s']=time.perf_counter()-tick
        result=dict(valid=True,success=bool(done),case=case,condition=config['id'],config=config,manifest_hash=manifest_hash,
            actions=steps,predictions=len(infers),wall_s=time.perf_counter()-begin,timing=timing,policy_wall_s=sum(r['timing']['policy_wall_ms'] for r in infers)/1000,
            policy_action_limit=max_policy_actions,settling_actions=settling_actions,
            environment_reused=not created,counts=infers[-1]['counts'],termination='success' if done else 'timeout')
        atomic(directory/'result.json',result); atomic(directory/'complete.json',dict(valid=True,manifest_hash=manifest_hash)); return result
    finally: pool.release(case['suite'],case['task'])

