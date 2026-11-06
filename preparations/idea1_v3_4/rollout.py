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
from preparations.idea1_v2_1.runtime_client import official_init_states,rpc,input_digest
from preparations.idea1_v2_2.common import seed


def episode(directory,port,case,config,pool,manifest_hash):
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
        ctl=env.env.robots[0].controller
        scales={k:np.asarray(getattr(ctl,k)).tolist() for k in ('input_min','input_max','output_min','output_max')}
        if not np.allclose(scales['output_max'],[.05,.05,.05,.5,.5,.5]): raise ValueError('Controller scale changed')
        rpc(port,dict(op='configure',config=config)); protocol=rpc(port,dict(op='health'))['protocol']
        atomic(directory/'identity.json',dict(case=case,config=config,protocol=protocol,manifest_hash=manifest_hash,
            init_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),controller=scales,env_seed=7,settling=10,
            prompt=definition.language))
        infers=[]; telemetry=[]; arrays={}; steps=0; done=False
        def capture(action=None):
            telemetry.append(dict(step=steps,eef_pos=np.asarray(obs['robot0_eef_pos']).tolist(),eef_quat=np.asarray(obs['robot0_eef_quat']).tolist(),
                action=None if action is None else action.tolist(),sim_time_s=float(env.env.sim.data.time)))
        capture(); timing.update(rpc_s=0.,step_s=0.,ack_s=0.,shadow_s=0.,input_save_s=0.)
        save_all=config['id']=='full_d10' and case['init']==14
        while steps<LIMITS[case['suite']] and not done:
            index=len(infers); generation_seed=seed(case['suite'],case['task'],case['init'],index)
            model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
            tick=time.perf_counter(); r=rpc(port,dict(op='predict',obs=model_obs,instruction=definition.language,suite=case['suite'],seed=generation_seed))
            elapsed=time.perf_counter()-tick; timing['rpc_s']+=elapsed
            actions=np.asarray(r.pop('actions'),np.float32); raw=r.pop('raw'); small=r.pop('arrays'); r.pop('ok',None)
            if actions.shape!=(50,7) or raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(raw).all(): raise ValueError('Invalid model output')
            arrays[f'actions_{index}']=actions; arrays[f'raw_{index}']=raw
            for name,value in small.items(): arrays[f'{name}_{index}']=value
            record=dict(r,rpc_ms=elapsed*1000,replan=index,step=steps,executed_length=0,input_hash=input_digest(model_obs,definition.language))
            infers.append(record)
            selected=r.get('shadow_selection',{}).get('selected',False)
            if save_all or selected or index==0:
                tick=time.perf_counter(); path=directory/f'input_{index:04d}.npz'
                np.savez_compressed(path,**model_obs,instruction=np.asarray(definition.language),seed=np.asarray(generation_seed))
                record.update(input_path=str(path.resolve()),input_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                timing['input_save_s']+=time.perf_counter()-tick
            if selected:
                tick=time.perf_counter(); shadow=rpc(port,dict(op='shadow')); shadow.pop('ok',None)
                shadow_arrays=shadow.pop('arrays'); np.savez_compressed(directory/f'shadow_{index:04d}.npz',**shadow_arrays)
                shadow['input_path']=record['input_path']; shadow['input_file_sha256']=record['input_file_sha256']
                atomic(directory/f'shadow_{index:04d}.json',shadow)
                timing['shadow_s']+=time.perf_counter()-tick
            executed=[]; poses=[]; tick=time.perf_counter()
            for action in actions[:min(r['execute_length'],LIMITS[case['suite']]-steps)]:
                obs,_,done,_=env.step(action.tolist()); steps+=1; executed.append(action); poses.append(np.asarray(obs['robot0_eef_quat']).copy()); record['executed_length']+=1; capture(action)
                if done: break
            timing['step_s']+=time.perf_counter()-tick; tick=time.perf_counter()
            rpc(port,dict(op='ack',executed=np.asarray(executed),quaternions=np.asarray(poses))); timing['ack_s']+=time.perf_counter()-tick
        tick=time.perf_counter(); write_rows(directory/'infer.jsonl',infers); write_rows(directory/'telemetry.jsonl',telemetry); np.savez(directory/'arrays.npz',**arrays); timing['save_s']=time.perf_counter()-tick
        result=dict(valid=True,success=bool(done),case=case,condition=config['id'],config=config,manifest_hash=manifest_hash,
            actions=steps,predictions=len(infers),wall_s=time.perf_counter()-begin,timing=timing,policy_wall_s=sum(r['timing']['policy_wall_ms'] for r in infers)/1000,
            environment_reused=not created,counts=infers[-1]['counts'],termination='success' if done else 'timeout')
        atomic(directory/'result.json',result); atomic(directory/'complete.json',dict(valid=True,manifest_hash=manifest_hash)); return result
    finally: pool.release(case['suite'],case['task'])


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--condition',required=True); p.add_argument('--output',required=True); p.add_argument('--port',required=True,type=int); p.add_argument('--shard',type=int,default=0); p.add_argument('--shards',type=int,default=3); p.add_argument('--stage',choices=['screen','expand'],default='screen'); a=p.parse_args()
    m=json.loads(Path(a.manifest).read_text()); cfg=next(c for c in m['conditions'] if c['id']==a.condition)
    owned=[c for c in m['cases'][cfg['id']] if c['task']%a.shards==a.shard and ((c['init']==14)==(a.stage=='screen'))]
    # Consecutive init reuse within a task; task keys and RNG independent of order.
    owned.sort(key=lambda c:(c['task'],c['suite'],c['init']))
    pool=Environments(reuse=True)
    try:
        for case in owned:
            directory=Path(a.output)/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            with lock(str(directory)+'.lock'):
                if (directory/'complete.json').exists(): continue
                for attempt in range(2):
                    if directory.exists(): directory.rename(directory.with_name(directory.name+'.invalid.'+uuid.uuid4().hex))
                    try:
                        result=episode(directory,a.port,case,cfg,pool,m['manifest_hash'])
                        print(json.dumps(result),flush=True); break
                    except Exception as exc:
                        directory.mkdir(parents=True,exist_ok=True)
                        atomic(directory/'technical_error.json',dict(error=repr(exc),attempt=attempt+1))
                        if attempt==1: raise
    finally: pool.close()

if __name__=='__main__': main()
