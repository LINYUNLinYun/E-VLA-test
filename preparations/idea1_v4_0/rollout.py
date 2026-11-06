"""Official-state closed loops, with explicit terminal selection accounting."""
import hashlib,random,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,write_rows
from preparations.idea1_v3_1.common import LIMITS
from preparations.idea1_v2_1.runtime_client import official_init_states,rpc,input_digest
from preparations.idea1_v2_2.common import seed


def episode(directory,port,case,config,pool,manifest_hash,pace_hz=0):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter();timing={};tick=begin
    env,definition,bddl,selected,created=pool.acquire(case['suite'],case['task'])
    timing['environment_acquire_s']=time.perf_counter()-tick
    try:
        vector=np.asarray(official_init_states(selected,case['task'])[case['init']])
        random.seed(7);np.random.seed(7);tick=time.perf_counter();env.seed(7);env.reset()
        obs=env.set_init_state(vector.copy())
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        timing['reset_s']=time.perf_counter()-tick
        assert float(env.env.control_freq)==20.
        ctl=env.env.robots[0].controller
        scales={k:np.asarray(getattr(ctl,k)).tolist() for k in ('input_min','input_max','output_min','output_max')}
        assert np.allclose(scales['output_max'],[.05,.05,.05,.5,.5,.5])
        rpc(port,dict(op='configure',config=config));protocol=rpc(port,dict(op='health'))['protocol']
        atomic(directory/'identity.json',dict(case=case,config=config,protocol=protocol,manifest_hash=manifest_hash,
            init_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            controller=scales,env_seed=7,settling=10,prompt=definition.language,pace_hz=pace_hz))
        infers=[];telemetry=[];arrays={};steps=0;done=False
        def capture(action=None):
            telemetry.append(dict(step=steps,eef_pos=np.asarray(obs['robot0_eef_pos']).tolist(),
                eef_quat=np.asarray(obs['robot0_eef_quat']).tolist(),action=None if action is None else action.tolist(),
                sim_time_s=float(env.env.sim.data.time)))
        capture();timing.update(rpc_s=0.,step_s=0.,pace_sleep_s=0.,ack_s=0.,input_save_s=0.,finish_s=0.)
        while steps<LIMITS[case['suite']] and not done:
            index=len(infers);generation_seed=seed(case['suite'],case['task'],case['init'],index)
            model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
            tick=time.perf_counter();r=rpc(port,dict(op='predict',obs=model_obs,instruction=definition.language,
                suite=case['suite'],seed=generation_seed));elapsed=time.perf_counter()-tick;timing['rpc_s']+=elapsed
            actions=np.asarray(r.pop('actions'),np.float32);raw=r.pop('raw');small=r.pop('arrays');r.pop('ok',None)
            assert actions.shape==(50,7) and raw.shape==(50,32) and np.isfinite(actions).all() and np.isfinite(raw).all()
            arrays[f'actions_{index}']=actions;arrays[f'raw_{index}']=raw
            for name,value in small.items():arrays[f'{name}_request{index}']=value
            record=dict(r,rpc_ms=elapsed*1000,replan=index,step=steps,executed_length=0,
                input_hash=input_digest(model_obs,definition.language))
            infers.append(record)
            # First two requests support prefix equivalence. A fixed small bank
            # from tasks 0/2 supports verifier diagnosis and deployment replay.
            save=index<2 or (case['task'] in (0,2) and index<15)
            if save:
                tick=time.perf_counter();path=directory/f'input_{index:04d}.npz'
                np.savez_compressed(path,**model_obs,instruction=np.asarray(definition.language),seed=np.asarray(generation_seed))
                record.update(input_path=str(path.resolve()),input_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                timing['input_save_s']+=time.perf_counter()-tick
            executed=[];poses=[];tick=time.perf_counter();sleep_s=0.
            for action in actions[:min(r['execute_length'],LIMITS[case['suite']]-steps)]:
                action_begin=time.perf_counter()
                obs,_,done,_=env.step(action.tolist());steps+=1;executed.append(action)
                poses.append(np.asarray(obs['robot0_eef_quat']).copy());record['executed_length']+=1;capture(action)
                if pace_hz:
                    remaining=1/pace_hz-(time.perf_counter()-action_begin)
                    if remaining>0:
                        sleep_begin=time.perf_counter();time.sleep(remaining);sleep_s+=time.perf_counter()-sleep_begin
                if done:break
            record['execution_wall_ms']=(time.perf_counter()-tick)*1000
            timing['step_s']+=time.perf_counter()-tick;timing['pace_sleep_s']+=sleep_s;tick=time.perf_counter()
            rpc(port,dict(op='ack',executed=np.asarray(executed),quaternions=np.asarray(poses)))
            ack_s=time.perf_counter()-tick
            timing['ack_s']+=ack_s
            record['ack_wall_ms']=ack_s*1000
        tick=time.perf_counter();terminal=rpc(port,dict(op='finish'));terminal.pop('ok',None)
        for name,value in terminal.pop('arrays').items():arrays[f'{name}_terminal']=value
        timing['finish_s']=time.perf_counter()-tick;atomic(directory/'terminal.json',terminal)
        assert terminal['counts']['selector']==terminal['counts']['completed_selector']
        tick=time.perf_counter();write_rows(directory/'infer.jsonl',infers);write_rows(directory/'telemetry.jsonl',telemetry)
        np.savez(directory/'arrays.npz',**arrays);timing['save_s']=time.perf_counter()-tick
        result=dict(valid=True,success=bool(done),case=case,condition=config['id'],config=config,manifest_hash=manifest_hash,
            actions=steps,predictions=len(infers),wall_s=time.perf_counter()-begin,timing=timing,
            policy_wall_s=sum(r['timing']['policy_wall_ms'] for r in infers)/1000+timing['finish_s'],
            capture_s=sum(r['timing']['selector_capture_ms'] for r in infers)/1000,
            counts=terminal['counts'],termination='success' if done else 'timeout',pace_hz=pace_hz,
            environment_reused=not created)
        atomic(directory/'result.json',result);atomic(directory/'complete.json',dict(valid=True,manifest_hash=manifest_hash))
        return result
    finally:pool.release(case['suite'],case['task'])
