"""Native LIBERO episode, consolidated arrays, fixed video keys, lightweight timing."""
from pathlib import Path
import os
import time
import numpy as np

from preparations.idea1_v2_1.runtime_client import LIMITS, official_init_states, rpc, telemetry_helpers
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3.io import atomic, write_rows, digest


def run(directory, port, row, config, profile_path, collect_inputs=False):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from preparations.idea1_v3.io import append
    import imageio.v2 as imageio
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    selected=benchmark.get_benchmark_dict()[row['suite']](); definition=selected.get_task(row['task'])
    initial_states=official_init_states(selected,row['task'])
    bddl=Path(get_libero_path('bddl_files'))/definition.problem_folder/definition.bddl_file
    clock=time.perf_counter(); timing={}; ticks=time.perf_counter()
    env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    timing['environment_create_s']=time.perf_counter()-ticks
    writer=None
    replay=telemetry_helpers(); states=[]; infers=[]; arrays={}; inputs=[]
    try:
        ticks=time.perf_counter(); env.seed(7); env.reset(); obs=env.set_init_state(initial_states[row['init']])
        for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
        timing['reset_s']=time.perf_counter()-ticks
        if float(env.env.control_freq)!=20.: raise ValueError('native control frequency changed')
        rpc(port,dict(op='reset')); protocol=rpc(port,dict(op='health'))['protocol']
        primary,target=replay._goal_object_names(env.env)
        initial={name: replay._object_state(env.env,name).get('pos') for name in primary}
        video=row['init']==6 and row['task'] in (0,5)
        if video: writer=imageio.get_writer(str(directory/'video.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=2)
        def capture(step,action):
            tick=time.perf_counter()
            state=replay.snapshot(env,obs,step=step,when='initial' if step==0 else 'post_action',
                action=action,primary_names=primary,target_names=target,initial_positions=initial)
            state.update(sim_time_s=float(env.env.sim.data.time),wall_timestamp_s=time.time())
            states.append(state)
            if writer: writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
            timing['telemetry_and_video_enqueue_s']=timing.get('telemetry_and_video_enqueue_s',0)+time.perf_counter()-tick
        capture(0,None); steps=0; done=False
        while steps<LIMITS[row['suite']] and not done:
            replan=len(infers); model_obs={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
            generation_seed=seed(row['suite'],row['task'],row['init'],replan)
            tick=time.perf_counter()
            response=rpc(port,dict(op='predict',obs=model_obs,instruction=definition.language,
                suite=row['suite'],seed=generation_seed))
            rpc_ms=(time.perf_counter()-tick)*1000
            actions=np.asarray(response.pop('actions')); raw=response.pop('raw'); small=response.pop('arrays')
            length=min(response['execute_length'],LIMITS[row['suite']]-steps)
            if not 0<length<=5 or actions.shape!=(50,7) or not np.isfinite(actions).all(): raise ValueError('invalid execution proposal')
            if collect_inputs and len(inputs)<5:
                # P0 real observations; never generate fake phase coverage.
                inputs.append(dict(obs=model_obs,instruction=definition.language,suite=row['suite'],seed=generation_seed))
            record=dict(response,replan=replan,step=steps,rpc_roundtrip_ms=rpc_ms,executed_length=0)
            record.pop('ok',None); infers.append(record)
            arrays[f'actions_{replan}']=actions; arrays[f'normalized_{replan}']=raw
            for name,value in small.items(): arrays[f'{name}_{replan}']=value
            executed=[]
            tick=time.perf_counter()
            for action in actions[:length]:
                obs,_,done,_=env.step(action.tolist()); steps+=1; executed.append(action)
                record['executed_length']+=1; capture(steps,action.tolist())
                if done: break
            timing['environment_step_and_capture_s']=timing.get('environment_step_and_capture_s',0)+time.perf_counter()-tick
            rpc(port,dict(op='ack',executed=np.asarray(executed)))
            append(profile_path,dict(kind='request',key=row['key'],replan=replan,rpc_ms=rpc_ms,**record['timing']))
        replay.assign_semantic_phases(states)
        tick=time.perf_counter()
        if writer: writer.close(); writer=None
        timing['video_drain_s']=time.perf_counter()-tick
        tick=time.perf_counter()
        write_rows(directory/'telemetry.jsonl',states); write_rows(directory/'infer.jsonl',infers)
        with (directory/'arrays.npz').open('xb') as stream:
            save=np.savez if row['execution']=='B' else np.savez_compressed
            save(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        if collect_inputs:
            import pickle
            with (directory/'profile_inputs.pkl').open('xb') as stream:
                pickle.dump(inputs,stream,protocol=5); stream.flush(); os.fsync(stream.fileno())
        timing['consolidated_save_s']=time.perf_counter()-tick
        if len(states)!=steps+1 or any(not np.isfinite(x['eef_pos']).all() for x in states): raise ValueError('invalid true XYZ records')
        if config['experiment']=='E4' and any(x['counts']['full'] or x['counts']['verify'] for x in infers): raise ValueError('E4 invoked Full or verifier')
        result=dict(row,config=config,protocol=protocol,success=bool(done),valid=True,actions=steps,
            predictions=len(infers),termination='success' if done else 'timeout',wall_s=time.perf_counter()-clock,
            timings=timing,video='video.mp4' if video else None,seed_rule='idea1-v22/1701 suite/task/init/replan',
            init_vector_hash=digest(np.asarray(initial_states[row['init']]).tolist()))
        atomic(directory/'result.json',result)
        # Completion follows all episode writes and fsync, including directory durability.
        atomic(directory/'complete.json',dict(key=row['key'],valid=True,success=bool(done)))
        append(profile_path,dict(kind='episode',**result))
        return result
    finally:
        if writer: writer.close()
        env.close()
