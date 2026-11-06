"""Native-H branch episode; no assumptions that pi0.5 has H50."""
import os
from pathlib import Path
import time
import numpy as np

from preparations.idea1_v2_1.runtime_client import LIMITS,official_init_states,rpc,telemetry_helpers
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3.io import atomic,append,write_rows,digest


def run(directory,port,row,profile_path):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import imageio.v2 as imageio
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    clock=time.perf_counter(); timers={}; suite=benchmark.get_benchmark_dict()[row['suite']]()
    task=suite.get_task(row['task']); initial_states=official_init_states(suite,row['task'])
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    tick=time.perf_counter(); env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
    timers['environment_create_s']=time.perf_counter()-tick
    replay=telemetry_helpers(); writer=None; states=[]; infers=[]; arrays={}
    try:
        tick=time.perf_counter(); env.seed(7); env.reset(); obs=env.set_init_state(initial_states[row['init']])
        for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
        timers['reset_s']=time.perf_counter()-tick
        if float(env.env.control_freq)!=20.: raise ValueError('control frequency changed')
        controller=env.env.robots[0].controller
        controller_protocol={name:np.asarray(getattr(controller,name)).tolist()
            for name in ('input_min','input_max','output_min','output_max')}
        rpc(port,dict(op='reset')); protocol=rpc(port,dict(op='health'))['protocol']; horizon=protocol['H']
        primary,target=replay._goal_object_names(env.env)
        initial={name:replay._object_state(env.env,name).get('pos') for name in primary}
        video=row['task'] in (0,5) and row['init']==6
        if video: writer=imageio.get_writer(str(directory/'video.mp4'),fps=20,codec='libx264',pixelformat='yuv420p',macro_block_size=2)
        def capture(step,action):
            tick=time.perf_counter(); state=replay.snapshot(env,obs,step=step,when='initial' if step==0 else 'post_action',
                action=action,primary_names=primary,target_names=target,initial_positions=initial)
            state.update(sim_time_s=float(env.env.sim.data.time),wall_timestamp_s=time.time())
            if action is not None:
                state['controller_command_sent']=action
                state['controller_command_after_input_clip']=np.r_[np.clip(np.asarray(action)[:6],
                    controller.input_min,controller.input_max),action[6]].tolist()
            states.append(state)
            if writer: writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))
            timers['telemetry_video_enqueue_s']=timers.get('telemetry_video_enqueue_s',0)+time.perf_counter()-tick
        capture(0,None); steps=0; done=False
        while steps<LIMITS[row['suite']] and not done:
            replan=len(infers); model_obs={key:np.asarray(value) for key,value in obs.items() if isinstance(value,np.ndarray)}
            tick=time.perf_counter(); response=rpc(port,dict(op='predict',obs=model_obs,instruction=task.language,
                suite=row['suite'],seed=seed(row['suite'],row['task'],row['init'],replan)))
            rpc_ms=(time.perf_counter()-tick)*1000
            actions=np.asarray(response.pop('actions'),np.float32); raw=response.pop('raw'); small=response.pop('arrays')
            if actions.shape!=(horizon,7) or not np.isfinite(actions).all(): raise ValueError('native H/finite action mismatch')
            length=min(response['execute_length'],LIMITS[row['suite']]-steps)
            if not 0<length<=5: raise ValueError('invalid execute length')
            record=dict(response,step=steps,replan=replan,rpc_roundtrip_ms=rpc_ms,executed_length=0); record.pop('ok',None)
            infers.append(record); arrays[f'actions_{replan}']=actions; arrays[f'normalized_{replan}']=raw
            for name,value in small.items(): arrays[f'{name}_{replan}']=value
            executed=[]; tick=time.perf_counter()
            for command in actions[:length]:
                obs,_,done,_=env.step(command.tolist()); steps+=1; executed.append(command)
                record['executed_length']+=1; capture(steps,command.tolist())
                if done: break
            timers['environment_step_and_capture_s']=timers.get('environment_step_and_capture_s',0)+time.perf_counter()-tick
            rpc(port,dict(op='ack',executed=np.asarray(executed,np.float32)))
            append(profile_path,dict(kind='request',key=row['key'],replan=replan,rpc_ms=rpc_ms,**record['timing']))
        replay.assign_semantic_phases(states); tick=time.perf_counter()
        if writer: writer.close(); writer=None
        timers['video_drain_s']=time.perf_counter()-tick; tick=time.perf_counter()
        write_rows(directory/'telemetry.jsonl',states); write_rows(directory/'infer.jsonl',infers)
        with (directory/'arrays.npz').open('xb') as stream:
            np.savez_compressed(stream,**arrays); stream.flush(); os.fsync(stream.fileno())
        timers['save_s']=time.perf_counter()-tick
        if len(states)!=steps+1 or any(not np.isfinite(state['eef_pos']).all() for state in states): raise ValueError('invalid XYZ records')
        result=dict(row,protocol=protocol,controller_protocol=controller_protocol,success=bool(done),valid=True,actions=steps,predictions=len(infers),
            termination='success' if done else 'timeout',wall_s=time.perf_counter()-clock,timings=timers,
            video='video.mp4' if video else None,init_vector_hash=digest(np.asarray(initial_states[row['init']]).tolist()))
        atomic(directory/'result.json',result); atomic(directory/'complete.json',dict(key=row['key'],valid=True,success=bool(done)))
        append(profile_path,dict(kind='episode',**result)); return result
    finally:
        if writer: writer.close()
        env.close()
