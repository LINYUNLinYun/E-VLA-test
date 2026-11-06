"""Full native pi0.5 rollouts with explicit per-request noise and rich trace."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import random
import time

import numpy as np

from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v3_1.server import rpc
from preparations.idea1_v2_1.runtime_client import official_init_states, telemetry_helpers, input_digest
from .common import LIMITS, PROTOCOL, atomic_json, episode_path, request_seed, write_jsonl


def run_episode(root, port, suite, task, init, pool):
    directory = episode_path(root,suite,task,init)
    if (directory/'complete.json').exists():
        return
    directory.mkdir(parents=True,exist_ok=False)
    begin = time.perf_counter()
    env, definition, bddl, selected, created = pool.acquire(suite,task)
    video_frames = []
    try:
        vector = np.asarray(official_init_states(selected,task)[init])
        random.seed(7); np.random.seed(7); env.seed(7); env.reset()
        obs = env.set_init_state(vector.copy())
        for _ in range(10):
            obs,_,_,_ = env.step([0.]*6+[-1.])
        controller = env.env.robots[0].controller
        scales = {name:np.asarray(getattr(controller,name)).tolist()
                  for name in ('input_min','input_max','output_min','output_max')}
        if float(env.env.control_freq)!=20. or not np.allclose(scales['output_max'],[.05]*3+[.5]*3):
            raise ValueError('Native control protocol differs')
        health = rpc(port,dict(op='health'))
        if health['protocol'] != PROTOCOL:
            raise ValueError('Service protocol mismatch')
        atomic_json(directory/'identity.json',dict(protocol=PROTOCOL,suite=suite,task=task,init=init,
                    service=dict(pid=health['pid'],cuda_visible_devices=health['cuda_visible_devices']),
                    render=dict(physical_gpu=os.environ.get('IDEA1_RENDER_PHYSICAL_GPU'),
                                egl_index=os.environ.get('MUJOCO_EGL_DEVICE_ID')),
                    instruction=definition.language,initial_state_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
                    bddl=str(bddl),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),controller=scales))
        helpers = telemetry_helpers()
        primary,target = helpers._goal_object_names(env.env)
        initial_positions = {name:helpers._object_state(env.env,name).get('pos') for name in primary}
        trace, infers = [], []

        def capture(step,action):
            row = helpers.snapshot(env,obs,step=step,when='initial' if step==0 else 'post_action',
                    action=action,primary_names=primary,target_names=target,initial_positions=initial_positions)
            for name in primary:
                try:
                    gripper,obj = env.env.robots[0].gripper,env.env.get_object(name)
                    row['robot_contacts'][name] = bool(env.env.check_contact(gripper,obj))
                    row['robot_grasps'][name] = bool(env.env._check_grasp(gripper=gripper,object_geoms=obj))
                except Exception:
                    row['robot_contacts'][name] = row['robot_grasps'][name] = None
            row.update(sim_time_s=float(env.env.sim.data.time),executed_action=action)
            trace.append(row)
            # Preselected task0/init0 videos support phase-label inspection.
            if task==0 and init==0:
                video_frames.append(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]))

        capture(0,None)
        steps,done = 0,False
        while steps<LIMITS[suite] and not done:
            request = len(infers)
            model_obs = {k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
            seed = request_seed(suite,task,init,request)
            result = rpc(port,dict(op='predict',obs=model_obs,instruction=definition.language,seed=seed))
            actions = np.asarray(result['actions'],np.float32)
            if actions.shape!=(10,7) or not np.isfinite(actions).all():
                raise ValueError('Invalid native H10 action output')
            if result['input_hash']!=input_digest(model_obs,definition.language):
                raise ValueError('Policy observation identity mismatch')
            request_path = directory/f'request_{request:04d}.npz'
            np.savez_compressed(request_path,**{f'obs__{k}':v for k,v in model_obs.items()},
                    instruction=np.asarray(definition.language),noise=result['noise'],
                    raw_actions=result['raw_actions'],actions=actions,seed=np.asarray(seed))
            row = dict(request=request,step=steps,path=str(request_path),seed=seed,
                       input_hash=result['input_hash'],noise_hash=result['noise_hash'],
                       policy_ms=result['policy_ms'],executed_length=0)
            infers.append(row)
            for action in actions[:min(5,LIMITS[suite]-steps)]:
                obs,_,done,_ = env.step(action.tolist())
                steps += 1; row['executed_length'] += 1
                capture(steps,action.tolist())
                if done:
                    break
        write_jsonl(directory/'trace.jsonl',trace)
        write_jsonl(directory/'infer.jsonl',infers)
        if video_frames:
            import imageio.v2 as imageio
            imageio.mimwrite(directory/'video.mp4',video_frames,fps=20,codec='libx264',macro_block_size=2)
        result = dict(valid=True,success=bool(done),suite=suite,task=task,init=init,
                      steps=steps,requests=len(infers),wall_s=time.perf_counter()-begin,
                      video=bool(video_frames),native_parity=rpc(port,dict(op='health'))['parity'])
        atomic_json(directory/'result.json',result)
        atomic_json(directory/'complete.json',dict(valid=True,protocol=PROTOCOL))
        print(result,flush=True)
        return result
    except Exception as exc:
        atomic_json(directory/'technical_error.json',dict(error=repr(exc)))
        raise
    finally:
        pool.release(suite,task)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--cases',nargs='+',required=True,help='suite/task/init keys')
    args=p.parse_args()
    pool=Environments(reuse=False)
    try:
        for key in args.cases:
            suite,task,init=key.split('/')
            run_episode(args.root,args.port,suite,int(task),int(init),pool)
    finally:
        pool.close()


if __name__=='__main__':
    main()
