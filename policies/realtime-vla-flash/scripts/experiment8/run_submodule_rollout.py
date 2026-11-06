#!/usr/bin/env python3
"""Paired-noise diagnostic rollout with N+1 measured states and RGB H264 frames."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/spec'))
sys.path.insert(0, str(ROOT / 'scripts/experiment6'))
import spec_client_libero as client_base
from libero.libero import benchmark
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from trajectory_replay import snapshot, _goal_object_names, _object_state, assign_semantic_phases


def mask(value):
    result = tuple(int(item) for item in value.split(',') if item.strip())
    if len(set(result)) != len(result) or any(index < 1 or index > 17 for index in result):
        raise argparse.ArgumentTypeError('Unique layer numbers must be in 1..17')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('libero_10','libero_90'), required=True)
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--official-init-index', type=int, default=0)
    parser.add_argument('--init-state', type=Path)
    parser.add_argument('--noise-seed', type=int, default=12345)
    parser.add_argument('--env-seed', type=int, default=7)
    parser.add_argument('--warmup-steps', type=int, default=10)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--replan-steps', type=int, default=12)
    parser.add_argument('--skip-attn-layers', type=mask, default=())
    parser.add_argument('--skip-ffn-layers', type=mask, default=())
    parser.add_argument('--condition-label', required=True)
    parser.add_argument('--expected-runtime', choices=('triton_eager','jax_reference'), default='triton_eager')
    parser.add_argument('--host', default='localhost')
    parser.add_argument('--port', type=int, default=8028)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite rollout directory')
    if not 1 <= args.replan_steps <= 50 or args.warmup_steps < 0:
        raise ValueError('Invalid action execution configuration')
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task)
    if args.init_state:
        initial = np.load(args.init_state, allow_pickle=False)
    else:
        initial = np.asarray(suite.get_task_init_states(args.task)[args.official_init_index])
    initial = np.asarray(initial, dtype=np.float64)
    initial_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    semantic_key = args.suite + '/' + task.name
    env, prompt = client_base._get_libero_env(task, client_base.LIBERO_ENV_RESOLUTION, args.env_seed)
    client = WebsocketClientPolicy(args.host, args.port)
    states, inferences = [], []
    success = False
    args.output_dir.mkdir(parents=True)
    try:
        env.reset()
        obs = env.set_init_state(initial)
        for _ in range(args.warmup_steps):
            obs, _, _, _ = env.step(client_base.LIBERO_DUMMY_ACTION)
        raw = env.env
        fps = float(raw.control_freq)
        if fps <= 0:
            raise ValueError('Invalid simulator control frequency')
        primary, targets = _goal_object_names(raw)
        initial_positions = {name:_object_state(raw,name).get('pos') for name in primary}
        video = args.output_dir/'rollout.mp4'
        def record(index, action, infer_id, offset, metadata):
            item = snapshot(env,obs,step=index,when='initial' if index == 0 else 'post_action',
                action=action,primary_names=primary,target_names=targets,initial_positions=initial_positions)
            item.update(state_index=index,action_index=index-1 if index else None,
                sim_time_s=float(raw.sim.data.time),video_frame_index=index,
                infer_id=infer_id,action_offset=offset,noise_sha256=metadata.get('noise_sha256') if metadata else None,
                done_after_step=bool(success),condition_label=args.condition_label)
            states.append(item)
            frame = np.ascontiguousarray(obs['agentview_image'][::-1,::-1])
            if frame.dtype != np.uint8:
                raise ValueError('Simulator RGB frame must be uint8')
            writer.append_data(frame)
        max_steps = args.max_steps or (520 if args.suite=='libero_10' else 400)
        with imageio.get_writer(str(video),fps=fps,codec='libx264',pixelformat='yuv420p',
                                ffmpeg_params=['-crf','20','-movflags','+faststart']) as writer:
            record(0,None,None,None,None)
            step, infer_id = 0, 0
            while step < max_steps and not success:
                if any(obs[key].dtype != np.uint8 for key in ('agentview_image','robot0_eye_in_hand_image')):
                    raise ValueError('Both simulator camera streams must be uint8 for canonical preprocessing')
                image = client_base.image_tools.convert_to_uint8(client_base.image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224))
                wrist = client_base.image_tools.convert_to_uint8(client_base.image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224))
                robot_state = np.concatenate((obs['robot0_eef_pos'],client_base._quat2axisangle(obs['robot0_eef_quat']),obs['robot0_gripper_qpos']))
                response = client.infer({'observation/image':image,'observation/wrist_image':wrist,
                    'observation/state':robot_state,'prompt':prompt,'__semantic_task_key__':semantic_key,
                    '__init_state_sha256__':initial_hash,'__noise_seed__':args.noise_seed,'__infer_id__':infer_id})
                metadata = response['diagnostic_metadata']
                actual_runtime = metadata.get('runtime', 'triton_eager' if metadata.get('eager_diagnostic') else None)
                if actual_runtime != args.expected_runtime:
                    raise ValueError('Server runtime does not match expected implementation')
                if (metadata['skip_attn_layers'] != list(args.skip_attn_layers) or
                    metadata['skip_ffn_layers'] != list(args.skip_ffn_layers) or metadata['denoise_steps'] != 10 or
                    metadata['action_space'] != 'pi0_libero_unnormalized_7d_policy_command'):
                    raise ValueError('Server intervention configuration mismatch')
                actions = np.asarray(response['actions'])
                if actions.shape != (50,7) or not np.isfinite(actions).all():
                    raise ValueError('Expected finite transformed [50,7] policy commands')
                inferences.append({'infer_id':infer_id,'start_action_index':step,'metadata':metadata,'actions':actions.tolist(),
                    'input_image_dtypes':{'agentview':str(image.dtype),'wrist':str(wrist.dtype)}})
                for offset, action in enumerate(actions[:args.replan_steps]):
                    if step >= max_steps:
                        break
                    obs, _, success, _ = env.step(action.tolist())
                    step += 1
                    record(step,action.tolist(),infer_id,offset,metadata)
                    if success:
                        break
                infer_id += 1
        assign_semantic_phases(states)
        for name, rows in [('telemetry.jsonl',states),('inferences.jsonl',inferences)]:
            (args.output_dir/name).write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
        xyz = np.asarray([state['eef_pos'] for state in states],dtype=float)
        np.savez_compressed(args.output_dir/'trajectory.npz',eef_xyz=xyz,
                            sim_time_s=np.asarray([state['sim_time_s'] for state in states]))
        with (args.output_dir/'trajectory.csv').open('w',newline='') as stream:
            writer_csv = csv.writer(stream)
            writer_csv.writerow(['state_index','sim_time_s','eef_x','eef_y','eef_z','video_frame_index'])
            writer_csv.writerows([state['state_index'],state['sim_time_s'],*state['eef_pos'],state['video_frame_index']] for state in states)
        report = {'condition_label':args.condition_label,'semantic_task_key':semantic_key,'success':bool(success),
            'executed_actions':len(states)-1,'state_count':len(states),'video_frame_count':len(states),
            'coordinate_frame':'MuJoCo world','units':'m','control_frequency_hz':fps,
            'init_state_sha256':initial_hash,'init_source':str(args.init_state) if args.init_state else 'official',
            'official_init_index':args.official_init_index,'noise_seed':args.noise_seed,'env_seed':args.env_seed,
            'warmup_steps':args.warmup_steps,'replan_steps':args.replan_steps,
            'skip_attn_layers':list(args.skip_attn_layers),'skip_ffn_layers':list(args.skip_ffn_layers),
            'denoise_steps':10,'runtime':args.expected_runtime,
            'termination_reason':'success' if success else 'step_limit'}
        (args.output_dir/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
