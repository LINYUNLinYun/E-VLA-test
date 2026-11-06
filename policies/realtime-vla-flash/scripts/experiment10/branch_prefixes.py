"""Py3.8 real-simulator validation branches from replayed init0 phase anchors.

No qpos-only clone, no physical rollback, and no Full18 recovery claim.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/spec'))
from scripts.experiment10.common import RESULTS, SUITES, HORIZONS, dump_json, json_lines, sha256


def validate_video(path, frames):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-count_frames', '-show_entries',
                            'stream=codec_name,pix_fmt,nb_read_frames', '-of', 'json', str(path)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
    stream = json.loads(result.stdout)['streams'][0]
    if (stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p' or
        int(stream['nb_read_frames']) != frames):
        raise RuntimeError('Branch video codec/frame alignment validation failed')
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(path),
                    '-f', 'null', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return stream


def rebuild(suite, meta, saved):
    import spec_client_libero as client
    if meta['episode_idx'] != 0:
        raise RuntimeError('This initial branch implementation requires init0; no controller lifecycle shortcut')
    trace = json_lines(Path(meta['source_trace']))
    telemetry = json_lines(Path(meta['source_telemetry']))
    if sha256(meta['source_trace']) != meta['source_trace_sha256']:
        raise RuntimeError('Baseline history changed')
    actions = [r['executed_action'] for r in trace if r.get('executed_action') is not None]
    np.random.seed(7)
    env, prompt = client._get_libero_env(suite.get_task(meta['task_id']), client.LIBERO_ENV_RESOLUTION, 7)
    try:
        env.reset()
        obs = env.set_init_state(suite.get_task_init_states(meta['task_id'])[0])
        for _ in range(10):
            obs, _, _, _ = env.step(client.LIBERO_DUMMY_ACTION)
        for step in range(meta['trajectory_step']+1):
            error = np.max(np.abs(np.asarray(obs['robot0_eef_pos'])-telemetry[step]['eef_pos']))
            if not np.isfinite(error) or error > 1e-5:
                raise RuntimeError('Branch history parity failure at step '+str(step))
            if step < meta['trajectory_step']:
                obs, _, done, _ = env.step(actions[step])
                if done:
                    raise RuntimeError('Terminal state before branch anchor')
        def image(value):
            return client.image_tools.convert_to_uint8(client.image_tools.resize_with_pad(
                np.ascontiguousarray(value[::-1, ::-1]), 224, 224))
        state = np.concatenate((obs['robot0_eef_pos'], client._quat2axisangle(obs['robot0_eef_quat']),
                                obs['robot0_gripper_qpos'])).astype(np.float32)
        if (not np.array_equal(image(obs['agentview_image']), saved['agent_image']) or
            not np.array_equal(image(obs['robot0_eye_in_hand_image']), saved['wrist_image']) or
            not np.array_equal(state, saved['robot_state']) or str(prompt) != str(saved['prompt'].item())):
            raise RuntimeError('Rebuilt branch observation differs from the model input')
        return env, obs, actions, telemetry
    except Exception:
        env.close()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    from libero.libero import benchmark
    from scripts.experiment6.trajectory_replay import snapshot, _goal_object_names, _object_state
    import imageio
    suite = benchmark.get_benchmark_dict()[args.suite]()
    entries = json.loads((RESULTS/'observations'/args.suite/'manifest.json').read_text())['observations']
    entries = [e for e in entries if e['split'] == 'validation' and e['episode_idx'] == 0 and e['ordinal'] == 0]
    conditions = ['D3', 'V16_D10', 'V13_D10', 'V13_D3']
    output = RESULTS/'branches'/args.suite
    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for entry in entries:
        observation_path = Path(entry['path'])
        if sha256(observation_path) != entry['sha256']:
            raise RuntimeError('Branch input changed')
        with np.load(observation_path, allow_pickle=False) as data:
            saved = {k: data[k] for k in data.files}
        meta = json.loads(str(saved['metadata_json'].item()))
        shadow = RESULTS/'shadow'/args.suite/(meta['sample_id']+'.npz')
        with np.load(shadow, allow_pickle=False) as data:
            info = json.loads(str(data['metadata_json'].item()))
            reference = data['baseline_commands'][0]
            variants = data['candidate_commands'][0]
        if info['observation_sha256'] != entry['sha256'] or info['noise_seeds'][0] != 12345:
            raise RuntimeError('Shadow input/noise mismatch')
        labels = [c[0] for c in info['conditions']]
        commands = {'Full18_D10': reference, **{c: variants[labels.index(c)] for c in conditions}}
        # Check the original next action before any model-generated branch.
        env, obs, history, telemetry = rebuild(suite, meta, saved)
        try:
            obs, _, _, _ = env.step(history[meta['trajectory_step']])
            if np.max(np.abs(np.asarray(obs['robot0_eef_pos'])-telemetry[meta['trajectory_step']+1]['eef_pos'])) > 1e-5:
                raise RuntimeError('Reconstructed controller next-step parity failure')
        finally:
            env.close()
        for horizon in HORIZONS:
            for name, action_sequence in commands.items():
                prefix = output/(meta['sample_id']+'_'+name+'_h'+str(horizon))
                record = prefix.with_suffix('.json')
                provenance = {'observation_sha256': entry['sha256'], 'shadow_sha256': sha256(shadow),
                              'condition': name, 'horizon': horizon, 'noise_seed': 12345,
                              'branch_code_sha256': sha256(__file__)}
                if record.exists():
                    previous = json.loads(record.read_text())
                    if previous['provenance'] != provenance:
                        raise RuntimeError('Branch resume mismatch')
                    if previous.get('video_sha256') != sha256(previous['video']):
                        raise RuntimeError('Branch video changed or lacks provenance')
                    validate_video(Path(previous['video']), len(previous['states']))
                    completed.append(str(record)); continue
                env, obs, _, _ = rebuild(suite, meta, saved)
                states, frames, done = [], [], False
                try:
                    primary, targets = _goal_object_names(env.env)
                    initial = {n: _object_state(env.env, n).get('pos') for n in primary}
                    for step in range(horizon+1):
                        states.append(snapshot(env, obs, step=step, when='initial' if step == 0 else 'post_action',
                                     action=None if step == 0 else action_sequence[step-1].tolist(),
                                     primary_names=primary, target_names=targets, initial_positions=initial))
                        frames.append(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
                        if done or step == horizon:
                            break
                        obs, _, done, _ = env.step(action_sequence[step].tolist())
                    video = prefix.with_suffix('.mp4')
                    imageio.mimwrite(video, frames, fps=20, codec='libx264', pixelformat='yuv420p',
                                      macro_block_size=2, ffmpeg_params=['-crf', '20', '-movflags', '+faststart'])
                    encoding = validate_video(video, len(states))
                    dump_json(record, {'provenance': provenance, 'states': states, 'video': str(video),
                              'actions_executed': len(states)-1, 'terminal_success': bool(done),
                              'controller_next_step_parity_passed': True, 'anchor_observation_bitwise_equal': True,
                              'recovery_evaluated': False, 'video_full_decode_verified': True,
                              'encoding_validation': encoding, 'video_sha256': sha256(video)})
                    completed.append(str(record))
                finally:
                    env.close()
        dump_json(output/'progress.json', {'completed_branches': len(completed), 'last_anchor': meta['sample_id']})
    dump_json(output/'summary.json', {'complete_prefix_branches': True, 'branches': completed,
              'sampling': 'validation tasks6-7 init0 first observation per phase; seed12345',
              'missing_phases_not_fabricated': True, 'recovery_complete': False, 'full_video_validation_complete': True})


if __name__ == '__main__':
    main()
