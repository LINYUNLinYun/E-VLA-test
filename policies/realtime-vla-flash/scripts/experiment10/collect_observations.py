"""Replay official Full18 init0-4; capture actual replanning observations with exact provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/spec'))
from scripts.experiment10.common import (RESULTS, SOURCE, SUITES, PHASES, INIT_INDICES, PROTOCOL,
                                         dump_json, json_lines, select_steps, sha256, split)


def collect(suite_name):
    import spec_client_libero as client
    from libero.libero import benchmark
    output = RESULTS / 'observations' / suite_name
    output.mkdir(parents=True, exist_ok=True)
    # Latest complete source only; require exact 10 task x 10 init source matrix.
    source = None
    for candidate in sorted(SOURCE.glob('exp9_pilot_Full18_' + suite_name + '_i00_10_a*'), reverse=True):
        rows = json.loads((candidate / 'episode_log.json').read_text())
        keys = [(int(r['task_id']), int(r['episode_idx'])) for r in rows]
        if len(keys) == 100 and set(keys) == {(t, i) for t in range(10) for i in range(10)}:
            source = candidate
            break
    if source is None:
        raise RuntimeError('No complete baseline source for ' + suite_name)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    np.random.seed(7)
    environments = {}
    episodes = []
    observations, absent = [], []
    for task_id in range(10):
        for init in INIT_INDICES:
            row = next(r for r in rows if r['task_id'] == task_id and r['episode_idx'] == init)
            prefix = 'task%02d_init%02d' % (task_id, init)
            episode_path = output / (prefix + '.json')
            if episode_path.exists():
                completed = json.loads(episode_path.read_text())
                if (completed['source_trace_sha256'] != sha256(source / row['trace_path'])
                        or completed['protocol'] != json.loads(json.dumps(PROTOCOL))):
                    raise RuntimeError('Resume provenance mismatch: ' + str(episode_path))
                for saved in completed['observations']:
                    if sha256(saved['path']) != saved['sha256']:
                        raise RuntimeError('Resume observation hash mismatch')
                # Replay even a completed preceding init: the source client reuses
                # one environment/controller across inits within each task.
            telemetry = json_lines(source / row['telemetry_path'])
            trace = json_lines(source / row['trace_path'])
            actions = [r['executed_action'] for r in trace if r.get('executed_action') is not None]
            if len(telemetry) != len(actions) + 1 or any(r['step'] != i for i, r in enumerate(telemetry)):
                raise RuntimeError('Unaligned source telemetry')
            replan_steps = {i for i, r in enumerate(trace) if r.get('infer_start_frame')}
            selected, missing = select_steps(telemetry, replan_steps)
            task = suite.get_task(task_id)
            if task_id not in environments:
                environments[task_id] = client._get_libero_env(task, client.LIBERO_ENV_RESOLUTION, 7)
            env, prompt = environments[task_id]
            saved, max_error, done = [], 0.0, False
            trace_hash = sha256(source / row['trace_path'])
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(task_id)[init])
                for _ in range(10):
                    obs, _, _, _ = env.step(client.LIBERO_DUMMY_ACTION)
                for step in range(len(actions) + 1):
                    expected = np.asarray(telemetry[step]['eef_pos'])
                    error = float(np.max(np.abs(np.asarray(obs['robot0_eef_pos']) - expected)))
                    max_error = max(max_error, error)
                    if not np.isfinite(error) or error > 1e-5:
                        raise RuntimeError('Replay EEF mismatch at %s/%s/%s/%s: %s' % (suite_name, task_id, init, step, error))
                    if step in selected:
                        phase, ordinal = selected[step]
                        sample_id = '%s_%s_step%04d_%s' % (suite_name, prefix, step, phase)
                        path = output / (sample_id + '.npz')
                        def image(value):
                            return client.image_tools.convert_to_uint8(client.image_tools.resize_with_pad(
                                np.ascontiguousarray(value[::-1, ::-1]), 224, 224))
                        state = np.concatenate((obs['robot0_eef_pos'], client._quat2axisangle(obs['robot0_eef_quat']),
                                                obs['robot0_gripper_qpos'])).astype(np.float32)
                        meta = {'sample_id': sample_id, 'suite': suite_name, 'task_id': task_id,
                                'episode_idx': init, 'trajectory_step': step, 'semantic_phase': phase,
                                'raw_phase': telemetry[step]['semantic_phase'], 'ordinal': ordinal,
                                'split': split(task_id), 'task_description': str(prompt),
                                'episode_success': bool(row['success']), 'source_trace': str(source / row['trace_path']),
                                'source_trace_sha256': trace_hash, 'source_telemetry': str(source / row['telemetry_path']),
                                'eef_replay_abs_error': error, 'seed': 7, 'warmup_steps': 10,
                                'diagnostic_snapshot': telemetry[step], 'protocol': PROTOCOL}
                        np.savez_compressed(path, agent_image=image(obs['agentview_image']),
                                            wrist_image=image(obs['robot0_eye_in_hand_image']), robot_state=state,
                                            prompt=np.asarray(str(prompt)), metadata_json=np.asarray(json.dumps(meta, sort_keys=True)))
                        saved.append({'path': str(path.resolve()), 'sha256': sha256(path), **{k: meta[k] for k in
                                      ('sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase', 'ordinal', 'split')}})
                    if step < len(actions):
                        obs, _, done, _ = env.step(actions[step])
                if bool(done) != bool(row['success']):
                    raise RuntimeError('Replay success mismatch')
            finally:
                if init == INIT_INDICES[-1]:
                    env.close()
                    del environments[task_id]
            missing_rows = [{'suite': suite_name, 'task_id': task_id, 'episode_idx': init,
                             'phase': phase, 'reason': 'absent_at_replanning_point'} for phase in missing]
            completed = {'suite': suite_name, 'task_id': task_id, 'episode_idx': init,
                         'source_trace_sha256': trace_hash, 'protocol': PROTOCOL,
                         'max_eef_abs_error': max_error, 'success_matched': True,
                         'observations': saved, 'missing_phases': missing_rows}
            dump_json(episode_path, completed)
            episodes.append(completed)
            observations.extend(saved)
            absent.extend(missing_rows)
            print(json.dumps({'event': 'episode_replayed', 'suite': suite_name, 'task': task_id,
                              'init': init, 'samples': len(saved), 'max_error': max_error}), flush=True)
    dump_json(output / 'manifest.json', {'complete': True, 'protocol': PROTOCOL, 'source_run': str(source),
               'episode_count': len(episodes), 'sample_count': len(observations), 'observations': observations,
               'missing_phases': absent, 'replay_audit': episodes})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    collect(parser.parse_args().suite)
