"""Official LIBERO client protocol with casewise receipts and no video encoding."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, REPO, atomic_json, case_key
sys.path[:0] = [str(ROOT), str(REPO), str(REPO / 'third_party/libero'),
                str(REPO / 'packages/openpi-client/src')]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--mode', required=True)
    p.add_argument('--source-suite', required=True)
    a = p.parse_args()
    from scripts.spec import spec_client_libero as client
    import numpy as np
    output = Path(a.output)
    rows = json.loads(Path(a.cases).read_text())
    state = dict(suite=None, rpc=0, rpc_ms=0., case_dir=None)
    def json_safe(value):
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k,v in value.items() if k != 'sim'}
        if isinstance(value, (list,tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value,np.ndarray):
            return value.tolist()
        if isinstance(value,np.generic):
            return value.item()
        if value is None or isinstance(value,(str,int,float,bool)):
            return value
        return type(value).__module__ + '.' + type(value).__name__
    original_env = client._get_libero_env
    def get_env(task, resolution, seed):
        env, description = original_env(task, resolution, seed)
        bddl = Path(client.get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        original_set = env.set_init_state
        def set_init(init):
            vector = np.asarray(init)
            observation = original_set(init)
            np.savez_compressed(state['case_dir'] / 'initial_state.npz', init_vector=vector)
            raw = env.env
            controllers = [json_safe(robot.controller_config) for robot in raw.robots]
            identity = dict(bddl_path=str(bddl), bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
                instruction=str(description), environment_seed=seed, control_freq=float(raw.control_freq),
                init_vector_sha256=hashlib.sha256(np.ascontiguousarray(vector).tobytes()).hexdigest(),
                init_vector_shape=list(vector.shape), init_vector_dtype=str(vector.dtype),
                controller_configs=controllers, action_spec=json_safe(raw.action_spec),
                initial_sim_state_sha256=hashlib.sha256(np.ascontiguousarray(raw.sim.get_state().flatten()).tobytes()).hexdigest())
            identity['controller_sha256'] = hashlib.sha256(json.dumps(controllers,sort_keys=True).encode()).hexdigest()
            atomic_json(state['case_dir'] / 'environment_identity.json', identity)
            return observation
        env.set_init_state = set_init
        return env,description
    client._get_libero_env = get_env
    original_infer = client._websocket_client_policy.WebsocketClientPolicy.infer
    def infer(self, obs):
        obs = dict(obs, __ease_suite__=state['suite'])
        if state['rpc'] == 0:
            np.savez_compressed(state['case_dir'] / 'first_policy_observation.npz',
                image=np.asarray(obs['observation/image']), wrist_image=np.asarray(obs['observation/wrist_image']),
                state=np.asarray(obs['observation/state']))
        tick = time.perf_counter()
        result = original_infer(self, obs)
        state['rpc'] += 1
        state['rpc_ms'] += (time.perf_counter() - tick) * 1000
        return result
    client._websocket_client_policy.WebsocketClientPolicy.infer = infer
    # Native client still writes actions and inference JSONL. Videos are explicitly omitted.
    client.imageio.mimwrite = lambda *args, **kwargs: None
    logging.basicConfig(level=logging.WARNING, force=True)
    done = []
    for case in rows:
        key = case_key(case)
        case_dir = output / 'episodes' / key
        receipt = case_dir / 'result.json'
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous.get('status') == 'complete':
                done.append(previous)
                continue
            raise RuntimeError('Existing incomplete case requires explicit technical retry directory: ' + key)
        case_dir.mkdir(parents=True, exist_ok=True)
        state.update(suite=case['suite'], rpc=0, rpc_ms=0., case_dir=case_dir)
        tick = time.perf_counter()
        try:
            client.eval_libero(client.Args(host='127.0.0.1', port=a.port,
                task_suite_name=case['suite'], task=str(case['task_id']),
                num_trials_per_task=1, episode_index_start=case['init_id'],
                episode_index_end=case['init_id'] + 1, num_steps_wait=10,
                replan_steps=12, resize_size=224, seed=7,
                record_semantic_telemetry=True, video_out_path=str(case_dir), run_name='native'))
            episode_log = case_dir / 'native' / 'episode_log.json'
            episodes = json.loads(episode_log.read_text())
            if len(episodes) != 1:
                raise ValueError('Expected exactly one official episode')
            row = episodes[0]
            if row['failure_reason'] is not None:
                raise RuntimeError('Native client reported technical failure: ' + str(row['failure_reason']))
            if (row['task_suite_name'], row['task_id'], row['episode_idx']) != (case['suite'], case['task_id'], case['init_id']):
                raise ValueError('Official case identity mismatch')
            if row['executed_action_count'] != row['trajectory_length']:
                raise ValueError('Action count mismatch')
            result = dict(case, case_key=key, mode=a.mode, source_suite=a.source_suite,
                status='complete', success=bool(row['success']), wall_seconds=time.perf_counter() - tick,
                rpc_calls=state['rpc'], client_roundtrip_sum_ms=state['rpc_ms'],
                mean_client_roundtrip_ms=state['rpc_ms'] / state['rpc'] if state['rpc'] else None,
                episode=row, native_output=str(case_dir / 'native'),
                video_recorded=False, video_reason='omitted prospectively to reduce I/O')
            row['video_path'] = None
            row['video_recorded'] = False
            atomic_json(episode_log, [row])
            atomic_json(receipt, result)
            done.append(result)
            atomic_json(output / 'progress.json', dict(completed=len(done), total=len(rows),
                successes=sum(int(r['success']) for r in done), last_case=key))
        except Exception:
            atomic_json(receipt, dict(case, case_key=key, status='technical_failure',
                source_suite=a.source_suite, mode=a.mode, wall_seconds=time.perf_counter() - tick,
                error=traceback.format_exc()))
            raise
    atomic_json(output / 'client_complete.json', dict(status='complete', cases=len(done),
                successes=sum(int(r['success']) for r in done)))

if __name__ == '__main__':
    main()
