"""Measured 20 Hz LIBERO episodes; model RPC transports numpy without copies to JSON."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle
import socket
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUITES = ('libero_object', 'libero_spatial', 'libero_goal', 'libero_10')
LIMITS = dict(zip(SUITES, (280, 220, 300, 520)))
LOADED_CLIENT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())
        f.write('\n')


def recv_exact(sock, n):
    parts = bytearray()
    while len(parts) < n:
        block = sock.recv(n - len(parts))
        if not block:
            raise ConnectionError('Model RPC ended early')
        parts.extend(block)
    return bytes(parts)


def rpc(port, request, host='127.0.0.1'):
    if host not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Trusted pickle RPC is restricted to localhost')
    payload = pickle.dumps(request, protocol=5)
    with socket.create_connection((host, port), timeout=600) as sock:
        sock.sendall(struct.pack('!I', len(payload)) + payload)
        size = struct.unpack('!I', recv_exact(sock, 4))[0]
        if size > 128 * 1024 * 1024:
            raise ValueError('RPC response exceeds limit')
        result = pickle.loads(recv_exact(sock, size))
    if not result.get('ok'):
        raise RuntimeError(str(result))
    return result


def model_seed(suite, task, init, replan):
    # Same source of model noise across all masks, independent of observations.
    key = f'idea1-v21/7/{suite}/{task}/{init}/{replan}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')


def value_digest(value):
    h = hashlib.sha256()
    def add(value):
        if isinstance(value, dict):
            for name in sorted(value):
                h.update(name.encode())
                add(value[name])
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)
        elif isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(str(value.shape).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True).encode())
    add(value)
    return h.hexdigest()


def input_digest(obs, instruction):
    return value_digest(dict(obs=obs, instruction=instruction))


def telemetry_helpers():
    repo = ROOT / 'policies/realtime-vla-flash'
    for path in (repo / 'scripts/spec', repo / 'scripts/experiment6', repo / 'packages/openpi-client/src'):
        sys.path.insert(0, str(path))
    import trajectory_replay as replay
    return replay


def capture_observation(path, suite='libero_object', task=0, init=0):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    selected = benchmark.get_benchmark_dict()[suite]()
    definition = selected.get_task(task)
    bddl = Path(get_libero_path('bddl_files')) / definition.problem_folder / definition.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(official_init_states(selected, task)[init])
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in obs.items() if isinstance(v, np.ndarray)},
                            instruction=np.asarray(definition.language), seed=np.asarray(model_seed(suite, task, init, 0)))
        return path
    finally:
        env.close()


def official_init_states(suite, task):
    # LIBERO's pinned loader predates torch2.6's weights_only default change.
    # Load only the exact local official suite file, never arbitrary user input.
    import torch
    from libero.libero import get_libero_path
    definition = suite.get_task(task)
    path = Path(get_libero_path('init_states')) / definition.problem_folder / definition.init_states_file
    return torch.load(path, weights_only=False)


def run_episode(directory, port, model, suite, task, init, mask, kind='formal', reference=False):
    """One official init. Exceptions are technical invalid; normal failure returns valid."""
    import imageio.v2 as imageio
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    replay = telemetry_helpers()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    selected = benchmark.get_benchmark_dict()[suite]()
    definition = selected.get_task(task)
    init_states = official_init_states(selected, task)
    if init >= len(init_states):
        raise ValueError('Official init index unavailable')
    bddl = Path(get_libero_path('bddl_files')) / definition.problem_folder / definition.bddl_file
    init_path = Path(get_libero_path('init_states')) / definition.problem_folder / definition.init_states_file
    # The source bytes and selected vector are separate provenance checks.
    state_hash = hashlib.sha256(np.asarray(init_states[init]).tobytes()).hexdigest()
    write_json(directory / 'identity.json', dict(model=model, suite=suite, task=task, init=init,
               skip=mask, kind=kind, reference=reference, env_seed=7, init_vector_hash=state_hash,
               bddl_sha256=sha(bddl), init_file=str(init_path), init_file_sha256=sha(init_path),
               instruction=definition.language))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    writer = None
    start = time.perf_counter()
    try:
        env.seed(7)
        env.reset()
        obs = env.set_init_state(init_states[init])
        for _ in range(10):
            obs, _, _, _ = env.step([0.] * 6 + [-1.])
        if float(env.env.control_freq) != 20.:
            raise ValueError('Expected native 20 Hz LIBERO control')
        controller = env.env.robots[0].controller
        controller_record = {name: np.asarray(getattr(controller, name)).tolist()
                             for name in ('input_min', 'input_max', 'output_min', 'output_max')}
        if (not np.allclose(controller_record['input_min'], -1.) or
            not np.allclose(controller_record['input_max'], 1.) or
            not np.allclose(controller_record['output_max'], [.05, .05, .05, .5, .5, .5]) or
            not np.allclose(controller_record['output_min'], [-.05, -.05, -.05, -.5, -.5, -.5])):
            raise ValueError('Native OSC controller scales differ from frozen 20 Hz LIBERO protocol')
        rpc(port, {'op': 'reset'})
        health = rpc(port, {'op': 'health'})
        protocol = health.get('protocol', health)
        r_by_suite = protocol.get('R_by_suite', protocol.get('R'))
        r = r_by_suite[suite] if isinstance(r_by_suite, dict) else r_by_suite
        expected_h = protocol.get('H')
        if not isinstance(r, int) or not isinstance(expected_h, int) or not 0 < r <= expected_h:
            raise ValueError('Server must freeze measured native H/R before episodes')
        write_json(directory / 'native_protocol.json', protocol)
        write_json(directory / 'environment_protocol.json', dict(controller=controller_record,
                   control_frequency_hz=float(env.env.control_freq), loaded_client_sha256=LOADED_CLIENT_SHA256,
                   python=sys.version, numpy=np.__version__, raw_health=health))
        primary, target = replay._goal_object_names(env.env)
        initial = {name: replay._object_state(env.env, name).get('pos') for name in primary}
        states, infers = [], []
        writer = imageio.get_writer(str(directory / 'video.mp4'), fps=20, codec='libx264',
                                   pixelformat='yuv420p', macro_block_size=2,
                                   ffmpeg_params=['-crf', '20', '-movflags', '+faststart'])
        io_s = 0.

        def capture(step, action):
            nonlocal io_s
            tick = time.perf_counter()
            row = replay.snapshot(env, obs, step=step, when='initial' if step == 0 else 'post_action',
                                  action=action, primary_names=primary, target_names=target, initial_positions=initial)
            # Keep missing contact evidence unknown, not a false negative.
            for name in primary:
                try:
                    gripper, obj = env.env.robots[0].gripper, env.env.get_object(name)
                    row['robot_contacts'][name] = bool(env.env.check_contact(gripper, obj))
                    row['robot_grasps'][name] = bool(env.env._check_grasp(gripper=gripper, object_geoms=obj))
                except Exception:
                    row['robot_contacts'][name] = row['robot_grasps'][name] = None
            row['sim_time_s'] = float(env.env.sim.data.time)
            row['wall_timestamp_s'] = time.time()
            states.append(row)
            writer.append_data(np.ascontiguousarray(obs['agentview_image'][::-1, ::-1]))
            io_s += time.perf_counter() - tick

        capture(0, None)
        steps, done = 0, False
        while steps < LIMITS[suite] and not done:
            replan = len(infers)
            seed = model_seed(suite, task, init, replan)
            inputs = directory / f'inputs/replan{replan:04d}.npz'
            inputs.parent.mkdir(exist_ok=True)
            np.savez_compressed(inputs, **{k: np.asarray(v) for k, v in obs.items() if isinstance(v, np.ndarray)},
                                instruction=np.asarray(definition.language), seed=np.asarray(seed))
            tick = time.perf_counter()
            model_obs = {k: np.asarray(v) for k, v in obs.items() if isinstance(v, np.ndarray)}
            response = rpc(port, dict(op='predict', obs=model_obs, instruction=definition.language,
                           suite=suite, seed=seed, skip=mask, reference=False, native=reference, probe=False,
                           category=kind, cache_path=str((directory / f'prediction{replan:04d}.npz').resolve())))
            roundtrip = (time.perf_counter() - tick) * 1000
            if response['input_hash'] != input_digest(model_obs, definition.language):
                raise ValueError('Server received observation differs from saved input')
            actions = np.asarray(response['actions'], dtype=np.float32)
            if actions.shape != (expected_h, 7) or not np.isfinite(actions).all():
                raise ValueError(f'Invalid native action shape/value: {actions.shape}')
            prediction = directory / f'chunk{replan:04d}.npz'
            np.savez_compressed(prediction, actions=actions)
            n = min(r, LIMITS[suite] - steps)
            record = dict(response)
            record.pop('actions')
            if 'prediction_path' in record:
                record['server_prediction_path'] = record.pop('prediction_path')
            record.update(replan=replan, step=steps, seed=seed, input_path=str(inputs),
                          input_file_sha256=sha(inputs), prediction_path=str(prediction),
                          prediction_sha256=sha(prediction), chunk_actions=actions.tolist(),
                          client_roundtrip_ms=roundtrip, executed_length=0)
            infers.append(record)
            for action in actions[:n]:
                # DB's native benchmark discretizes only the final gripper value.
                executed = action.copy()
                if model == 'db-cogact':
                    executed[-1] = -1. if executed[-1] < 0 else 1.
                obs, _, done, _ = env.step(executed.tolist())
                steps += 1
                record['executed_length'] += 1
                capture(steps, executed.tolist())
                if done:
                    break
        replay.assign_semantic_phases(states)
        for filename, rows in [('telemetry.jsonl', states), ('infer.jsonl', infers)]:
            with (directory / filename).open('x') as stream:
                for row in rows:
                    stream.write(json.dumps(row, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')
        writer.close()
        writer = None
        phase_order = ['initial', 'approach', 'contact', 'grasp_or_manipulate', 'transport', 'goal_alignment', 'release_or_final_action', 'success']
        furthest = max((row['semantic_phase'] for row in states), key=lambda phase: phase_order.index(phase))
        result = dict(model=model, suite=suite, task=task, init=init, success=bool(done),
                      valid=True, kind=kind, actions=steps, predictions=len(infers), frames=len(states),
                      termination='success' if done else 'action_limit', wall_s=time.perf_counter()-start,
                      record_io_s=io_s, video='video.mp4', control_hz=20, H=expected_h, R=r,
                      directory=str(directory.resolve()), furthest_phase=furthest,
                      final_phase=states[-1]['semantic_phase'])
        write_json(directory / 'result.json', result)
        return result
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--directory', required=True, type=Path)
    p.add_argument('--port', required=True, type=int)
    p.add_argument('--model', required=True, choices=('xiaomi-robotic0', 'db-cogact'))
    p.add_argument('--suite', required=True, choices=SUITES)
    p.add_argument('--task', required=True, type=int)
    p.add_argument('--init', required=True, type=int)
    p.add_argument('--mask', default='[]', type=json.loads)
    p.add_argument('--kind', default='formal')
    p.add_argument('--reference', action='store_true')
    a = p.parse_args()
    run_episode(a.directory, a.port, a.model, a.suite, a.task, a.init, a.mask, a.kind, a.reference)
