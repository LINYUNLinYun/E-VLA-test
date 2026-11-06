"""Four-GPU, five-init static/initial-prefix-recovery evaluations with per-episode validation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from scripts.experiment10.common import ROOT, RESULTS, SUITES, PROTOCOL, dump_json, json_lines, claim_gpu, managed_gpus
from scripts.experiment10.run_pipeline import free_gpus, event


class AdoptedWorker:
    """Track a worker left running after its scheduler process exited."""

    def __init__(self, pid, validation_path):
        self.pid = int(pid)
        self.validation_path = Path(validation_path)

    def poll(self):
        try:
            state = (Path('/proc') / str(self.pid) / 'stat').read_text().split(') ', 1)[1][0]
        except (FileNotFoundError, IndexError):
            state = 'X'
        if state not in ('Z', 'X'):
            return None
        return 0 if self.validation_path.exists() else 1


def process_is_live(pid):
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0]
    except (FileNotFoundError, IndexError):
        return False
    return state not in ('Z', 'X')


def validate_run(directory, condition, suite):
    rows = json.loads((directory / 'episode_log.json').read_text())
    task_ids = tuple(condition.get('task_ids', range(10)))
    init_indices = tuple(condition.get('init_indices', range(5)))
    expected = {(t, i) for t in task_ids for i in init_indices}
    actual = [(r['task_id'], r['episode_idx']) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise RuntimeError('Missing or duplicate official episodes')
    for row in rows:
        if row.get('failure_reason') or row.get('episode_error'):
            raise RuntimeError('Episode exception is not a valid task failure')
        states = json_lines(directory / row['telemetry_path'])
        trace = json_lines(directory / row['trace_path'])
        inference = json_lines(directory / row['infer_path'])
        if not inference or len(inference) != row['infer_record_count']:
            raise RuntimeError('Missing actual inference records')
        if len(states) != len(trace)+1 or row['frame_count'] != len(states):
            raise RuntimeError('N+1 telemetry/video mismatch')
        if [r['step'] for r in states] != list(range(len(states))):
            raise RuntimeError('Noncontiguous telemetry')
        xyz = np.asarray([r['eef_pos'] for r in states])
        if xyz.shape != (len(states), 3) or not np.isfinite(xyz).all():
            raise RuntimeError('Invalid EEF XYZ')
        for index, infer in enumerate(inference):
            metadata_key = os.environ.get('EXP11_METADATA_KEY', 'experiment10')
            metadata = infer[metadata_key]
            if metadata['condition'] != condition['name'] or metadata['checkpoint_sha256'] != condition['checkpoint_sha256']:
                raise RuntimeError('Inference provenance mismatch')
            if metadata['trace_infer_id'] != index or metadata['trace_episode_idx'] != row['episode_idx']:
                raise RuntimeError('Noise index mismatch')
            noise = np.random.default_rng(np.random.SeedSequence([12345, row['episode_idx'], index])).standard_normal((50, 32)).astype(np.float32)
            if hashlib.sha256(noise.tobytes()).hexdigest() != metadata['noise_sha256']:
                raise RuntimeError('Noise tensor hash mismatch')
            if condition.get('save_full_inputs'):
                input_path = Path(metadata.get('full_input_path', ''))
                if not input_path.is_file() or not metadata.get('full_input_sha256'):
                    raise RuntimeError('Full-observation replay artifact missing')
                with np.load(input_path, allow_pickle=False) as saved:
                    input_metadata = json.loads(str(saved['metadata_json'].item()))
                    digest = hashlib.sha256()
                    for name in ('agent_image', 'wrist_image', 'robot_state', 'noise'):
                        digest.update(np.ascontiguousarray(saved[name]).tobytes())
                    digest.update(str(saved['prompt'].item()).encode('utf-8'))
                    digest.update(str(input_metadata.get('suite', '')).encode('utf-8'))
                    if (digest.hexdigest() != input_metadata.get('input_sha256') or
                            input_metadata.get('input_sha256') != metadata['full_input_sha256'] or
                            input_metadata.get('noise_sha256') != metadata['noise_sha256'] or
                            input_metadata.get('checkpoint_sha256') != condition['checkpoint_sha256'] or
                            input_metadata.get('skip_layers') != [] or input_metadata.get('denoise_steps') != 3 or
                            int(input_metadata.get('task_id', -1)) != int(row['task_id']) or
                            int(input_metadata.get('episode_idx', -1)) != int(row['episode_idx']) or
                            int(input_metadata.get('infer_id', -1)) != int(index)):
                        raise RuntimeError('Full-observation replay artifact provenance mismatch')
            recovered = condition['mode'] == 'prefix_once' and index > 0
            if condition['mode'] in ('verify_then_execute', 'parallel_verify'):
                if (metadata['candidate_skip_layers_1based'] != condition['skip'] or
                    metadata['candidate_denoise_steps'] != condition['steps'] or
                    metadata['reference_skip_layers_1based'] != [] or
                    metadata['reference_denoise_steps'] != 10 or
                    metadata['verification_translation_threshold_m'] != condition['translation_threshold_m'] or
                    metadata['verification_rotation_threshold_rad'] != condition['rotation_threshold_rad']):
                    raise RuntimeError('Verify-gate condition/provenance mismatch')
                if condition['mode'] == 'parallel_verify':
                    if (metadata.get('mode') != 'parallel_verify' or
                        metadata.get('parallel_inference_wall_ms', -1) < 0 or
                        metadata.get('candidate_noise_sha256') != metadata['noise_sha256'] or
                        metadata.get('reference_noise_sha256') != metadata['noise_sha256'] or
                        metadata.get('candidate_checkpoint_sha256') != condition['checkpoint_sha256'] or
                        metadata.get('reference_checkpoint_sha256') != condition['checkpoint_sha256']):
                        raise RuntimeError('Parallel verifier timing/noise/checkpoint provenance mismatch')
                used_full = bool(metadata['used_full_fallback'])
                if used_full:
                    if (metadata['candidate_accepted_prefix_len'] != 0 or
                        metadata['vlm_kv_shell_indices_1based'] != [] or metadata['denoise_steps'] != 10):
                        raise RuntimeError('Full fallback metadata mismatch')
                    expected_prefix = 12
                else:
                    candidate_prefix = int(metadata['candidate_accepted_prefix_len'])
                    if (candidate_prefix < 1 or candidate_prefix > condition['horizon'] or
                        metadata['vlm_kv_shell_indices_1based'] != condition['skip'] or
                        metadata['denoise_steps'] != condition['steps']):
                        raise RuntimeError('Accepted draft metadata mismatch')
                    expected_prefix = candidate_prefix
            elif condition['mode'] == 'kalman_gate':
                if (metadata['candidate_skip_layers_1based'] != condition['skip'] or
                    metadata['candidate_denoise_steps'] != condition['steps'] or
                    metadata['reference_skip_layers_1based'] != [] or
                    metadata['reference_denoise_steps'] != 10 or
                    metadata['command_prediction_translation_threshold'] != condition['translation_threshold'] or
                    metadata['command_prediction_rotation_threshold'] != condition['rotation_threshold'] or
                    metadata['max_observation_age_ms'] != condition['max_observation_age_ms'] or
                    metadata['predictor_minimum_executed_history'] != condition['minimum_executed_history'] or
                    metadata['predictor_process_variance'] != condition['predictor_process_variance'] or
                    metadata['predictor_measurement_variance'] != condition['predictor_measurement_variance']):
                    raise RuntimeError('Kalman-gate condition/provenance mismatch')
                used_full = bool(metadata['used_full_fallback'])
                if used_full:
                    if (metadata['candidate_accepted_prefix_len'] != 0 or
                        metadata['vlm_kv_shell_indices_1based'] != [] or metadata['denoise_steps'] != 10):
                        raise RuntimeError('Kalman-gate Full18 fallback metadata mismatch')
                    expected_prefix = 12
                else:
                    candidate_prefix = int(metadata['candidate_accepted_prefix_len'])
                    if (candidate_prefix < 1 or candidate_prefix > condition['horizon'] or
                        metadata['vlm_kv_shell_indices_1based'] != condition['skip'] or
                        metadata['denoise_steps'] != condition['steps']):
                        raise RuntimeError('Kalman-gate accepted draft metadata mismatch')
                    expected_prefix = candidate_prefix
            else:
                mask, steps = ([], 10) if recovered else (condition['skip'], condition['steps'])
                if metadata['vlm_kv_shell_indices_1based'] != mask or metadata['denoise_steps'] != steps:
                    raise RuntimeError('Unexpected actual layer/schedule intervention')
                expected_prefix = (condition['horizon'] if condition['mode'] == 'prefix_once' and index == 0
                                   else condition.get('replan_steps', 12))
            if infer['accepted_prefix_len'] != expected_prefix or infer['chunk_exec_len'] != expected_prefix:
                raise RuntimeError('Unexpected planned/executed prefix length')
            chunk = [r for r in trace if r['infer_id'] == index]
            offsets = [r['action_offset_in_chunk'] for r in chunk]
            if not offsets or offsets != list(range(len(chunk))) or len(chunk) > expected_prefix:
                raise RuntimeError('Actual action-chunk execution mismatch')
            if index < len(inference)-1 and len(chunk) != expected_prefix:
                raise RuntimeError('Nonterminal chunk shortened unexpectedly')
        video = directory / row['video_path']
        probe = subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                    '-show_entries', 'stream=codec_name,pix_fmt,nb_read_frames', '-of', 'json', str(video)],
                    check=True, capture_output=True, text=True)
        stream = json.loads(probe.stdout)['streams'][0]
        if stream['codec_name'] != 'h264' or stream['pix_fmt'] != 'yuv420p' or int(stream['nb_read_frames']) != len(states):
            raise RuntimeError('Video encoding/frame mismatch')
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True, capture_output=True)
    dump_json(directory / 'validated.json', {'complete': True, 'suite': suite, 'condition': condition,
                                             'episodes': len(expected), 'full_video_decodes': len(expected),
                                             'protocol': PROTOCOL})


def worker(gpu, condition, suite, output):
    server = None
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_EGL_DEVICE_ID': str(gpu), 'MUJOCO_GL': 'egl',
                'LIBERO_CONFIG_PATH': str(ROOT / '.libero'), 'JAX_PLATFORMS': 'cpu',
                'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
                'PYTHONPATH': ':'.join([str(ROOT / 'src'), str(ROOT), '/root/nfs/embodied/benchmark/LIBERO']),
                'PYTHONUNBUFFERED': '1'})
    if condition.get('save_full_inputs'):
        env['EXP11_SAVE_FULL_INPUTS'] = str(RESULTS / 'full_observations')
    env['EXP11_SUITE'] = suite
    env['EXP11_METADATA_KEY'] = os.environ.get('EXP11_METADATA_KEY', 'experiment10')
    env['EXP11_INIT_INDICES_JSON'] = json.dumps(list(condition.get('init_indices', range(5))))
    server_command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.serve_policy',
                '--suite', suite, '--skip', ','.join(map(str, condition['skip'])), '--steps', str(condition['steps']),
                '--mode', condition['mode'], '--horizon', str(condition['horizon']),
                '--condition', condition['name'], '--port', str(port)]
    if condition['mode'] == 'verify_then_execute':
        server_command.extend(['--translation-threshold-m', str(condition['translation_threshold_m']),
                               '--rotation-threshold-rad', str(condition['rotation_threshold_rad'])])
    elif condition['mode'] == 'kalman_gate':
        server_command.extend(['--translation-threshold-m', str(condition['translation_threshold']),
                               '--rotation-threshold-rad', str(condition['rotation_threshold']),
                               '--predictor-process-variance', ','.join(map(str, condition['predictor_process_variance'])),
                               '--predictor-measurement-variance', ','.join(map(str, condition['predictor_measurement_variance'])),
                               '--max-observation-age-ms', str(condition['max_observation_age_ms']),
                               '--minimum-executed-history', str(condition['minimum_executed_history'])])
    started = time.monotonic()
    with (logs / 'server.log').open('a') as stream:
        try:
            server = subprocess.Popen(server_command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Policy server exited before ready')
                try:
                    with socket.create_connection(('127.0.0.1', port), timeout=1):
                        break
                except OSError:
                    if time.monotonic() - started > 600:
                        raise TimeoutError('Policy server not ready')
                    time.sleep(2)
            task_ids = tuple(condition.get('task_ids', range(10)))
            init_indices = tuple(condition.get('init_indices', range(5)))
            task_spec = ','.join(map(str, task_ids))
            client_command = [str(ROOT / 'examples/libero/.venv/bin/python'), '-m', 'scripts.experiment10.client_libero',
                    '--host', '127.0.0.1', '--port', str(port), '--task-suite-name', suite, '--task', task_spec,
                    '--num-trials-per-task', str(len(init_indices)), '--episode-index-start', str(min(init_indices)),
                    '--episode-index-end', str(max(init_indices) + 1),
                    '--num-steps-wait', '10', '--replan-steps', str(condition.get('replan_steps', 12)),
                    '--seed', '7', '--record-semantic-telemetry',
                    '--video-out-path', str(output), '--run-name', 'run']
            with (logs / 'client.log').open('a') as client_stream:
                subprocess.run(client_command, env=env, cwd=ROOT, check=True, stdout=client_stream, stderr=subprocess.STDOUT)
            validate_run(output / 'run', condition, suite)
        finally:
            if server is not None and server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--status-path', type=Path,
                        help='Scheduler status file; defaults to results/.../closed_loop_status.json')
    parser.add_argument('--condition-index', type=int)
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    conditions = json.loads(args.conditions.read_text())['conditions']
    for condition in conditions:
        import re
        from scripts.experiment10.backend import validate_mask
        if not re.fullmatch('[A-Za-z0-9_]+', condition['name']):
            raise ValueError('Condition names must be safe local path components')
        validate_mask(condition['skip'])
        task_ids = condition.get('task_ids', list(range(10)))
        init_indices = condition.get('init_indices', list(range(5)))
        condition_suites = condition.get('suites', SUITES)
        if (not task_ids or not init_indices or len(set(task_ids)) != len(task_ids) or
                len(set(init_indices)) != len(init_indices) or any(type(t) is not int or t < 0 or t > 9 for t in task_ids) or
                any(type(i) is not int or i < 0 or i > 4 for i in init_indices) or
                not condition_suites or any(suite not in SUITES for suite in condition_suites) or
                len(set(condition_suites)) != len(condition_suites) or
                sorted(init_indices) != list(range(min(init_indices), max(init_indices) + 1))):
            raise ValueError('Invalid task/init coverage for condition ' + condition['name'])
        if (condition['steps'] not in (1, 2, 3, 4, 5, 7, 10) or
            condition['horizon'] not in (1, 2, 4, 5, 8, 12) or
            condition['mode'] not in ('static', 'prefix_once', 'verify_then_execute', 'kalman_gate')):
            raise ValueError('Unknown schedule or execution mode')
        if condition.get('replan_steps', 12) not in (1, 2, 3, 4, 5, 8, 12):
            raise ValueError('Unsupported execution window')
        if condition['mode'] == 'static' and condition['horizon'] != condition.get('replan_steps', 12):
            raise ValueError('Static action prefix must match the execution window')
        if condition['mode'] == 'verify_then_execute':
            task_ids, init_indices = condition.get('task_ids', []), condition.get('init_indices', [])
            if (not task_ids or not init_indices or
                any(type(t) is not int or t < 0 or t > 9 for t in task_ids) or
                any(type(i) is not int or i < 0 or i > 4 for i in init_indices) or
                len(set(task_ids)) != len(task_ids) or len(set(init_indices)) != len(init_indices) or
                sorted(init_indices) != list(range(min(init_indices), max(init_indices)+1)) or
                not np.isfinite(condition.get('translation_threshold_m', float('nan'))) or
                condition['translation_threshold_m'] <= 0 or
                not np.isfinite(condition.get('rotation_threshold_rad', float('nan'))) or
                condition['rotation_threshold_rad'] <= 0):
                raise ValueError('Verify-then-execute needs task/init lists and frozen positive thresholds')
    if condition['mode'] == 'kalman_gate':
        q, r = condition.get('predictor_process_variance', []), condition.get('predictor_measurement_variance', [])
        if (len(q) != 6 or len(r) != 6 or
            any(not np.isfinite(v) or v <= 0 for v in q + r) or
            any(not np.isfinite(condition.get(k, float('nan'))) or condition[k] <= 0
                for k in ('translation_threshold', 'rotation_threshold', 'max_observation_age_ms')) or
            type(condition.get('minimum_executed_history')) is not int or condition['minimum_executed_history'] < 2):
            raise ValueError('Kalman-gate needs frozen positive Q/R, thresholds, age limit and history length')
    if args.worker:
        condition = conditions[args.condition_index]
        destination = RESULTS / 'closed_loop' / condition['name'] / args.suite
        worker(args.gpu, condition, args.suite, destination)
        return
    status_path = (args.status_path or (RESULTS/'closed_loop_status.json')).resolve()
    if RESULTS.resolve() not in status_path.parents:
        raise ValueError('Scheduler status path must stay inside experiment results')
    if not args.launch:
        counts = {c['name']: len(c.get('task_ids', range(10))) *
                  len(c.get('init_indices', range(5))) * len(c.get('suites', SUITES)) for c in conditions}
        print(json.dumps({'conditions': conditions, 'episodes_by_condition': counts,
                          'suites': sorted({s for c in conditions for s in c.get('suites', SUITES)}),
                          'default_init_indices': [0,1,2,3,4]}, indent=2))
        return
    status_path.parent.mkdir(parents=True, exist_ok=True)
    scheduler_lock = status_path.with_name(status_path.name + '.scheduler.lock').open('a')
    try:
        fcntl.flock(scheduler_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('A scheduler already owns %s' % status_path) from exc

    prior = json.loads(status_path.read_text()) if status_path.exists() else None
    if prior and process_is_live(prior.get('pid', -1)):
        cmdline = (Path('/proc') / str(prior['pid']) / 'cmdline').read_bytes().replace(b'\0', b' ')
        if b'scripts.experiment10.run_closed_loop' in cmdline and b'--worker' not in cmdline:
            raise RuntimeError('Scheduler PID %s is still live for %s' % (prior['pid'], status_path))

    pending = ([(i, suite) for i, c in enumerate(conditions) for suite in c.get('suites', SUITES)]
               if prior is None else
               [(int(i), str(suite)) for i, suite in prior.get('pending', [])])
    active, failures, next_poll = {}, list(prior.get('failures', [])) if prior else [], 0.
    leases = {}
    adopted_tasks = set()
    if prior and not prior.get('complete', False):
        names = {condition['name']: i for i, condition in enumerate(conditions)}
        for record in prior.get('active', []):
            if (record.get('condition') not in names or
                    record.get('suite') not in conditions[names[record['condition']]].get('suites', SUITES)):
                raise RuntimeError('Cannot resume unknown active task: %r' % record)
            index, suite = names[record['condition']], record['suite']
            gpu, pid = int(record['gpu']), int(record['pid'])
            destination = RESULTS / 'closed_loop' / conditions[index]['name'] / suite
            validation = destination / 'run/validated.json'
            if process_is_live(pid):
                active[gpu] = (AdoptedWorker(pid, validation), index, suite, open(os.devnull, 'w'))
                adopted_tasks.add((index, suite))
                lease = claim_gpu(gpu)
                if lease is not None:
                    leases[gpu] = lease
                event(event='closed_loop_worker_adopted', gpu=gpu, condition=conditions[index]['name'],
                      suite=suite, pid=pid)
            elif validation.exists():
                saved = json.loads(validation.read_text())
                if saved.get('condition') != conditions[index]:
                    raise RuntimeError('Closed-loop resume identity mismatch')
                event(event='closed_loop_finished_while_unmanaged', gpu=gpu,
                      condition=conditions[index]['name'], suite=suite)
            else:
                pending.insert(0, (index, suite))

    def prune_validated():
        remaining = []
        for index, suite in pending:
            destination = RESULTS / 'closed_loop' / conditions[index]['name'] / suite
            validation = destination / 'run/validated.json'
            if not validation.exists():
                remaining.append((index, suite))
                continue
            saved = json.loads(validation.read_text())
            if saved.get('condition') != conditions[index]:
                raise RuntimeError('Closed-loop resume identity mismatch')
            event(event='closed_loop_already_validated', condition=conditions[index]['name'], suite=suite)
        return remaining

    pending = [(i, s) for i, s in pending if (i, s) not in adopted_tasks]
    pending = prune_validated()
    previous_managed = managed_gpus()
    event(event='closed_loop_scheduler_started', status_path=str(status_path),
          resumed=prior is not None, adopted=len(active), pending=len(pending))
    while pending or active:
        currently_managed = managed_gpus()
        if previous_managed - currently_managed:
            next_poll = 0.
        previous_managed = currently_managed
        for gpu, (process, index, suite, stream) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                if code:
                    failures.append({'condition': conditions[index]['name'], 'suite': suite, 'code': code})
                del active[gpu]
                lease = leases.pop(gpu, None)
                if lease is not None:
                    lease.close()
                next_poll = 0.
                event(event='closed_loop_job_finished', gpu=gpu, condition=conditions[index]['name'], suite=suite, code=code)
        pending = prune_validated()
        # Do not ask nvidia-smi while all four devices are already owned by our
        # live workers. Completion/PID changes wake dispatch without GPU polling.
        if pending and len(currently_managed) < PROTOCOL['max_workers'] and time.monotonic() >= next_poll:
            free = [g for g in free_gpus() if g not in active]
            next_poll = time.monotonic() + 300
            for gpu in free:
                if not pending:
                    break
                lease = claim_gpu(gpu)
                if lease is None:
                    continue
                index, suite = pending.pop(0)
                destination = RESULTS / 'closed_loop' / conditions[index]['name'] / suite
                validation = destination / 'run/validated.json'
                if validation.exists():
                    saved = json.loads(validation.read_text())
                    if saved['condition'] != conditions[index]:
                        raise RuntimeError('Closed-loop resume identity mismatch')
                    lease.close()
                    next_poll = 0.
                    continue
                leases[gpu] = lease
                destination.mkdir(parents=True, exist_ok=True)
                stream = (destination / 'worker.log').open('a')
                command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.run_closed_loop',
                          '--conditions', str(args.conditions.resolve()), '--worker', '--condition-index', str(index),
                          '--suite', suite, '--gpu', str(gpu)]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, index, suite, stream)
                event(event='closed_loop_job_started', gpu=gpu, condition=conditions[index]['name'], suite=suite, pid=process.pid)
        dump_json(status_path, {'pid': os.getpid(), 'pending': pending, 'failures': failures,
                    'active': [{'gpu': g, 'pid': p.pid, 'condition': conditions[i]['name'], 'suite': s}
                               for g, (p,i,s,_) in active.items()],
                    'conditions': [condition['name'] for condition in conditions],
                    'complete': not pending and not active and not failures})
        if pending or active:
            time.sleep(2)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
