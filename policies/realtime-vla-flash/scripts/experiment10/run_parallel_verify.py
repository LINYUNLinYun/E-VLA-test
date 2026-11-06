"""Two-GPU same-observation parallel D3/Full18 verification rollouts."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import numpy as np

from scripts.experiment10.common import (
    ROOT, RESULTS, SUITES, PROTOCOL, claim_gpu, dump_json, managed_gpus,
)
from scripts.experiment10.run_closed_loop import AdoptedWorker, process_is_live, validate_run
from scripts.experiment10.run_pipeline import event, free_gpus


def reserve_port():
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        return reservation.getsockname()[1]


def server_env(gpu):
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'MUJOCO_EGL_DEVICE_ID': str(gpu),
        'MUJOCO_GL': 'egl',
        'LIBERO_CONFIG_PATH': str(ROOT / '.libero'),
        'JAX_PLATFORMS': 'cpu',
        'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
        'PYTHONPATH': ':'.join([str(ROOT / 'src'), str(ROOT), '/root/nfs/embodied/benchmark/LIBERO']),
        'PYTHONUNBUFFERED': '1',
    })
    return env


def wait_server(process, port, label, started):
    from websockets.sync.client import connect
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError('%s server exited before ready (code=%s)' % (label, code))
        try:
            with connect('ws://127.0.0.1:%d' % port, open_timeout=1,
                         compression=None, max_size=None) as connection:
                connection.recv()
                event(event='parallel_verify_server_ready', role=label, port=port,
                      elapsed_s=round(time.monotonic() - started, 2))
                return
        except (OSError, TimeoutError):
            time.sleep(2)
    raise TimeoutError('%s policy server not ready after 600 seconds' % label)


def worker(gpus, condition, suite, output):
    if len(gpus) != 2 or gpus[0] == gpus[1]:
        raise ValueError('Parallel verification requires two distinct GPUs')
    output.mkdir(parents=True, exist_ok=True)
    logs = output / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    candidate_port, reference_port = reserve_port(), reserve_port()
    if candidate_port == reference_port:
        reference_port = reserve_port()
    policy_command = [
        str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.serve_policy',
        '--suite', suite, '--skip', ','.join(map(str, condition['skip'])),
        '--mode', 'static', '--horizon', str(condition['horizon']),
    ]
    candidate_command = policy_command + [
        '--steps', str(condition['steps']), '--condition', condition['name'] + '_D3',
        '--port', str(candidate_port),
    ]
    reference_command = policy_command + [
        '--steps', '10', '--condition', condition['name'] + '_Full18_D10',
        '--port', str(reference_port),
    ]
    processes, streams = [], []
    started = time.monotonic()
    try:
        for role, gpu, command in (
            ('candidate_d3', gpus[0], candidate_command),
            ('reference_full18', gpus[1], reference_command),
        ):
            stream = (logs / (role + '.log')).open('a')
            streams.append(stream)
            process = subprocess.Popen(command, env=server_env(gpu), cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
        wait_server(processes[0], candidate_port, 'candidate_d3', started)
        wait_server(processes[1], reference_port, 'reference_full18', started)

        task_ids = tuple(condition.get('task_ids', range(10)))
        init_indices = tuple(condition.get('init_indices', range(5)))
        client_command = [
            str(ROOT / 'examples/libero/.venv/bin/python'), '-m',
            'scripts.experiment10.client_libero', '--host', '127.0.0.1',
            '--port', str(candidate_port), '--task-suite-name', suite,
            '--task', ','.join(map(str, task_ids)),
            '--num-trials-per-task', str(len(init_indices)),
            '--episode-index-start', str(min(init_indices)),
            '--episode-index-end', str(max(init_indices) + 1),
            '--num-steps-wait', '10', '--replan-steps', '12', '--seed', '7',
            '--record-semantic-telemetry', '--video-out-path', str(output),
            '--run-name', 'run',
        ]
        client_env = server_env(gpus[0])
        client_env.update({
            'EXP10_PARALLEL_REFERENCE_PORT': str(reference_port),
            'EXP10_PARALLEL_VERIFY_JSON': json.dumps(condition, separators=(',', ':')),
        })
        with (logs / 'client.log').open('a') as stream:
            subprocess.run(client_command, env=client_env, cwd=ROOT, check=True,
                           stdout=stream, stderr=subprocess.STDOUT)
        validate_run(output / 'run', condition, suite)
        event(event='parallel_verify_worker_validated', condition=condition['name'], suite=suite,
              gpus=list(gpus), episodes=len(task_ids) * len(init_indices))
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()


def validate_condition(condition):
    import re
    from scripts.experiment10.backend import validate_mask

    if not re.fullmatch('[A-Za-z0-9_]+', condition['name']):
        raise ValueError('Condition names must be safe local path components')
    validate_mask(condition['skip'])
    tasks, inits = condition.get('task_ids', []), condition.get('init_indices', [])
    if (condition.get('mode') != 'parallel_verify' or
        condition['steps'] not in (1, 2, 3, 4, 5, 7) or
        condition['horizon'] not in (1, 2, 4, 8, 12) or
        not tasks or not inits or
        any(type(t) is not int or t < 0 or t > 9 for t in tasks) or
        any(type(i) is not int or i < 0 or i > 4 for i in inits) or
        len(set(tasks)) != len(tasks) or len(set(inits)) != len(inits) or
        sorted(inits) != list(range(min(inits), max(inits) + 1)) or
        min(inits) != 0 or max(inits) != 4 or
        not np.isfinite(condition.get('translation_threshold_m', float('nan'))) or
        condition['translation_threshold_m'] <= 0 or
        not np.isfinite(condition.get('rotation_threshold_rad', float('nan'))) or
        condition['rotation_threshold_rad'] <= 0 or
        not re.fullmatch('[0-9a-f]{64}', condition.get('checkpoint_sha256', ''))):
        raise ValueError('Invalid parallel-verify mode, split, horizon or frozen thresholds')


class LiveWorker:
    def __init__(self, process, condition_index, suite, gpus, stream):
        self.process, self.condition_index, self.suite = process, condition_index, suite
        self.gpus, self.stream = gpus, stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=Path, required=True)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--status-path', type=Path)
    parser.add_argument('--condition-index', type=int)
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--gpus', help='Two physical GPU IDs, e.g. 0,1')
    args = parser.parse_args()
    conditions = json.loads(args.conditions.read_text())['conditions']
    for condition in conditions:
        validate_condition(condition)

    if args.worker:
        if args.condition_index is None or args.suite is None or not args.gpus:
            raise ValueError('Worker requires condition index, suite and a two-GPU pair')
        gpus = tuple(int(gpu) for gpu in args.gpus.split(','))
        condition = conditions[args.condition_index]
        output = RESULTS / 'closed_loop' / condition['name'] / args.suite
        worker(gpus, condition, args.suite, output)
        return

    counts = {c['name']: len(c['task_ids']) * len(c['init_indices']) * len(SUITES)
              for c in conditions}
    if not args.launch:
        print(json.dumps({'conditions': conditions, 'episodes_by_condition': counts,
                          'suites': SUITES, 'gpus_per_worker': 2}, indent=2))
        return

    status_path = (args.status_path or RESULTS / 'parallel_verify_status.json').resolve()
    if RESULTS.resolve() not in status_path.parents:
        raise ValueError('Scheduler status path must stay inside experiment results')
    lock = status_path.with_name(status_path.name + '.scheduler.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('A scheduler already owns %s' % status_path) from exc

    prior = json.loads(status_path.read_text()) if status_path.exists() else None
    if prior and process_is_live(prior.get('pid', -1)):
        command_line = (Path('/proc') / str(prior['pid']) / 'cmdline').read_bytes().replace(b'\0', b' ')
        if b'scripts.experiment10.run_parallel_verify' in command_line and b'--worker' not in command_line:
            raise RuntimeError('Scheduler PID %s is still live for %s' % (prior['pid'], status_path))
    pending = ([(index, suite) for index in range(len(conditions)) for suite in SUITES]
               if prior is None else [(int(i), str(suite)) for i, suite in prior.get('pending', [])])
    failures = list(prior.get('failures', [])) if prior else []
    active = {}
    leases = {}
    next_poll = 0.0
    if prior and not prior.get('complete', False):
        for record in prior.get('active', []):
            name_to_index = {condition['name']: i for i, condition in enumerate(conditions)}
            condition_index, suite = name_to_index[record['condition']], record['suite']
            gpus = tuple(int(gpu) for gpu in record['gpus'])
            validation = RESULTS / 'closed_loop' / conditions[condition_index]['name'] / suite / 'run/validated.json'
            if process_is_live(record['pid']):
                worker_process = AdoptedWorker(record['pid'], validation)
                active[(condition_index, suite)] = LiveWorker(
                    worker_process, condition_index, suite, gpus, open(os.devnull, 'w'))
                pending = [item for item in pending if item != (condition_index, suite)]
                for gpu in gpus:
                    lease = claim_gpu(gpu)
                    if lease is not None:
                        leases[(condition_index, suite, gpu)] = lease
            elif validation.exists():
                pending = [item for item in pending if item != (condition_index, suite)]
            else:
                pending.insert(0, (condition_index, suite))

    def prune_validated():
        remaining = []
        for condition_index, suite in pending:
            validation = RESULTS / 'closed_loop' / conditions[condition_index]['name'] / suite / 'run/validated.json'
            if not validation.exists():
                remaining.append((condition_index, suite))
                continue
            saved = json.loads(validation.read_text())
            if saved.get('condition') != conditions[condition_index]:
                raise RuntimeError('Parallel-verify resume identity mismatch')
        return remaining

    pending = prune_validated()
    previous_managed = managed_gpus()
    event(event='parallel_verify_scheduler_started', status_path=str(status_path),
          pending=len(pending), gpus_per_worker=2)
    while pending or active:
        currently_managed = managed_gpus()
        if previous_managed - currently_managed:
            next_poll = 0.0
        previous_managed = currently_managed

        for key, item in list(active.items()):
            code = item.process.poll()
            if code is None:
                continue
            item.stream.close()
            if code:
                failures.append({'condition': conditions[item.condition_index]['name'],
                                 'suite': item.suite, 'code': code})
            del active[key]
            for gpu in item.gpus:
                lease = leases.pop((item.condition_index, item.suite, gpu), None)
                if lease is not None:
                    lease.close()
            event(event='parallel_verify_worker_finished', condition=conditions[item.condition_index]['name'],
                  suite=item.suite, gpus=list(item.gpus), code=code)
            next_poll = 0.0
        pending = prune_validated()

        currently_managed = managed_gpus()
        if pending and len(currently_managed) < PROTOCOL['max_workers'] and time.monotonic() >= next_poll:
            available = [gpu for gpu in free_gpus() if gpu not in currently_managed]
            next_poll = time.monotonic() + PROTOCOL['no_idle_poll_seconds']
            while len(available) >= 2 and pending:
                pair = (available.pop(0), available.pop(0))
                pair_leases = [claim_gpu(gpu) for gpu in pair]
                if any(lease is None for lease in pair_leases):
                    for lease in pair_leases:
                        if lease is not None:
                            lease.close()
                    continue
                condition_index, suite = pending.pop(0)
                output = RESULTS / 'closed_loop' / conditions[condition_index]['name'] / suite
                (output / 'logs').mkdir(parents=True, exist_ok=True)
                stream = (output / 'logs/parallel_worker.log').open('a')
                command = [
                    str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.run_parallel_verify',
                    '--conditions', str(args.conditions.resolve()), '--worker',
                    '--condition-index', str(condition_index), '--suite', suite,
                    '--gpus', ','.join(map(str, pair)),
                ]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                active[(condition_index, suite)] = LiveWorker(process, condition_index, suite, pair, stream)
                for gpu, lease in zip(pair, pair_leases):
                    leases[(condition_index, suite, gpu)] = lease
                event(event='parallel_verify_worker_started', condition=conditions[condition_index]['name'],
                      suite=suite, gpus=list(pair), pid=process.pid)

        dump_json(status_path, {
            'pid': os.getpid(), 'pending': pending, 'failures': failures,
            'active': [
                {'gpu': item.gpus[0], 'gpus': list(item.gpus), 'pid': item.process.pid,
                 'condition': conditions[item.condition_index]['name'], 'suite': item.suite}
                for item in active.values()
            ],
            'conditions': [condition['name'] for condition in conditions],
            'complete': not pending and not active and not failures,
        })
        if pending or active:
            time.sleep(2)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
