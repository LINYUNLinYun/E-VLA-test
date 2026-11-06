"""Dedicated same-GPU latency comparison for representative Experiment 10 variants."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from scripts.experiment10.backend import Backend
from scripts.experiment10.common import (ROOT, RESULTS, SUITES, NOISE_SEEDS, PROTOCOL,
                                         claim_gpu, dump_json, managed_gpus, sha256)
from scripts.experiment10.run_pipeline import event, free_gpus


CONDITIONS = (
    ('Full18_D10', (), 10),
    ('Full18_D1', (), 1),
    ('Full18_D2', (), 2),
    ('Full18_D3', (), 3),
    ('Full18_D4', (), 4),
    ('Full18_D5', (), 5),
    ('V16_D10', (9, 10), 10),
    ('V13_D10', (3, 4, 7, 12, 15), 10),
)
PHASES = ('approach', 'contact_grasp', 'transport', 'alignment_release')
REPEATS = 5
WARMUPS = 3


def select_samples(entries):
    """One real observation per phase, preferring validation task 6 then 7."""
    selected = []
    for phase in PHASES:
        candidates = [e for e in entries if e['task_id'] in (6, 7) and e['semantic_phase'] == phase]
        candidates.sort(key=lambda e: (e['task_id'] != 6, e['task_id'], e['episode_idx'], e['ordinal']))
        if candidates:
            selected.append(candidates[0])
    return selected


def identity(entries, backend):
    return {
        'schema': 1,
        'suite': entries[0]['suite'],
        'checkpoint_sha256': backend.weight_hash,
        'conditions': [[name, list(skip), steps] for name, skip, steps in CONDITIONS],
        'noise_seeds': list(NOISE_SEEDS),
        'repeats': REPEATS,
        'warmups': WARMUPS,
        'phases': list(PHASES),
        'sample_role': 'validation_tasks_6_7_timing_only',
        'protocol': json.loads(json.dumps(PROTOCOL)),
        'timing_is_dedicated_benchmark': True,
    }


def benchmark_suite(suite, gpu):
    import torch
    torch.cuda.init()
    entries = json.loads((RESULTS / 'observations' / suite / 'manifest.json').read_text())['observations']
    samples = select_samples(entries)
    if not samples:
        raise RuntimeError('No real validation observations available for %s' % suite)
    metadata = []
    for entry in entries:
        with np.load(entry['path'], allow_pickle=False) as data:
            metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(metadata)
    expected = identity(entries, backend)
    output = RESULTS / 'dedicated_timing' / suite
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261009 + SUITES.index(suite))

    for sample_index, entry in enumerate(samples):
        target = output / (entry['sample_id'] + '.json')
        if target.exists():
            saved = json.loads(target.read_text())
            if any(saved.get(key) != value for key, value in expected.items()):
                raise RuntimeError('Timing resume identity mismatch: %s' % target)
            continue
        if sha256(entry['path']) != entry['sha256']:
            raise RuntimeError('Timing source observation changed: %s' % entry['sample_id'])
        with np.load(entry['path'], allow_pickle=False) as data:
            observation = {
                'observation/image': data['agent_image'],
                'observation/wrist_image': data['wrist_image'],
                'observation/state': data['robot_state'],
                'prompt': str(data['prompt'].item()),
            }
            meta = json.loads(str(data['metadata_json'].item()))

        warm_noise = np.random.default_rng(NOISE_SEEDS[0]).standard_normal((50, 32)).astype(np.float32)
        for _, skip, steps in CONDITIONS:
            backend.prepare(observation, meta, warm_noise)
            backend.infer(skip, steps)
        torch.cuda.synchronize()

        measurements = []
        for noise_seed in NOISE_SEEDS:
            noise = np.random.default_rng(noise_seed).standard_normal((50, 32)).astype(np.float32)
            for repeat in range(REPEATS):
                order = list(range(len(CONDITIONS)))
                rng.shuffle(order)
                for condition_index in order:
                    name, skip, steps = CONDITIONS[condition_index]
                    torch.cuda.synchronize()
                    prepare_started = time.perf_counter()
                    backend.prepare(observation, meta, noise)
                    torch.cuda.synchronize()
                    prepare_ms = (time.perf_counter() - prepare_started) * 1000
                    infer_started = time.perf_counter()
                    _, _, timing = backend.infer(skip, steps)
                    infer_wall_ms = (time.perf_counter() - infer_started) * 1000
                    measurements.append({
                        'condition': name,
                        'skip_layers_1based': list(skip),
                        'denoise_steps': steps,
                        'noise_seed': noise_seed,
                        'repeat': repeat,
                        'prepare_ms': prepare_ms,
                        'cuda_policy_ms': timing['cuda_policy_ms'],
                        'infer_wall_ms': infer_wall_ms,
                        'pipeline_wall_ms': prepare_ms + infer_wall_ms,
                    })
        dump_json(target, {
            **expected,
            'sample_id': entry['sample_id'],
            'task_id': entry['task_id'],
            'episode_idx': entry['episode_idx'],
            'semantic_phase': entry['semantic_phase'],
            'observation_sha256': entry['sha256'],
            'gpu_index': int(gpu),
            'gpu_name': torch.cuda.get_device_name(0),
            'measurements': measurements,
        })
        dump_json(output / 'progress.json', {
            'completed_samples': sample_index + 1,
            'expected_samples': len(samples),
            'last_sample': entry['sample_id'],
            'gpu_index': int(gpu),
        })
        print(json.dumps({'event': 'timing_sample_complete', 'suite': suite,
                          'sample': entry['sample_id'], 'samples': '%s/%s' % (sample_index+1, len(samples))}),
              flush=True)

    rows = []
    cells = {}
    for entry in samples:
        data = json.loads((output / (entry['sample_id'] + '.json')).read_text())
        for row in data['measurements']:
            enriched = {k: data[k] for k in ('sample_id', 'task_id', 'episode_idx', 'semantic_phase',
                                                'gpu_index', 'gpu_name')}
            enriched.update(row)
            rows.append(enriched)
            for metric in ('prepare_ms', 'cuda_policy_ms', 'infer_wall_ms', 'pipeline_wall_ms'):
                cells.setdefault((row['condition'], data['task_id'], data['semantic_phase'], metric), []).append(row[metric])

    metrics = {}
    for condition, _, _ in CONDITIONS:
        metrics[condition] = {}
        for metric in ('prepare_ms', 'cuda_policy_ms', 'infer_wall_ms', 'pipeline_wall_ms'):
            values = np.asarray([r[metric] for r in rows if r['condition'] == condition], dtype=float)
            cell_medians = [float(np.median(v)) for (c, _, _, m), v in cells.items() if c == condition and m == metric]
            metrics[condition][metric] = {
                'n': int(len(values)),
                'median_ms': float(np.median(values)),
                'p95_ms': float(np.quantile(values, .95)),
                'task_phase_equal_median_ms': float(np.median(cell_medians)),
            }
    csv_path = output / 'raw_timings.csv'
    columns = ['condition', 'skip_layers_1based', 'denoise_steps', 'noise_seed', 'repeat',
               'sample_id', 'task_id', 'episode_idx', 'semantic_phase', 'gpu_index', 'gpu_name',
               'prepare_ms', 'cuda_policy_ms', 'infer_wall_ms', 'pipeline_wall_ms']
    temporary = csv_path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    dump_json(output / 'summary.json', {
        **expected,
        'complete': True,
        'sample_count': len(samples),
        'samples': [e['sample_id'] for e in samples],
        'gpu_index': int(gpu),
        'gpu_name': torch.cuda.get_device_name(0),
        'metrics': metrics,
        'limitations': ['Validation tasks 6-7 are timing samples only; not independent task-level latency CIs',
                        'CUDA policy time includes vision, VLM and action decoder as one graph',
                        'Concurrent measurements on other GPUs do not affect within-GPU condition comparisons'],
    })
    print(json.dumps({'event': 'suite_timing_complete', 'suite': suite, 'gpu': gpu,
                      'samples': len(samples), 'output': str(output)}), flush=True)


class AdoptedWorker:
    def __init__(self, pid, completion):
        self.pid = int(pid)
        self.completion = Path(completion)

    def poll(self):
        try:
            state = (Path('/proc') / str(self.pid) / 'stat').read_text().split(') ', 1)[1][0]
        except (FileNotFoundError, IndexError):
            state = 'X'
        if state not in ('Z', 'X'):
            return None
        if self.completion.exists() and json.loads(self.completion.read_text()).get('complete'):
            return 0
        return 1


def live(pid):
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0]
    except (FileNotFoundError, IndexError):
        return False
    return state not in ('Z', 'X')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--status-path', type=Path,
                        default=RESULTS / 'dedicated_timing_status.json')
    args = parser.parse_args()
    if args.worker:
        benchmark_suite(args.suite, args.gpu)
        return
    if not args.launch:
        print(json.dumps({'suites': SUITES, 'conditions': CONDITIONS, 'samples_per_suite': 'up to 4',
                          'repeats': REPEATS, 'noise_seeds': NOISE_SEEDS,
                          'gpu_idle_poll_seconds': PROTOCOL['no_idle_poll_seconds']}, indent=2))
        return

    status_path = args.status_path.resolve()
    if RESULTS.resolve() not in status_path.parents:
        raise ValueError('Scheduler status path must stay inside experiment results')
    lock = status_path.with_name(status_path.name + '.scheduler.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('A timing scheduler already owns %s' % status_path) from exc
    prior = json.loads(status_path.read_text()) if status_path.exists() else None
    if prior and live(prior.get('pid', -1)):
        cmdline = (Path('/proc') / str(prior['pid']) / 'cmdline').read_bytes().replace(b'\0', b' ')
        if b'scripts.experiment10.benchmark_latency' in cmdline and b'--worker' not in cmdline:
            raise RuntimeError('Timing scheduler PID %s is still live' % prior['pid'])
    pending = list(SUITES) if prior is None else list(prior.get('pending', []))
    active, failures, leases = {}, (list(prior.get('failures', [])) if prior else []), {}
    if prior and not prior.get('complete', False):
        for record in prior.get('active', []):
            suite, gpu, pid = record['suite'], int(record['gpu']), int(record['pid'])
            completion = RESULTS / 'dedicated_timing' / suite / 'summary.json'
            if live(pid):
                active[gpu] = (AdoptedWorker(pid, completion), suite, open(os.devnull, 'w'))
                lease = claim_gpu(gpu)
                if lease is not None:
                    leases[gpu] = lease
                event(event='timing_worker_adopted', suite=suite, gpu=gpu, pid=pid)
            elif completion.exists() and json.loads(completion.read_text()).get('complete'):
                if suite in pending:
                    pending.remove(suite)
            elif suite not in pending:
                pending.insert(0, suite)
    pending = [suite for suite in pending if not (
        (RESULTS / 'dedicated_timing' / suite / 'summary.json').exists() and
        json.loads((RESULTS / 'dedicated_timing' / suite / 'summary.json').read_text()).get('complete'))]
    active_suites = {suite for _, suite, _ in active.values()}
    pending = [suite for suite in pending if suite not in active_suites]
    next_poll, previous_managed = 0., managed_gpus()
    event(event='timing_scheduler_started', status_path=str(status_path), resumed=prior is not None,
          adopted=len(active), pending=len(pending))
    while pending or active:
        currently_managed = managed_gpus()
        if previous_managed - currently_managed:
            next_poll = 0.
        previous_managed = currently_managed
        for gpu, (process, suite, stream) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                lease = leases.pop(gpu, None)
                if lease is not None:
                    lease.close()
                del active[gpu]
                next_poll = 0.
                if code:
                    failures.append({'suite': suite, 'code': code})
                event(event='timing_job_finished', suite=suite, gpu=gpu, code=code)
        if (pending and len(currently_managed | set(active)) < PROTOCOL['max_workers'] and
                time.monotonic() >= next_poll):
            available = [gpu for gpu in free_gpus() if gpu not in active]
            next_poll = time.monotonic() + PROTOCOL['no_idle_poll_seconds']
            for gpu in available:
                if not pending:
                    break
                lease = claim_gpu(gpu)
                if lease is None:
                    continue
                suite = pending.pop(0)
                env = os.environ.copy()
                env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_EGL_DEVICE_ID': str(gpu),
                            'MUJOCO_GL': 'egl', 'LIBERO_CONFIG_PATH': str(ROOT / '.libero'),
                            'JAX_PLATFORMS': 'cpu', 'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
                            'PYTHONPATH': ':'.join([str(ROOT / 'src'), str(ROOT),
                                                    '/root/nfs/embodied/benchmark/LIBERO']),
                            'PYTHONUNBUFFERED': '1'})
                command = [str(ROOT / '.venv/bin/python'), '-m',
                           'scripts.experiment10.benchmark_latency', '--worker', '--suite', suite, '--gpu', str(gpu)]
                output = RESULTS / 'dedicated_timing' / suite
                output.mkdir(parents=True, exist_ok=True)
                stream = (output / 'worker.log').open('a')
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, suite, stream)
                leases[gpu] = lease
                event(event='timing_job_started', suite=suite, gpu=gpu, pid=process.pid)
        dump_json(status_path, {
            'pid': os.getpid(), 'pending': pending, 'failures': failures,
            'active': [{'gpu': gpu, 'pid': proc.pid, 'suite': suite}
                       for gpu, (proc, suite, _) in active.items()],
            'complete': not pending and not active and not failures,
        })
        if pending or active:
            time.sleep(2)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
