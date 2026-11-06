"""Real, paired multi-GPU combination oracle. Imports CUDA only inside owned workers."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
import traceback

import numpy as np

from scripts.experiment10.common import RESULTS, ROOT, SUITES, NOISE_SEEDS, PROTOCOL, dump_json, sha256, claim_gpu, managed_gpus
from scripts.experiment10.numerics import action_metrics
from scripts.experiment10.run_pipeline import free_gpus, event


def load_entries(role, coarse=False, phase=None):
    entries = []
    for suite in SUITES:
        source = json.loads((RESULTS / 'observations' / suite / 'manifest.json').read_text())
        for entry in source['observations']:
            if entry['split'] != role or phase and entry['semantic_phase'] != phase:
                continue
            if coarse and (entry['episode_idx'] != 0 or entry['ordinal'] != 0):
                continue
            entries.append(entry)
    return sorted(entries, key=lambda r: r['sample_id'])


def cache_path(mask, role, coarse, phase):
    entries = load_entries(role, coarse, phase)
    probes = [json.loads((RESULTS / 'probe' / s / 'summary.json').read_text()) for s in SUITES]
    weights = {p['checkpoint_sha256'] for p in probes}
    if len(weights) != 1 or not all(p['complete'] for p in probes):
        raise RuntimeError('Completed probe weights are inconsistent')
    evidence = {'mask': mask, 'role': role, 'coarse': coarse, 'phase': phase, 'protocol': PROTOCOL,
                'entries': [(r['sample_id'], r['sha256']) for r in entries],
                'checkpoint_sha256': next(iter(weights)),
                'code': [sha256(Path(__file__)), sha256(Path(__file__).with_name('backend.py')),
                         sha256(Path(__file__).with_name('numerics.py'))]}
    digest = hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest()
    return RESULTS / 'search' / 'cache' / (digest + '.json'), entries, evidence


def aggregate(rows):
    """Episode -> task/phase -> task; task-equal means and descriptive within-task p95."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row['suite'], row['task_id'], row['episode_idx'], row['phase'], row['horizon'])].append(row)
    task_phase = defaultdict(list)
    for (suite, task, init, phase, horizon), values in cells.items():
        task_phase[(suite, task, phase, horizon)].append((
            np.mean([v['normalized_continuous_rmse'] for v in values]),
            np.quantile([v['normalized_continuous_rmse'] for v in values], .95),
            np.mean([v['gripper_flip_rate'] for v in values]),
            np.median([v['paired_cuda_ratio'] for v in values])))
    tasks = defaultdict(list)
    for (suite, task, phase, horizon), values in task_phase.items():
        tasks[(suite, task, horizon)].append(np.mean(values, axis=0))
    summary = {}
    objectives = []
    ratios = []
    for horizon in (1, 2, 4, 8, 12):
        values = [np.mean(v, axis=0) for (s, t, h), v in tasks.items() if h == horizon]
        if not values:
            raise ValueError('No finite horizon observations')
        means = np.mean(values, axis=0)
        summary[str(horizon)] = {'task_count': len(values), 'mean_rmse': float(means[0]),
                                 'mean_within_episode_p95_rmse': float(means[1]),
                                 'mean_gripper_flip': float(means[2]), 'paired_cuda_ratio': float(means[3])}
        objectives.extend(means[:3].tolist())
        ratios.append(means[3])
    objectives.append(float(np.mean(ratios)))
    if not np.isfinite(objectives).all():
        raise ValueError('Nonfinite measured objectives')
    return objectives, summary


def worker_main(gpu, requests, responses, stop):
    os.environ.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'JAX_PLATFORMS': 'cpu',
                      'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi'})
    from scripts.experiment10.backend import Backend
    import torch
    torch.set_num_threads(4)
    metadata = []
    for entry in load_entries('search') + load_entries('validation'):
        with np.load(entry['path'], allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json'].item()))
        if not any(m['suite'] == meta['suite'] and m['task_id'] == meta['task_id'] for m in metadata):
            metadata.append(meta)
    try:
        backend = Backend(metadata)
        while not stop.is_set():
            task = requests.get()
            if task is None:
                return
            key, mask, role, coarse, phase = task
            path, entries, evidence = cache_path(mask, role, coarse, phase)
            if evidence['checkpoint_sha256'] != backend.weight_hash:
                raise RuntimeError('Model weights differ from the registered phase probes')
            if path.exists():
                saved = json.loads(path.read_text())
                if saved['checkpoint_sha256'] != backend.weight_hash or saved['evidence'] != json.loads(json.dumps(evidence)):
                    raise RuntimeError('Combination cache provenance mismatch')
                responses.put((key, saved['objectives'], None))
                continue
            descriptions = {}
            for entry in entries:
                with np.load(entry['path'], allow_pickle=False) as data:
                    descriptions[entry['sample_id']] = json.loads(str(data['metadata_json'].item()))
            entries.sort(key=lambda e: (len(backend.embeddings[e['suite'], descriptions[e['sample_id']]['task_description']]),
                                        e['sample_id']))
            rows, cross_instance = [], []
            started = time.monotonic()
            for entry in entries:
                if stop.is_set():
                    return
                if sha256(entry['path']) != entry['sha256']:
                    raise RuntimeError('Combination input changed')
                meta = descriptions[entry['sample_id']]
                with np.load(entry['path'], allow_pickle=False) as data:
                    observation = {'observation/image': data['agent_image'], 'observation/wrist_image': data['wrist_image'],
                                   'observation/state': data['robot_state'], 'prompt': str(data['prompt'].item())}
                anchor = RESULTS / 'probe' / entry['suite'] / 'samples' / (entry['sample_id'] + '.npz')
                with np.load(anchor, allow_pickle=False) as reference:
                    for seed_index in range(1 if coarse else 3):
                        rng = np.random.default_rng(np.random.SeedSequence(
                            [NOISE_SEEDS[seed_index], int(entry['sha256'][:8], 16)]))
                        expected_noise = rng.standard_normal((50, 32), dtype=np.float32)
                        if not np.array_equal(expected_noise, reference['noise'][seed_index]):
                            raise RuntimeError('Noise provenance mismatch')
                        backend.prepare(observation, meta, reference['noise'][seed_index])
                        baseline, baseline_command, baseline_timing = backend.infer()
                        candidate, command, timing = backend.infer(mask)
                        repeated, _, _ = backend.infer()
                        if not np.array_equal(baseline, repeated):
                            raise RuntimeError('Same-instance full baseline changed during mask evaluation')
                        cross_instance.append(float(np.max(np.abs(baseline - reference['baseline_raw'][seed_index]))))
                        for metrics in action_metrics(candidate, baseline, command, baseline_command):
                            rows.append({'sample_id': entry['sample_id'], 'suite': entry['suite'],
                                         'task_id': entry['task_id'], 'episode_idx': entry['episode_idx'],
                                         'phase': entry['semantic_phase'], 'noise_seed': NOISE_SEEDS[seed_index],
                                         'paired_cuda_ratio': timing['cuda_policy_ms'] / baseline_timing['cuda_policy_ms'],
                                         **metrics})
            objectives, summary = aggregate(rows)
            saved = {'complete': True, 'evidence': evidence, 'checkpoint_sha256': backend.weight_hash,
                     'objectives': objectives, 'horizon_summary': summary, 'rows': rows, 'gpu': gpu,
                     'seconds': time.monotonic() - started, 'sample_count': len(entries),
                     'same_instance_reference': True, 'not_a_dedicated_timing_benchmark': True,
                     'max_cross_instance_baseline_difference': max(cross_instance, default=0.)}
            dump_json(path, saved)
            responses.put((key, objectives, None))
    except Exception:
        responses.put((None, None, traceback.format_exc()))


class GpuPool:
    """Grow workers on genuinely idle cards; five-minute polling without completion events."""
    def __init__(self):
        self.context = mp.get_context('spawn')
        self.requests, self.responses = self.context.Queue(), self.context.Queue()
        self.stop = self.context.Event()
        self.workers, self.next_poll, self.counter = {}, 0., 0
        self.leases = {}
        self.previous_managed = managed_gpus()

    def acquire(self):
        current = managed_gpus()
        if self.previous_managed - current:
            self.next_poll = 0.
        self.previous_managed = current
        if time.monotonic() < self.next_poll:
            return
        if len(current | set(self.workers)) >= 4:
            return
        for gpu in free_gpus():
            if gpu in self.workers or len(self.workers) >= 4:
                continue
            lease = claim_gpu(gpu)
            if lease is None:
                continue
            self.leases[gpu] = lease
            process = self.context.Process(target=worker_main, args=(gpu, self.requests, self.responses, self.stop))
            process.start()
            self.workers[gpu] = process
            event(event='search_worker_started', gpu=gpu, pid=process.pid)
        self.next_poll = time.monotonic() + 300

    def evaluate(self, masks, *, role='search', coarse=True, phase=None):
        masks = list(dict.fromkeys(tuple(m) for m in masks))
        if not masks:
            return {}
        pending, scores = {}, {}
        for mask in masks:
            path, _, _ = cache_path(mask, role, coarse, phase)
            if path.exists():
                cached = json.loads(path.read_text())
                _, _, evidence = cache_path(mask, role, coarse, phase)
                if not cached.get('complete') or cached.get('evidence') != json.loads(json.dumps(evidence)):
                    raise RuntimeError('Root combination cache identity mismatch')
                scores[mask] = cached['objectives']
            else:
                self.counter += 1
                pending[self.counter] = mask
                self.requests.put((self.counter, mask, role, coarse, phase))
        while pending:
            self.acquire()
            for gpu, process in self.workers.items():
                if not process.is_alive():
                    raise RuntimeError('Search worker exited: gpu %s code %s' % (gpu, process.exitcode))
            try:
                key, value, error = self.responses.get(timeout=2)
            except queue.Empty:
                continue
            if error:
                raise RuntimeError(error)
            if key not in pending:
                raise RuntimeError('Unknown mask response')
            mask = pending.pop(key)
            scores[mask] = value
            dump_json(RESULTS / 'search' / 'progress.json', {'completed_batch': len(scores), 'pending_batch': len(pending),
                         'last_mask': mask, 'role': role, 'coarse': coarse, 'phase': phase,
                         'workers': [{'gpu': g, 'pid': p.pid} for g, p in self.workers.items()]})
            print(json.dumps({'event': 'mask_measured', 'mask': mask, 'role': role, 'coarse': coarse,
                              'phase': phase, 'remaining_batch': len(pending)}), flush=True)
        return scores

    def close(self):
        self.stop.set()
        for _ in self.workers:
            self.requests.put(None)
        for gpu, process in self.workers.items():
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            event(event='search_worker_stopped', gpu=gpu, pid=process.pid, code=process.exitcode)
            self.leases.pop(gpu).close()
