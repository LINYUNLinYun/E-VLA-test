"""Experiment 10 protocol and CPU-only helpers (also usable by the LIBERO py3.8 client)."""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get('EXP11_RESULTS_ROOT', str(ROOT / 'results/experiment10_phase_draft_20261009'))).resolve()
SOURCE = ROOT / 'results/experiment9_standard_vlm_pruning_20261004'
CONVERTED = Path('/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted')
SUITES = ('libero_object', 'libero_spatial', 'libero_goal', 'libero_10')
PHASE_MAP = {'initial': 'approach', 'approach': 'approach', 'contact': 'contact_grasp',
             'grasp_or_manipulate': 'contact_grasp', 'transport': 'transport',
             'goal_alignment': 'alignment_release', 'release_or_final_action': 'alignment_release'}
PHASES = ('approach', 'contact_grasp', 'transport', 'alignment_release')
INIT_INDICES = tuple(range(5))
NOISE_SEEDS = (12345, 23456, 34567)
HORIZONS = (1, 2, 4, 8, 12)
PROTOCOL = json.loads(os.environ['EXP11_PROTOCOL_JSON']) if os.environ.get('EXP11_PROTOCOL_JSON') else {
            'schema': 1, 'experiment_id': 'experiment10', 'init_indices': INIT_INDICES, 'env_seed': 7, 'warmup': 10,
            'replan': 12, 'noise_seeds': NOISE_SEEDS, 'horizons': HORIZONS,
            'split': {'search': list(range(6)), 'validation': [6, 7], 'evaluation': [8, 9]},
            'layer_skip_semantics': 'skip hidden updates; preserve each layer KV shell',
            'max_workers': 4, 'no_idle_poll_seconds': 300}


def claim_gpu(gpu):
    """Atomic experiment-local ownership; CUDA process checks still protect external users."""
    directory = RESULTS / 'gpu_leases'
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / ('gpu%s.lock' % gpu)).open('a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def managed_gpus():
    """Live owned worker handles, used for completion events rather than GPU polling."""
    occupied = set()
    # Include custom suffixes such as ``attn16_multi_init_status_manifestfix.json``;
    # otherwise a live worker can hold a lease while schedulers fail to count its GPUs.
    filenames = [p.name for p in RESULTS.glob('*status*.json')]
    filenames.append('search/progress.json')
    for filename in filenames:
        path = RESULTS / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        active = data.get('active') or []
        workers = data.get('workers') or []
        if isinstance(active, dict):
            active = [active]
        elif not isinstance(active, list):
            active = []
        if isinstance(workers, dict):
            workers = [workers]
        elif not isinstance(workers, list):
            workers = []
        for record in active + workers:
            if not isinstance(record, dict):
                continue
            try:
                state = (Path('/proc') / str(record['pid']) / 'stat').read_text().split(') ', 1)[1][0]
                if state not in ('Z', 'X'):
                    if 'gpu' in record:
                        occupied.add(int(record['gpu']))
                    occupied.update(int(gpu) for gpu in record.get('gpus', []))
            except (FileNotFoundError, KeyError):
                continue
    return occupied


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def json_lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def split(task_id):
    return 'search' if task_id < 6 else 'validation' if task_id < 8 else 'evaluation'


def select_steps(telemetry, inference_steps):
    """First/middle/last *replanning* observation per phase; never sample the terminal state."""
    groups = {p: [] for p in PHASES}
    for row in telemetry:
        step = int(row['step'])
        phase = PHASE_MAP.get(row.get('semantic_phase'))
        if phase and step in inference_steps and not row.get('done_after_step', False):
            groups[phase].append(step)
    selected = {}
    missing = []
    for phase, steps in groups.items():
        if not steps:
            missing.append(phase)
        else:
            for ordinal, step in enumerate(sorted({steps[0], steps[len(steps)//2], steps[-1]})):
                selected[step] = (phase, ordinal)
    return selected, missing
