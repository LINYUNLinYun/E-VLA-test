"""Frozen Experiment 11 protocol shared with the existing execution backend."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/experiment11_layer_skip_prefix_20261010'
SOURCE = ROOT / 'results/experiment10_phase_draft_20261009'
CONVERTED = Path('/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted')
SUITES = ('libero_object', 'libero_spatial', 'libero_goal', 'libero_10')
INIT_INDICES = tuple(range(5))
PROTOCOL = {
    'schema': 1,
    'experiment_id': 'experiment11_layer_skip_prefix_20261010',
    'suites': list(SUITES),
    'task_ids': list(range(10)),
    'init_indices': list(INIT_INDICES),
    'env_seed': 7,
    'warmup': 10,
    'action_horizon': 50,
    'replan': 5,
    'denoise_steps_primary': 3,
    'denoise_steps_diagnostic': 10,
    'noise_seed_rule': 'SeedSequence(12345, episode_idx, replan_idx), float32[50,32]',
    'layer_skip_semantics': 'skip hidden updates; preserve every layer KV shell; Action Expert full18',
    'max_workers': 4,
    'no_idle_poll_seconds': 300,
    'episode_key': ['suite', 'task_id', 'official_init_index'],
}


def environment():
    import os
    env = os.environ.copy()
    env['EXP11_RESULTS_ROOT'] = str(RESULTS)
    env['EXP11_PROTOCOL_JSON'] = json.dumps(PROTOCOL, separators=(',', ':'))
    env['EXP11_METADATA_KEY'] = 'experiment11'
    env['EXP11_SOURCE_RESULTS'] = str(SOURCE)
    return env
