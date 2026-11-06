"""Immutable replay identities; replay units are not physical episodes."""
import itertools
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT, SUITES, atomic_json, digest, lock
from preparations.idea1_v3_1.common import read_manifest as original_read_manifest
from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import sha

MODULE = 'preparations.ease_insights_20261103.ae_complement_beam_v1'
MODEL_PYTHON = ROOT / 'policies/realtime-vla-flash/.venv/bin/python'
OUTPUT = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v1'
MASKS = [list(v) for v in itertools.combinations(range(1, 19), 2)]
CONDITION = dict(id='joint16_d2', AE_keep=16, D=2, H=50, evaluated_prefix=5,
                 VLM_keep=18, vision_keep=27, cameras=['agentview', 'wrist'])


def require(value, message):
    if not value:
        raise ValueError(message)


def read_manifest(path, verify_sources=False):
    value = original_read_manifest(path, verify_sources)
    require(value['revision'] == 'ease-AE16-joint-mask-screen-v1' and value['mode'] == 'offline_snapshot_replay', 'Wrong replay revision')
    require(value['masks'] == MASKS and value['noise_seeds'] == [12345, 23456, 34567] and value['conditions'] == [CONDITION], 'Frozen replay matrix differs')
    require(value['planned_conditions'] == ['joint16_d2'] and value['physical_episodes'] == 0 and len(value['observations']) == 110, 'Replay budget differs')
    require(value['normalization']['sha256'] == sha(value['normalization']['path']), 'Current normalization changed')
    return value


def validate_output(output, manifest):
    require(Path(output).resolve() == OUTPUT and manifest['output_root'] == str(OUTPUT), 'Replay output differs')


def planned_config(manifest, cid):
    require(cid == CONDITION['id'] and manifest['conditions'] == [CONDITION], 'Wrong replay condition')
    return dict(CONDITION)


def stage_for(manifest):
    require(manifest['mode'] == 'offline_snapshot_replay', 'Not an offline screen')
    return 'screen'


def cases(manifest, stage=None, shard_index=0, shards=1):
    require(stage in (None, 'screen') and type(shards) is int and shards in (1, 2) and 0 <= shard_index < shards, 'Wrong replay shard')
    return [dict(index=i, sample_id=o['sample_id'], suite=o['suite'], task=o['task'], phase=o['phase'])
            for i, o in enumerate(manifest['observations']) if (SUITES.index(o['suite']) * 10 + o['task']) % shards == shard_index]


def runtime_revision(manifest):
    return dict(mode='offline_snapshot_replay', manifest_hash=manifest['manifest_hash'], physical_episodes=0,
                scheduler_unit='one archived snapshot, three repeated noise seeds, all153 masks plus a fresh denseD2 reference')


def replay_dir(output, cid, scene):
    require(cid == CONDITION['id'], 'Wrong replay condition')
    return Path(output) / cid / 'replays' / scene['sample_id']


def load_observation(row):
    import numpy as np
    require(sha(row['path']) == row['sha256'], 'Archived snapshot changed')
    with np.load(row['path'], allow_pickle=False) as data:
        obs = {k: data[k].copy() for k in ('agent_image', 'wrist_image', 'robot_state')}
    for name in ('agent_image', 'wrist_image'):
        require(obs[name].shape == (224, 224, 3) and obs[name].dtype == np.uint8, 'Expected already transformed uint8 images')
    require(obs['robot_state'].shape == (8,) and np.isfinite(obs['robot_state']).all(), 'Expected direct finite8D state')
    return obs
