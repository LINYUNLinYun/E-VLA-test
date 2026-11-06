"""Audit frozen inputs and source artifacts before any Experiment 11 GPU run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from scripts.experiment11.build_candidates import sha256
from scripts.experiment11.protocol import CONVERTED, PROTOCOL, RESULTS, SOURCE, SUITES
from scripts.experiment10.common import dump_json


PROMPT_DIR = {'libero_object': 'draft_object', 'libero_spatial': 'draft_spatial',
              'libero_goal': 'draft', 'libero_10': 'draft_10'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-observation-hashes', action='store_true',
                        help='Read and hash all historical sampled observations')
    args = parser.parse_args()
    protocol_dir = RESULTS / 'protocol'
    manifest_path = protocol_dir / 'candidate_manifest.json'
    config_path = protocol_dir / 'closed_loop_config_formal.json'
    manifest = json.loads(manifest_path.read_text())
    config = json.loads(config_path.read_text())
    errors = []
    candidates = manifest['candidates']
    counts = {family: sum(row['family'] == family for row in candidates)
              for family in ('V16', 'V15', 'V14', 'V13')}
    if counts != {'V16': 10, 'V15': 8, 'V14': 6, 'V13': 5}:
        errors.append('candidate family counts mismatch')
    conditions = config['conditions']
    if len(conditions) != 31 or len({c['name'] for c in conditions}) != 31:
        errors.append('expected 31 unique conditions including two D3 references')
    for condition in conditions:
        if (condition['steps'] != 3 or condition['horizon'] != 5 or condition.get('replan_steps') != 5 or
                condition['mode'] != 'static' or condition['checkpoint_sha256'] != manifest['checkpoint_sha256']):
            errors.append('condition violates D3 / horizon50 / execute5 / checkpoint: ' + condition['name'])
        skip = condition['skip']
        if len(skip) != len(set(skip)) or any(x in (1, 17, 18) for x in skip) or any(x < 1 or x > 18 for x in skip):
            errors.append('invalid/protected skip set: ' + condition['name'])
        if condition['name'].startswith('Full18') and skip:
            errors.append('Full18 unexpectedly skips VLM layers')
    checkpoint = CONVERTED / 'base/base_weights.pkl'
    norm_stats = CONVERTED / 'base/assets/physical-intelligence/libero/norm_stats.json'
    if not checkpoint.is_file() or sha256(checkpoint) != manifest['checkpoint_sha256']:
        errors.append('converted model checkpoint hash mismatch')
    if not norm_stats.is_file():
        errors.append('LIBERO normalization asset missing')
    source_manifests, observation_hash_audit = {}, {'checked': 0, 'errors': []}
    prompt_assets = {}
    for suite in SUITES:
        path = SOURCE / 'observations' / suite / 'manifest.json'
        if not path.is_file():
            errors.append('source observation manifest missing: ' + suite)
            continue
        source_manifest = json.loads(path.read_text())
        source_manifests[suite] = {'path': str(path), 'sha256': sha256(path),
                                   'complete': source_manifest.get('complete'),
                                   'episode_count': source_manifest.get('episode_count'),
                                   'sample_count': source_manifest.get('sample_count')}
        if not source_manifest.get('complete') or source_manifest.get('episode_count') != 50:
            errors.append('source observation coverage incomplete: ' + suite)
        prompts = set()
        for row in source_manifest.get('observations', []):
            obs_path = Path(row['path'])
            if not obs_path.is_file():
                errors.append('source observation missing: ' + str(obs_path))
                continue
            if args.verify_observation_hashes:
                observed_hash = sha256(obs_path)
                observation_hash_audit['checked'] += 1
                if observed_hash != row['sha256']:
                    observation_hash_audit['errors'].append(str(obs_path))
            with __import__('numpy').load(obs_path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata_json'].item()))
                prompts.add(str(metadata['task_description']))
        artifact_root = CONVERTED / PROMPT_DIR[suite]
        prompt_manifest_path = artifact_root / 'manifest.json'
        prompt_manifest = json.loads(prompt_manifest_path.read_text())['prompts']
        task_assets = {}
        for prompt in sorted(prompts):
            if prompt not in prompt_manifest:
                errors.append('prompt embedding missing: %s / %s' % (suite, prompt))
                continue
            embedding = artifact_root / prompt_manifest[prompt]['embed_path']
            if not embedding.is_file():
                errors.append('prompt embedding file missing: ' + str(embedding))
            else:
                task_assets[prompt] = {'path': str(embedding), 'sha256': sha256(embedding)}
        prompt_assets[suite] = {'manifest_path': str(prompt_manifest_path),
                                'manifest_sha256': sha256(prompt_manifest_path),
                                'used_prompt_embeddings': task_assets}
    if observation_hash_audit['errors']:
        errors.append('historical observation content hash mismatch')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        errors.append('ffmpeg/ffprobe unavailable')
    versions = {}
    for name, command in (('ffmpeg', ['ffmpeg', '-version']), ('ffprobe', ['ffprobe', '-version'])):
        try:
            response = subprocess.run(command, check=True, capture_output=True, text=True)
            versions[name] = response.stdout.splitlines()[0]
        except Exception as exc:
            errors.append('%s version check failed: %s' % (name, exc))
    code_paths = [
        Path(__file__), Path(__file__).with_name('build_candidates.py'),
        SOURCE.parent.parent / 'scripts/experiment10/run_closed_loop.py',
        SOURCE.parent.parent / 'scripts/experiment10/serve_policy.py',
        SOURCE.parent.parent / 'scripts/experiment10/backend.py',
        SOURCE.parent.parent / 'scripts/experiment10/client_libero.py',
        SOURCE.parent.parent / 'scripts/spec/spec_client_libero.py',
        SOURCE.parent.parent / 'src/openpi/training/config.py',
        SOURCE.parent.parent / 'src/openpi/transforms.py',
    ]
    code_hashes = {str(path): sha256(path) for path in code_paths if path.is_file()}
    disk = shutil.disk_usage(RESULTS.parent)
    provenance = {
        'experiment_id': PROTOCOL['experiment_id'],
        'protocol': PROTOCOL,
        'candidate_manifest_sha256': sha256(manifest_path),
        'closed_loop_config_sha256': sha256(config_path),
        'checkpoint': {'path': str(checkpoint), 'sha256': manifest['checkpoint_sha256']},
        'normalization_asset': {'path': str(norm_stats), 'sha256': sha256(norm_stats) if norm_stats.is_file() else None},
        'source_observation_manifests': source_manifests,
        'source_observation_hash_audit': observation_hash_audit,
        'prompt_embeddings': prompt_assets,
        'code_sha256': code_hashes,
        'video_tools': versions,
        'disk_bytes': {'total': disk.total, 'used': disk.used, 'free': disk.free},
        'cpu_preflight_errors': errors,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    dump_json(RESULTS / 'provenance/preflight.json', provenance)
    print(json.dumps({'errors': errors, 'candidate_counts': counts, 'condition_count': len(conditions),
                      'episode_count_primary': len(conditions) * 200,
                      'observation_hashes_checked': observation_hash_audit['checked'],
                      'disk_free_bytes': disk.free, 'video_tools': versions,
                      'preflight_path': str(RESULTS / 'provenance/preflight.json')}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
