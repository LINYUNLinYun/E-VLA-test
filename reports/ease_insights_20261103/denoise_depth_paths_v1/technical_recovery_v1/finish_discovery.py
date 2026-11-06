"""Complete-only acceptance with the separately pinned technical recovery auditors."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/nfs/embodied/benchmark')
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
STUDY = ROOT / 'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'
PYTHON = ROOT / 'policies/realtime-vla-flash/.venv/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_amendment(expected):
    path = HERE / 'acceptance_amendment.json'
    if sha(path) != expected:
        raise ValueError('Technical acceptance amendment changed')
    value = read(path)
    for name, pin in value['source_files'].items():
        if sha(ROOT / name) != pin:
            raise ValueError('Technical acceptance source changed: ' + name)
    if sha(STUDY / 'manifest.json') != value['scientific_manifest_file_sha256']:
        raise ValueError('Frozen scientific manifest changed')
    return value


def ready():
    value = read(REPORT / 'progress.json')
    return (value['all_complete'] is True and value['total_new'] == 320 and
            value['completed_shard_markers'] == 8 and not value['attention']), value


def run(amendment_sha256):
    amendment = verify_amendment(amendment_sha256)
    from preparations.ease_insights_20261103.denoise_depth_paths_v1 import common
    target = REPORT / 'discovery_pipeline_acceptance.json'
    if target.exists():
        prior = read(target)
        if (prior['status'] != 'pass' or prior.get('technical_acceptance_amendment_sha256') != amendment_sha256 or
                any(sha(p) != h for p, h in prior['artifacts'].items())):
            raise ValueError('Prior acceptance changed')
        return dict(status='already_complete', selection_status=prior['selection_status'])
    ok, progress = ready()
    if not ok:
        return dict(status='pending', total_new=progress['total_new'])
    manifest = common.read_manifest(STUDY / 'manifest.json', verify_sources=True)
    analysis = STUDY / 'analysis/analysis.json'
    commands = [
        [str(PYTHON), '-B', str(HERE / 'discovery_audit.py')],
        [str(PYTHON), '-B', '-m', 'preparations.ease_insights_20261103.denoise_depth_paths_v1.selection',
         '--root', str(STUDY), '--output', str(analysis.parent)],
        [str(PYTHON), '-B', str(HERE / 'statistics_audit.py'), '--analysis', str(analysis),
         '--case-audit', str(REPORT / 'discovery_acceptance.json'), '--output', str(REPORT / 'independent_statistics.json')],
    ]
    with (HERE / 'acceptance_pipeline.log').open('a') as log:
        for command in commands:
            log.write(json.dumps(dict(time=time.time(), command=command)) + '\n')
            log.flush()
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    cases, statistics, value = read(REPORT / 'discovery_acceptance.json'), read(REPORT / 'independent_statistics.json'), read(analysis)
    if (cases['status'] != 'pass' or statistics['status'] != 'pass' or statistics['differences'] != [] or
            not value['complete'] or value['new_discovery_episodes'] != 320 or value['preserved_reference_episodes'] != 160):
        raise ValueError('Complete acceptance failed')
    if any(v['manifest_hash'] != manifest['manifest_hash'] for v in (cases, statistics, value)):
        raise ValueError('Mixed scientific identities')
    if any(v['technical_recovery']['plan_sha256'] != amendment['technical_resume_plan_sha256'] for v in (cases, statistics)):
        raise ValueError('Mixed technical recovery identities')
    verify_amendment(amendment_sha256)
    from preparations.ease_insights_20261103.denoise_depth_paths_v1.selection import freeze
    selection = freeze(analysis.parent, value)
    result = dict(status='pass', time=time.time(), manifest_hash=manifest['manifest_hash'], selection_status=value['selection']['status'],
                  new_cases=320, preserved_control_cells=160, preserved_controls_are_new_replications=False, source_pins=len(manifest['source_files']),
                  technical_acceptance_amendment_sha256=amendment_sha256,
                  artifacts={str(p): sha(p) for p in (STUDY / 'manifest.json', REPORT / 'discovery_acceptance.json',
                             REPORT / 'independent_statistics.json', analysis, selection, HERE / 'acceptance_amendment.json')},
                  condition_counts={r['condition']: r['successes'] for r in value['condition_summaries'] if r['level'] == 'all'},
                  source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)), scientific_records_modified=False,
                  goal_complete=False, confirmation_launched=False,
                  next='Prepare independent confirmation only if selected; otherwise preserve no_candidate and continue the overall goal')
    common.atomic_json(target, result)
    return {k: v for k, v in result.items() if k not in ('artifacts', 'source')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--amendment-sha256', required=True)
    args = parser.parse_args()
    verify_amendment(args.amendment_sha256)
    lock = (REPORT / 'discovery_completion.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.wait:
        while not ready()[0]:
            _, progress = ready()
            if progress['attention']:
                raise RuntimeError('Science needs attention; do not retry or select')
            time.sleep(2)
    print(json.dumps(run(args.amendment_sha256)), flush=True)
