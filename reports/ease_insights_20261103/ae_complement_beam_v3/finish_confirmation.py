"""CPU-only full360 acceptance; never allocates a GPU or retries an outcome."""
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
REPORT = Path(__file__).resolve().parent
STUDY = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v3/confirmation'
PYTHON = ROOT / 'policies/realtime-vla-flash/.venv/bin/python'


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(manifest_pin, source_pin):
    if sha(__file__) != source_pin or sha(STUDY / 'manifest.json') != manifest_pin:
        raise ValueError('Confirmation driver or frozen manifest changed')
    from preparations.ease_insights_20261103.ae_complement_beam_v3 import common
    manifest = common.read_manifest(STUDY / 'manifest.json', verify_sources=True)
    if common.stage_for(manifest) != 'confirm' or (len(manifest['planned_conditions']) != 3 or manifest['planned_conditions'][0] != 'full_d10'):
        raise ValueError('Not the frozen independent confirmation')
    return manifest


def progress(identity, manifest):
    counts = {cid: sum(p.parent.name.startswith('init') and p.parent.name[4:].isdigit()
                      for p in (STUDY / cid / 'episodes').glob('**/episode_receipt.json')) for cid in manifest['planned_conditions']}
    markers = len(list(STUDY.glob('*/confirm_shard*of02_complete.json')))
    attention = any(read(p)['phase'] == 'attention_holding' for p in (ROOT / 'results/ease_insights_20261103/resource_pool_v4/sessions').glob('*/status.json'))
    value = dict(time=time.time(), manifest_hash=manifest['manifest_hash'], observer=identity, completed=counts,
                 new_cases=sum(counts.values()), expected_cases=360, completed_shards=markers, expected_shards=6,
                 attention=attention, all_complete=all(n == 120 for n in counts.values()) and markers == 6,
                 successes_read_for_partial_analysis=False)
    from preparations.idea1_v3_1.common import atomic_json
    atomic_json(REPORT / 'confirmation_progress.json', value)
    return value


def run(manifest_pin, source_pin, wait):
    manifest = verify(manifest_pin, source_pin)
    lock = (REPORT / 'confirmation_completion.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(manifest, manifest_pin, source_pin, wait)
    finally:
        lock.close()


def run_locked(manifest, manifest_pin, source_pin, wait):
    target = REPORT / 'confirmation_pipeline_acceptance.json'
    if target.exists():
        old = read(target)
        if old['status'] != 'pass' or any(sha(p) != pin for p, pin in old['artifacts'].items()):
            raise ValueError('Prior confirmation acceptance changed')
        return dict(status='already_complete', conclusion=old['conclusion'])
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19],
                    source_sha256=source_pin, manifest_file_sha256=manifest_pin, time=time.time(), CPU_only=True, GPU_jobs_launched=False)
    from preparations.idea1_v3_1.common import atomic_json
    atomic_json(REPORT / 'confirmation_driver.json', identity)
    last = None
    while True:
        value = progress(identity, manifest)
        if (value['new_cases'] // 10, value['completed_shards'], value['attention']) != last:
            print(json.dumps({k: v for k, v in value.items() if k != 'observer'}), flush=True)
            last = (value['new_cases'] // 10, value['completed_shards'], value['attention'])
        if value['attention']:
            raise RuntimeError('Technical attention; no retry, exclusion or partial significance analysis')
        if value['all_complete']:
            break
        if not wait:
            return dict(status='pending', new_cases=value['new_cases'])
        time.sleep(2)
    verify(manifest_pin, source_pin)
    commands = [
        [str(PYTHON), '-B', str(REPORT / 'confirmation_audit.py')],
        [str(PYTHON), '-B', '-m', 'preparations.ease_insights_20261103.ae_complement_beam_v3.analyze', '--root', str(STUDY), '--output', str(STUDY / 'analysis')],
        [str(PYTHON), '-B', str(REPORT / 'confirmation_statistics_audit.py')],
    ]
    with (REPORT / 'confirmation_acceptance_pipeline.log').open('a') as log:
        for command in commands:
            log.write(json.dumps(dict(time=time.time(), command=command)) + '\n')
            log.flush()
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    cases = read(REPORT / 'confirmation_acceptance.json')
    statistics = read(REPORT / 'confirmation_independent_statistics.json')
    analysis = read(STUDY / 'analysis/analysis.json')
    if (cases['status'] != 'pass' or cases['new_cases'] != 360 or statistics['status'] != 'pass' or
            statistics['differences'] or not analysis['complete'] or analysis['valid_episodes'] != 360 or
            statistics['conclusion'] != analysis['conclusion']):
        raise ValueError('Complete independent confirmation acceptance failed')
    if any(v['manifest_hash'] != manifest['manifest_hash'] for v in [cases, statistics, analysis]):
        raise ValueError('Mixed confirmation identities')
    verify(manifest_pin, source_pin)
    result = dict(status='pass', time=time.time(), manifest_hash=manifest['manifest_hash'], conclusion=analysis['conclusion'],
                  new_cases=360, preserved_control_cells=0, source_pins=len(manifest['source_files']),
                  source=dict(path=str(Path(__file__).resolve()), sha256=source_pin),
                  artifacts={str(p): sha(p) for p in [STUDY / 'manifest.json', REPORT / 'confirmation_acceptance.json',
                             REPORT / 'confirmation_independent_statistics.json', STUDY / 'analysis/analysis.json']},
                  success_counts=statistics['success_counts'], directions=statistics['directions'],
                  quality_eligibility=statistics['candidate_qualified'], native_cost_ratios=statistics['cost_ratios'],
                  scientific_records_modified=False, goal_complete=False,
                  scope='All360 fresh cases and independent fixed-direction arithmetic. Overall goal completion still requires a separate evidence audit.')
    atomic_json(target, result)
    return {k: v for k, v in result.items() if k not in ['artifacts', 'directions']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.manifest_sha256, args.source_sha256, args.wait)), flush=True)
