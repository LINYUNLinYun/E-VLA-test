"""User-approved V14_A03 supplement; separate queue, no environment execution.

Reuse the tested original scheduler via explicit process-local callbacks. Never
change its frozen manifest, status, prediction files, or original analysis.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from pathlib import Path

from scripts.experiment11 import launch_shadow as scheduler
from scripts.experiment11.protocol import RESULTS, ROOT, SUITES
from scripts.experiment10.common import dump_json, sha256

MANIFEST = RESULTS / 'protocol/selected_shadow_supplement_manifest.json'
STATUS = RESULTS / 'experiment11_v14_shadow_supplement_status.json'
OUTPUT = RESULTS / 'shadow_supplement_v14/attempt1_20261011'
ANALYSIS = RESULTS / 'analysis/shadow_supplement_v14/attempt1_20261011'
VARIANT = 'V14_A03'
SKIP = [3, 4, 9, 11]
COUNTS = dict(zip(SUITES, [1466, 1128, 1218, 3167]))
ATTEMPT = 'attempt1_20261011'


def set_attempt(attempt: str):
    global ATTEMPT, STATUS, OUTPUT, ANALYSIS
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        raise ValueError('Invalid attempt id')
    ATTEMPT = attempt
    STATUS = RESULTS / ('experiment11_v14_shadow_supplement_status.json' if attempt == 'attempt1_20261011'
                        else f'experiment11_v14_shadow_supplement_status_{attempt}.json')
    OUTPUT = RESULTS / 'shadow_supplement_v14' / attempt
    ANALYSIS = RESULTS / 'analysis/shadow_supplement_v14' / attempt
    os.environ['EXP11_V14_CALL_LEDGER'] = str(OUTPUT / 'call_budget.sqlite')


def validate_manifest(data: dict) -> None:
    expected = [{'variant_id': VARIANT, 'condition_name': VARIANT + '_D3_R5_FORMAL',
                 'skip_layers': SKIP}]
    if (data.get('admission') != 'user_approved_exception' or
            data.get('selected_new_masks') != expected or data.get('denoise_steps') != 3 or
            data.get('action_horizon') != 50 or data.get('replan_execute_prefix') != 5 or
            data.get('expected_suite_predictions') != COUNTS):
        raise RuntimeError('Only the explicit V14_A03 D3/50/5 exception is authorized')
    for source, digest in data.get('source_hashes', {}).items():
        if sha256(RESULTS / source) != digest:
            raise RuntimeError(f'Supplement frozen source changed: {source}')
    required = {'protocol/candidate_manifest.json', 'analysis/closed_loop_d3/suite_results.csv'}
    required.update(f'closed_loop/{scheduler.BASELINE}/{s}/run/episode_log.json' for s in SUITES)
    if set(data.get('source_hashes', {})) != required:
        raise RuntimeError('Supplement is missing frozen source hashes')


def freeze_manifest() -> None:
    candidate_path = RESULTS / 'protocol/candidate_manifest.json'
    candidates = json.loads(candidate_path.read_text())['candidates']
    candidate = next(r for r in candidates if r['variant_id'] == VARIANT)
    if candidate['skip_layers'] != SKIP:
        raise RuntimeError('V14_A03 mask differs from the existing candidate')
    scores = RESULTS / 'analysis/closed_loop_d3/suite_results.csv'
    with scores.open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['condition'] == VARIANT + '_D3_R5_FORMAL']
    if (len(rows) != 4 or {r['suite'] for r in rows} != set(SUITES) or
            any(int(r['episodes']) != 50 for r in rows) or
            sum(int(r['successes']) for r in rows) != 148):
        raise RuntimeError('V14_A03 148/200 evidence is missing')
    sources = [candidate_path, scores]
    for suite in SUITES:
        if scheduler.expected_prediction_count(suite) != COUNTS[suite]:
            raise RuntimeError(f'Unexpected frozen Full replan count: {suite}')
        sources.append(RESULTS / 'closed_loop' / scheduler.BASELINE / suite / 'run/episode_log.json')
    data = {
        'schema': 1, 'status': 'eligible', 'admission': 'user_approved_exception',
        'user_direction_date': '2026-10-11', 'user_direction': '补充V14的实验4数据',
        'selected_new_masks': [{'variant_id': VARIANT, 'condition_name': VARIANT + '_D3_R5_FORMAL',
                                'skip_layers': SKIP}],
        'overall_successes': 148, 'overall_episodes': 200, 'overall_success_rate': 0.74,
        'denoise_steps': 3, 'action_horizon': 50, 'replan_execute_prefix': 5,
        'expected_suite_predictions': COUNTS, 'expected_variant_predictions': sum(COUNTS.values()),
        'extra_full_parity_predictions': 4, 'environment_steps_executed_by_variant': 0,
        'source_hashes': {str(p.relative_to(RESULTS)): sha256(p) for p in sources},
    }
    if MANIFEST.exists():
        if json.loads(MANIFEST.read_text()) != data:
            raise RuntimeError('Refusing to overwrite a different frozen supplemental manifest')
    else:
        dump_json(MANIFEST, data)


def frozen_jobs():
    status = json.loads((RESULTS / 'experiment11_closed_loop_status.json').read_text())
    audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    audit = json.loads(audit_path.read_text())
    if (not status.get('complete') or status.get('active') or status.get('pending') or
            status.get('failures') or audit.get('condition_suite_runs_validated') != 124 or
            audit.get('episodes_validated') != 6200):
        raise RuntimeError('The original D3 matrix must remain complete and validated')
    validate_manifest(json.loads(MANIFEST.read_text()))
    return ([{'job_id': f'{VARIANT}__{s}', 'variant_id': VARIANT, 'suite': s,
              'skip_layers': SKIP, 'condition_name': VARIANT + '_D3_R5_FORMAL'} for s in SUITES],
            sha256(MANIFEST), sha256(audit_path))


def output_paths(job):
    directory = OUTPUT / job['variant_id'] / job['suite']
    return directory / 'predictions.jsonl', directory / 'validated.json', directory / 'predictions.jsonl.partial'


def validate_job(job):
    predictions, marker_path, partial = output_paths(job)
    if not predictions.is_file() or not marker_path.is_file():
        return False
    m = json.loads(marker_path.read_text())
    parity = m.get('full_recomputed_parity_max_abs_error')
    return bool(m.get('complete') and m.get('variant_id') == VARIANT and
                m.get('suite') == job['suite'] and m.get('skip_layers') == SKIP and
                m.get('condition_name') == job['condition_name'] and m.get('episode_count') == 50 and
                m.get('replan_prediction_count') == COUNTS[job['suite']] and
                m.get('action_horizon') == 50 and m.get('denoise_steps') == 3 and
                m.get('replan_execute_prefix') == 5 and m.get('environment_steps_executed_by_variant') == 0 and
                isinstance(parity, (float, int)) and math.isfinite(parity) and 0 <= parity <= 1e-5 and
                m.get('source_hashes') == {
                    'full_episode_log_sha256': sha256(RESULTS / 'closed_loop' / scheduler.BASELINE /
                                                     job['suite'] / 'run/episode_log.json'),
                    'selected_shadow_manifest_sha256': sha256(MANIFEST),
                    'candidate_manifest_sha256': sha256(RESULTS / 'protocol/candidate_manifest.json')} and
                m.get('predictions_sha256') == sha256(predictions) and not partial.exists())


def child_command(job, gpu):
    return [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment11.launch_shadow_supplement',
            '--worker-suite', job['suite'], '--gpu-index', str(gpu), '--attempt-id', ATTEMPT]


def configure_scheduler():
    from scripts.experiment11.gpu_discovery import throttled_free_gpus
    scheduler.STATUS = STATUS
    scheduler.LOCK = STATUS.with_name(STATUS.name + '.scheduler.lock')
    scheduler.frozen_jobs = frozen_jobs
    scheduler.output_paths = output_paths
    scheduler.validate_job = validate_job
    scheduler.child_command = child_command
    scheduler.LOG_ROOT = RESULTS / 'logs/shadow_supplement_v14' / ATTEMPT
    scheduler.free_gpus = throttled_free_gpus


def analyze():
    configure_scheduler()
    from scripts.experiment11 import analyze_shadow
    analyze_shadow.frozen_jobs = frozen_jobs
    analyze_shadow.validate_job = validate_job
    analyze_shadow.SHADOW_ROOT = OUTPUT
    analyze_shadow.SHADOW_STATUS = STATUS
    analyze_shadow.ANALYSIS = ANALYSIS
    return analyze_shadow.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--worker-suite', choices=SUITES)
    parser.add_argument('--gpu-index', type=int)
    parser.add_argument('--attempt-id', default='attempt1_20261011')
    args = parser.parse_args()
    set_attempt(args.attempt_id)
    if args.worker_suite:
        if args.gpu_index is None:
            parser.error('--worker-suite requires --gpu-index')
        from scripts.experiment11.shadow_worker import run
        return run(VARIANT, args.worker_suite, args.gpu_index, selection_path=MANIFEST,
                   output_dir=OUTPUT / VARIANT / args.worker_suite)
    if args.freeze:
        freeze_manifest()
    configure_scheduler()
    jobs, selection_hash, audit_hash = frozen_jobs()
    print(json.dumps({'jobs': len(jobs), 'variant_predictions': sum(COUNTS.values()),
                      'full_parity_predictions': 4, 'status': str(STATUS),
                      'dry_run': not args.launch}, indent=2), flush=True)
    if args.launch:
        if ATTEMPT == 'attempt1_20261011':
            raise RuntimeError('Interrupted attempt1 is frozen; pass a new --attempt-id')
        code = scheduler.run_manager(jobs, selection_hash, audit_hash)
        if code:
            return code
        return analyze()
    if args.analyze:
        return analyze()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
