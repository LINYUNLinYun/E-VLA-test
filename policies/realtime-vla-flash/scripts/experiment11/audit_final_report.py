"""Audit the assembled Experiment 11 report against all frozen stage artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json


WORKSPACE = Path(__file__).resolve().parents[4]
ANALYSIS = RESULTS / 'analysis'
OUTPUT = ANALYSIS / 'completion_audit.json'
REPORT = ANALYSIS / 'experiment11_final_report.md'
D3_AUDIT = ANALYSIS / 'closed_loop_d3/completion_audit.json'
D3_STATUS = RESULTS / 'experiment11_closed_loop_status.json'
D3_EPISODES = ANALYSIS / 'closed_loop_d3/episode_results.csv'
LATENCY_META = RESULTS / 'latency/isolated_d3_metadata.json'
LATENCY_SAMPLES = RESULTS / 'latency/isolated_d3_samples.csv'
LATENCY_SUMMARY = RESULTS / 'latency/isolated_d3.csv'
SHADOW_AUDIT = ANALYSIS / 'shadow_d3/completion_audit.json'
SHADOW_STATUS = RESULTS / 'experiment11_shadow_status.json'
D10_AUDIT = ANALYSIS / 'd10_followup_final/completion_audit.json'
D10_STATUS = RESULTS / 'experiment11_d10_closed_loop_status.json'
D10_CONFIG = RESULTS / 'protocol/closed_loop_config_d10_followup.json'
CANDIDATES = RESULTS / 'protocol/candidate_manifest.json'
D3_FORMAL_RUNS = RESULTS / 'closed_loop'
D3_FAILURES = ANALYSIS / 'closed_loop_d3/failure_stage_counts.csv'
SHADOW_ANALYSIS = ANALYSIS / 'shadow_d3'
D10_ANALYSIS = ANALYSIS / 'd10_followup_final'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def _clean_status(path: Path) -> dict:
    status = read_json(path)
    require(status.get('complete') is True, f'Scheduler is not complete: {path}')
    require(not status.get('active') and not status.get('pending') and not status.get('failures'),
            f'Scheduler has active/pending/failing jobs: {path}')
    return status


def audit_latency() -> dict:
    metadata = read_json(LATENCY_META)
    require(metadata.get('attempt_id') == 'freshproc_suite_retry1_20261011',
            'Unexpected latency attempt selected')
    require(metadata.get('conditions') == 6 and metadata.get('samples_per_condition') == 200 and
            metadata.get('total_samples') == 1200, 'Latency metadata coverage mismatch')
    with LATENCY_SAMPLES.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1200, f'Expected 1200 latency samples, got {len(rows)}')
    pair_counts: dict[tuple[str, str], int] = {}
    checkpoint_hashes = set()
    full_parity = []
    for row in rows:
        key = (row['condition'], row['suite'])
        pair_counts[key] = pair_counts.get(key, 0) + 1
        checkpoint_hashes.add(row['checkpoint_sha256'])
        require(int(row['steps']) == 3 and int(row['horizon']) == 50 and
                int(row['replan_steps']) == 5 and int(row['gpu_index']) == 1,
                f'Latency protocol mismatch in {key}')
        require(bool(row['input_sha256']), f'Missing latency input hash in {key}')
        for field in ('cuda_policy_ms', 'backend_infer_wall_ms'):
            value = float(row[field])
            require(math.isfinite(value) and value > 0, f'Invalid {field} in {key}')
        if row['condition'].startswith('Full18'):
            full_parity.append(float(row['full_parity_max_abs_error']))
    require(len(pair_counts) == 24 and all(count == 50 for count in pair_counts.values()),
            'Latency condition×suite coverage is incomplete')
    require(len(checkpoint_hashes) == 1, 'Latency checkpoint hashes differ')
    require(len(full_parity) == 200 and all(math.isfinite(value) for value in full_parity),
            'Full parity errors are missing or non-finite')
    parity_max = max(full_parity)
    require(parity_max <= 1e-5, f'Full latency parity exceeded tolerance: {parity_max}')
    with LATENCY_SUMMARY.open(newline='', encoding='utf-8') as stream:
        aggregates = list(csv.DictReader(stream))
    require(len(aggregates) == 6, 'Latency aggregate does not have six conditions')
    return {
        'complete': True,
        'attempt_id': metadata['attempt_id'],
        'condition_suite_pairs': len(pair_counts),
        'samples': len(rows),
        'gpu_index': metadata['gpu_index'],
        'measurement_scope': metadata['measurement_scope'],
        'full_parity_samples': len(full_parity),
        'full_parity_max_abs_error': parity_max,
        'checkpoint_sha256': next(iter(checkpoint_hashes)),
        'excluded_attempts': {
            'freshproc_retry1_20261011': 'Excluded: Full18 isolated parity failed; retained under latency/attempts.'
        },
    }


def audit_report_links() -> tuple[list[str], list[Path]]:
    text = REPORT.read_text(encoding='utf-8')
    targets = []
    files = []
    for target in re.findall(r'\]\(([^)]+)\)', text):
        if target.startswith(('http://', 'https://', 'mailto:')):
            continue
        local = (REPORT.parent / target).resolve()
        require(local.exists(), f'Broken local report link: {target}')
        targets.append(target)
        if local.is_file():
            files.append(local)
    require('最终版' in text and 'D10 收尾中' not in text, 'Final report is still marked as a draft')
    require(len(targets) > 0, 'No local report links were checked')
    return targets, sorted(set(files))


def build_audit() -> dict:
    d3_status = _clean_status(D3_STATUS)
    shadow_status = _clean_status(SHADOW_STATUS)
    d10_status = _clean_status(D10_STATUS)
    d3 = read_json(D3_AUDIT)
    shadow = read_json(SHADOW_AUDIT)
    d10 = read_json(D10_AUDIT)

    require(d3.get('condition_suite_runs_expected') == 124 and
            d3.get('condition_suite_runs_validated') == 124 and
            d3.get('episodes_expected') == 6200 and d3.get('episodes_validated') == 6200 and
            d3.get('full_observation_input_artifacts_missing') == 0 and
            d3.get('validated_video_files') == 6200 and d3.get('video_pairs_rendered') == 29,
            'D3 completion audit does not match expected coverage')
    with D3_EPISODES.open(newline='', encoding='utf-8') as stream:
        d3_episode_rows = list(csv.DictReader(stream))
    require(len(d3_episode_rows) == 6200, 'D3 episode table does not have 6200 rows')
    require(shadow.get('complete') is True and shadow.get('expected_jobs') == 12 and
            shadow.get('validated_jobs') == 12 and
            shadow.get('environment_steps_executed_by_variants') == 0 and
            shadow.get('replan_prediction_records') == 20937,
            'Shadow completion audit does not match expected coverage')
    require(d10.get('complete') is True and d10.get('conditions_expected') == 7 and
            d10.get('conditions_validated') == 7 and
            d10.get('condition_suite_runs_expected') == 28 and
            d10.get('condition_suite_runs_validated') == 28 and
            d10.get('episodes_expected') == 1400 and d10.get('episodes_validated') == 1400 and
            d10.get('video_decodes_validated') == 1400,
            'D10 completion audit does not match expected coverage')
    require(len(D10_CONFIG.read_text(encoding='utf-8')) > 0, 'D10 frozen config is missing')
    latency = audit_latency()
    link_targets, linked_files = audit_report_links()

    # Include per-job D3 evidence and the numerical Shadow/D10 analysis products
    # directly in the aggregate hash inventory, rather than relying only on the
    # summary audits to carry nested hashes.
    d3_formal_runs = sorted(
        path for path in D3_FORMAL_RUNS.glob('*_D3_R5_FORMAL/*/run')
        if (path / 'validated.json').is_file()
    )
    require(len(d3_formal_runs) == 124, f'Expected 124 formal D3 jobs, got {len(d3_formal_runs)}')
    d3_job_artifacts = []
    for run_dir in d3_formal_runs:
        for name in ('validated.json', 'episode_log.json'):
            artifact = run_dir / name
            require(artifact.is_file(), f'Missing formal D3 job artifact: {artifact}')
            d3_job_artifacts.append(artifact)

    shadow_artifacts = [SHADOW_ANALYSIS / 'report.md']
    shadow_artifacts += sorted(SHADOW_ANALYSIS.glob('*.csv'))
    shadow_artifacts += sorted((SHADOW_ANALYSIS / 'figures').glob('*.png'))
    d10_artifacts = [D10_ANALYSIS / 'report.md']
    d10_artifacts += sorted(D10_ANALYSIS.glob('*.csv'))
    d10_artifacts += sorted(D10_ANALYSIS.glob('*.png'))
    for artifact in (*shadow_artifacts, *d10_artifacts, D3_FAILURES):
        require(artifact.is_file(), f'Missing analysis artifact for aggregate audit: {artifact}')

    source_paths = [
        D3_AUDIT, D3_STATUS, D3_EPISODES, D3_FAILURES, *d3_job_artifacts,
        LATENCY_META, LATENCY_SAMPLES, LATENCY_SUMMARY,
        SHADOW_AUDIT, SHADOW_STATUS, D10_AUDIT, D10_STATUS, D10_CONFIG, CANDIDATES, REPORT,
        *shadow_artifacts, *d10_artifacts, *linked_files,
    ]
    source_hashes = {}
    for path in sorted(set(source_paths)):
        try:
            source_key = str(path.relative_to(RESULTS))
        except ValueError:
            source_key = str(path.relative_to(WORKSPACE))
        source_hashes[source_key] = file_sha256(path)
    return {
        'complete': True,
        'experiment_id': 'experiment11_layer_skip_prefix_20261010',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'active_experiment11_workers': 0,
        'stages': {
            'd3_closed_loop': {
                'complete': True, 'condition_suite_runs': '124/124', 'episodes': '6200/6200',
                'missing_full_inputs': 0, 'validated_videos': 6200, 'trajectory_pairs_rendered': 29,
            },
            'isolated_latency': latency,
            'full_only_shadow': {
                'complete': True, 'jobs': '12/12', 'matched_replan_predictions': 20937,
                'variant_environment_steps': 0,
            },
            'd10_diagnostic': {
                'complete': True, 'conditions': '7/7', 'condition_suite_runs': '28/28',
                'episodes': '1400/1400', 'validated_video_decodes': 1400, 'invalid_runs': 0,
                'interpretation': 'Finite post-hoc interaction diagnostic, not held-out validation.',
            },
            'task_type_layer_probe': {
                'launched': False,
                'reason': 'Planned in idea1_experiment_plan.md §10.7; awaits a separate user handoff/launch.',
            },
        },
        'report': str(REPORT.relative_to(RESULTS)),
        'report_sha256': file_sha256(REPORT),
        'checked_local_link_targets': link_targets,
        'source_inventory': {
            'd3_formal_jobs_hashed': len(d3_formal_runs),
            'd3_job_files_hashed': len(d3_job_artifacts),
            'shadow_analysis_files_hashed': len(shadow_artifacts),
            'd10_analysis_files_hashed': len(d10_artifacts),
            'raw_episode_videos_individually_hashed': False,
        },
        'source_sha256': source_hashes,
        'scheduler_statuses': {
            'd3': {'complete': d3_status['complete'], 'active': 0, 'pending': 0, 'failures': 0},
            'shadow': {'complete': shadow_status['complete'], 'active': 0, 'pending': 0, 'failures': 0},
            'd10': {'complete': d10_status['complete'], 'active': 0, 'pending': 0, 'failures': 0},
        },
    }


def main() -> int:
    audit = build_audit()
    dump_json(OUTPUT, audit)
    print(json.dumps({'audit': str(OUTPUT), 'complete': audit['complete'],
                      'checked_local_links': len(audit['checked_local_link_targets']),
                      'hashed_source_files': len(audit['source_sha256'])}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
