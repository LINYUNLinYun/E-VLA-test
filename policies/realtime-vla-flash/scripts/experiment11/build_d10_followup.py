"""Freeze the post-D3, at-most-seven-condition D10 interaction diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.experiment11.protocol import RESULTS, SUITES
from scripts.experiment10.common import dump_json, sha256


BASELINE_D3 = 'Full18_D3_R5_FORMAL'
CUSTOM_D3 = 'V13_Custom_D3_R5_FORMAL'


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Print frozen choices without writing config')
    args = parser.parse_args()

    status_path = RESULTS / 'experiment11_closed_loop_status.json'
    status = json.loads(status_path.read_text())
    if not status.get('complete') or status.get('active') or status.get('pending') or status.get('failures'):
        raise RuntimeError('D3 matrix is not terminal/clean; cannot select D10 conditions')
    audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    if not audit_path.is_file():
        raise RuntimeError('Run analyze_closed_loop.py first and audit its complete coverage')
    audit = json.loads(audit_path.read_text())
    if audit.get('condition_suite_runs_validated') != 124 or audit.get('episodes_validated') != 6200:
        raise RuntimeError('D3 completion audit does not prove the required coverage')

    suite_rows = load_csv(RESULTS / 'analysis/closed_loop_d3/suite_results.csv')
    manifest = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
    d3_config = json.loads((RESULTS / 'protocol/closed_loop_config_formal.json').read_text())
    by_condition = {row['name']: row for row in d3_config['conditions']}
    rows_by_id: dict[str, dict[str, float]] = {}
    for candidate in manifest['candidates']:
        variant_id = candidate['variant_id']
        name = candidate['condition']['name'] + '_FORMAL'
        data = [row for row in suite_rows if row['condition'] == name]
        if {row['suite'] for row in data} != set(SUITES) or any(int(row['episodes']) != 50 for row in data):
            raise RuntimeError(f'Incomplete D3 suite summary for {variant_id}')
        rows_by_id[variant_id] = {
            'family': candidate['family'],
            'overall_sr': sum(float(row['success_rate']) for row in data) / len(SUITES),
        }

    full_rows = [row for row in suite_rows if row['condition'] == BASELINE_D3]
    custom_rows = [row for row in suite_rows if row['condition'] == CUSTOM_D3]
    if len(full_rows) != 4 or len(custom_rows) != 4:
        raise RuntimeError('Missing D3 reference results')
    full_sr = sum(float(row['success_rate']) for row in full_rows) / 4
    custom_sr = sum(float(row['success_rate']) for row in custom_rows) / 4

    selected: list[tuple[str, str]] = []
    for family in ('V16', 'V15', 'V14', 'V13'):
        members = [(variant_id, values) for variant_id, values in rows_by_id.items()
                   if values['family'] == family]
        if not members:
            continue
        winner = sorted(members, key=lambda item: (-item[1]['overall_sr'], item[0]))[0]
        selected.append((winner[0], f'best D3 overall success in budget {family}; deterministic id tie-break'))
    already = {variant_id for variant_id, _ in selected}
    weaker = sorted(((variant_id, values['overall_sr'] - full_sr)
                     for variant_id, values in rows_by_id.items() if variant_id not in already),
                    key=lambda item: (item[1], item[0]))
    if weaker and weaker[0][1] < 0:
        selected.append((weaker[0][0], f'largest observed D3 loss versus Full18 ({weaker[0][1]:+.4f}); post-hoc diagnostic'))

    candidate_by_id = {row['variant_id']: row for row in manifest['candidates']}
    d10_conditions: list[dict] = []

    def add_control(source_name: str, d10_name: str, role: str, d3_sr: float) -> None:
        condition = dict(by_condition[source_name])
        condition.update({'name': d10_name, 'steps': 10, 'horizon': 5, 'replan_steps': 5,
                          'save_full_inputs': False, 'd10_role': role, 'matched_d3_success_rate': d3_sr})
        d10_conditions.append(condition)

    add_control(BASELINE_D3, 'Full18_D10_R5_DIAG', 'reference_full', full_sr)
    add_control(CUSTOM_D3, 'V13_Custom_D10_R5_DIAG', 'reference_custom_v13', custom_sr)
    for variant_id, reason in selected:
        candidate = candidate_by_id[variant_id]
        source_name = candidate['condition']['name'] + '_FORMAL'
        condition = dict(by_condition[source_name])
        condition.update({'name': candidate['condition']['name'].replace('_D3_R5', '_D10_R5_DIAG'),
                          'steps': 10, 'horizon': 5, 'replan_steps': 5,
                          'save_full_inputs': False, 'd10_role': 'selected_variant',
                          'variant_id': variant_id, 'selection_reason': reason,
                          'matched_d3_success_rate': rows_by_id[variant_id]['overall_sr']})
        d10_conditions.append(condition)
    if len(d10_conditions) > 7 or len({row['name'] for row in d10_conditions}) != len(d10_conditions):
        raise RuntimeError('D10 follow-up exceeds its frozen seven-condition limit or has duplicate names')

    config = {
        'schema': 1,
        'experiment_id': 'experiment11_d10_interaction_diagnostic',
        'selection_source': str(RESULTS / 'analysis/closed_loop_d3/suite_results.csv'),
        'selection_source_sha256': sha256(RESULTS / 'analysis/closed_loop_d3/suite_results.csv'),
        'selection_audit_sha256': sha256(audit_path),
        'comparison_boundary': 'D10 is a post-hoc denoising-step interaction diagnostic, not a replacement for the D3 primary matrix.',
        'paired_noise_rule': 'SeedSequence(12345, episode_idx, replan_idx), float32[50,32]',
        'suite_coverage': list(SUITES),
        'task_ids': list(range(10)),
        'init_indices': list(range(5)),
        'episodes_per_condition': 200,
        'conditions': d10_conditions,
    }
    output_path = RESULTS / 'protocol/closed_loop_config_d10_followup.json'
    if args.dry_run:
        print(json.dumps({'conditions': [
            {'name': row['name'], 'variant_id': row.get('variant_id'), 'skip': row['skip'],
             'steps': row['steps'], 'matched_d3_success_rate': row.get('matched_d3_success_rate'),
             'selection_reason': row.get('selection_reason', row['d10_role'])}
            for row in d10_conditions], 'max_episodes': len(d10_conditions) * 200}, indent=2))
        return 0
    if output_path.exists():
        raise FileExistsError(f'Refusing to replace existing frozen D10 config: {output_path}')
    dump_json(output_path, config)
    print(json.dumps({'config': str(output_path), 'conditions': len(d10_conditions),
                      'max_episodes': len(d10_conditions) * 200}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
