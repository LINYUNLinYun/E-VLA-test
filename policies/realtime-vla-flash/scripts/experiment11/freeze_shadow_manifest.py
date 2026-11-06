"""Select Experiment 4 shadow variants using only completed D3 closed-loop results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from scripts.experiment11.protocol import RESULTS, SUITES
from scripts.experiment10.common import dump_json, sha256


THRESHOLD = 0.80
FAMILIES = ('V16', 'V15', 'V14', 'V13')


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write selected_shadow_manifest.json; default only previews')
    args = parser.parse_args()

    status_path = RESULTS / 'experiment11_closed_loop_status.json'
    status = json.loads(status_path.read_text())
    audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    if (not status.get('complete') or status.get('active') or status.get('pending') or
            status.get('failures') or not audit_path.is_file()):
        raise RuntimeError('D3 matrix and its complete audit must pass before selecting shadow variants')
    audit = json.loads(audit_path.read_text())
    if audit.get('condition_suite_runs_validated') != 124 or audit.get('episodes_validated') != 6200:
        raise RuntimeError('D3 audit does not prove 124 validated suite runs / 6200 episodes')

    suite_rows = load_csv(RESULTS / 'analysis/closed_loop_d3/suite_results.csv')
    manifest_path = RESULTS / 'protocol/candidate_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    latency_path = RESULTS / 'latency/isolated_d3.csv'
    latency_by_id = {}
    if latency_path.is_file():
        for row in load_csv(latency_path):
            if row.get('variant_id') and row.get('p50_cuda_policy_ms'):
                latency_by_id[row['variant_id']] = float(row['p50_cuda_policy_ms'])

    candidate_stats = {}
    for candidate in manifest['candidates']:
        variant_id = candidate['variant_id']
        name = candidate['condition']['name'] + '_FORMAL'
        rows = [row for row in suite_rows if row['condition'] == name]
        if {row['suite'] for row in rows} != set(SUITES) or any(int(row['episodes']) != 50 for row in rows):
            raise RuntimeError(f'Incomplete D3 success summary for {variant_id}')
        successes = sum(int(row['successes']) for row in rows)
        candidate_stats[variant_id] = {
            'variant_id': variant_id, 'family': candidate['family'],
            'condition_name': name, 'skip_layers': candidate['skip_layers'],
            'overall_successes': successes, 'overall_episodes': 200,
            'overall_success_rate': successes / 200,
            'suite_results': {row['suite']: {'successes': int(row['successes']),
                                              'episodes': int(row['episodes']),
                                              'success_rate': float(row['success_rate'])}
                              for row in rows},
            'isolated_latency_p50_cuda_policy_ms': latency_by_id.get(variant_id),
        }

    selected = []
    family_ranking = {}
    for family in FAMILIES:
        members = [row for row in candidate_stats.values() if row['family'] == family]
        ordered = sorted(members, key=lambda row: (-row['overall_success_rate'], row['variant_id']))
        best_rate = ordered[0]['overall_success_rate']
        tied = [row for row in ordered if row['overall_success_rate'] == best_rate]
        qualified = best_rate >= THRESHOLD
        if len(tied) > 1 and qualified:
            if any(row['isolated_latency_p50_cuda_policy_ms'] is None for row in tied):
                raise RuntimeError(
                    f'{family} has tied top D3 success; dedicated latency is needed for the frozen tie-break')
            winner = min(tied, key=lambda row: (row['isolated_latency_p50_cuda_policy_ms'], row['variant_id']))
            tie_rule = 'highest D3 success; tied candidates ranked by isolated CUDA p50, then variant id'
        elif len(tied) > 1:
            winner = min(tied, key=lambda row: row['variant_id'])
            tie_rule = 'highest D3 success is below the 80% gate; id tie-break only for descriptive ranking'
        else:
            winner = tied[0]
            tie_rule = 'highest D3 success; unique top candidate'
        family_ranking[family] = {
            'best_candidate': winner['variant_id'], 'best_success_rate': winner['overall_success_rate'],
            'qualified_at_80_percent': qualified, 'tie_rule': tie_rule,
            'ranked_candidates': [{**row, 'rank': rank + 1}
                                  for rank, row in enumerate(sorted(
                                      members,
                                      key=lambda row: (-row['overall_success_rate'],
                                                       row['isolated_latency_p50_cuda_policy_ms']
                                                       if row['isolated_latency_p50_cuda_policy_ms'] is not None else float('inf'),
                                                       row['variant_id'])))],
        }
        if qualified:
            selected.append({**winner, 'selection_reason': tie_rule + '; meets >=80% gate'})

    custom_skip = manifest['custom_v13_reference_skip_layers']
    custom_rows = [row for row in suite_rows if row['condition'] == 'V13_Custom_D3_R5_FORMAL']
    if len(custom_rows) != 4:
        raise RuntimeError('Missing original V13-Custom D3 reference result')
    custom_successes = sum(int(row['successes']) for row in custom_rows)
    original_v13 = {
        'variant_id': 'V13_Custom', 'family': 'V13-reference',
        'condition_name': 'V13_Custom_D3_R5_FORMAL', 'skip_layers': custom_skip,
        'overall_successes': custom_successes, 'overall_episodes': 200,
        'overall_success_rate': custom_successes / 200,
        'suite_results': {row['suite']: {'successes': int(row['successes']),
                                          'episodes': int(row['episodes']),
                                          'success_rate': float(row['success_rate'])}
                          for row in custom_rows},
        'selection_reason': 'Original user-provided V13 reference; diagnostic only, does not satisfy the new-mask 80% gate.',
    }
    output = {
        'schema': 1,
        'experiment_id': 'experiment11_shadow_prefix_d3',
        'status': 'eligible' if selected else 'no_qualified_new_mask',
        'manual_user_decision_required': not bool(selected),
        'gate': {'metric': 'overall D3 closed-loop success across 4 suites, 200 episodes',
                 'minimum_rate': THRESHOLD, 'minimum_successes': 160,
                 'max_new_masks': 4, 'original_v13_reference_added': True},
        'source_artifacts': {
            'completion_audit': str(audit_path), 'completion_audit_sha256': sha256(audit_path),
            'suite_results': str(RESULTS / 'analysis/closed_loop_d3/suite_results.csv'),
            'suite_results_sha256': sha256(RESULTS / 'analysis/closed_loop_d3/suite_results.csv'),
            'candidate_manifest_sha256': sha256(manifest_path),
            'latency_csv': str(latency_path) if latency_path.is_file() else None,
            'latency_csv_sha256': sha256(latency_path) if latency_path.is_file() else None,
        },
        'family_ranking': family_ranking,
        'selected_new_masks': selected,
        'original_v13_reference': original_v13,
        'interpretation_warning': 'Selection uses the same five-init exploratory set; it is not an independent confirmation or physical-safety guarantee.',
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.write:
        path = RESULTS / 'protocol/selected_shadow_manifest.json'
        if path.exists():
            raise FileExistsError(f'Refusing to replace frozen shadow selection: {path}')
        dump_json(path, output)
        print(json.dumps({'written': str(path), 'selected_new_masks': len(selected)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
