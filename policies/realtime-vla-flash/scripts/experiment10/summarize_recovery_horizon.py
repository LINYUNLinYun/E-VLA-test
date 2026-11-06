"""Summarize paired short-prefix D3 then Full18-D10 recovery rollouts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, json_lines


def summarize(config_path):
    config = json.loads(Path(config_path).read_text())
    output = RESULTS / 'recovery_horizon'
    output.mkdir(parents=True, exist_ok=True)
    condition_summaries, episode_rows, inference_rows = {}, [], []

    for condition in config['conditions']:
        name = condition['name']
        condition_episodes, condition_inferences = [], []
        for suite in SUITES:
            root = RESULTS / 'closed_loop' / name / suite / 'run'
            validated = json.loads((root / 'validated.json').read_text())
            if not validated.get('complete') or validated.get('condition') != condition:
                raise RuntimeError('Unvalidated/config-mismatched recovery run: %s/%s' % (name, suite))
            rows = json.loads((root / 'episode_log.json').read_text())
            selected = {(r['task_id'], r['episode_idx']): r for r in rows}

            baseline_root = RESULTS / 'closed_loop' / 'Full18_D10' / suite / 'run'
            baseline_validation = json.loads((baseline_root / 'validated.json').read_text())
            if not baseline_validation.get('complete'):
                raise RuntimeError('Full18-D10 paired baseline is not validated: %s' % suite)
            baseline_rows = json.loads((baseline_root / 'episode_log.json').read_text())
            baseline = {(r['task_id'], r['episode_idx']): r for r in baseline_rows
                        if r['task_id'] in condition['task_ids']}
            expected = {(task, init) for task in condition['task_ids'] for init in condition['init_indices']}
            if set(selected) != expected or set(baseline) != expected:
                raise RuntimeError('Incomplete paired validation task/init set: %s/%s' % (name, suite))

            for key in sorted(expected):
                row, reference = selected[key], baseline[key]
                requests = json_lines(root / row['infer_path'])
                baseline_requests = json_lines(baseline_root / reference['infer_path'])
                actual_route_counts = Counter()
                for request in requests:
                    meta = request['experiment10']
                    route = 'full18_recovery' if bool(meta['recovery_round']) else 'd3_prefix'
                    actual_route_counts[route] += 1
                    record = {
                        'condition': name, 'suite': suite, 'task_id': key[0],
                        'episode_idx': key[1], 'infer_id': request['infer_id'],
                        'route': route, 'accepted_prefix_len': request['accepted_prefix_len'],
                        'chunk_exec_len': request['chunk_exec_len'],
                        'sample_actions_ms': meta['sample_actions_ms'],
                        'denoise_steps': meta['denoise_steps'],
                        'recovery_round': bool(meta['recovery_round']),
                    }
                    inference_rows.append(record)
                    condition_inferences.append(record)
                gate_total = float(sum(r['experiment10']['sample_actions_ms'] for r in requests))
                baseline_total = float(sum(r['experiment10']['sample_actions_ms'] for r in baseline_requests))
                condition_episodes.append({
                    'condition': name, 'suite': suite, 'task_id': key[0], 'episode_idx': key[1],
                    'horizon': condition['horizon'], 'gate_success': int(bool(row['success'])),
                    'baseline_success': int(bool(reference['success'])),
                    'paired_outcome': ('both_success' if row['success'] and reference['success'] else
                                       'gate_only_success' if row['success'] else
                                       'baseline_only_success' if reference['success'] else 'both_failure'),
                    'gate_furthest_phase_on_failure': row.get('furthest_semantic_phase') if not row['success'] else None,
                    'infer_calls': len(requests), 'd3_prefix_calls': actual_route_counts['d3_prefix'],
                    'full18_recovery_calls': actual_route_counts['full18_recovery'],
                    'gate_policy_inference_total_ms': gate_total,
                    'baseline_policy_inference_total_ms': baseline_total,
                    'gate_policy_mean_ms_per_request': gate_total / len(requests) if requests else None,
                    'baseline_policy_mean_ms_per_request': baseline_total / len(baseline_requests) if baseline_requests else None,
                })

        successes = sum(r['gate_success'] for r in condition_episodes)
        baseline_successes = sum(r['baseline_success'] for r in condition_episodes)
        gate_ms = sum(r['gate_policy_inference_total_ms'] for r in condition_episodes)
        baseline_ms = sum(r['baseline_policy_inference_total_ms'] for r in condition_episodes)
        condition_summaries[name] = {
            'horizon': condition['horizon'], 'episodes': len(condition_episodes),
            'gate_successes': successes, 'baseline_successes': baseline_successes,
            'gate_success_rate': successes / len(condition_episodes),
            'baseline_success_rate': baseline_successes / len(condition_episodes),
            'paired_both_success': sum(r['paired_outcome'] == 'both_success' for r in condition_episodes),
            'gate_only_success': sum(r['paired_outcome'] == 'gate_only_success' for r in condition_episodes),
            'baseline_only_success': sum(r['paired_outcome'] == 'baseline_only_success' for r in condition_episodes),
            'd3_prefix_calls': sum(r['d3_prefix_calls'] for r in condition_episodes),
            'full18_recovery_calls': sum(r['full18_recovery_calls'] for r in condition_episodes),
            'gate_policy_inference_total_ms': gate_ms,
            'baseline_policy_inference_total_ms': baseline_ms,
            'total_policy_inference_ratio_vs_baseline': gate_ms / baseline_ms,
            'furthest_phase_of_gate_failures': dict(Counter(
                r['gate_furthest_phase_on_failure'] for r in condition_episodes
                if r['gate_furthest_phase_on_failure'] is not None)),
            'suite_summaries': {
                suite: {
                    'episodes': sum(r['suite'] == suite for r in condition_episodes),
                    'gate_successes': sum(r['suite'] == suite and r['gate_success'] for r in condition_episodes),
                    'baseline_successes': sum(r['suite'] == suite and r['baseline_success'] for r in condition_episodes),
                    'gate_policy_inference_total_ms': sum(
                        r['gate_policy_inference_total_ms'] for r in condition_episodes if r['suite'] == suite),
                    'baseline_policy_inference_total_ms': sum(r['baseline_policy_inference_total_ms'] for r in condition_episodes if r['suite'] == suite),
                } for suite in SUITES
            },
        }
        episode_rows.extend(condition_episodes)

    summary = {
        'complete': True, 'reference': 'Full18_D10', 'evaluation_split': 'validation tasks6-7/init0-4',
        'conditions': condition_summaries,
        'limitations': [
            'Only two validation tasks and five official inits per task; exploratory horizon screen',
            'Validation tasks were also used for command-gate calibration, so this is not untouched holdout evidence',
            'First D3 prefix is followed by synchronous Full18-D10 recovery; this does not test asynchronous overlap',
            'Policy inference time sums exclude simulator/environment wall-clock time',
            'Highest reached phase is not the first trajectory deviation or a causal failure stage',
        ],
    }
    dump_json(output / 'recovery_horizon_summary.json', summary)
    for filename, rows in (('recovery_horizon_episodes.csv', episode_rows),
                           ('recovery_horizon_inferences.csv', inference_rows)):
        temporary = output / (filename + '.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / filename)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=Path,
                        default=Path('scripts/experiment10/recovery_horizon_screen_config.json'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.conditions), indent=2), flush=True)


if __name__ == '__main__':
    main()
