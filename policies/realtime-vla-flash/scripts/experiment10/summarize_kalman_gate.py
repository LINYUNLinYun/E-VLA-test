"""Pair executed-command Kalman-gate rollouts with Full18-D10 and summarize routes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, json_lines


def summarize(config_path):
    config = json.loads(Path(config_path).read_text())
    output = RESULTS / 'kalman_gate'
    output.mkdir(parents=True, exist_ok=True)
    all_episode_rows, all_inference_rows, condition_summaries = [], [], {}
    baseline_name = 'Full18_D10'

    for condition in config['conditions']:
        name = condition['name']
        episode_rows, inference_rows = [], []
        for suite in SUITES:
            root = RESULTS / 'closed_loop' / name / suite / 'run'
            validation = json.loads((root / 'validated.json').read_text())
            if not validation.get('complete') or validation.get('condition') != condition:
                raise RuntimeError('Kalman-gate suite is not validated/config-matched: %s/%s' % (name, suite))
            candidate_episodes = json.loads((root / 'episode_log.json').read_text())
            baseline_root = RESULTS / 'closed_loop' / baseline_name / suite / 'run'
            baseline_episodes = json.loads((baseline_root / 'episode_log.json').read_text())
            baseline = {(r['task_id'], r['episode_idx']): r for r in baseline_episodes
                        if r['task_id'] in condition['task_ids']}
            selected = [r for r in candidate_episodes if r['task_id'] in condition['task_ids']]
            expected_count = len(condition['task_ids']) * len(condition['init_indices'])
            if len(selected) != expected_count or len(baseline) != expected_count:
                raise RuntimeError('Incomplete paired Kalman-gate episodes: %s/%s' % (name, suite))

            for row in selected:
                key = (row['task_id'], row['episode_idx'])
                if key not in baseline:
                    raise RuntimeError('Missing paired Full18-D10 episode: %s/%s %r' % (name, suite, key))
                reference = baseline[key]
                requests = json_lines(root / row['infer_path'])
                baseline_requests = json_lines(baseline_root / reference['infer_path'])
                routes = []
                for request in requests:
                    meta = request['experiment10']
                    fallback = bool(meta['used_full_fallback'])
                    route = 'full_fallback' if fallback else 'draft_prefix'
                    routes.append(route)
                    inference_rows.append({
                        'condition': name, 'suite': suite, 'task_id': row['task_id'],
                        'episode_idx': row['episode_idx'], 'infer_id': request['infer_id'],
                        'route': route, 'accepted_prefix_len': request['accepted_prefix_len'],
                        'candidate_accepted_prefix_len': meta['candidate_accepted_prefix_len'],
                        'chunk_exec_len': request['chunk_exec_len'],
                        'execution_gate_reason': meta['execution_gate_reason'],
                        'predictor_history_count': meta['predictor_history_count'],
                        'observation_age_ms_at_decision': meta['observation_age_ms_at_decision'],
                        'candidate_cuda_policy_ms': meta['candidate_cuda_policy_ms'],
                        'full_cuda_policy_ms': meta['reference_cuda_policy_ms'],
                        'sample_actions_ms': meta['sample_actions_ms'],
                    })
                draft_lengths = [r['experiment10']['candidate_accepted_prefix_len'] for r in requests
                                 if not bool(r['experiment10']['used_full_fallback'])]
                episode_rows.append({
                    'condition': name, 'suite': suite, 'task_id': row['task_id'],
                    'episode_idx': row['episode_idx'], 'gate_success': int(bool(row['success'])),
                    'baseline_success': int(bool(reference['success'])),
                    'paired_outcome': ('both_success' if row['success'] and reference['success'] else
                                       'gate_only_success' if row['success'] else
                                       'baseline_only_success' if reference['success'] else 'both_failure'),
                    'infer_calls': len(requests), 'draft_prefix_calls': routes.count('draft_prefix'),
                    'full_fallback_calls': routes.count('full_fallback'),
                    'draft_prefix_ratio': routes.count('draft_prefix') / len(requests) if requests else 0.,
                    'mean_accepted_draft_prefix': float(np.mean(draft_lengths)) if draft_lengths else None,
                    'gate_policy_mean_ms': float(np.mean([r['experiment10']['sample_actions_ms'] for r in requests])) if requests else None,
                    'baseline_policy_mean_ms': reference.get('infer_latency_mean_ms'),
                    'gate_policy_total_ms': float(sum(r['experiment10']['sample_actions_ms'] for r in requests)),
                    'baseline_policy_total_ms': float(sum(r['experiment10']['sample_actions_ms'] for r in baseline_requests)),
                })

        condition_episodes = [r for r in episode_rows]
        condition_calls = [r for r in inference_rows]
        drafts = [r for r in condition_calls if r['route'] == 'draft_prefix']
        fallbacks = [r for r in condition_calls if r['route'] == 'full_fallback']
        successes = sum(r['gate_success'] for r in condition_episodes)
        baseline_successes = sum(r['baseline_success'] for r in condition_episodes)
        condition_summaries[name] = {
            'threshold_quantile': name.rsplit('Q', 1)[-1],
            'episodes': len(condition_episodes), 'gate_successes': successes,
            'baseline_successes': baseline_successes,
            'gate_success_rate': successes / len(condition_episodes),
            'baseline_success_rate': baseline_successes / len(condition_episodes),
            'paired_both_success': sum(r['paired_outcome'] == 'both_success' for r in condition_episodes),
            'gate_only_success': sum(r['paired_outcome'] == 'gate_only_success' for r in condition_episodes),
            'baseline_only_success': sum(r['paired_outcome'] == 'baseline_only_success' for r in condition_episodes),
            'inference_calls': len(condition_calls), 'draft_prefix_calls': len(drafts),
            'full_fallback_calls': len(fallbacks),
            'draft_prefix_ratio': len(drafts) / len(condition_calls) if condition_calls else 0.,
            'mean_accepted_draft_prefix': float(np.mean([r['candidate_accepted_prefix_len'] for r in drafts])) if drafts else None,
            'gate_request_latency_median_ms': float(np.median([r['sample_actions_ms'] for r in condition_calls])) if condition_calls else None,
            'gate_request_latency_p95_ms': float(np.percentile([r['sample_actions_ms'] for r in condition_calls], 95)) if condition_calls else None,
            'gate_policy_inference_total_ms': float(sum(r['gate_policy_total_ms'] for r in condition_episodes)),
            'baseline_policy_inference_total_ms': float(sum(r['baseline_policy_total_ms'] for r in condition_episodes)),
            'total_policy_inference_ratio_vs_baseline': (
                sum(r['gate_policy_total_ms'] for r in condition_episodes) /
                sum(r['baseline_policy_total_ms'] for r in condition_episodes)),
            'route_reasons': {reason: sum(r['execution_gate_reason'] == reason for r in condition_calls)
                              for reason in sorted({r['execution_gate_reason'] for r in condition_calls})},
            'suite_summaries': {
                suite: {
                    'episodes': sum(r['suite'] == suite for r in condition_episodes),
                    'gate_successes': sum(r['suite'] == suite and r['gate_success'] for r in condition_episodes),
                    'baseline_successes': sum(r['suite'] == suite and r['baseline_success'] for r in condition_episodes),
                    'draft_prefix_ratio': (
                        sum(r['suite'] == suite and r['route'] == 'draft_prefix' for r in condition_calls) /
                        max(1, sum(r['suite'] == suite for r in condition_calls))),
                } for suite in SUITES
            },
            'frozen_parameters': {k: condition[k] for k in (
                'skip', 'steps', 'horizon', 'translation_threshold', 'rotation_threshold',
                'max_observation_age_ms', 'minimum_executed_history',
                'predictor_process_variance', 'predictor_measurement_variance')},
        }
        all_episode_rows.extend(episode_rows)
        all_inference_rows.extend(inference_rows)

    summary = {
        'complete': True, 'reference': baseline_name,
        'evaluation_split': 'tasks_8_9_init0_4', 'conditions': condition_summaries,
        'limitations': [
            'Exploratory 40 episodes per threshold, using only tasks 8-9 and init0-4',
            'Command-prediction agreement is not semantic or physical safety verification',
            'Q/R and quantile thresholds are heuristic and calibrated on validation tasks 6-7',
            '100 ms observation-age cutoff is an engineering screen, not a validated real-robot limit',
            'The synchronous simulator screen does not establish asynchronous wall-clock overlap',
        ],
    }
    dump_json(output / 'kalman_gate_summary.json', summary)
    for filename, rows in (('kalman_gate_episodes.csv', all_episode_rows),
                           ('kalman_gate_inferences.csv', all_inference_rows)):
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
                        default=Path('scripts/experiment10/kalman_gate_closed_loop_config.json'))
    args = parser.parse_args()
    print(json.dumps(summarize(args.conditions), indent=2), flush=True)


if __name__ == '__main__':
    main()
