"""Pair VerifyD3_H8 outcomes with Full18-D10 and summarize actual per-infer routes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, json_lines


def summarize(condition_name='VerifyD3_H8'):
    output = RESULTS / 'verify_gate'
    episode_rows, inference_rows, suite_summaries = [], [], {}
    for suite in SUITES:
        root = RESULTS / 'closed_loop' / condition_name / suite / 'run'
        validation = json.loads((root / 'validated.json').read_text())
        if not validation.get('complete'):
            raise RuntimeError('Verify-gate suite is not validated: %s' % suite)
        candidate_episodes = json.loads((root / 'episode_log.json').read_text())
        baseline_root = RESULTS / 'closed_loop' / 'Full18_D10' / suite / 'run'
        baseline_episodes = json.loads((baseline_root / 'episode_log.json').read_text())
        baseline = {(r['task_id'], r['episode_idx']): r for r in baseline_episodes if r['task_id'] in (8, 9)}
        if len(candidate_episodes) != 10 or len(baseline) != 10:
            raise RuntimeError('Expected paired tasks 8-9 and init0-4 in %s' % suite)

        candidate_success = baseline_success = both_success = candidate_only = baseline_only = 0
        suite_inferences = []
        for row in candidate_episodes:
            key = (row['task_id'], row['episode_idx'])
            if key not in baseline:
                raise RuntimeError('Missing paired Full18-D10 episode: %s %r' % (suite, key))
            reference = baseline[key]
            candidate_ok, baseline_ok = bool(row['success']), bool(reference['success'])
            candidate_success += int(candidate_ok)
            baseline_success += int(baseline_ok)
            both_success += int(candidate_ok and baseline_ok)
            candidate_only += int(candidate_ok and not baseline_ok)
            baseline_only += int(baseline_ok and not candidate_ok)

            requests = json_lines(root / row['infer_path'])
            routes = []
            for request in requests:
                meta = request['experiment10']
                fallback = bool(meta['used_full_fallback'])
                route = 'full_fallback' if fallback else 'draft_prefix'
                routes.append(route)
                record = {
                    'suite': suite, 'task_id': row['task_id'], 'episode_idx': row['episode_idx'],
                    'infer_id': request['infer_id'], 'route': route,
                    'accepted_prefix_len': request['accepted_prefix_len'],
                    'candidate_accepted_prefix_len': meta['candidate_accepted_prefix_len'],
                    'chunk_exec_len': request['chunk_exec_len'],
                    'verification_reason': meta['verification_reason'],
                    'sample_actions_ms': meta['sample_actions_ms'],
                    'candidate_cuda_policy_ms': meta['candidate_cuda_policy_ms'],
                    'reference_cuda_policy_ms': meta['reference_cuda_policy_ms'],
                }
                inference_rows.append(record)
                suite_inferences.append(record)
            draft = [r for r in suite_inferences if r['suite'] == suite and
                     r['task_id'] == row['task_id'] and r['episode_idx'] == row['episode_idx'] and
                     r['route'] == 'draft_prefix']
            fallback = [r for r in suite_inferences if r['suite'] == suite and
                        r['task_id'] == row['task_id'] and r['episode_idx'] == row['episode_idx'] and
                        r['route'] == 'full_fallback']
            episode_rows.append({
                'suite': suite, 'task_id': row['task_id'], 'episode_idx': row['episode_idx'],
                'verify_success': int(candidate_ok), 'baseline_success': int(baseline_ok),
                'paired_outcome': ('both_success' if candidate_ok and baseline_ok else
                                   'verify_only_success' if candidate_ok else
                                   'baseline_only_success' if baseline_ok else 'both_failure'),
                'infer_calls': len(requests), 'draft_prefix_calls': len(draft),
                'full_fallback_calls': len(fallback),
                'draft_prefix_ratio': len(draft) / len(requests) if requests else 0.,
                'mean_accepted_draft_prefix': float(np.mean([r['candidate_accepted_prefix_len'] for r in draft])) if draft else None,
                'mean_verify_policy_ms': float(np.mean([r['sample_actions_ms'] for r in suite_inferences
                                                        if r['task_id'] == row['task_id'] and r['episode_idx'] == row['episode_idx']])) if requests else None,
                'baseline_mean_policy_ms': reference.get('infer_latency_mean_ms'),
            })

        suite_rows = [r for r in episode_rows if r['suite'] == suite]
        suite_calls = [r for r in inference_rows if r['suite'] == suite]
        draft_calls = [r for r in suite_calls if r['route'] == 'draft_prefix']
        fallback_calls = [r for r in suite_calls if r['route'] == 'full_fallback']
        suite_summaries[suite] = {
            'episodes': len(suite_rows),
            'verify_successes': candidate_success,
            'baseline_successes': baseline_success,
            'paired_both_success': both_success,
            'verify_only_success': candidate_only,
            'baseline_only_success': baseline_only,
            'verify_success_rate': candidate_success / len(suite_rows),
            'baseline_success_rate': baseline_success / len(suite_rows),
            'inference_calls': len(suite_calls),
            'draft_prefix_calls': len(draft_calls),
            'full_fallback_calls': len(fallback_calls),
            'draft_prefix_ratio': len(draft_calls) / len(suite_calls) if suite_calls else 0.,
            'mean_accepted_draft_prefix': float(np.mean([r['candidate_accepted_prefix_len'] for r in draft_calls])) if draft_calls else None,
            'verify_policy_median_ms': float(np.median([r['sample_actions_ms'] for r in suite_calls])) if suite_calls else None,
            'baseline_episode_mean_policy_ms': float(np.mean([r['baseline_mean_policy_ms'] for r in suite_rows])),
            'route_reasons': {reason: sum(r['verification_reason'] == reason for r in suite_calls)
                              for reason in sorted({r['verification_reason'] for r in suite_calls})},
        }

    all_episodes = episode_rows
    all_calls = inference_rows
    successes = sum(r['verify_success'] for r in all_episodes)
    baseline_successes = sum(r['baseline_success'] for r in all_episodes)
    draft_calls = [r for r in all_calls if r['route'] == 'draft_prefix']
    fallback_calls = [r for r in all_calls if r['route'] == 'full_fallback']
    summary = {
        'condition': condition_name,
        'reference': 'Full18_D10 same task/init; existing validated run',
        'evaluation_split': 'tasks_8_9_init0_4',
        'complete': True,
        'episodes': len(all_episodes),
        'verify_successes': successes,
        'baseline_successes': baseline_successes,
        'paired_both_success': sum(r['paired_outcome'] == 'both_success' for r in all_episodes),
        'verify_only_success': sum(r['paired_outcome'] == 'verify_only_success' for r in all_episodes),
        'baseline_only_success': sum(r['paired_outcome'] == 'baseline_only_success' for r in all_episodes),
        'verify_success_rate': successes / len(all_episodes),
        'baseline_success_rate': baseline_successes / len(all_episodes),
        'inference_calls': len(all_calls),
        'draft_prefix_calls': len(draft_calls),
        'full_fallback_calls': len(fallback_calls),
        'draft_prefix_ratio': len(draft_calls) / len(all_calls) if all_calls else 0.,
        'mean_accepted_draft_prefix': float(np.mean([r['candidate_accepted_prefix_len'] for r in draft_calls])) if draft_calls else None,
        'verify_policy_median_ms': float(np.median([r['sample_actions_ms'] for r in all_calls])) if all_calls else None,
        'baseline_episode_mean_policy_ms': float(np.mean([r['baseline_mean_policy_ms'] for r in all_episodes])),
        'suite_summaries': suite_summaries,
        'limitations': ['Only evaluation tasks 8-9 and init0-4; exploratory screening sample',
                        'Gate agreement with Full18 is not semantic correctness',
                        'Current run route labels are reconstructed from used_full_fallback because its client summary used the older full/draft flag'],
    }
    dump_json(output / 'verify_d3_h8_summary.json', summary)
    for filename, rows, fields in (
        ('verify_d3_h8_episodes.csv', episode_rows, list(episode_rows[0].keys())),
        ('verify_d3_h8_inferences.csv', inference_rows, list(inference_rows[0].keys())),
    ):
        temporary = output / (filename + '.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / filename)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', default='VerifyD3_H8')
    args = parser.parse_args()
    print(json.dumps(summarize(args.condition), indent=2), flush=True)


if __name__ == '__main__':
    main()
