"""Pair a parallel verifier horizon screen against the Full18-D10 rollout."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, json_lines


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def summarize(condition_name, config):
    condition = next((row for row in config['conditions'] if row['name'] == condition_name), None)
    if condition is None:
        raise ValueError('Condition missing from config: %s' % condition_name)

    episode_rows, inference_rows, suite_summaries = [], [], {}
    all_parallel_episode_totals, all_baseline_episode_totals = [], []
    for suite in SUITES:
        candidate_root = RESULTS / 'closed_loop' / condition_name / suite / 'run'
        baseline_root = RESULTS / 'closed_loop' / 'Full18_D10' / suite / 'run'
        for root in (candidate_root, baseline_root):
            validation = json.loads((root / 'validated.json').read_text())
            if not validation.get('complete'):
                raise RuntimeError('Unvalidated paired rollout: %s' % root)
        candidate_episodes = json.loads((candidate_root / 'episode_log.json').read_text())
        baseline_episodes = json.loads((baseline_root / 'episode_log.json').read_text())
        task_ids = set(condition['task_ids'])
        expected = {(task, init) for task in task_ids for init in condition['init_indices']}
        candidate = {(row['task_id'], row['episode_idx']): row for row in candidate_episodes}
        baseline = {(row['task_id'], row['episode_idx']): row for row in baseline_episodes
                    if row['task_id'] in task_ids}
        if set(candidate) != expected or set(baseline) != expected:
            raise RuntimeError('Paired episode coverage mismatch for %s' % suite)

        candidate_success = baseline_success = both_success = candidate_only = baseline_only = 0
        suite_rows = []
        candidate_latencies, baseline_latencies = [], []
        cold_key = (min(task_ids), min(condition['init_indices']))
        for key in sorted(expected):
            candidate_episode, baseline_episode = candidate[key], baseline[key]
            c_ok, b_ok = bool(candidate_episode['success']), bool(baseline_episode['success'])
            candidate_success += int(c_ok)
            baseline_success += int(b_ok)
            both_success += int(c_ok and b_ok)
            candidate_only += int(c_ok and not b_ok)
            baseline_only += int(b_ok and not c_ok)
            c_requests = json_lines(candidate_root / candidate_episode['infer_path'])
            b_requests = json_lines(baseline_root / baseline_episode['infer_path'])
            accepted = sum(not bool(req['experiment10']['used_full_fallback']) for req in c_requests)
            parallel_episode_total = float(sum(float(req['client_roundtrip_ms']) for req in c_requests))
            baseline_episode_total = float(sum(float(req['client_roundtrip_ms']) for req in b_requests))
            parallel_episode_steady = float(sum(
                float(req['client_roundtrip_ms']) for req in c_requests
                if not (key == cold_key and int(req['infer_id']) == 0)))
            baseline_episode_steady = float(sum(
                float(req['client_roundtrip_ms']) for req in b_requests
                if not (key == cold_key and int(req['infer_id']) == 0)))
            for request in c_requests:
                meta = request['experiment10']
                cold = (key == cold_key and int(request['infer_id']) == 0)
                latency = float(request['client_roundtrip_ms'])
                row = {
                    'suite': suite, 'task_id': key[0], 'episode_idx': key[1],
                    'infer_id': int(request['infer_id']),
                    'route': 'full_fallback' if meta['used_full_fallback'] else 'draft_prefix',
                    'accepted_prefix_len': int(request['accepted_prefix_len']),
                    'candidate_accepted_prefix_len': int(meta['candidate_accepted_prefix_len']),
                    'verification_reason': meta['verification_reason'],
                    'client_roundtrip_ms': latency,
                    'parallel_inference_wall_ms': float(meta['parallel_inference_wall_ms']),
                    'candidate_cuda_policy_ms': float(meta['candidate_cuda_policy_ms']),
                    'reference_cuda_policy_ms': float(meta['reference_cuda_policy_ms']),
                    'cuda_compute_sum_ms': float(meta['cuda_policy_ms']),
                    'excluded_first_request_cold_start': cold,
                }
                inference_rows.append(row)
                suite_rows.append(row)
                if not cold:
                    candidate_latencies.append(latency)
            # Baseline latency is summarized on its own replan requests; request IDs
            # are not paired because a different executed prefix changes replanning.
            baseline_latencies.extend(
                float(request['client_roundtrip_ms']) for request in b_requests
                if not (key == cold_key and int(request['infer_id']) == 0))
            all_parallel_episode_totals.append(parallel_episode_steady)
            all_baseline_episode_totals.append(baseline_episode_steady)
            episode_rows.append({
                'suite': suite, 'task_id': key[0], 'episode_idx': key[1],
                'parallel_success': int(c_ok), 'baseline_success': int(b_ok),
                'paired_outcome': ('both_success' if c_ok and b_ok else
                                   'parallel_only_success' if c_ok else
                                   'baseline_only_success' if b_ok else 'both_failed'),
                'parallel_inference_calls': len(c_requests),
                'draft_prefix_calls': accepted,
                'full_fallback_calls': len(c_requests) - accepted,
                'parallel_roundtrip_total_ms': parallel_episode_total,
                'parallel_steady_roundtrip_total_ms': parallel_episode_steady,
                'baseline_inference_calls': len(b_requests),
                'baseline_roundtrip_total_ms': baseline_episode_total,
                'baseline_steady_roundtrip_total_ms': baseline_episode_steady,
            })

        suite_drafts = [row for row in suite_rows if row['route'] == 'draft_prefix']
        suite_summaries[suite] = {
            'episodes': len(expected),
            'parallel_successes': candidate_success,
            'baseline_successes': baseline_success,
            'paired_both_success': both_success,
            'parallel_only_success': candidate_only,
            'baseline_only_success': baseline_only,
            'inference_calls': len(suite_rows),
            'draft_prefix_calls': len(suite_drafts),
            'full_fallback_calls': len(suite_rows) - len(suite_drafts),
            'draft_prefix_ratio': len(suite_drafts) / len(suite_rows),
            'parallel_client_roundtrip_mean_ms': float(np.mean(candidate_latencies)),
            'parallel_client_roundtrip_median_ms': float(np.median(candidate_latencies)),
            'parallel_client_roundtrip_p95_ms': float(np.quantile(candidate_latencies, .95)),
            'baseline_client_roundtrip_mean_ms': float(np.mean(baseline_latencies)),
            'parallel_mean_episode_roundtrip_total_ms': float(np.mean([
                row['parallel_steady_roundtrip_total_ms'] for row in episode_rows if row['suite'] == suite])),
            'baseline_mean_episode_roundtrip_total_ms': float(np.mean([
                row['baseline_steady_roundtrip_total_ms'] for row in episode_rows if row['suite'] == suite])),
            'parallel_cuda_compute_sum_ms': float(sum(row['cuda_compute_sum_ms'] for row in suite_rows)),
        }

    all_drafts = [row for row in inference_rows if row['route'] == 'draft_prefix']
    all_candidate_latency = [row['client_roundtrip_ms'] for row in inference_rows
                             if not row['excluded_first_request_cold_start']]
    all_baseline_latency = [value for summary in suite_summaries.values()
                            for value in [summary['baseline_client_roundtrip_mean_ms']]]
    output = RESULTS / 'parallel_verify_horizons' / condition_name
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'condition': condition_name,
        'horizon': int(condition['horizon']),
        'evaluation_split': 'tasks8-9/init0-4 (40 episodes)',
        'reference': 'paired Full18_D10 episode outcomes; unpaired per-request latency means',
        'complete': True,
        'episodes': len(episode_rows),
        'parallel_successes': sum(row['parallel_success'] for row in episode_rows),
        'baseline_successes': sum(row['baseline_success'] for row in episode_rows),
        'paired_both_success': sum(row['paired_outcome'] == 'both_success' for row in episode_rows),
        'parallel_only_success': sum(row['paired_outcome'] == 'parallel_only_success' for row in episode_rows),
        'baseline_only_success': sum(row['paired_outcome'] == 'baseline_only_success' for row in episode_rows),
        'inference_calls': len(inference_rows),
        'draft_prefix_calls': len(all_drafts),
        'full_fallback_calls': len(inference_rows) - len(all_drafts),
        'draft_prefix_ratio': len(all_drafts) / len(inference_rows),
        'latency_excludes_first_inference_per_suite': True,
        'parallel_client_roundtrip_mean_ms': float(np.mean(all_candidate_latency)),
        'parallel_client_roundtrip_median_ms': float(np.median(all_candidate_latency)),
        'parallel_client_roundtrip_p95_ms': float(np.quantile(all_candidate_latency, .95)),
        'parallel_mean_episode_roundtrip_total_ms': float(np.mean(all_parallel_episode_totals)),
        'baseline_mean_episode_roundtrip_total_ms': float(np.mean(all_baseline_episode_totals)),
        'parallel_vs_baseline_mean_episode_roundtrip_ratio': (
            float(np.mean(all_parallel_episode_totals) / np.mean(all_baseline_episode_totals))),
        'parallel_cuda_compute_sum_ms': float(sum(row['cuda_compute_sum_ms'] for row in inference_rows)),
        'suite_summaries': suite_summaries,
        'limitations': [
            'Only 40 paired episodes on tasks8-9/init0-4; exploratory screen',
            'Two GPUs are used for each parallel rollout; latency does not equal system throughput',
            'Same-observation agreement with Full18 is not semantic correctness',
            'Latency means are per-run replan requests; episode success comparisons are paired by task/init',
        ],
    }
    # Average the four suite means, rather than weighting by differing replan-call counts,
    # for the reported baseline reference; retain per-suite values above.
    summary['baseline_client_roundtrip_mean_ms_suite_equal'] = float(np.mean(all_baseline_latency))
    dump_json(output / 'summary.json', summary)
    write_csv(output / 'episodes.csv', episode_rows)
    write_csv(output / 'inferences.csv', inference_rows)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions', type=argparse.FileType('r'), required=True)
    parser.add_argument('--condition', required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.condition, json.load(args.conditions)), indent=2), flush=True)


if __name__ == '__main__':
    main()
