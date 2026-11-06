"""Summarize paired parallel D3/Full18 verification rollouts and request latency."""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, json_lines


def summarize(condition_name='ParallelVerifyD3_H8_T89'):
    output = RESULTS / 'parallel_verify'
    episode_rows, inference_rows, suite_summaries = [], [], {}
    all_parallel_latency, all_sync_latency = [], []
    for suite in SUITES:
        root = RESULTS / 'closed_loop' / condition_name / suite / 'run'
        validation = json.loads((root / 'validated.json').read_text())
        if not validation.get('complete'):
            raise RuntimeError('Parallel-verifier suite is not validated: %s' % suite)
        episodes = json.loads((root / 'episode_log.json').read_text())
        baseline_root = RESULTS / 'closed_loop' / 'Full18_D10' / suite / 'run'
        sync_root = RESULTS / 'closed_loop' / 'VerifyD3_H8' / suite / 'run'
        baseline_rows = json.loads((baseline_root / 'episode_log.json').read_text())
        sync_rows = json.loads((sync_root / 'episode_log.json').read_text())
        baseline = {(row['task_id'], row['episode_idx']): row for row in baseline_rows
                    if row['task_id'] in (8, 9)}
        synchronous = {(row['task_id'], row['episode_idx']): row for row in sync_rows
                       if row['task_id'] in (8, 9)}
        if len(episodes) != 10 or len(baseline) != 10 or len(synchronous) != 10:
            raise RuntimeError('Expected paired tasks8-9/init0-4 runs in %s' % suite)

        parallel_success = baseline_success = sync_success = both_success = 0
        parallel_only = baseline_only = 0
        suite_inferences = []
        cold_episode_key = (episodes[0]['task_id'], episodes[0]['episode_idx'])
        for episode in episodes:
            key = (episode['task_id'], episode['episode_idx'])
            if key not in baseline or key not in synchronous:
                raise RuntimeError('Missing paired reference episode: %s %r' % (suite, key))
            baseline_episode, sync_episode = baseline[key], synchronous[key]
            p_ok, b_ok, s_ok = bool(episode['success']), bool(baseline_episode['success']), bool(sync_episode['success'])
            parallel_success += int(p_ok)
            baseline_success += int(b_ok)
            sync_success += int(s_ok)
            both_success += int(p_ok and b_ok)
            parallel_only += int(p_ok and not b_ok)
            baseline_only += int(b_ok and not p_ok)

            requests = json_lines(root / episode['infer_path'])
            parallel_ms, sync_ms, baseline_ms = [], [], []
            sync_requests = json_lines(sync_root / sync_episode['infer_path'])
            sync_by_id = {int(request['infer_id']): request for request in sync_requests}
            baseline_requests = json_lines(baseline_root / baseline_episode['infer_path'])
            for request in requests:
                meta = request['experiment10']
                route = 'full_fallback' if bool(meta['used_full_fallback']) else 'draft_prefix'
                latency = float(request['client_roundtrip_ms'])
                parallel_ms.append(latency)
                sync_request = sync_by_id.get(int(request['infer_id']))
                sync_meta = sync_request.get('experiment10', {}) if sync_request else {}
                action_diff = None
                action_match = None
                if sync_request is not None:
                    parallel_actions = np.asarray(request['chunk_actions'], dtype=float)
                    sync_actions = np.asarray(sync_request['chunk_actions'], dtype=float)
                    executed = min(int(request['chunk_exec_len']), int(sync_request['chunk_exec_len']))
                    if parallel_actions.shape != sync_actions.shape or executed < 1:
                        raise RuntimeError('Parallel/sync action comparison has incompatible chunks')
                    action_diff = float(np.max(np.abs(parallel_actions[:executed] - sync_actions[:executed])))
                    action_match = bool(np.allclose(parallel_actions[:executed], sync_actions[:executed],
                                                    atol=1e-6, rtol=0.0))
                cold_start_request = ((episode['task_id'], episode['episode_idx']) == cold_episode_key and
                                      int(request['infer_id']) == 0)
                row = {
                    'suite': suite, 'task_id': episode['task_id'], 'episode_idx': episode['episode_idx'],
                    'infer_id': request['infer_id'], 'route': route,
                    'accepted_prefix_len': request['accepted_prefix_len'],
                    'candidate_accepted_prefix_len': meta['candidate_accepted_prefix_len'],
                    'verification_reason': meta['verification_reason'],
                    'parallel_client_roundtrip_ms': latency,
                    'sync_client_roundtrip_ms': (
                        float(sync_request['client_roundtrip_ms']) if sync_request is not None else None),
                    'parallel_policy_sample_actions_ms': meta['sample_actions_ms'],
                    'candidate_server_sample_actions_ms': meta['candidate_server_sample_actions_ms'],
                    'reference_server_sample_actions_ms': meta['reference_server_sample_actions_ms'],
                    'parallel_inference_wall_ms': meta['parallel_inference_wall_ms'],
                    'candidate_cuda_policy_ms': meta['candidate_cuda_policy_ms'],
                    'reference_cuda_policy_ms': meta['reference_cuda_policy_ms'],
                    'parallel_cuda_sum_ms': meta['cuda_policy_ms'],
                    'excluded_as_first_request_cold_start': cold_start_request,
                    'sync_candidate_accepted_prefix_len': sync_meta.get('candidate_accepted_prefix_len'),
                    'sync_verification_reason': sync_meta.get('verification_reason'),
                    'sync_candidate_cuda_policy_ms': sync_meta.get('candidate_cuda_policy_ms'),
                    'sync_reference_cuda_policy_ms': sync_meta.get('reference_cuda_policy_ms'),
                    'sync_cuda_sum_ms': (float(sync_meta.get('candidate_cuda_policy_ms', 0.0)) +
                                         float(sync_meta.get('reference_cuda_policy_ms', 0.0))),
                    'route_matches_sync_verifier': (
                        bool(sync_meta) and
                        int(sync_meta.get('candidate_accepted_prefix_len', -1)) ==
                        int(meta['candidate_accepted_prefix_len']) and
                        bool(sync_meta.get('used_full_fallback')) == bool(meta['used_full_fallback'])
                    ),
                    'executed_prefix_action_max_abs_diff_vs_sync': action_diff,
                    'executed_prefix_actions_match_sync_atol_1e-6': action_match,
                }
                inference_rows.append(row)
                suite_inferences.append(row)
            sync_ms = [float(request['client_roundtrip_ms']) for request in sync_requests]
            baseline_ms = [float(request['client_roundtrip_ms']) for request in baseline_requests]
            episode_rows.append({
                'suite': suite, 'task_id': episode['task_id'], 'episode_idx': episode['episode_idx'],
                'parallel_success': int(p_ok), 'sync_verify_success': int(s_ok),
                'baseline_success': int(b_ok),
                'paired_outcome': ('both_success' if p_ok and b_ok else
                                   'parallel_only_success' if p_ok else
                                   'baseline_only_success' if b_ok else 'both_failed'),
                'inference_calls': len(requests),
                'draft_prefix_calls': sum(not bool(r['experiment10']['used_full_fallback']) for r in requests),
                'parallel_client_roundtrip_total_ms': float(np.sum(parallel_ms)),
                'parallel_client_roundtrip_mean_ms': float(np.mean(parallel_ms)),
                'sync_client_roundtrip_mean_ms': float(np.mean(sync_ms)),
                'baseline_client_roundtrip_mean_ms': float(np.mean(baseline_ms)),
            })

        suite_calls = [row for row in suite_inferences if row['suite'] == suite]
        draft_calls = [row for row in suite_calls if row['route'] == 'draft_prefix']
        route_comparable = [row for row in suite_calls if row['sync_candidate_accepted_prefix_len'] is not None]
        latencies = [row['parallel_client_roundtrip_ms'] for row in suite_calls
                     if not row['excluded_as_first_request_cold_start']]
        sync_latencies = [float(request['client_roundtrip_ms']) for key, episode in synchronous.items()
                          for request in json_lines(sync_root / episode['infer_path'])
                          if not (key == cold_episode_key and int(request['infer_id']) == 0)]
        all_parallel_latency.extend(latencies)
        all_sync_latency.extend(sync_latencies)
        suite_summaries[suite] = {
            'episodes': len(episodes),
            'parallel_successes': parallel_success,
            'sync_verify_successes': sync_success,
            'baseline_successes': baseline_success,
            'paired_both_success': both_success,
            'parallel_only_success': parallel_only,
            'baseline_only_success': baseline_only,
            'inference_calls': len(suite_calls),
            'draft_prefix_calls': len(draft_calls),
            'full_fallback_calls': len(suite_calls) - len(draft_calls),
            'draft_prefix_ratio': len(draft_calls) / len(suite_calls),
            'sync_route_comparable_calls': len(route_comparable),
            'route_agreement_rate_vs_sync_verifier': (
                float(np.mean([row['route_matches_sync_verifier'] for row in route_comparable]))
                if route_comparable else None
            ),
            'parallel_client_roundtrip_mean_ms': float(np.mean(latencies)),
            'parallel_client_roundtrip_median_ms': float(np.median(latencies)),
            'parallel_client_roundtrip_p95_ms': float(np.quantile(latencies, .95)),
            'parallel_client_roundtrip_p99_ms': float(np.quantile(latencies, .99)),
            'parallel_client_roundtrip_max_ms': float(np.max(latencies)),
            'parallel_calls_over_100ms': sum(value > 100.0 for value in latencies),
            'sync_verify_client_roundtrip_mean_ms': float(np.mean(sync_latencies)),
            'parallel_vs_sync_mean_latency_ratio': float(np.mean(latencies) / np.mean(sync_latencies)),
            'parallel_cuda_compute_sum_ms': float(sum(row['parallel_cuda_sum_ms'] for row in suite_calls)),
            'sync_cuda_compute_sum_ms': float(sum(row['sync_cuda_sum_ms'] for row in suite_calls)),
        }

    all_calls = inference_rows
    route_comparable = [row for row in all_calls if row['sync_candidate_accepted_prefix_len'] is not None]
    action_comparable = [row for row in all_calls
                         if row['executed_prefix_actions_match_sync_atol_1e-6'] is not None]
    summary = {
        'condition': condition_name,
        'reference': 'paired Full18_D10 plus prior synchronous VerifyD3_H8 on identical tasks/inits',
        'evaluation_split': 'tasks8-9/init0-4 (40 episodes)',
        'complete': True,
        'episodes': len(episode_rows),
        'parallel_successes': sum(row['parallel_success'] for row in episode_rows),
        'sync_verify_successes': sum(row['sync_verify_success'] for row in episode_rows),
        'baseline_successes': sum(row['baseline_success'] for row in episode_rows),
        'paired_both_success': sum(row['paired_outcome'] == 'both_success' for row in episode_rows),
        'parallel_only_success': sum(row['paired_outcome'] == 'parallel_only_success' for row in episode_rows),
        'baseline_only_success': sum(row['paired_outcome'] == 'baseline_only_success' for row in episode_rows),
        'inference_calls': len(all_calls),
        'draft_prefix_calls': sum(row['route'] == 'draft_prefix' for row in all_calls),
        'full_fallback_calls': sum(row['route'] == 'full_fallback' for row in all_calls),
        'draft_prefix_ratio': sum(row['route'] == 'draft_prefix' for row in all_calls) / len(all_calls),
        'sync_route_comparable_calls': len(route_comparable),
        'route_matches_sync_verifier': sum(row['route_matches_sync_verifier'] for row in route_comparable),
        'route_agreement_rate_vs_sync_verifier': (
            sum(row['route_matches_sync_verifier'] for row in route_comparable) / len(route_comparable)
            if route_comparable else None
        ),
        'executed_prefix_action_comparable_calls': len(action_comparable),
        'executed_prefix_actions_match_sync_calls': sum(
            row['executed_prefix_actions_match_sync_atol_1e-6'] for row in action_comparable),
        'executed_prefix_action_match_rate_vs_sync': (
            float(np.mean([row['executed_prefix_actions_match_sync_atol_1e-6'] for row in action_comparable]))
            if action_comparable else None
        ),
        'executed_prefix_max_abs_action_diff_vs_sync': (
            float(max(row['executed_prefix_action_max_abs_diff_vs_sync'] for row in action_comparable))
            if action_comparable else None
        ),
        'latency_summary_excludes_first_inference_per_suite': True,
        'parallel_client_roundtrip_mean_ms': float(np.mean(all_parallel_latency)),
        'parallel_client_roundtrip_median_ms': float(np.median(all_parallel_latency)),
        'parallel_client_roundtrip_p95_ms': float(np.quantile(all_parallel_latency, .95)),
        'parallel_client_roundtrip_p99_ms': float(np.quantile(all_parallel_latency, .99)),
        'parallel_client_roundtrip_max_ms': float(np.max(all_parallel_latency)),
        'parallel_calls_over_100ms': sum(value > 100.0 for value in all_parallel_latency),
        'sync_verify_client_roundtrip_mean_ms': float(np.mean(all_sync_latency)),
        'parallel_vs_sync_mean_latency_ratio': float(np.mean(all_parallel_latency) / np.mean(all_sync_latency)),
        'task9_first_inference_calls': sum(row['task_id'] == 9 and row['infer_id'] == 0 for row in all_calls),
        'parallel_task9_first_inference_mean_ms': float(np.mean([
            row['parallel_client_roundtrip_ms'] for row in all_calls
            if row['task_id'] == 9 and row['infer_id'] == 0] or [float('nan')])),
        'parallel_task9_first_inference_median_ms': float(np.median([
            row['parallel_client_roundtrip_ms'] for row in all_calls
            if row['task_id'] == 9 and row['infer_id'] == 0] or [float('nan')])),
        'parallel_task9_first_inference_max_ms': float(np.max([
            row['parallel_client_roundtrip_ms'] for row in all_calls
            if row['task_id'] == 9 and row['infer_id'] == 0] or [float('nan')])),
        'sync_task9_first_inference_mean_ms': float(np.mean([
            row['sync_client_roundtrip_ms'] for row in all_calls
            if row['task_id'] == 9 and row['infer_id'] == 0 and row['sync_client_roundtrip_ms'] is not None]
            or [float('nan')])),
        'sync_task9_first_inference_median_ms': float(np.median([
            row['sync_client_roundtrip_ms'] for row in all_calls
            if row['task_id'] == 9 and row['infer_id'] == 0 and row['sync_client_roundtrip_ms'] is not None]
            or [float('nan')])),
        'sync_task9_first_inference_max_ms': float(np.max([
            row['sync_client_roundtrip_ms'] for row in all_calls
            if row['task_id'] == 9 and row['infer_id'] == 0 and row['sync_client_roundtrip_ms'] is not None]
            or [float('nan')])),
        'parallel_cuda_compute_sum_ms': float(sum(row['parallel_cuda_sum_ms'] for row in all_calls)),
        'sync_cuda_compute_sum_ms': float(sum(row['sync_cuda_sum_ms'] for row in all_calls)),
        'suite_summaries': suite_summaries,
        'limitations': [
            'Only 40 paired episodes on tasks8-9/init0-4; exploratory comparison',
            'Same-observation agreement with Full18 is not semantic correctness',
            'Both GPU workers run concurrently; latency uses client websocket roundtrip and does not represent total GPU compute or system throughput',
            'The first inference of each suite process is excluded from latency comparisons to remove cold-start/import effects',
            'Reference sync verifier was run earlier; environments are deterministic by fixed task/init/seed but process/runtime conditions can differ',
        ],
    }
    dump_json(output / 'parallel_verify_summary.json', summary)
    for filename, rows in (('parallel_verify_episodes.csv', episode_rows),
                           ('parallel_verify_inferences.csv', inference_rows)):
        temporary = output / (filename + '.tmp')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output / filename)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', default='ParallelVerifyD3_H8_T89')
    args = parser.parse_args()
    print(json.dumps(summarize(args.condition), indent=2), flush=True)


if __name__ == '__main__':
    main()
