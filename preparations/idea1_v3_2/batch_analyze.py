"""Audit completed E1 evidence without attributing free-rollout drift to batching."""
import argparse
import json
from pathlib import Path
import numpy as np
from .batch_policy import compare
from preparations.idea1_v3_1.common import atomic_json, SUITES


def analyze(output):
    output = Path(output)
    if (output/'result.json').exists():
        receipt = json.loads((output/'result.json').read_text())
    else:
        receipt = dict(json.loads((output/'wait_scan_controller.json').read_text()), bank_results=[],arms=[],controls=[],
            official_scenes=[dict(suite=s,task=0,init=i) for s in SUITES for i in (0,1)])
    rescue = json.loads((output/'rescue.json').read_text()) if (output/'rescue.json').exists() else {}
    arm_metadata = receipt['arms'] + rescue.get('arms', [])
    bank = []
    for row in receipt['bank_results']:
        checks = [c[k] for c in row['checks'] for k in ('raw', 'decoded')]
        bank.append(dict(batch_size=row['batch_size'], completed_examples=row['completed_examples'],
            examples_per_s=row['completed_examples_per_s'], relaxed_passed=row['numerical_passed'],
            strict_1e5_passed=all(c['strict_1e5_passed'] for c in checks),
            largest_raw_abs=max(c['raw']['max_abs'] for c in row['checks']),
            largest_decoded_abs=max(c['decoded']['max_abs'] for c in row['checks'])))
    scenes = receipt['official_scenes']
    arms = {}
    for arm in arm_metadata:
        rows = {}
        for scene in scenes:
            directory = output/arm['arm']/scene['suite']/f"init{scene['init']}"
            if not (directory/'complete.json').exists():
                continue
            key = (scene['suite'], scene['task'], scene['init'])
            rows[key] = dict(result=json.loads((directory/'result.json').read_text()),
                identity=json.loads((directory/'identity.json').read_text()),
                infers=[json.loads(line) for line in (directory/'infer.jsonl').read_text().splitlines()])
        arms[arm['arm']] = rows
    pairs = []
    baseline_name = 'b1_before' if 'b1_before' in arms else 'rescue_b1_after'
    baseline = arms.get(baseline_name, {})
    for name, rows in arms.items():
        if name == baseline_name:
            continue
        for key, actual in rows.items():
            if key not in baseline:
                continue
            ref = baseline[key]
            matched, drifted, differences = 0, 0, []
            for a, b in zip(ref['infers'], actual['infers']):
                if (a['input_hash'], a['noise_hash'], a['seed']) == (b['input_hash'], b['noise_hash'], b['seed']):
                    matched += 1
                    differences.append(compare(a['actions'], b['actions']))
                else:
                    drifted += 1
            drifted += abs(len(ref['infers'])-len(actual['infers']))
            pairs.append(dict(arm=name, scene=list(key),
                same_official_init=ref['identity']['init_vector_sha256']==actual['identity']['init_vector_sha256'],
                same_first_input=ref['infers'][0]['input_hash']==actual['infers'][0]['input_hash'],
                baseline_success=ref['result']['success'], actual_success=actual['result']['success'],
                baseline_actions=ref['result']['actions'], actual_actions=actual['result']['actions'],
                baseline_predictions=ref['result']['predictions'], actual_predictions=actual['result']['predictions'],
                matched_input_noise_requests=matched, drifted_or_unpaired_requests=drifted,
                matched_action_checks=differences))
    metrics = []
    base_wall = next((a.get('wall_s') for a in arm_metadata if a['arm']=='rescue_b1_after'), None)
    if base_wall is None:
        base_wall = next((a.get('wall_s') for a in arm_metadata if a['arm']=='b1_before'), None)
    for arm in arm_metadata:
        if 'wall_s' not in arm:
            continue
        infer_rows=[row for scene in arms[arm['arm']].values() for row in scene['infers']]
        unique_batches={tuple(row['timing'][k] for k in ('prepare_ms','model_completed_ms','request_completed_ms','batch_size')):
                        row['timing'] for row in infer_rows}
        queue_collect=[row['server_dispatch_ms']-row['timing']['request_completed_ms'] for row in infer_rows]
        model_by_size={str(size):dict(count=sum(t['batch_size']==size for t in unique_batches.values()),
            median_completed_ms=float(np.median([t['model_completed_ms'] for t in unique_batches.values() if t['batch_size']==size])))
            for size in sorted(set(t['batch_size'] for t in unique_batches.values()))}
        metrics.append(dict(arm=arm['arm'], wall_s=arm['wall_s'],
            completed_episodes=arm['completed_episodes'], successes=arm['successes'],
            completed_actions=arm['completed_actions'], predictions=arm['predictions'],
            episodes_per_hour=arm['episodes_per_hour'], actions_per_s=arm['completed_actions']/arm['wall_s'],
            requests_per_s=arm['predictions']/arm['wall_s'],
            throughput_vs_measured_b1=base_wall/arm['wall_s'], actual_batch_histogram=arm['actual_batch_histogram'],
            unique_batch_model_timings=model_by_size,
            summed_unique_batch_model_completed_s=sum(t['model_completed_ms'] for t in unique_batches.values())/1000,
            queue_plus_collect_ms_p50_p95=np.percentile(queue_collect,[50,95]).tolist(),
            timing_note='Model-completed timing includes host materialization. Queue+collect is enqueue-to-response minus completed request timing. Unique batches deduplicated by shared timing tuple.'))
    summary = dict(run_status=receipt['status'], rescue_status=rescue.get('status'), bank=bank, arms=metrics, pairs=pairs,
        numerical_parity_passed=False if rescue else all(r['strict_1e5_passed'] for r in bank),
        quality_reference_arm=baseline_name,
        same_service_fixed_controls=rescue.get('controls',receipt['controls']),
        throughput_comparator='measured rescue_b1_after; initial B1 arm wall unavailable' if rescue else 'b1_before',
        caveats=['Eight fixed scenes per arm support a bounded paired check, not a full success-rate claim.',
                 'Pointwise actions are compared only where input, noise and seed match; free closed-loop drift is reported.',
                 'Episode wall includes environment construction/reset, RPC, rendering, simulation and lean telemetry.',
                 'Closed-loop action/prediction counts can differ, so episode throughput includes changed policy trajectories and is not identical-work hardware acceleration.',
                 'GPU memory figures are snapshots after completed bank configurations, not sampled peak memory.'])
    atomic_json(output/'analysis.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    args = parser.parse_args()
    result=analyze(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','same_service_fixed_controls')}, indent=2))
