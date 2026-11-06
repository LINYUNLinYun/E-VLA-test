"""Independent CPU-only runtime/hardware summary after the 160-case audit.

Read the final requests/summary/episodes CSVs once. Hot statistics use complete
five-request cycles starting at replan15, matching the frozen report window.
Free closed-loop wall differences are descriptive; matched-input U/W routes
with the same A/F mask cannot establish an algorithmic speed advantage.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparations.idea1_v3.io import atomic

OUT = ROOT / 'results/idea1_v3_4_E2_mixed_cost'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, field):
    value = row.get(field)
    return None if value in (None, '', 'None', 'nan') else float(value)


def stats(values):
    values = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if not len(values):
        return dict(n=0, mean=None, median=None, p95=None, maximum=None, minimum=None, sum=0.)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()), minimum=float(values.min()), sum=float(values.sum()))


def complete_cycles(rows):
    cases = defaultdict(list)
    for row in rows:
        cases[tuple(row[k] for k in ('suite', 'task', 'init'))].append(row)
    cycles, covered = [], 0
    for key, bank in cases.items():
        bank.sort(key=lambda r: int(r['replan']))
        if [int(r['replan']) for r in bank] != list(range(len(bank))):
            raise ValueError(f'Non-contiguous finalized request journal: {key}')
        own = [bank[index:index + 5] for index in range(15, len(bank) - 4, 5)]
        covered += bool(own)
        cycles.extend(own)
    return cycles, covered


def module_counts(rows):
    values = defaultdict(list)
    for row in rows:
        a, f = set(json.loads(row['actual_attn'])), set(json.loads(row['actual_ffn']))
        for name, value in dict(attn=len(a), ffn=len(f), af_joint=len(a & f),
                                a_only=len(a - f), f_only=len(f - a), support=len(a | f), individual_modules=len(a) + len(f)).items():
            values[name].append(value)
    result = {name: stats(v) for name, v in values.items()}
    denominator = sum(values['support'])
    result['support_state_fractions'] = {name: None if denominator == 0 else sum(values[name]) / denominator for name in ('a_only', 'f_only', 'af_joint')}
    return result


def mask_cost(row, costs):
    a, f = set(json.loads(row['actual_attn'])), set(json.loads(row['actual_ffn']))
    terms = [(layer, 'af') for layer in a & f] + [(layer, 'attn') for layer in a - f] + [(layer, 'ffn') for layer in f - a]
    saved = math.fsum(costs[layer][kind + '_saved_ms'] for layer, kind in terms)
    noise_sum = math.fsum(costs[layer][kind + '_noise_ms'] for layer, kind in terms)
    return saved, noise_sum


def condition_report(cfg, rows, episode_rows, official, costs, budget):
    cycles, coverage = complete_cycles(rows)
    steady = [row for cycle in cycles for row in cycle]
    full = [row for row in steady if row['path'] == 'full']
    fast = [row for row in steady if row['path'] == 'fast']
    all_full = [row for row in rows if row['path'] == 'full']
    all_fast = [row for row in rows if row['path'] == 'fast']
    captures = [row for row in rows if number(row, 'selector_capture_ms') > 0]
    for row in rows:
        if abs(number(row, 'policy_hot_ms') - (number(row, 'policy_ms') - number(row, 'selector_capture_ms'))) > 1e-5:
            raise ValueError('Stored hot-time subtraction differs from policy wall minus selector capture')
    saved, uncertainty = [], []
    lower, upper = budget['effective_window_ms']
    for row in all_fast:
        cost, noise = mask_cost(row, costs)
        if abs(cost - number(row, 'estimated_saved_ms')) > 1e-8 or not lower <= cost <= upper:
            raise ValueError('Actual selected mask differs from the frozen joint-state cost/window')
        saved.append(cost)
        uncertainty.append(noise)
    amortized = {}
    for field in ('selector_hot_ms', 'selection_ms', 'error_build_ms', 'selector_capture_ms'):
        values = [number(row, field) for row in steady]
        amortized[field] = dict(per_request_ms=stats(values)['mean'],
            per_complete_five_request_cycle_ms=stats([sum(number(row, field) for row in cycle) for cycle in cycles])['mean'],
            per_Full_ms=stats([number(row, field) for row in full]))
    row = dict(condition=cfg['id'], budget=cfg['budget'], objective='W' if cfg['cost_weighted'] else 'U',
        episodes=len(episode_rows), requests=len(rows), full_requests=len(all_full), fast_requests=len(all_fast),
        hot_window_start_replan=15, complete_hot_cycles=len(cycles), hot_covered_episodes=coverage,
        full_hot_ms=stats([number(r, 'policy_hot_ms') for r in full]),
        fast_hot_ms=stats([number(r, 'policy_hot_ms') for r in fast]),
        five_request_hot_per_request_ms=stats([number(r, 'policy_hot_ms') for r in steady]),
        five_request_hot_cycle_mean_ms=stats([np.mean([number(r, 'policy_hot_ms') for r in cycle]) for cycle in cycles]),
        five_request_hot_cycle_total_ms=stats([sum(number(r, 'policy_hot_ms') for r in cycle) for cycle in cycles]),
        fast_hot_by_position={str(p): stats([number(r, 'policy_hot_ms') for r in fast if int(r['fast_position']) == p]) for p in range(1, 5)},
        selector_and_CPU_mask_selection_amortization=amortized,
        cold_graph_capture_ms=stats([number(r, 'selector_capture_ms') for r in captures]),
        first_R_graph_capture_ms=stats([number(r, 'selector_capture_ms') for r in captures if int(r['replan']) == 0]),
        E_switch_graph_capture_ms=stats([number(r, 'selector_capture_ms') for r in captures if int(r['replan']) == 10]),
        all_request_policy_ms_including_cold=stats([number(r, 'policy_ms') for r in rows]),
        all_request_RPC_ms_including_cold_queue=stats([number(r, 'rpc_ms') for r in rows]),
        all_Full_policy_ms_including_cold=stats([number(r, 'policy_ms') for r in all_full]),
        actual_skip_counts_all_Fast=module_counts(all_fast), actual_skip_counts_hot_Fast=module_counts(fast),
        estimated_net_hardware_saved_ms_all_Fast=stats(saved), sum_of_marginal_state_MADs_ms=stats(uncertainty),
        estimated_cost_window_ms=[lower, upper],
        episode_wall_s=stats([number(r, 'wall_s') for r in episode_rows]),
        official_summary=official)
    checks = dict(complete_hot_cycles=int(official['complete_hot_cycles']) == len(cycles),
        hot_covered_episodes=int(official['hot_covered_episodes']) == coverage,
        full_hot_mean_ms=math.isclose(number(official, 'full_hot_ms'), row['full_hot_ms']['mean'], abs_tol=1e-6),
        five_request_hot_mean_ms=math.isclose(number(official, 'five_request_hot_mean_ms'), row['five_request_hot_per_request_ms']['mean'], abs_tol=1e-6),
        selector_capture_total_ms=math.isclose(number(official, 'selector_capture_total_ms'), row['cold_graph_capture_ms']['sum'], abs_tol=1e-5))
    if not all(checks.values()):
        raise ValueError(f'Independent runtime aggregation differs from official summary: {cfg["id"]} {checks}')
    row['official_summary_cross_checks'] = checks
    return row


def same_input_report(data):
    result = []
    for budget in ('B4', 'B6'):
        rows = {r['route']: r for r in data['routes'] if r['budget'] == budget}
        u, w = rows['U'], rows['W']
        same = u['attn'] == w['attn'] and u['ffn'] == w['ffn']
        paired = [r for r in data['paired'] if r['budget'] == budget]
        result.append(dict(budget=budget, identical_U_W_masks=same,
            U_attn=u['attn'], U_ffn=u['ffn'], W_attn=w['attn'], W_ffn=w['ffn'],
            U_median_wall_ms=u['median_wall_ms'], W_median_wall_ms=w['median_wall_ms'],
            U_median_vlm_ms=u['median_vlm_ms'], W_median_vlm_ms=w['median_vlm_ms'],
            paired_W_minus_U_wall_ms=stats([r['W_minus_U_wall_ms'] for r in paired]),
            paired_W_minus_U_vlm_ms=stats([r['W_minus_U_vlm_ms'] for r in paired]),
            algorithmic_speedup_demonstrated=False,
            interpretation='U and W executed identical masks; wall/event differences measure order and runtime variation, not a change in skipped computation.' if same else 'Four repeated measurements on one input are descriptive and do not establish a general algorithmic speedup.'))
    return dict(input_scope=data['input_scope'], gpu_uuid=data['gpu_uuid'], native_requests=data['native_requests'], budgets=result)


def run(output=OUT, destination=None):
    output = Path(output)
    complete = output / 'complete.json'
    if not complete.exists():
        raise RuntimeError('Final complete.json is absent; do not scan partially completed cases')
    marker, audit = json.loads(complete.read_text()), json.loads((output / 'audit.json').read_text())
    if not marker.get('passed') or not audit.get('passed') or audit.get('observed_records') != 160:
        raise ValueError('The final 160-case audit must pass before this independent analysis')
    paths = {name: output / name for name in ('complete.json', 'audit.json', 'manifest.json', 'requests.csv', 'summary.csv', 'episodes.csv', 'throughput.json', 'cost_validation.json')}
    manifest = json.loads(paths['manifest.json'].read_text())
    requests, summaries, episodes = (csv_rows(paths[name]) for name in ('requests.csv', 'summary.csv', 'episodes.csv'))
    official = {r['condition']: r for r in summaries}
    table_path = Path(manifest['cost_table_path'])
    table = json.loads(table_path.read_text())
    costs = {int(row['layer']): row for row in table['layer_costs']}
    reports = []
    for cfg in manifest['conditions']:
        rows = [r for r in requests if r['condition'] == cfg['id']]
        erows = [r for r in episodes if r['condition'] == cfg['id']]
        if len(erows) != 40:
            raise ValueError('A finalized condition must retain all forty episodes')
        reports.append(condition_report(cfg, rows, erows, official[cfg['id']], costs, manifest['budgets'][cfg['budget']]))
    per_gpu = defaultdict(list)
    for row in episodes:
        per_gpu[row['gpu_uuid']].append(row)
    by_gpu = {gpu: dict(episodes=len(rows), episode_wall_s=stats([number(r, 'wall_s') for r in rows]),
        conditions={cfg['id']: sum(r['condition'] == cfg['id'] for r in rows) for cfg in manifest['conditions']}) for gpu, rows in per_gpu.items()}
    hardware = dict(gpu=table['hardware'], calls=table['provenance']['native_requests'],
        layer_distribution=table['layer_distribution'],
        per_layer=[{key: row[key] for key in ('layer', 'attn_saved_ms', 'attn_noise_ms', 'ffn_saved_ms', 'ffn_noise_ms', 'af_saved_ms', 'af_noise_ms', 'attn_eligible', 'ffn_eligible', 'af_eligible', 'profile')} for row in table['layer_costs']],
        uncertainty='Per-state median absolute deviation across measured repetitions/inputs. The sum of selected marginal MADs is an arithmetic diagnostic, not an independence assumption, confidence interval, or error bound.')
    source_hashes = {str(path.resolve()): sha(path) for path in list(paths.values()) + [table_path, Path(__file__)]}
    result = dict(status='complete', analysis='independent mixed-cost runtime and native hardware summary',
        manifest_hash=manifest['manifest_hash'], conditions=reports,
        throughput=json.loads(paths['throughput.json'].read_text()), per_gpu_episode_coverage=by_gpu,
        hardware=hardware, same_input_cost_validation=same_input_report(json.loads(paths['cost_validation.json'].read_text())),
        limitations=['Hot means cover only complete five-request cycles starting at the fourth Full/replan15, with coverage counts reported.','Full hot time remains a peak; five-request averages do not establish per-request deadline compliance.','Cold graph capture is subtracted in hot time but retained in all-request latency, total capture, and real evaluation throughput.','Condition trajectories differ; observed free-loop VLM, RPC and wall differences cannot by themselves identify a hardware-operator or objective speedup.','RPC time includes queue/transport and episode walls overlap across workers; summed episode walls are not the global evaluation duration.','Same-input cost-validation U/W masks may coincide; their timing differences are explicitly not algorithmic gains.','Hardware layer costs and joint AF savings are locally additive estimates; use actual validation and closed-loop runtime alongside them.'],
        source_hashes=source_hashes, no_GPU_or_model_calls=True, time=time.time())
    destination = Path(destination) if destination else output / 'runtime_hardware_analysis.json'
    atomic(destination, result)
    return result


def self_test():
    rows = []
    for index in range(27):
        rows.append(dict(suite='s', task='1', init='19', replan=str(index)))
    cycles, covered = complete_cycles(rows)
    assert len(cycles) == 2 and covered == 1 and [int(cycles[0][0]['replan']), int(cycles[-1][-1]['replan'])] == [15, 24]
    assert stats([1, 2, 3])['mean'] == 2 and stats([])['n'] == 0
    c = {1: dict(af_saved_ms=2., af_noise_ms=.2), 3: dict(attn_saved_ms=.5, attn_noise_ms=.1), 5: dict(ffn_saved_ms=1., ffn_noise_ms=.05)}
    row = dict(actual_attn='[1,3]', actual_ffn='[1,5]')
    saved, noise = mask_cost(row, c)
    assert saved == 3.5 and abs(noise - .35) < 1e-10
    count = module_counts([row])
    assert count['af_joint']['mean'] == 1 and count['support']['mean'] == 3 and count['individual_modules']['mean'] == 4
    print('CPU self-test passed: complete hot-cycle window, joint-AF costs/noise, actual module counts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--check-ready', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.check_ready:
        print(json.dumps(dict(ready=(args.output / 'complete.json').exists())))
    else:
        result = run(args.output, args.destination)
        print(json.dumps(dict(status=result['status'], conditions=[dict(condition=r['condition'], full_hot_ms=r['full_hot_ms']['mean'], fast_hot_ms=r['fast_hot_ms']['mean'], five_request_hot_ms=r['five_request_hot_per_request_ms']['mean']) for r in result['conditions']])))


if __name__ == '__main__':
    main()
