"""Audit the frozen forty-case matrix and report exploratory paired evidence."""
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .common import ROOT, OUT, PLAN, SUITES, BASE_MANIFEST_HASH, _read_frozen, ck, directory, sha
from .evaluate import identity
from .selection import mask_record, select_ffn
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.common import choose

REPORT = ROOT / 'docs/experiments/idea1/idea1_v3_4_E2_module_skip_report.md'
PAIRS = (('granularity', 'B-R+', 'F-R+'), ('compensation', 'F-R0', 'F-R+'),
         ('error_scoring', 'F-R+', 'F-E+'), ('controlled_granularity', 'B-E+', 'F-E+'),
         ('dynamic_vs_fixed', 'F-Fixed+', 'F-E+'))
BOOTSTRAP_SEED = 281034
BOOTSTRAP_REPEATS = 2000


def key(case):
    return tuple(case[name] for name in ('suite', 'task', 'init'))


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def write_csv(path, records):
    path = Path(path)
    names = list(dict.fromkeys(name for row in records for name in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def hot_ms(request):
    timing = request['timing']
    return float(timing.get('policy_hot_ms', timing['policy_wall_ms'] - timing.get('selector_capture_ms', 0)))


def complete_cycles(requests):
    """Identical replan15/fourth-Full window, complete five-request blocks."""
    return [requests[index:index + 5] for index in range(15, len(requests) - 4, 5)
            if [row['replan'] for row in requests[index:index + 5]] == list(range(index, index + 5))]


def estimated_savings(cfg, mask, manifest):
    if cfg['family'] == 'ffn':
        return mask_record(mask, manifest['budgets'][cfg['budget']])['saved_ms'], 'frozen measured FFN sum'
    costs = {row['layer']: row for row in manifest['cost_table']['layer_costs']}
    if all('af_saved_ms' in costs[layer] for layer in mask):
        return math.fsum(costs[layer]['af_saved_ms'] for layer in mask), 'measured AF sum; local cost approximation'
    return manifest['budgets'][cfg['budget']]['anchor_saved_ms'], 'budget anchor; per-mask AF cost unavailable'


def audit_episode(cfg, path, result, manifest, reused):
    requests = read_rows(path / 'infer.jsonl')
    errors, history, mask, selections, fulls, fasts, mask_changes = [], [], None, 0, 0, 0, 0
    prefix_arrays, request_rows = {}, []
    fixed = cfg['scoring'] == 'fixed'

    def check(passed, reason, replan=None):
        if not passed:
            errors.append(dict(replan=replan, reason=reason))

    with np.load(path / 'arrays.npz', allow_pickle=False) as arrays:
        for index, request in enumerate(requests):
            route, timing = request['route'], request['timing']
            full = index % 5 == 0
            chosen = route.get('selected_skip', [])
            legal = chosen == sorted(set(chosen)) and all(1 <= layer <= 17 for layer in chosen)
            legal = legal and all(b - a > 1 for a, b in zip(chosen, chosen[1:]))
            check(legal, 'illegal spaced module mask', index)
            check(request.get('replan') == request.get('round') == index and request['step'] == index * 5,
                  'request/replan/env-step identity', index)
            check(route['period'] == 5 and route['path'] == ('full' if full else 'fast') and
                  route['D'] == (10 if full else 2), 'Full/Fast-D schedule', index)
            check(route.get('steady_vjp_count') == int(full and not fixed), 'Full-only one VJP / Fast zero VJP', index)
            check(route.get('adjacency_rule') == 'spaced' and route.get('compensation') == cfg['compensation'],
                  'frozen rule or compensation', index)
            if not reused:
                check(route.get('current_observation_kv') is True, 'current-observation KV receipt', index)
                check(route.get('family') == cfg['family'] and route.get('budget') == cfg['budget'],
                      'module family/budget identity', index)
            check(len(timing.get('nodes', [])) == (11 if full else 3), 'AE D10/D2 integration nodes', index)
            raw, actions = arrays[f'raw_{index}'], arrays[f'actions_{index}']
            check(raw.shape == (50, 32) and actions.shape == (50, 7) and
                  np.isfinite(raw).all() and np.isfinite(actions).all(), 'finite raw/actions with original shapes', index)
            if index <= 10:
                prefix_arrays[index] = dict(raw=raw.copy(), actions=actions.copy())
            refresh = full and (not fixed or index == 0)
            check(route.get('mask_refresh') == refresh, 'mask refresh only at declared Full', index)
            if refresh:
                mask_changes += int(mask is not None and mask != chosen)
                mask = list(chosen)
            check(chosen == mask, 'mask held across four Fast requests', index)
            check(route.get('mask_age_env_steps') == (request['step'] if fixed else index % 5 * 5), 'mask age', index)
            check(route.get('full_age_env_steps') == index % 5 * 5 and
                  route.get('full_age_rounds') == index % 5, 'Full history age', index)
            if cfg['family'] == 'block':
                check(len(chosen) == cfg['block_skip_count'], 'block anchor skip count', index)
            else:
                try:
                    record = mask_record(chosen, manifest['budgets'][cfg['budget']])
                    if not reused:
                        check(route.get('budget_hash') == record['budget_hash'] and
                              math.isclose(route.get('saved_ms', math.nan), record['saved_ms'], abs_tol=1e-8),
                              'frozen cost-window receipt', index)
                except ValueError:
                    check(False, 'FFN mask outside frozen same-budget candidate set', index)
            if full:
                fulls += 1
                history = (history + [request['step']])[-2:]
                check(route.get('skip') == [], 'Full path never skips', index)
                if not fixed:
                    selections += 1
                    score_name = f"{'layer_score' if reused else 'module_score'}_{index}"
                    score, gradient = arrays[score_name], arrays[f'gate_gradient_{index}']
                    check(score.shape == gradient.shape == (17,) and np.isfinite(score).all() and
                          np.isfinite(gradient).all() and np.array_equal(score, np.abs(gradient)),
                          'one-VJP sensitivity score', index)
                    optimal = choose(score, cfg['block_skip_count'], 'spaced') if cfg['family'] == 'block' else select_ffn(score, manifest['budgets'][cfg['budget']])
                    check(list(optimal) == chosen, 'minimum sensitivity over frozen legal candidates', index)
                else:
                    check(chosen == manifest['fixed_episode_masks'][cfg['id']][ck(result['case'])],
                          'frozen episode-identity random mask', index)
            else:
                fasts += 1
                check(route.get('skip') == mask, 'native Fast skip receipt', index)
                check(timing.get('selector_total_ms', 0) == 0, 'Fast has no selector/VJP work', index)
            check(route.get('history_full_steps') == history and route.get('history_bytes', 0) > 0,
                  'two real Full histories, read before write', index)
            if not full:
                ratio = 0 if len(history) < 2 else (request['step'] - history[-1]) / (history[-1] - history[-2])
                check(math.isclose(route.get('extrapolation_ratio', math.nan), ratio, abs_tol=1e-8),
                      'actual-step hold/extrapolation ratio', index)
                check(route.get('has_second_history') == (len(history) == 2), 'second Full history availability', index)
            check(route.get('patch_layers') == ([] if full or cfg['compensation'] == 'none' else mask),
                  'compensation on selected modules only', index)
            if cfg['scoring'] == 'error':
                expected_mode = 'warm_original' if index < 10 else 'comp_error'
                check(route.get('scoring_mode') == expected_mode, 'third Full first E score / step55 first E Fast', index)
                if index >= 10:
                    step = (index // 5) * 25
                    check(route.get('score_source_full_steps') == [step - 50, step - 25, step],
                          'E score uses previous two real Fulls before current commit', index)
            check(request['counts']['full'] == fulls and request['counts']['draft'] == fasts and
                  request['counts']['selector'] == selections, 'per-episode Full/Fast/VJP counters', index)
            saving, basis = estimated_savings(cfg, chosen, manifest)
            request_rows.append(dict(condition=cfg['id'], **result['case'], replan=index,
                step=request['step'], path=route['path'], fast_position=0 if full else index % 5,
                reused=reused, timing_scope='historical baseline background' if reused else 'own free closed-loop',
                hot_complete_cycle_eligible=index >= 15 and (index // 5) * 5 + 4 < len(requests),
                skip_layers=json.dumps(route['skip']), selected_skip_layers=json.dumps(chosen),
                support_layers=0 if full else len(chosen), selected_support_layers=len(chosen),
                skipped_modules=0 if full else len(chosen) * (2 if cfg['family'] == 'block' else 1),
                selected_modules=len(chosen) * (2 if cfg['family'] == 'block' else 1),
                estimated_saved_ms=0 if full else saving, selected_estimated_saved_ms=saving,
                saving_basis=basis, policy_ms=timing['policy_wall_ms'], policy_hot_ms=hot_ms(request),
                selector_ms=timing.get('selector_total_ms', 0), selector_hot_ms=timing.get('selector_hot_ms', 0),
                selector_capture_ms=timing.get('selector_capture_ms', 0), error_build_ms=timing.get('error_build_ms', 0),
                selection_ms=timing.get('mask_selection_ms', 0), rpc_ms=request['rpc_ms'],
                vlm_ms=timing.get('vlm_ms'), ae_ms=timing.get('ae_ms'),
                scoring_mode=route.get('scoring_mode', 'original'),
                history_steps=json.dumps(route.get('history_full_steps', [])), extrapolation_ratio=route.get('extrapolation_ratio'),
                steady_vjp_count=route.get('steady_vjp_count'), current_observation_kv=route.get('current_observation_kv', 'historical path')))
    check(len(requests) == result['predictions'] and sum(row['executed_length'] for row in requests) == result['actions'],
          'episode request/action totals')
    check(bool(requests), 'nonempty completed episode')
    return dict(result=result, identity=identity(path), requests=requests, prefix_arrays=prefix_arrays,
                audit_errors=errors, mask_changes=mask_changes), request_rows


def prefix_errors(baseline, candidate, warm_error_pair=False):
    reasons = []
    for field in ('init_vector_sha256', 'bddl_sha256', 'controller', 'prompt', 'env_seed', 'settling', 'first_input_hash'):
        if baseline['identity'][field] != candidate['identity'][field]:
            reasons.append(field)
    for field in ('raw', 'actions'):
        if not np.allclose(baseline['prefix_arrays'][0][field], candidate['prefix_arrays'][0][field], atol=1e-5, rtol=1e-5):
            reasons.append('first_common_Full_' + field)
    aa, bb = baseline['requests'], candidate['requests']
    if len(aa) < 2 or len(bb) < 2 or aa[1]['input_hash'] != bb[1]['input_hash']:
        reasons.append('first_Fast_observation')
    if warm_error_pair:
        for index in range(min(11, len(aa), len(bb))):
            if aa[index]['input_hash'] != bb[index]['input_hash']:
                reasons.append(f'warm_input_{index}')
            for field in ('raw', 'actions'):
                if not np.allclose(baseline['prefix_arrays'][index][field], candidate['prefix_arrays'][index][field], atol=1e-5, rtol=1e-5):
                    reasons.append(f'warm_{field}_{index}')
            if index < 10 and aa[index]['route']['selected_skip'] != bb[index]['route']['selected_skip']:
                reasons.append(f'warm_mask_{index}')
        if len(bb) > 10:
            route = bb[10]['route']
            if route.get('scoring_mode') != 'comp_error' or route.get('score_source_full_steps') != [0, 25, 50]:
                reasons.append('first_E_at_replan10_step50')
    if baseline['audit_errors'] or candidate['audit_errors']:
        reasons.append('route_or_history_audit')
    if baseline.get('warm_pair_errors') or candidate.get('warm_pair_errors'):
        reasons.append('R/E_warm_reference')
    return sorted(set(reasons))


def clustered_delta(records):
    """Suite-stratified resampling of tasks, keeping both inits in a cluster."""
    if not records:
        return None, None, None
    clusters = []
    for suite in SUITES:
        tasks = sorted({row['task'] for row in records if row['suite'] == suite})
        if tasks:
            clusters.append(np.asarray([np.mean([row['delta'] for row in records if row['suite'] == suite and row['task'] == task]) for task in tasks]))
    point = float(np.mean([values.mean() for values in clusters]))
    random = np.random.default_rng(BOOTSTRAP_SEED)
    samples = np.zeros(BOOTSTRAP_REPEATS)
    for values in clusters:
        samples += values[random.integers(0, len(values), size=(BOOTSTRAP_REPEATS, len(values)))].mean(axis=1) / len(clusters)
    lower, upper = np.quantile(samples, [.025, .975])
    return point * 100, float(lower * 100), float(upper * 100)


def paired_rows(manifest, data):
    output, exclusions = [], []
    for budget in ('B4', 'B6'):
        for comparison, base, cand in PAIRS:
            a, b = f'{budget}_{base}', f'{budget}_{cand}'
            aa, bb = data[a], data[b]
            shared = sorted(set(aa) & set(bb))
            omitted = {case: prefix_errors(aa[case], bb[case], comparison == 'error_scoring') for case in shared}
            omitted = {case: reasons for case, reasons in omitted.items() if reasons}
            exclusions += [dict(baseline=a, candidate=b, case=list(case), reasons=reasons) for case, reasons in omitted.items()]
            for scope in ('all40', 'strict'):
                selected = [case for case in shared if scope == 'all40' or case not in omitted]
                records = [dict(suite=case[0], task=case[1], delta=int(bb[case]['result']['success']) - int(aa[case]['result']['success'])) for case in selected]
                point, lower, upper = clustered_delta(records)
                output.append(dict(budget=budget, comparison=comparison, baseline=a, candidate=b, scope=scope,
                    n=len(selected), target_n=40, missing_pairs=40 - len(shared), excluded_prefix_cases=len(omitted),
                    baseline_success=sum(aa[case]['result']['success'] for case in selected),
                    candidate_success=sum(bb[case]['result']['success'] for case in selected),
                    recovered=sum(not aa[case]['result']['success'] and bb[case]['result']['success'] for case in selected),
                    new_failures=sum(aa[case]['result']['success'] and not bb[case]['result']['success'] for case in selected),
                    task_balanced_delta_pp=point, episode_delta_pp=mean([row['delta'] * 100 for row in records]),
                    ci_low_pp=lower, ci_high_pp=upper, bootstrap_repeats=BOOTSTRAP_REPEATS, bootstrap_seed=BOOTSTRAP_SEED,
                    suites_present=len({case[0] for case in selected}), tasks_present=len({case[:2] for case in selected})))
    return output, exclusions


def condition_summary(cfg, bank, request_rows):
    episodes = list(bank.values())
    requests = [request for episode in episodes for request in episode['requests']]
    cycles = [cycle for episode in episodes for cycle in complete_cycles(episode['requests'])]
    steady = [request for cycle in cycles for request in cycle]
    result = dict(condition=cfg['id'], budget=cfg['budget'], family=cfg['family'], scoring=cfg['scoring'],
                  compensation=cfg['compensation'], n=len(episodes), target=40,
                  success=sum(episode['result']['success'] for episode in episodes),
                  success_percent=None if not episodes else sum(episode['result']['success'] for episode in episodes) * 100 / len(episodes),
                  reused=bool(cfg['reused']), timing_scope='historical baseline background' if cfg['reused'] else 'own free closed-loop',
                  hot_window_start_replan=15, complete_hot_cycles=len(cycles),
                  hot_covered_episodes=sum(bool(complete_cycles(episode['requests'])) for episode in episodes),
                  five_request_hot_mean_ms=mean([hot_ms(request) for request in steady]),
                  five_request_wall_mean_ms=mean([request['timing']['policy_wall_ms'] for request in steady]),
                  full_hot_ms=mean([hot_ms(request) for request in steady if request['replan'] % 5 == 0]),
                  all_request_policy_ms=mean([request['timing']['policy_wall_ms'] for request in requests]),
                  all_request_rpc_ms=mean([request['rpc_ms'] for request in requests]),
                  selector_capture_total_ms=sum(request['timing'].get('selector_capture_ms', 0) for request in requests),
                  error_switch_capture_total_ms=sum(request['timing'].get('selector_capture_ms', 0) for request in requests if request['replan'] == 10 and cfg['scoring'] == 'error'),
                  mask_changes=sum(episode['mask_changes'] for episode in episodes),
                  mean_skip_layers=mean([row['support_layers'] for row in request_rows if row['fast_position']]),
                  skip_layers_min=min((row['support_layers'] for row in request_rows if row['fast_position']), default=None),
                  skip_layers_max=max((row['support_layers'] for row in request_rows if row['fast_position']), default=None),
                  mean_estimated_saved_ms=mean([row['estimated_saved_ms'] for row in request_rows if row['fast_position']]),
                  episode_wall_s=sum(episode['result']['wall_s'] for episode in episodes),
                  mean_episode_wall_s=mean([episode['result']['wall_s'] for episode in episodes]),
                  actions=sum(episode['result']['actions'] for episode in episodes), predictions=len(requests))
    result['actions_per_episode_wall_s'] = result['actions'] / result['episode_wall_s'] if result['episode_wall_s'] else None
    result['episodes_per_episode_wall_hour'] = len(episodes) * 3600 / result['episode_wall_s'] if result['episode_wall_s'] else None
    for position in range(1, 5):
        result[f'fast{position}_hot_ms'] = mean([hot_ms(request) for request in steady if request['replan'] % 5 == position])
    for suite in SUITES:
        subset = [episode for case, episode in bank.items() if case[0] == suite]
        result[f'{suite}_n'] = len(subset)
        result[f'{suite}_success'] = sum(episode['result']['success'] for episode in subset)
    return result


def _number(value, digits=2):
    return '—' if value is None else f'{value:.{digits}f}'


def main():
    manifest = _read_frozen(OUT / 'manifest.json')
    errors, missing, data, summaries, requests, episode_rows = [], [], {}, [], [], []
    for filename, expected in manifest['source_files'].items():
        if sha(ROOT / filename) != expected:
            errors.append(dict(reason='source_hash', path=filename))
    for filename, expected in manifest['references'].items():
        if sha(ROOT / filename) != expected:
            errors.append(dict(reason='reference_hash', path=filename))
    if sha(PLAN) != manifest['plan_sha256']:
        errors.append(dict(reason='plan_hash'))
    if sha(manifest['cost_table_path']) != manifest['cost_table_file_sha256']:
        errors.append(dict(reason='measured_cost_table_hash'))
    for cfg in manifest['conditions']:
        bank, condition_requests = {}, []
        for case in manifest['small_cases']:
            ref = manifest['baseline'].get(cfg['id'], {}).get(ck(case))
            path = Path(ref['directory']) if ref else directory(cfg['id'], case)
            if not (path / 'complete.json').exists() or not (path / 'result.json').exists():
                missing.append(dict(condition=cfg['id'], case=case, directory=str(path)))
                continue
            try:
                result = json.loads((path / 'result.json').read_text())
                if not result.get('valid') or result.get('case') != case:
                    raise ValueError('invalid fixed case identity')
                if ref:
                    if result['manifest_hash'] != BASE_MANIFEST_HASH or result['condition'] != ref['condition']:
                        raise ValueError('baseline manifest identity')
                    for filename, expected in ref['hashes'].items():
                        if sha(path / filename) != expected:
                            raise ValueError(f'baseline reference hash: {filename}')
                elif result['manifest_hash'] != manifest['manifest_hash'] or result['condition'] != cfg['id']:
                    raise ValueError('new record frozen manifest/condition identity')
                if not ref and any(result['config'].get(field) != cfg[field] for field in cfg):
                    raise ValueError('new record frozen scientific configuration')
                episode, rows = audit_episode(cfg, path, result, manifest, bool(ref))
                bank[key(case)] = episode
                condition_requests += rows
                errors += [dict(condition=cfg['id'], case=case, **error) for error in episode['audit_errors']]
                episode_rows.append(dict(condition=cfg['id'], **case, reused=bool(ref), success=result['success'],
                    actions=result['actions'], predictions=result['predictions'], wall_s=result['wall_s'],
                    cold_capture_ms=sum(row['timing'].get('selector_capture_ms', 0) for row in episode['requests']),
                    route_audit_passed=not episode['audit_errors']))
            except Exception as error:
                errors.append(dict(condition=cfg['id'], case=case, reason='episode_load_or_audit', error=repr(error)))
        data[cfg['id']] = bank
        requests += condition_requests
        summaries.append(condition_summary(cfg, bank, condition_requests))
    warm_exclusions = []
    for budget in ('B4', 'B6'):
        for family in ('B', 'F'):
            raw_name, error_name = f'{budget}_{family}-R+', f'{budget}_{family}-E+'
            for case in sorted(set(data[raw_name]) & set(data[error_name])):
                reasons = prefix_errors(data[raw_name][case], data[error_name][case], warm_error_pair=True)
                if reasons:
                    data[error_name][case]['warm_pair_errors'] = reasons
                    warm_exclusions.append(dict(baseline=raw_name, candidate=error_name, case=list(case), reasons=reasons))
    pairs, prefix_exclusions = paired_rows(manifest, data)
    gates, startups = [], []
    for work in sorted((OUT / 'workers').glob('gpu*')):
        if not (work / 'startup.json').exists():
            continue
        for name in ('compatibility_gate.json', 'selector_gate.json', 'route_gate.json', 'isolation_gate.json'):
            path = work / name
            row = json.loads(path.read_text()) if path.exists() else dict(passed=False, reason='missing')
            gates.append(dict(worker=work.name, gate=name, passed=row.get('passed') is True))
            if row.get('passed') is not True:
                errors.append(dict(reason='worker_preflight_gate', worker=work.name, gate=name))
        for path in work.glob('startup_attempt*.json'):
            startups.append(dict(worker=work.name, **json.loads(path.read_text())))
    for name in ('pilot_complete.json', 'pilot_mechanism_complete.json'):
        path = OUT / name
        if not path.exists() or json.loads(path.read_text()).get('passed') is not True:
            errors.append(dict(reason='stage_barrier', path=name))
    attempts = [dict(path=str(path), **json.loads(path.read_text())) for path in (OUT / 'technical_attempts').glob('*/*.json')]
    if any(row['attempts'] > 2 or row.get('fatal') for row in attempts):
        errors.append(dict(reason='technical_attempt_budget_or_fatal_failure'))
    technical_errors = list(OUT.glob('*/episodes/*/task*/init*/technical_error.json'))
    technical_errors += list(OUT.glob('*/episodes/*/task*/init*.invalid.*/technical_error.json'))
    evidence = sorted(set(OUT.glob('cost_validation*.json')) | set(OUT.glob('workers/gpu*/cost_validation*.json')))
    offline = OUT / 'pilot_mechanism' / 'summary.json'
    evidence += [offline] if offline.exists() else []
    evidence_rows = [dict(path=str(path.relative_to(OUT)), sha256=sha(path), content=json.loads(path.read_text())) for path in evidence]
    actual_cost_rows = []
    for item in evidence_rows:
        content = item['content']
        if item['path'].endswith('cost_validation.json'):
            if content.get('passed') is not True:
                errors.append(dict(reason='same_input_cost_numerical_gate', path=item['path']))
            actual_cost_rows += content.get('paired_costs', [])
    if not (OUT / 'cost_validation.json').exists():
        errors.append(dict(reason='missing_same_input_cost_validation'))
    if not offline.exists() or json.loads(offline.read_text()).get('status') != 'complete':
        errors.append(dict(reason='missing_or_incomplete_offline_mechanism'))
    new_episodes = [row for row in episode_rows if not row['reused']]
    reused_episodes = [row for row in episode_rows if row['reused']]
    completed = len(episode_rows) == 480 and len(new_episodes) == 400 and len(reused_episodes) == 80 and not missing
    start = min((row['time'] - row['startup_s'] for row in startups), default=None)
    evaluation_marker = OUT / 'evaluation_complete.json'
    end = json.loads(evaluation_marker.read_text())['time'] if evaluation_marker.exists() else None
    wall = end - start if start is not None and end is not None and end > start else None
    throughput = dict(new_episodes=len(new_episodes), reused_episodes=len(reused_episodes),
        evaluation_window_s=wall, episodes_per_hour=None if wall is None else len(new_episodes) * 3600 / wall,
        actions_per_s=None if wall is None else sum(row['actions'] for row in new_episodes) / wall,
        requests_per_s=None if wall is None else sum(row['predictions'] for row in new_episodes) / wall,
        model_preparation_s=[row['startup_s'] for row in startups],
        extra_technical_attempts=sum(max(row['attempts'] - 1, 0) for row in attempts),
        technical_error_records=len(set(technical_errors)), technical_attempts=attempts)
    audit = dict(passed=completed and not errors, matrix_complete=completed, target_records=480,
        observed_records=len(episode_rows), reused_episodes=len(reused_episodes), new_episodes=len(new_episodes),
        missing=missing, errors=errors, strict_prefix_exclusions=prefix_exclusions,
        raw_error_warm_prefix_exclusions=warm_exclusions, worker_gates=gates,
        route_checks='Full-only VJP; Fast zero VJP; actual-step history; current KV receipt; frozen masks and minimum scores',
        all_valid_successes_and_failures_retained=True, same_input_cost_evidence=evidence_rows,
        throughput=throughput, exploration='40 reused exploratory cases, not an unseen validation set', time=time.time())
    for name, rows in (('summary.csv', summaries), ('requests.csv', requests), ('paired.csv', pairs), ('episodes.csv', episode_rows)):
        write_csv(OUT / name, rows)
    atomic(OUT / 'audit.json', audit)
    atomic(OUT / 'throughput.json', throughput)
    status = '完整' if audit['passed'] else '未完成或审计未通过'
    lines = ['# VLM模块级跳过：40例探索结果', '',
        f'状态：**{status}**。本轮是此前使用过的40个探索case，并非未见验证集；不扩至200例，不包含可选混合阶段。', '',
        f"冻结目标480条：80条B-R+历史有效结果复用、400条新测；当前读取{len(episode_rows)}条（复用{len(reused_episodes)}、新测{len(new_episodes)}），缺失{len(missing)}条，审计异常{len(errors)}项。成功与有效失败全部保留。", '',
        'Full-D10→Fast-D2×4，H50/R5，L1–L17间隔约束；FFN成本窗口、B4/B6锚点及固定随机根341101在运行前冻结。', '',
        '|条件|成功/有效例|Spatial|Object|Goal|Long|实际FFN/层支持均值|估计省时ms|',
        '|---|---|---|---|---|---|---|---|']
    for row in summaries:
        suite_values = [f"{row[s + '_success']}/{row[s + '_n']}" for s in SUITES]
        lines.append(f"|{row['condition']}|{row['success']}/{row['n']}|" + '|'.join(suite_values) +
                     f"|{_number(row['mean_skip_layers'])}|{_number(row['mean_estimated_saved_ms'])}|")
    lines += ['', 'FFN支持层即跳过FFN数；整层条件每个支持层同时跳A/F。估计省时来自冻结的实测局部成本，可加和仍是近似；逐请求真实mask与成本见requests.csv。', '',
        '|条件|完整热态周期/覆盖回合|Full ms|Fast1 ms|Fast2 ms|Fast3 ms|Fast4 ms|五轮均摊ms|冷捕获总秒|E图切换捕获秒|',
        '|---|---|---|---|---|---|---|---|---|---|']
    for row in summaries:
        values = [row['full_hot_ms']] + [row[f'fast{position}_hot_ms'] for position in range(1, 5)] + [row['five_request_hot_mean_ms']]
        lines.append(f"|{row['condition']}|{row['complete_hot_cycles']}/{row['hot_covered_episodes']}|" + '|'.join(_number(value) for value in values) +
                     f"|{row['selector_capture_total_ms']/1000:.3f}|{row['error_switch_capture_total_ms']/1000:.3f}|")
    lines += ['', '全部条件统一从第四Full/replan15起，仅取完整五请求块；热态policy_hot排除另列的selector_capture。所有请求含冷启动的policy/RPC均值、回合墙钟和动作吞吐见summary.csv。Full峰值与五轮均值分别列出。', '',
        '**B-R+的自由闭环时延为历史背景。不同自由轨迹的时延不能直接证明模块的同输入省时或同预算优势。** 主要同输入成本证据与离线动作扰动单列如下：', '']
    if evidence_rows:
        for row in evidence_rows:
            lines.append(f"- [{row['path']}]({OUT / row['path']})（sha256 `{row['sha256']}`）。")
    else:
        lines.append('目前没有读取到同输入cost_validation或离线summary；不据自由闭环均值声称同预算优势。')
    if actual_cost_rows:
        lines += ['', '|同输入预算|估计FFN总省时ms|实测整层Fast wall ms|实测FFN Fast wall ms|配对wall差ms|冻结tau ms|实际满足预算|',
                  '|---|---|---|---|---|---|---|']
        for row in actual_cost_rows:
            lines.append(f"|{row['budget']}|{_number(row['frozen_ffn_mask_record']['saved_ms'])}|{_number(row['block_fast_request_ms'])}|{_number(row['ffn_fast_request_ms'])}|{_number(row['paired_request_ffn_minus_block_ms'])}|{_number(row['tau_ms'])}|{row['matched_actual_cost']}|")
        lines += ['', '实测wall包含Fast补偿操作，为预声明同一真实观测、同噪声的四个热态AB/BA配对；估计总省时与实测整请求wall分别列出。此检查的输入和mask覆盖有限，不推广为所有闭环mask均同预算；超过tau的候选不称同预算收益。']
    lines += ['', '|预算/比较|范围|n|基准→候选成功|恢复/新增失败|task等权差pp|探索95%区间pp|',
        '|---|---|---|---|---|---|---|']
    for row in pairs:
        lines.append(f"|{row['budget']}/{row['comparison']}|{row['scope']}|{row['n']}|{row['baseline_success']}→{row['candidate_success']}|{row['recovered']}/{row['new_failures']}|{_number(row['task_balanced_delta_pp'])}|[{_number(row['ci_low_pp'])}, {_number(row['ci_high_pp'])}]|")
    lines += ['', '五组方向依次为B-R+→F-R+（粒度）、F-R0→F-R+（补偿）、F-R+→F-E+（误差评分）、B-E+→F-E+（控制评分后的粒度）、F-Fixed+→F-E+（动态选择）。', '',
        f'严格配对按首输入、共同首Full raw/actions（atol=rtol=1e-5）、首Fast观测与官方身份排除异常；R/E同结构另查replan0..10共同暖前缀、第三Full的首次E点。发现{len(prefix_exclusions)}个主要比较-case异常、{len(warm_exclusions)}个同结构R/E暖前缀异常，列入audit.json；all40保留全部有效描述结果，不重跑到匹配。短回合未到E点如实保留，缺少首Fast时不能进入严格比较。', '',
        '区间按suite分层、task聚类bootstrap2000次，seed281034；同一task两个init作为同一簇，不当成独立任务。严格子集按仍可用task等权，列出实际n，未校正多重比较。固定随机仅一个预设根，单次正权VJP有抵消、多模块有非线性交互，PyTorch回放为原生BF16近似，周期5残差变化仍有限制。', '',
        f"新测实际窗口{_number(wall)}秒，{_number(throughput['episodes_per_hour'])}回合/小时，{_number(throughput['actions_per_s'])}动作/秒，{_number(throughput['requests_per_s'])}请求/秒；模型加载/核验秒数{throughput['model_preparation_s']}。额外技术尝试{throughput['extra_technical_attempts']}次，技术错误记录{throughput['technical_error_records']}份。窗口含模型准备、pilot机制检查与调度，分母不混入80条历史复用回合。", '',
        f"缺失与路由/源码/历史审计详情见audit.json；只有480条齐备且审计通过才生成complete.json。冻结manifest hash：`{manifest['manifest_hash']}`。", '',
        '[冻结方案](idea1_v3_4_E2_module_skip_plan.md)']
    text = '\n'.join(lines) + '\n'
    (OUT / 'report.md').write_text(text)
    REPORT.write_text(text)
    if audit['passed']:
        atomic(OUT / 'complete.json', dict(passed=True, report=str(REPORT), manifest_hash=manifest['manifest_hash'], time=time.time()))
        atomic(OUT / 'status.json', dict(status='complete', observed_records=480, time=time.time()))
    return audit


if __name__ == '__main__':
    main()
