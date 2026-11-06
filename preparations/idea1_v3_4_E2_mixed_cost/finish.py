"""Audit mixed native routes and paired U/W quality at frozen RTX4090 costs."""
import json
import math
from pathlib import Path
import time

import numpy as np

from .common import BASE, ROOT, OUT, SUITES, ck, directory, read_manifest, sha, verify_frozen
from .selection import mask_record, objective_scores, select_mixed
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_2.common import seed as model_seed
from preparations.idea1_v3_4_E2_module_skip.evaluate import identity
from preparations.idea1_v3_4_E2_module_skip.finish import (
    clustered_delta, complete_cycles, condition_summary, hot_ms, key, mean,
    prefix_errors, write_csv,
)

REPORT = ROOT / 'docs/experiments/idea1/idea1_v3_4_E2_mixed_cost_report.md'


def audit_episode(cfg, path, result, manifest):
    requests = [json.loads(line) for line in (path / 'infer.jsonl').read_text().splitlines() if line]
    errors, rows, prefixes, history = [], [], {}, []
    mask_a = mask_f = None
    fulls = fasts = changes = 0
    budget = manifest['budgets'][cfg['budget']]

    def check(passed, reason, index=None):
        if not passed:
            errors.append(dict(replan=index, reason=reason))

    recorded_protocol = json.loads((path / 'identity.json').read_text())['protocol']
    measured_protocol = manifest['cost_table']['provenance']['protocol']
    for field in ('checkpoint_sha256', 'dtype', 'attention', 'backend'):
        if field in ('checkpoint_sha256', 'dtype') or field in measured_protocol:
            check(measured_protocol.get(field) is not None and recorded_protocol.get(field) == measured_protocol.get(field),
                  'identity protocol matches measured checkpoint/precision/native backend: ' + field)

    with np.load(path / 'arrays.npz', allow_pickle=False) as arrays:
        for index, request in enumerate(requests):
            route, timing = request['route'], request['timing']
            full = index % 5 == 0
            case = result['case']
            check(request.get('seed') == model_seed(case['suite'], case['task'], case['init'], index),
                  'frozen case/replan noise seed', index)
            a, f = route['selected_attn'], route['selected_ffn']
            support = sorted(set(a) | set(f))
            check(request['replan'] == request['round'] == index and request['step'] == index * 5,
                  'case/replan/env-step identity', index)
            check(a == sorted(set(a)) and f == sorted(set(f)) and all(1 <= layer <= 17 for layer in support) and
                  all(y - x > 1 for x, y in zip(support, support[1:])), 'A/F union support spacing', index)
            check(route['selected_skip'] == support and route['selected_af'] == sorted(set(a) & set(f)),
                  'support and joint-AF selection receipt', index)
            check(route['period'] == 5 and route['path'] == ('full' if full else 'fast') and route['D'] == (10 if full else 2),
                  'Full-D10/Fast-D2x4 schedule', index)
            check(route['steady_vjp_count'] == int(full), 'one Full VJP; Fast zero VJP', index)
            check(route['family'] == 'mixed' and route['cost_weighted'] == cfg['cost_weighted'] and
                  route['compensation'] == 'extrapolate' and route['adjacency_rule'] == 'spaced',
                  'frozen family/objective/compensation', index)
            check(route['current_observation_kv'] is True, 'current-observation KV', index)
            check(len(timing.get('nodes', [])) == (11 if full else 3), 'AE D10/D2 integration nodes', index)
            raw, actions = arrays[f'raw_{index}'], arrays[f'actions_{index}']
            check(raw.shape == (50, 32) and actions.shape == (50, 7) and np.isfinite(raw).all() and np.isfinite(actions).all(),
                  'finite original raw/actions shapes', index)
            if index <= 10:
                prefixes[index] = dict(raw=raw.copy(), actions=actions.copy())
            record = mask_record(a, f, budget)
            check(route.get('budget_hash') == record['budget_hash'] and
                  math.isclose(route.get('saved_ms', math.nan), record['saved_ms'], abs_tol=1e-8),
                  'frozen candidate membership and measured joint-AF estimated cost', index)
            if full:
                fulls += 1
                changes += int(mask_a is not None and (mask_a != a or mask_f != f))
                mask_a, mask_f = list(a), list(f)
                history = (history + [request['step']])[-2:]
                score_name = f'module_score_{index}' if f'module_score_{index}' in arrays else f'model_score_{index}'
                score, gradient = arrays[score_name], arrays[f'gate_gradient_{index}']
                check(score.shape == gradient.shape == (17, 2) and np.isfinite(score).all() and
                      np.isfinite(gradient).all() and np.array_equal(score, np.abs(gradient)), '34-gate one-VJP module scores', index)
                selected_a, selected_f = select_mixed(score, budget, cost_weighted=cfg['cost_weighted'])
                check(list(selected_a) == a and list(selected_f) == f, 'minimum U/W objective within shared frozen candidate set', index)
                if f'selection_score_{index}' in arrays:
                    check(np.array_equal(arrays[f'selection_score_{index}'], objective_scores(score, budget, cost_weighted=cfg['cost_weighted'])),
                          'unweighted or standalone-cost-normalized objective scores', index)
            else:
                fasts += 1
                check(a == mask_a and f == mask_f, 'hold mixed mask across four Fast requests', index)
                check(timing.get('selector_total_ms', 0) == 0, 'Fast has no scoring work', index)
            actual_a, actual_f = ([] if full else a), ([] if full else f)
            check(route['skip'] == ([] if full else support) and route['actual_skip_attn'] == actual_a and
                  route['actual_skip_ffn'] == actual_f and route['mask_refresh'] == full,
                  'actual Full/Fast skip and refresh semantics', index)
            check(route['patch_attn'] == actual_a and route['patch_ffn'] == actual_f and
                  route['patch_layers'] == ([] if full else support), 'separate A-before-F compensation', index)
            check(route['history_full_steps'] == history and route['history_bytes'] > 0, 'two real Full A/F banks', index)
            ratio = 0 if full or len(history) < 2 else (request['step'] - history[-1]) / (history[-1] - history[-2])
            check(math.isclose(route['extrapolation_ratio'], ratio, abs_tol=1e-8), 'actual-step hold/extrapolation ratio', index)
            if not full:
                check(route['has_second_history'] == (len(history) == 2), 'history availability', index)
            check(route['mask_age_env_steps'] == index % 5 * 5 and route['full_age_env_steps'] == index % 5 * 5,
                  'mask/Full age', index)
            check(route['scoring_mode'] == ('warm_original' if index < 10 else 'comp_error'), 'third Full first error score', index)
            if index >= 10:
                current_step = (index // 5) * 25
                check(route['score_source_full_steps'] == [current_step - 50, current_step - 25, current_step],
                      'error reads previous Full banks before commit', index)
            check(request['counts']['full'] == fulls and request['counts']['draft'] == fasts and
                  request['counts']['selector'] == fulls, 'isolated episode Full/Fast/VJP counters', index)
            native = route['native_module_receipt']
            check(native['current_observation_kv'] is True and native['skip_attn'] == actual_a and native['skip_ffn'] == actual_f,
                  'native current KV and real operator bypass', index)
            check(native['patched']['attn'] == actual_a and native['patched']['ffn'] == actual_f,
                  'native module residual patch receipt', index)
            for receipt in native['routes']:
                layer = receipt['layer']
                attention = layer < 18 and layer not in actual_a
                ffn = layer < 18 and layer not in actual_f
                projection = 'qkv' if attention or layer == 18 else 'kv'
                check(receipt['attn'] == attention and receipt['ffn'] == ffn and receipt['projection'] == projection,
                      f'native L{layer} A/F/projection route', index)
            if full:
                check(native['captured'] == dict(attn=list(range(1, 18)), ffn=list(range(1, 18))), 'native full capture all 34 residuals', index)
            rows.append(dict(condition=cfg['id'], **result['case'], budget=cfg['budget'], cost_weighted=cfg['cost_weighted'],
                replan=index, step=request['step'], path=route['path'], fast_position=0 if full else index % 5,
                actual_attn=json.dumps(actual_a), actual_ffn=json.dumps(actual_f),
                selected_attn=json.dumps(a), selected_ffn=json.dumps(f), selected_support=json.dumps(support),
                actual_support_count=0 if full else len(support), support_layers=0 if full else len(support),
                actual_module_count=len(actual_a) + len(actual_f), attn_count=len(a), ffn_count=len(f),
                af_count=len(set(a) & set(f)), a_only_count=len(set(a) - set(f)), f_only_count=len(set(f) - set(a)),
                estimated_saved_ms=0 if full else record['saved_ms'], selected_estimated_saved_ms=record['saved_ms'],
                policy_ms=timing['policy_wall_ms'], policy_hot_ms=hot_ms(request), rpc_ms=request['rpc_ms'],
                selector_ms=timing.get('selector_total_ms', 0), selector_capture_ms=timing.get('selector_capture_ms', 0),
                selector_hot_ms=timing.get('selector_hot_ms', 0), error_build_ms=timing.get('error_build_ms', 0),
                selection_ms=timing.get('mask_selection_ms', 0), vlm_ms=timing.get('vlm_ms'), ae_ms=timing.get('ae_ms'),
                scoring_mode=route['scoring_mode'], extrapolation_ratio=ratio, steady_vjp_count=route['steady_vjp_count']))
    check(len(requests) == result['predictions'] and sum(row['executed_length'] for row in requests) == result['actions'], 'episode request/action totals')
    return dict(result=result, requests=requests, identity=identity(path), prefix_arrays=prefixes,
                audit_errors=errors, mask_changes=changes), rows


def paired(manifest, data):
    pairs, excluded = [], []
    for budget in ('B4', 'B6'):
        baseline, candidate = f'{budget}_M-E-U+', f'{budget}_M-E-W+'
        aa, bb = data[baseline], data[candidate]
        shared = sorted(set(aa) & set(bb))
        exceptions = {case: prefix_errors(aa[case], bb[case]) for case in shared}
        exceptions = {case: reasons for case, reasons in exceptions.items() if reasons}
        excluded += [dict(budget=budget, case=list(case), reasons=reasons) for case, reasons in exceptions.items()]
        for scope in ('all40', 'strict'):
            selected = [case for case in shared if scope == 'all40' or case not in exceptions]
            records = [dict(suite=case[0], task=case[1], delta=int(bb[case]['result']['success']) - int(aa[case]['result']['success'])) for case in selected]
            point, lower, upper = clustered_delta(records)
            pairs.append(dict(budget=budget, baseline=baseline, candidate=candidate, scope=scope, n=len(selected), target=40,
                missing=40 - len(shared), excluded_prefix_cases=len(exceptions),
                baseline_success=sum(aa[case]['result']['success'] for case in selected),
                candidate_success=sum(bb[case]['result']['success'] for case in selected),
                recovered=sum(not aa[case]['result']['success'] and bb[case]['result']['success'] for case in selected),
                new_failures=sum(aa[case]['result']['success'] and not bb[case]['result']['success'] for case in selected),
                episode_delta_pp=mean([record['delta'] * 100 for record in records]), task_balanced_delta_pp=point,
                ci_low_pp=lower, ci_high_pp=upper, bootstrap_repeats=2000, bootstrap_seed=281034))
    return pairs, excluded


def number(value):
    return '—' if value is None else f'{value:.2f}'


def main():
    manifest = read_manifest()
    verify_frozen(manifest)
    errors, missing, data, summary, request_rows, episodes, control_exceptions = [], [], {}, [], [], [], []
    for cfg in manifest['conditions']:
        bank, rows = {}, []
        for case in manifest['small_cases']:
            path = directory(cfg['id'], case)
            if not (path / 'complete.json').exists():
                missing.append(dict(condition=cfg['id'], case=case))
                continue
            try:
                result = json.loads((path / 'result.json').read_text())
                if not result['valid'] or result['case'] != case or result['manifest_hash'] != manifest['manifest_hash'] or result['condition'] != cfg['id']:
                    raise ValueError('invalid fixed episode identity')
                if any(result['config'].get(field) != cfg[field] for field in cfg):
                    raise ValueError('scientific config differs from frozen condition')
                episode, records = audit_episode(cfg, path, result, manifest)
                bank[key(case)] = episode
                rows += records
                errors += [dict(condition=cfg['id'], case=case, **error) for error in episode['audit_errors']]
                expected = identity(manifest['controls'][ck(case)]['directory'])
                if any(episode['identity'][field] != expected[field] for field in expected):
                    control_exceptions.append(dict(condition=cfg['id'], case=case, reason='historical control initial identity/input'))
                episodes.append(dict(condition=cfg['id'], **case, success=result['success'], actions=result['actions'],
                    predictions=result['predictions'], wall_s=result['wall_s'], policy_wall_s=result['policy_wall_s'],
                    gpu_uuid=json.loads((path / 'identity.json').read_text())['protocol'].get('gpu_uuid'),
                    route_audit_passed=not episode['audit_errors']))
            except Exception as error:
                errors.append(dict(condition=cfg['id'], case=case, reason='episode_load_or_audit', error=repr(error)))
        data[cfg['id']] = bank
        request_rows += rows
        row = condition_summary(dict(cfg, reused=False), bank, [dict(record, timing_scope='own free closed-loop') for record in rows])
        selected = [record for record in rows if record['fast_position'] == 0]
        support_total = sum(len(json.loads(record['selected_support'])) for record in selected)
        row.update(mean_selected_attn=mean([record['attn_count'] for record in selected]),
                   mean_selected_ffn=mean([record['ffn_count'] for record in selected]),
                   mean_selected_af=mean([record['af_count'] for record in selected]),
                   a_only_support_fraction=None if not support_total else sum(record['a_only_count'] for record in selected) / support_total,
                   f_only_support_fraction=None if not support_total else sum(record['f_only_count'] for record in selected) / support_total,
                   af_support_fraction=None if not support_total else sum(record['af_count'] for record in selected) / support_total)
        summary.append(row)
    for case in manifest['small_cases']:
        uuids = {row['gpu_uuid'] for row in episodes if all(row[field] == case[field] for field in ('suite', 'task', 'init'))}
        if len(uuids) > 1 or None in uuids:
            errors.append(dict(reason='conditions for the same case must share a recorded physical GPU UUID', case=case, gpu_uuids=list(uuids)))
    pair_rows, exclusions = paired(manifest, data)
    gates, startups = [], []
    for work in sorted((OUT / 'workers').glob('gpu*')):
        if not (work / 'startup.json').exists():
            continue
        for name in ('compatibility_gate.json', 'selector_gate.json', 'route_gate.json', 'isolation_gate.json'):
            path = work / name
            passed = path.exists() and json.loads(path.read_text()).get('passed') is True
            gates.append(dict(worker=work.name, name=name, passed=passed))
            if not passed:
                errors.append(dict(reason='worker_gate', worker=work.name, name=name))
        startups += [dict(worker=work.name, **json.loads(path.read_text())) for path in work.glob('startup_attempt*.json')]
    hardware = OUT / 'hardware_probe/summary.json'
    if not hardware.exists() or json.loads(hardware.read_text()).get('passed') is not True:
        errors.append(dict(reason='incomplete_hardware_probe'))
    cost_checks = OUT / 'cost_validation.json'
    if not cost_checks.exists() or json.loads(cost_checks.read_text()).get('passed') is not True:
        errors.append(dict(reason='incomplete_same_input_cost_gate'))
    attempts = [dict(path=str(path), **json.loads(path.read_text())) for path in (OUT / 'technical_attempts').glob('*/*.json')]
    if any(row['attempts'] > 2 or row.get('fatal') for row in attempts):
        errors.append(dict(reason='technical_retry_budget_or_fatal'))
    complete = len(episodes) == 160 and not missing and all(row['n'] == 40 for row in summary)
    start = min((row['time'] - row['startup_s'] for row in startups), default=None)
    end_marker = OUT / 'evaluation_complete.json'
    end = json.loads(end_marker.read_text())['time'] if end_marker.exists() else None
    wall = end - start if start is not None and end is not None and end > start else None
    throughput = dict(new_episodes=len(episodes), evaluation_window_s=wall,
        episodes_per_hour=None if wall is None else len(episodes) * 3600 / wall,
        actions_per_s=None if wall is None else sum(row['actions'] for row in episodes) / wall,
        requests_per_s=None if wall is None else sum(row['predictions'] for row in episodes) / wall,
        model_preparation_s=[row['startup_s'] for row in startups],
        extra_technical_attempts=sum(max(row['attempts'] - 1, 0) for row in attempts))
    evidence = [dict(path=str(path), sha256=sha(path)) for path in (hardware, Path(manifest['cost_table_path']), cost_checks) if path.exists()]
    audit = dict(passed=complete and not errors, matrix_complete=complete, target_records=160, observed_records=len(episodes),
        reused_episodes=0, new_episodes=len(episodes), missing=missing, errors=errors, strict_prefix_exclusions=exclusions,
        historical_control_initial_exceptions=control_exceptions, gates=gates, hardware_and_same_input_evidence=evidence,
        all_valid_successes_and_failures_retained=True, throughput=throughput, technical_attempts=attempts, time=time.time())
    for filename, records in (('summary.csv', summary), ('requests.csv', request_rows), ('paired.csv', pair_rows), ('episodes.csv', episodes)):
        write_csv(OUT / filename, records)
    atomic(OUT / 'audit.json', audit)
    atomic(OUT / 'throughput.json', throughput)
    lines = ['# Attention/FFN混合选择：4090成本加权对照', '',
        f"状态：{'完整且审计通过' if audit['passed'] else '未完成或审计未通过'}。冻结160条新测，当前{len(episodes)}条；四个条件各40例，均为既有探索case，不是未见验证集。", '',
        'U最小化sum(s)，W最小化sum(s/c_m)；二者共享同一实测总成本窗口、相同合法候选集及union层间隔约束。时间动作投影权重均为[1,2,4,8,16]/31，U/W仅指成本归一化。AF预算取直接测得的joint cost，AF评分为局部模块分数相加近似。', '',
        '|条件|成功/有效例|Spatial|Object|Goal|Long|平均选A/F/AF|估计Fast省时ms|',
        '|---|---|---|---|---|---|---|---|']
    for row in summary:
        suites = [f"{row[s + '_success']}/{row[s + '_n']}" for s in SUITES]
        lines.append(f"|{row['condition']}|{row['success']}/{row['n']}|" + '|'.join(suites) +
            f"|{number(row['mean_selected_attn'])}/{number(row['mean_selected_ffn'])}/{number(row['mean_selected_af'])}|{number(row['mean_estimated_saved_ms'])}|")
    lines += ['', 'A、F计数包含同层AF，AF交集另列；A-only/F-only/AF支持占比及每次真实选择见CSV。Full实际跳过为0，selected字段表示之后Fast将使用的mask。', '',
        '|预算|范围|n|U→W成功|恢复/新增失败|task等权差pp|探索95%区间pp|', '|---|---|---|---|---|---|---|']
    for row in pair_rows:
        lines.append(f"|{row['budget']}|{row['scope']}|{row['n']}|{row['baseline_success']}→{row['candidate_success']}|{row['recovered']}/{row['new_failures']}|{number(row['task_balanced_delta_pp'])}|[{number(row['ci_low_pp'])},{number(row['ci_high_pp'])}]|")
    lines += ['', f'严格共同首Full raw/actions及首Fast观测异常{len(exclusions)}项；all40保留所有有效结果，严格比较只排除前缀不兼容项，不重跑到匹配。suite分层task聚类bootstrap2000次、seed281034，同task两init同簇，探索区间不作多重比较校正。', '',
        '|条件|完整热态周期/覆盖回合|Full ms|Fast1 ms|Fast2 ms|Fast3 ms|Fast4 ms|五轮均摊ms|冷捕获总秒|',
        '|---|---|---|---|---|---|---|---|---|']
    for row in summary:
        values = [row['full_hot_ms']] + [row[f'fast{p}_hot_ms'] for p in range(1, 5)] + [row['five_request_hot_mean_ms']]
        lines.append(f"|{row['condition']}|{row['complete_hot_cycles']}/{row['hot_covered_episodes']}|" + '|'.join(number(v) for v in values) + f"|{row['selector_capture_total_ms']/1000:.3f}|")
    lines += ['', '统一从第四Full/replan15起取完整五请求热态块，图构建与E图切换冷捕获及全请求policy/RPC和回合墙钟另列CSV。每个task组留在同一GPU、四条件按case索引轮换；服务B1串行、两个同task隔离环境，实际单环境回退写入gate。', '',
        '硬件完整Attention/保留KV/FFN/补偿的真实测量，以及同输入U/W/FFN整请求含补偿成本，分别查看：', '']
    for item in evidence:
        lines.append(f"- [{Path(item['path']).name}]({item['path']})，sha256 `{item['sha256']}`。")
    lines += ['', '局部成本相加的估计省时与原生整Fast请求wall分别判读；自由闭环轨迹分叉，时延差不能直接归因为单一模块。历史FFN-E的成本表/候选集与本次不同，不把旧自由闭环时延当作新同输入因果证据。', '',
        f"评测窗口{number(wall)}秒，{number(throughput['episodes_per_hour'])}回合/小时、{number(throughput['actions_per_s'])}动作/秒、{number(throughput['requests_per_s'])}请求/秒；模型准备秒数{throughput['model_preparation_s']}，额外技术尝试{throughput['extra_technical_attempts']}次。", '',
        f"审计异常{len(errors)}项，缺失{len(missing)}条；所有成功/有效失败保留。仅160条齐备且审计通过才生成complete.json。manifest hash：`{manifest['manifest_hash']}`。", '',
        '[冻结方案](idea1_v3_4_E2_mixed_cost_plan.md)']
    text = '\n'.join(lines) + '\n'
    (OUT / 'report.md').write_text(text)
    REPORT.write_text(text)
    if audit['passed']:
        atomic(OUT / 'complete.json', dict(passed=True, report=str(REPORT), manifest_hash=manifest['manifest_hash'], time=time.time()))
        atomic(OUT / 'status.json', dict(status='complete', observed_records=160, time=time.time()))
    return audit


if __name__ == '__main__':
    main()
