"""Apply the frozen prefix-exclusion rule before comparing first gradients.

The original analyzer compared gradients even when first inputs differed.
This review preserves its source, all tolerances, and every rollout result.
"""
import csv
import json
import time
from pathlib import Path

import numpy as np

from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v3_4_E2_cycle5_q4_v12.common import (
    ROOT, OUT, BASE, PLAN, CONDITIONS, SUITES, sha, ck, directory,
)
from preparations.idea1_v3_4_E2_cycle5_q4_v12.finish import audit_episode, key
from preparations.idea1_v3_4_E2_selection_200.finish import header, paired


def input_difference(a, b):
    """Describe stored first-input differences without changing either input."""
    differences = []
    with np.load(a / 'input_0000.npz') as x, np.load(b / 'input_0000.npz') as y:
        assert set(x.files) == set(y.files)
        for name in x.files:
            xx, yy = x[name], y[name]
            if np.array_equal(xx, yy):
                continue
            row = dict(field=name, new_shape=list(xx.shape), reference_shape=list(yy.shape))
            if xx.shape == yy.shape:
                row['changed_elements'] = int(np.count_nonzero(xx != yy))
                if np.issubdtype(xx.dtype, np.number) and np.issubdtype(yy.dtype, np.number):
                    row['max_abs'] = float(np.max(np.abs(xx.astype(float) - yy.astype(float))))
            differences.append(row)
    return differences


def main():
    m = json.loads((OUT / 'manifest.json').read_text())
    assert sha(PLAN) == m['plan_sha256']
    assert all(sha(ROOT / p) == v for p, v in m['source_files'].items())
    assert all(sha(ROOT / p) == v for p, v in m['references'].items())
    assert json.loads((OUT / 'cpu_gate.json').read_text())['passed']
    review_sha = sha(__file__)
    atomic(OUT / 'review_manifest.json', dict(
        source=str(Path(__file__).relative_to(ROOT)), source_sha256=review_sha,
        original_manifest_sha256=sha(OUT / 'manifest.json'),
        original_failure_sha256=sha(OUT / 'failed.json'),
        rule='Compare first positive gradients only for identical first inputs; '
             'retain all valid outcomes and apply the frozen strict-prefix exclusion.',
        gradient_atol=1e-4, gradient_rtol=1e-3, rollout_reruns=0, time=time.time(),
    ))
    keys = {key(c) for c in m['cases']}
    data, quality, episodes, requests, costs, gradient_checks = {}, [], [], [], [], []
    anomalies = []
    baseline_costs = {r['condition']: r for r in csv.DictReader((BASE / 'costs.csv').open())}
    for name, bank in m['baseline'].items():
        dd = {}
        for c in m['cases']:
            ref = bank[ck(c)]
            d = Path(ref['directory'])
            assert all(sha(d / f) == v for f, v in ref['hashes'].items())
            h = header(d)
            with np.load(d / 'arrays.npz') as z:
                h['gradient0'] = z['gate_gradient_0'].copy()
            dd[key(c)] = h
        data['q1_' + name] = dd
        quality.append(dict(
            condition='q1_' + name, projections=1,
            compensation='none' if name.endswith('_none') else 'extrapolate',
            n=200, success=sum(h['result']['success'] for h in dd.values()), reused=True,
            **{s: sum(h['result']['success'] for k, h in dd.items() if k[0] == s) for s in SUITES},
        ))
    for cfg in CONDITIONS:
        dd, blocks, cold = {}, [], []
        changes = selectors = coverage = 0
        for c in m['cases']:
            d = directory(cfg['id'], c)
            r = json.loads((d / 'result.json').read_text())
            assert (d / 'complete.json').exists() and r['valid'] and r['case'] == c
            assert r['config'] == cfg and r['manifest_hash'] == m['manifest_hash']
            h, a = audit_episode(cfg, d, r)
            dd[key(c)] = h
            baseline = data['q1_' + cfg['base_id']][key(c)]
            assert {k: v for k, v in h['identity'].items() if k != 'first_input_hash'} == {
                k: v for k, v in baseline['identity'].items() if k != 'first_input_hash'
            }
            same_input = h['identity']['first_input_hash'] == baseline['identity']['first_input_hash']
            error = float(np.max(np.abs(h['gradient0'] - baseline['gradient0'])))
            check = dict(condition=cfg['id'], case=c, same_input=same_input, max_abs=error)
            if same_input:
                assert np.allclose(h['gradient0'], baseline['gradient0'], atol=1e-4, rtol=1e-3), (
                    'first positive direction changed on identical input', cfg['id'], c,
                )
                check.update(status='same_input_passed', passed=True)
            else:
                ref = Path(m['baseline'][cfg['base_id']][ck(c)]['directory'])
                q = json.loads((d / 'infer.jsonl').open().readline())
                old_q = json.loads((ref / 'infer.jsonl').open().readline())
                differences = input_difference(d, ref)
                assert differences and q['seed'] == old_q['seed']
                evidence = dict(
                    condition=cfg['id'], case=c, directory=str(d), reference=str(ref),
                    new_input_hash=h['identity']['first_input_hash'],
                    reference_input_hash=baseline['identity']['first_input_hash'],
                    new_input_sha256=sha(d / 'input_0000.npz'),
                    reference_input_sha256=sha(ref / 'input_0000.npz'),
                    seed=q['seed'], differences=differences, gradient_max_abs=error,
                    raw0_max_abs=float(np.max(np.abs(h['raw0'] - baseline['raw0']))),
                    actions0_max_abs=float(np.max(np.abs(h['actions0'] - baseline['actions0']))),
                    cause='Stored first inputs differ; rendering cause is not established.',
                )
                anomalies.append(evidence)
                check.update(status='different_input_not_comparable', passed=None)
            gradient_checks.append(check)
            blocks += a['blocks']
            cold.append(a['cold'])
            changes += a['changes']
            selectors += a['selectors']
            coverage += bool(a['blocks'])
            for q in a['requests']:
                requests.append(dict(
                    condition=cfg['id'], **c, replan=q['replan'], path=q['route']['path'],
                    policy_ms=q['timing']['policy_wall_ms'], selector_ms=q['timing']['selector_total_ms'],
                    selector_capture_ms=q['timing']['selector_capture_ms'], rpc_ms=q['rpc_ms'],
                ))
            episodes.append(dict(
                condition=cfg['id'], **c, success=r['success'], actions=r['actions'],
                predictions=r['predictions'], wall_s=r['wall_s'],
                policy_wall_s=r['policy_wall_s'], cold_s=a['cold'],
            ))
        data[cfg['id']] = dd
        quality.append(dict(
            condition=cfg['id'], projections=4, compensation=cfg['compensation'],
            n=200, success=sum(h['result']['success'] for h in dd.values()), reused=False,
            **{s: sum(h['result']['success'] for k, h in dd.items() if k[0] == s) for s in SUITES},
        ))
        steady = [q for block in blocks for q in block]
        costs.append(dict(
            condition=cfg['id'], n=200, steady_covered_episodes=coverage,
            steady_complete_cycles=len(blocks),
            cycle_policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in steady])),
            full_policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in steady if q['route']['path'] == 'full'])),
            fast_policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in steady if q['route']['path'] == 'fast'])),
            mean_cold_s=float(np.mean(cold)), mask_changes=changes, selection_events=selectors, max_run=1,
        ))
        atomic(OUT / 'condition_audits' / f"{cfg['id']}.json", dict(
            passed=True, n=200, projections=4, full_forward_count=1, max_run=1,
        ))
        print(json.dumps(dict(audited_condition=cfg['id'], n=200)), flush=True)
    comparisons = [('q1_' + cfg['base_id'], cfg['id'], False) for cfg in CONDITIONS]
    comparisons.append((CONDITIONS[0]['id'], CONDITIONS[1]['id'], True))
    strict, sensitivity, exceptions = [], [], []
    for a, b, same_mask in comparisons:
        for scope, target in (('strict', strict), ('all200', sensitivity)):
            row, ex = paired(a, b, data[a], data[b], keys, scope, same_mask)
            target.append(row)
            if scope == 'strict':
                exceptions += ex
    # Every non-comparable first gradient must also be excluded from its strict pair.
    for anomaly in anomalies:
        assert any(e['candidate'] == anomaly['condition'] and tuple(e['case']) == key(anomaly['case'])
                   and 'first_input_hash' in e['errors'] for e in exceptions)
    for filename, rs in (
        ('summary.csv', quality), ('episodes.csv', episodes), ('requests.csv', requests),
        ('costs.csv', costs), ('paired.csv', strict), ('paired_sensitivity.csv', sensitivity),
    ):
        write_csv(OUT / filename, rs)
    checked = sum(c['status'] == 'same_input_passed' for c in gradient_checks)
    assert checked + len(anomalies) == 400
    atomic(OUT / 'first_projection_checks.json', dict(
        passed=True, total=400, compared_identical_inputs=checked,
        different_inputs_not_comparable=len(anomalies), checks=gradient_checks,
    ))
    atomic(OUT / 'first_input_anomalies.json', dict(anomalies=anomalies, rollout_reruns=0))
    atomic(OUT / 'prefix_audit.json', dict(pair_exceptions=exceptions, all_valid_outcomes_retained=True))
    workers = []
    for wd in sorted((OUT / 'workers').glob('gpu*')):
        if not (wd / 'startup.json').exists():
            continue
        startup = json.loads((wd / 'startup.json').read_text())
        state = json.loads((wd / 'state.json').read_text())
        exits = json.loads((wd / 'exits.json').read_text())
        assert state['status'] == 'complete' and exits['service'] in (-15, 0)
        assert all(code == 0 for code in exits['evaluations'])
        assert json.loads((wd / 'compatibility_gate.json').read_text())['passed']
        iso = json.loads((wd / 'isolation_gate.json').read_text())
        assert startup['actors'] == 1 or iso['passed']
        attempts = [json.loads(p.read_text()) for p in wd.glob('startup_attempt*.json')]
        workers.append(dict(
            identity=json.loads((wd / 'identity.json').read_text()), startup=startup,
            startup_attempts=attempts, state=state, exits=exits, isolation=iso,
        ))
    start = min(s['time'] for w in workers for s in (w['startup_attempts'] or [w['startup']]))
    end = max(w['state']['time'] for w in workers)
    wall = end - start
    retries = [str(p) for p in (OUT / 'technical_attempts').rglob('*.json')
               if json.loads(p.read_text())['attempts'] > 1]
    throughput = dict(
        new_episodes=400, evaluation_wall_s=wall, episodes_per_hour=400 * 3600 / wall,
        actions_per_s=sum(r['actions'] for r in episodes) / wall,
        requests_per_s=sum(r['predictions'] for r in episodes) / wall,
        service_startup_s=[w['startup']['startup_s'] for w in workers],
        technical_retries=retries, workers=workers,
        controller_start=json.loads((OUT / 'process.json').read_text())['time'],
    )
    assert len(episodes) == 400
    atomic(OUT / 'throughput.json', throughput)
    report = ROOT / 'docs/experiments/idea1/idea1_v3_4_E2_cycle5_q4_v12_report.md'
    lines = [
        '# 周期5、V12：一次与四次反推，有无外推各200', '',
        'Full-D10→Fast-D2×4，只允许间隔跳6层。四次从首Full启用，共享一次可微Full-D10前向，'
        '第一方向为原正权、额外三组固定seed340610正负方向；原残差门控、RMS评分，不使用补偿误差方向。'
        '新400回合，旧一次基线400有效记录复用。', '',
        '|反推次数|无补偿|外推补偿|', '|---|---|---|',
    ]
    for q in (1, 4):
        vs = [next(r for r in quality if r['projections'] == q and r['compensation'] == comp)['success']
              for comp in ('none', 'extrapolate')]
        lines.append(f'|{q}|' + '|'.join(f'{v}/200（{v/2:.1f}%）' for v in vs) + '|')
    lines += ['', '|条件|均摊ms/请求|Full ms|Fast ms|覆盖回合|平均冷捕获秒|', '|---|---|---|---|---|---|']
    for cfg in CONDITIONS:
        old = baseline_costs[cfg['base_id']]
        lines.append(f"|q1_{cfg['base_id']}|{float(old['cycle_policy_ms']):.2f}|{float(old['full_policy_ms']):.2f}|"
                     f"{float(old['fast_policy_ms']):.2f}|{old['steady_covered_episodes']}/200|{float(old['mean_cold_s']):.2f}|")
    for r in costs:
        lines.append(f"|{r['condition']}|{r['cycle_policy_ms']:.2f}|{r['full_policy_ms']:.2f}|"
                     f"{r['fast_policy_ms']:.2f}|{r['steady_covered_episodes']}/200|{r['mean_cold_s']:.2f}|")
    lines += ['', '稳态从第二Full起仅取完整五请求块，不含冷捕获与RPC排队；Full峰值单列。'
              '自由轨迹耗时不当作同输入因果加速。', '',
              '|基准→候选|严格n|恢复/新增失败|task等权变化pp|探索95%区间pp|', '|---|---|---|---|---|']
    for r in strict:
        lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|"
                     f"{r['task_balanced_delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines += [
        '', f"实际闭环窗口{wall/60:.2f}分钟，{throughput['episodes_per_hour']:.1f}回合/小时，"
        f"{throughput['actions_per_s']:.2f}动作/秒，{throughput['requests_per_s']:.2f}请求/秒；"
        f"服务准备秒数{throughput['service_startup_s']}单列。技术重试{len(retries)}回合。", '',
        f'同首输入第一投影{checked}条按原容差通过，{len(anomalies)}条因首输入不同不作梯度一致性比较。'
        f'共同前缀配对异常累计{len(exceptions)}项；完整200描述保留，严格配对与all200敏感性分别输出。', '',
        'libero_goal/task08/init23的q4外推与旧q1外推首帧腕部图像有3个通道值相差1，'
        '其余已存输入相同；渲染差异的根因尚未确认。该案例保留有效结果，从相关严格配对排除，未重跑。'
        '原汇总器在应用前缀排除前就要求梯度一致，因而中断；冻结源码和failed.json保留，'
        '本次独立分析只修正比较的适用条件，未放宽任何容差。证据见first_input_anomalies.json和review_manifest.json。', '',
        'suite分层task聚类bootstrap2000次，seed281034，探索区间未校正多重比较。'
        '四组方向仅用一个预设随机种子，并非随机种子总体均值。近似PyTorch Full-D10回放、'
        '局部门控、多层交互及补偿历史时效性局限延续。', '',
        '[冻结方案](idea1_v3_4_E2_cycle5_q4_v12_plan.md)',
    ]
    report.write_text('\n'.join(lines) + '\n')
    resolution = dict(
        resolved=True, analysis_only=True, frozen_sources_unchanged=True,
        rollout_reruns=0, review_source_sha256=review_sha, report=str(report), time=time.time(),
    )
    atomic(OUT / 'failure_resolution.json', resolution)
    atomic(OUT / 'audit.json', dict(
        passed=True, new_episodes=400, source_hashes=True, period=5, projections=4,
        shared_forward=1, max_run=1, first_projection_reference=True,
        first_projection_compared=checked, first_projection_ineligible=len(anomalies),
        prefix_exceptions=len(exceptions), review_source_sha256=review_sha, time=time.time(),
    ))
    atomic(OUT / 'complete.json', dict(report=str(report), review=True, time=time.time()))
    atomic(OUT / 'status.json', dict(status='complete', completed=400, review=True, time=time.time()))


if __name__ == '__main__':
    main()
