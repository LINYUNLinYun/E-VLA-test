import json
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv, paired_ci
from .common import ROOT, OUT, REPORT, SUITES, MASK, directory, key, sha


def finish():
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert all(sha(ROOT/p)==expected for p,expected in manifest['source_files'].items())
    data, flat, summary, requests = {}, [], [], []
    checked = 0
    for cfg in manifest['conditions']:
        records = []
        for case in manifest['cases']:
            d = directory(cfg['id'], case)
            assert (d/'complete.json').exists()
            result = json.loads((d/'result.json').read_text())
            assert result['valid'] and result['manifest_hash']==manifest['manifest_hash'] and result['config']==cfg
            rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
            refresh_steps = []
            for i, row in enumerate(rows):
                route = row['route']
                refresh = cfg['compensation'] is not None and i % 3 == 0
                assert route['residual_refresh'] == refresh
                if refresh:
                    refresh_steps.append(row['step'])
                assert route['history_steps'] == refresh_steps[-2:]
                assert route['D']==2 and route['skip']==list(MASK) and route['action_source']=='v12_d2'
                assert not route['full_ae_executed'] and not route['full_action_executed']
                assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
                assert row['counts']['full']==0 and row['counts']['verify']==0
                assert row['counts']['draft']==i+1 and row['counts']['residual_refresh']==len(refresh_steps)
                ratio = 0. if len(refresh_steps)<2 else (row['step']-refresh_steps[-1])/(refresh_steps[-1]-refresh_steps[-2])
                assert np.isclose(route['extrapolation_ratio'], ratio)
                assert row['timing']['nodes']['actual_NFE']==2 and row['timing']['time_embedding']=='openpi_linspace'
                requests.append(dict(condition=cfg['id'], **case, request=i, step=row['step'], executed=row['executed_length'], refresh=refresh,
                    policy_ms=row['timing']['policy_wall_ms'], rpc_ms=row['rpc_ms'], ae_ms=row['timing']['ae_ms'],
                    refresh_vlm_ms=row['timing'].get('refresh_vlm_ms',0.), refresh_vision_ms=row['timing'].get('refresh_vision_ms',0.)))
                checked += 1
            hot = rows[3:]
            row = dict(condition=cfg['id'], **case, success=result['success'], actions=result['actions'], requests=result['predictions'],
                residual_refreshes=result['counts']['residual_refresh'], wall_s=result['wall_s'], policy_ms=result['policy_wall_s']*1000,
                hot_ms=sum(r['timing']['policy_wall_ms'] for r in hot), hot_actions=sum(r['executed_length'] for r in hot),
                first_input_hash=rows[0]['input_hash'], source=str(d))
            records.append(row)
            flat.append(row)
        data[cfg['id']] = {key(r):r for r in records}
        for suite in ('all', *SUITES):
            group = [r for r in records if suite=='all' or r['suite']==suite]
            total = lambda field:sum(r[field] for r in group)
            summary.append(dict(condition=cfg['id'], suite=suite, n=len(group), success=sum(r['success'] for r in group),
                policy_ms_per_action=total('policy_ms')/total('actions'), hot_ms_per_action=total('hot_ms')/total('hot_actions'),
                policy_ms_per_request=total('policy_ms')/total('requests'), wall_s_per_episode=total('wall_s')/len(group),
                episodes_per_hour=3600*len(group)/total('wall_s'), actions_per_second=total('actions')/total('wall_s'),
                residual_refreshes=total('residual_refreshes'), full_actions=0))
    baseline, candidate = [cfg['id'] for cfg in manifest['conditions']]
    a, b = data[baseline], data[candidate]
    assert set(a)==set(b) and len(a)==40
    assert all(a[k]['first_input_hash']==b[k]['first_input_hash'] for k in a)
    pairs = [dict(suite=a[k]['suite'], task=a[k]['task'], delta=int(b[k]['success'])-int(a[k]['success'])) for k in a]
    ci = paired_ci(pairs)
    paired = dict(baseline=baseline, candidate=candidate, n=40,
        recovered=sum(not a[k]['success'] and b[k]['success'] for k in a), lost=sum(a[k]['success'] and not b[k]['success'] for k in a),
        delta_pp=100*np.mean([p['delta'] for p in pairs]), ci_low_pp=100*ci[0], ci_high_pp=100*ci[1])
    workers = sorted((OUT/'workers').glob('gpu*'))
    assert workers and all(json.loads((w/'exits.json').read_text())['released'] for w in workers)
    assert all(json.loads((w/'preflight.json').read_text())['passed'] and json.loads((w/'cross_service.json').read_text())['passed'] for w in workers)
    for name, rows in [('episodes.csv',flat), ('summary.csv',summary), ('requests.csv',requests), ('paired.csv',[paired])]:
        write_csv(OUT/name, rows)
    lines = ['# V12子路径：无补偿与仅刷新残差的外推补偿', '',
        '两组各40条：四suite各10task、官方init19。固定V12跳层[2,3,5,14,16,17]，H50/R5、20Hz；所有实际动作均来自当前V12前缀和D2，零Full-D10动作、零验证、零拒绝接管。两组统一修正后的OpenPI时间编码。', '',
        '补偿组在重规划0/3/6…用当前观测完整VLM捕获真残差，随后重新运行V12并执行D2动作；刷新时ratio0，其余按实际age/gap外推。只保留自身最近两次刷新，第一周期hold。无补偿组完全不运行额外Full刷新。', '',
        '|条件|Spatial|Object|Goal|LIBERO-10|合计|热态策略ms/动作|含冷态策略ms/动作|',
        '|---|---|---|---|---|---|---|---|']
    for cfg in manifest['conditions']:
        group = {r['suite']:r for r in summary if r['condition']==cfg['id']}
        all_row = group['all']
        values = [cfg['id'], *[f"{group[s]['success']}/10" for s in SUITES], f"{all_row['success']}/40", f"{all_row['hot_ms_per_action']:.3f}", f"{all_row['policy_ms_per_action']:.3f}"]
        lines.append('|'+ '|'.join(values)+'|')
    lines += ['', f"补偿相对无补偿：恢复{paired['recovered']}例、新增失败{paired['lost']}例；成功率差{paired['delta_pp']:+.1f}个百分点，suite分层task聚类探索95%区间[{paired['ci_low_pp']:+.1f},{paired['ci_high_pp']:+.1f}]。40例为探索评测，不是官方全量。", '',
        '热态统计从每回合第4次请求开始，实际刷新、残差提取、输入处理、AE与解码全部计入。本次刷新轮重复准备视觉/VLM，未做共享视觉优化；计时描述本实现的实际成本，不把自由轨迹的成本差称作受控部署加速。无encoder/AE CUDA Graph。', '',
        '旧31/40与169/200属于周期执行Full-D10动作且使用旧D2时间编码的历史混合策略，不作为这两组的新对照。', '',
        '[逐回合结果](../../../results/idea1_v4_2_refresh_only/episodes.csv) · [逐请求](../../../results/idea1_v4_2_refresh_only/requests.csv)']
    REPORT.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json', dict(passed=True, episodes=80, route_checks=checked, initial_pairs=40, full_actions=0, full_ae_calls=0, sources_match=True, time=time.time()))
    atomic(OUT/'complete.json', dict(passed=True, report=str(REPORT), time=time.time()))
    atomic(OUT/'status.json', dict(status='complete', completed=80, target=80, summary=summary, paired=paired, time=time.time()))


if __name__ == '__main__':
    finish()
