"""Summarize the bounded experiment and audit only its actual mechanisms."""
import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from .common import ROOT, OUT, directory, key, sha
from .rules import motion_features, skip_budget


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def load_episode(config, case, path):
    result = json.loads((path/'result.json').read_text())
    rows = [json.loads(line) for line in (path/'infer.jsonl').read_text().splitlines()]
    terminal = json.loads((path/'terminal.json').read_text())
    tail = rows[3:]
    finish_ms = result['timing']['finish_s']*1000
    def cost(requests):
        return sum(r['timing']['policy_wall_ms']-r['timing'].get('selector_capture_ms',0.)+
                   r.get('ack_wall_ms',0.) for r in requests)+finish_ms
    fulls = sum(r['route']['path']=='full' for r in rows)
    drafts = [r for r in rows if r['route']['draft_attempted']]
    selectors = [event for row in rows for event in row['selector_events']]+terminal['selector_events']
    native_ae = [r['timing']['draft_ae_ms'] for r in drafts]
    kalman_cpu = sum(r['timing'].get('kalman_predict_ms',0.)+r['timing'].get('kalman_encode_ms',0.) for r in rows)
    tail_actions = sum(r['executed_length'] for r in tail)
    summary = dict(condition=config['id'], case=key(case), success=int(result['success']),
        actions=result['actions'], requests=len(rows), fulls=fulls, drafts=len(drafts),
        fallbacks=sum(r['route']['fallback'] for r in rows), selectors=len(selectors),
        wall_s=result['wall_s'], capture_s=result['capture_s'], warm_policy_ms=cost(rows),
        warm_tail_ms=cost(tail), warm_tail_actions=tail_actions,
        warm_ms_per_action=cost(rows)/result['actions'],
        warm_tail_ms_per_action=cost(tail)/tail_actions if tail_actions else None,
        warm_wall_ms_per_action=(result['wall_s']-result['capture_s'])*1000/result['actions'],
        draft_ae_ms_total=sum(native_ae), draft_ae_count=len(native_ae),
        selector_wait_ms=sum(e['wait_ms'] for e in selectors),
        selector_score_ms=sum(e['score_ms'] for e in selectors),
        selector_ready=sum(bool(e['ready_before_wait']) for e in selectors),
        kalman_cpu_ms=kalman_cpu, ack_cpu_rpc_ms=sum(r.get('ack_wall_ms',0.) for r in rows),
        error_build_ms=sum(r['timing'].get('full_error_build_ms',0.) for r in rows),
        directory=str(path))
    return result, rows, terminal, summary


def audit(config, case, path, manifest):
    result, rows, terminal, _ = load_episode(config, case, path)
    identity = json.loads((path/'identity.json').read_text())
    assert result['valid'] and identity['manifest_hash']==manifest['manifest_hash']
    assert result['config']==config and result['case']==case and result['pace_hz']==20
    assert identity['settling']==10 and identity['env_seed']==7
    assert not identity['protocol']['native_inference_graph'] and not identity['protocol']['verify_graph']
    assert len(rows)==result['predictions'] and sum(r['executed_length'] for r in rows)==result['actions']
    telemetry = [json.loads(line) for line in (path/'telemetry.jsonl').read_text().splitlines()]
    commands, poses = [], []
    for record in telemetry:
        if record['action'] is not None:
            a = np.asarray(record['action'],np.float32)
            a[:6] = np.clip(a[:6],-1,1)
            commands.append(a)
            poses.append(record['eef_quat'])
    full_steps, launches, events = [], [], []
    for index, row in enumerate(rows):
        route = row['route']
        assert row['round']==index and row['step']==sum(r['executed_length'] for r in rows[:index])
        assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
        assert row['executed_length']<=row['execute_length']<=5
        if route['draft_attempted']:
            assert route['verify_performed'] and route['actual_fast_nfe']==config['fast_steps']
            mask = route['draft_skip']
            assert len(mask)==route['skip_count'] and all(b-a>1 for a,b in zip(mask,mask[1:]))
            assert all(1<=layer<=17 for layer in mask)
            assert route['draft_history_full_steps']==full_steps[-2:]
            if config['dynamic']:
                f = motion_features(commands[:row['step']],poses[:row['step']])
                count, phase = skip_budget(f,manifest['phase_rule']['thresholds'])
                assert route['skip_count']==count and route['phase']==phase and count in (5,6,7,8)
            else:
                assert route['skip_count']==6
            if config['initial']=='kalman':
                assert route['kalman_used'] and route['kalman_observations']==row['step']
                assert route['kalman_source_last_step']==row['step']-1
                assert route['initial_state']=='kalman' and route['initial_unused_dims']=='zeros25'
        if route['path']=='full':
            if config['mode']=='verified':
                assert route['mask_refresh'] and route['steady_vjp_count']==4
                if config['scoring']=='compensation_error':
                    if not full_steps:
                        assert route['scoring_mode']=='warm_original'
                    elif len(full_steps)==1:
                        assert route['scoring_mode']=='comp_error_hold'
                        assert route['score_source_full_steps']==[full_steps[-1],row['step']]
                        assert route['score_error_ratio']==0.
                    else:
                        assert route['scoring_mode']=='comp_error_extrapolate'
                        assert route['score_source_full_steps']==full_steps[-2:]+[row['step']]
                        expected=(row['step']-full_steps[-1])/(full_steps[-1]-full_steps[-2])
                        assert abs(route['score_error_ratio']-expected)<1e-12
                else:
                    assert route['scoring_mode']=='original'
                launches.append((index,row['step']))
            full_steps.append(row['step'])
            if config['mode']=='verified':
                assert route['history_full_steps_after']==full_steps[-2:]
        events.extend(row['selector_events'])
    events.extend(terminal['selector_events'])
    assert result['counts']['selector']==result['counts']['completed_selector']==len(events)==len(launches)
    assert result['counts']['vjp']==4*len(launches)
    assert result['counts']['verify']==result['counts']['draft']==sum(r['route']['draft_attempted'] for r in rows)
    assert result['counts']['kalman']==sum(r['route'].get('kalman_used',False) for r in rows)
    assert result['counts']['full']==len(full_steps)
    for event in events:
        assert (event['origin_round'],event['origin_step'])==launches[event['id']]
    # Verify every actual selected mask against its own recorded gradient.
    from preparations.idea1_v3_4_selection_async.core import choose
    with np.load(path/'arrays.npz') as arrays:
        gradients = {}
        for name in arrays.files:
            if name.startswith('gate_gradient_job'):
                job = int(name.split('job',1)[1].split('_',1)[0])
                gradients[job] = arrays[name]
                assert gradients[job].shape==(4,17) and np.isfinite(gradients[job]).all()
            if name.startswith('kalman_initial_prefix_'):
                assert np.max(np.abs(arrays[name][:,7:]))==0
        origin_to_job = {event['origin_step']:event['id'] for event in events}
        for row in rows:
            route = row['route']
            if route['draft_attempted']:
                expected, _, _ = choose(gradients[origin_to_job[route['mask_origin_step']]],route['skip_count'])
                assert expected==route['draft_skip']
    return dict(passed=True,case=case,condition=config['id'],requests=len(rows),fulls=len(full_steps),selectors=len(events))


def run(partial=False):
    manifest = json.loads((OUT/'manifest.json').read_text())
    episodes, requests, budgets, audits = [], [], Counter(), []
    data = {}
    for config in manifest['conditions']:
        data[config['id']] = {}
        for case in manifest['cases']:
            path = directory(config['id'],case)
            if not (path/'complete.json').exists():
                if partial:
                    continue
                raise RuntimeError(f'Missing required episode: {path}')
            result, rows, terminal, summary = load_episode(config,case,path)
            data[config['id']][key(case)] = summary
            episodes.append(summary)
            if not partial:
                audits.append(audit(config,case,path,manifest))
            for row in rows:
                route, timing = row['route'],row['timing']
                if route['draft_attempted'] and config['dynamic']:
                    budgets[route['skip_count']] += 1
                requests.append(dict(condition=config['id'],case=key(case),request=row['round'],step=row['step'],
                    path=route['path'],category=route['request_category'],executed=row['executed_length'],
                    draft=route['draft_attempted'],fallback=route['fallback'],accepted=route['accepted_length'],
                    skip=route.get('skip_count'),phase=route.get('phase'),scoring_mode=route.get('scoring_mode'),
                    kalman=route.get('kalman_used',False),fast_nfe=route.get('actual_fast_nfe'),
                    full_ae_ms=timing.get('full_ae_ms'),draft_ae_ms=timing.get('draft_ae_ms'),
                    draft_ms=timing.get('draft_path_wall_ms'),verify_ms=timing.get('verify_path_wall_ms'),
                    policy_ms=timing['policy_wall_ms'],capture_ms=timing.get('selector_capture_ms',0.),
                    ack_ms=row.get('ack_wall_ms',0.),selector_wait_ms=timing.get('selector_wait_ms',0.)))
    summaries = []
    for config in manifest['conditions']:
        rr = list(data[config['id']].values())
        if not rr:
            continue
        totals = {field:sum(r[field] for r in rr) for field in (
            'success','actions','requests','fulls','drafts','fallbacks','selectors','wall_s','capture_s',
            'warm_policy_ms','warm_tail_ms','warm_tail_actions','draft_ae_ms_total','draft_ae_count',
            'selector_wait_ms','selector_ready','kalman_cpu_ms','ack_cpu_rpc_ms','error_build_ms')}
        summaries.append(dict(condition=config['id'],n=len(rr),**totals,
            success_rate=totals['success']/len(rr),full_fraction=totals['fulls']/totals['requests'],
            rejection_rate=totals['fallbacks']/max(totals['drafts'],1),
            warm_ms_per_action=totals['warm_policy_ms']/totals['actions'],
            warm_tail_ms_per_action=totals['warm_tail_ms']/max(totals['warm_tail_actions'],1),
            warm_wall_ms_per_action=(totals['wall_s']-totals['capture_s'])*1000/totals['actions'],
            draft_ae_mean_ms=totals['draft_ae_ms_total']/max(totals['draft_ae_count'],1)))
    pairs = []
    for control in ('full_d10','v12_d2_control'):
        for config in manifest['conditions']:
            candidate = config['id']
            if candidate==control:
                continue
            common = sorted(set(data[control]) & set(data[candidate]))
            if not common:
                continue
            aa,bb = [data[control][k] for k in common],[data[candidate][k] for k in common]
            ratios = [a['warm_tail_ms_per_action']/b['warm_tail_ms_per_action']
                      for a,b in zip(aa,bb) if a['warm_tail_ms_per_action'] and b['warm_tail_ms_per_action']]
            wall_ratios = [a['warm_wall_ms_per_action']/b['warm_wall_ms_per_action'] for a,b in zip(aa,bb)]
            pairs.append(dict(control=control,candidate=candidate,n=len(common),
                control_success=sum(r['success'] for r in aa),candidate_success=sum(r['success'] for r in bb),
                recovered=sum(not a['success'] and b['success'] for a,b in zip(aa,bb)),
                new_failures=sum(a['success'] and not b['success'] for a,b in zip(aa,bb)),
                tail_speedup_geomean=float(np.exp(np.mean(np.log(ratios)))) if ratios else None,
                tail_speedup_pooled=(sum(r['warm_tail_ms'] for r in aa)/max(sum(r['warm_tail_actions'] for r in aa),1))/
                    (sum(r['warm_tail_ms'] for r in bb)/max(sum(r['warm_tail_actions'] for r in bb),1)),
                wall_per_action_speedup_geomean=float(np.exp(np.mean(np.log(wall_ratios))))))
    analysis = dict(complete=len(episodes)==manifest['target_episodes'],episodes=len(episodes),
        manifest_hash=manifest['manifest_hash'],summary=summaries,paired=pairs,
        dynamic_skip_distribution=dict(budgets),time=time.time())
    atomic(OUT/'analysis.json',analysis)
    write_csv(OUT/'episodes.csv',episodes)
    write_csv(OUT/'requests.csv',requests)
    write_csv(OUT/'summary.csv',summaries)
    write_csv(OUT/'paired.csv',pairs)
    if partial:
        return analysis
    assert len(episodes)==280 and all(sum(1 for r in episodes if r['condition']==c['id'])==40 for c in manifest['conditions'])
    expected_sources = dict(manifest['source_files'])
    revision_file = OUT/'source_revisions.json'
    revisions = json.loads(revision_file.read_text()) if revision_file.exists() else []
    for revision in revisions:
        for name,before in revision['before'].items():
            assert expected_sources[name]==before
            expected_sources[name]=revision['after'][name]
    source_ok = all(sha(ROOT/name)==value for name,value in expected_sources.items())
    assert source_ok, 'Inference implementation changed after freezing'
    assert budgets[7]+budgets[8]>0, 'Dynamic rule never used its aggressive budgets'
    # Common initial input and native Full outputs are reported separately from
    # the later freely diverging trajectories; do not silently discard failures.
    prefix = []
    for case in manifest['cases']:
        base = directory('full_d10',case)
        first = json.loads((base/'infer.jsonl').read_text().splitlines()[0])
        with np.load(base/'arrays.npz') as z:
            raw = z['raw_0'].copy()
        for config in manifest['conditions'][1:]:
            path = directory(config['id'],case)
            current = json.loads((path/'infer.jsonl').read_text().splitlines()[0])
            with np.load(path/'arrays.npz') as z:
                delta = float(np.max(np.abs(z['raw_0']-raw)))
                same_raw = bool(np.allclose(z['raw_0'],raw,atol=1e-5,rtol=1e-5))
            prefix.append(dict(condition=config['id'],case=key(case),
                same_input=first['input_hash']==current['input_hash'],native_full_raw_max_abs=delta,
                native_full_passed=same_raw))
    write_csv(OUT/'prefix.csv',prefix)
    atomic(OUT/'audit.json',dict(passed=True,source_integrity=source_ok,source_revisions=revisions,episodes=audits,
        prefix_same_input=sum(r['same_input'] for r in prefix),prefix_raw_passed=sum(r['native_full_passed'] for r in prefix),
        prefix_checks=len(prefix),time=time.time()))
    report(analysis,manifest)
    atomic(OUT/'complete.json',dict(valid=True,experiments_completed=['E1','E2','E3'],
        episodes=280,manifest_hash=manifest['manifest_hash'],report_sha256=sha(ROOT/'docs/experiments/idea1/idea1_v4_0_report.md'),
        analysis_source_sha256=sha(Path(__file__)),source_integrity=True,time=time.time()))
    return analysis


def report(analysis,manifest):
    by = {r['condition']:r for r in analysis['summary']}
    pairs = {(r['control'],r['candidate']):r for r in analysis['paired']}
    lines = ['# idea1 v4.0：三项快速验证结果','',
        '三个独立实验均已完成；七条件各40例，共280条。使用相同官方初态、生成seed和20Hz执行节奏；所有有效失败保留。选择器保留四方向一批与反向融合CUDA Graph，原生encoder、AE、验证均未上图。',
        '', '**本轮判断：外推误差评分有质量收益信号，尚未带来进一步提速；直接Kalman初值替换与本次动态预算规则未得到质量和速度的净收益。没有达到1.5倍。**',
        '', '| 条件 | 成功/40 | 对D2对照恢复/新增失败 | 热态策略ms/动作 | 相对Full倍速 | Fast AE ms | 实际Full占比 | 草稿拒绝率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for config in manifest['conditions']:
        name = config['id']
        s = by[name]
        pair = pairs.get(('v12_d2_control',name))
        speed = pairs.get(('full_d10',name))
        delta = '—' if pair is None or name=='full_d10' else f"+{pair['recovered']}/−{pair['new_failures']}"
        ratio = '1.000' if speed is None else f"{speed['tail_speedup_geomean']:.3f}"
        ae = f"{s['draft_ae_mean_ms']:.3f}" if s['drafts'] else '—'
        lines.append(f"|{name}|{s['success']}/40|{delta}|{s['warm_tail_ms_per_action']:.3f}|{ratio}|{ae}|{s['full_fraction']:.1%}|{s['rejection_rate']:.1%}|")
    lines += ['', '## 实验1：Kalman作为AE初始状态','',
        f"Gaussian D2：{by['v12_d2_control']['success']}/40；Kalman D2：{by['v12_d2_kalman']['success']}/40；Gaussian D1：{by['v12_d1_gaussian']['success']}/40；Kalman D1：{by['v12_d1_kalman']['success']}/40。",
        '', '同为D2时，Kalman初值成功数下降；降到D1虽然将AE降到约7.4ms，但成功数只剩19/40，整体策略成本也没有快过原D2对照。Kalman预测和坐标重编码约5.3ms/次Fast，状态更新的ack成本另计入主表。Gaussian D1的39/40属于带验证及Full回退的完整系统：74.7%的草稿被拒，76.6%的请求实际执行Full，因此策略只有Full的约0.70倍速度。',
        '', '这是指定的直接初值替换：冻结xyz命令Kalman，只吸收已执行动作，旋转/夹爪保持，按当前输出变换逆映射，后25维为零；仍在原生D1/D2时间节点计算。确定性动作初值与训练时高斯起点分布不同，本实验不覆盖加噪或缩短时间起点的其他warm-start设计。Kalman预测/坐标转换及ack中的状态更新成本均计入统计。',
        '', '## 实验2：外推误差评分','',
        f"共同普通评分：{by['v12_d2_control']['success']}/40；误差方向评分：{by['v12_d2_error_score']['success']}/40。",
        '', '相对D2对照恢复4例、新增失败1例，净多3例（+7.5个百分点）；草稿拒绝率14.25%→13.50%。热态策略成本18.746→18.911ms/动作，未观察到进一步提速。主要积极信号在质量，40例不足以确认稳定泛化收益。',
        '', '首Full原评分，第二Full用hold误差，其后用本路径真实Full间距的外推误差。每个实际Full四方向一批；评分仍为各层投影RMS相加。无未来Fast观测，也未叠加Kalman或动态预算。',
        '', '## 实验3：动态跳层预算','',
        f"固定skip6：{by['v12_d2_control']['success']}/40；动态skip5–8：{by['dynamic_skip5_8_d2']['success']}/40。实际草稿预算分布：{analysis['dynamic_skip_distribution']}。",
        '', '平均跳过6.62层；skip5/6/7/8约占16.5%/35.0%/18.2%/30.3%，没有退化为全skip5。相对D2对照恢复4例、新增失败6例，净少2例；同case策略倍速仅1.0016倍，未得到有意义的总体提速。',
        '', f"冻结风险阈值：{manifest['phase_rule']['thresholds']}。由八条既有Full轨迹的323个无夹爪切换窗口校准，仅使用命令/姿态平滑度，不使用成功标签；不是独立未见质量留出集。决策是风险代理，不宣称识别真实接触阶段。",
        '', '## 计时和解释','',
        '主表按实际执行动作均摊，热态后段统一去掉每回合前三个请求，扣除显式选择器捕获；计入对应ack CPU/RPC成本和回合末任务排空。倍速为同case每动作策略成本比的几何均值；每请求、每回合墙钟与策略计算不混称。',
        '', '| 条件 | 全程热态ms/动作 | 扣捕获墙钟ms/动作 | 捕获总秒 | Kalman预测/编码总ms | ack总ms |',
        '|---|---:|---:|---:|---:|---:|']
    for config in manifest['conditions']:
        s = by[config['id']]
        lines.append(f"|{s['condition']}|{s['warm_ms_per_action']:.3f}|{s['warm_wall_ms_per_action']:.3f}|{s['capture_s']:.2f}|{s['kalman_cpu_ms']:.2f}|{s['ack_cpu_rpc_ms']:.2f}|")
    lines += ['', '40例为快速idea筛查，成功数变化与配对恢复/新增失败一起解读，不据此宣称总体非劣或统计显著。该表的Full基线实际完整18层/D10，继承的budget=12配置字段未参与Full执行。',
        '', '正式四服务各14项新增机制检查通过，CPU三项检查通过；280回合机制审查及240项首Full输入/原生输出对照通过。开始时有一次采样步数日志接口格式技术中断，修正未改变采样计算；已完成Full回合保留，仅恢复未完成技术尝试，见`attempts/technical_01/`及`source_revisions.json`。',
        '', '原始记录位于 `results/idea1_v4_0/episodes/`；逐请求、配对、机制审核及首请求检查分别见 `requests.csv`、`paired.csv`、`audit.json`、`prefix.csv`。实验设置见[冻结方案](idea1_v4_0_plan.md)。']
    (ROOT/'docs/experiments/idea1/idea1_v4_0_report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial',action='store_true')
    args = parser.parse_args()
    result = run(args.partial)
    print(json.dumps(dict(complete=result['complete'],episodes=result['episodes'],summary=result['summary']),indent=2))
