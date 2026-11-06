"""Keep fixed-parameter dimension comparisons distinct from retuning results."""
import json
import math
import time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v4_0.finish import load_episode, write_csv
from .common import ROOT, OUT, OLD, REPORT, read, key, config, control, directory, sha, anchors, unique


def aggregate(records):
    summaries=[]
    for name in dict.fromkeys(r['condition'] for r in records):
        for split in ('tune','confirm','all'):
            group=[r for r in records if r['condition']==name and (split=='all' or r['split']==split)]
            if not group:
                continue
            fields=('actions','requests','fulls','drafts','fallbacks','warm_tail_ms','warm_tail_actions',
                'draft_ae_ms_total','draft_ae_count','wall_s','capture_s','kalman_cpu_ms','rotation_forecast_ms','ack_cpu_rpc_ms')
            totals={field:sum(r.get(field,0.) for r in group) for field in fields}
            summaries.append(dict(condition=name,split=split,n=len(group),success=sum(r['success'] for r in group),
                **totals,hot_ms_per_action=totals['warm_tail_ms']/totals['warm_tail_actions'],
                ae_ms=totals['draft_ae_ms_total']/max(totals['draft_ae_count'],1),
                kalman_ms_per_draft=totals['kalman_cpu_ms']/max(totals['drafts'],1),
                full_fraction=totals['fulls']/totals['requests'],rejection_rate=totals['fallbacks']/max(totals['drafts'],1),
                episodes_per_hour=3600*len(group)/totals['wall_s'],actions_per_second=totals['actions']/totals['wall_s']))
    return summaries


def finish():
    manifest=read()
    screen=json.loads((OUT/'replay_selection.json').read_text())
    winners=json.loads((OUT/'champions.json').read_text())
    complete_candidates=unique(anchors()+winners['candidates'])
    full_ids={c['id'] for c in complete_candidates}|{control()['id']}
    configurations=[control()]+[config(c) for c in screen['candidates']]
    records,first,checks=[],{},[]
    def load(cfg,case,path,name,new):
        result,rows,terminal,summary=load_episode(cfg,case,path)
        assert result['valid'] and result['config']==cfg and result['case']==case
        assert result['pace_hz']==20
        if new:
            assert result['manifest_hash']==manifest['manifest_hash']
            assert terminal['counts']['selector']==terminal['counts']['completed_selector']
            assert terminal['counts']['vjp']==4*terminal['counts']['full']
            for row in rows:
                route=row['route']
                assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
                assert route['kalman_dimensions']==cfg['kalman_dimensions']
                assert not route['kalman_gripper_prediction']
                if route['draft_attempted']:
                    assert route['actual_fast_nfe']==cfg['fast_steps']
                    assert route['fast_time_embedding']=='openpi_linspace'
                    assert len(route['draft_skip'])==6 and all(b-a>1 for a,b in zip(route['draft_skip'],route['draft_skip'][1:]))
                    if cfg['initial']=='kalman':
                        assert route['kalman_observations']==row['step']==route['kalman_rotation_observations']
                        assert route['kalman_source_last_step']==row['step']-1
                        assert np.allclose(route['initial_time_nodes'],[cfg['start_time']*(1-i/cfg['fast_steps']) for i in range(cfg['fast_steps'])])
                        assert np.isclose(sum(row['timing']['draft_nodes']['dt']),-cfg['start_time'])
            checks.append(dict(condition=name,case=key(case),requests=len(rows),passed=True))
        with np.load(path/'arrays.npz') as values:
            raw=values['raw_0'].copy()
        first[(name,key(case))]=dict(input_hash=rows[0]['input_hash'],raw=raw)
        summary.update(condition=name,split='tune' if case in manifest['tune_cases'] else 'confirm',
            rotation_forecast_ms=sum(r['timing'].get('kalman_rotation_forecast_ms',0.) for r in rows),
            reused=not new)
        records.append(summary)
    for cfg in configurations:
        cases=manifest['cases'] if cfg['id'] in full_ids else manifest['tune_cases']
        for case in cases:
            path=directory(cfg['id'],case)
            assert (path/'complete.json').exists()
            load(cfg,case,path,cfg['id'],True)
    for reference in manifest['references']:
        for case in manifest['cases']:
            path=Path(reference['root'])/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            load(reference['config'],case,path,reference['id'],False)
    summaries=aggregate(records)
    paired=[]
    comparisons=[]
    for cfg in configurations:
        name=cfg['id']
        for baseline in (control()['id'],'previous_v12_d2_time_fixed_control'):
            if baseline!=name:
                comparisons.append((baseline,name,'gaussian_reference'))
    for anchor in anchors():
        comparisons.append(('previous_'+anchor['id'].replace('kf6_','kf_'),anchor['id'],'matched_3d_to_6d'))
    for baseline,candidate,kind in comparisons:
        for split in ('tune','confirm','all'):
            a={r['case']:r for r in records if r['condition']==baseline and (split=='all' or r['split']==split)}
            b={r['case']:r for r in records if r['condition']==candidate and (split=='all' or r['split']==split)}
            common=sorted(set(a)&set(b))
            if not common:
                continue
            matched,excluded=[],[]
            for case in common:
                x,y=first[(baseline,case)],first[(candidate,case)]
                passed=x['input_hash']==y['input_hash'] and np.allclose(x['raw'],y['raw'],atol=1e-5,rtol=1e-5)
                if passed:
                    matched.append(case)
                else:
                    excluded.append(dict(case=case,input_match=x['input_hash']==y['input_hash'],raw_max_abs=float(np.abs(x['raw']-y['raw']).max())))
            atomic(OUT/'prefix_comparisons'/f'{baseline}__{candidate}__{split}.json',dict(matched=matched,excluded=excluded))
            for scope,keys in (('all_valid',common),('strict_prefix',matched)):
                if not keys:
                    continue
                ratios=[a[k]['warm_tail_ms_per_action']/b[k]['warm_tail_ms_per_action'] for k in keys]
                paired.append(dict(baseline=baseline,candidate=candidate,kind=kind,split=split,scope=scope,n=len(keys),
                    baseline_success=sum(a[k]['success'] for k in keys),candidate_success=sum(b[k]['success'] for k in keys),
                    recovered=sum(not a[k]['success'] and b[k]['success'] for k in keys),
                    lost=sum(a[k]['success'] and not b[k]['success'] for k in keys),
                    speedup_geomean=math.exp(sum(map(math.log,ratios))/len(ratios)),excluded_initial_cases=excluded))
    expected=dict(manifest['source_files'])
    revisions=OUT/'source_revisions.json'
    if revisions.exists():
        expected.update(json.loads(revisions.read_text())['expected_sources'])
    integrity={p:sha(ROOT/p)==value for p,value in expected.items()}
    assert all(integrity.values()),'Implementation changed after manifest freeze'
    assert all(sha(p)==value for p,value in manifest['reference_files'].items()),'Historical results changed'
    workers=sorted((OUT/'workers').glob('gpu*'))
    assert workers and all(json.loads((w/'exits.json').read_text())['released'] for w in workers)
    write_csv(OUT/'episodes.csv',records)
    write_csv(OUT/'summary.csv',summaries)
    write_csv(OUT/'paired.csv',paired)
    write_csv(OUT/'replay_summary.csv',screen['summary'])
    atomic(OUT/'analysis.json',dict(complete=True,new_episodes=len(checks),summaries=summaries,paired=paired,
        winners=winners,manifest_hash=manifest['manifest_hash'],time=time.time()))
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=len(checks),checks=checks,source_integrity=integrity,
        previous_results_unchanged=True,first_full_tolerance=dict(atol=1e-5,rtol=1e-5),time=time.time()))
    by={(r['condition'],r['split']):r for r in summaries}
    lines=['# Kalman六维控制指令预测与流匹配起点', '',
        f"新增{len(checks)}条闭环，复用120条v4.1历史对照。36组起点/噪声/NFE候选先在48个相同观测筛选，再用8例调参、32例确认；固定旧起点参数的两个六维锚点始终补齐40例。",'',
        '位置滤波器保持原样；旋转三个增量控制指令使用独立Kalman、相同q_u/q_w/g/beta与单位Q/R形状。只读取实际已执行命令，夹爪保持，25维padding先验为零。这里仍是带验证和正常Full-D10动作/接管的完整系统。', '',
        '## 同观测离线误差', '', '|方案|前5步七维命令MSE|位置RMSE|旋转RMSE|验证拒绝率|', '|---|---:|---:|---:|---:|']
    chosen={c['id'] for c in complete_candidates}
    for r in screen['summary']:
        if r['id'] in chosen or r['id'] in ('raw_kalman_xyz','raw_kalman_six','gaussian_time_fixed_d1','gaussian_time_fixed_d2','matched3_d1_t75_n100','matched3_d2_t05_n100'):
            lines.append(f"|{r['id']}|{r['mse']:.6f}|{r['xyz_rmse']:.6f}|{r['rotation_rmse']:.6f}|{r.get('rejection_rate',0.):.1%}|")
    lines+=['','参考是同观测Full-D10输出，不是真值；位置/旋转RMSE单位为各自decoded控制命令单位，不混作EEF毫米或绝对姿态角。', '',
        '## 闭环与实际成本', '', '|方案|调参8例|确认32例|全部|热态ms/动作|AE ms|Kalman预测/编码ms/草稿|Full比例|拒绝率|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    names=[r['id'] for r in manifest['references']]+[control()['id']]+[c['id'] for c in complete_candidates]
    for name in dict.fromkeys(names):
        total=by[(name,'all')];tune=by[(name,'tune')];confirmation=by[(name,'confirm')]
        lines.append(f"|{name}|{tune['success']}/{tune['n']}|{confirmation['success']}/{confirmation['n']}|{total['success']}/{total['n']}|{total['hot_ms_per_action']:.3f}|{total['ae_ms']:.3f}|{total['kalman_ms_per_draft']:.3f}|{total['full_fraction']:.1%}|{total['rejection_rate']:.1%}|")
    lines+=['','## 相同起点参数：只增加旋转预测', '', '|基线→候选|共同首Full匹配数|基线成功|六维成功|恢复/新增失败|策略配对倍速|', '|---|---:|---:|---:|---:|---:|']
    for r in paired:
        if r['kind']=='matched_3d_to_6d' and r['split']=='all' and r['scope']=='strict_prefix':
            lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['baseline_success']}|{r['candidate_success']}|{r['recovered']}/{r['lost']}|{r['speedup_geomean']:.3f}×|")
    lines+=['', '所有有效40例成绩保留；未通过原1e-5容差的共同首Full从相关严格配对剔除，具体差值与case见prefix_comparisons。all_valid配对另见paired.csv，不重跑到成功或放宽容差。历史对照在不同时间执行，速度只作描述参考。', '',
        '## 准备与评测吞吐', '', '|服务|模型构造秒（不含预核验）|启动准备秒|预核验状态|', '|---|---:|---:|---|']
    for worker in workers:
        environment=json.loads((worker/'environment.json').read_text()) if (worker/'environment.json').exists() else {}
        identity=json.loads((worker/'identity.json').read_text())
        gate=worker/'preflight/complete.json'
        preparation_end=gate.stat().st_mtime if gate.exists() else json.loads((worker/'exits.json').read_text())['time']
        preparation_s=preparation_end-identity['time']
        construction=f"{environment['model_load_s']:.2f}" if 'model_load_s' in environment else '—'
        state='完成' if gate.exists() else '主矩阵完成前未投入评测，已退出'
        lines.append(f"|{worker.name}|{construction}|{preparation_s:.2f}|{state}|")
    lines+=['', '热态计时排除每回合前三请求和显式选择器捕获，包含六维预测、坐标编码、混合、ack、验证/Full接管及终端选择器排空。实际回合wall、episode/h和动作/s见summary.csv，墙钟包含固定20Hz执行等待。选择器图保留，encoder/AE/验证无图。', '',
        '36组仅做离线筛选，并非全部闭环。这里只改旋转预测维度；旋转噪声参数未重新校准，仍不保证Kalman起点符合AE训练分布。40例为探索评测，不是官方全量或严格未见基准。', '',
        '资源记录：GPU0曾在外部任务切换间隙短暂空闲，随后其他进程启动导致本轮预核验OOM；仅退出本轮失败服务，健康服务继续。为尊重这些任务切换，GPU0后来增加60秒持续观察空闲门槛，读取同一主调度器的快照，不增加GPU查询者。模型与科学参数未变，详见resource_amendment.json及attempts/resource_conflict_*。报告计时字段澄清记录在source_revisions.json。', '',
        '[逐例](../../../results/idea1_v4_3_kalman6/episodes.csv) · [配对](../../../results/idea1_v4_3_kalman6/paired.csv) · [离线](../../../results/idea1_v4_3_kalman6/replay_summary.csv)']
    REPORT.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(valid=True,new_episodes=len(checks),replay_samples=48,report=str(REPORT),report_sha256=sha(REPORT),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',new_episodes=len(checks),summary=summaries,winners=winners,time=time.time()))


if __name__=='__main__':
    finish()
