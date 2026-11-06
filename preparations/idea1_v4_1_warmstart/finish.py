"""Summarize both parameter-selection and confirmation cases without mixing them."""
import csv
import json
import math
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v4_0.finish import load_episode,write_csv
from .common import ROOT,OUT,OLD,read,key,config,control,directory,sha

REPORT=ROOT/'docs/experiments/idea1/idea1_v4_1_warmstart_report.md'


def finish():
    m=read()
    screen=json.loads((OUT/'replay_selection.json').read_text())
    winners=json.loads((OUT/'champions.json').read_text())
    selected={c['id'] for c in winners['candidates']}
    control_id=control()['id']
    configs=[control()]+[config(c) for c in screen['candidates']]
    records,checks=[],[]
    previous_prefix_anomalies=set()
    for cfg in configs:
        cases=m['cases'] if cfg['id'] in selected or cfg['id']==control_id else m['tune_cases']
        for case in cases:
            d=directory(cfg['id'],case)
            result,rows,terminal,summary=load_episode(cfg,case,d)
            assert result['valid'] and result['manifest_hash']==m['manifest_hash'] and result['config']==cfg
            assert terminal['counts']['selector']==terminal['counts']['completed_selector']
            assert terminal['counts']['vjp']==4*terminal['counts']['full']
            for r in rows:
                route=r['route']
                assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
                if route['draft_attempted']:
                    mask=route['draft_skip']
                    assert len(mask)==6 and all(b-a>1 for a,b in zip(mask,mask[1:]))
                    assert route['actual_fast_nfe']==cfg['fast_steps']
                    assert route['fast_time_embedding']=='openpi_linspace'
                    if cfg.get('warmstart'):
                        n,t=cfg['fast_steps'],cfg['start_time']
                        assert route['kalman_observations']==r['step']
                        assert np.allclose(route['initial_time_nodes'],[t*(1-i/n) for i in range(n)])
                        assert route['warmstart_noise_scale']==cfg['noise_scale']
                        assert np.isclose(sum(r['timing']['draft_nodes']['dt']),-t)
            old=OLD/'episodes/v12_d2_control'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            old_first=json.loads((old/'infer.jsonl').read_text().splitlines()[0])
            with np.load(old/'arrays.npz') as ref,np.load(d/'arrays.npz') as got:
                max_abs=float(np.abs(ref['raw_0']-got['raw_0']).max())
                passed=bool(np.allclose(ref['raw_0'],got['raw_0'],atol=1e-5,rtol=1e-5))
            previous_input_match=rows[0]['input_hash']==old_first['input_hash']
            if not previous_input_match or not passed:
                previous_prefix_anomalies.add(key(case))
            # The fresh control is the primary paired comparison. Historical
            # prefixes that fail the original tolerance are disclosed/excluded
            # from historical paired comparisons, never rerun or relabelled.
            fresh=directory(control_id,case)
            fresh_first=json.loads((fresh/'infer.jsonl').read_text().splitlines()[0])
            with np.load(fresh/'arrays.npz') as ref,np.load(d/'arrays.npz') as got:
                fresh_max_abs=float(np.abs(ref['raw_0']-got['raw_0']).max())
                fresh_match=bool(np.allclose(ref['raw_0'],got['raw_0'],atol=1e-5,rtol=1e-5))
            assert rows[0]['input_hash']==fresh_first['input_hash'] and fresh_match
            checks.append(dict(condition=cfg['id'],case=key(case),first_full_max_abs=max_abs,
                previous_input_match=previous_input_match,previous_full_match=passed,
                fresh_first_full_max_abs=fresh_max_abs,passed=True))
            summary['split']='tune' if case in m['tune_cases'] else 'confirm'
            summary['warmstart_mix_ms']=sum(r['timing'].get('warmstart_mix_ms',0.) for r in rows)
            records.append(summary)
    old_configs=json.loads((OLD/'manifest.json').read_text())['conditions']
    for cfg in [c for c in old_configs if c['id'] in ('full_d10','v12_d2_control')]:
        for case in m['cases']:
            d=OLD/'episodes'/cfg['id']/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            summary=load_episode(cfg,case,d)[3]
            summary['condition']='previous_'+cfg['id']
            summary['split']='tune' if case in m['tune_cases'] else 'confirm'
            records.append(summary)
    summaries=[]
    condition_names=list(dict.fromkeys(r['condition'] for r in records))
    for name in condition_names:
        for split in ('tune','confirm','all'):
            group=[r for r in records if r['condition']==name and (split=='all' or r['split']==split)]
            if not group:continue
            totals={field:sum(r.get(field,0.) for r in group) for field in (
                'actions','requests','fulls','drafts','fallbacks','warm_tail_ms','warm_tail_actions',
                'draft_ae_ms_total','draft_ae_count','wall_s','capture_s','kalman_cpu_ms','warmstart_mix_ms')}
            summaries.append(dict(condition=name,split=split,n=len(group),success=sum(r['success'] for r in group),
                **totals,hot_ms_per_action=totals['warm_tail_ms']/totals['warm_tail_actions'],
                full_fraction=totals['fulls']/totals['requests'],rejection_rate=totals['fallbacks']/max(totals['drafts'],1),
                ae_ms=totals['draft_ae_ms_total']/max(totals['draft_ae_count'],1)))
    paired=[]
    for name in condition_names:
        if name.startswith('previous_'):continue
        for ref_name in (control_id,'previous_v12_d2_control','previous_full_d10'):
            if name==ref_name:continue
            for split in ('tune','confirm','all'):
                a={r['case']:r for r in records if r['condition']==ref_name and (split=='all' or r['split']==split)}
                b={r['case']:r for r in records if r['condition']==name and (split=='all' or r['split']==split)}
                common=set(a)&set(b)
                excluded=sorted(common&previous_prefix_anomalies) if ref_name.startswith('previous_') else []
                keys=sorted(common-set(excluded))
                if not keys:continue
                ratios=[a[k]['warm_tail_ms_per_action']/b[k]['warm_tail_ms_per_action'] for k in keys]
                paired.append(dict(control=ref_name,candidate=name,split=split,n=len(keys),
                    control_success=sum(a[k]['success'] for k in keys),candidate_success=sum(b[k]['success'] for k in keys),
                    recovered=sum(b[k]['success'] and not a[k]['success'] for k in keys),
                    new_failures=sum(a[k]['success'] and not b[k]['success'] for k in keys),
                    hot_speedup_geomean=math.exp(sum(map(math.log,ratios))/len(ratios)),
                    excluded_initial_cases=excluded))
    expected_sources=dict(m['source_files'])
    revisions_path=OUT/'source_revisions.json'
    if revisions_path.exists():
        revisions=json.loads(revisions_path.read_text())
        expected_sources.update(revisions['expected_sources'])
    integrity={p:sha(ROOT/p)==expected for p,expected in expected_sources.items()}
    assert all(integrity.values()),'Frozen implementation changed; record technical revisions explicitly before pooling'
    for worker in (OUT/'workers').glob('gpu*'):
        assert json.loads((worker/'exits.json').read_text())['released']
    analysis=dict(complete=True,manifest_hash=m['manifest_hash'],new_episodes=len(checks),
        replay_samples=m['replay_samples'],summary=summaries,paired=paired,champions=winners,
        historical_prefix_exclusions=sorted(previous_prefix_anomalies),time=time.time())
    write_csv(OUT/'episodes.csv',records)
    write_csv(OUT/'summary.csv',summaries)
    write_csv(OUT/'paired.csv',paired)
    write_csv(OUT/'replay_summary.csv',screen['summary'])
    atomic(OUT/'analysis.json',analysis)
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=len(checks),checks=checks,source_integrity=integrity,
        fresh_control_prefixes_match=True,historical_prefix_exclusions=sorted(previous_prefix_anomalies),
        historical_tolerance_unchanged=True))
    by={(r['condition'],r['split']):r for r in summaries}
    pby={(r['candidate'],r['control'],r['split']):r for r in paired}
    lines=['# v4.1：Kalman 流匹配起点调参结果','',
        f"36组参数先在{m['replay_samples']}个同观测请求上筛选，每种D取两组；四候选和Gaussian D2对照各跑8例，再固定D1/D2冠军在其余32例确认。新增{len(checks)}回合。",'',
        '## 离线筛选','',
        '| 条件 | 前5步命令MSE对Full | 夹爪符号差异率 | 验证拒绝率 | AE ms |',
        '|---|---:|---:|---:|---:|']
    selected_ids={c['id'] for c in screen['candidates']}
    for r in screen['summary']:
        if r['id'] in selected_ids or r['id'] in ('raw_kalman','raw_noise','gaussian_d1','gaussian_d2','gaussian_time_fixed_d1','gaussian_time_fixed_d2','legacy_kf_t1_d1','legacy_kf_t1_d2'):
            reject=f"{r['rejection_rate']:.1%}" if 'rejection_rate' in r else '—'
            lines.append(f"|{r['id']}|{r['mse']:.6f}|{r['gripper_mismatch']:.1%}|{reject}|{r['ae_ms']:.3f}|")
    lines+=['','参考为同观测、同seed的Full-D10输出，并非真值。固定输入的误差和验证接受率只用于筛查，不能代替闭环成功率。',
        '', '## 八例调参闭环','', '| 条件 | 成功/8 | 热态策略ms/动作 | 拒绝率 |', '|---|---:|---:|---:|']
    for cfg in configs:
        r=by[(cfg['id'],'tune')]
        lines.append(f"|{cfg['id']}|{r['success']}/{r['n']}|{r['hot_ms_per_action']:.3f}|{r['rejection_rate']:.1%}|")
    lines+=['','## 冻结参数后的确认','',
        '| 条件 | 调参8例 | 确认32例 | 合计40例 | 确认对D2倍速 | 合计对Full倍速 |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['previous_full_d10','previous_v12_d2_control',control_id]+[c['id'] for c in winners['candidates']]:
        a,b,c=[by[(name,s)] for s in ('tune','confirm','all')]
        if name.startswith('previous_'):
            s1=s2='—'
        else:
            s1='1.000' if name==control_id else f"{pby[(name,control_id,'confirm')]['hot_speedup_geomean']:.3f}"
            s2=f"{pby[(name,'previous_full_d10','all')]['hot_speedup_geomean']:.3f}"
        lines.append(f"|{name}|{a['success']}/8|{b['success']}/32|{c['success']}/40|{s1}|{s2}|")
    lines+=['','## 参数与计时','',
        '最终参数：`'+json.dumps(winners['candidates'],ensure_ascii=False)+'`。',
        '', 'Full/周期/外推/选层/验证均保持原配置。encoder、AE和验证未上图，选择器图保留。热态均摊计时去掉每回合前三请求和显式选择器捕获，包含Kalman预测、坐标编码、混合、ack以及终端选择器排空。',
        '', '本次开始GPU实验前发现并修正了自定义D1/D2时间频率公式：原适配器采用i/512，官方及D10导出采用i/511。新Gaussian D2和全部新候选统一修正；Full-D10导出权重不变。修正前的计划和CPU差异诊断保存在attempts/pre_gpu_time_embedding_review，修正前无本轮GPU数据。',
        '', '主对照是新跑40例的修正Gaussian D2；旧Full/Gaussian D2仅复用v4.0作为历史对照。调参和确认分开报告，旧数据此前已观察，不宣称完全未见基准或统计显著。',
        '', f"历史共同前缀核验发现{len(previous_prefix_anomalies)}个case未过原1e-5容差：{sorted(previous_prefix_anomalies)}。保留全部有效结果，不放宽容差或重跑；与旧数据的配对耗时/恢复统计排除这些case，40例成功数仍为全部描述。所有136个新回合与本轮修正D2对照的首Full均通过原容差，因此本轮候选对新对照的比较不受该历史异常影响。详见first_full_review.json、audit.json和paired.csv中的实际n。",
        '', '资源恢复：新增GPU预检查遇到显存冲突；原调度器随后终止了其他worker。已完成有效回合保留，仅归档恢复未完成技术尝试。调度改为按UUID绑定模型卡，单卡预检查OOM不再中止其他worker；模型推理、参数、筛选规则均未变。修订见source_revisions.json，原日志见attempts/technical_resource_01。',
        '', '源文件、本轮共同首Full、历史前缀异常、实际时间节点/NFE和因果状态检查见`audit.json`；逐例及配对统计见本目录`episodes.csv`、`paired.csv`。所有本轮服务退出并释放GPU租约。']
    REPORT.write_text('\n'.join(lines)+'\n')
    note='\n\n### v4.1 Kalman流匹配起点调参结果\n\n'
    note+=f"36组起点/噪声/NFE参数，{m['replay_samples']}请求离线筛选，8例调参+32例确认；新增{len(checks)}回合。\n\n"
    for candidate in winners['candidates']:
        name=candidate['id'];r=by[(name,'all')];confirm=by[(name,'confirm')]
        note+=f"- `{name}`：{r['success']}/40，其中参数冻结后{confirm['success']}/32；对Full热态倍速{pby[(name,'previous_full_d10','all')]['hot_speedup_geomean']:.3f}。\n"
    note+=f'\n与旧Full的配对倍速排除{len(previous_prefix_anomalies)}个共同首Full未过原容差的case；全部40例成功数保留。本轮与新D2主对照的首Full均匹配。\n'
    note+='\n完整参数、离线误差及对照见[实验报告](docs/experiments/idea1/idea1_v4_1_warmstart_report.md)。\n'
    idea=ROOT/'idea1.md'
    existing=idea.read_text()
    marker='\n\n### v4.1 Kalman流匹配起点调参结果\n'
    if marker in existing:existing=existing.split(marker)[0]
    idea.write_text(existing+note)
    atomic(OUT/'complete.json',dict(valid=True,new_episodes=len(checks),replay_samples=m['replay_samples'],
        source_integrity=True,report_sha256=sha(REPORT),manifest_hash=m['manifest_hash'],time=time.time()))


if __name__=='__main__':
    finish()
