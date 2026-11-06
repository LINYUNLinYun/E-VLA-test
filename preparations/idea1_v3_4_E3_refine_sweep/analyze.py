import csv,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from .common import ROOT,OUT,SUITES,METRICS,sha,metrics,balanced,tag,select_q


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def read_stage(stage):
    m=json.loads((OUT/'manifest.json').read_text());rows=[];errors=[];maxdiff=0.
    specpath=OUT/'manifest.json' if stage==1 else OUT/'stage2_manifest.json';spec=json.loads(specpath.read_text());steps=2 if stage==1 else 1
    expected={'fresh2','draft',*[tag(q) for q in spec['grid']]}|({'fresh1'} if stage==2 else set())
    for item in m['selected']:
        d=OUT/f'stage{stage}'/f"sample{item['id']:03d}"
        if not (d/'complete.json').exists():errors.append(dict(id=item['id'],error='missing'));continue
        c=json.loads((d/'complete.json').read_text());data=json.loads((d/'metrics.json').read_text())
        assert c['manifest_hash']==m['manifest_hash'] and c['stage_spec_sha256']==sha(specpath)
        assert data['grid']==spec['grid'] and data['steps']==steps and all(r['passed'] for r in data['parity'])
        with np.load(d/'actions.npz') as z:
            assert set(k[:-3] for k in z.files if k.endswith('_e0') and not k.startswith('ref_'))==expected
            for k in z.files:assert z[k].shape==((50,32) if k.endswith('_raw') else (50,7)) and np.isfinite(z[k]).all(),k
            for row in data['rows']:
                assert row['id']==item['id'] and row['suite']==item['case']['suite'] and row['task']==item['case']['task'] and row['split']==item['split']
                name=row['candidate']+'_'+row['noise'];ref='ref_'+row['reference']
                calculated=metrics(z[name],z[name+'_raw'],z[ref],z[ref+'_raw'],row['horizon'])
                for k,v in calculated.items():
                    diff=abs(v-row[k]);maxdiff=max(maxdiff,diff)
                    if diff>1e-10:errors.append(dict(id=item['id'],error='metric mismatch',field=k,difference=diff))
                rows.append(row)
    result=dict(passed=not errors and len({r['id'] for r in rows})==127,requests=len({r['id'] for r in rows}),errors=errors,max_abs_metric_difference=maxdiff,time=time.time())
    atomic(OUT/f'stage{stage}_audit.json',result)
    if not result['passed']:raise RuntimeError(f'Stage{stage} audit failed')
    write_csv(OUT/f'stage{stage}_requests.csv',rows)
    return rows


def summaries(rows):
    output=[]
    for split in ('calibration','heldout','all'):
        selected=[r for r in rows if (split=='all' or r['split']==split) and r['noise']==r['reference']]
        for horizon in (5,50):
            for name in sorted({r['candidate'] for r in selected}):
                a=[r for r in selected if r['candidate']==name and r['horizon']==horizon]
                output.append(dict(split=split,horizon=horizon,candidate=name,requests=len({r['id'] for r in a}),
                    **{k:balanced(a,k) for k in METRICS},**{k+'_request_mean':float(np.mean([r[k] for r in a])) for k in METRICS},
                    ae_ms=float(np.mean([r['ae_ms'] for r in a])),wall_ms=float(np.mean([r['wall_ms'] for r in a]))))
    return output


def paired(rows,name,base,split,h=5):
    a=[r for r in rows if (split=='all' or r['split']==split) and r['noise']==r['reference'] and r['horizon']==h]
    aa={(r['id'],r['noise']):r for r in a if r['candidate']==name};bb={(r['id'],r['noise']):r for r in a if r['candidate']==base}
    assert aa.keys()==bb.keys()
    deltas=[dict(suite=aa[k]['suite'],task=aa[k]['task'],**{m:aa[k][m]-bb[k][m] for m in METRICS}) for k in aa]
    rng=np.random.default_rng(291030);boot=np.zeros((2000,len(METRICS)))
    for suite in SUITES:
        rs=[r for r in deltas if r['suite']==suite];tasks=sorted({r['task'] for r in rs})
        matrix=np.array([[np.mean([r[m] for r in rs if r['task']==t]) for m in METRICS] for t in tasks])
        weights=rng.multinomial(len(tasks),np.ones(len(tasks))/len(tasks),size=2000)/len(tasks)
        boot+=weights@matrix/4
    lo,hi=np.percentile(boot,[2.5,97.5],axis=0)
    return [dict(candidate=name,baseline=base,split=split,horizon=h,metric=m,delta=balanced(deltas,m),ci_low=float(lo[i]),ci_high=float(hi[i])) for i,m in enumerate(METRICS)]


def finish():
    m=json.loads((OUT/'manifest.json').read_text());rows1=read_stage(1);rows2=read_stage(2)
    assert all(sha(ROOT/p)==h for p,h in m['sources'].items())
    from .common import PLAN
    assert sha(PLAN)==m['plan_sha256']
    spec=json.loads((OUT/'stage2_manifest.json').read_text());decision1=json.loads((OUT/'stage1_decision.json').read_text())
    sealed=dict(spec);sealed.pop('spec_hash');assert digest(sealed)==spec['spec_hash'] and spec['time']>=decision1['time']
    recalculated=select_q([r for r in rows1 if r['split']=='calibration' and r['horizon']==5 and r['noise']==r['reference']],m['grid'],'fresh2')
    assert decision1['selection']==recalculated and spec['stage1_decision_sha256']==sha(OUT/'stage1_decision.json')
    decision2=select_q([r for r in rows2 if r['split']=='calibration' and r['horizon']==5 and r['noise']==r['reference']],spec['grid'],'fresh1')
    atomic(OUT/'stage2_decision.json',dict(selection=decision2,selection_split='calibration',time=time.time()))
    summ1=summaries(rows1);summ2=summaries(rows2);write_csv(OUT/'stage1_summary.csv',summ1);write_csv(OUT/'stage2_summary.csv',summ2)
    effects=[]
    for rows,stage,grid,bases in [(rows1,1,m['grid'],['fresh2']),(rows2,2,spec['grid'],['fresh1','fresh2'])]:
        for name in [tag(q) for q in grid]:
            for base in bases:
                for split in ('calibration','heldout','all'):
                    effects += [dict(stage=stage,**r) for r in paired(rows,name,base,split)]
    write_csv(OUT/'paired_effects.csv',effects)
    exits=json.loads((OUT/'service_exits.json').read_text());assert exits and all(r['exit'] is not None for r in exits.values())
    gates=[]
    for d in (OUT/'workers').glob('gpu*'):
        for name in ('gate.json','one_step_gate.json'):
            g=json.loads((d/name).read_text());assert g['passed'];gates.append(str(d/name))
    bootstrap_note='均值先平均每请求双噪声，再按task、suite等权；区间为suite分层task聚类bootstrap2000次，探索性逐点区间，未作多重比较校正。'
    lines=['# E3：噪声强度与失败draft一步修复','',
        '两阶段已按顺序完成，均使用127个实际被拒请求、当前Full KV与双噪声Full-D10参考。偶数task的65请求仅用于选参，奇数task的62请求作为留出；结果是离线动作差，不是修复策略闭环成功率。',
        '',bootstrap_note,'',f"阶段一在调参组选择q={decision1['selection']['q']}；夹爪均值不高于fresh D2的候选存在：{decision1['selection']['has_gripper_not_higher_candidate']}。",'',
        f"阶段二在第一阶段结束后冻结q网格{spec['grid']}，用调参组选择q={decision2['q']}；夹爪均值不高于fresh D1的候选存在：{decision2['has_gripper_not_higher_candidate']}。该条件只是点估计筛选，不是统计非劣结论。"]
    for stage,summ in [(1,summ1),(2,summ2)]:
        for split in ('all','heldout','calibration'):
            lines += ['',f'## 阶段{stage} / {split} / 前五步匹配噪声参考','', '|候选|请求|平移mm代理|旋转L2|夹爪符号差|AE ms|候选wall ms|','|---|---|---|---|---|---|---|']
            for r in summ:
                if r['split']==split and r['horizon']==5:lines.append(f"|{r['candidate']}|{r['requests']}|{r['translation_mm']:.3f}|{r['rotation_l2']:.4f}|{r['gripper_sign']:.2%}|{r['ae_ms']:.3f}|{r['wall_ms']:.3f}|")
    lines += ['', '## 留出组主候选配对差','', '|阶段/候选|对照|指标|均值差|95%区间|','|---|---|---|---|---|']
    for stage,q in [(1,decision1['selection']['q']),(2,decision2['q'])]:
        for r in effects:
            if r['stage']==stage and r['candidate']==tag(q) and r['split']=='heldout' and r['metric'] in ('translation_mm','rotation_l2','gripper_sign'):
                lines.append(f"|{stage}/{tag(q)}|{r['baseline']}|{r['metric']}|{r['delta']:+.5f}|[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]|")
    lines += ['', '![加噪强度曲线](../../../results/idea1_v3_4_E3_refine_sweep/noise_curves.png)',
        '', 'q是动作/噪声线性混合系数，并同时改变积分起点和步长；不等同于只增加独立噪声方差。draft锚点始终为实际ε0生成动作，e1分析分支只是把同一动作对另一个参考作比较。q=1与fresh同NFE输出逐元素一致，不能将其结果当成失败动作信息带来的收益。',
        '', '候选wall包含加噪/AE/解码，当前Full视觉/VLM准备成本另存每请求metrics.json；原已付draft/verify成本没有被抵扣。所有候选先预热后AB/BA三次，次数一致。当前状态/条件保持一致，NFE1/D1与NFE2/D2明确区分。全50步与未clip/raw误差、跨噪声参考结果见逐请求CSV。',
        '', '[冻结方案](idea1_v3_4_E3_refine_sweep_plan.md) · [阶段一逐请求](../../../results/idea1_v3_4_E3_refine_sweep/stage1_requests.csv) · [阶段二逐请求](../../../results/idea1_v3_4_E3_refine_sweep/stage2_requests.csv) · [配对结果](../../../results/idea1_v3_4_E3_refine_sweep/paired_effects.csv)']
    plots(summ1,summ2,m['grid'],spec['grid'])
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E3_refine_sweep_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,stage1_requests=127,stage2_requests=127,sources_match=True,selection_reproduced=True,stage2_frozen_after_stage1=True,all_services_exited=True,gates=gates,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),report_sha256=sha(report),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',stage1_requests=127,stage2_requests=127,selected_q_stage1=decision1['selection']['q'],selected_q_stage2=decision2['q'],time=time.time()))


def plots(summ1,summ2,grid1,grid2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for index,(summ,grid,fresh) in enumerate([(summ1,grid1,'fresh2'),(summ2,grid2,'fresh1')]):
        for col,metric in enumerate(['translation_mm','rotation_l2','gripper_sign']):
            ax=axes[index,col]
            for split,color in [('calibration','#7570b3'),('heldout','#1b9e77')]:
                selected={r['candidate']:r for r in summ if r['split']==split and r['horizon']==5}
                scale=100 if metric=='gripper_sign' else 1
                ax.plot(grid,[selected[tag(q)][metric]*scale for q in grid],marker='o',color=color,label=split)
                ax.axhline(selected[fresh][metric]*scale,color=color,linestyle='--',alpha=.7,label=f'{split} {fresh}')
            ax.set_title(f'Stage {index+1}: {metric}'+(' (%)' if metric=='gripper_sign' else ''));ax.set_xlabel('q');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Failed-draft repair: task/suite-balanced matched-noise H5 differences');fig.tight_layout();fig.savefig(OUT/'noise_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':finish()
