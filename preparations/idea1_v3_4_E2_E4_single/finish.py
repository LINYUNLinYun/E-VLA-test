import csv,json,time
from collections import defaultdict
import numpy as np
from .common import ROOT,OUT,PLAN,LAYERS,SUITES,sha,dump,metrics

METRICS=('translation_mm','rotation_l2','gripper_sign','raw_rms6')

def write_csv(p,rows):
    if not rows:return
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)

from pathlib import Path

def balanced(rows,field):
    groups=defaultdict(list)
    for r in rows:
        v=r[field]
        if v is not None:groups[r['suite'],r['task']].append(v)
    suites=defaultdict(list)
    for (s,t),v in groups.items():suites[s].append(np.mean(v))
    return float(np.mean([np.mean(v) for v in suites.values()]))

def paired_ci(rows,field):
    rng=np.random.default_rng(340429);b=np.zeros(2000)
    for suite in SUITES:
        rs=[r for r in rows if r['suite']==suite];tasks=sorted({r['task'] for r in rs})
        values=np.array([np.mean([r[field] for r in rs if r['task']==t]) for t in tasks])
        b+=rng.choice(values,(2000,len(values)),replace=True).mean(1)/4
    return np.percentile(b,[2.5,97.5]).tolist()

def read_all():
    pairs=json.loads((OUT/'selection.json').read_text());e2=[];ranks=[];projections=[];costs=[];e4=[];hidden=[];timing=[]
    maxdiff=0.;checks=[]
    for p in pairs:
        if p['e2']:
            d=OUT/'E2'/f"sample{p['id']:03d}";assert (d/'complete.json').exists()
            m=json.loads((d/'metrics.json').read_text());e2+=m['rows'];ranks+=m['ranks'];projections+=m['projections']
            costs.append(dict(id=p['id'],**p['case'],**{k:m[k] for k in ('forward_ms','graph_forward_ms','backward35_ms','jacobian_total_ms')}))
            with np.load(d/'arrays.npz') as z:
                assert z['jacobian'].shape==(5,7,17) and np.isfinite(z['jacobian']).all()
                for row in m['rows']:
                    name=f"L{row['layer']}_g{row['gate']:g}"
                    v=metrics(z['full_actions'],z[name+'_actions'],z['full_raw'],z[name])
                    for k,value in v.items():maxdiff=max(maxdiff,abs(row[k]-value))
        d=OUT/'E4'/f"sample{p['id']:03d}";assert (d/'complete.json').exists()
        m=json.loads((d/'metrics.json').read_text());e4+=m['rows'];hidden+=m['hidden'];timing+=m['timing']
        assert len(m['rows'])==17*7*2 and len(m['hidden'])==17*2*9
        with np.load(d/'arrays.npz') as z:
            for row in m['rows']:
                name=f"L{row['layer']}_{row['candidate']}"
                v=metrics(z['full_actions'],z[name+'_actions'],z['full_raw'],z[name+'_raw'],row['horizon'])
                for k,value in v.items():maxdiff=max(maxdiff,abs(row[k]-value))
    assert len(costs)==40 and len({r['id'] for r in e4})==160 and maxdiff<1e-9
    dump(OUT/'metrics_audit.json',dict(passed=True,max_abs_metric_difference=maxdiff,E2_requests=40,E4_requests=160))
    return e2,ranks,projections,costs,e4,hidden,timing

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in m['sources'].items())
    assert sha(PLAN)==m['plan_sha256'] and sha(OUT/'selection.json')==m['selection_sha256']
    e2,ranks,projections,costs,e4,hidden,timing=read_all()
    for name,rows in [('E2_requests',e2),('E2_ranks',ranks),('E2_projections',projections),('E2_costs',costs),('E4_requests',e4),('E4_hidden',hidden),('E4_timing',timing)]:write_csv(OUT/f'{name}.csv',rows)
    e2summary=[]
    for gate in (.9,.5,0.):
        for group in ('translation','rotation','gripper'):
            rows=[r for r in e2 if r['gate']==gate and r['group']==group];rr=[r for r in ranks if r['gate']==gate and r['group']==group]
            e2summary.append(dict(gate=gate,group=group,relative_vector_error=balanced(rows,'relative_vector_error'),direction_cosine=balanced(rows,'direction_cosine'),
                **{k:balanced(rr,k) for k in ('jacobian_spearman','cosine_spearman','residual_spearman','low4_actual_mean','all_actual_mean','low4_in_actual_high4')}))
    write_csv(OUT/'E2_summary.csv',e2summary)
    # Selection code sees validation rows only, never heldout outcomes.
    selected={};selection=[]
    for l in LAYERS:
        rs=[r for r in e4 if r['layer']==l and r['split']=='validation' and r['horizon']==5]
        base=[r for r in rs if r['candidate']=='skip'];bg=balanced(base,'gripper_sign');candidates=[]
        for name in ('linear_r8','linear_r32'):
            rr=[r for r in rs if r['candidate']==name];candidates.append(dict(name=name,gripper=balanced(rr,'gripper_sign'),translation=balanced(rr,'translation_mm')))
        valid=[r for r in candidates if r['gripper']<=bg+1e-12]
        best=min(valid,key=lambda r:(r['translation'],r['gripper'],r['name'])) if valid else min(candidates,key=lambda r:(r['gripper'],r['translation'],r['name']))
        selected[l]=best['name'];selection.append(dict(layer=l,selected=best['name'],baseline_gripper=bg,passed_gripper_point_gate=bool(valid),candidates=candidates))
    dump(OUT/'rank_selection.json',dict(selection_split='validation_only',rows=selection,time=time.time()))
    # Fixed and selected variants remain separately available in the raw tables.
    summary=[];effects=[];names=('skip','channel','hadamard','reuse','extrapolate','linear_selected')
    for split in ('calibration','validation','heldout','all'):
        for l in LAYERS:
            for h in (5,50):
                for name in names:
                    actual=selected[l] if name=='linear_selected' else name
                    rs=[r for r in e4 if r['layer']==l and r['horizon']==h and r['candidate']==actual and (split=='all' or r['split']==split)]
                    summary.append(dict(split=split,layer=l,horizon=h,candidate=name,selected_rank=selected[l] if name=='linear_selected' else '',requests=len(rs),**{k:balanced(rs,k) for k in METRICS}))
                    if split=='heldout' and h==5 and name!='skip':
                        base={r['id']:r for r in e4 if r['layer']==l and r['horizon']==h and r['candidate']=='skip' and r['split']==split}
                        delta=[dict(suite=r['suite'],task=r['task'],**{k:r[k]-base[r['id']][k] for k in METRICS}) for r in rs]
                        for k in METRICS:
                            ci=paired_ci(delta,k);effects.append(dict(layer=l,candidate=name,metric=k,delta=balanced(delta,k),ci_low=ci[0],ci_high=ci[1]))
    write_csv(OUT/'E4_summary.csv',summary);write_csv(OUT/'E4_heldout_paired.csv',effects)
    history_summary=[]
    for l in LAYERS:
        for available in (False,True):
            for name in ('skip','reuse','extrapolate'):
                rs=[r for r in e4 if r['layer']==l and r['horizon']==5 and r['candidate']==name and r['split']=='heldout' and r['has_second_history']==available]
                if rs:history_summary.append(dict(layer=l,has_second_history=available,candidate=name,requests=len(rs),**{k:balanced(rs,k) for k in METRICS}))
    write_csv(OUT/'E4_history_subsets.csv',history_summary)
    hidden_summary=[]
    for l in LAYERS:
        for domain in ('full','shifted'):
            for name in ('skip','channel','hadamard','reuse','extrapolate','linear_r8','linear_r32','shiftlinear_r8','shiftlinear_r32'):
                rs=[r for r in hidden if r['layer']==l and r['domain']==domain and r['candidate']==name and r['split']=='heldout']
                hidden_summary.append(dict(layer=l,domain=domain,candidate=name,hidden_mse=balanced(rs,'hidden_mse'),hidden_relative_l2=balanced(rs,'hidden_relative_l2')))
    write_csv(OUT/'E4_hidden_summary.csv',hidden_summary)
    speed=[]
    for l in LAYERS:
        for name in ('full_refresh','skip','channel','hadamard','reuse','extrapolate','linear_r8','linear_r32'):
            rs=[r for r in timing if r['layer']==l and r['candidate']==name]
            speed.append(dict(layer=l,candidate=name,requests=len(rs),model_wall_ms=balanced(rs,'model_wall_ms')))
    write_csv(OUT/'E4_timing_summary.csv',speed)
    lines=['# v3.4 E2/E4：PyTorch 单层实验报告','',
        '仅执行新增E2/E4；复用真实观测，不重跑E1/E3，不开展多层选型或闭环。E2为40请求×17层×三档gate，E4为160请求×17层、五类补偿。所有主对照均使用相同FP32 PyTorch functional模型、原fused权重与固定视觉嵌入，D2/H50/R5。',
        '', '## E2：局部动作敏感度与实际单层跳过','',
        '|gate|动作组|向量相对预测误差|方向余弦|Jacobian排序ρ|层余弦排序ρ|残差范数排序ρ|预测低4层落入实际高4层的平均数|',
        '|---|---|---|---|---|---|---|---|']
    for r in e2summary:lines.append(f"|{r['gate']}|{r['group']}|{r['relative_vector_error']:.3f}|{r['direction_cosine']:.3f}|{r['jacobian_spearman']:.3f}|{r['cosine_spearman']:.3f}|{r['residual_spearman']:.3f}|{r['low4_in_actual_high4']:.3f}|")
    lines += ['',f"平均单次完整forward {np.mean([r['forward_ms'] for r in costs]):.2f}ms；完整35维Jacobian（含forward）{np.mean([r['jacobian_total_ms'] for r in costs]):.2f}ms。这是FP32诊断成本；随机投影表由完整J计算，仅衡量近似精度，不声称其已获得实测部署加速。",'',
        'gate=0与真实bypass逐请求17层核验通过；四suite真实输入有限差分检查通过。连续raw夹爪Jacobian与decoded符号差单列，局部线性或单层排序表现不解释为多层组合有效。',
        '', '## E4：留出任务上的单层补偿','',
        '校准：task0/2/4/6/8；选rank：task1/5；留出：task3/7/9。每suite相同划分，同轨迹不跨split。以下每行是单独跳过对应层，其他层Full；平移为控制命令差×50mm代理，不是真实EEF误差。',
        '', '|层|候选|H5平移mm代理|旋转L2|夹爪符号差|相对skip平移变化mm及探索95%区间|', '|---|---|---|---|---|---|']
    for l in LAYERS:
        for name in names:
            r=next(r for r in summary if r['split']=='heldout' and r['layer']==l and r['horizon']==5 and r['candidate']==name)
            eff=next((r for r in effects if r['layer']==l and r['candidate']==name and r['metric']=='translation_mm'),None)
            text='—' if eff is None else f"{eff['delta']:+.3f} [{eff['ci_low']:+.3f},{eff['ci_high']:+.3f}]"
            lines.append(f"|{l}|{name}|{r['translation_mm']:.3f}|{r['rotation_l2']:.4f}|{r['gripper_sign']:.2%}|{text}|")
    lines += ['', '## 输入偏移、历史与成本边界','',
        'E4_hidden_summary.csv对每个补偿分别报告Full输入和V14上游skip产生的偏移输入上的留出hidden误差，并比较仅Full输入拟合与实际偏移输入拟合的相同rank映射；这里没有执行多层动作策略。',
        'reuse/extrapolate使用Full参考轨迹上的历史，不是候选自身闭环缓存。最近Full年龄5步，第二历史间隔至少15步；缺第二历史回退reuse，逐请求has_second_history保留。不能把这些shadow质量或单次patch耗时解释为部署收益，周期Full刷新与轨迹分叉尚需后续闭环评估。',
        'E4_timing_summary.csv是四suite固定真实输入完整模型路径的热态AB/BA三次计时，包含相应patch，不含固定视觉嵌入提取、RPC/仿真。full_refresh包含该单层历史残差提取。科学比较中共享前缀只节省实验重算，没有计作候选部署加速。',
        'a/b匹配激活均值和标准差；linear为带固定ridge的闭式残差回归及随机截断SVD，rank8/32只在验证任务上选择。配对区间为suite分层task聚类bootstrap2000次，探索性且未校正多重比较；不能据本表宣称总体非劣或任务成功率提高。',
        '', '## 数值与复现','',
        '固定视觉嵌入由原BF16 native视觉路径导出；本轮可微VLM/AE使用FP32 PyTorch，保留原checkpoint的融合权重、RMSNorm epsilon、RoPE表、prefix KV shell与state注意力mask。engineering_probe.json报告与原BF16 native输出的差异，不拼接两后端的成功率或误差。原始完整H50动作、Jacobian、逐请求指标与校准参数均已保存；指标从原始数组重算验收。',
        '', '[冻结方案](idea1_v3_4_E2_E4_single_plan.md) · [最终验收](../../../results/idea1_v3_4_E2_E4_single/audit.json) · [E2汇总](../../../results/idea1_v3_4_E2_E4_single/E2_summary.csv) · [E4留出配对](../../../results/idea1_v3_4_E2_E4_single/E4_heldout_paired.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_E4_single_report.md';report.write_text('\n'.join(lines)+'\n')
    plots(e2summary,summary,hidden_summary)
    dump(OUT/'audit.json',dict(passed=True,source_hashes_match=True,plan_match=True,requests=dict(E2=40,E4=160),
        layers=17,E2_bypass_checks=680,raw_metrics_recomputed=True,rank_selection_split='validation_only',time=time.time()))
    dump(OUT/'complete.json',dict(report=str(report),report_sha256=sha(report),time=time.time()))
    dump(OUT/'status.json',dict(status='complete',time=time.time()))

def plots(e2,summary,hidden):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,group in zip(axes,('translation','rotation','gripper')):
        rs=[r for r in e2 if r['group']==group]
        for name in ('jacobian_spearman','cosine_spearman','residual_spearman'):ax.plot([1-r['gate'] for r in rs],[r[name] for r in rs],marker='o',label=name)
        ax.set_title(group);ax.set_xlabel('Gate reduction');ax.set_ylabel('Spearman vs actual single-layer effect');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'E2_rank.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,metric in zip(axes,METRICS[:3]):
        for name in ('skip','channel','hadamard','reuse','extrapolate','linear_selected'):
            rs=[r for r in summary if r['split']=='heldout' and r['horizon']==5 and r['candidate']==name]
            ax.plot([r['layer'] for r in rs],[r[metric] for r in rs],marker='.',label=name)
        ax.set_title('Heldout '+metric);ax.set_xlabel('Single bypassed layer');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'E4_heldout.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
