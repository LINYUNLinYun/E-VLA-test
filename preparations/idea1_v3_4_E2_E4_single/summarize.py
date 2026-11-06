"""Readable results from audited tables; no new selection or inference."""
import csv,json,time
from .common import ROOT,OUT,sha,dump
from .finish import balanced,paired_ci,METRICS

def read(p,ints=(),floats=()):
    with p.open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for k in ints:r[k]=int(r[k])
        for k in floats:r[k]=float(r[k])
    return rows

def main():
    assert json.loads((OUT/'audit.json').read_text())['passed']
    assert json.loads((OUT/'matched_shift/audit.json').read_text())['passed']
    data=read(OUT/'E4_requests.csv',ints=('id','task','layer','horizon'),floats=METRICS)
    selected={r['layer']:r['selected'] for r in json.loads((OUT/'rank_selection.json').read_text())['rows']}
    summary=read(OUT/'E4_summary.csv',ints=('layer','horizon'),floats=METRICS)
    timing=read(OUT/'E4_timing_summary.csv',ints=('layer',),floats=('model_wall_ms',))
    names=('skip','channel','hadamard','reuse','extrapolate','linear_selected')
    table=[]
    for name in names:
        rs=[r for r in data if r['split']=='heldout' and r['horizon']==5 and r['candidate']==(selected[r['layer']] if name=='linear_selected' else name)]
        base={(r['id'],r['layer']):r for r in data if r['split']=='heldout' and r['horizon']==5 and r['candidate']=='skip'}
        delta=[dict(suite=r['suite'],task=r['task'],delta=r['translation_mm']-base[r['id'],r['layer']]['translation_mm']) for r in rs]
        ci=paired_ci(delta,'delta')
        good=0;better_translation=0
        for l in range(1,18):
            a=next(r for r in summary if r['split']=='heldout' and r['horizon']==5 and r['layer']==l and r['candidate']==name)
            b=next(r for r in summary if r['split']=='heldout' and r['horizon']==5 and r['layer']==l and r['candidate']=='skip')
            better_translation+=a['translation_mm']<b['translation_mm']
            good+=a['translation_mm']<b['translation_mm'] and a['rotation_l2']<b['rotation_l2'] and a['gripper_sign']<=b['gripper_sign']+1e-12
        cost=sum(next(r['model_wall_ms'] for r in timing if r['layer']==l and r['candidate']==(selected[l] if name=='linear_selected' else name)) for l in range(1,18))/17
        basecost=sum(r['model_wall_ms'] for r in timing if r['candidate']=='skip')/17
        table.append(dict(candidate=name,**{k:balanced(rs,k) for k in METRICS},translation_delta=balanced(delta,'delta'),translation_ci=ci,
            translation_better_layers=int(better_translation),all_three_better_layers=int(good),model_ms=cost,patch_extra_ms=cost-basecost))
    h=read(OUT/'matched_shift/heldout_summary.csv',ints=('layer',),floats=('hidden_mse','hidden_relative_l2'))
    robustness=[]
    for domain in ('full','shifted'):
        for rank in (8,32):
            for kind in ('linear','shiftlinear'):
                vals=[]
                for l in range(4,18):
                    a=next(r for r in h if r['layer']==l and r['domain']==domain and r['candidate']==f'{kind}_r{rank}')
                    b=next(r for r in h if r['layer']==l and r['domain']==domain and r['candidate']=='skip')
                    vals.append(a['hidden_mse']/b['hidden_mse'])
                robustness.append(dict(domain=domain,rank=rank,fit_input='full' if kind=='linear' else 'shifted',mse_ratio_to_no_patch=sum(vals)/len(vals)))
    dump(OUT/'readable_summary.json',dict(E4_layer_grid_macro=table,matched_input_robustness=robustness,time=time.time()))
    labels={'skip':'无补偿','channel':'通道配准','hadamard':'Hadamard配准','reuse':'历史残差复用','extrapolate':'残差外推','linear_selected':'闭式低秩（验证组选rank）'}
    lines=['# E4：五种单层补偿与输入分布检查','',
        '17层分别独立跳过，其余层Full；共160个真实观测，80校准、32验证、48留出，按完整task隔离。没有新闭环、多层动作策略或E1/E3重跑。',
        '', '以下是留出组中17个单层条件等权的描述性汇总，不是一条模型的闭环成绩，也没有用留出组选层。平移为前5步控制命令差代理，夹爪为相对同输入Full-D2的符号差。',
        '', '|方法|平移mm代理|旋转L2|夹爪差|平移降低的层数|平移/旋转降低且夹爪不增的层数|额外模型ms/请求|',
        '|---|---|---|---|---|---|---|']
    for r in table:lines.append(f"|{labels[r['candidate']]}|{r['translation_mm']:.3f}|{r['rotation_l2']:.4f}|{r['gripper_sign']:.2%}|{r['translation_better_layers']}/17|{r['all_three_better_layers']}/17|{r['patch_extra_ms']:+.3f}|")
    lines += ['', '上述成本为四suite固定真实输入的热态FP32 PyTorch完整VLM+两步AE/FM路径，固定视觉嵌入提取、RPC和仿真不在其中。未专门将KV shell的未用Q投影消除；它是功能参考后端的测量，不是优化过的部署kernel。历史方法仍需要周期Full刷新；full_refresh成本另有记录，不能将patch请求成本直接解释为部署加速。',
        '', '## 相同SVD种子的输入分布检查','',
        '同一Full轨迹观测分别经Full和V14上游跳层生成该层输入。以下仅统计受上游skip影响的L4–17，将每层hidden MSE除以该输入域的无补偿MSE后，再对14层等权；低于1表示改善。两组拟合的rank、ridge、随机SVD种子相同，只改变校准输入。',
        '', '|评估输入|rank|Full输入拟合 / 无补偿|偏移输入拟合 / 无补偿|','|---|---|---|---|']
    for domain in ('full','shifted'):
        for rank in (8,32):
            a=next(r for r in robustness if r['domain']==domain and r['rank']==rank and r['fit_input']=='full')
            b=next(r for r in robustness if r['domain']==domain and r['rank']==rank and r['fit_input']=='shifted')
            lines.append(f"|{domain}|{rank}|{a['mse_ratio_to_no_patch']:.3f}|{b['mse_ratio_to_no_patch']:.3f}|")
    lines += ['', '尚未受上游skip影响的L1–3为负对照：相同随机数下，两组拟合的均值与rank8/32线性算子一致，控制验收通过。原先不同种子的诊断原样保存，主解释使用matched_shift结果。',
        '', '这是固定观测上的hidden输入鲁棒性检查；没有在V14自身闭环状态分布上评测，也不能从hidden误差推出多层动作或任务成功率。历史残差同样来自Full参考历史，不是候选自身闭环缓存。',
        '', '![留出任务的逐层动作差](../../../results/idea1_v3_4_E2_E4_single/E4_heldout.png)',
        '', '[共同冻结方案](idea1_v3_4_E2_E4_single_plan.md) · [逐层完整结果](idea1_v3_4_E2_E4_single_report.md) · [主验收](../../../results/idea1_v3_4_E2_E4_single/audit.json) · [匹配随机数控制](../../../results/idea1_v3_4_E2_E4_single/matched_shift/audit.json)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E4_single_report.md';report.write_text('\n'.join(lines)+'\n')
    e2=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_single_report.md'
    note='\n排序和向量相对预测误差在normalized raw动作空间内计算；decoded平移/旋转/夹爪指标逐请求另存，不把normalized排序ρ当成闭环成功率预测。\n'
    if '排序和向量相对预测误差在normalized' not in e2.read_text():e2.write_text(e2.read_text()+note)
    detail=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_E4_single_report.md'
    detail.write_text(detail.read_text()+'\n输入分布检查请以相同SVD种子的[独立控制汇总](idea1_v3_4_E4_single_report.md)为主，原不同种子指标保留。\n')
    status=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_E4_single_execution_status.md'
    status.write_text('# v3.4 E2/E4 单层实验：已完成\n\nE2：40请求×17层×三档gate，680项bypass核验通过。E4：160请求×17层、五种补偿，80/32/48按task隔离校准/验证/留出；原始数组重算通过。相同SVD随机数的输入偏移检查及L1–3负对照通过。\n\n未重跑E1/E3，未开展多层动作策略或新闭环。\n\n[E2单层结果](idea1_v3_4_E2_single_report.md) · [E4单层结果](idea1_v3_4_E4_single_report.md) · [完整逐层报告](idea1_v3_4_E2_E4_single_report.md)\n')
    dump(OUT/'complete.json',dict(reports={str(p.relative_to(ROOT)):sha(p) for p in (detail,e2,report,status)},
        main_audit_sha256=sha(OUT/'audit.json'),matched_shift_audit_sha256=sha(OUT/'matched_shift/audit.json'),time=time.time()))
    dump(OUT/'status.json',dict(status='complete',E2_requests=40,E4_requests=160,matched_shift_control='passed',time=time.time()))
    print(json.dumps(dict(E4=table,robustness=robustness),indent=2))

if __name__=='__main__':main()
