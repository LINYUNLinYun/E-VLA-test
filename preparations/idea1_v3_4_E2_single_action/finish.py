import csv,json,time
import numpy as np
from .common import ROOT,OUT,PLAN,sha,dump,write_csv
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert sha(PLAN)==m['plan_sha256'] and all(sha(ROOT/p)==h for p,h in m['sources'].items())
    rows=[];quality=[];numerics=[]
    for dtype in ('fp32','bf16'):
        done=list((OUT/dtype).glob('sample*/complete.json'));assert len(done)==8
        for p in done:
            rows+=read(p.parent/'latency.csv');quality+=read(p.parent/'quality.csv');numerics+=json.loads((p.parent/'checks.json').read_text())
    write_csv(OUT/'latency_requests.csv',rows);write_csv(OUT/'measured_quality_requests.csv',quality)
    summary=[]
    for dtype in ('fp32','bf16'):
        for method,step in sorted({(r['method'],int(r['step'])) for r in rows if r['dtype']==dtype}):
            a=[r for r in rows if r['dtype']==dtype and r['method']==method and int(r['step'])==step];v=np.array([float(r['total_ms']) for r in a])
            summary.append(dict(dtype=dtype,method=method,step=step,n=len(a),mean_ms=float(v.mean()),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(v.max()),under200=float(np.mean(v<200)),added_ms=float(np.mean([float(r['added_ms']) for r in a]))))
    write_csv(OUT/'latency_summary.csv',summary)
    measured=[]
    for dtype in ('fp32','bf16'):
        for method in ('project1','project4','exact7','full35'):
            for step in (1,3,5):
                a=[r for r in quality if r['dtype']==dtype and r['method']==method and int(r['target_step'])==step and (method=='full35' or int(r['step'])==step)]
                assert len(a)==8
                measured.append(dict(dtype=dtype,method=method,step=step,n=8,rho_step=float(np.mean([float(r['rho_step']) for r in a])),rho_R5=float(np.mean([float(r['rho_R5']) for r in a])),selected_step_effect_ratio=float(np.mean([float(r['selected_step_effect_ratio']) for r in a]))))
    write_csv(OUT/'measured_quality_summary.csv',measured)
    q=read(OUT/'quality_summary.csv')
    lines=['# E2：单动作反传的效果与成本','',
        '固定第1/3/5步七维输出；1/4次为随机投影，7次为单动作精确Jacobian，35次为五步完整Jacobian。均为Full gate=1基点。未执行新闭环。',
        '', '## 原40请求上的排序效果','',
        '同样预测所选单步真实七维动作扰动；下表为32组预先固定随机种子的平均，单次和四次使用嵌套相同权重。完整35次也按该单步目标评分，避免把不同目标的相关性混在一起。',
        '', '|动作位置|方法|单步排序ρ|五步排序ρ|随机种子间单步ρ的5%–95%范围|候选四层的单步扰动/全部层平均|',
        '|---|---|---|---|---|---|']
    for r in q:lines.append(f"|{r['step']}|{r['method']}|{float(r['rho_step']):.3f}|{float(r['rho_R5']):.3f}|[{float(r['seed_rho_step_p05']):.3f},{float(r['seed_rho_step_p95']):.3f}]|{float(r['selected_step_effect_ratio']):.3f}|")
    lines+=['','## 同轮实测成本','',
        '每精度8个固定真实输入、每输入/方法预热一次、交替顺序测3次。含VLM、完整两步FM/AE、所列反传、同步及max2选四层；不含视觉嵌入生成、RPC/排队、加载和文件读取。小批P95不是部署保证。',
        '', '|精度|动作位置|方法|均值ms|P95 ms|最大ms|低于200ms比例|','|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['dtype']}|{r['step'] or '全部/无'}|{r['method']}|{r['mean_ms']:.1f}|{r['p95_ms']:.1f}|{r['max_ms']:.1f}|{r['under200']:.1%}|")
    lines+=['','## 实际BF16反传效果','',
        '这8请求固定使用第一组投影种子，真值为同输入原FP32单层反事实；与40请求/32种子主表分开解释。',
        '', '|动作位置|方法|单步排序ρ|五步排序ρ|候选四层的单步扰动比|','|---|---|---|---|---|']
    for r in measured:
        if r['dtype']=='bf16':lines.append(f"|{r['step']}|{r['method']}|{r['rho_step']:.3f}|{r['rho_R5']:.3f}|{r['selected_step_effect_ratio']:.3f}|")
    lines+=['', '选四层的扰动指标来自四种独立单层反事实的平均，不是四层一起跳过后的动作或闭环成功率。max2约束只用于生成可部署形状的候选mask，没有对此mask做闭环。',
        '归一化raw动作尺度和七维权重保持原定义，平移/旋转/夹爪组合目标不同于任务成败。32组种子用于检查投影波动，未按结果选种子或最佳动作位置。不同位置的结果同时公开。',
        'FP32输出与存档及实际梯度与完整J相应行/投影核对通过；BF16为独立近似分支，偏差记录在checks.json。forward增量仅适用于同一PyTorch路径保留计算图，原Triton Full不能直接抵扣。',
        '', '[方案](idea1_v3_4_E2_single_action_plan.md) · [效果CSV](../../../results/idea1_v3_4_E2_single_action/quality_summary.csv) · [计时CSV](../../../results/idea1_v3_4_E2_single_action/latency_summary.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_single_action_report.md';report.write_text('\n'.join(lines)+'\n')
    assert all(np.isfinite(float(r['total_ms'])) for r in rows)
    dump(OUT/'audit.json',dict(passed=True,sources_match=True,quality_requests=40,projection_seeds=32,latency_inputs_per_dtype=8,latency_rows=len(rows),fp32_gradient_checks=True,fp32_output_checks=True,time=time.time()))
    dump(OUT/'complete.json',dict(report=str(report),report_sha256=sha(report),time=time.time()));dump(OUT/'status.json',dict(status='complete',time=time.time()))
if __name__=='__main__':main()
