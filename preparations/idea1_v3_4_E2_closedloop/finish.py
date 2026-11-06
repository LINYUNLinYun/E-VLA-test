import csv,json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,SUITES
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def results():
    m=json.loads((OUT/'manifest.json').read_text());allrows={}
    for cfg in m['conditions']:
        data={}
        for p in (OUT/cfg['id']/'episodes').glob('*/task*/init*/result.json'):
            if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
            r=json.loads(p.read_text());assert r['valid'] and r['manifest_hash']==m['manifest_hash']
            data[tuple(r['case'][k] for k in ('suite','task','init'))]=r
        allrows[cfg['id']]=data
    return m,allrows

def progress():
    m,data=results();return {name:dict(completed=len(rs),success=sum(r['success'] for r in rs.values())) for name,rs in data.items()}

def main():
    import hashlib
    m,data=results();expected={tuple(c[k] for k in ('suite','task','init')) for c in m['cases']}
    assert all(set(rs)==expected for rs in data.values())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in m['source_files'].items())
    rows=[];summary=[]
    for name,rs in data.items():
        rows.extend(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']) for r in rs.values())
        for suite in ('all',*SUITES):
            a=[r for k,r in rs.items() if suite=='all' or k[0]==suite]
            summary.append(dict(condition=name,suite=suite,n=len(a),success=sum(r['success'] for r in a),
                policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in a)/sum(r['actions'] for r in a),
                episodes_per_hour=3600*len(a)/sum(r['wall_s'] for r in a)))
    comparisons=[]
    for a,b in [('cosine4','action_mse4'),('cosine4','jacobian4'),('v14','cosine4'),('full_d2','cosine4')]:
        an=m['aliases'].get(a,a);bn=m['aliases'].get(b,b);aa,bb=data[an],data[bn]
        pairs=[dict(suite=k[0],task=k[1],delta=int(bb[k]['success'])-int(aa[k]['success'])) for k in sorted(expected)]
        ci=paired_ci(pairs)
        comparisons.append(dict(baseline=a,candidate=b,n=len(expected),delta_pp=100*np.mean([r['delta'] for r in pairs]),
            ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],recovered=sum(not aa[k]['success'] and bb[k]['success'] for k in expected),lost=sum(aa[k]['success'] and not bb[k]['success'] for k in expected)))
    write_csv(OUT/'episodes.csv',rows);write_csv(OUT/'summary.csv',summary);write_csv(OUT/'paired.csv',comparisons)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')]
    assert len(gates)==4 and all(g['passed'] for g in gates)
    lines=['# E2选层指标：留出闭环快速评测','',
        '四suite各5个留出奇数task、每task init19/20，40条/条件。校准只使用偶数task的既有E2数据；所有mask和评分规则在闭环前冻结。原BF16 Triton、D2/H50/R5，纯剪枝，无在线Jacobian或验证器。',
        '', '|条件|跳过层|Spatial|Object|Goal|Long|总成功|策略ms/执行动作|','|---|---|---|---|---|---|---|---|']
    for cfg in m['conditions']:
        g={r['suite']:r for r in summary if r['condition']==cfg['id']}
        lines.append('|'+ '|'.join([cfg['id'],str(cfg['skip']),*[f"{g[s]['success']}/{g[s]['n']}" for s in SUITES],f"{g['all']['success']}/40",f"{g['all']['policy_ms_per_action']:.3f}"])+'|')
    lines+=['','|基线→候选|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in comparisons:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['',f"相同mask别名：{m['aliases']}。",'',
        '区间为suite分层task聚类bootstrap探索区间，样本为快速评测而非官方50-init全量基准。不同选层方法得到的mask即使表现不同，也不能唯一归因于余弦指标本身，更不能推广为所有模型/预算的结论。校准成本与控制成本分开；本轮是离线全局静态选型，不是在线梯度路由。',
        '', '[冻结方案](idea1_v3_4_E2_closedloop_plan.md) · [逐case](../../../results/idea1_v3_4_E2_closedloop/episodes.csv) · [配对结果](../../../results/idea1_v3_4_E2_closedloop/paired.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_closedloop_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,cases_per_condition=40,conditions=len(data),sources_match=True,reference_gates=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))

if __name__=='__main__':main()
