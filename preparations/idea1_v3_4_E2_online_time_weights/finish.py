import json,time
from collections import Counter
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,sha
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def collect(name):
    out={}
    for p in (OUT/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
        r=json.loads(p.read_text());assert r['valid'];out[tuple(r['case'][k] for k in ['suite','task','init'])]=(p.parent,r)
    return out
def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(a),success=sum(r['success'] for d,r in a.values())) for c in m['conditions'] for a in [collect(c['id'])]}
def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    data={c['id']:collect(c['id']) for c in m['conditions']};assert all(set(a)==keys for a in data.values())
    rows=[];requests=[];summaries=[];mask_counts=Counter();identity_checks=0
    for case in keys:
        values=[identity(data[c['id']][case][0]) for c in m['conditions']];assert all(v==values[0] for v in values);identity_checks+=len(values)
    for name,a in data.items():
        for case,(d,r) in a.items():
            assert r['manifest_hash']==m['manifest_hash'];rows.append(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],rpc_s=r['timing']['rpc_s']))
            previous=None
            for infer in [json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()]:
                skip=infer['route']['skip'];assert len(skip)==4 and len(set(skip))==4 and min(skip)>=1 and max(skip)<=17 and not any({i,i+1,i+2}<=set(skip) for i in range(1,16))
                mask_counts[(name,tuple(skip))]+=1
                requests.append(dict(condition=name,**r['case'],replan=infer['replan'],skip=str(skip),mask_changed=previous is not None and skip!=previous,rpc_ms=infer['rpc_ms'],**{k:v for k,v in infer['timing'].items() if k.endswith('_ms')}));previous=skip
        for suite in ['all',*SUITES]:
            rr=[r for r in rows if r['condition']==name and (suite=='all' or r['suite']==suite)];qq=[r for r in requests if r['condition']==name and (suite=='all' or r['suite']==suite)]
            wall=sum(r['wall_s'] for r in rr);actions=sum(r['actions'] for r in rr)
            summary=dict(condition=name,suite=suite,n=len(rr),success=sum(r['success'] for r in rr),mean_steps=actions/len(rr),mean_episode_wall_s=wall/len(rr),episode_per_hour=3600*len(rr)/wall,actions_per_second=actions/wall,requests=len(qq),mask_change_fraction=float(np.mean([q['mask_changed'] for q in qq if q['replan']>0])))
            for metric in ['rpc_ms','policy_wall_ms','policy_hot_ms','selector_total_ms','selector_hot_ms','selector_capture_ms','vision_ms','vlm_ms','ae_ms']:
                v=[q[metric] for q in qq];summary[metric+'_mean']=float(np.mean(v));summary[metric+'_p95']=float(np.percentile(v,95))
            summaries.append(summary)
    pairs=[]
    for an,bn in [('uniform_random','linear_random'),('uniform_fixed','exponential_fixed'),('uniform_random','exponential_fixed'),('uniform_random','uniform_fixed')]:
        aa,bb=data[an],data[bn];deltas=[dict(suite=k[0],task=k[1],delta=int(bb[k][1]['success'])-int(aa[k][1]['success'])) for k in keys];ci=paired_ci(deltas)
        pairs.append(dict(baseline=an,candidate=bn,recovered=sum(not aa[k][1]['success'] and bb[k][1]['success'] for k in keys),lost=sum(aa[k][1]['success'] and not bb[k][1]['success'] for k in keys),delta_pp=float(np.mean([d['delta'] for d in deltas])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    write_csv(OUT/'episodes.csv',rows);write_csv(OUT/'requests.csv',requests);write_csv(OUT/'summary.csv',summaries);write_csv(OUT/'paired.csv',pairs);write_csv(OUT/'mask_frequency.csv',[dict(condition=k[0],skip=str(k[1]),requests=v) for k,v in mask_counts.items()])
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')];assert gates and all(g['passed'] for g in gates)
    lines=['# E2 时间权重在线选层闭环结果','','40个匹配case/条件，160条新闭环；实时当前观测单次VJP，每次重规划重新选skip4/max2。D2/H50/R5，原生BF16执行，可微BF16/FP32 RMS选层。固定视觉输入微基准不代替此处完整策略延迟。','','|条件|Spatial|Object|Goal|Long|成功|平均步数|策略均值ms|策略热态ms|选层热态ms|平均episode秒|','|---|---|---|---|---|---|---|---|---|---|']
    for c in m['conditions']:
        ss={r['suite']:r for r in summaries if r['condition']==c['id']};a=ss['all']
        lines.append('|'+ '|'.join([c['id'],*[f"{ss[s]['success']}/10" for s in SUITES],f"{a['success']}/40",f"{a['mean_steps']:.1f}",f"{a['policy_wall_ms_mean']:.2f}",f"{a['policy_hot_ms_mean']:.2f}",f"{a['selector_hot_ms_mean']:.2f}",f"{a['mean_episode_wall_s']:.2f}"])+'|')
    lines+=['','|配对比较|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for x in pairs:lines.append(f"|{x['baseline']}→{x['candidate']}|{x['recovered']}/{x['lost']}|{x['delta_pp']:+.1f}|[{x['ci_low_pp']:+.1f},{x['ci_high_pp']:+.1f}]|")
    lines+=['','随机七维方向固定seed340710；fixed全+1。uniform为五步各0.2，linear为1:2:3:4:5归一化，exponential为1:2:4:8:16归一化。两候选分别与匹配七维方向的等权基准配对；跨方向对照同时改变时间和七维权重。','',f'同case四条件新建环境、构建前seed7，初态/BDDL/controller/prompt/首输入一致，完成{identity_checks}项身份核验。冷捕获计入策略均值与episode墙钟，另报扣除捕获的热态。完整RPC、分套件、P95、吞吐、mask变化率见summary.csv。服务预热/加载另见workers/*/state.json。','', '不与历史静态Jacobian mask成功率直接合并；未引入动作执行空档、延迟梯度或周期更新。区间按套件分层、task聚类bootstrap，为小批探索结果。','', '[冻结方案](idea1_v3_4_E2_online_time_weights_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_online_time_weights_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,identity_checks=identity_checks,requests=len(requests),source_hashes=True,masks=True,graph_eager_gate=True,time=time.time()));atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
