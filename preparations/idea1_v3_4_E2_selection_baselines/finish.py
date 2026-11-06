import json,time
from collections import Counter
import numpy as np
from .common import ROOT,OUT,REFERENCE,PLAN,SUITES,sha
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def collect(root,name):
    out={}
    for p in (root/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
        r=json.loads(p.read_text());assert r['valid'];out[tuple(r['case'][k] for k in ['suite','task','init'])]=(p.parent,r)
    return out
def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(a),success=sum(r['success'] for d,r in a.values())) for c in m['conditions'] for a in [collect(OUT,c['id'])]}
def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256'] and sha(REFERENCE/'manifest.json')==m['reference_manifest_sha256']
    data={c['id']:collect(OUT,c['id']) for c in m['conditions']};data['jacobian']=collect(REFERENCE,'exponential_fixed');assert all(set(a)==keys for a in data.values())
    for case in keys:
        values=[identity(a[case][0]) for a in data.values()];assert all(v==values[0] for v in values)
    episodes=[];requests=[];summary=[]
    for name,a in data.items():
        for case,(d,r) in a.items():
            episodes.append(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']))
            previous=None
            for infer in [json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()]:
                skip=infer['route']['skip'];assert len(skip)==4 and min(skip)>=1 and max(skip)<=17 and not any({i,i+1,i+2}<=set(skip) for i in range(1,16))
                requests.append(dict(condition=name,**r['case'],replan=infer['replan'],skip=str(skip),mask_changed=previous is not None and skip!=previous,rpc_ms=infer['rpc_ms'],**{k:v for k,v in infer['timing'].items() if k.endswith('_ms')}));previous=skip
        for suite in ['all',*SUITES]:
            ee=[r for r in episodes if r['condition']==name and (suite=='all' or r['suite']==suite)];qq=[r for r in requests if r['condition']==name and (suite=='all' or r['suite']==suite)];hot=[q for q in qq if q['replan']>0]
            wall=sum(r['wall_s'] for r in ee);acts=sum(r['actions'] for r in ee)
            row=dict(condition=name,suite=suite,n=len(ee),success=sum(r['success'] for r in ee),mean_steps=acts/len(ee),mean_episode_wall_s=wall/len(ee),episode_per_hour=3600*len(ee)/wall,actions_per_second=acts/wall,requests=len(qq),hot_requests=len(hot),mask_change_fraction=float(np.mean([q['mask_changed'] for q in hot])))
            for metric in ['policy_wall_ms','selector_hot_ms','rpc_ms']:
                v=[q[metric] for q in hot];row[metric+'_mean']=float(np.mean(v));row[metric+'_p95']=float(np.percentile(v,95))
            row['all_request_policy_ms_mean']=float(np.mean([q['policy_wall_ms'] for q in qq]));summary.append(row)
    comparisons=[]
    baselines=[c['id'] for c in m['conditions']]
    for name in baselines+['random_online_seed_mean']:
        def baseline_success(k):
            return float(np.mean([data[f'random_online_s{i}'][k][1]['success'] for i in [1,2,3]])) if name=='random_online_seed_mean' else float(data[name][k][1]['success'])
        deltas=[dict(suite=k[0],task=k[1],delta=float(data['jacobian'][k][1]['success'])-baseline_success(k)) for k in keys];ci=paired_ci(deltas)
        comparisons.append(dict(baseline=name,candidate='jacobian',baseline_rate=float(np.mean([baseline_success(k) for k in keys])),candidate_rate=.5,delta_pp=float(np.mean([r['delta'] for r in deltas])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100,recovered=None if name.endswith('_mean') else sum(not data[name][k][1]['success'] and data['jacobian'][k][1]['success'] for k in keys),lost=None if name.endswith('_mean') else sum(data[name][k][1]['success'] and not data['jacobian'][k][1]['success'] for k in keys)))
    write_csv(OUT/'episodes.csv',episodes);write_csv(OUT/'requests.csv',requests);write_csv(OUT/'summary.csv',summary);write_csv(OUT/'paired.csv',comparisons)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/cosine_gate.json')];assert gates and all(g['passed'] for g in gates)
    lines=['# E2 Jacobian与随机/余弦闭环对照','','Jacobian指数后重+七维正等权复用已完成40条；新余弦、三随机种子、单种子回合固定随机共200条。均为动态V14预算、D2/H50/R5、skip4/max2。新旧场景初态/首观测身份一致。热态统一排除每回合首请求；冷捕获计入完整episode墙钟。','','|方法|Spatial|Object|Goal|Long|成功|选层热态ms|完整策略热态ms|平均episode秒|','|---|---|---|---|---|---|---|---|---|']
    for name in ['jacobian',*baselines]:
        ss={r['suite']:r for r in summary if r['condition']==name};a=ss['all'];lines.append('|'+ '|'.join([name,*[f"{ss[s]['success']}/10" for s in SUITES],f"{a['success']}/40",f"{a['selector_hot_ms_mean']:.3f}",f"{a['policy_wall_ms_mean']:.2f}",f"{a['mean_episode_wall_s']:.2f}"])+'|')
    lines+=['','|基准→Jacobian|成功率差pp|探索95%区间pp|恢复/新增失败|','|---|---|---|---|']
    for r in comparisons:lines.append(f"|{r['baseline']}→Jacobian|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|{r['recovered']}/{r['lost']}|")
    lines+=['','随机在线种子341101/341102/341103均匀采样合法四层集合，每次重规划重新选；回合固定随机与online_s1共享首mask，后续不更换。余弦是当前Full VLM逐token输入/输出cosine的层均值，使用无梯度BF16/FP32 RMS encode+CUDA Graph，取高相似层。每个方法均计入自身必要成本，不为便宜基准额外运行Jacobian。','', '三随机种子均公开；随机均值先在同case内平均，bootstrap按suite分层task聚类，不将120轨迹当120独立case。无显著性导向扩样。Jacobian候选是前轮四方案中选出的最佳者，因此同批重复比较仍是探索证据；泛化结论需要新场景。','', '原始FP32完整J/单层删除反事实对照见offline_summary.csv：仅衡量独立删除层的扰动，不能代替四层联合删除或闭环成败。旧离线固定余弦10/40不混入本次实时余弦比较。','', '[冻结方案](idea1_v3_4_E2_selection_baselines_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_selection_baselines_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,identity_checks=len(keys)*len(data),source_hashes=True,reference_reused=True,requests=len(requests),time=time.time()));atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
