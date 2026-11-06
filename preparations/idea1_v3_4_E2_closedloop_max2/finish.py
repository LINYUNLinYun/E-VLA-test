import json,time
import numpy as np
from .common import ROOT,OUT,OLD,PLAN,SUITES,sha,legal
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def collect(root,name):
    data={}
    for p in (root/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
        r=json.loads(p.read_text());assert r['valid']
        data[tuple(r['case'][k] for k in ('suite','task','init'))]=(p.parent,r)
    return data

def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(rs),success=sum(r['success'] for d,r in rs.values())) for c in m['conditions'] for rs in [collect(OUT,c['id'])]}

def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ('suite','task','init')) for c in m['cases']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    assert sha(OLD/'manifest.json')==m['original_manifest_sha256']
    data={c['id']:collect(OUT,c['id']) for c in m['conditions']};historical={c['id']:collect(OLD,c['id']) for c in json.loads((OLD/'manifest.json').read_text())['conditions']}
    rows=[];summary=[];checks=0
    for name,rs in data.items():
        assert set(rs)==keys and legal(m['masks'][name])
        for k,(d,r) in rs.items():
            assert r['manifest_hash']==m['manifest_hash']
            od,old=historical['full_d2'][k]
            identity=json.loads((d/'identity.json').read_text());reference=json.loads((od/'identity.json').read_text())
            assert all(identity[f]==reference[f] for f in ('init_vector_sha256','bddl_sha256','prompt','env_seed','settling','controller'))
            a=json.loads((d/'infer.jsonl').read_text().splitlines()[0]);b=json.loads((od/'infer.jsonl').read_text().splitlines()[0]);assert a['input_hash']==b['input_hash'];checks+=1
            rows.append(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']))
        for suite in ('all',*SUITES):
            rr=[r for k,(d,r) in rs.items() if suite=='all' or k[0]==suite]
            summary.append(dict(condition=name,suite=suite,n=len(rr),success=sum(r['success'] for r in rr),policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in rr)/sum(r['actions'] for r in rr)))
    comparisons=[]
    pairspec=[(name+'_old',historical[name],name,rs) for name,rs in data.items()]
    pairspec += [('cosine4',data['cosine4'],name,data[name]) for name in ('jacobian4','action_mse4')]
    pairspec += [(name,historical[name],candidate,rs) for name in ('v14','full_d2') for candidate,rs in data.items()]
    for an,aa,bn,bb in pairspec:
        assert set(aa)==keys and set(bb)==keys
        pairs=[dict(suite=k[0],task=k[1],delta=int(bb[k][1]['success'])-int(aa[k][1]['success'])) for k in sorted(keys)];ci=paired_ci(pairs)
        comparisons.append(dict(baseline=an,candidate=bn,delta_pp=100*np.mean([r['delta'] for r in pairs]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],
            recovered=sum(not aa[k][1]['success'] and bb[k][1]['success'] for k in keys),lost=sum(aa[k][1]['success'] and not bb[k][1]['success'] for k in keys)))
    write_csv(OUT/'episodes.csv',rows);write_csv(OUT/'summary.csv',summary);write_csv(OUT/'paired.csv',comparisons)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')]
    assert gates and all(g['passed'] for g in gates)
    lines=['# E2闭环：最多连续跳两层结果','',
        '固定原校准分数，在四层预算下禁止连续3层；合法组合中最小化原平均秩总和。40个原留出case/条件，共120条新闭环。π0 BF16 Triton、D2/H50/R5及其他协议不变。',
        '', '|方法|原mask|原成功|max2 mask|Spatial|Object|Goal|Long|新成功|', '|---|---|---|---|---|---|---|---|---|']
    oldm=json.loads((OLD/'manifest.json').read_text())
    for name in data:
        g={r['suite']:r for r in summary if r['condition']==name}
        lines.append('|'+ '|'.join([name,str(oldm['masks'][name]),f"{sum(r['success'] for d,r in historical[name].values())}/40",str(m['masks'][name]),*[f"{g[s]['success']}/10" for s in SUITES],f"{g['all']['success']}/40"])+'|')
    lines+=['','|配对对照|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in comparisons:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['', '历史V14为26/40，Full-D2为40/40；服务原生参考与逐case首观测/初态/BDDL/prompt身份核对后复用。',
        '本轮没有根据成绩调分数或挑mask。区间为suite分层task聚类bootstrap探索区间，小批结果不解释为总体非劣或某指标在所有模型/预算上失效。',
        '', '[冻结方案](idea1_v3_4_E2_closedloop_max2_plan.md) · [配对CSV](../../../results/idea1_v3_4_E2_closedloop_max2/paired.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_closedloop_max2_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,identity_checks=checks,case_keys=True,source_hashes=True,mask_constraint=True,reference_gates=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))

if __name__=='__main__':main()
