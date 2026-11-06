import json,time
import numpy as np
from .common import ROOT,OUT,OLD,NEW,sha,PLAN
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv
def main():
    m=json.loads((OUT/'manifest.json').read_text());assert sha(PLAN)==m['plan_sha256'] and all(sha(ROOT/p)==h for p,h in m['sources'].items())
    bad={tuple(c[k] for k in ('suite','task','init')) for c in m['cases']};keys=[tuple(c[k] for k in ('suite','task','init')) for c in m['original_cases']]
    data={};rows=[];checks=[]
    for name,mapping in m['mapping'].items():
        rs={}
        for k in keys:
            root=OUT if k in bad else __import__('pathlib').Path(mapping['root']);identifier=name if k in bad else mapping['id']
            d=root/identifier/'episodes'/k[0]/f'task{k[1]:02d}'/f'init{k[2]:02d}'
            assert (d/'complete.json').exists();r=json.loads((d/'result.json').read_text());assert r['valid'];rs[k]=(d,r)
            rows.append(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],source=str(d),repaired=k in bad,policy_wall_s=r['policy_wall_s']))
        data[name]=rs
    for k in keys:
        fd=data['full_d2'][k][0];fi=json.loads((fd/'identity.json').read_text());fr=json.loads((fd/'infer.jsonl').read_text().splitlines()[0])
        for name,rs in data.items():
            d=rs[k][0];i=json.loads((d/'identity.json').read_text());r=json.loads((d/'infer.jsonl').read_text().splitlines()[0])
            passed=r['input_hash']==fr['input_hash'] and all(i[f]==fi[f] for f in ('init_vector_sha256','bddl_sha256','prompt','controller','env_seed','settling'))
            checks.append(dict(case=dict(suite=k[0],task=k[1],init=k[2]),condition=name,passed=passed,source=str(d)))
    atomic(OUT/'identity_audit.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
    assert all(c['passed'] for c in checks),'Fresh reset input gate failed; inspect identity_audit.json'
    pairs=[]
    specs=[('old_'+n,'new_'+n) for n in ('cosine4','jacobian4','action_mse4')]+[('new_cosine4','new_'+n) for n in ('jacobian4','action_mse4')]
    for a,b in specs:
        aa,bb=data[a],data[b];delta=[dict(suite=k[0],task=k[1],delta=int(bb[k][1]['success'])-int(aa[k][1]['success'])) for k in keys];ci=paired_ci(delta)
        pairs.append(dict(baseline=a,candidate=b,delta_pp=100*np.mean([r['delta'] for r in delta]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],recovered=sum(not aa[k][1]['success'] and bb[k][1]['success'] for k in keys),lost=sum(aa[k][1]['success'] and not bb[k][1]['success'] for k in keys)))
    write_csv(OUT/'paired.csv',pairs);write_csv(OUT/'episodes.csv',rows)
    counts={name:sum(r['success'] for d,r in rs.values()) for name,rs in data.items()}
    lines=['# E2闭环：最多连续跳两层（经初始观测修复）','',
        '120条主实验已完成。原首观测验收发现4个Long case在条件间不一致；补跑这些case的全部8个新旧对照，共32条。各条件构造前固定seed7并新建环境，其他协议不变。原记录未删除，4case所有条件统一用修复数据，其余36case复用原数据。全部8条件×40case的首观测及初态/BDDL/prompt/controller身份逐位核对通过；没有放宽hash门。',
        '', '|指标|原mask|原成功（修复后）|max2 mask|新成功（修复后）|','|---|---|---|---|---|']
    oldm=json.loads((OLD/'manifest.json').read_text());newm=json.loads((NEW/'manifest.json').read_text())
    for n in ('cosine4','jacobian4','action_mse4'):lines.append(f"|{n}|{oldm['masks'][n]}|{counts['old_'+n]}/40|{newm['masks'][n]}|{counts['new_'+n]}/40|")
    lines+=['',f"同一修复矩阵内V14为{counts['v14']}/40，Full-D2为{counts['full_d2']}/40。",'',
        '|对照|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['', '主120条修复前原始计数：余弦10/40、Jacobian19/40、动作影响21/40；它们是执行记录，不应替代上表经相同首观测核对的结果。修复涉及的原旧对照也统一更新，不能混用原旧聚合数。',
        '这是针对既有结果提出约束后的探索性配对检查，非全新独立确认集。区间按suite分层task聚类bootstrap，未证明总体非劣，也不把单个mask输赢推广为指标普遍有效/失效。',
        '', '[原冻结方案](idea1_v3_4_E2_closedloop_max2_plan.md) · [技术修复方案](idea1_v3_4_E2_max2_reset_repair_plan.md) · [全部数据来源](../../../results/idea1_v3_4_E2_max2_repair/episodes.csv) · [身份验收](../../../results/idea1_v3_4_E2_max2_repair/identity_audit.json)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_closedloop_max2_report.md';report.write_text('\n'.join(lines)+'\n')
    for root in (OUT,NEW):
        atomic(root/'audit.json',dict(passed=True,identity_checks=320,repair_directory=str(OUT),counts=counts,time=time.time()))
        atomic(root/'complete.json',dict(report=str(report),repair_directory=str(OUT),time=time.time()))
        atomic(root/'status.json',dict(status='complete',repaired=True,counts=counts,time=time.time()))
    print(json.dumps(counts),flush=True)
if __name__=='__main__':main()
