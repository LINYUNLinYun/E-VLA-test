import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OLD,OUT,PLAN,SUITES,casekey,sha
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def ledgers():return [json.loads(p.read_text()) for p in (OUT/'case_done').glob('*.json')]

def progress():
    ls=ledgers();counts={};reused=0
    for group in ls:
        for r in group['records']:
            c=counts.setdefault(r['condition'],dict(completed=0,success=0));c['completed']+=1;c['success']+=int(r['success']);reused+=int(r['reused'])
    return dict(verified_cases=len(ls),completed_records=sum(r['completed'] for r in counts.values()),reused_records=reused,conditions=counts)

def main():
    m=json.loads((OUT/'manifest.json').read_text());ls=ledgers();assert len(ls)==200
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    assert sha(OLD/'adapters.pt')==m['adapter_sha256']
    assert all(sha(p)==h for p,h in m['timing_files'].items())
    data={c['id']:{} for c in m['conditions']};flat=[];route_checks=0;new_service_pids={}
    for group in ls:
        assert len(group['records'])==10
        hashes=[];vectors=[]
        for record in group['records']:
            d=Path(record['directory']);r=json.loads((d/'result.json').read_text());cfg=r['config'];name=cfg['id'];key=tuple(r['case'][k] for k in ('suite','task','init'))
            identity=json.loads((d/'identity.json').read_text());assert r['valid'] and cfg==next(c for c in m['conditions'] if c['id']==name)
            assert identity['protocol']['adapter_sha256']==m['adapter_sha256']
            rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()];hashes.append(rows[0]['input_hash']);vectors.append(identity['init_vector_sha256'])
            full=[]
            for index,row in enumerate(rows):
                expected_full=cfg['mode']=='periodic' and index%3==0
                assert row['route']['refresh']==expected_full and row['round']==index
                assert len(row['timing']['nodes'])==(11 if expected_full else 3)
                if expected_full:full.append(row['step'])
                if cfg['mode']=='periodic' and not expected_full:
                    assert row['route']['history_steps']==full[-2:]
                    assert row['full_cache_age']==row['step']-full[-1]
                route_checks+=1
            if record['reused']:
                receipt=m['reuse'][casekey(r['case'])][name]
                assert all(sha(d/n)==h for n,h in receipt['hashes'].items())
            else:
                assert r['manifest_hash']==m['manifest_hash']
                new_service_pids.setdefault(key,set()).add(identity['protocol']['server_pid'])
            r.update(directory=str(d),reused=record['reused']);data[name][key]=r
            flat.append(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],
                rpc_s=r['timing']['rpc_s'],full_requests=r['counts']['full'],draft_requests=r['counts']['draft'],reused=r['reused'],directory=str(d)))
        assert len(set(hashes))==len(set(vectors))==1
    assert all(len(v)==200 for v in data.values()) and all(len(p)==1 for p in new_service_pids.values())
    summary=[];comparisons=[]
    specs=[('periodic_v12','periodic_'+k) for k in ('channel','hadamard','linear','reuse','extrapolate')]
    specs += [('v12_'+k,'periodic_'+k) for k in ('channel','hadamard','linear')]
    specs += [('pure_v12','v12_'+k) for k in ('channel','hadamard','linear')]+[('pure_v12',k) for k in ('periodic_v12','periodic_reuse','periodic_extrapolate')]
    for cohort in ('all200','new160','pilot40'):
        selected=lambda k:cohort=='all200' or (k[2]!=19 if cohort=='new160' else k[2]==19)
        for name,rs in data.items():
            for suite in ('all',*SUITES):
                a=[r for k,r in rs.items() if selected(k) and (suite=='all' or k[0]==suite)]
                summary.append(dict(condition=name,cohort=cohort,suite=suite,n=len(a),success=sum(r['success'] for r in a),
                    policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in a)/sum(r['actions'] for r in a),
                    rpc_ms_per_request=1000*sum(r['timing']['rpc_s'] for r in a)/sum(r['predictions'] for r in a),
                    episode_wall_s=sum(r['wall_s'] for r in a)/len(a),actions_per_second=sum(r['actions'] for r in a)/sum(r['wall_s'] for r in a)))
        for baseline,candidate in specs:
            a,b=data[baseline],data[candidate];ks=[k for k in sorted(a) if selected(k)]
            pairs=[dict(suite=k[0],task=k[1],delta=int(b[k]['success'])-int(a[k]['success'])) for k in ks];ci=paired_ci(pairs)
            comparisons.append(dict(cohort=cohort,baseline=baseline,candidate=candidate,n=len(ks),
                delta_pp=100*np.mean([r['delta'] for r in pairs]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],
                recovered=sum(not a[k]['success'] and b[k]['success'] for k in ks),lost=sum(a[k]['success'] and not b[k]['success'] for k in ks)))
    timing=[]
    for path in m['timing_files']:
        rows=json.loads(Path(path).read_text())['rows'];mean=lambda n:float(np.mean([r['policy_wall_ms'] for r in rows if r['path']==n]))
        costs=dict(pure_v12=mean('v12_d2'),**{'v12_'+k:mean(k) for k in ('channel','hadamard','linear')})
        costs.update(periodic_v12=(mean('full10')+2*mean('v12_d2'))/3,
            periodic_reuse=(mean('full10_capture')+2*mean('reuse'))/3,
            periodic_extrapolate=(mean('full10_capture')+mean('extrapolate5')+mean('extrapolate10'))/3)
        costs.update({'periodic_'+k:(mean('full10')+2*mean(k))/3 for k in ('channel','hadamard','linear')})
        for name,value in costs.items():timing.append(dict(worker=Path(path).parent.name,condition=name,component_estimate_ms=value,speedup_vs_full=mean('full10')/value))
    write_csv(OUT/'episodes.csv',flat);write_csv(OUT/'summary.csv',summary);write_csv(OUT/'paired.csv',comparisons);write_csv(OUT/'deployment_estimate.csv',timing)
    lines=['# V12十条件：200条扩样与静态补偿周期化','',
        'mask=[2,3,5,14,16,17]，原π0 BF16 Triton，H50/R5；周期为Full-D10→两次V12-D2。四suite各10task、每task init19–23。补偿参数及rank8完全冻结。',
        '', '|条件|原init19|新增init20–23|合计|Spatial|Object|Goal|Long|组件摊销ms/请求|','|---|---|---|---|---|---|---|---|---|']
    for cfg in m['conditions']:
        name=cfg['id'];get=lambda cohort,s='all':next(r for r in summary if r['condition']==name and r['cohort']==cohort and r['suite']==s)
        t=np.mean([r['component_estimate_ms'] for r in timing if r['condition']==name])
        lines.append('|'+ '|'.join([name,f"{get('pilot40')['success']}/40",f"{get('new160')['success']}/160",f"{get('all200')['success']}/200",
            *[f"{get('all200',s)['success']}/50" for s in SUITES],f'{t:.3f}'])+'|')
    lines+=['','|基线→候选（全部200）|恢复/新增失败|成功率差pp|探索95%区间pp|','|---|---|---|---|']
    for r in comparisons:
        if r['cohort']=='all200':lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','以上区间为suite分层task聚类bootstrap2000次；五init作为task内重复，不按200个完全独立样本计算。未校正多重比较，200条不是官方50-init全量。原40和新增160的配对结果分别保存在paired.csv。',
        '', f"共2000条有效记录，其中复用旧结果{sum(r['reused'] for r in flat)}条，所有200个case十条件首观测及初态向量一致。旧结果只在新条件首观测匹配后接受，拒绝复用的case见reuse_rejected目录。模型/缓存/参数与源码核验通过。",'',
        '同GPU模型服务串行B1，两个独立环境客户端按task分组交叠运行；四suite交错/独立数值门禁通过才启用双客户端，否则使用单客户端。新条件在Full轮不提取静态方法不需要的残差。历史缓存始终来自该候选自身路径。',
        '', '组件摊销延迟复用前轮2160次独占模型测量，新静态周期按(Full+2×静态draft)/3估计，不声称新跑端到端微基准。实际策略wall/动作、RPC含队列、回合墙钟、动作/s见summary.csv；新增160的运行基础设施一致，原40包含历史单环境结果，分析运行成本优先使用new160。',
        '', '[冻结方案](idea1_v3_4_v12_compensation_200_plan.md) · [逐case](../../../results/idea1_v3_4_v12_compensation_200/episodes.csv) · [配对比较](../../../results/idea1_v3_4_v12_compensation_200/paired.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_v12_compensation_200_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,conditions=10,cases_per_condition=200,initial_observation_cases=200,route_checks=route_checks,
        reused=sum(r['reused'] for r in flat),sources_match=True,adapter_match=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))

if __name__=='__main__':main()
