import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,sha,key,directory
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

NAMES=[f'{model}_p{p}' for model in ('v12','prefix3') for p in (2,4,6)]
def progress():
    rows={}
    for name in ('v12_p4','v12_p6'):
        results=[json.loads(p.read_text()) for p in (OUT/name/'episodes').glob('*/task*/init19/result.json') if (p.parent/'complete.json').exists()]
        rows[name]=dict(completed=len(results),success=sum(r['success'] for r in results))
    return rows

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items());assert sha(PLAN)==m['plan_sha256']
    assert len(list((OUT/'case_done').glob('*.json')))==40
    data={n:{} for n in NAMES};episodes=[];route_checks=0;prefix_checks=[]
    for c in m['cases']:
        k=key(c);done=json.loads((OUT/'case_done'/f'{k}.json').read_text());sources={};pids=[];initial=[];vectors=[]
        for cfg in m['conditions']:sources[cfg['id']]=directory(cfg['id'],c)
        sources['v12_p2']=Path(done['reference_directory'])
        sources.update({n:Path(ref['directory']) for n,ref in m['prefix_references'][k].items()})
        for name,d in sources.items():
            r=json.loads((d/'result.json').read_text());identity=json.loads((d/'identity.json').read_text());assert r['valid'] and r['case']==c
            rows=[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()]
            initial.append(rows[0]['input_hash']);vectors.append(identity['init_vector_sha256'])
            if name in ('v12_p4','v12_p6'):
                pids.append(identity['protocol']['server_pid']);assert r['manifest_hash']==m['manifest_hash'];period=int(name[-1]);last=None
                for i,row in enumerate(rows):
                    full=i%(period+1)==0;assert row['route']['refresh']==full and row['round']==i
                    assert len(row['timing']['nodes'])==(11 if full else 3)
                    if full:last=row['step'];assert row['route']['replayed_layers']==[]
                    else:
                        assert row['route']['replayed_layers']==m['reference_config']['skip']
                        assert row['route']['history_step']==last and row['full_cache_age']==row['step']-last
                    route_checks+=1
            else:
                ref=m['references'][k] if name=='v12_p2' else m['prefix_references'][k][name]
                if name!='v12_p2' or done['reference_reused']:assert all(sha(d/n)==h for n,h in ref['hashes'].items())
            data[name][k]=dict(result=r,directory=d)
            episodes.append(dict(condition=name,**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],rpc_s=r['timing']['rpc_s'],directory=str(d),reused=name.startswith('prefix3') or (name=='v12_p2' and done['reference_reused'])))
        assert len(set(initial))==len(set(vectors))==len(set(pids))==1
        with np.load(sources['v12_p2']/'arrays.npz') as a,np.load(sources['v12_p4']/'arrays.npz') as b,np.load(sources['v12_p6']/'arrays.npz') as cc:
            for name,x,y,n in [('p2_p4',a,b,3),('p2_p6',a,cc,3),('p4_p6',b,cc,5)]:
                keys=[f'{kind}_{i}' for kind in ('actions','raw') for i in range(n)]
                assert all(z in x and z in y for z in keys)
                passed=all(np.allclose(x[z],y[z],atol=1e-5,rtol=1e-5) for z in keys)
                prefix_checks.append(dict(case=c,comparison=name,requests=n,passed=passed,exact=all(np.array_equal(x[z],y[z]) for z in keys),max_abs=max(float(np.max(np.abs(x[z]-y[z]))) for z in keys)))
                assert passed
    paired=[]
    comparisons=[('v12_p2','v12_p4'),('v12_p2','v12_p6'),('v12_p4','v12_p6'),('prefix3_p2','prefix3_p4'),('prefix3_p2','prefix3_p6')]+[(f'prefix3_p{p}',f'v12_p{p}') for p in (2,4,6)]
    for baseline,candidate in comparisons:
        a,b=data[baseline],data[candidate];pairs=[];recovered=lost=0
        for c in m['cases']:
            k=key(c);av=bool(a[k]['result']['success']);bv=bool(b[k]['result']['success'])
            pairs.append(dict(suite=c['suite'],task=c['task'],delta=int(bv)-int(av)));recovered+=not av and bv;lost+=av and not bv
        ci=paired_ci(pairs)
        paired.append(dict(baseline=baseline,candidate=candidate,n=40,delta_pp=100*np.mean([r['delta'] for r in pairs]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],recovered=recovered,lost=lost))
    stability=[]
    for period in (4,6):
        pairs=[]
        for c in m['cases']:
            k=key(c);s=lambda n:int(data[n][k]['result']['success'])
            pairs.append(dict(suite=c['suite'],task=c['task'],delta=(s(f'v12_p{period}')-s('v12_p2'))-(s(f'prefix3_p{period}')-s('prefix3_p2'))))
        ci=paired_ci(pairs);stability.append(dict(period=period,v12_minus_prefix3_change_pp=100*np.mean([r['delta'] for r in pairs]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1]))
    deployment=[]
    for wd in sorted((OUT/'workers').glob('gpu*')):
        assert json.loads((wd/'gate.json').read_text())['passed'] and json.loads((wd/'cross_service.json').read_text())['passed']
    for path,expected in m['timing_files'].items():
        assert sha(path)==expected;bench=json.loads(Path(path).read_text());assert bench['exclusive'] and len(bench['rows'])==560
        means={n:float(np.mean([r['policy_wall_ms'] for r in bench['rows'] if r['path']==n])) for n in {r['path'] for r in bench['rows']}}
        for model,variant in [('v12','v12_reuse_reference'),('prefix3','prefix3_reuse')]:
            F,D=means['full_'+variant],means['draft_'+variant]
            for period in (2,4,6):
                cost=(F+period*D)/(period+1)
                deployment.append(dict(worker=Path(path).parent.name,condition=f'{model}_p{period}',period=period,estimated_request_ms=cost,full_refresh_ms=F,draft_ms=D,speedup_vs_own_p2=((F+2*D)/3)/cost,source='reused measured components'))
    summary=[];ages=[]
    for name,rs in data.items():
        for suite in ('all',*SUITES):
            values=[v['result'] for v in rs.values() if suite=='all' or v['result']['case']['suite']==suite]
            wall=sum(r['wall_s'] for r in values);actions=sum(r['actions'] for r in values);predictions=sum(r['predictions'] for r in values)
            summary.append(dict(condition=name,suite=suite,n=len(values),success=sum(r['success'] for r in values),policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in values)/actions,episode_wall_s=wall/len(values),actions_per_second=actions/wall,requests_per_second=predictions/wall,episodes_per_hour=3600*len(values)/wall,full_request_fraction=sum(r['counts']['full'] for r in values)/predictions))
        byage={}
        for v in rs.values():
            for line in (v['directory']/'infer.jsonl').read_text().splitlines():
                row=json.loads(line)
                if not row['route']['refresh']:byage.setdefault(row['full_cache_age'],[]).append(row['timing']['policy_wall_ms'])
        for age,values in sorted(byage.items()):ages.append(dict(condition=name,age_steps=age,age_seconds=age/20,requests=len(values),policy_wall_ms=float(np.mean(values))))
    for file,rows in [('episodes.csv',episodes),('summary.csv',summary),('paired.csv',paired),('stability.csv',stability),('deployment_estimate.csv',deployment),('cache_age_summary.csv',ages)]:write_csv(OUT/file,rows)
    lines=['# V12残差复用：刷新周期2/4/6与前三层方案对照','','V12跳层[2,3,5,14,16,17]，重放最近本路径Full残差；前三层方案仅L1–3当前hidden更新、L4–17重放残差。两者所有18层当前K/V、视觉、完整AE均实时。周期P=Full-D10→P次draft-D2，H50/R5。四suite各10task、init19，各40条。V12周期4/6为80条新闭环，其他160条兼容结果复用。','',
        '|方案|周期|Spatial|Object|Goal|Long|成功数|Full请求占比|策略ms/动作|组件估算ms/请求|','|---|---|---|---|---|---|---|---|---|---|']
    for name in NAMES:
        a={r['suite']:r for r in summary if r['condition']==name};cost=np.mean([r['estimated_request_ms'] for r in deployment if r['condition']==name])
        lines.append('|'+ '|'.join([name.split('_')[0],name[-1],*[f"{a[s]['success']}/10" for s in SUITES],f"{a['all']['success']}/40",f"{a['all']['full_request_fraction']:.1%}",f"{a['all']['policy_ms_per_action']:.3f}",f'{cost:.3f}'])+'|')
    lines+=['','|基线→候选|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','稳定性比较：以各自周期2为基准，V12的成功率变化减去前三层方案的变化，正值表示V12降幅更小。']
    for r in stability:lines.append(f"- 周期{r['period']}：{r['v12_minus_prefix3_change_pp']:+.2f}pp，探索95%区间[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]pp。")
    lines+=['','两新条件同case同服务轮换顺序；全部六条件首观测和init向量一致。所有新请求刷新日程、缓存年龄与D核对通过；V12共同调度前缀120项输出核对通过，逐值相同项数'+str(sum(r['exact'] for r in prefix_checks))+'。四suite服务门禁和跨卡核对通过，技术错误单列保存，不丢弃有效失败。',
        '', '周期2/4/6最大draft缓存年龄为10/20/30控制步，20Hz下0.5/1/1.5秒。改变周期同时改变残差年龄和Full-D10动作比例，不能把效果独归因于残差老化。两mask不嵌套：V12跳过L2/3，前三层方案实时计算L2/3，因此也不能把差异直接解释为某几层的单独贡献。',
        '', '组件耗时复用前轮三卡1680次独占固定输入测量并逐文件核hash，按(Full_capture+P×draft)/(P+1)摊销；不是本轮新测端到端加速。实际闭环策略wall、RPC、环境墙钟和吞吐见episodes.csv及summary.csv；旧轮耗时只作描述。环境构建前seed7、新建独立环境、官方init及10步settling保持一致。40条为探索，不是官方全量，不作等效/非劣声明。',
        '', '[冻结方案](idea1_v3_4_v12_refresh_sweep_plan.md) · [逐case](../../../results/idea1_v3_4_v12_refresh_sweep/episodes.csv) · [配对](../../../results/idea1_v3_4_v12_refresh_sweep/paired.csv) · [稳定性](../../../results/idea1_v3_4_v12_refresh_sweep/stability.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_v12_refresh_sweep_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'common_prefix_audit.json',dict(passed=True,cases=40,comparisons=len(prefix_checks),checks=prefix_checks))
    atomic(OUT/'audit.json',dict(passed=True,cases=40,new_candidates=2,initial_observation_cases=40,route_checks=route_checks,sources_match=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
