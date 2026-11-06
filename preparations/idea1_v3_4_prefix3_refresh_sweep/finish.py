import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,sha,key,directory
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def progress():
    rows={}
    for name in ('prefix3_p4','prefix3_p6'):
        results=[json.loads(p.read_text()) for p in (OUT/name/'episodes').glob('*/task*/init19/result.json') if (p.parent/'complete.json').exists() and '.technical.' not in p.parent.name]
        rows[name]=dict(completed=len(results),success=sum(r['success'] for r in results))
    return rows

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in m['source_files'].items());assert sha(PLAN)==m['plan_sha256']
    assert len(list((OUT/'case_done').glob('*.json')))==40
    data={k:{} for k in ('prefix3_p4','prefix3_p6','prefix3_p2')};episodes=[];route_checks=0
    for c in m['cases']:
        done=json.loads((OUT/'case_done'/f'{key(c)}.json').read_text());initial=[];vectors=[];pids=[]
        for cfg in m['conditions']+[m['reference_config']]:
            name=cfg['id'];d=Path(done['reference_directory']) if name=='prefix3_p2' else directory(name,c)
            r=json.loads((d/'result.json').read_text());identity=json.loads((d/'identity.json').read_text());assert r['valid']
            rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()];initial.append(rows[0]['input_hash']);vectors.append(identity['init_vector_sha256'])
            if name!='prefix3_p2':
                pids.append(identity['protocol']['server_pid']);assert r['manifest_hash']==m['manifest_hash']
                last=None
                for i,row in enumerate(rows):
                    full=i%(cfg['period']+1)==0;assert row['route']['refresh']==full and row['round']==i
                    assert len(row['timing']['nodes'])==(11 if full else 3)
                    if full:last=row['step'];assert row['route']['replayed_layers']==[]
                    else:
                        assert row['route']['replayed_layers']==cfg['skip']
                        assert row['route']['history_step']==last and row['full_cache_age']==row['step']-last
                    route_checks+=1
            elif done['reference_reused']:
                assert all(sha(d/n)==h for n,h in m['references'][key(c)]['hashes'].items())
            k=tuple(c[x] for x in ('suite','task','init'));data[name][k]=r
            episodes.append(dict(condition=name,**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],directory=str(d),reference_reused=done['reference_reused'] if name=='prefix3_p2' else False))
        assert len(set(initial))==len(set(vectors))==len(set(pids))==1
    paired=[]
    for baseline,candidate in [('prefix3_p2','prefix3_p4'),('prefix3_p2','prefix3_p6'),('prefix3_p4','prefix3_p6')]:
        a,b=data[baseline],data[candidate];ks=sorted(a)
        p=[dict(suite=k[0],task=k[1],delta=int(b[k]['success'])-int(a[k]['success'])) for k in ks];ci=paired_ci(p)
        paired.append(dict(baseline=baseline,candidate=candidate,n=40,delta_pp=100*np.mean([r['delta'] for r in p]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],
            recovered=sum(not a[k]['success'] and b[k]['success'] for k in ks),lost=sum(a[k]['success'] and not b[k]['success'] for k in ks)))
    summary=[];deployment=[];workers=sorted((OUT/'workers').glob('gpu*'));ages=[]
    for wd in workers:
        assert json.loads((wd/'gate.json').read_text())['passed'] and json.loads((wd/'cross_service.json').read_text())['passed']
    for path,expected in m['timing_files'].items():
        assert sha(path)==expected
        bench=json.loads(Path(path).read_text());assert bench['exclusive'] and len(bench['rows'])==560
        means={n:float(np.mean([r['policy_wall_ms'] for r in bench['rows'] if r['path']==n])) for n in {r['path'] for r in bench['rows']}}
        F,D=means['full_prefix3_reuse'],means['draft_prefix3_reuse']
        baseline=(F+2*D)/3
        for name,period in [('prefix3_p2',2),('prefix3_p4',4),('prefix3_p6',6)]:
            cost=(F+period*D)/(period+1)
            deployment.append(dict(worker=Path(path).parent.name,condition=name,period=period,estimated_request_ms=cost,
                full_refresh_ms=F,draft_ms=D,speedup_vs_p2=baseline/cost,source='reused measured components'))
    for name,rs in data.items():
        for suite in ('all',*SUITES):
            a=[r for k,r in rs.items() if suite=='all' or k[0]==suite]
            summary.append(dict(condition=name,suite=suite,n=len(a),success=sum(r['success'] for r in a),
                policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in a)/sum(r['actions'] for r in a),episode_wall_s=np.mean([r['wall_s'] for r in a]),
                actions_per_second=sum(r['actions'] for r in a)/sum(r['wall_s'] for r in a),requests_per_second=sum(r['predictions'] for r in a)/sum(r['wall_s'] for r in a),
                episodes_per_hour=3600*len(a)/sum(r['wall_s'] for r in a),full_request_fraction=sum(r['counts']['full'] for r in a)/sum(r['predictions'] for r in a)))
    for name,rs in data.items():
        byage={}
        for r in rs.values():
            # Reconstruct the directory from the already saved episode rows.
            c=r['case'];d=next(Path(x['directory']) for x in episodes if x['condition']==name and x['suite']==c['suite'] and x['task']==c['task'])
            for line in (d/'infer.jsonl').read_text().splitlines():
                row=json.loads(line)
                if not row['route']['refresh']:
                    age=row['full_cache_age'];byage.setdefault(age,[]).append(row['timing']['policy_wall_ms'])
        for age,values in sorted(byage.items()):ages.append(dict(condition=name,age_steps=age,age_seconds=age/20,requests=len(values),policy_wall_ms=float(np.mean(values))))
    for file,rows in [('episodes.csv',episodes),('summary.csv',summary),('paired.csv',paired),('deployment_estimate.csv',deployment),('cache_age_summary.csv',ages)]:write_csv(OUT/file,rows)
    lines=['# 前三层更新：刷新周期2/4/6对照','',
        '原π0 BF16 Triton，L1–3实时hidden更新、L4–17重放最近一次本路径Full残差，L18仅K/V。周期P=Full-D10→P次draft-D2，H50/R5。四suite各10task、init19，各40条；周期4/6为80条新闭环，周期2为同批兼容参考。',
        '', '|周期|Spatial|Object|Goal|Long|总成功|实际Full请求占比|策略ms/动作|组件估算ms/请求|相对周期2加速|','|---|---|---|---|---|---|---|---|---|---|']
    for name,period in [('prefix3_p2',2),('prefix3_p4',4),('prefix3_p6',6)]:
        a={r['suite']:r for r in summary if r['condition']==name};ds=[r for r in deployment if r['condition']==name];mean=lambda k:float(np.mean([r[k] for r in ds]))
        lines.append('|'+ '|'.join([str(period),*[f"{a[s]['success']}/10" for s in SUITES],f"{a['all']['success']}/40",f"{a['all']['full_request_fraction']:.1%}",f"{a['all']['policy_ms_per_action']:.3f}",f"{mean('estimated_request_ms'):.3f}",f"{mean('speedup_vs_p2'):.3f}×"] )+'|')
    lines+=['','|基线→候选|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','周期2/4/6的最大draft缓存年龄分别10/20/30控制步（0.5/1/1.5秒），实际分布见cache_age_summary.csv。周期拉长同时减少Full-D10动作占比，因此质量差不能单独归因于残差年龄。',
        '', '延迟为前轮同一backend的1680次独占组件测量复用估计，按(Full_capture+P×draft)/(P+1)摊销，不是本轮新跑微基准；文件hash逐一核对。本轮实际闭环策略wall、RPC、回合墙钟及吞吐保留在逐case及summary。周期2的回合耗时来自前轮，仅作描述，不直接与新条件作独占速度因果比较。',
        '', '环境构建前seed7、独立新环境、官方init、10步settling；全部条件首观测相同才合并。没有新训练、选层、外推、verifier或夹爪保护，Full/残差提取及修补均计入路径。40条为探索评测，不能声称等效、非劣或官方全量成绩。',
        '', '[冻结方案](idea1_v3_4_prefix3_refresh_sweep_plan.md) · [逐case](../../../results/idea1_v3_4_prefix3_refresh_sweep/episodes.csv) · [组件估计](../../../results/idea1_v3_4_prefix3_refresh_sweep/deployment_estimate.csv) · [配对结果](../../../results/idea1_v3_4_prefix3_refresh_sweep/paired.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_prefix3_refresh_sweep_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,cases_per_candidate=40,new_candidates=2,initial_observation_cases=40,route_checks=route_checks,sources_match=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
