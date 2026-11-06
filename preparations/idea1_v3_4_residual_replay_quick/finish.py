import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,sha,key,directory
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def progress():
    rows={}
    for name in ('all_reuse','prefix3_reuse'):
        results=[json.loads(p.read_text()) for p in (OUT/name/'episodes').glob('*/task*/init19/result.json') if (p.parent/'complete.json').exists() and '.technical.' not in p.parent.name]
        rows[name]=dict(completed=len(results),success=sum(r['success'] for r in results))
    return rows

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in m['source_files'].items());assert sha(PLAN)==m['plan_sha256']
    assert len(list((OUT/'case_done').glob('*.json')))==40
    data={k:{} for k in ('all_reuse','prefix3_reuse','v12_reuse_reference')};episodes=[];route_checks=0
    for c in m['cases']:
        done=json.loads((OUT/'case_done'/f'{key(c)}.json').read_text());initial=[];vectors=[];pids=[]
        for cfg in m['conditions']+[m['reference_config']]:
            name=cfg['id'];d=Path(done['reference_directory']) if name=='v12_reuse_reference' else directory(name,c)
            r=json.loads((d/'result.json').read_text());identity=json.loads((d/'identity.json').read_text());assert r['valid']
            rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()];initial.append(rows[0]['input_hash']);vectors.append(identity['init_vector_sha256'])
            if name!='v12_reuse_reference':
                pids.append(identity['protocol']['server_pid']);assert r['manifest_hash']==m['manifest_hash']
                last=None
                for i,row in enumerate(rows):
                    full=i%3==0;assert row['route']['refresh']==full and row['round']==i
                    assert len(row['timing']['nodes'])==(11 if full else 3)
                    if full:last=row['step'];assert row['route']['replayed_layers']==[]
                    else:
                        assert row['route']['replayed_layers']==cfg['skip']
                        assert row['route']['history_step']==last and row['full_cache_age']==row['step']-last
                    route_checks+=1
            elif done['reference_reused']:
                assert all(sha(d/n)==h for n,h in m['references'][key(c)]['hashes'].items())
            k=tuple(c[x] for x in ('suite','task','init'));data[name][k]=r
            episodes.append(dict(condition=name,**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],directory=str(d),reference_reused=done['reference_reused'] if name=='v12_reuse_reference' else False))
        assert len(set(initial))==len(set(vectors))==len(set(pids))==1
    paired=[]
    for baseline,candidate in [('v12_reuse_reference','all_reuse'),('v12_reuse_reference','prefix3_reuse'),('all_reuse','prefix3_reuse')]:
        a,b=data[baseline],data[candidate];ks=sorted(a)
        p=[dict(suite=k[0],task=k[1],delta=int(b[k]['success'])-int(a[k]['success'])) for k in ks];ci=paired_ci(p)
        paired.append(dict(baseline=baseline,candidate=candidate,n=40,delta_pp=100*np.mean([r['delta'] for r in p]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],
            recovered=sum(not a[k]['success'] and b[k]['success'] for k in ks),lost=sum(a[k]['success'] and not b[k]['success'] for k in ks)))
    summary=[];deployment=[];timing=[];workers=sorted((OUT/'workers').glob('gpu*'))
    for wd in workers:
        assert json.loads((wd/'gate.json').read_text())['passed'] and json.loads((wd/'cross_service.json').read_text())['passed']
        bench=json.loads((wd/'benchmark.json').read_text());assert len(bench['rows'])==560
        if not bench['exclusive']:continue
        means={n:float(np.mean([r['policy_wall_ms'] for r in bench['rows'] if r['path']==n])) for n in {r['path'] for r in bench['rows']}}
        costs={name:(means['full_'+name]+2*means['draft_'+name])/3 for name in data}
        for name,value in costs.items():deployment.append(dict(worker=wd.name,condition=name,steady_request_ms=value,draft_ms=means['draft_'+name],full_refresh_ms=means['full_'+name],
            speedup_vs_v12=costs['v12_reuse_reference']/value,speedup_vs_full=means['full_no_capture']/value,cache_bytes=bench['cache_bytes'][name]))
        timing.extend(dict(worker=wd.name,**r) for r in bench['rows'])
    assert deployment,'No exclusive timing measurements'
    for name,rs in data.items():
        for suite in ('all',*SUITES):
            a=[r for k,r in rs.items() if suite=='all' or k[0]==suite]
            summary.append(dict(condition=name,suite=suite,n=len(a),success=sum(r['success'] for r in a),
                policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in a)/sum(r['actions'] for r in a),episode_wall_s=np.mean([r['wall_s'] for r in a]),
                actions_per_second=sum(r['actions'] for r in a)/sum(r['wall_s'] for r in a),requests_per_second=sum(r['predictions'] for r in a)/sum(r['wall_s'] for r in a),
                episodes_per_hour=3600*len(a)/sum(r['wall_s'] for r in a)))
    for file,rows in [('episodes.csv',episodes),('summary.csv',summary),('paired.csv',paired),('deployment.csv',deployment),('fixed_input_requests.csv',timing)]:write_csv(OUT/file,rows)
    lines=['# 周期2残差重放快速测试','',
        '四suite各10task、init19，两个候选各40条新闭环。周期为Full-D10→两次draft-D2，H50/R5。all_reuse跳过L1–17的实时hidden更新；prefix3_reuse仅L1–3实时更新，L4–17重放残差。L18仅生成K/V。所有请求仍处理当前图像、投影当前K/V并运行完整AE，不使用旧K/V或旧动作。',
        '', '|条件|Spatial|Object|Goal|Long|总成功|策略ms/动作|稳态ms/请求|draft ms|相对V12复用加速|','|---|---|---|---|---|---|---|---|---|---|']
    for name in data:
        a={r['suite']:r for r in summary if r['condition']==name};ds=[r for r in deployment if r['condition']==name];mean=lambda k:float(np.mean([r[k] for r in ds]))
        lines.append('|'+ '|'.join([name,*[f"{a[s]['success']}/10" for s in SUITES],f"{a['all']['success']}/40",f"{a['all']['policy_ms_per_action']:.3f}",f"{mean('steady_request_ms'):.3f}",f"{mean('draft_ms'):.3f}",f"{mean('speedup_vs_v12'):.3f}×"] )+'|')
    lines+=['','|基线→候选|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','V12复用为显式复用的历史参考，逐case首观测、官方init及模型路径门禁匹配才使用；不兼容则本轮重跑参考。新候选同case在同服务执行，环境每条件独立构建且构建前设seed7。',
        '', '40固定真实输入、每输入每路径先预热3次再AB/BA两遍，共每卡560次测量。稳态按(含残差提取Full+2×draft)/3摊销。仅未见共卡的测速纳入上表，准备和预热成本另列。固定输入时延与不同轨迹闭环ms/动作不混为同一口径。',
        '', '缓存FP32残差只保留最近一次本路径Full，Full刷新、复制/提取及修补成本均计入。历史参考的闭环墙钟来自前轮，回合耗时只作描述，不与本轮作独占因果速度比较。40条探索样本不能宣称等效、总体非劣或官方全量结果。',
        '', '[冻结方案](idea1_v3_4_residual_replay_quick_plan.md) · [逐case](../../../results/idea1_v3_4_residual_replay_quick/episodes.csv) · [部署计时](../../../results/idea1_v3_4_residual_replay_quick/deployment.csv) · [配对结果](../../../results/idea1_v3_4_residual_replay_quick/paired.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_residual_replay_quick_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,cases_per_candidate=40,new_candidates=2,initial_observation_cases=40,route_checks=route_checks,sources_match=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
