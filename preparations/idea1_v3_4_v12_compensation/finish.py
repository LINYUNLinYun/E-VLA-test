import hashlib,json,time
import numpy as np
from .common import ROOT,OUT,SUITES
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def results():
    m=json.loads((OUT/'manifest.json').read_text());data={}
    for cfg in m['conditions']:
        rs={}
        for path in (OUT/cfg['id']/'episodes').glob('*/task*/init*/result.json'):
            if '.invalid.' in path.parent.name or not (path.parent/'complete.json').exists():continue
            r=json.loads(path.read_text())
            assert r['valid'] and r['manifest_hash']==m['manifest_hash']
            r['directory']=str(path.parent);rs[tuple(r['case'][k] for k in ('suite','task','init'))]=r
        data[cfg['id']]=rs
    return m,data

def progress():
    _,data=results()
    return {name:dict(completed=len(rs),success=sum(r['success'] for r in rs.values())) for name,rs in data.items()}

def main():
    m,data=results();expected={tuple(c[k] for k in ('suite','task','init')) for c in m['cases']}
    assert all(set(rs)==expected for rs in data.values())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in m['source_files'].items())
    assert hashlib.sha256((ROOT/'docs/experiments/idea1/idea1_v3_4_v12_compensation_plan.md').read_bytes()).hexdigest()==m['plan_sha256']
    summary=[];flat=[];inference=[];identities={};route_checks=0
    for name,rs in data.items():
        for key,r in rs.items():
            d=__import__('pathlib').Path(r['directory'])
            identity=json.loads((d/'identity.json').read_text());identities[(name,key)]=identity
            rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
            full_steps=[]
            for i,row in enumerate(rows):
                cfg=r['config'];mode=cfg['mode'];isfull=mode=='full' or (mode=='periodic' and i%3==0)
                assert row['route']['refresh']==isfull and row['round']==i
                if isfull:full_steps.append(row['step'])
                if mode=='periodic' and not isfull:
                    assert row['route']['history_steps']==full_steps[-2:]
                    assert row['full_cache_age']==row['step']-full_steps[-1]
                    assert row['route']['has_second_history']==(len(full_steps)>=2)
                    ratio=0. if len(full_steps)<2 else (row['step']-full_steps[-1])/(full_steps[-1]-full_steps[-2])
                    assert np.isclose(row['route']['extrapolation_ratio'],ratio)
                route_checks+=1
                inference.append(dict(condition=name,**r['case'],request=i,path=row['route']['path'],
                    policy_wall_ms=row['timing']['policy_wall_ms'],rpc_ms=row['rpc_ms'],executed=row['executed_length'],
                    vlm_ms=row['timing']['vlm_ms'],vision_ms=row['timing']['vision_ms'],ae_ms=row['timing']['ae_ms']))
            flat.append(dict(condition=name,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],
                wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],full_requests=r['counts']['full'],draft_requests=r['counts']['draft'],
                rpc_s=r['timing']['rpc_s'],step_s=r['timing']['step_s'],reset_s=r['timing']['reset_s'],source=r['directory']))
        for suite in ('all',*SUITES):
            a=[r for k,r in rs.items() if suite=='all' or k[0]==suite]
            total=lambda k:sum(r[k] for r in a)
            summary.append(dict(condition=name,suite=suite,n=len(a),success=sum(r['success'] for r in a),
                policy_ms_per_action=1000*total('policy_wall_s')/total('actions'),policy_ms_per_request=1000*total('policy_wall_s')/total('predictions'),
                episode_wall_s=total('wall_s')/len(a),episodes_per_hour=3600*len(a)/total('wall_s'),
                actions_per_second=total('actions')/total('wall_s'),requests_per_second=total('predictions')/total('wall_s'),
                actions_mean=total('actions')/len(a),full_request_ratio=sum(r['counts']['full'] for r in a)/total('predictions')))
    for key in expected:
        ids=[identities[(name,key)] for name in data]
        assert len({r['init_vector_sha256'] for r in ids})==1
        assert len({r['protocol']['server_pid'] for r in ids})==1
        assert len({r['prompt'] for r in ids})==1
    pairs=[]
    for baseline in ('pure_v12','periodic_v12'):
        for candidate in (('v12_channel','v12_hadamard','v12_linear','periodic_v12','periodic_reuse','periodic_extrapolate') if baseline=='pure_v12' else ('periodic_reuse','periodic_extrapolate')):
            aa,bb=data[baseline],data[candidate]
            ps=[dict(suite=k[0],task=k[1],delta=int(bb[k]['success'])-int(aa[k]['success'])) for k in sorted(expected)]
            ci=paired_ci(ps)
            metric=lambda rs:sum(r['policy_wall_s'] for r in rs.values())/sum(r['actions'] for r in rs.values())
            pairs.append(dict(baseline=baseline,candidate=candidate,n=len(expected),delta_pp=100*np.mean([p['delta'] for p in ps]),
                ci_low_pp=100*ci[0],ci_high_pp=100*ci[1],recovered=sum(not aa[k]['success'] and bb[k]['success'] for k in expected),
                lost=sum(aa[k]['success'] and not bb[k]['success'] for k in expected),closedloop_policy_speedup=metric(aa)/metric(bb)))
    benches=[];timings=[];deploy=[]
    workers=sorted((OUT/'workers').glob('gpu*'))
    for wd in workers:
        if not (wd/'benchmark.json').exists():continue
        assert json.loads((wd/'gate.json').read_text())['passed']
        assert json.loads((wd/'cross_service.json').read_text())['passed']
        b=json.loads((wd/'benchmark.json').read_text())
        assert len(b['rows'])==720
        benches.extend(dict(worker=wd.name,**r) for r in b['rows'])
        means={}
        for path in sorted({r['path'] for r in b['rows']}):
            a=[r for r in b['rows'] if r['path']==path];means[path]=np.mean([r['policy_wall_ms'] for r in a])
            timings.append(dict(worker=wd.name,path=path,n=len(a),mean_ms=means[path],
                p50_ms=np.median([r['policy_wall_ms'] for r in a]),p95_ms=np.percentile([r['policy_wall_ms'] for r in a],95),
                vlm_ms=np.mean([r['vlm_ms'] for r in a]),vision_ms=np.mean([r['vision_ms'] for r in a]),ae_ms=np.mean([r['ae_ms'] for r in a])))
        costs=dict(pure_v12=means['v12_d2'],v12_channel=means['channel'],v12_hadamard=means['hadamard'],v12_linear=means['linear'],periodic_v12=(means['full10']+2*means['v12_d2'])/3,
            periodic_reuse=(means['full10_capture']+2*means['reuse'])/3,
            periodic_extrapolate=(means['full10_capture']+means['extrapolate5']+means['extrapolate10'])/3,
            full10_reference=means['full10'])
        for name,value in costs.items():deploy.append(dict(worker=wd.name,condition=name,steady_request_ms=value,
            ms_per_action=value/5,speedup_vs_periodic=costs['periodic_v12']/value,speedup_vs_pure=costs['pure_v12']/value,
            speedup_vs_full=costs['full10_reference']/value))
    assert len({r['worker'] for r in deploy})==len(m['gpus'])
    assert all(json.loads((wd/'benchmark.json').read_text())['exclusive'] for wd in workers)
    calibration=json.loads((OUT/'calibration.json').read_text())
    assert calibration['passed'] and len(calibration['calibration_ids'])==80
    assert set(calibration['calibration_ids']).isdisjoint(calibration['validation_ids'])
    assert hashlib.sha256((OUT/'adapters.pt').read_bytes()).hexdigest()==calibration['adapter_sha256']
    for filename,rows in [('episodes.csv',flat),('summary.csv',summary),('paired.csv',pairs),('inference.csv',inference),
                          ('fixed_input_requests.csv',benches),('fixed_input_summary.csv',timings),('deployment.csv',deploy)]:write_csv(OUT/filename,rows)
    lines=['# 指定V12与五种补偿：闭环及部署成本','',
        '四suite各10 task、官方init19，40条/条件、共280条新闭环。原BF16 Triton π0，V12跳[2,3,5,14,16,17]，H50/R5；纯V12及三种静态补偿全程D2；后三个周期条件为Full-D10→两次draft-D2。复用/外推缓存只来自各自路径已发生的Full，包含提取、补偿及刷新成本。无验证器或夹爪保护。',
        '', '|条件|Spatial|Object|Goal|Long|总成功|策略ms/动作|回合墙钟s|episode/h|动作/s|请求/s|','|---|---|---|---|---|---|---|---|---|---|---|']
    for cfg in m['conditions']:
        g={r['suite']:r for r in summary if r['condition']==cfg['id']};a=g['all']
        lines.append('|'+ '|'.join([cfg['id'],*[f"{g[s]['success']}/10" for s in SUITES],f"{a['success']}/40",
            f"{a['policy_ms_per_action']:.3f}",f"{a['episode_wall_s']:.2f}",f"{a['episodes_per_hour']:.2f}",f"{a['actions_per_second']:.2f}",f"{a['requests_per_second']:.2f}"])+'|')
    lines+=['','|基线→候选|恢复/新增失败|成功率差pp|探索95%区间pp|闭环策略成本比（基线/候选）|','|---|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|{r['closedloop_policy_speedup']:.3f}×|")
    lines+=['','固定真实输入、相同输入/噪声、每卡40输入×两遍，每输入每路径先预热5次；表中为每卡点估计的等权平均。周期值按稳态1次Full+2次draft摊销，包含Full残差提取；启动时无第二历史的回退成本已计入上表真实闭环。',
        '', '|条件|固定输入摊销ms/请求|相对周期V12加速|相对纯V12加速|相对Full-D10加速|','|---|---|---|---|---|']
    for cfg in m['conditions']:
        a=[r for r in deploy if r['condition']==cfg['id']]
        avg=lambda k:float(np.mean([r[k] for r in a]))
        lines.append(f"|{cfg['id']}|{avg('steady_request_ms'):.3f}|{avg('speedup_vs_periodic'):.3f}×|{avg('speedup_vs_pure'):.3f}×|{avg('speedup_vs_full'):.3f}×|")
    lines+=['',f"静态校准及验证选择成本 {calibration['wall_s']:.2f}s；低秩全局rank={calibration['selected_rank']}。80个校准及32个验证请求均为旧init14观测，闭环使用init19；任务身份有重叠，不称跨任务泛化。",'','服务准备成本：','']
    for wd in workers:
        if (wd/'protocol.json').exists():lines.append(f"- {wd.name}：加载及工程核验 {json.loads((wd/'protocol.json').read_text())['service_load_and_gate_s']:.2f}s；固定输入测速及其预热 {json.loads((wd/'benchmark.json').read_text())['wall_s']:.2f}s。")
    lines+=['','成功率区间为suite分层task聚类bootstrap2000次探索区间，未校正多重比较。40条快速评测不能宣称总体非劣、等效或官方基准成绩。真实闭环轨迹不同，策略ms/动作反映实际执行成本；因果时延对照使用固定输入测速。回合墙钟含仿真、环境准备及落盘；上述episode/h为各回合墙钟汇总的worker吞吐，不是三卡联合吞吐。',
        '', '[冻结方案](idea1_v3_4_v12_compensation_plan.md) · [逐case](../../../results/idea1_v3_4_v12_compensation/episodes.csv) · [配对比较](../../../results/idea1_v3_4_v12_compensation/paired.csv) · [固定输入成本](../../../results/idea1_v3_4_v12_compensation/deployment.csv)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_v12_compensation_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,cases_per_condition=40,conditions=7,sources_match=True,
        route_checks=route_checks,own_history_checks=True,paired_initial_states=True,gpus=m['gpus'],time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))

if __name__=='__main__':main()
