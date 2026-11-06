import csv,json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,BASE,PLAN,CONDITIONS,SUITES,sha,ck,directory,legal,choose
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v3_4_E2_selection_200.finish import header,prefix,paired

def key(c):return tuple(c[k] for k in ('suite','task','init'))
def rows(d):return [json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]

def audit_episode(cfg,d,r):
    qs=rows(d);method=cfg['method'];comp=cfg['compensation']=='extrapolate';k=cfg['skip_count']
    mask=None;mask_step=None;selectors=changes=fulls=fasts=0;history=[];cold=0.
    with np.load(d/'arrays.npz') as z:
        for i,q in enumerate(qs):
            rt=q['route'];full=i%5==0
            refresh=full if method in ('cosine','jacobian') else i==0 or (method=='random_online' and not full and i!=1)
            assert q['replan']==q['round']==i and q['step']==i*5
            assert rt['period']==5 and rt['path']==('full' if full else 'fast') and rt['D']==(10 if full else 2)
            assert len(q['timing']['nodes'])==(11 if full else 3)
            assert rt['method']==method and rt['compensation']==cfg['compensation'] and rt['adjacency_rule']=='spaced'
            assert rt['steady_vjp_count']==(4 if full else 0) and rt['selector_forward_count']==int(full)
            assert rt['configured_projections']==4 and rt['scoring_mode']=='original_residual' and rt['score_aggregation']=='projection_rms' and rt['projection_seed']==340610
            assert legal(rt['selected_skip'],k,'spaced') and rt['selected_longest_run']==1
            assert np.isfinite(z[f'raw_{i}']).all() and np.isfinite(z[f'actions_{i}']).all()
            assert rt['mask_refresh']==refresh
            if refresh:
                candidate=rt['selected_skip'];selectors+=1;changes+=int(mask is not None and mask!=candidate);mask=candidate;mask_step=q['step']
                if method in ('random_online','random_fixed'):assert mask==random_mask(cfg['random_seed'],q['seed'],k)
                elif method=='global_fixed':assert mask==cfg['fixed_skip']
                else:
                    score=z[f'layer_score_{i}'];assert score.shape==(17,) and np.isfinite(score).all()
                    assert choose(score,k,'spaced')==mask
                    if method=='jacobian':
                        grad=z[f'gate_gradient_{i}'];assert grad.shape==(4,17) and np.isfinite(grad).all() and np.array_equal(score,np.sqrt(np.mean(np.square(grad),axis=0)))
            else:assert q['timing']['selector_total_ms']==0
            assert rt['selected_skip']==mask and rt['mask_age_env_steps']==q['step']-mask_step
            assert rt['full_age_rounds']==i%5 and rt['full_age_env_steps']==i%5*5
            assert q['counts']['selector']==selectors
            if full:
                fulls+=1;assert rt['skip']==[]
                if comp:history=(history+[q['step']])[-2:]
            else:fasts+=1;assert rt['skip']==mask
            assert q['counts']['full']==fulls and q['counts']['draft']==fasts and q['counts']['verify']==0
            if method=='jacobian':assert rt['jacobian_model_steps']==10 and rt['jacobian_output_steps']==[1,2,3,4,5] and rt['jacobian_full_layers']==18
            if comp:
                assert rt['captured_layers']==list(range(1,18)) and rt['history_full_steps']==history and rt['history_bytes']>0
                assert rt['patch_layers']==([] if full else mask)
                if full:assert rt['extrapolation_ratio']==0.
                else:
                    gap=None if len(history)<2 else history[-1]-history[-2]
                    assert rt['has_second_history']==(gap is not None)
                    assert np.isclose(rt['extrapolation_ratio'],0. if gap is None else (q['step']-history[-1])/gap)
            cold+=q['timing']['selector_capture_ms']/1000
        h=dict(result=r,identity=identity(d),first_mask=qs[0]['route']['selected_skip'],raw0=z['raw_0'].copy(),actions0=z['actions_0'].copy(),fast_input=qs[1]['input_hash'] if len(qs)>1 else None)
        if method=='jacobian':h['gradient0']=z['gate_gradient_0'][0].copy()
    assert len(qs)==r['predictions'] and sum(q['executed_length'] for q in qs)==r['actions']
    blocks=[qs[i:i+5] for i in range(5,len(qs)-4,5)]
    return h,dict(requests=qs,cold=cold,changes=changes,selectors=selectors,blocks=blocks)

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==v for p,v in m['source_files'].items()) and all(sha(ROOT/p)==v for p,v in m['references'].items())
    assert json.loads((OUT/'cpu_gate.json').read_text())['passed']
    keys={key(c) for c in m['cases']};data={};quality=[];episodes=[];requests=[];costs=[];gradient_checks=[]
    baseline_costs={r['condition']:r for r in csv.DictReader((BASE/'costs.csv').open())}
    for name,bank in m['baseline'].items():
        dd={}
        for c in m['cases']:
            ref=bank[ck(c)];d=Path(ref['directory']);assert all(sha(d/f)==v for f,v in ref['hashes'].items())
            h=header(d)
            with np.load(d/'arrays.npz') as z:h['gradient0']=z['gate_gradient_0'].copy()
            dd[key(c)]=h
        data['q1_'+name]=dd
        quality.append(dict(condition='q1_'+name,projections=1,compensation='none' if name.endswith('_none') else 'extrapolate',n=200,success=sum(h['result']['success'] for h in dd.values()),reused=True,
                            **{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
    for cfg in CONDITIONS:
        dd={};blocks=[];cold=[];changes=selectors=coverage=0
        for c in m['cases']:
            d=directory(cfg['id'],c);r=json.loads((d/'result.json').read_text())
            assert (d/'complete.json').exists() and r['valid'] and r['case']==c and r['config']==cfg and r['manifest_hash']==m['manifest_hash']
            h,a=audit_episode(cfg,d,r);dd[key(c)]=h;baseline=data['q1_'+cfg['base_id']][key(c)]
            assert {k:v for k,v in h['identity'].items() if k!='first_input_hash'}=={k:v for k,v in baseline['identity'].items() if k!='first_input_hash'}
            first_gradient_ok=np.allclose(h['gradient0'],baseline['gradient0'],atol=1e-4,rtol=1e-3)
            assert first_gradient_ok,('first positive direction changed',cfg['id'],c)
            gradient_checks.append(dict(condition=cfg['id'],case=c,passed=True,max_abs=float(abs(h['gradient0']-baseline['gradient0']).max())))
            blocks+=a['blocks'];cold.append(a['cold']);changes+=a['changes'];selectors+=a['selectors'];coverage+=bool(a['blocks'])
            for q in a['requests']:requests.append(dict(condition=cfg['id'],**c,replan=q['replan'],path=q['route']['path'],policy_ms=q['timing']['policy_wall_ms'],selector_ms=q['timing']['selector_total_ms'],selector_capture_ms=q['timing']['selector_capture_ms'],rpc_ms=q['rpc_ms']))
            episodes.append(dict(condition=cfg['id'],**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],cold_s=a['cold']))
        data[cfg['id']]=dd;quality.append(dict(condition=cfg['id'],projections=4,compensation=cfg['compensation'],n=200,success=sum(h['result']['success'] for h in dd.values()),reused=False,
                                             **{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
        steady=[q for block in blocks for q in block]
        def mean(a):return float(np.mean(a)) if a else None
        costs.append(dict(condition=cfg['id'],n=200,steady_covered_episodes=coverage,steady_complete_cycles=len(blocks),cycle_policy_ms=mean([q['timing']['policy_wall_ms'] for q in steady]),
                          full_policy_ms=mean([q['timing']['policy_wall_ms'] for q in steady if q['route']['path']=='full']),fast_policy_ms=mean([q['timing']['policy_wall_ms'] for q in steady if q['route']['path']=='fast']),
                          mean_cold_s=mean(cold),mask_changes=changes,selection_events=selectors,max_run=1))
        atomic(OUT/'condition_audits'/f"{cfg['id']}.json",dict(passed=True,n=200,projections=4,full_forward_count=1,max_run=1));print(json.dumps(dict(audited_condition=cfg['id'],n=200)),flush=True)
    comparisons=[('q1_'+cfg['base_id'],cfg['id'],False) for cfg in CONDITIONS]+[(CONDITIONS[0]['id'],CONDITIONS[1]['id'],True)]
    strict=[];sensitivity=[];exceptions=[]
    for a,b,same_mask in comparisons:
        for scope,target in (('strict',strict),('all200',sensitivity)):
            row,ex=paired(a,b,data[a],data[b],keys,scope,same_mask);target.append(row)
            if scope=='strict':exceptions+=ex
    for file,rs in (('summary.csv',quality),('episodes.csv',episodes),('requests.csv',requests),('costs.csv',costs),('paired.csv',strict),('paired_sensitivity.csv',sensitivity)):write_csv(OUT/file,rs)
    atomic(OUT/'first_projection_checks.json',dict(passed=True,n=400,checks=gradient_checks));atomic(OUT/'prefix_audit.json',dict(pair_exceptions=exceptions,all_valid_outcomes_retained=True))
    workers=[]
    for wd in sorted((OUT/'workers').glob('gpu*')):
        if not (wd/'startup.json').exists():continue
        startup=json.loads((wd/'startup.json').read_text());state=json.loads((wd/'state.json').read_text());exits=json.loads((wd/'exits.json').read_text())
        assert state['status']=='complete' and exits['service'] in (-15,0) and all(code==0 for code in exits['evaluations'])
        assert json.loads((wd/'compatibility_gate.json').read_text())['passed']
        iso=json.loads((wd/'isolation_gate.json').read_text());assert startup['actors']==1 or iso['passed']
        attempts=[json.loads(p.read_text()) for p in wd.glob('startup_attempt*.json')]
        workers.append(dict(identity=json.loads((wd/'identity.json').read_text()),startup=startup,startup_attempts=attempts,state=state,exits=exits,isolation=iso))
    start=min(s['time'] for w in workers for s in (w['startup_attempts'] or [w['startup']]))
    end=max(w['state']['time'] for w in workers);wall=end-start
    retries=[str(p) for p in (OUT/'technical_attempts').rglob('*.json') if json.loads(p.read_text())['attempts']>1]
    throughput=dict(new_episodes=400,evaluation_wall_s=wall,episodes_per_hour=400*3600/wall,actions_per_s=sum(r['actions'] for r in episodes)/wall,requests_per_s=sum(r['predictions'] for r in episodes)/wall,
                    service_startup_s=[w['startup']['startup_s'] for w in workers],technical_retries=retries,workers=workers,controller_start=json.loads((OUT/'process.json').read_text())['time'])
    assert len(episodes)==400;atomic(OUT/'throughput.json',throughput)
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle5_q4_v12_report.md'
    lines=['# 周期5、V12：一次与四次反推，有无外推各200','','Full-D10→Fast-D2×4，只允许间隔跳6层。四次从首Full启用，共享一次可微Full-D10前向，第一方向为原正权、额外三组固定seed340610正负方向；原残差门控、RMS评分，不使用补偿误差方向。新400回合，旧一次基线400有效记录复用。','','|反推次数|无补偿|外推补偿|','|---|---|---|']
    for q in (1,4):
        vs=[next(r for r in quality if r['projections']==q and r['compensation']==comp)['success'] for comp in ('none','extrapolate')]
        lines.append(f'|{q}|'+'|'.join(f'{v}/200（{v/2:.1f}%）' for v in vs)+'|')
    lines+=['','|条件|均摊ms/请求|Full ms|Fast ms|覆盖回合|平均冷捕获秒|','|---|---|---|---|---|---|']
    for cfg in CONDITIONS:
        old=baseline_costs[cfg['base_id']]
        lines.append(f"|q1_{cfg['base_id']}|{float(old['cycle_policy_ms']):.2f}|{float(old['full_policy_ms']):.2f}|{float(old['fast_policy_ms']):.2f}|{old['steady_covered_episodes']}/200|{float(old['mean_cold_s']):.2f}|")
    for r in costs:lines.append(f"|{r['condition']}|{r['cycle_policy_ms']:.2f}|{r['full_policy_ms']:.2f}|{r['fast_policy_ms']:.2f}|{r['steady_covered_episodes']}/200|{r['mean_cold_s']:.2f}|")
    lines+=['','稳态从第二Full起仅取完整五请求块，不含冷捕获与RPC排队；Full峰值单列。自由轨迹耗时不当作同输入因果加速。','','|基准→候选|严格n|恢复/新增失败|task等权变化pp|探索95%区间pp|','|---|---|---|---|---|']
    for r in strict:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['task_balanced_delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['',f"实际闭环窗口{wall/60:.2f}分钟，{throughput['episodes_per_hour']:.1f}回合/小时，{throughput['actions_per_s']:.2f}动作/秒，{throughput['requests_per_s']:.2f}请求/秒；服务准备秒数{throughput['service_startup_s']}单列。技术重试{len(retries)}回合。",'', f'共同前缀配对异常记录{len(exceptions)}项，完整200描述保留；严格配对与all200敏感性分别输出。第一投影400例核验通过。suite分层task聚类bootstrap2000次，seed281034，探索区间未校正多重比较。', '', '四组方向仅用一个预设随机种子，并非随机种子总体均值。近似PyTorch Full-D10回放、局部门控、多层交互及补偿历史时效性局限延续。', '', '[冻结方案](idea1_v3_4_E2_cycle5_q4_v12_plan.md)']
    report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=400,source_hashes=True,period=5,projections=4,shared_forward=1,max_run=1,first_projection_reference=True,prefix_exceptions=len(exceptions),time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',completed=400,time=time.time()))

if __name__=='__main__':main()

