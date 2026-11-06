"""Audit all executed routes, then publish matched quality and cost tables."""
import itertools,json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,CONDITIONS,SUITES,sha,ck,directory,legal,choose,random_mask
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
            assert rt['steady_vjp_count']==int(full and method=='jacobian')
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
                        grad=z[f'gate_gradient_{i}'];assert grad.shape==(17,) and np.isfinite(grad).all() and np.array_equal(score,abs(grad))
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
        if method=='jacobian':h['gradient0']=z['gate_gradient_0'].copy()
    assert len(qs)==r['predictions'] and sum(q['executed_length'] for q in qs)==r['actions']
    blocks=[qs[i:i+5] for i in range(5,len(qs)-4,5)]
    return h,dict(requests=qs,cold=cold,changes=changes,selectors=selectors,blocks=blocks)

def quality(name,cfg,bank,keys):
    return dict(condition=name,budget=cfg['budget'],method=cfg['method'],compensation=cfg['compensation'],period=5,rule='spaced',n=len(keys),success=sum(bank[k]['result']['success'] for k in keys),
                **{s:sum(bank[k]['result']['success'] for k in keys if k[0]==s) for s in SUITES})

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert sha(PLAN)==m['plan_sha256'] and all(sha(ROOT/p)==v for p,v in m['source_files'].items()) and all(sha(ROOT/p)==v for p,v in m['references'].items())
    assert json.loads((OUT/'cpu_gate.json').read_text())['passed']
    data={};quality_all=[];quality_small=[];episodes=[];requests=[];costs=[];reference_exceptions=[]
    all_keys={key(c) for c in m['cases']};small_keys={key(c) for c in m['small_cases']}
    for cfg in CONDITIONS:
        dd={};blocks=[];cold=[];changes=selectors=coverage=0
        for c in (m['small_cases'] if cfg['n']==40 else m['cases']):
            d=directory(cfg['id'],c);r=json.loads((d/'result.json').read_text())
            assert (d/'complete.json').exists() and r['valid'] and r['case']==c and r['config']==cfg and r['manifest_hash']==m['manifest_hash']
            h,a=audit_episode(cfg,d,r);dd[key(c)]=h;blocks+=a['blocks'];cold.append(a['cold']);changes+=a['changes'];selectors+=a['selectors'];coverage+=bool(a['blocks'])
            ref=Path(m['baseline']['v14_jacobian_spaced_extrapolate'][ck(c)]['directory']);old_identity=identity(ref)
            assert {k:v for k,v in h['identity'].items() if k!='first_input_hash'}=={k:v for k,v in old_identity.items() if k!='first_input_hash'}
            if h['identity']['first_input_hash']!=old_identity['first_input_hash']:reference_exceptions.append(dict(condition=cfg['id'],case=c,errors=['first_input_hash']))
            for q in a['requests']:
                requests.append(dict(condition=cfg['id'],**c,replan=q['replan'],step=q['step'],path=q['route']['path'],policy_ms=q['timing']['policy_wall_ms'],selector_ms=q['timing']['selector_total_ms'],selector_capture_ms=q['timing']['selector_capture_ms'],rpc_ms=q['rpc_ms']))
            episodes.append(dict(condition=cfg['id'],**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],cold_s=a['cold'],mask_changes=a['changes'],selectors=a['selectors']))
        data[cfg['id']]=dd;quality_all.append(quality(cfg['id'],cfg,dd,set(dd)))
        if cfg['method']!='global_fixed':quality_small.append(quality(cfg['id'],cfg,dd,small_keys))
        steady=[q for block in blocks for q in block]
        def mean(values):return float(np.mean(values)) if values else None
        costs.append(dict(condition=cfg['id'],n=cfg['n'],steady_covered_episodes=coverage,steady_complete_cycles=len(blocks),cycle_policy_ms=mean([q['timing']['policy_wall_ms'] for q in steady]),
                          full_policy_ms=mean([q['timing']['policy_wall_ms'] for q in steady if q['route']['path']=='full']),fast_policy_ms=mean([q['timing']['policy_wall_ms'] for q in steady if q['route']['path']=='fast']),
                          mean_cold_s=mean(cold),mask_changes=changes,selection_events=selectors,max_run=1))
        atomic(OUT/'condition_audits'/f"{cfg['id']}.json",dict(passed=True,n=cfg['n'],period=5,max_run=1,selectors=selectors,mask_changes=changes))
        print(json.dumps(dict(audited_condition=cfg['id'],n=cfg['n'])),flush=True)
    # Historical comparisons are restricted to Jacobian spaced, whose legal
    # masks and scoring match; other old rows also changed the legal space.
    for name,bank in m['baseline'].items():
        dd={}
        for c in m['cases']:
            ref=bank[ck(c)];d=Path(ref['directory']);assert all(sha(d/f)==v for f,v in ref['hashes'].items())
            h=header(d)
            with np.load(d/'arrays.npz') as z:h['gradient0']=z['gate_gradient_0'].copy()
            dd[key(c)]=h
        data['cycle3_'+name]=dd
    comparisons=[]
    for budget in (14,12):
        for method in ('cosine','jacobian','random_online','random_fixed'):
            comparisons.append((f'v{budget}_{method}_none',f'v{budget}_{method}_extrapolate',small_keys if method.startswith('random') else all_keys,True,'compensation'))
        for comp in ('none','extrapolate'):
            comparisons.append((f'v{budget}_cosine_{comp}',f'v{budget}_jacobian_{comp}',all_keys,False,'main200_methods'))
            for a,b in itertools.combinations(('cosine','jacobian','random_online','random_fixed'),2):
                comparisons.append((f'v{budget}_{a}_{comp}',f'v{budget}_{b}_{comp}',small_keys,False,'matched40_methods'))
            comparisons.append((f'cycle3_v{budget}_jacobian_spaced_{comp}',f'v{budget}_jacobian_{comp}',all_keys,True,'cycle3_to_cycle5'))
    comparisons.append(('v12_global_fixed_none','v12_global_fixed_extrapolate',all_keys,True,'compensation'))
    for comp in ('none','extrapolate'):comparisons.append((f'v12_global_fixed_{comp}',f'v12_jacobian_{comp}',all_keys,False,'main200_methods'))
    strict=[];sensitivity=[];exceptions=[]
    for a,b,keys,same_mask,cohort in comparisons:
        for scope,target in (('strict',strict),('all_cases',sensitivity)):
            row,ex=paired(a,b,data[a],data[b],keys,scope,same_mask);row['comparison_cohort']=cohort;target.append(row)
            if scope=='strict':exceptions+=ex
    for file,rs in (('summary.csv',quality_all),('four_methods_40.csv',quality_small),('episodes.csv',episodes),('requests.csv',requests),('costs.csv',costs),('paired.csv',strict),('paired_sensitivity.csv',sensitivity)):write_csv(OUT/file,rs)
    atomic(OUT/'prefix_audit.json',dict(reference_initial_exceptions=reference_exceptions,pair_exceptions=exceptions,all_valid_outcomes_retained=True))
    assert len(episodes)==2320 and sum(r['n'] for r in quality_all)==2320
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
    throughput=dict(new_episodes=2320,evaluation_wall_s=wall,episodes_per_hour=2320*3600/wall,actions_per_s=sum(r['actions'] for r in episodes)/wall,requests_per_s=sum(r['predictions'] for r in episodes)/wall,
                    service_startup_s=[w['startup']['startup_s'] for w in workers],technical_retries=retries,workers=workers,controller_start=json.loads((OUT/'process.json').read_text())['time'])
    atomic(OUT/'throughput.json',throughput)
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle5_spaced_report.md'
    lines=['# 周期5：完整选层结果，全部只允许间隔跳层','','Full-D10→Fast-D2×4，V14跳4层、V12跳6层，原始一次Full10正权Jacobian。外推首周期hold，其后ratio0.2/0.4/0.6/0.8。全部2320条为新评测。','','|方法|V14无补偿|V14外推|V12无补偿|V12外推|','|---|---|---|---|---|']
    def value(method,budget,comp,table):
        r=next((r for r in table if r['method']==method and r['budget']==budget and r['compensation']==comp),None)
        return '—' if r is None else f"{r['success']}/{r['n']}（{100*r['success']/r['n']:.1f}%）"
    for method in ('cosine','jacobian','global_fixed'):lines.append('|'+method+'|'+'|'.join(value(method,b,c,quality_all) for b in (14,12) for c in ('none','extrapolate'))+'|')
    lines+=['',f"全局固定V12为新预声明合法组合{m['global_fixed']}；旧组合含相邻跳层，未沿用。",'', '相同40条子集：','','|方法|V14无补偿|V14外推|V12无补偿|V12外推|','|---|---|---|---|---|']
    for method in ('cosine','jacobian','random_online','random_fixed'):lines.append('|'+method+'|'+'|'.join(value(method,b,c,quality_small) for b in (14,12) for c in ('none','extrapolate'))+'|')
    lines+=['','|条件|稳态均摊ms/请求|Full ms|Fast ms|覆盖回合|完整周期数|平均冷捕获秒|','|---|---|---|---|---|---|---|']
    def number(v):return '—' if v is None else f'{v:.2f}'
    for r in costs:lines.append(f"|{r['condition']}|{number(r['cycle_policy_ms'])}|{number(r['full_policy_ms'])}|{number(r['fast_policy_ms'])}|{r['steady_covered_episodes']}/{r['n']}|{r['steady_complete_cycles']}|{number(r['mean_cold_s'])}|")
    lines+=['','稳态仅统计第二Full起完整五请求块，冷捕获、首周期、RPC队列单列，不把Full峰值当作均摊时延。','', '|基准→候选|比较组|严格n|恢复/新增失败|task等权变化pp|探索95%区间pp|','|---|---|---|---|---|---|']
    for r in strict:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['comparison_cohort']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['task_balanced_delta_pp']:+.2f}|[{number(r['ci_low_pp'])},{number(r['ci_high_pp'])}]|")
    lines+=['',f"实际评测窗口{wall/60:.2f}分钟，{throughput['episodes_per_hour']:.1f}回合/小时，{throughput['actions_per_s']:.2f}动作/秒，{throughput['requests_per_s']:.2f}请求/秒；模型准备秒数{throughput['service_startup_s']}。技术重试{len(retries)}回合。",'', '同一随机根seed，非随机种子总体均值；40条为200条子集。suite分层task聚类bootstrap2000次、seed281034，未校正多重比较；完整样本描述与严格配对范围不同，前缀异常在prefix_audit.json中列出，有效回合全部保留。', '', '仅Jacobian spaced与旧周期3构成相同选层规则的周期比较；余弦和随机的旧周期3使用max2，旧全局固定mask也不同。PyTorch Full10为原生BF16近似回放、一次正权VJP的局限延续。', '', '[冻结方案](idea1_v3_4_E2_cycle5_spaced_plan.md)']
    report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=2320,source_hashes=True,period=5,max_run=1,own_all17_two_full_histories=True,full10_one_vjp=True,pair_prefix_exceptions=len(exceptions),time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',completed=2320,time=time.time()))

if __name__=='__main__':main()
