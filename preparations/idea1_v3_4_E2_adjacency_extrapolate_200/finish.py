import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,BASE,SUITES,CONDITIONS,sha,ck,directory,choose,legal,longest_run
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v3_4_E2_selection_200.finish import header,prefix,paired

def key(c):return tuple(c[k] for k in ('suite','task','init'))
def rows(d):return [json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]

def audit_episode(cfg,d,r):
    qs=rows(d);mask=None;changes=0;selections=0;run_counts={};history=[];cold=0.;k=cfg['skip_count'];rule=cfg['adjacency_rule']
    with np.load(d/'arrays.npz') as z:
        for i,q in enumerate(qs):
            rt=q['route'];full=i%3==0
            assert q['step']==i*5 and rt['path']==('full' if full else 'fast') and rt['D']==(10 if full else 2)
            assert len(q['timing']['nodes'])==(11 if full else 3) and rt['mask_refresh']==full
            assert rt['adjacency_rule']==rule and rt['compensation']=='extrapolate' and rt['steady_vjp_count']==int(full)
            assert rt['jacobian_model_steps']==10 and rt['jacobian_output_steps']==[1,2,3,4,5] and rt['jacobian_full_layers']==18
            assert legal(rt['selected_skip'],k,rule) and rt['selected_longest_run']==longest_run(rt['selected_skip'])
            assert np.isfinite(z[f'raw_{i}']).all() and np.isfinite(z[f'actions_{i}']).all()
            if full:
                score=z[f'layer_score_{i}'];grad=z[f'gate_gradient_{i}'];assert score.shape==grad.shape==(17,) and np.isfinite(grad).all()
                assert np.array_equal(score,abs(grad)) and choose(score,k,rule)==rt['selected_skip']
                assert rt['skip']==[];selections+=1;changes+=int(mask is not None and mask!=rt['selected_skip']);mask=rt['selected_skip']
                run=longest_run(mask);run_counts[run]=run_counts.get(run,0)+1
            else:
                assert rt['skip']==mask==rt['selected_skip'] and q['timing']['selector_total_ms']==0
            assert rt['mask_age_env_steps']==i%3*5 and rt['full_age_env_steps']==i%3*5 and q['counts']['selector']==selections
            if full:history=(history+[q['step']])[-2:]
            assert rt['captured_layers']==list(range(1,18)) and rt['history_full_steps']==history
            assert rt['patch_layers']==([] if full else mask)
            if not full:
                gap=None if len(history)<2 else history[-1]-history[-2]
                assert rt['has_second_history']==(gap is not None)
                assert np.isclose(rt['extrapolation_ratio'],0. if gap is None else (q['step']-history[-1])/gap)
            cold+=q['timing']['selector_capture_ms']/1000
        h=dict(result=r,identity=identity(d),first_mask=qs[0]['route']['selected_skip'],raw0=z['raw_0'].copy(),actions0=z['actions_0'].copy(),fast_input=None if len(qs)<2 else qs[1]['input_hash'],gradient0=z['gate_gradient_0'].copy())
    cycles=[float(np.mean([q['timing']['policy_wall_ms'] for q in qs[i:i+3]])) for i in range(3,len(qs)-2,3)]
    return h,dict(changes=changes,selections=selections,runs=run_counts,cold=cold,cycles=cycles,requests=qs)

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==v for p,v in m['source_files'].items()) and all(sha(ROOT/p)==v for p,v in m['references'].items())
    data={};quality=[];episodes=[];requests=[];costs=[];exceptions=[];keys={key(c) for c in m['cases']}
    for name,bank in m['baseline'].items():
        dd={}
        for c in m['cases']:
            ref=bank[ck(c)];d=Path(ref['directory']);assert all(sha(d/p)==v for p,v in ref['hashes'].items())
            if name=='global_fixed_v12_extrapolate':
                r=json.loads((d/'result.json').read_text());cfg=r['config'];assert cfg['skip']==[2,3,5,14,16,17] and cfg['compensation']=='extrapolate' and cfg['period']==2
                qs=rows(d)
                with np.load(d/'arrays.npz') as z:h=dict(result=r,identity=identity(d),first_mask=cfg['skip'],raw0=z['raw_0'].copy(),actions0=z['actions_0'].copy(),fast_input=qs[1]['input_hash'] if len(qs)>1 else None)
            else:h=header(d)
            if 'jacobian' in name:
                with np.load(d/'arrays.npz') as z:h['gradient0']=z['gate_gradient_0'].copy()
            dd[key(c)]=h
        data[name]=dd
        quality.append(dict(condition=name,budget=12 if name.startswith('global_fixed') else int(name[1:3]),rule='unrestricted' if 'unrestricted' in name else 'spaced' if 'spaced' in name else 'fixed' if 'global_fixed' in name else 'max2',method='jacobian' if 'jacobian' in name else 'fixed' if name.startswith('global_fixed') else 'cosine',compensation='none' if name.endswith('_none') else 'extrapolate',reused=True,n=200,success=sum(h['result']['success'] for h in dd.values()),**{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
    for cfg in CONDITIONS:
        dd={};cycles=[];cold=[];hist={};changes=selections=0;full_ms=[];fast_ms=[]
        for c in m['cases']:
            d=directory(cfg['id'],c);r=json.loads((d/'result.json').read_text())
            assert (d/'complete.json').exists() and r['valid'] and r['case']==c and r['config']==cfg and r['manifest_hash']==m['manifest_hash']
            h,a=audit_episode(cfg,d,r);dd[key(c)]=h;baseline=data[f"v{cfg['budget']}_jacobian_extrapolate"][key(c)]
            assert {k:v for k,v in h['identity'].items() if k!='first_input_hash'}=={k:v for k,v in baseline['identity'].items() if k!='first_input_hash'}
            ex=prefix(baseline,h)
            if not ex:assert np.allclose(h['gradient0'],baseline['gradient0'],atol=1e-4,rtol=1e-3),('Common Full gradient',cfg['id'],c)
            else:exceptions.append(dict(condition=cfg['id'],case=c,errors=ex))
            cycles+=a['cycles'];cold.append(a['cold']);changes+=a['changes'];selections+=a['selections']
            for run,n in a['runs'].items():hist[run]=hist.get(run,0)+n
            for q in a['requests'][1:]:
                ms=q['timing']['policy_wall_ms'];(full_ms if q['route']['path']=='full' else fast_ms).append(ms)
                requests.append(dict(condition=cfg['id'],**c,path=q['route']['path'],policy_ms=ms,selector_ms=q['timing']['selector_total_ms'],rpc_ms=q['rpc_ms']))
            episodes.append(dict(condition=cfg['id'],**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],cold_s=a['cold'],mask_changes=a['changes'],selectors=a['selections']))
        data[cfg['id']]=dd
        quality.append(dict(condition=cfg['id'],budget=cfg['budget'],rule=cfg['adjacency_rule'],method='jacobian',compensation='extrapolate',reused=False,n=200,success=sum(h['result']['success'] for h in dd.values()),**{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
        costs.append(dict(condition=cfg['id'],cycle_policy_ms=float(np.mean(cycles)),full_policy_ms=float(np.mean(full_ms)),fast_policy_ms=float(np.mean(fast_ms)),mean_cold_s=float(np.mean(cold)),mask_changes=changes,selection_events=selections,max_run=max(hist),run_histogram=json.dumps(hist),fraction_run_ge3=sum(n for run,n in hist.items() if run>=3)/selections))
        atomic(OUT/'condition_audits'/f"{cfg['id']}.json",dict(passed=True,n=200,run_histogram=hist,selections=selections,mask_changes=changes))
        print(json.dumps(dict(audited_condition=cfg['id'],n=200)),flush=True)
    pairs=[];sensitivity=[];pair_exceptions=[]
    for budget in (14,12):
        base=f'v{budget}_jacobian_extrapolate';cos=f'v{budget}_cosine_extrapolate';u=f'v{budget}_jacobian_unrestricted_extrapolate';sp=f'v{budget}_jacobian_spaced_extrapolate'
        comparisons=[(base,u,False),(base,sp,False),(u,sp,False),(cos,u,False),(cos,sp,False),(f'v{budget}_jacobian_unrestricted_none',u,True),(f'v{budget}_jacobian_spaced_none',sp,True)]
        if budget==12:comparisons += [('global_fixed_v12_extrapolate',u,False),('global_fixed_v12_extrapolate',sp,False)]
        for a,b,same_mask in comparisons:
            for scope,target in (('strict',pairs),('all200',sensitivity)):
                row,ex=paired(a,b,data[a],data[b],keys,scope,same_mask);target.append(row)
                if scope=='strict':pair_exceptions+=ex
    for name,rs in (('summary.csv',quality),('episodes.csv',episodes),('requests.csv',requests),('costs.csv',costs),('paired.csv',pairs),('paired_sensitivity.csv',sensitivity)):write_csv(OUT/name,rs)
    atomic(OUT/'prefix_audit.json',dict(baseline_exceptions=exceptions,pair_exceptions=pair_exceptions,all_valid_outcomes_retained=True))
    assert len(episodes)==800
    workers=[]
    for wd in sorted((OUT/'workers').glob('gpu*')):
        startup=json.loads((wd/'startup.json').read_text());state=json.loads((wd/'state.json').read_text());exits=json.loads((wd/'exits.json').read_text())
        assert state['status']=='complete' and exits['service']==-15 and all(code==0 for code in exits['evaluations'])
        assert json.loads((wd/'compatibility_gate.json').read_text())['passed']
        iso=json.loads((wd/'isolation_gate.json').read_text());assert startup['actors']==1 or iso['passed']
        workers.append(dict(identity=json.loads((wd/'identity.json').read_text()),startup=startup,state=state,exits=exits,isolation=iso))
    start=min(w['startup']['time'] for w in workers);end=max(w['state']['time'] for w in workers);wall=end-start
    retries=[str(p) for p in (OUT/'technical_attempts').rglob('*.json') if json.loads(p.read_text())['attempts']>1]
    throughput=dict(new_episodes=800,evaluation_wall_s=wall,episodes_per_hour=800*3600/wall,actions_per_s=sum(r['actions'] for r in episodes)/wall,requests_per_s=sum(r['predictions'] for r in episodes)/wall,service_startup_s=[w['startup']['startup_s'] for w in workers],technical_retries=retries,workers=workers,controller_start=json.loads((OUT/'process.json').read_text())['time'])
    atomic(OUT/'throughput.json',throughput)
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=800,baseline_references=1800,source_hashes=True,mask_optimizer_and_execution=True,own_all17_two_full_histories=True,full10_one_vjp=True,prefix_exceptions=len(exceptions),time=time.time()))
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_adjacency_extrapolate_200_report.md'
    lines=['# Jacobian连续性规则：200例外推补偿对照','','两新规则×V14/V12各200，共800新回合。每Full采集自身全部17层真实残差，保留最近两个Full；首周期hold，随后Fast ratio 1/3、2/3。Full-D10→Fast-D2→Fast-D2、原始一次正权VJP、每Full换mask，V14跳4、V12跳6、AE18；仅增加同一外推补偿。','','|方法/规则|V14无补偿|V14外推|V12无补偿|V12外推|','|---|---|---|---|---|']
    prior_summary=list(__import__('csv').DictReader((BASE/'summary.csv').open()))
    def value(name):
        if name in data:return sum(h['result']['success'] for h in data[name].values())
        return int(next(r['success'] for r in prior_summary if r['condition']==name and r['scope']=='all'))
    for label,template in [('余弦，最多连续2','v{}_cosine_{}'),('Jacobian，最多连续2','v{}_jacobian_{}'),('Jacobian，无连续上限','v{}_jacobian_unrestricted_{}'),('Jacobian，只许间隔跳','v{}_jacobian_spaced_{}')]:
        values=[value(template.format(b,c)) for b in (14,12) for c in ('none','extrapolate')]
        lines.append('|'+label+'|'+'|'.join(f'{v}/200（{v/2:.1f}%）' for v in values)+'|')
    lines+=['','全局固定V12外推同case既有参考：169/200（84.5%），固定组合[2,3,5,14,16,17]；不等同于每回合固定随机。','','|基准→候选|严格配对n|恢复/新增失败|task等权差值pp|探索95%区间pp|','|---|---|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['task_balanced_delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','|条件|实际最大连续跳层|选层中三连及以上比例|稳态周期均摊ms/请求|Full ms|Fast ms|平均冷捕获秒|','|---|---|---|---|---|---|---|']
    for r in costs:lines.append(f"|{r['condition']}|{r['max_run']}|{r['fraction_run_ge3']*100:.2f}%|{r['cycle_policy_ms']:.2f}|{r['full_policy_ms']:.2f}|{r['fast_policy_ms']:.2f}|{r['mean_cold_s']:.2f}|")
    lines += ['',f'实际闭环窗口{wall/60:.2f}分钟，{throughput["episodes_per_hour"]:.1f}回合/小时、{throughput["actions_per_s"]:.2f}执行动作/秒、{throughput["requests_per_s"]:.2f}模型请求/秒；含各回合冷启动、环境、排队及落盘，模型准备单列throughput.json。技术重试{len(retries)}回合。',f'相对max2的基线前缀异常记录{len(exceptions)}项，相关严格配对异常累计{len(pair_exceptions)}项；不同条件或比较可重复涉及同case。全部200描述结果保留，严格配对剔除对应异常、完整200敏感性另表。', '', '4 suites×task0..9×init19..23。探索bootstrap采用suite分层task聚类2000次、seed281034，未校正多重比较；已用于探索的case不视作未见验证。存在整task排除时，严格task等权区间不直接对应完整200的逐episode点估计。', '', '规则优化器、单次Full10梯度、真实残差双历史与各层捕获均核验；每个候选使用自己的观测、KV与历史。首个完整周期及冷捕获计入实际wall；稳态均摊从第二Full起按完整三请求块计，policy耗时不含双环境RPC队列。自由轨迹成本差不作同输入因果加速证明。PyTorch Full10近似原生BF16和单加权VJP局限不变。', '', '[冻结方案](idea1_v3_4_E2_adjacency_extrapolate_200_plan.md)']
    report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',time=time.time()))

if __name__=='__main__':main()
