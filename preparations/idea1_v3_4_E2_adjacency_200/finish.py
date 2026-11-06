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
    qs=rows(d);mask=None;changes=0;selections=0;run_counts={};cold=0.;k=cfg['skip_count'];rule=cfg['adjacency_rule']
    with np.load(d/'arrays.npz') as z:
        for i,q in enumerate(qs):
            rt=q['route'];full=i%3==0
            assert q['step']==i*5 and rt['path']==('full' if full else 'fast') and rt['D']==(10 if full else 2)
            assert len(q['timing']['nodes'])==(11 if full else 3) and rt['mask_refresh']==full
            assert rt['adjacency_rule']==rule and rt['compensation']=='none' and rt['steady_vjp_count']==int(full)
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
            h=header(d)
            if 'jacobian' in name:
                with np.load(d/'arrays.npz') as z:h['gradient0']=z['gate_gradient_0'].copy()
            dd[key(c)]=h
        data[name]=dd
        quality.append(dict(condition=name,budget=int(name[1:3]),rule='max2',method='jacobian' if 'jacobian' in name else 'cosine',reused=True,n=200,success=sum(h['result']['success'] for h in dd.values()),**{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
    for cfg in CONDITIONS:
        dd={};cycles=[];cold=[];hist={};changes=selections=0;full_ms=[];fast_ms=[]
        for c in m['cases']:
            d=directory(cfg['id'],c);r=json.loads((d/'result.json').read_text())
            assert (d/'complete.json').exists() and r['valid'] and r['case']==c and r['config']==cfg and r['manifest_hash']==m['manifest_hash']
            h,a=audit_episode(cfg,d,r);dd[key(c)]=h;baseline=data[f"v{cfg['budget']}_jacobian_none"][key(c)]
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
        quality.append(dict(condition=cfg['id'],budget=cfg['budget'],rule=cfg['adjacency_rule'],method='jacobian',reused=False,n=200,success=sum(h['result']['success'] for h in dd.values()),**{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
        costs.append(dict(condition=cfg['id'],cycle_policy_ms=float(np.mean(cycles)),full_policy_ms=float(np.mean(full_ms)),fast_policy_ms=float(np.mean(fast_ms)),mean_cold_s=float(np.mean(cold)),mask_changes=changes,selection_events=selections,max_run=max(hist),run_histogram=json.dumps(hist),fraction_run_ge3=sum(n for run,n in hist.items() if run>=3)/selections))
        atomic(OUT/'condition_audits'/f"{cfg['id']}.json",dict(passed=True,n=200,run_histogram=hist,selections=selections,mask_changes=changes))
        print(json.dumps(dict(audited_condition=cfg['id'],n=200)),flush=True)
    pairs=[];sensitivity=[];pair_exceptions=[]
    for budget in (14,12):
        base=f'v{budget}_jacobian_none';cos=f'v{budget}_cosine_none';u=f'v{budget}_jacobian_unrestricted_none';s=f'v{budget}_jacobian_spaced_none'
        for a,b in ((base,u),(base,s),(u,s),(cos,u),(cos,s)):
            for scope,target in (('strict',pairs),('all200',sensitivity)):
                row,ex=paired(a,b,data[a],data[b],keys,scope);target.append(row)
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
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=800,baseline_references=800,source_hashes=True,mask_optimizer_and_execution=True,full10_one_vjp=True,prefix_exceptions=len(exceptions),time=time.time()))
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_adjacency_200_report.md'
    lines=['# 无补偿Jacobian：连续跳层规则200例对照','','每个预算/新规则200条，共800条新回合。Full-D10→Fast-D2→Fast-D2，原始一次正权VJP，每Full更新mask；V14跳4、V12跳6、候选L1–17、AE全18层；其余既有协议保持。unrestricted允许但不强制三连及以上；spaced完全禁止相邻跳过。max2和余弦是同200case旧结果复用。','','|方法/规则|V14成功数|V12成功数|','|---|---|---|']
    for label,template in [('余弦，最多连续2','v{}_cosine_none'),('Jacobian，最多连续2','v{}_jacobian_none'),('Jacobian，无连续上限','v{}_jacobian_unrestricted_none'),('Jacobian，只许间隔跳','v{}_jacobian_spaced_none')]:
        values=[next(r['success'] for r in quality if r['condition']==template.format(b)) for b in (14,12)]
        lines.append(f'|{label}|{values[0]}/200（{values[0]/2:.1f}%）|{values[1]}/200（{values[1]/2:.1f}%）|')
    lines+=['','|基准→候选|严格配对n|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['task_balanced_delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','|条件|实际最大连续跳层|选层中三连及以上比例|稳态周期均摊ms/请求|Full ms|Fast ms|平均冷捕获秒|','|---|---|---|---|---|---|---|']
    for r in costs:lines.append(f"|{r['condition']}|{r['max_run']}|{r['fraction_run_ge3']*100:.2f}%|{r['cycle_policy_ms']:.2f}|{r['full_policy_ms']:.2f}|{r['fast_policy_ms']:.2f}|{r['mean_cold_s']:.2f}|")
    lines+=['',f'实际闭环窗口{wall/60:.2f}分钟，{throughput["episodes_per_hour"]:.1f}回合/小时、{throughput["actions_per_s"]:.2f}执行动作/秒、{throughput["requests_per_s"]:.2f}模型请求/秒；窗口含各回合冷启动、环境、排队与落盘，模型服务准备单列throughput.json。技术重试{len(retries)}回合。',f'相对max2共同前缀异常{len(exceptions)}项、相关配对累计异常{len(pair_exceptions)}项；所有有效结果保留，严格配对剔除对应异常，完整200敏感性见paired_sensitivity.csv。', '','200case为4 suites×task0..9×init19..23，完整suite细目见summary.csv。探索bootstrap为suite分层task聚类2000次，未校正多重比较；本批case曾用于此前探索，不声称未见验证。每个mask属于各自自由轨迹，不能从三连频率与成功率关系作额外因果归因。', '', '稳态周期从第二Full起按完整三请求块平均；policy计时不含RPC排队，Full/Fast表排除各回合第一个请求，冷捕获单列。不同自由轨迹时延不作为同输入加速证明。评分所在Full的额外反传成本保留，单VJP及PyTorch D10近似原生BF16的局限不变。', '', '[冻结方案](idea1_v3_4_E2_adjacency_200_plan.md)']
    report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',time=time.time()))

if __name__=='__main__':main()
