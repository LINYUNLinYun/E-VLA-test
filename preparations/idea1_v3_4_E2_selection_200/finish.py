import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,PLAN,BASE,EXTRAP,FIXED,SUITES,CONDITIONS,sha,ck,source,old_directory,legal,random_mask
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv,paired_ci

def key(c):return tuple(c[k] for k in ('suite','task','init'))
def rows(d):return [json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()]

def header(d,r=None):
    if r is None:r=json.loads((d/'result.json').read_text())
    qs=rows(d)
    with np.load(d/'arrays.npz') as z:
        h=dict(result=r,identity=identity(d),first_mask=qs[0]['route']['selected_skip'],raw0=z['raw_0'].copy(),actions0=z['actions_0'].copy(),fast_input=None if len(qs)<2 else qs[1]['input_hash'])
        if len(qs)>1:h.update(raw1=z['raw_1'].copy(),actions1=z['actions_1'].copy())
    return h

def prefix(a,b,same_mask=False,first_fast_output=False):
    errors=[]
    if a['identity']['first_input_hash']!=b['identity']['first_input_hash']:errors.append('first_input_hash')
    for field in ('raw0','actions0'):
        if not np.allclose(a[field],b[field],atol=1e-5,rtol=1e-5):errors.append(field)
    if a['fast_input']!=b['fast_input']:errors.append('first_fast_input')
    if same_mask and a['first_mask']!=b['first_mask']:errors.append('first_mask')
    if first_fast_output:
        for field in ('raw1','actions1'):
            if field in a and field in b and not np.allclose(a[field],b[field],atol=1e-5,rtol=1e-5):errors.append(field)
    return errors

def paired(label,name,aa,bb,keys,scope,same_mask=False,fast_output=False):
    excluded={k:prefix(aa[k],bb[k],same_mask,fast_output) for k in keys};excluded={k:e for k,e in excluded.items() if e}
    selected=sorted(keys-set(excluded)) if scope=='strict' else sorted(keys)
    ds=[dict(suite=k[0],task=k[1],delta=int(bb[k]['result']['success'])-int(aa[k]['result']['success'])) for k in selected]
    ci=paired_ci(ds)
    point=np.mean([np.mean([np.mean([r['delta'] for r in ds if r['suite']==s and r['task']==t]) for t in sorted({r['task'] for r in ds if r['suite']==s})]) for s in SUITES])
    row=dict(baseline=label,candidate=name,scope=scope,n=len(selected),excluded_prefix_cases=len(excluded),baseline_success=sum(aa[k]['result']['success'] for k in selected),candidate_success=sum(bb[k]['result']['success'] for k in selected),recovered=sum(not aa[k]['result']['success'] and bb[k]['result']['success'] for k in selected),lost=sum(aa[k]['result']['success'] and not bb[k]['result']['success'] for k in selected),task_balanced_delta_pp=float(point*100),episode_delta_pp=float(np.mean([r['delta'] for r in ds])*100),ci_low_pp=None if ci is None else ci[0]*100,ci_high_pp=None if ci is None else ci[1]*100)
    return row,[dict(baseline=label,candidate=name,case=k,errors=e) for k,e in excluded.items()]

def finish_random():
    m=json.loads((OUT/'manifest.json').read_text());small=m['small_cases'];keys={key(c) for c in small};data={};quality=[];audit=[]
    for budget in (14,12):
        for method in ('cosine','jacobian','random_online','random_fixed'):
            for comp in ('none','extrapolate'):
                name=f'v{budget}_{method}_{comp}';dd={}
                for c in small:
                    if method.startswith('random') and comp=='none':d=old_directory(BASE,f'v{budget}_{method}',c)
                    else:
                        cfg=next(cfg for cfg in CONDITIONS if cfg['id']==name);d,_=source(m,cfg,c)
                    assert d is not None and (d/'complete.json').exists(),(name,c)
                    h=header(d);dd[key(c)]=h
                data[name]=dd
                quality.append(dict(condition=name,budget=budget,method=method,compensation=comp,n=40,success=sum(h['result']['success'] for h in dd.values()),**{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
    pairs=[];sensitivity=[]
    for budget in (14,12):
        for method in ('cosine','jacobian','random_online','random_fixed'):
            a=f'v{budget}_{method}_none';b=f'v{budget}_{method}_extrapolate'
            for scope,target in [('strict',pairs),('all40',sensitivity)]:
                row,errors=paired(a,b,data[a],data[b],keys,scope,same_mask=True);target.append(row)
                if scope=='strict':audit+=errors
        a=f'v{budget}_random_online_extrapolate';b=f'v{budget}_random_fixed_extrapolate'
        errors=[dict(budget=budget,case=k,errors=prefix(data[a][k],data[b][k],True,True)) for k in keys];errors=[e for e in errors if e['errors']];audit+=errors
    for file,rs in [('four_methods_40.csv',quality),('four_methods_40_paired.csv',pairs),('four_methods_40_sensitivity.csv',sensitivity)]:write_csv(OUT/file,rs)
    atomic(OUT/'random40_prefix_audit.json',dict(exceptions=audit,paired_comparisons=8,random_pair_first_request_checks=80))
    lines=['# 周期3四种选层：40场景补偿完整对照','','随机补偿160条新回合，其他结果复用已完成有效记录。相同case、Full-D10→Fast-D2→Fast-D2、原始一次Full10 Jacobian、随机根seed341101。','','|方法|V14无补偿|V14外推|V12无补偿|V12外推|','|---|---|---|---|---|']
    for method in ('cosine','jacobian','random_online','random_fixed'):
        values=[next(r['success'] for r in quality if r['budget']==b and r['method']==method and r['compensation']==c) for b,c in [(14,'none'),(14,'extrapolate'),(12,'none'),(12,'extrapolate')]]
        lines.append('|'+method+'|'+'|'.join(f'{v}/40' for v in values)+'|')
    lines+=['',f'相关共同前缀异常记录{len(audit)}项，严格配对按冻结计划排除，完整40条描述保留；细节见random40_prefix_audit.json。一个预设随机根seed，不代表随机种子总体。']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_selection_random40_report.md';report.write_text('\n'.join(lines)+'\n');atomic(OUT/'random40_complete.json',dict(report=str(report),time=time.time()))

def audit_episode(cfg,d,r,reused):
    qs=rows(d);comp=cfg['compensation']=='extrapolate';method=cfg['method'];k=cfg['skip_count'];hist=[];mask=None;mask_step=None;selectors=0;changes=0
    with np.load(d/'arrays.npz') as z:
        for i,q in enumerate(qs):
            rt=q['route'];full=i%3==0;refresh=full if method in ('cosine','jacobian') else i==0 or (method=='random_online' and not full and i!=1)
            assert rt['path']==('full' if full else 'fast') and rt['D']==(10 if full else 2) and len(q['timing']['nodes'])==(11 if full else 3)
            assert legal(rt['selected_skip'],k) and rt['mask_refresh']==refresh
            assert np.isfinite(z[f'raw_{i}']).all() and np.isfinite(z[f'actions_{i}']).all()
            if refresh:
                selectors+=1;changes+=int(mask is not None and mask!=rt['selected_skip']);mask=rt['selected_skip'];mask_step=q['step']
                if method.startswith('random'):assert mask==random_mask(cfg['random_seed'],q['seed'],k)
                else:
                    assert z[f'layer_score_{i}'].shape==(17,) and np.isfinite(z[f'layer_score_{i}']).all()
                    if method=='jacobian':assert z[f'gate_gradient_{i}'].shape==(17,) and np.isfinite(z[f'gate_gradient_{i}']).all()
            else:assert q['timing']['selector_total_ms']==0
            assert rt['selected_skip']==mask and rt['mask_age_env_steps']==q['step']-mask_step and q['counts']['selector']==selectors
            if method=='jacobian':assert rt['jacobian_model_steps']==10
            if not reused:assert rt['steady_vjp_count']==int(full and method=='jacobian')
            if full:
                assert rt['skip']==[]
                if comp:hist=(hist+[q['step']])[-2:]
            else:
                assert rt['skip']==mask and rt['full_age_env_steps']==i%3*5
                if method in ('cosine','jacobian'):assert q['timing']['selector_total_ms']==0
            if comp:
                assert rt['captured_layers']==list(range(1,18)) and rt['history_full_steps']==hist
                assert rt['patch_layers']==([] if full else mask)
                if not full:
                    age=q['step']-hist[-1];gap=None if len(hist)<2 else hist[-1]-hist[-2]
                    assert age==i%3*5 and rt['has_second_history']==(gap is not None)
                    assert np.isclose(rt['extrapolation_ratio'],0 if gap is None else age/gap)
        h=dict(result=r,identity=identity(d),first_mask=qs[0]['route']['selected_skip'],raw0=z['raw_0'].copy(),actions0=z['actions_0'].copy(),fast_input=None if len(qs)<2 else qs[1]['input_hash'])
    cycles=[float(np.mean([q['timing']['policy_wall_ms'] for q in qs[i:i+3]])) for i in range(3,len(qs)-2,3)]
    return h,dict(mask_changes=changes,selectors=selectors,cold_s=sum(q['timing']['selector_capture_ms'] for q in qs)/1000,cycle_ms=float(np.mean(cycles)) if cycles else None,cycles=cycles,requests=qs)

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in m['references'].items())
    for refs in m['reuse'].values():
        for ref in refs.values():assert all(sha(Path(ref['directory'])/p)==h for p,h in ref['hashes'].items())
    data={};episodes=[];quality=[];timings=[];first_fixed=[];keys={key(c) for c in m['cases']};small={key(c) for c in m['small_cases']};fresh=keys-small
    fixed={comp:{} for comp in ('none','extrapolate')}
    for comp,bank in [('none',m['fixed_none']),('extrapolate',m['fixed'])]:
        for c in m['cases']:
            ref=bank[ck(c)];d=Path(ref['directory']);assert sha(d/'result.json')==ref['result_sha256'] and sha(d/'identity.json')==ref['identity_sha256'];fixed[comp][key(c)]=header(d)
    for cfg in CONDITIONS:
        dd={};cycle_all=[];cold=[];new_wall=0.;new_n=0;reused_n=0
        for c in (m['small_cases'] if cfg['phase']=='random40' else m['cases']):
            d,reused=source(m,cfg,c);assert d is not None and (d/'complete.json').exists();r=json.loads((d/'result.json').read_text());assert r['valid'] and r['case']==c
            if not reused:assert r['manifest_hash']==m['manifest_hash']
            h,extra=audit_episode(cfg,d,r,reused);dd[key(c)]=h;baseline=fixed['extrapolate'][key(c)]
            assert {k:v for k,v in h['identity'].items() if k!='first_input_hash'}=={k:v for k,v in baseline['identity'].items() if k!='first_input_hash'}
            errors=prefix(baseline,h)
            if errors:first_fixed.append(dict(condition=cfg['id'],case=c,errors=errors))
            cycle_all+=extra['cycles'];cold.append(extra['cold_s']);qs=extra['requests']
            for q in qs[1:]:timings.append(dict(condition=cfg['id'],suite=c['suite'],task=c['task'],init=c['init'],reused=reused,path=q['route']['path'],policy_ms=q['timing']['policy_wall_ms'],rpc_ms=q['rpc_ms'],selector_ms=q['timing']['selector_total_ms']))
            episodes.append(dict(condition=cfg['id'],**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],reused=reused,cold_s=extra['cold_s'],mask_changes=extra['mask_changes'],cycle_ms=extra['cycle_ms']))
            if reused:reused_n+=1
            else:new_n+=1;new_wall+=r['wall_s']
        data[cfg['id']]=dd
        for scope,kk in [('all',set(dd)),('original40',set(dd)&small),('new160',set(dd)&fresh)]:
            if not kk:continue
            quality.append(dict(condition=cfg['id'],budget=cfg['budget'],method=cfg['method'],compensation=cfg['compensation'],scope=scope,n=len(kk),success=sum(dd[k]['result']['success'] for k in kk),success_rate=float(np.mean([dd[k]['result']['success'] for k in kk])),**{s:sum(dd[k]['result']['success'] for k in kk if k[0]==s) for s in SUITES}))
        atomic(OUT/'condition_audits'/f"{cfg['id']}.json",dict(passed=True,n=len(dd),reused=reused_n,new=new_n,cycle_policy_ms=float(np.mean(cycle_all)),mean_cold_s=float(np.mean(cold)),sum_new_episode_wall_s=new_wall))
        print(json.dumps(dict(audited_condition=cfg['id'],n=len(dd))),flush=True)
    pairs=[];sens=[];exceptions=[]
    comparisons=[]
    for budget in (14,12):
        for method in ('cosine','jacobian'):comparisons.append((f'v{budget}_{method}_none',f'v{budget}_{method}_extrapolate',True))
        for comp in ('none','extrapolate'):comparisons.append((f'v{budget}_cosine_{comp}',f'v{budget}_jacobian_{comp}',False))
    for a,b,same in comparisons:
        for cohort,kk in [('all200',keys),('new160',fresh),('original40',small)]:
            for scope,target in [('strict',pairs),('all_cases',sens)]:
                r,ex=paired(a,b,data[a],data[b],kk,scope,same);r['cohort']=cohort;target.append(r)
                if scope=='strict':exceptions+=ex
    for cfg in [c for c in CONDITIONS if c['budget']==12 and c['phase']=='expand200']:
        comp=cfg['compensation'];a=f'global_fixed_v12_{comp}';b=cfg['id']
        for scope,target in [('strict',pairs),('all_cases',sens)]:
            r,ex=paired(a,b,fixed[comp],data[b],keys,scope);r['cohort']='all200';target.append(r)
            if scope=='strict':exceptions+=ex
    for file,rs in [('episodes.csv',episodes),('summary.csv',quality),('requests.csv',timings),('paired.csv',pairs),('paired_sensitivity.csv',sens)]:write_csv(OUT/file,rs)
    atomic(OUT/'prefix_audit.json',dict(pair_exceptions=exceptions,fixed_reference_exceptions=first_fixed,all_valid_outcomes_retained=True))
    assert len(episodes)==1760 and sum(r['reused'] for r in episodes)==320
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/compatibility_gate.json')];assert gates and all(g['passed'] for g in gates)
    atomic(OUT/'audit.json',dict(passed=True,records=1760,new_episodes=1440,reused_episodes=320,source_hashes=True,route_and_history=True,shared_policy_isolation_gated=True,scientific_failures_retained=True,time=time.time()))
    lines=['# 周期3余弦/Jacobian：200场景扩样','','4 suites×10 task×init19..23，每条件200；原40复用、新160独立列出。8个扩样条件共1600，另有4个随机补偿条件各40，共1760记录；本轮新1440，复用320。Jacobian为原始一次正权Full-D10评分，每Full刷新，不混用后续误差评分/三周期变体。','','|方法|预算|无补偿200|外推200|无补偿新160|外推新160|','|---|---|---|---|---|---|']
    for budget in (14,12):
        for method in ('cosine','jacobian'):
            nums=[next(r for r in quality if r['condition']==f'v{budget}_{method}_{comp}' and r['scope']==scope) for scope in ('all','new160') for comp in ('none','extrapolate')]
            lines.append('|'+ '|'.join([method,f'V{budget}',*[f"{r['success']}/{r['n']}" for r in nums]])+'|')
    lines+=['','全局固定V12同case既有参考：无补偿'+str(sum(h['result']['success'] for h in fixed['none'].values()))+'/200，外推'+str(sum(h['result']['success'] for h in fixed['extrapolate'].values()))+'/200。不等同于每回合固定随机。','','|配对|cohort|n|恢复/新增失败|task等权差值pp|探索95%区间pp|','|---|---|---|---|---|---|']
    for r in pairs:
        if r['cohort']=='original40':continue
        lines.append(f"|{r['baseline']}→{r['candidate']}|{r['cohort']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['task_balanced_delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['',f'共同前缀异常：相关配对累计{len(exceptions)}项，完整描述保留、严格配对剔除，all-case敏感性另表；同一个case可在多项比较中重复计数。固定参考异常{len(first_fixed)}项。所有官方初态/控制协议、源hash、Full/Fast路由、单次Full10梯度、随机seed和本路径双历史核验通过。','', '新160为任务不均初态数（偶task5个、奇task3个），配对点与bootstrap统一suite/task等权，同时保留逐episode差值。总体200每task5个，点估计一致。探索区间未校正多重比较；原40参与多次先前探索，不能当作全新验证。','', '成本与吞吐见condition_audits、requests.csv与throughput.json。复用40含旧单环境时延、新160可能双环境；双环境RPC含队列、policy内计时不含队列。自由轨迹均值不是同输入加速证明。旧PyTorch D10导数仍为原生BF16近似，不是精确原生Jacobian。','','[随机补偿40完整表](idea1_v3_4_E2_selection_random40_report.md)','[冻结方案](idea1_v3_4_E2_selection_200_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_selection_200_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',time=time.time()))

if __name__=='__main__':main()
