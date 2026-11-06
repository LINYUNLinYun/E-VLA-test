import json,time
import numpy as np
from .common import ROOT,OUT,REFERENCE,STATIC_REFERENCE,PLAN,SUITES,sha,legal
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv
from preparations.idea1_v3_4_E2_cycle3_extrapolate.finish import collect

def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(a),success=sum(r['success'] for d,r in a.values())) for c in m['conditions'] for a in [collect(OUT,c['id'])]}

def blocks(qs):return [qs[i:i+18] for i in range(18,len(qs)-17,18)]

def prefix_check(d,od,qs,oq):
    errors=[]
    with np.load(d/'arrays.npz') as a,np.load(od/'arrays.npz') as b:
        assert all(np.isfinite(a[k]).all() for k in a.files)
        for i in range(min(4,len(qs),len(oq))):
            for field in ('input_hash','seed'):
                if qs[i][field]!=oq[i][field]:errors.append(dict(replan=i,field=field))
            for field in ('raw','actions'):
                x,y=a[f'{field}_{i}'],b[f'{field}_{i}']
                if not np.allclose(x,y,atol=1e-5,rtol=1e-5):errors.append(dict(replan=i,field=field,max_abs=float(abs(x-y).max())))
            if i<3 and qs[i]['route']['selected_skip']!=oq[i]['route']['selected_skip']:errors.append(dict(replan=i,field='first_mask'))
        if len(qs)>4 and (len(oq)<=4 or qs[4]['input_hash']!=oq[4]['input_hash'] or qs[4]['seed']!=oq[4]['seed']):errors.append(dict(replan=4,field='first_new_fast_input'))
    return errors

def paired(label,name,base,candidate,keys,scope):
    diffs=[dict(suite=k[0],task=k[1],delta=int(candidate[k]['success'])-int(base[k]['success'])) for k in sorted(keys)]
    ci=paired_ci(diffs)
    balanced=np.mean([np.mean([np.mean([r['delta'] for r in diffs if r['suite']==s and r['task']==t]) for t in sorted({r['task'] for r in diffs if r['suite']==s})]) for s in SUITES])
    return dict(baseline=label,candidate=name,scope=scope,n=len(keys),baseline_success=sum(base[k]['success'] for k in keys),candidate_success=sum(candidate[k]['success'] for k in keys),recovered=sum(not base[k]['success'] and candidate[k]['success'] for k in keys),lost=sum(base[k]['success'] and not candidate[k]['success'] for k in keys),delta_pp=float(balanced*100),episode_delta_pp=float(np.mean([r['delta'] for r in diffs])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100)

def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']};configs={c['id']:c for c in m['conditions']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in m['references'].items())
    data={name:collect(OUT,name) for name in configs};baseline=collect(REFERENCE,'v12_comp_error_extrapolate')
    assert all(set(a)==keys for a in list(data.values())+[baseline])
    fixed={tuple(r[k] for k in ['suite','task','init']):r for r in json.loads((STATIC_REFERENCE/'static_mask_reference.json').read_text())['records'] if r['condition']=='periodic_extrapolate'};assert set(fixed)==keys
    episodes=[];requests=[];block_rows=[];prefix=[];excluded={name:set() for name in configs};base_blocks=[]
    for key,(od,old) in baseline.items():
        oq=[json.loads(s) for s in (od/'infer.jsonl').read_text().splitlines()]
        base_blocks += [float(np.mean([q['timing']['policy_wall_ms'] for q in block])) for block in blocks(oq)]
    for name,aa in data.items():
        stride=configs[name]['refresh_stride']
        for key,(d,r) in aa.items():
            od,old=baseline[key];assert identity(d)==identity(od) and r['manifest_hash']==m['manifest_hash']
            qs=[json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()];oq=[json.loads(s) for s in (od/'infer.jsonl').read_text().splitlines()]
            errors=prefix_check(d,od,qs,oq)
            if min(len(qs),len(oq))<=4 and (len(qs)!=len(oq) or r['success']!=old['success']):errors.append(dict(field='short_prefix_outcome'))
            if errors:excluded[name].add(key)
            prefix.append(dict(condition=name,case=key,passed=not errors,errors=errors))
            full_steps=[];mask=None;mask_step=None;mask_cycle=None;source=[];error_ratio=None;score_mode=None;error_fulls=0;changes=0;selectors=0
            with np.load(d/'arrays.npz') as arrays:
                for i,q in enumerate(qs):
                    z=q['route'];full=i%3==0;cycle=i//3;refresh=full and cycle%stride==0
                    assert z['path']==('full' if full else 'fast') and z['D']==(10 if full else 2)
                    assert len(q['timing']['nodes'])==(11 if full else 3) and legal(z['selected_skip'],6)
                    assert z['captured_layers']==list(range(1,18)) and z['jacobian_model_steps']==10
                    assert z['refresh_stride']==stride and z['full_cycle_index']==cycle
                    assert z['selector_forward_count']==z['steady_vjp_count']==int(refresh)
                    assert z['mask_refresh']==refresh and z['full_refresh']==full
                    if refresh:
                        selectors+=1;assert f'gate_gradient_{i}' in arrays and f'layer_score_{i}' in arrays
                        assert arrays[f'gate_gradient_{i}'].shape==arrays[f'layer_score_{i}'].shape==(17,)
                        if len(full_steps)>=2:
                            source=full_steps[-2:]+[q['step']];error_ratio=(source[2]-source[1])/(source[1]-source[0]);error_fulls+=1;score_mode='comp_error'
                            assert f'prediction_error_rms_{i}' in arrays
                        else:score_mode='warm_original'
                        changes+=int(mask is not None and mask!=z['selected_skip']);mask=z['selected_skip'];mask_step=q['step'];mask_cycle=cycle
                    else:
                        assert q['timing']['selector_total_ms']==q['timing']['error_build_ms']==0
                        assert f'gate_gradient_{i}' not in arrays and f'layer_score_{i}' not in arrays
                    if full:
                        assert z['skip']==[] and z['patch_layers']==[]
                        full_steps.append(q['step'])
                    else:
                        assert z['skip']==mask and z['patch_layers']==mask
                        age=q['step']-full_steps[-1];gap=None if len(full_steps)<2 else full_steps[-1]-full_steps[-2]
                        assert age==i%3*5 and z['full_age_env_steps']==age and z['has_second_history']==(gap is not None)
                        assert np.isclose(z['extrapolation_ratio'],0 if gap is None else age/gap)
                    assert z['selected_skip']==mask and z['mask_cycle_index']==mask_cycle
                    assert z['mask_age_full_cycles']==cycle-mask_cycle and z['mask_age_env_steps']==q['step']-mask_step
                    assert z['history_full_steps']==full_steps[-2:] and z['score_source_full_steps']==source and z['scoring_mode']==score_mode
                    assert z['score_error_ratio']==error_ratio
                    assert q['counts']['selector']==selectors and q['counts']['full']==len(full_steps)
                    requests.append(dict(condition=name,stride=stride,**r['case'],replan=i,path=z['path'],refresh=refresh,mask_age_env_steps=z['mask_age_env_steps'],full_age_env_steps=z['full_age_env_steps'],rpc_ms=q['rpc_ms'],**{k:v for k,v in q['timing'].items() if k.endswith('_ms')}))
            for block in blocks(qs):
                assert sum(q['route']['steady_vjp_count'] for q in block)==6//stride
                block_rows.append(dict(condition=name,**r['case'],start_replan=block[0]['replan'],policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in block])),selector_ms=float(np.mean([q['timing']['selector_total_ms'] for q in block])),vjps=sum(q['route']['steady_vjp_count'] for q in block)))
            episodes.append(dict(condition=name,stride=stride,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],fulls=len(full_steps),vjps=selectors,error_fulls=error_fulls,mask_changes=changes,selector_cold_s=sum(q['timing']['selector_capture_ms'] for q in qs)/1000))
    summary=[];pairs=[];sensitivity=[]
    for name,cfg in configs.items():
        ee=[r for r in episodes if r['condition']==name];qq=[q for q in requests if q['condition']==name and q['replan']>=18];bb=[r for r in block_rows if r['condition']==name];aa={k:r for k,(d,r) in data[name].items()}
        row=dict(condition=name,stride=cfg['refresh_stride'],n=40,success=sum(r['success'] for r in ee),warm_prefix_passed=40-len(excluded[name]),steady_blocks=len(bb),episodes_with_steady_blocks=len({(r['suite'],r['task'],r['init']) for r in bb}),cycle_policy_ms=float(np.mean([r['policy_ms'] for r in bb])),reference_cycle_policy_ms=float(np.mean(base_blocks)),cycle_selector_ms=float(np.mean([r['selector_ms'] for r in bb])),mean_episode_wall_s=float(np.mean([r['wall_s'] for r in ee])),mean_steps=float(np.mean([r['actions'] for r in ee])),mean_selector_cold_s=float(np.mean([r['selector_cold_s'] for r in ee])),episodes_using_error_score=sum(r['error_fulls']>0 for r in ee),total_vjps=sum(r['vjps'] for r in ee),total_fulls=sum(r['fulls'] for r in ee),mean_mask_changes=float(np.mean([r['mask_changes'] for r in ee])))
        for suite in SUITES:row[suite+'_success']=sum(r['success'] for r in ee if r['suite']==suite)
        for label,select in [('full_scored',lambda q:q['path']=='full' and q['refresh']),('full_held',lambda q:q['path']=='full' and not q['refresh']),('fast',lambda q:q['path']=='fast')]:
            vals=[q['policy_wall_ms'] for q in qq if select(q)];row[label+'_policy_ms']=float(np.mean(vals));row[label+'_p95_ms']=float(np.percentile(vals,95))
        summary.append(row)
        for label,base in [('every_cycle',{k:r for k,(d,r) in baseline.items()}),('fixed_v12',fixed)]:
            pairs.append(paired(label,name,base,aa,keys-excluded[name],'strict_prefix'))
            sensitivity.append(paired(label,name,base,aa,keys,'all40'))
    for file,rows in [('episodes.csv',episodes),('requests.csv',requests),('blocks.csv',block_rows),('summary.csv',summary),('paired.csv',pairs),('paired_all40_sensitivity.csv',sensitivity)]:write_csv(OUT/file,rows)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')];assert gates and all(g['passed'] for g in gates)
    costs=[]
    for gate in gates:
        for r in gate['checks']:costs.append(dict(stride=r['stride'],mean_request_ms=r['mean_request_ms'],mean_selector_ms=r['mean_selector_ms']))
    write_csv(OUT/'fixed_input_cost.csv',costs)
    atomic(OUT/'prefix_audit.json',dict(checks=prefix,exact=sum(p['passed'] for p in prefix),total=80,exclusions={name:[list(k) for k in sorted(kk)] for name,kk in excluded.items()}))
    atomic(OUT/'audit.json',dict(passed=True,episodes=80,source_hashes=True,causal_history=True,all_full_histories_refresh=True,hold_mask_between_refresh=True,own_history=True,fast_no_probe=True,prefix_exact=sum(p['passed'] for p in prefix),prefix_total=80,time=time.time()))
    lines=['# V12 Jacobian每2/3周期刷新组合','','一次加权VJP；周期仍为Full-D10→Fast-D2→Fast-D2。Full周期index0起每stride个周期刷新组合。每个Full都更新真实残差，未选层Full不做可微重放/反传/误差方向构造；Fast使用最新两个Full历史。两个条件各40自由闭环。','','|刷新间隔/周期|成功|Spatial|Object|Goal|Long|稳态ms/请求|计时覆盖回合/40|冷选层秒/回合|','|---|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append('|'+ '|'.join([str(r['stride']),f"{r['success']}/40",*[f"{r[s+'_success']}/10" for s in SUITES],f"{r['cycle_policy_ms']:.2f}",str(r['episodes_with_steady_blocks']),f"{r['mean_selector_cold_s']:.2f}"] )+'|')
    lines+=['','同场景旧每周期一次为31/40，固定V12外推为36/40。稳态成本取replan18起完整18请求（6周期）块，避免2/3刷新相位偏置；较短回合不满足完整块，计时覆盖数已列，不能把这一子集均值当所有回合平均。原每周期参考同截取均值为'+f"{np.mean(base_blocks):.2f}"+'ms/请求。','','|固定同输入刷新间隔|三卡平均ms/请求|','|---|---|']
    for stride in (1,2,3):lines.append(f"|{stride}|{np.mean([r['mean_request_ms'] for r in costs if r['stride']==stride]):.2f}|")
    lines+=['','固定输入是一个真实Spatial观测、三卡各18预热+18测量请求；不是8个新输入成本测试。恒定观测后误差为零，用于调度/数值/成本核验；非零误差模型核验复用已冻结8真实输入证明。冷图单列，完整计入真实回合墙钟。','','|配对比较|n|恢复/新增失败|task等权差值pp|探索95%区间pp|','|---|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f},{r['ci_high_pp']:+.1f}]|")
    lines+=['',f"共同前缀核验{sum(p['passed'] for p in prefix)}/80；主配对按冻结规则排除有差异case，全部40敏感性另表。保留所有有效结果，不替换/放宽容差。",'', '探索40场景，suite分层task聚类bootstrap2000次，未校正多重比较，不按结果扩样。自由轨迹成本不是同输入因果差值。组合稳定性/评分过时/扰动交互均未通过本对照单独分离。','','[冻结方案](idea1_v3_4_E2_sparse_refresh_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_sparse_refresh_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))

if __name__=='__main__':main()
