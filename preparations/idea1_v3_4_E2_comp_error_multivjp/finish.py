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

def cycles(qs):return [qs[i:i+3] for i in range(9,len(qs)-2,3)]

def prefix_check(d,od,qs,oq):
    errors=[]
    with np.load(d/'arrays.npz') as a,np.load(od/'arrays.npz') as b:
        assert all(np.isfinite(a[k]).all() for k in a.files)
        for i in range(min(7,len(qs),len(oq))):
            for field in ('input_hash','seed'):
                if qs[i][field]!=oq[i][field]:errors.append(dict(replan=i,field=field))
            for field in ('raw','actions'):
                x,y=a[f'{field}_{i}'],b[f'{field}_{i}']
                if not np.allclose(x,y,atol=1e-5,rtol=1e-5):errors.append(dict(replan=i,field=field,max_abs=float(abs(x-y).max())))
            if i<6 and qs[i]['route']['selected_skip']!=oq[i]['route']['selected_skip']:
                errors.append(dict(replan=i,field='warm_mask'))
        if len(qs)>7 and (len(oq)<=7 or qs[7]['input_hash']!=oq[7]['input_hash'] or qs[7]['seed']!=oq[7]['seed']):
            errors.append(dict(replan=7,field='first_new_fast_input'))
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
    episodes=[];requests=[];cycle_rows=[];prefix=[];excluded={name:set() for name in configs}
    for name,aa in data.items():
        cfg=configs[name];nproj=cfg['projections']
        for key,(d,r) in aa.items():
            od,old=baseline[key];assert identity(d)==identity(od) and r['manifest_hash']==m['manifest_hash']
            qs=[json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()];oq=[json.loads(s) for s in (od/'infer.jsonl').read_text().splitlines()]
            errors=prefix_check(d,od,qs,oq)
            if min(len(qs),len(oq))<=7 and (len(qs)!=len(oq) or r['success']!=old['success']):errors.append(dict(field='short_prefix_outcome'))
            if errors:excluded[name].add(key)
            prefix.append(dict(condition=name,case=key,passed=not errors,errors=errors))
            full_steps=[];mask=None;source=[];error_ratio=None;error_fulls=0;changes=0
            for i,q in enumerate(qs):
                z=q['route'];full=i%3==0;mode='warm_original' if i<6 else 'comp_error'
                assert z['path']==('full' if full else 'fast') and z['D']==(10 if full else 2)
                assert len(q['timing']['nodes'])==(11 if full else 3) and legal(z['selected_skip'],6)
                assert z['captured_layers']==list(range(1,18)) and z['jacobian_model_steps']==10
                assert z['configured_projections']==nproj and z['selector_forward_count']==int(full)
                assert z['steady_vjp_count']==((1 if i<6 else nproj) if full else 0) and z['scoring_mode']==mode
                if full:
                    assert z['skip']==[] and z['patch_layers']==[] and z['mask_refresh']
                    if len(full_steps)>=2:
                        source=full_steps[-2:]+[q['step']];error_ratio=(source[2]-source[1])/(source[1]-source[0]);error_fulls+=1
                        assert np.isclose(z['score_error_ratio'],error_ratio)
                    else:assert z['score_error_ratio'] is None
                    full_steps.append(q['step']);changes+=int(mask is not None and mask!=z['selected_skip']);mask=z['selected_skip']
                else:
                    assert z['skip']==mask and z['patch_layers']==mask and not z['mask_refresh']
                    assert q['timing']['selector_total_ms']==q['timing']['error_build_ms']==0
                    age=q['step']-full_steps[-1];gap=None if len(full_steps)<2 else full_steps[-1]-full_steps[-2]
                    assert age==i%3*5 and z['mask_age_env_steps']==age and z['has_second_history']==(gap is not None)
                    assert np.isclose(z['extrapolation_ratio'],0 if gap is None else age/gap)
                assert z['history_full_steps']==full_steps[-2:] and z['score_source_full_steps']==source
                if i>=6:assert z['score_error_ratio']==error_ratio and z['error_buffer_bytes']>0
                requests.append(dict(condition=name,projections=nproj,**r['case'],replan=i,path=z['path'],steady_vjp_count=z['steady_vjp_count'],rpc_ms=q['rpc_ms'],**{k:v for k,v in q['timing'].items() if k.endswith('_ms')}))
            for block in cycles(qs):cycle_rows.append(dict(condition=name,**r['case'],start_replan=block[0]['replan'],policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in block])),selector_ms=float(np.mean([q['timing']['selector_total_ms'] for q in block])),rpc_ms=float(np.mean([q['rpc_ms'] for q in block]))))
            episodes.append(dict(condition=name,projections=nproj,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],error_fulls=error_fulls,mask_changes=changes,selector_cold_s=sum(q['timing']['selector_capture_ms'] for q in qs)/1000))
    summary=[];pairs=[];sensitivity=[]
    for name,cfg in configs.items():
        ee=[r for r in episodes if r['condition']==name];qq=[q for q in requests if q['condition']==name and q['replan']>=9];cc=[r for r in cycle_rows if r['condition']==name];aa={k:r for k,(d,r) in data[name].items()}
        row=dict(condition=name,projections=cfg['projections'],n=40,success=sum(r['success'] for r in ee),warm_prefix_passed=40-len(excluded[name]),cycle_policy_ms=float(np.mean([r['policy_ms'] for r in cc])),cycle_selector_ms=float(np.mean([r['selector_ms'] for r in cc])),mean_episode_wall_s=float(np.mean([r['wall_s'] for r in ee])),mean_steps=float(np.mean([r['actions'] for r in ee])),mean_selector_cold_s=float(np.mean([r['selector_cold_s'] for r in ee])),episodes_using_error_score=sum(r['error_fulls']>0 for r in ee))
        for suite in SUITES:row[suite+'_success']=sum(r['success'] for r in ee if r['suite']==suite)
        for path in ('full','fast'):
            vals=[q['policy_wall_ms'] for q in qq if q['path']==path];row[path+'_policy_ms']=float(np.mean(vals));row[path+'_p95_ms']=float(np.percentile(vals,95))
        summary.append(row)
        for label,bb in [('one_vjp',{k:r for k,(d,r) in baseline.items()}),('fixed_v12',fixed)]:
            pairs.append(paired(label,name,bb,aa,keys-excluded[name],'strict_prefix'))
            sensitivity.append(paired(label,name,bb,aa,keys,'all40'))
    for file,rows in [('episodes.csv',episodes),('requests.csv',requests),('cycles.csv',cycle_rows),('summary.csv',summary),('paired.csv',pairs),('paired_all40_sensitivity.csv',sensitivity)]:write_csv(OUT/file,rows)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')];assert gates and all(g['passed'] for g in gates)
    atomic(OUT/'prefix_audit.json',dict(checks=prefix,exact=sum(p['passed'] for p in prefix),total=120,exclusions={name:[list(k) for k in sorted(kk)] for name,kk in excluded.items()}))
    atomic(OUT/'audit.json',dict(passed=True,episodes=120,source_hashes=True,causal_history=True,own_history=True,fast_no_probe=True,shared_forward=True,prefix_exact=sum(p['passed'] for p in prefix),prefix_total=120,predeclared_prefix_exclusions=True,time=time.time()))
    cost=json.loads((OUT/'cost/summary.json').read_text())
    lines=['# 补偿误差选层：V12多次反传闭环','','同40场景、Full-D10→Fast-D2→Fast-D2、外推补偿，前两次Full保持一次原评分；第三Full起共用一次可微D10前向，做q次反传。第一方向沿用正权重，新增固定seed340610随机正负投影，时间权重幅度相同；按投影梯度RMS选合法六层。不是完整Jacobian。','','|反传次数|成功|Spatial|Object|Goal|Long|稳态周期ms/请求|Full ms|Fast ms|冷构建秒/回合|','|---|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append('|'+ '|'.join([str(r['projections']),f"{r['success']}/40",*[f"{r[s+'_success']}/10" for s in SUITES],f"{r['cycle_policy_ms']:.2f}",f"{r['full_policy_ms']:.2f}",f"{r['fast_policy_ms']:.2f}",f"{r['mean_selector_cold_s']:.2f}"] )+'|')
    lines+=['','旧同场景一次误差方向V12为31/40，固定V12外推为36/40。旧有效回合只复用不重跑。','','|同输入反传次数|selector含误差构造 ms|','|---|---|']
    for q in (1,2,3,4):lines.append(f"|{q}|{cost['timing'][str(q)]['mean_ms']:.2f}|")
    lines+=['','GPU成本为8真实输入各20热身10测量、条件分块平衡顺序；中性Full/Graph梯度/首投影均已核验。冷构建不计入热态，完整包含于闭环回合时间。自由轨迹的周期均值不是同输入因果差值。','','|严格配对|n|恢复/新增失败|task等权差值pp|探索95%区间pp|','|---|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f},{r['ci_high_pp']:+.1f}]|")
    lines+=['',f"前缀核验{sum(p['passed'] for p in prefix)}/120；若有差异，按冻结计划保留全部原结果、从对应主配对排除并在prefix_audit.json逐项公开。all40敏感性另表。没有改变容差或替换回合。",'', '每项40回合为探索规模；suite分层task聚类bootstrap2000次，未校正多重比较；不能把最优点估计当独立确认。代理依旧是局部多层近似，已发生误差不等于未来Fast误差。','', '[冻结方案](idea1_v3_4_E2_comp_error_multivjp_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_comp_error_multivjp_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))

if __name__=='__main__':main()
