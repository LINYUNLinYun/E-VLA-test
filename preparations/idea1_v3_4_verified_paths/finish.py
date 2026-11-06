import collections,json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,BASE,PRIOR,PLAN,SETTINGS,sha,ck,SUITES
from .evaluate import header,identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_3.rules import acceptance
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v3_4_E2_selection_200.finish import paired
from preparations.idea1_v3_4_E2_adjacency_200.common import choose,legal

def key(c):return tuple(c[k] for k in ('suite','task','init'))
def rows(d):return [json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()]

def audit(cfg,d,r,reused):
    qs=rows(d);counts=dict(full=0,draft=0,verify=0,selector=0,vjp=0,kalman=0)
    step=0;full_round=full_step=None;history=[];mask=list(cfg['skip']) if cfg['selection']=='fixed' else None;mask_step=0 if mask is not None else None
    draft_rounds=0;pending_gripper=False;last_gripper=None;categories=[];requests=[]
    with np.load(d/'arrays.npz') as z:
        for i,q in enumerate(qs):
            rt=q['route'];t=q['timing'];assert q['step']==step and q['replan']==i and q['round']==i
            assert np.isfinite(z[f'raw_{i}']).all() and np.isfinite(z[f'actions_{i}']).all()
            if reused:
                assert cfg['mode']=='full' and rt['path']=='full' and len(t['nodes'])==11
                counts['full']+=1;category='full_baseline';proposal=5;full=True;attempt=False
            else:
                assert not rt['kalman'] and rt['period']==cfg['period'] and rt['selection']==cfg['selection'] and rt['compensation']==cfg['compensation']
                entry_age=None if full_round is None else dict(rounds=i-full_round,env_steps=step-full_step)
                assert rt['full_age_at_entry']==entry_age and rt['draft_rounds_at_entry']==draft_rounds
                reason='baseline' if cfg['mode']=='full' else 'startup' if full_round is None else 'pending_gripper' if pending_gripper else 'periodic' if draft_rounds>=cfg['period']-1 else None
                attempt=reason is None;assert rt['draft_attempted']==rt['verify_performed']==attempt
                if attempt:
                    assert rt['verify_full_round']==full_round and rt['verify_full_step']==full_step and rt['draft_skip']==mask and len(t['draft_nodes'])==3
                    counts['draft']+=1;counts['verify']+=1;draft_rounds+=1
                    if cfg['compensation']=='extrapolate':
                        gap=None if len(history)<2 else history[-1]-history[-2];age=step-history[-1]
                        assert rt['draft_history_full_steps']==history and rt['draft_residual_age']==age and rt['draft_full_gap']==gap
                        assert np.isclose(rt['draft_extrapolation_ratio'],0. if gap is None else age/gap) and rt['captured_layers']==list(range(1,18))
                    length,detail=acceptance(z[f'draft_future_{i}'],z[f'reconstructed_future_{i}'],cfg['tau'],5,last_gripper=last_gripper,protect=True)
                    assert length==rt['accepted_length'] and all(rt[k]==detail[k] for k in ('spatial_length','verify_gripper','post_verify_gripper'))
                    switch=detail['verify_gripper'] or detail['post_verify_gripper']
                    if length==0:reason='reject_gripper' if switch else 'reject_spatial';proposal=5
                    else:proposal=length;pending_gripper=switch
                else:assert rt['accepted_length'] is None and not rt['fallback'];proposal=5
                full=reason is not None
                assert (rt['path']=='full')==full and rt['fallback']==(attempt and full)
                if full:
                    assert rt['full_reason']==reason and len(t['full_nodes'])==11
                    counts['full']+=1;full_round=i;full_step=step;draft_rounds=0;pending_gripper=False
                    if cfg['compensation']=='extrapolate':history=(history+[step])[-2:]
                    if cfg['selection']=='jacobian':
                        qn=cfg['projections'];g=z[f'gate_gradient_{i}'];score=z[f'layer_score_{i}']
                        assert g.shape==(qn,17) and np.isfinite(g).all()
                        expected=abs(g[0]) if qn==1 else np.sqrt(np.mean(g*g,axis=0));assert np.array_equal(score,expected)
                        mask=choose(score,18-cfg['budget'],'spaced');mask_step=step;counts['selector']+=1;counts['vjp']+=qn
                    category='full_'+reason
                else:category='fast_accept_5' if proposal==5 else 'fast_accept_partial'
                assert rt['request_category']==category and rt['selected_skip']==([] if mask is None else mask)
                if cfg['mode']=='verified':assert legal(mask,18-cfg['budget'],'spaced' if cfg['selection']=='jacobian' else 'max2')
                assert rt['mask_refresh']==(full and cfg['selection']=='jacobian')
                assert rt['steady_vjp_count']==(cfg['projections'] if full and cfg['selection']=='jacobian' else 0)
                assert rt['history_full_steps_after']==history and rt['full_round_after']==full_round and rt['full_step_after']==full_step
                assert rt['draft_rounds_after']==draft_rounds and rt['pending_gripper_after']==pending_gripper and q['counts']==counts
            n=q['executed_length'];assert 0<n<=proposal<=5
            step+=n;last_gripper=float(z[f'raw_{i}'][n-1,6]);categories.append(category)
            requests.append(dict(condition=cfg['id'],**r['case'],replan=i,executed=n,category=category,full=full,draft=attempt,verify=attempt,accepted_length=rt.get('accepted_length'),
                                 policy_ms=t['policy_wall_ms'],full_ms=t.get('full_path_wall_ms',t['policy_wall_ms'] if reused else 0.),draft_ms=t.get('draft_path_wall_ms',0.),verify_ms=t.get('verify_path_wall_ms',0.),
                                 selector_ms=t.get('selector_total_ms',0.),selector_capture_ms=t.get('selector_capture_ms',0.),ratio=rt.get('draft_extrapolation_ratio'),residual_age=rt.get('draft_residual_age'),full_gap=rt.get('draft_full_gap'),reused=reused))
    assert step==r['actions'] and len(qs)==r['predictions'] and r['counts']['full']==counts['full'] and r['counts']['draft']==counts['draft'] and r['counts']['verify']==counts['verify']
    return header(d,r),requests

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and all(sha(ROOT/p)==h for p,h in m['references'].items())
    dep=json.loads((OUT/'dependency_receipt.json').read_text());assert sha(PRIOR/'complete.json')==dep['complete_sha256'] and sha(PRIOR/'audit.json')==dep['audit_sha256']
    data={};quality=[];episodes=[];request_rows=[];reuse_flags={};route_summary=[];call_summary=[];costs=[];latencies=[]
    for cfg in m['conditions']:
        dd={};rq=[]
        for c in m['cases']:
            receipt=json.loads((OUT/'completions'/cfg['id']/f'{ck(c)}.json').read_text());d=Path(receipt['directory']);r=json.loads((d/'result.json').read_text());reused=receipt['reused']
            assert (d/'complete.json').exists() and r['valid'] and r['case']==c
            if reused:
                ref=m['baseline_candidates'][ck(c)];assert cfg['mode']=='full' and str(d)==ref['directory'] and all(sha(d/f)==h for f,h in ref['hashes'].items())
                assert json.loads((OUT/'baseline_reuse'/f'{ck(c)}.json').read_text())['accepted']
            else:assert r['config']==cfg and r['manifest_hash']==m['manifest_hash']
            h,qs=audit(cfg,d,r,reused);dd[key(c)]=h;rq+=qs
            expected=identity(Path(m['controls'][ck(c)]['directory']));assert {k:v for k,v in h['identity'].items() if k!='first_input_hash'}=={k:v for k,v in expected.items() if k!='first_input_hash'}
            reuse_flags[(cfg['id'],key(c))]=reused
            episodes.append(dict(condition=cfg['id'],**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],policy_wall_s=r['policy_wall_s'],wall_s=r['wall_s'],reused=reused,cold_s=sum(q['selector_capture_ms'] for q in qs)/1000))
        data[cfg['id']]=dd;request_rows+=rq
        quality.append(dict(condition=cfg['id'],n=200,success=sum(h['result']['success'] for h in dd.values()),reused=sum(reuse_flags[(cfg['id'],k)] for k in dd),**{s:sum(h['result']['success'] for k,h in dd.items() if k[0]==s) for s in SUITES}))
        counts=collections.Counter(q['category'] for q in rq)
        for category,n in counts.items():route_summary.append(dict(condition=cfg['id'],category=category,n=n,total=len(rq),request_fraction=n/len(rq)))
        verify=[q for q in rq if q['verify']];ratios=[q['ratio'] for q in rq if q['ratio'] is not None];time_total=sum(q['policy_ms'] for q in rq)
        call_summary.append(dict(condition=cfg['id'],requests=len(rq),full_calls=sum(q['full'] for q in rq),draft_calls=len(verify),verify_calls=len(verify),full_per_request=sum(q['full'] for q in rq)/len(rq),
                                 verify_zero_fraction=sum(q['accepted_length']==0 for q in verify)/len(verify) if verify else None,acceptance_histogram=json.dumps(dict(collections.Counter(q['accepted_length'] for q in verify))),
                                 fast_executed_action_fraction=sum(q['executed'] for q in rq if q['category'].startswith('fast_'))/sum(q['executed'] for q in rq),
                                 full_time_fraction=sum(q['full_ms'] for q in rq)/time_total,draft_time_fraction=sum(q['draft_ms'] for q in rq)/time_total,verify_time_fraction=sum(q['verify_ms'] for q in rq)/time_total,selector_time_fraction=sum(q['selector_ms'] for q in rq)/time_total,
                                 ratio_p50=float(np.median(ratios)) if ratios else None,ratio_p95=float(np.percentile(ratios,95)) if ratios else None,ratio_max=max(ratios) if ratios else None,ratio_gt1_fraction=sum(r>1 for r in ratios)/len(ratios) if ratios else None))
        latencies.append(dict(condition=cfg['id'],all_policy_ms_per_request=float(np.mean([q['policy_ms'] for q in rq])),policy_p50=float(np.median([q['policy_ms'] for q in rq])),policy_p95=float(np.percentile([q['policy_ms'] for q in rq],95)),selector_capture_s=sum(q['selector_capture_ms'] for q in rq)/1000))
        atomic(OUT/'condition_audits'/f"{cfg['id']}.json",dict(passed=True,n=200,kalman_calls=0));print(json.dumps(dict(audited=cfg['id'],n=200)),flush=True)
    fresh={key(c) for c in m['cases'] if not reuse_flags[('full_d10',key(c))]};allkeys={key(c) for c in m['cases']}
    base=[e for e in episodes if e['condition']=='full_d10' and (e['suite'],e['task'],e['init']) in fresh]
    base_cost=sum(e['policy_wall_s'] for e in base)/sum(e['actions'] for e in base)*1000
    for cfg in m['conditions']:
        ee=[e for e in episodes if e['condition']==cfg['id'] and (e['suite'],e['task'],e['init']) in fresh];cost=sum(e['policy_wall_s'] for e in ee)/sum(e['actions'] for e in ee)*1000
        costs.append(dict(condition=cfg['id'],scope='matched_new_baseline_cases',n=len(ee),policy_ms_per_executed_action=cost,baseline_policy_ms_per_executed_action=base_cost,speedup=base_cost/cost,policy_ms_per_request=sum(e['policy_wall_s'] for e in ee)*1000/sum(e['predictions'] for e in ee),cold_capture_s=sum(e['cold_s'] for e in ee)))
    pairs=[];sensitivity=[];exceptions=[]
    comparisons=[('full_d10',c['id'],False) for c in m['conditions'] if c['mode']=='verified']+[(f'v{b}_extrapolate_d2_verify_p3',f'v{b}_extrapolate_d2_verify_p5',True) for b in (12,14)]
    for a,b,same_mask in comparisons:
        for scope,target in (('strict',pairs),('all200',sensitivity)):
            row,ex=paired(a,b,data[a],data[b],allkeys,scope,same_mask);target.append(row)
            if scope=='strict':exceptions+=ex
    profile_rows=[];profile_files=[]
    for s in SUITES:
        for task in (0,2):
            f=OUT/'profiles'/s/f'task{task:02d}'/'complete.json';r=json.loads(f.read_text());assert r['passed'];profile_rows+=r['rows'];profile_files.append(dict(path=str(f),wall_s=r['wall_s']))
    profile_summary=[]
    for cfg in [c for c in m['conditions'] if c['mode']=='verified']:
        rr=[r for r in profile_rows if r['condition']==cfg['id']];ref=sum(r['full_reference_ms'] for r in rr);candidate=sum(r['candidate_policy_ms'] for r in rr);executed=sum(r['executed_length'] for r in rr)
        profile_summary.append(dict(condition=cfg['id'],requests=len(rr),unique_input_hashes=len({r['input_hash'] for r in rr}),same_gpu_and_input=True,full_reference_ms_per_request=ref/len(rr),candidate_ms_per_request=candidate/len(rr),request_speedup=ref/candidate,
                                    execution_normalized_speedup=(ref/(5*len(rr)))/(candidate/executed),candidate_ms_excluding_explicit_selector_capture=(candidate-sum(r['candidate_selector_capture_ms'] for r in rr))/len(rr)))
    for name,rr in (('summary.csv',quality),('episodes.csv',episodes),('requests.csv',request_rows),('request_mix.csv',route_summary),('calls_and_ratios.csv',call_summary),('costs.csv',costs),('latency.csv',latencies),('paired.csv',pairs),('paired_sensitivity.csv',sensitivity),('profile_requests.csv',profile_rows),('profile_summary.csv',profile_summary)):write_csv(OUT/name,rr)
    atomic(OUT/'prefix_audit.json',dict(pair_exceptions=exceptions,all_valid_outcomes_retained=True))
    workers=[]
    for wd in sorted((OUT/'workers').glob('gpu*')):
        if not (wd/'startup.json').exists():continue
        start=json.loads((wd/'startup.json').read_text());state=json.loads((wd/'state.json').read_text());exits=json.loads((wd/'exits.json').read_text())
        assert state['status']=='complete' and exits['service'] in (-15,0) and all(code==0 for code in exits['evaluations']) and json.loads((wd/'compatibility_gate.json').read_text())['passed']
        iso=json.loads((wd/'isolation_gate.json').read_text());assert start['actors']==1 or iso['passed']
        attempts=[json.loads(f.read_text()) for f in wd.glob('startup_attempt*.json')];workers.append(dict(startup=start,startup_attempts=attempts,state=state,exits=exits,identity=json.loads((wd/'identity.json').read_text())))
    start=min(s['time'] for w in workers for s in (w['startup_attempts'] or [w['startup']]));end=max(w['state']['time'] for w in workers);wall=end-start
    new=[e for e in episodes if not e['reused']];retries=[str(f) for f in (OUT/'technical_attempts').rglob('*.json') if json.loads(f.read_text())['attempts']>1]
    throughput=dict(new_episodes=len(new),reused=1000-len(new),evaluation_wall_s=wall,episodes_per_hour=len(new)*3600/wall,actions_per_s=sum(e['actions'] for e in new)/wall,requests_per_s=sum(e['predictions'] for e in new)/wall,
                    profiling_included_in_wall=True,profiling=profile_files,service_startup_s=[w['startup']['startup_s'] for w in workers],technical_retries=retries,workers=workers)
    assert len(episodes)==1000;atomic(OUT/'throughput.json',throughput)
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_verified_paths_report.md'
    lines=['# V12/V14外推D2＋验证：周期3/5对Full-D10','',f"选层设置：{m['selection']}。验证tau=.10、t=.10/.05、原夹爪保护，无Kalman draft。每项同200例；加速比为同轮新baseline共同{len(fresh)}例的完整策略ms/实际执行动作成本比，包含被拒Fast、verify、Full及反推/冷捕获。",'', '|条件|成功率|策略ms/实际动作|相对Full-D10成本加速比|','|---|---|---|---|']
    for q in quality:
        c=next(c for c in costs if c['condition']==q['condition']);lines.append(f"|{q['condition']}|{q['success']}/200（{q['success']/2:.1f}%）|{c['policy_ms_per_executed_action']:.3f}|{c['speedup']:.3f}×|")
    lines+=['','|条件|请求数|Full调用/请求|零接受/verify|Fast执行动作占比|外推ratio>1占比|','|---|---|---|---|---|']
    def percent(v):return '—' if v is None else f'{100*v:.1f}%'
    for r in call_summary:lines.append(f"|{r['condition']}|{r['requests']}|{percent(r['full_per_request'])}|{percent(r['verify_zero_fraction'])}|{percent(r['fast_executed_action_fraction'])}|{percent(r['ratio_gt1_fraction'])}|")
    lines+=['','互斥请求类型占比（每条件合计100%）：','','|条件|类别|次数|比例|','|---|---|---|---|']
    for r in route_summary:lines.append(f"|{r['condition']}|{r['category']}|{r['n']}|{percent(r['request_fraction'])}|")
    lines+=['','同卡同输入Full参考；候选耗时取原自由闭环请求，参考在task组结束后顺/逆两次测量，时间顺序未完全反平衡，作为参考而非严格因果速度证明：','','|条件|输入请求数|Full参考ms/请求|候选ms/请求|请求成本比|执行长度归一化成本比|','|---|---|---|---|---|---|']
    for r in profile_summary:lines.append(f"|{r['condition']}|{r['requests']}|{r['full_reference_ms_per_request']:.2f}|{r['candidate_ms_per_request']:.2f}|{r['request_speedup']:.3f}×|{r['execution_normalized_speedup']:.3f}×|")
    lines+=['','速度比>1表示候选更快。真实闭环轨迹及请求组成不同；冷捕获、排队、模型准备、同输入参考和实际评测吞吐分别保留。Full/Draft/Verify可在同一拒绝请求并存，调用比例不要求合计100%；互斥请求类别才合计100%。验证零接受与夹爪接管是规则触发因素，不直接等同于任务失败原因。', '',f"本轮新测{len(new)}条、兼容复用{1000-len(new)}条Full参考；评测窗口{wall/60:.2f}分钟，{throughput['episodes_per_hour']:.1f}新回合/小时，含独立profiling。技术重试{len(retries)}回合。模型准备单列。",'',f'共同前缀配对异常记录{len(exceptions)}项；全部有效结果保留，严格配对与完整200敏感性分别输出。suite分层task聚类bootstrap2000次、seed281034，探索区间未校正多重比较。','', '[冻结方案](idea1_v3_4_verified_paths_plan.md)']
    report.write_text('\n'.join(lines)+'\n');atomic(OUT/'audit.json',dict(passed=True,target_records=1000,new_episodes=len(new),kalman_calls=0,source_hashes=True,verified_actual_prefix=True,request_categories_complete=True,profile_passed=True,time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',completed=1000,time=time.time()))

if __name__=='__main__':main()
