import json,time
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,BASE,PLAN,sha,ck
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v3_4_verified_paths.finish import audit as parent_audit
from preparations.idea1_v3_4_verified_paths.evaluate import header
from preparations.idea1_v3_4_E2_selection_200.finish import paired

def key(c):return tuple(c[k] for k in ('suite','task','init'))
def audit(cfg,d,r):
    h,qs=parent_audit(cfg,d,r,False);history=[]
    raw=[json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
    with np.load(d/'arrays.npz') as z:
        last=None
        for i,q in enumerate(raw):
            rt=q['route']
            if rt['path']=='full':
                step=q['step'];mode='warm_original' if not history else 'comp_error_hold' if len(history)==1 else 'comp_error_extrapolate'
                source=[] if not history else history+[step]
                ratio=None if not history else 0. if len(history)==1 else (step-history[-1])/(history[-1]-history[-2])
                assert rt['scoring_mode']==mode and rt['score_source_full_steps']==source and rt['score_error_ratio']==ratio
                if history:assert z[f'compensation_error_rms_{i}'].shape==(17,) and np.isfinite(z[f'compensation_error_rms_{i}']).all() and (z[f'compensation_error_rms_{i}']>=0).all()
                history=(history+[step])[-2:];last=(mode,source,ratio)
            else:assert (rt['scoring_mode'],rt['score_source_full_steps'],rt['score_error_ratio'])==last
            assert rt['configured_projections']==4 and rt['projection_seed']==340610 and rt['score_aggregation']=='projection_rms'
    return h,qs,raw

def main():
    m=json.loads((OUT/'manifest.json').read_text());assert sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and all(sha(ROOT/p)==h for p,h in m['references'].items())
    assert json.loads((OUT/'cpu_gate.json').read_text())['passed']
    summary=[];pairs=[];sensitivity=[];exceptions=[];episodes=[];requests=[];checks=[];keys={key(c) for c in m['cases']}
    for cfg in m['conditions']:
        old={};new={};cost=[];original_cost=[]
        for c in m['cases']:
            rec=json.loads((OUT/'completions'/cfg['id']/f'{ck(c)}.json').read_text());d=Path(rec['directory']);r=json.loads((d/'result.json').read_text())
            assert r['valid'] and r['case']==c and r['config']==cfg and r['manifest_hash']==m['manifest_hash'] and (d/'complete.json').exists()
            h,qs,raw=audit(cfg,d,r);new[key(c)]=h
            ref=m['baseline'][cfg['base_id']][ck(c)];od=Path(ref['directory']);assert all(sha(od/f)==v for f,v in ref['hashes'].items());oh=header(od);old[key(c)]=oh
            assert {k:v for k,v in h['identity'].items() if k!='first_input_hash'}=={k:v for k,v in oh['identity'].items() if k!='first_input_hash'}
            with np.load(d/'arrays.npz') as a,np.load(od/'arrays.npz') as b:
                eligible=h['identity']['first_input_hash']==oh['identity']['first_input_hash']
                if eligible:assert np.allclose(a['gate_gradient_0'],b['gate_gradient_0'],atol=1e-4,rtol=1e-3)
                checks.append(dict(condition=cfg['id'],case=c,identical_input=eligible,compared=eligible))
            old_q=[json.loads(s) for s in (od/'infer.jsonl').read_text().splitlines()]
            for label,rows,out in [('error_score',raw,cost),('original_score',old_q,original_cost)]:
                for q in rows:
                    t=q['timing'];out.append(dict(policy_ms=t['policy_wall_ms'],cold_ms=t['selector_capture_ms'],full=q['route']['path']=='full',selector_ms=t['selector_total_ms'],error_build_ms=t.get('error_build_ms',0.),mode=q['route'].get('scoring_mode','original')))
            episodes.append(dict(condition=cfg['id'],**c,success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']))
            requests.extend(dict(condition=cfg['id'],**c,**q) for q in qs)
        for scope,target in [('strict',pairs),('all40',sensitivity)]:
            row,ex=paired(cfg['base_id'],cfg['id'],old,new,keys,scope,True);target.append(row)
            if scope=='strict':exceptions+=ex
        summary.append(dict(condition=cfg['id'],budget=cfg['budget'],period=cfg['period'],n=40,original_success=sum(h['result']['success'] for h in old.values()),error_score_success=sum(h['result']['success'] for h in new.values()),
                            original_policy_ms=np.mean([q['policy_ms'] for q in original_cost]),error_policy_ms=np.mean([q['policy_ms'] for q in cost]),original_minus_capture_ms=np.mean([q['policy_ms']-q['cold_ms'] for q in original_cost]),error_minus_capture_ms=np.mean([q['policy_ms']-q['cold_ms'] for q in cost]),
                            error_selection_events=sum(q['full'] and q['mode']!='warm_original' for q in cost),error_build_ms=np.mean([q['error_build_ms'] for q in cost if q['full'] and q['mode']!='warm_original']),full_fraction=np.mean([q['full'] for q in cost])))
        print(json.dumps(dict(audited=cfg['id'],n=40)),flush=True)
    for name,rows in [('summary.csv',summary),('paired.csv',pairs),('paired_sensitivity.csv',sensitivity),('episodes.csv',episodes),('requests.csv',requests)]:write_csv(OUT/name,rows)
    atomic(OUT/'prefix_audit.json',dict(exceptions=exceptions,all_valid_outcomes_retained=True,first_gradient_checks=checks))
    workers=[]
    for wd in (OUT/'workers').glob('gpu*'):
        if not (wd/'startup.json').exists():continue
        state=json.loads((wd/'state.json').read_text());startup=json.loads((wd/'startup.json').read_text());exits=json.loads((wd/'exits.json').read_text())
        assert state['status']=='complete' and exits['service'] in (-15,0) and all(x==0 for x in exits['evaluations']) and json.loads((wd/'compatibility_gate.json').read_text())['passed']
        workers.append(dict(state=state,startup=startup,identity=json.loads((wd/'identity.json').read_text()),exits=exits))
    wall=max(w['state']['time'] for w in workers)-min(w['startup']['time'] for w in workers)
    retries=[str(f) for f in (OUT/'technical_attempts').rglob('*.json') if json.loads(f.read_text())['attempts']>1]
    throughput=dict(new_episodes=160,evaluation_wall_s=wall,episodes_per_hour=160*3600/wall,technical_retries=retries,service_startup_s=[w['startup']['startup_s'] for w in workers],workers=workers)
    atomic(OUT/'throughput.json',throughput)
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_verified_error40_report.md'
    lines=['# 补偿误差评分：四次反推、间隔跳层、外推D2及验证，40例快速对照','','各条件4 suites×10 tasks×init19，共160条新回合；原评分同案例160条复用。只改变选层方向：首Full原评分，第二Full用hold预测误差，之后用实际Full间距外推误差。每Full均四次反推；周期、D、验证阈值、夹爪规则及外推比率未改。','','|变体|原评分|补偿误差评分|变化pp|','|---|---|---|---|']
    for r in summary:lines.append(f"|V{r['budget']}，周期{r['period']}|{r['original_success']}/40|{r['error_score_success']}/40|{(r['error_score_success']-r['original_success'])*2.5:+.1f}|")
    lines+=['','|配对|严格n|恢复/新增失败|变化pp|探索95%区间pp|','|---|---|---|---|---|']
    for r in pairs:lines.append(f"|{r['candidate']}|{r['n']}|{r['recovered']}/{r['lost']}|{r['task_balanced_delta_pp']:+.2f}|[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]|")
    lines+=['','|条件|原/新ms每请求|原/新扣显式图捕获ms每请求|实际误差评分Full次数|误差构造ms|','|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['condition']}|{r['original_policy_ms']:.2f}/{r['error_policy_ms']:.2f}|{r['original_minus_capture_ms']:.2f}/{r['error_minus_capture_ms']:.2f}|{r['error_selection_events']}|{r['error_build_ms']:.3f}|")
    full_success=0
    for ref in m['full_reference'].values():
        d=Path(ref['directory']);assert sha(d/'result.json')==ref['result_sha256'];full_success+=json.loads((d/'result.json').read_text())['success']
    lines+=['',f'同40例Full-D10参考为{full_success}/40。新160回合实际评测窗口{wall/60:.2f}分钟，吞吐{throughput["episodes_per_hour"]:.1f}回合/小时；准备成本单列。技术重试{len(retries)}回合。', '',f'共同前缀配对异常累计{len(exceptions)}项，全部有效结果保留、严格配对剔除，另报all40敏感性。suite分层task聚类bootstrap2000次、seed281034，未校正多重比较；40例是探索小样本。', '', '新结果单客户端，历史参考含双/单客户端；策略内计时不含RPC排队，跨轮自由轨迹均值不是严格同输入因果测速。误差评分仍为Full-D10局部近似，不是未来Fast误差oracle。只从本策略已观测Full历史构造方向。', '', '[冻结方案](idea1_v3_4_verified_error40_plan.md)']
    report.write_text('\n'.join(lines)+'\n')
    assert len(episodes)==160
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=160,source_hashes=True,own_history_error_directions=True,projections=4,max_run=1,kalman_calls=0,prefix_exceptions=len(exceptions),time=time.time()))
    atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',completed=160,target=160,time=time.time()))

if __name__=='__main__':main()
