import json,time
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,sha,legal
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def collect(name):
    out={}
    for p in (OUT/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
        r=json.loads(p.read_text());assert r['valid'];out[tuple(r['case'][k] for k in ['suite','task','init'])]=(p.parent,r)
    return out
def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(a),success=sum(r['success'] for d,r in a.values())) for c in m['conditions'] for a in [collect(c['id'])]}
def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']};configs={c['id']:c for c in m['conditions']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    data={name:collect(name) for name in configs};assert all(set(a)==keys for a in data.values())
    episodes=[];requests=[];cycles=[];first_checks=[]
    for case in keys:
        reference=None;first_raw=None;first_actions=None;second_input=None
        for name,a in data.items():
            d,r=a[case];assert r['manifest_hash']==m['manifest_hash'];actual=identity(d)
            if reference is None:reference=actual
            assert actual==reference
            infer=[json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()]
            with np.load(d/'arrays.npz') as z:
                raw=z['raw_0'].copy();actions=z['actions_0'].copy()
            if first_raw is None:first_raw=raw;first_actions=actions
            assert np.allclose(raw,first_raw,atol=1e-5,rtol=1e-5) and np.allclose(actions,first_actions,atol=1e-5,rtol=1e-5)
            if len(infer)>1:
                if second_input is None:second_input=infer[1]['input_hash']
                assert second_input==infer[1]['input_hash']
            cfg=configs[name];full_mask=None;fast_masks=[]
            for index,q in enumerate(infer):
                route=q['route'];full=index%3==0
                assert route['path']==('full' if full else 'fast') and route['D']==(10 if full else 2)
                assert legal(route['selected_skip'],cfg['skip_count']) and len(q['timing']['nodes'])==(11 if full else 3)
                if full:
                    assert route['skip']==[] and route['full_age_rounds']==0 and route['full_age_env_steps']==0
                    full_mask=route['selected_skip']
                else:
                    assert legal(route['skip'],cfg['skip_count']) and route['skip']==route['selected_skip']
                    assert route['full_age_rounds']==index%3 and route['full_age_env_steps']==index%3*5
                    if cfg['method'] in ['cosine','jacobian']:
                        assert q['timing']['selector_total_ms']==0 and route['skip']==full_mask and route['mask_age_env_steps']==index%3*5
                    fast_masks.append(route['skip'])
                requests.append(dict(condition=name,budget=cfg['budget'],method=cfg['method'],**r['case'],replan=index,path=route['path'],skip=str(route['skip']),selected_skip=str(route['selected_skip']),mask_refresh=route['mask_refresh'],mask_age_env_steps=route['mask_age_env_steps'],rpc_ms=q['rpc_ms'],**{k:v for k,v in q['timing'].items() if k.endswith('_ms')}))
            if cfg['method']=='random_fixed' and fast_masks:assert all(x==fast_masks[0] for x in fast_masks)
            assert infer[-1]['counts']['verify']==0 and infer[-1]['counts']['full']==sum(i%3==0 for i in range(len(infer)))
            for start in range(3,len(infer)-2,3):
                block=infer[start:start+3]
                cycles.append(dict(condition=name,budget=cfg['budget'],method=cfg['method'],**r['case'],start_replan=start,policy_ms_per_request=float(np.mean([q['timing']['policy_wall_ms'] for q in block])),rpc_ms_per_request=float(np.mean([q['rpc_ms'] for q in block])),selector_ms_per_request=float(np.mean([q['timing']['selector_total_ms'] for q in block])),full_ms=block[0]['timing']['policy_wall_ms'],fast_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in block[1:]]))))
            episodes.append(dict(condition=name,budget=cfg['budget'],method=cfg['method'],**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']))
        for budget in [14,12]:
            da,db=[data[f'v{budget}_{method}'][case][0] for method in ['random_online','random_fixed']]
            ia,ib=[[json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()] for d in [da,db]]
            if len(ia)>1 and len(ib)>1:
                assert ia[1]['route']['skip']==ib[1]['route']['skip']
                with np.load(da/'arrays.npz') as a,np.load(db/'arrays.npz') as b:
                    assert np.allclose(a['raw_1'],b['raw_1'],atol=1e-5,rtol=1e-5) and np.allclose(a['actions_1'],b['actions_1'],atol=1e-5,rtol=1e-5)
        first_checks.append(dict(case=case,passed=True,conditions=8,first_full_actions=True,first_fast_input=True,random_first_fast=True))
    summary=[]
    for name,cfg in configs.items():
        for suite in ['all',*SUITES]:
            ee=[r for r in episodes if r['condition']==name and (suite=='all' or r['suite']==suite)];qq=[q for q in requests if q['condition']==name and (suite=='all' or q['suite']==suite)];cc=[c for c in cycles if c['condition']==name and (suite=='all' or c['suite']==suite)]
            wall=sum(r['wall_s'] for r in ee);acts=sum(r['actions'] for r in ee);hot=[q for q in qq if q['replan']>0]
            row=dict(condition=name,budget=cfg['budget'],method=cfg['method'],suite=suite,n=len(ee),success=sum(r['success'] for r in ee),mean_steps=acts/len(ee),mean_episode_wall_s=wall/len(ee),episode_per_hour=3600*len(ee)/wall,actions_per_second=acts/wall,requests=len(qq),full_fraction=float(np.mean([q['path']=='full' for q in qq])),all_policy_ms_mean=float(np.mean([q['policy_wall_ms'] for q in qq])),steady_cycles=len(cc))
            for metric in ['policy_ms_per_request','rpc_ms_per_request','selector_ms_per_request','full_ms','fast_ms']:row['cycle_'+metric]=float(np.mean([c[metric] for c in cc])) if cc else None
            for path in ['full','fast']:
                selected=[q for q in hot if q['path']==path]
                for metric in ['policy_wall_ms','rpc_ms','selector_total_ms']:
                    v=[q[metric] for q in selected];row[path+'_'+metric+'_mean']=float(np.mean(v)) if v else None;row[path+'_'+metric+'_p95']=float(np.percentile(v,95)) if v else None
            summary.append(row)
    paired=[]
    specs=[(f'v{budget}_{base}',f'v{budget}_jacobian') for budget in [14,12] for base in ['cosine','random_online','random_fixed']]+[(f'v14_{method}',f'v12_{method}') for method in ['cosine','random_online','jacobian','random_fixed']]
    for an,bn in specs:
        aa,bb=data[an],data[bn];diff=[dict(suite=k[0],task=k[1],delta=int(bb[k][1]['success'])-int(aa[k][1]['success'])) for k in keys];ci=paired_ci(diff)
        paired.append(dict(baseline=an,candidate=bn,recovered=sum(not aa[k][1]['success'] and bb[k][1]['success'] for k in keys),lost=sum(aa[k][1]['success'] and not bb[k][1]['success'] for k in keys),delta_pp=float(np.mean([r['delta'] for r in diff])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    for filename,rows in [('episodes.csv',episodes),('requests.csv',requests),('cycles.csv',cycles),('summary.csv',summary),('paired.csv',paired)]:write_csv(OUT/filename,rows)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/cycle_gate.json')];assert gates and all(g['passed'] for g in gates)
    atomic(OUT/'first_request_audit.json',dict(passed=True,checks=first_checks))
    lines=['# E2 周期3选层：V14/V12结果','','真实执行Full-D10→Fast-D2→Fast-D2，H50/R5，V14跳4层/V12跳6层，max2，8条件×40case=320新闭环。余弦与Jacobian仅Full选层，后两Fast复用；余弦取原生Full已有trace，Jacobian额外D2目标单次VJP；每次随机仅Fast换mask，固定随机整回合不变。无额外验证或补偿。','','|方法|V14成功|V12成功|V14稳态周期ms/请求|V12稳态周期ms/请求|V14选层摊销ms/请求|V12选层摊销ms/请求|','|---|---|---|---|---|---|---|']
    allrows={r['condition']:r for r in summary if r['suite']=='all'}
    for method in ['cosine','random_online','jacobian','random_fixed']:
        a,b=[allrows[f'v{budget}_{method}'] for budget in [14,12]];lines.append(f"|{method}|{a['success']}/40|{b['success']}/40|{a['cycle_policy_ms_per_request']:.2f}|{b['cycle_policy_ms_per_request']:.2f}|{a['cycle_selector_ms_per_request']:.2f}|{b['cycle_selector_ms_per_request']:.2f}|")
    lines+=['','|条件|Spatial|Object|Goal|Long|Full热态ms|Fast热态ms|实际Full占比|平均episode秒|','|---|---|---|---|---|---|---|---|---|']
    for name,cfg in configs.items():
        ss={r['suite']:r for r in summary if r['condition']==name};a=ss['all'];lines.append('|'+ '|'.join([name,*[f"{ss[s]['success']}/10" for s in SUITES],f"{a['full_policy_wall_ms_mean']:.2f}",f"{a['fast_policy_wall_ms_mean']:.2f}",f"{a['full_fraction']:.3f}",f"{a['mean_episode_wall_s']:.2f}"])+'|')
    lines+=['','|配对比较|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f},{r['ci_high_pp']:+.1f}]|")
    lines+=['','周期热态仅统计每回合从第2个Full开始的完整Full/Fast/Fast块，按真实完整3轮均值统计；各条请求和冷捕获仍保留在episodes和requests中。余弦selector列只含统计/约束选层，trace复制已计入Full VLM及完整策略耗时。Jacobian选择目标仍为原D2五步指数权重+七维全正权，执行Full为D10，不把两者混淆。','', '所有case八条件的初态/首输入/首Full动作与首Fast输入一致；两随机策略同预算首Fast的mask和动作一致。所有Full轮实际skip=[]/D10，Fast skip=4或6/D2；余弦/Jacobian Fast探测耗时必须为0，mask年龄为5或10动作。已完成路由与身份审核，不将旧纯Fast结果作为本周期成功率。','', '单一预设随机种子341101，非随机种子总体保证。配对区间为suite分层task聚类探索区间，未按结果增加样本。','', '[冻结方案](idea1_v3_4_E2_cycle3_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle3_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,identity_checks=len(keys)*len(configs),source_hashes=True,schedule=True,fast_no_probe=True,budgets=True,requests=len(requests),steady_cycles=len(cycles),time=time.time()));atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
