import csv,json,time
import numpy as np
from .common import ROOT,OUT,REFERENCE,PLAN,sha,legal
from .evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv

def collect(root,name):
    result={}
    for p in (root/name/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists():continue
        r=json.loads(p.read_text());assert r['valid'];result[tuple(r['case'][k] for k in ['suite','task','init'])]=(p.parent,r)
    return result
def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(a),success=sum(r['success'] for d,r in a.values())) for c in m['conditions'] for a in [collect(OUT,c['id'])]}
def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    assert sha(REFERENCE/'manifest.json')==m['reference_manifest_sha256'] and sha(REFERENCE/'summary.csv')==m['reference_summary_sha256']
    data={c['id']:collect(OUT,c['id']) for c in m['conditions']};assert all(set(a)==keys for a in data.values())
    baseline={name:collect(REFERENCE,name) for name in [f'v{budget}_{method}' for budget in [14,12] for method in ['cosine','random_online','random_fixed']]}
    assert all(set(a)==keys for a in baseline.values())
    episodes=[];requests=[];cycles=[];checks=[]
    for name,a in data.items():
        cfg=next(c for c in m['conditions'] if c['id']==name)
        for case,(d,r) in a.items():
            assert r['manifest_hash']==m['manifest_hash'];refdir,ref=baseline[f"v{cfg['budget']}_cosine"][case]
            assert identity(d)==identity(refdir)
            qs=[json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()];oldqs=[json.loads(x) for x in (refdir/'infer.jsonl').read_text().splitlines()]
            with np.load(d/'arrays.npz') as z,np.load(refdir/'arrays.npz') as old:
                assert np.allclose(z['raw_0'],old['raw_0'],atol=1e-5,rtol=1e-5) and np.allclose(z['actions_0'],old['actions_0'],atol=1e-5,rtol=1e-5)
                assert all(np.isfinite(z[key]).all() for key in z.files if key.startswith(('gate_gradient_','layer_score_')))
            if len(qs)>1:assert qs[1]['input_hash']==oldqs[1]['input_hash']
            selected=None
            for i,q in enumerate(qs):
                route=q['route'];full=i%3==0;assert route['jacobian_model_steps']==10 and route['jacobian_full_layers']==18 and route['jacobian_output_steps']==[1,2,3,4,5]
                assert route['path']==('full' if full else 'fast') and route['D']==(10 if full else 2) and len(q['timing']['nodes'])==(11 if full else 3)
                assert legal(route['selected_skip'],cfg['skip_count'])
                if full:assert route['skip']==[];selected=route['selected_skip']
                else:assert route['skip']==selected and q['timing']['selector_total_ms']==0 and route['mask_age_env_steps']==i%3*5
                requests.append(dict(condition=name,budget=cfg['budget'],**r['case'],replan=i,path=route['path'],skip=str(route['skip']),jacobian_model_steps=10,rpc_ms=q['rpc_ms'],**{k:v for k,v in q['timing'].items() if k.endswith('_ms')}))
            for start in range(3,len(qs)-2,3):
                block=qs[start:start+3];cycles.append(dict(condition=name,budget=cfg['budget'],**r['case'],start_replan=start,policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in block])),selector_ms=float(np.mean([q['timing']['selector_total_ms'] for q in block])),rpc_ms=float(np.mean([q['rpc_ms'] for q in block]))))
            episodes.append(dict(condition=name,budget=cfg['budget'],**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']));checks.append(dict(condition=name,case=case,passed=True))
    summary=[]
    for name in data:
        a=[r for r in episodes if r['condition']==name];q=[x for x in requests if x['condition']==name and x['replan']>0];cc=[x for x in cycles if x['condition']==name];budget=a[0]['budget']
        row=dict(condition=name,budget=budget,n=len(a),success=sum(r['success'] for r in a),mean_steps=float(np.mean([r['actions'] for r in a])),mean_episode_wall_s=float(np.mean([r['wall_s'] for r in a])),cycle_policy_ms=float(np.mean([r['policy_ms'] for r in cc])),cycle_selector_ms=float(np.mean([r['selector_ms'] for r in cc])),cycle_rpc_ms=float(np.mean([r['rpc_ms'] for r in cc])))
        for path in ['full','fast']:
            for metric in ['policy_wall_ms','selector_total_ms','rpc_ms']:
                v=[x[metric] for x in q if x['path']==path];row[path+'_'+metric+'_mean']=float(np.mean(v));row[path+'_'+metric+'_p95']=float(np.percentile(v,95))
        for suite in sorted({r['suite'] for r in a}):row[suite+'_success']=sum(r['success'] for r in a if r['suite']==suite)
        summary.append(row)
    paired=[]
    for name,aa in data.items():
        budget=next(c['budget'] for c in m['conditions'] if c['id']==name)
        for method in ['cosine','random_online','random_fixed']:
            refname=f'v{budget}_{method}';bb=baseline[refname];delta=[dict(suite=k[0],task=k[1],delta=int(aa[k][1]['success'])-int(bb[k][1]['success'])) for k in keys];ci=paired_ci(delta)
            paired.append(dict(baseline=refname,candidate=name+'_full10',recovered=sum(not bb[k][1]['success'] and aa[k][1]['success'] for k in keys),lost=sum(bb[k][1]['success'] and not aa[k][1]['success'] for k in keys),delta_pp=float(np.mean([r['delta'] for r in delta])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    for file,rows in [('episodes.csv',episodes),('requests.csv',requests),('cycles.csv',cycles),('summary.csv',summary),('paired.csv',paired)]:write_csv(OUT/file,rows)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')];assert gates and all(g['passed'] and g['NFE']==10 for g in gates)
    reference=list(csv.DictReader((REFERENCE/'summary.csv').open()));refall={r['condition']:r for r in reference if r['suite']=='all'}
    lines=['# 周期3 Jacobian修正：Full-D10与正确初始输入','','本轮修正两处实现问题：重放日程D2→D10；Full结束后恢复原始语言嵌入，不再使用已更新的语言隐藏状态。旧Jacobian周期3比较结论撤回。余弦/随机执行路径不受该重放错误影响，在初态、首Full动作和首Fast输入核验后复用。','', '仍为18层Full路径输出前5步动作的指数时间加权、七维全+1、一次VJP；高影响层保留，低影响4/6层跳过，max2；后两次Fast复用。未更改为未来第6–15步。执行Full-D10→Fast-D2→Fast-D2、H50/R5。新补80条闭环，原六组余弦/随机240条不重复。','','|策略|V14成功|V12成功|V14稳态周期ms/请求|V12稳态周期ms/请求|','|---|---|---|---|---|']
    for method in ['cosine','random_online','jacobian_full10','random_fixed']:
        if method=='jacobian_full10':
            a,b=[next(r for r in summary if r['budget']==budget) for budget in [14,12]];values=[a['success'],b['success'],a['cycle_policy_ms'],b['cycle_policy_ms']]
        else:
            a,b=[refall[f'v{budget}_{method}'] for budget in [14,12]];values=[int(a['success']),int(b['success']),float(a['cycle_policy_ms_per_request']),float(b['cycle_policy_ms_per_request'])]
        lines.append(f"|{method}|{values[0]}/40|{values[1]}/40|{values[2]:.2f}|{values[3]:.2f}|")
    lines+=['','|修正Jacobian|Full热态ms|Fast热态ms|选层摊销ms/请求|平均episode秒|','|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['condition']}|{r['full_policy_wall_ms_mean']:.2f}|{r['fast_policy_wall_ms_mean']:.2f}|{r['cycle_selector_ms']:.2f}|{r['mean_episode_wall_s']:.2f}|")
    lines+=['','|基准→修正Jacobian|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f},{r['ci_high_pp']:+.1f}]|")
    diagnostics=[c for g in gates for c in g['checks']]
    lines+=['','## 重放数值与限制','',f"Graph/eager输出、梯度和mask核验通过，Full初始输入与导出输入逐位一致。原生执行输出不变。PyTorch重放与原生Triton并非逐位相同：四suite跨服务R5 raw RMS最大{max(r['native_replay_prefix7_rms'] for r in diagnostics):.6f}，前5步平移最大差{max(r['native_replay_translation_max_mm'] for r in diagnostics):.3f}mm，夹爪符号翻转检查总数{sum(r['native_replay_gripper_sign_flips'] for r in diagnostics)}。这些差异为诊断，不隐称原生Triton有自动求导。",'', '一次标量VJP是动作Jacobian的投影近似，非完整35行矩阵范数。两处修正同时发生，不能把与旧结果的变化单独归因于D10。周期稳态排除首周期，冷捕获仍计入全部episode墙钟。小批配对区间为探索证据。','', '[修正方案](idea1_v3_4_E2_cycle3_full10_jacobian_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle3_full10_jacobian_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'audit.json',dict(passed=True,identity_checks=len(checks),initial_full_and_fast=True,full_initial_input=True,jacobian_NFE=10,schedule=True,fast_no_probe=True,time=time.time()));atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
