import csv,json,time
import numpy as np
from .common import ROOT,OUT,PLAN,SUITES,sha,legal,reference_root
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
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']};configs={c['id']:c for c in m['conditions']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    for path,h in m['references'].items():
        assert sha(ROOT/path/'manifest.json')==h['manifest_sha256'] and sha(ROOT/path/'summary.csv')==h['summary_sha256']
    data={name:collect(OUT,name) for name in configs};baseline={name:collect(reference_root(cfg),cfg['base_id']) for name,cfg in configs.items()}
    assert all(set(a)==keys for a in list(data.values())+list(baseline.values()))
    episodes=[];requests=[];cycles=[];checks=[]
    for name,aa in data.items():
        cfg=configs[name]
        for case,(d,r) in aa.items():
            assert r['manifest_hash']==m['manifest_hash'];old_dir,_=baseline[name][case];assert identity(d)==identity(old_dir)
            qs=[json.loads(x) for x in (d/'infer.jsonl').read_text().splitlines()];oldqs=[json.loads(x) for x in (old_dir/'infer.jsonl').read_text().splitlines()]
            with np.load(d/'arrays.npz') as z,np.load(old_dir/'arrays.npz') as old:
                assert np.allclose(z['raw_0'],old['raw_0'],atol=1e-5,rtol=1e-5) and np.allclose(z['actions_0'],old['actions_0'],atol=1e-5,rtol=1e-5)
                assert all(np.isfinite(z[key]).all() for key in z.files if key.startswith(('gate_gradient_','layer_score_')))
            assert qs[0]['route']['selected_skip']==oldqs[0]['route']['selected_skip']
            if len(qs)>1:assert qs[1]['input_hash']==oldqs[1]['input_hash']
            full_mask=None;history_steps=[];mask_changes=0;previous_mask=None
            for i,q in enumerate(qs):
                route=q['route'];full=i%3==0;assert route['path']==('full' if full else 'fast') and route['D']==(10 if full else 2) and len(q['timing']['nodes'])==(11 if full else 3)
                assert legal(route['selected_skip'],cfg['skip_count']) and route['captured_layers']==list(range(1,18))
                if cfg['method']=='jacobian':assert route['jacobian_model_steps']==10
                if full:
                    assert route['skip']==[] and route['patch_layers']==[]
                    full_mask=route['selected_skip'];history_steps=(history_steps+[q['step']])[-2:]
                    if previous_mask is not None:mask_changes+=int(previous_mask!=full_mask)
                    previous_mask=full_mask
                else:
                    assert route['skip']==full_mask and route['patch_layers']==full_mask and q['timing']['selector_total_ms']==0
                    assert route['mask_age_env_steps']==i%3*5
                    age=q['step']-history_steps[-1];expected=0. if len(history_steps)<2 else age/(history_steps[-1]-history_steps[-2])
                    assert route['has_second_history']==(len(history_steps)==2) and np.isclose(route['extrapolation_ratio'],expected)
                assert route['history_full_steps']==history_steps
                requests.append(dict(condition=name,method=cfg['method'],budget=cfg['budget'],**r['case'],replan=i,path=route['path'],skip=str(route['skip']),history_steps=str(history_steps),ratio=route['extrapolation_ratio'],history_bytes=route['history_bytes'],rpc_ms=q['rpc_ms'],**{k:v for k,v in q['timing'].items() if k.endswith('_ms')}))
            for start in range(3,len(qs)-2,3):
                block=qs[start:start+3];cycles.append(dict(condition=name,budget=cfg['budget'],method=cfg['method'],**r['case'],start_replan=start,policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in block])),selector_ms=float(np.mean([q['timing']['selector_total_ms'] for q in block])),capture_ms=float(np.mean([q['timing']['residual_capture_ms'] for q in block])),rpc_ms=float(np.mean([q['rpc_ms'] for q in block]))))
            episodes.append(dict(condition=name,budget=cfg['budget'],method=cfg['method'],**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],full_mask_changes=mask_changes));checks.append(dict(condition=name,case=case,passed=True,initial_mask=True,first_full_action=True,first_fast_input=True))
    summary=[];paired=[]
    for name,cfg in configs.items():
        ee=[r for r in episodes if r['condition']==name];qq=[q for q in requests if q['condition']==name and q['replan']>0];cc=[c for c in cycles if c['condition']==name]
        root=reference_root(cfg);base=baseline[name];reference=list(csv.DictReader((root/'summary.csv').open()))
        ref=next(r for r in reference if r['condition']==cfg['base_id'] and (r.get('suite','all')=='all'))
        base_cycle=float(ref.get('cycle_policy_ms',ref.get('cycle_policy_ms_per_request')))
        row=dict(condition=name,method=cfg['method'],budget=cfg['budget'],n=len(ee),success=sum(r['success'] for r in ee),baseline_success=sum(r['success'] for d,r in base.values()),baseline_cycle_ms=base_cycle,cycle_policy_ms=float(np.mean([r['policy_ms'] for r in cc])),cycle_selector_ms=float(np.mean([r['selector_ms'] for r in cc])),cycle_capture_ms=float(np.mean([r['capture_ms'] for r in cc])),cycle_rpc_ms=float(np.mean([r['rpc_ms'] for r in cc])),mean_steps=float(np.mean([r['actions'] for r in ee])),mean_episode_wall_s=float(np.mean([r['wall_s'] for r in ee])),max_history_mib=max(q['history_bytes'] for q in qq)/2**20)
        for suite in SUITES:row[suite+'_success']=sum(r['success'] for r in ee if r['suite']==suite)
        for path in ['full','fast']:
            for metric in ['policy_wall_ms','rpc_ms']:
                v=[q[metric] for q in qq if q['path']==path];row[path+'_'+metric+'_mean']=float(np.mean(v));row[path+'_'+metric+'_p95']=float(np.percentile(v,95))
        summary.append(row)
        diff=[dict(suite=k[0],task=k[1],delta=int(data[name][k][1]['success'])-int(base[k][1]['success'])) for k in keys];ci=paired_ci(diff)
        paired.append(dict(baseline=cfg['base_id'],candidate=name,recovered=sum(not base[k][1]['success'] and data[name][k][1]['success'] for k in keys),lost=sum(base[k][1]['success'] and not data[name][k][1]['success'] for k in keys),delta_pp=float(np.mean([r['delta'] for r in diff])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    for budget in [14,12]:
        an,bn=[f'v{budget}_{method}_extrapolate' for method in ['cosine','jacobian']];aa,bb=data[an],data[bn]
        diff=[dict(suite=k[0],task=k[1],delta=int(bb[k][1]['success'])-int(aa[k][1]['success'])) for k in keys];ci=paired_ci(diff)
        paired.append(dict(baseline=an,candidate=bn,recovered=sum(not aa[k][1]['success'] and bb[k][1]['success'] for k in keys),lost=sum(aa[k][1]['success'] and not bb[k][1]['success'] for k in keys),delta_pp=float(np.mean([r['delta'] for r in diff])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    for file,rows in [('episodes.csv',episodes),('requests.csv',requests),('cycles.csv',cycles),('summary.csv',summary),('paired.csv',paired)]:write_csv(OUT/file,rows)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/patch_gate.json')];assert gates and all(g['passed'] for g in gates)
    lines=['# 动态选层＋残差外推：周期3结果','','四条件×40个匹配case，共160条新闭环；无补偿基准经首输入/首Full动作/首mask身份核验复用。Jacobian仅用已修正输入和D10可微重放版本。Full-D10→Fast-D2→Fast-D2，V14/V12 max2，AE全18。Full采集全部17层真实残差，保留本路径两次历史；首周期复用，后续Fast按age/gap=1/3、2/3外推。','','|方法|无补偿成功|加外推成功|无补偿周期ms/请求|加外推周期ms/请求|Full热态ms|Fast热态ms|','|---|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|{r['condition']}|{r['baseline_success']}/40|{r['success']}/40|{r['baseline_cycle_ms']:.2f}|{r['cycle_policy_ms']:.2f}|{r['full_policy_wall_ms_mean']:.2f}|{r['fast_policy_wall_ms_mean']:.2f}|")
    lines+=['','|条件|Spatial|Object|Goal|Long|平均步数|平均episode秒|两份历史峰值MiB|','|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append('|'+ '|'.join([r['condition'],*[f"{r[s+'_success']}/10" for s in SUITES],f"{r['mean_steps']:.1f}",f"{r['mean_episode_wall_s']:.2f}",f"{r['max_history_mib']:.1f}"])+'|')
    lines+=['','|配对比较|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f},{r['ci_high_pp']:+.1f}]|")
    lines+=['','补偿缓存全部17层，因动态mask可能换层；仅缓存当次选中4/6层无法保证下一Full的同层双历史。缓存来自本条件自身真实Full，不读取其他条件轨迹。K/V始终从当前观测生成，不使用旧K/V。余弦与Jacobian选层公式完全沿用无补偿版本；没有针对补偿重新选指标。','', '前置same-observation oracle重建Full VLM+D2动作、FP32参考外推公式、Full-D10/Jacobian输入身份与Graph梯度门通过。首次无第二历史的回退、后续历史年龄与按层取缓存、Full不补偿/Fast零探测均已审核。周期稳态从第二个Full开始取完整三轮，冷捕获和首周期仍包含在完整episode墙钟。','', '另一个实验固定V12的134→169/200仅作已有动机，不能与本轮不同动态mask和case直接合并。这里40条/条件为探索，区间按suite分层task聚类bootstrap，未作显著性驱动扩样。','', '[冻结方案](idea1_v3_4_E2_cycle3_extrapolate_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_cycle3_extrapolate_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'first_request_audit.json',dict(passed=True,checks=checks));atomic(OUT/'audit.json',dict(passed=True,identity_checks=len(checks),source_hashes=True,own_history=True,all17_history=True,history_age=True,first_fallback=True,fast_no_probe=True,time=time.time()));atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
