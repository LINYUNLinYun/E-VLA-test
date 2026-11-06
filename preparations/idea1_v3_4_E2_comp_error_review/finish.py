from preparations.idea1_v3_4_E2_comp_error_review.audit import PREFIX_EXCEPTION,validate_prefix_exception
import csv,json,time
import numpy as np
from preparations.idea1_v3_4_E2_comp_error_closedloop.common import ROOT,OUT,REFERENCE,PLAN,SUITES,sha,legal
from preparations.idea1_v3_4_E2_comp_error_closedloop.evaluate import identity
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv
from preparations.idea1_v3_4_E2_cycle3_extrapolate.finish import collect

def progress():
    m=json.loads((OUT/'manifest.json').read_text())
    return {c['id']:dict(completed=len(a),success=sum(r['success'] for d,r in a.values())) for c in m['conditions'] for a in [collect(OUT,c['id'])]}

def cycles(qs):
    return [qs[start:start+3] for start in range(9,len(qs)-2,3)]

def main():
    m=json.loads((OUT/'manifest.json').read_text());keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']};configs={c['id']:c for c in m['conditions']}
    assert all(sha(ROOT/p)==h for p,h in m['source_files'].items()) and sha(PLAN)==m['plan_sha256']
    assert all(sha(ROOT/p)==h for p,h in m['references'].items())
    data={name:collect(OUT,name) for name in configs};baseline={name:collect(REFERENCE,cfg['base_id']) for name,cfg in configs.items()}
    assert all(set(a)==keys for a in list(data.values())+list(baseline.values()))
    fixed_rows=json.loads((REFERENCE/'static_mask_reference.json').read_text())['records']
    fixed={tuple(r[k] for k in ['suite','task','init']):r for r in fixed_rows if r['condition']=='periodic_extrapolate'};assert set(fixed)==keys
    episodes=[];requests=[];cycle_rows=[];audits=[];base_costs={};prefix_requests=0
    for name,aa in data.items():
        cfg=configs[name];base_costs[name]=[]
        for key,(d,r) in aa.items():
            assert r['manifest_hash']==m['manifest_hash'];od,old_result=baseline[name][key];assert identity(d)==identity(od)
            qs=[json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()];oq=[json.loads(s) for s in (od/'infer.jsonl').read_text().splitlines()]
            excluded=(name,key)==PREFIX_EXCEPTION
            if excluded:
                validate_prefix_exception(name,key,d,od,qs,oq,r,old_result);prefix_requests+=4
            else:
                with np.load(d/'arrays.npz') as a,np.load(od/'arrays.npz') as b:
                    for i in range(min(7,len(qs),len(oq))):
                        assert qs[i]['input_hash']==oq[i]['input_hash'] and qs[i]['seed']==oq[i]['seed'],(name,key,i,'prefix input')
                        assert np.allclose(a[f'raw_{i}'],b[f'raw_{i}'],atol=1e-5,rtol=1e-5),(name,key,i,'prefix raw')
                        assert np.allclose(a[f'actions_{i}'],b[f'actions_{i}'],atol=1e-5,rtol=1e-5)
                        if i<6:assert qs[i]['route']['selected_skip']==oq[i]['route']['selected_skip']
                        prefix_requests+=1
                    assert all(np.isfinite(a[k]).all() for k in a.files)
                if min(len(qs),len(oq))<=7:assert len(qs)==len(oq) and r['success']==old_result['success']
                if len(qs)>7:
                    assert len(oq)>7 and qs[7]['input_hash']==oq[7]['input_hash'] and qs[7]['seed']==oq[7]['seed']
            full_steps=[];mask=None;last_source=[];error_ratio=None;mask_changes=0;previous_mask=None;error_fulls=0
            for i,q in enumerate(qs):
                z=q['route'];full=i%3==0;assert z['path']==('full' if full else 'fast') and z['D']==(10 if full else 2)
                assert len(q['timing']['nodes'])==(11 if full else 3) and legal(z['selected_skip'],cfg['skip_count'])
                assert z['captured_layers']==list(range(1,18)) and z['jacobian_model_steps']==10 and z['steady_vjp_count']==int(full)
                mode='warm_original' if i<6 else 'comp_error';assert z['scoring_mode']==mode
                if full:
                    assert z['skip']==[] and z['patch_layers']==[] and z['mask_refresh']
                    if len(full_steps)>=2:
                        last_source=full_steps[-2:]+[q['step']];error_ratio=(last_source[2]-last_source[1])/(last_source[1]-last_source[0]);error_fulls+=1
                        assert np.isclose(z['score_error_ratio'],error_ratio)
                    else:assert z['score_error_ratio'] is None
                    full_steps.append(q['step']);mask=z['selected_skip']
                    if previous_mask is not None:mask_changes+=int(mask!=previous_mask)
                    previous_mask=mask
                else:
                    assert z['skip']==mask and z['patch_layers']==mask and not z['mask_refresh']
                    assert q['timing']['selector_total_ms']==q['timing']['error_build_ms']==0
                    age=q['step']-full_steps[-1];gap=None if len(full_steps)<2 else full_steps[-1]-full_steps[-2]
                    assert age==i%3*5 and z['mask_age_env_steps']==age
                    assert z['has_second_history']==(gap is not None)
                    assert np.isclose(z['extrapolation_ratio'],0 if gap is None else age/gap)
                assert z['history_full_steps']==full_steps[-2:] and z['score_source_full_steps']==last_source
                if i>=6:assert z['score_error_ratio']==error_ratio and z['error_buffer_bytes']>0
                requests.append(dict(condition=name,**r['case'],replan=i,path=z['path'],scoring_mode=mode,skip=str(z['skip']),score_source_full_steps=str(z['score_source_full_steps']),score_error_ratio=z['score_error_ratio'],history_bytes=z['history_bytes'],error_buffer_bytes=z['error_buffer_bytes'],rpc_ms=q['rpc_ms'],**{k:v for k,v in q['timing'].items() if k.endswith('_ms')}))
            for block in cycles(qs):
                cycle_rows.append(dict(condition=name,**r['case'],start_replan=block[0]['replan'],policy_ms=float(np.mean([q['timing']['policy_wall_ms'] for q in block])),selector_ms=float(np.mean([q['timing']['selector_total_ms'] for q in block])),error_build_ms=float(np.mean([q['timing']['error_build_ms'] for q in block])),rpc_ms=float(np.mean([q['rpc_ms'] for q in block]))))
            base_costs[name]+=[float(np.mean([q['timing']['policy_wall_ms'] for q in block])) for block in cycles(oq)]
            episodes.append(dict(condition=name,budget=cfg['budget'],**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],error_fulls=error_fulls,full_mask_changes=mask_changes,selector_cold_s=sum(q['timing']['selector_capture_ms'] for q in qs)/1000))
            audits.append(dict(condition=name,case=key,passed=True,warm_prefix_requests=4 if excluded else min(7,len(qs)),warm_prefix_exact=not excluded,primary_pair_included=not excluded,error_fulls=error_fulls))
    summary=[];paired=[]
    for name,cfg in configs.items():
        ee=[r for r in episodes if r['condition']==name];qq=[q for q in requests if q['condition']==name and q['replan']>=9];cc=[r for r in cycle_rows if r['condition']==name];aa=data[name];bb=baseline[name]
        row=dict(condition=name,budget=cfg['budget'],n=len(ee),success=sum(r['success'] for r in ee),original_success=sum(r['success'] for d,r in bb.values()),fixed_v12_success=sum(r['success'] for r in fixed.values()),steady_cycles=len(cc),cycle_policy_ms=float(np.mean([r['policy_ms'] for r in cc])),original_cycle_policy_ms=float(np.mean(base_costs[name])),cycle_selector_ms=float(np.mean([r['selector_ms'] for r in cc])),cycle_error_build_ms=float(np.mean([r['error_build_ms'] for r in cc])),mean_episode_wall_s=float(np.mean([r['wall_s'] for r in ee])),mean_steps=float(np.mean([r['actions'] for r in ee])),mean_selector_cold_s=float(np.mean([r['selector_cold_s'] for r in ee])),episodes_using_error_score=sum(r['error_fulls']>0 for r in ee),max_history_mib=max(q['history_bytes'] for q in qq)/2**20,max_error_buffer_mib=max(q['error_buffer_bytes'] for q in qq)/2**20)
        for suite in SUITES:row[suite+'_success']=sum(r['success'] for r in ee if r['suite']==suite)
        for path in ['full','fast']:
            vals=[q['policy_wall_ms'] for q in qq if q['path']==path];row[path+'_policy_ms']=float(np.mean(vals));row[path+'_p95_ms']=float(np.percentile(vals,95))
        summary.append(row)
        pair_keys=keys-{PREFIX_EXCEPTION[1]} if cfg['budget']==14 else keys
        for label,base in [(cfg['base_id'],{k:r for k,(d,r) in bb.items()}),('fixed_v12_extrapolate',fixed)]:
            diffs=[dict(suite=k[0],task=k[1],delta=int(aa[k][1]['success'])-int(base[k]['success'])) for k in sorted(pair_keys)];ci=paired_ci(diffs)
            paired.append(dict(baseline=label,candidate=name,n=len(pair_keys),same_budget=label!='fixed_v12_extrapolate' or cfg['budget']==12,recovered=sum(not base[k]['success'] and aa[k][1]['success'] for k in pair_keys),lost=sum(base[k]['success'] and not aa[k][1]['success'] for k in pair_keys),delta_pp=float(np.mean([r['delta'] for r in diffs])*100),ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    for file,rows in [('episodes.csv',episodes),('requests.csv',requests),('cycles.csv',cycle_rows),('summary.csv',summary),('paired.csv',paired)]:write_csv(OUT/file,rows)
    gates=[json.loads(p.read_text()) for p in (OUT/'workers').glob('gpu*/gate.json')];assert gates and all(g['passed'] for g in gates)
    lines=['# 补偿误差方向选层：闭环初筛','','两个预算各40条同场景自由闭环，共80条新记录。Full-D10→Fast-D2→Fast-D2，时间指数偏后、七维正等权、热态一次VJP、max2。前两次Full沿用原Jacobian；第三次起按本路径最近两次Full对当前Full的残差预测误差方向评分。代理只使用已发生的观测；没有提前获得后两个Fast的真实误差。','','|预算|原Jacobian＋外推|误差方向＋外推|固定V12＋外推参考|原/新稳态周期ms/请求|新Full/Fast ms|','|---|---|---|---|---|---|']
    for r in summary:lines.append(f"|V{r['budget']}|{r['original_success']}/40|{r['success']}/40|{r['fixed_v12_success']}/40|{r['original_cycle_policy_ms']:.2f}/{r['cycle_policy_ms']:.2f}|{r['full_policy_ms']:.2f}/{r['fast_policy_ms']:.2f}|")
    lines+=['','固定V12对V14为跨预算参照，V12才是同预算比较。固定结果从已完成200条中按相同40个case取交集，首输入/初态/Full身份已核验，不混用全200条成功率。','','|预算|Spatial|Object|Goal|Long|平均步数|平均回合秒|冷选层秒/回合|启用新评分回合|','|---|---|---|---|---|---|---|---|---|']
    for r in summary:lines.append('|'+ '|'.join([f"V{r['budget']}",*[f"{r[s+'_success']}/10" for s in SUITES],f"{r['mean_steps']:.1f}",f"{r['mean_episode_wall_s']:.2f}",f"{r['mean_selector_cold_s']:.2f}",f"{r['episodes_using_error_score']}/40"] )+'|')
    lines+=['','|配对比较|恢复/新增失败|差值pp|探索95%区间pp|','|---|---|---|---|']
    for r in paired:lines.append(f"|{r['baseline']}→{r['candidate']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.1f}|[{r['ci_low_pp']:+.1f},{r['ci_high_pp']:+.1f}]|")
    lines+=['','区间为suite分层task聚类bootstrap2000次，未校正多重比较；固定40条探索规模，未按显著性扩样。79/80条的新旧前缀动作及输入严格通过核验。V14 Spatial task7/init19的前20动作、末端位姿与模拟时间完全相同，但step20输入hash出现一次未复现的差异，早于新评分启用；新旧都成功。固定动作同卡/跨卡四次重放均复现旧hash，具体致因未确定。保留全部80条原结果，不替换或重跑到成功；该场景从V14主配对分析中排除，因此V14主配对n=39、V12 n=40。总体成功数仍按40条描述，另列all40敏感性分析。所有80条均自由运行，误差和补偿历史只来自本条件自身路径。', '', '稳态周期从第4次Full（replan9）开始取完整三轮；原参考也用相同截取规则。第一次原图构建、第三Full新图构建、切换释放原图及误差buffer准备等成本全部包含在真实episode墙钟；CUDA图冷构建会执行额外预热/捕获，不应把热态一次VJP误读为冷构建也只执行一次。', '', '原实测8输入240组显示误差方向评分平均新增1.018ms/Full，但本轮是真实自由轨迹成本，不能用热态微基准替代episode成本。缓存两次Full各17层FP32残差；误差方向另有一个工作buffer和图输入副本，内存不等同于只保存两份历史。', '', 'Jacobian为一次加权投影，而非完整35维范数；有限幅度多层交互仍未直接优化。PyTorch D10重放与BF16原生Full已知不是逐位相同，前轮核验前5动作raw RMS最大0.007581、平移最大差6.367mm；中性门控Full重放输出和Graph/eager梯度一致性已在8真实输入验证并复用。','','[冻结方案](idea1_v3_4_E2_comp_error_closedloop_plan.md)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_comp_error_closedloop_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'prefix_audit.json',dict(passed=True,checks=audits,requests=prefix_requests));atomic(OUT/'audit.json',dict(passed=True,episodes=80,source_hashes=True,causal_three_full_history=True,warm_prefix_exact_episodes=79,primary_pair_exclusions=[dict(condition=PREFIX_EXCEPTION[0],case=PREFIX_EXCEPTION[1])],own_history=True,fast_no_probe=True,time=time.time()));atomic(OUT/'complete.json',dict(report=str(report),time=time.time()));atomic(OUT/'status.json',dict(status='complete',progress=progress(),time=time.time()))
if __name__=='__main__':main()
