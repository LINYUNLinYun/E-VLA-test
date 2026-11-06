"""Incremental descriptive results; final audit is separate from green tests."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .manifest import ROOT,OUT,SUITES


def read_csv(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def write_csv(path,rows):
    if not rows: return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(rows)


def key(r):
    return tuple(r['case'][k] for k in ('suite','task','init'))


def paired_ci(pairs,iterations=2000):
    """Suite stratified task cluster resampling, init nested in each task."""
    rng=np.random.default_rng(281034); estimates=[]
    groups=[]
    for suite in SUITES:
        tasks=sorted({r['task'] for r in pairs if r['suite']==suite})
        groups.append(np.array([np.mean([r['delta'] for r in pairs if r['suite']==suite and r['task']==task]) for task in tasks]))
    if any(len(g)==0 for g in groups): return None
    for _ in range(iterations): estimates.append(np.mean([rng.choice(g,len(g),replace=True).mean() for g in groups]))
    return np.percentile(estimates,[2.5,97.5]).tolist()


def quality(out):
    m=json.loads((OUT/'manifest.json').read_text()); allrows={}; flat=[]; summary=[]
    for cfg in m['conditions']:
        rows=[]
        for p in (OUT/cfg['id']/'episodes').glob('*/task*/init*/result.json'):
            if '.invalid.' in p.parent.name or not (p.parent/'complete.json').exists(): continue
            r=json.loads(p.read_text())
            if not r.get('valid'): continue
            if r['manifest_hash']!=m['manifest_hash']: raise ValueError('Result manifest mismatch')
            r['directory']=str(p.parent); rows.append(r)
            flat.append(dict(condition=cfg['id'],**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],
                wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],rpc_s=r['timing']['rpc_s'],step_s=r['timing']['step_s'],
                reset_s=r['timing']['reset_s'],shadow_s=r['timing']['shadow_s'],input_save_s=r['timing']['input_save_s'],
                full_requests=r['counts']['full'],draft_requests=r['counts']['draft'],verify_requests=r['counts']['verify'],source=str(p)))
        allrows[cfg['id']]={key(r):r for r in rows}
        for scope in ('first40','all'):
            selected=[r for r in rows if scope=='all' or r['case']['init']==14]
            for suite in ('all',*SUITES):
                rs=[r for r in selected if suite=='all' or r['case']['suite']==suite]
                if not rs: continue
                wall=sum(r['wall_s'] for r in rs); actions=sum(r['actions'] for r in rs)
                summary.append(dict(condition=cfg['id'],scope=scope,suite=suite,n=len(rs),target=(cfg['n'] if scope=='all' else 40)//(1 if suite=='all' else 4),
                    success=sum(r['success'] for r in rs),success_rate=np.mean([r['success'] for r in rs]),
                    actions_mean=actions/len(rs),policy_ms_per_action=sum(r['policy_wall_s'] for r in rs)*1000/actions,
                    policy_ms_per_request=sum(r['policy_wall_s'] for r in rs)*1000/sum(r['predictions'] for r in rs),
                    worker_episode_per_hour=len(rs)*3600/wall,wall_mean_s=wall/len(rs),
                    worker_actions_per_second=actions/wall,worker_requests_per_second=sum(r['predictions'] for r in rs)/wall,
                    rpc_wall_fraction=sum(r['timing']['rpc_s'] for r in rs)/wall,
                    physics_step_wall_fraction=sum(r['timing']['step_s'] for r in rs)/wall,
                    reset_wall_fraction=sum(r['timing']['reset_s'] for r in rs)/wall,
                    environment_acquire_wall_fraction=sum(r['timing']['environment_acquire_s'] for r in rs)/wall,
                    input_save_wall_fraction=sum(r['timing']['input_save_s'] for r in rs)/wall,
                    shadow_wall_fraction=sum(r['timing']['shadow_s'] for r in rs)/wall,
                    full_request_ratio=sum(r['counts']['full'] for r in rs)/sum(r['predictions'] for r in rs)))
    comparisons=[]
    for baseline,candidate in (('periodic_v14','periodic_old16'),('periodic_v14','periodic_old13'),('periodic_v14','periodic_old12'),
                               ('v14_d2','periodic_old16'),('full_d10','periodic_old16'),('full_d10','v14_d2')):
        a,b=allrows[baseline],allrows[candidate]
        for scope in ('first40','all'):
            shared=sorted(set(a)&set(b)); shared=[k for k in shared if scope=='all' or k[2]==14]
            if not shared: continue
            pairs=[dict(suite=k[0],task=k[1],delta=int(b[k]['success'])-int(a[k]['success'])) for k in shared]
            for suite in ('all',*SUITES):
                keys=[k for k in shared if suite=='all' or k[0]==suite]
                if not keys: continue
                ci=paired_ci(pairs) if suite=='all' else None
                comparisons.append(dict(baseline=baseline,candidate=candidate,scope=scope,suite=suite,n=len(keys),
                    baseline_success=sum(a[k]['success'] for k in keys),candidate_success=sum(b[k]['success'] for k in keys),
                    recovered=sum(not a[k]['success'] and b[k]['success'] for k in keys),lost=sum(a[k]['success'] and not b[k]['success'] for k in keys),
                    delta_pp=np.mean([int(b[k]['success'])-int(a[k]['success']) for k in keys])*100,
                    delta_ci_low_pp=None if ci is None else ci[0]*100,delta_ci_high_pp=None if ci is None else ci[1]*100,
                    paired_policy_cost_ratio=(sum(a[k]['policy_wall_s'] for k in keys)/sum(a[k]['actions'] for k in keys))/(sum(b[k]['policy_wall_s'] for k in keys)/sum(b[k]['actions'] for k in keys))))
    tasks=[]
    for name,rows in allrows.items():
        for suite in SUITES:
            for task in range(10):
                selected=[r for k,r in rows.items() if k[0]==suite and k[1]==task]
                if selected: tasks.append(dict(condition=name,suite=suite,task=task,n=len(selected),success=sum(r['success'] for r in selected),success_rate=np.mean([r['success'] for r in selected]),actions_mean=np.mean([r['actions'] for r in selected])))
    write_csv(out/'episodes.csv',flat); write_csv(out/'quality_summary.csv',summary); write_csv(out/'paired_quality.csv',comparisons); write_csv(out/'task_summary.csv',tasks)
    return summary,comparisons


def mechanism(out):
    records=[]; layers=[]; timings=[]; pairmeta=[]; effects=[]
    for complete in (OUT/'mechanism').glob('shard*/pair*/complete.json'):
        d=complete.parent; p=json.loads((d/'pair.json').read_text()); suite,task,_=p['episode'].split('/')
        pairid=str(d.relative_to(OUT/'mechanism'))
        for r in read_csv(d/'action_metrics.csv'):
            records.append(dict(r,pair=pairid,episode=p['episode'],suite=suite,task=int(task[4:])))
        for r in read_csv(d/'layer_metrics.csv'):
            layers.append(dict(r,pair=pairid,episode=p['episode'],suite=suite,task=int(task[4:])))
        timings += [dict(r,pair=pairid,episode=p['episode'],suite=suite) for r in json.loads((d/'timings.json').read_text())]
        with np.load(d/'actions.npz') as saved:
            arrays={name:saved[name] for name in saved.files if name.endswith('_raw')}
            for denoise in (2,10):
                base=arrays[f'v14_D{denoise}_raw'][:5,:6].astype(float)
                target=arrays[f'full_D{denoise}_raw'][:5,:6].astype(float)
                error=base-target
                for start in range(1,19):
                    intervention=arrays[f'old_single_L{start}_D{denoise}_raw'][:5,:6].astype(float)
                    delta=intervention-base
                    effects.append(dict(pair=pairid,suite=suite,task=int(task[4:]),D=denoise,kind='single',layer=start,
                        response_rms=float(np.sqrt(np.mean(delta**2))),error_alignment=float(np.mean(error*delta)),
                        squared_response=float(np.mean(delta**2)),delta_mse=float(np.mean((intervention-target)**2)-np.mean(error**2))))
                for start in (12,13,14,15,16,17,18):
                    intervention=arrays[f'old_suffix_L{start}_D{denoise}_raw'][:5,:6].astype(float)
                    delta=intervention-base
                    linear=sum((arrays[f'old_single_L{i}_D{denoise}_raw'][:5,:6].astype(float)-base) for i in range(start,19))
                    effects.append(dict(pair=pairid,suite=suite,task=int(task[4:]),D=denoise,kind='suffix',layer=start,
                        response_rms=float(np.sqrt(np.mean(delta**2))),error_alignment=float(np.mean(error*delta)),
                        squared_response=float(np.mean(delta**2)),delta_mse=float(np.mean((intervention-target)**2)-np.mean(error**2)),
                        interaction_rms=float(np.sqrt(np.mean((delta-linear)**2)))))
        with np.load(p['previous']['path']) as old,np.load(p['current']['path']) as current:
            pairmeta.append(dict(pair=pairid,episode=p['episode'],suite=suite,task=int(task[4:]),
                eef_displacement_m=float(np.linalg.norm(current['robot0_eef_pos']-old['robot0_eef_pos'])),
                agent_rgb_mae=float(np.abs(current['agentview_image'].astype(float)-old['agentview_image'].astype(float)).mean()),
                wrist_rgb_mae=float(np.abs(current['robot0_eye_in_hand_image'].astype(float)-old['robot0_eye_in_hand_image'].astype(float)).mean())))
    if not records: return {},[]
    layer_summary=[]; layer_groups={}
    for r in layers:
        for suite in ('all',r['suite']):
            layer_groups.setdefault((r['candidate'],r['kv'],r['region'],suite,int(r['layer'])),[]).append(r)
    for (candidate,kv,region,suite,layer),chosen in sorted(layer_groups.items()):
        layer_summary.append(dict(candidate=candidate,kv=kv,region=region,suite=suite,layer=layer,n=len(chosen),
            **{field:np.mean([float(r[field]) for r in chosen]) for field in ('relative_l2','cosine','rms_ratio')}))
    write_csv(out/'mechanism_actions.csv',records); write_csv(out/'mechanism_layers.csv',layers); write_csv(out/'mechanism_layer_summary.csv',layer_summary); write_csv(out/'observation_changes.csv',pairmeta); write_csv(out/'mechanism_layer_effects.csv',effects)
    fields=('candidate','D','horizon','component')
    groups={}
    for r in records: groups.setdefault(tuple(r[k] for k in fields),[]).append(r)
    summary=[]
    for ks,rs in groups.items():
        for suite in ('all',*SUITES):
            chosen=[r for r in rs if suite=='all' or r['suite']==suite]
            if not chosen: continue
            summary.append(dict(zip(fields,ks),suite=suite,n=len(chosen),mean_l2=float(np.mean([float(r['mean_l2']) for r in chosen])),
                rmse=float(np.mean([float(r['rmse']) for r in chosen])),
                controller_delta_proxy_mm=float(np.mean([float(r['clipped_controller_translation_mm_proxy']) for r in chosen])) if ks[-1]=='translation' else None,
                sign_disagreement=float(np.mean([float(r['sign_mismatch_fraction']) for r in chosen])) if ks[-1]=='gripper' else None))
    write_csv(out/'mechanism_summary.csv',summary)
    # Paired errors, clustered by original task rather than individual request.
    paired=[]
    for d in ('2','10'):
        selected=[r for r in records if r['D']==d and r['horizon']=='5' and r['component']=='translation']
        base={r['pair']:float(r['clipped_controller_translation_mm_proxy']) for r in selected if r['candidate']=='v14'}
        for candidate in sorted({r['candidate'] for r in selected}-{'v14'}):
            rs=[r for r in selected if r['candidate']==candidate]
            diff=[dict(suite=r['suite'],task=r['task'],delta=float(r['clipped_controller_translation_mm_proxy'])-base[r['pair']]) for r in rs]
            ci=paired_ci(diff)
            paired.append(dict(candidate=candidate,D=d,n=len(rs),delta_mm=np.mean([r['delta'] for r in diff]),
                improved_ratio=np.mean([r['delta']<0 for r in diff]),low=None if ci is None else ci[0],high=None if ci is None else ci[1]))
    write_csv(out/'mechanism_paired.csv',paired)
    historical=[]
    for denoise in ('2','10'):
        for scope,tasks in (('all',set(range(10))),('prior_tasks_current_init14',{0,5}),('other_tasks_current_init14',set(range(10))-{0,5})):
            chosen=[r for r in records if r['D']==denoise and r['horizon']=='5' and r['component']=='translation' and r['task'] in tasks]
            base={r['pair']:float(r['clipped_controller_translation_mm_proxy']) for r in chosen if r['candidate']=='v14'}
            hybrid={r['pair']:float(r['clipped_controller_translation_mm_proxy']) for r in chosen if r['candidate']=='old_suffix_L16'}
            shared=sorted(set(base)&set(hybrid))
            if shared: historical.append(dict(D=denoise,scope=scope,n=len(shared),v14_mm=np.mean([base[k] for k in shared]),
                old16_mm=np.mean([hybrid[k] for k in shared]),delta_mm=np.mean([hybrid[k]-base[k] for k in shared]),posthoc=True))
    write_csv(out/'historical_task_scope_diagnostic.csv',historical)
    changes=[]
    base={r['pair']:float(r['clipped_controller_translation_mm_proxy']) for r in records if r['D']=='2' and r['horizon']=='5' and r['component']=='translation' and r['candidate']=='v14'}
    old16={r['pair']:float(r['clipped_controller_translation_mm_proxy']) for r in records if r['D']=='2' and r['horizon']=='5' and r['component']=='translation' and r['candidate']=='old_suffix_L16'}
    for suite in ('all',*SUITES):
        chosen=[r for r in pairmeta if suite=='all' or r['suite']==suite]
        if not chosen: continue
        delta=[old16[r['pair']]-base[r['pair']] for r in chosen]
        changes.append(dict(suite=suite,n=len(chosen),old16_delta_mm=np.mean(delta),
            **{field+'_mean':np.mean([r[field] for r in chosen]) for field in ('eef_displacement_m','agent_rgb_mae','wrist_rgb_mae')},
            **{field+'_delta_pearson':correlation([r[field] for r in chosen],delta) for field in ('eef_displacement_m','agent_rgb_mae','wrist_rgb_mae')}))
    write_csv(out/'observation_change_summary.csv',changes)
    timingrows=[]
    for path in sorted({r['path'] for r in timings}):
        rs=[r for r in timings if r['path']==path]
        timingrows.append(dict(path=path,n=len(rs),wall_ms_mean=np.mean([r['wall_ms'] for r in rs]),wall_ms_p50=np.median([r['wall_ms'] for r in rs]),
            vision_ms=np.mean([r['prepare']['vision_ms'] for r in rs]),vlm_ms=np.mean([r['prepare']['vlm_ms'] for r in rs]),ae_ms=np.mean([r['sample']['ae_ms'] for r in rs])))
    write_csv(out/'same_input_timing.csv',timingrows)
    return dict(pairs=len(pairmeta),summary=summary,paired=paired,timing=timingrows,observation_changes=changes,historical_scope=historical),layers


def correlation(a,b):
    a,b=np.asarray(a).ravel(),np.asarray(b).ravel()
    if a.size<2 or np.std(a)==0 or np.std(b)==0: return None
    return float(np.corrcoef(a,b)[0,1])


def e3(out):
    records=[]; routes=[]; population=[]
    vectors={}; costs=[]
    for path in (OUT/'E3_parent'/'episodes').glob('*/task*/init*/result.json'):
        if '.invalid.' in path.parent.name or not (path.parent/'complete.json').exists(): continue
        if not json.loads(path.read_text()).get('valid'): continue
        d=path.parent; case=json.loads(path.read_text())['case']
        infer_by_round={}
        for line in (d/'infer.jsonl').read_text().splitlines():
            r=json.loads(line); route=r['route']
            infer_by_round[r['round']]=r
            routes.append(dict(**case,replan=r['replan'],path=route['path'],reason=route.get('full_reason'),
                accepted=route.get('accepted_length'),selected=r.get('shadow_selection',{}).get('selected',False),
                verify_gripper=route.get('verify_gripper'),spatial_length=route.get('spatial_length')))
        for shadow in sorted(d.glob('shadow_*.json')):
            data=json.loads(shadow.read_text())
            with np.load(shadow.with_suffix('.npz')) as saved:
                arrays={name:saved[name] for name in saved.files}
            group=data['group']; idx=shadow.stem
            parent=infer_by_round[data['round']]; route=parent['route']
            spatial_zero=route.get('spatial_length')==0
            gripper=bool(route.get('verify_gripper') or route.get('post_verify_gripper'))
            cause=('spatial_and_gripper' if spatial_zero and gripper else 'spatial' if spatial_zero else 'gripper' if gripper else 'nonzero')
            for item in data['metrics']:
                name=item['candidate']; a=arrays[name]; b=arrays['full_d10']; raw=arrays[name+'_raw']; rawref=arrays['full_d10_raw']
                first=a[:5]; ref=b[:5]
                delta=first-ref
                records.append(dict(**case,request=idx,group=group,cause=cause,candidate=name,accepted_length=item['accepted_length'],
                    parent_accepted_length=route['accepted_length'],parent_spatial_length=route['spatial_length'],
                    cache_age_steps=parent['full_cache_age']['env_steps'],
                    raw_first5_rmse=item['raw_first5_rmse'],translation_proxy_mm=float(np.linalg.norm(np.clip(first[:,:3],-1,1)-np.clip(ref[:,:3],-1,1),axis=1).mean()*50),
                    rotation_l2=float(np.linalg.norm(delta[:,3:6],axis=1).mean()),gripper_sign=float(np.mean((first[:,6]<0)!=(ref[:,6]<0))),
                    decoded_cosine=float(np.dot(first[:,:6].ravel(),ref[:,:6].ravel())/(np.linalg.norm(first[:,:6])*np.linalg.norm(ref[:,:6]))) if np.linalg.norm(first[:,:6])*np.linalg.norm(ref[:,:6]) else None,
                    ae_ms=item['cost'].get('ae_ms',data['parent_timing'].get('draft_ae_ms') if name=='v14_d2' else None),
                    current_full_prepare_ms=sum(float(v) for v in data['reference_full_prepare'].values()),previous_available=data['previous_available']))
                for comp,sl in (('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))):
                    vectors.setdefault((group,name,comp),[]).append((first[:,sl].ravel(),ref[:,sl].ravel()))
            costs.append(dict(**case,request=idx,group=group,shadow_wall_s=data['shadow_wall_s'],full_prepare_ms=sum(float(v) for v in data['reference_full_prepare'].values()),full_ae_ms=data['reference_full_sample']['ae_ms']))
    baseline={(r['suite'],r['task'],r['init'],r['request']):r for r in records if r['candidate']=='v14_d2'}
    for r in records:
        base=baseline[(r['suite'],r['task'],r['init'],r['request'])]
        r['translation_delta_mm']=r['translation_proxy_mm']-base['translation_proxy_mm']
        r['rotation_delta']=r['rotation_l2']-base['rotation_l2']
    summary=[]
    for group,name in sorted({(r['group'],r['candidate']) for r in records}):
        rs=[r for r in records if r['group']==group and r['candidate']==name]
        summary.append(dict(group=group,candidate=name,n=len(rs),translation_proxy_mm=np.mean([r['translation_proxy_mm'] for r in rs]),
            rotation_l2=np.mean([r['rotation_l2'] for r in rs]),gripper_sign=np.mean([r['gripper_sign'] for r in rs]),
            verify_nonzero_rate=np.mean([r['accepted_length']>0 for r in rs]),acceptance_mean=np.mean([r['accepted_length'] for r in rs]),
            ae_ms=np.mean([r['ae_ms'] for r in rs if r['ae_ms'] is not None]) if any(r['ae_ms'] is not None for r in rs) else None,
            current_full_prepare_ms=np.mean([r['current_full_prepare_ms'] for r in rs]),
            translation_improved_ratio=np.mean([r['translation_delta_mm']<0 for r in rs]),
            translation_delta_mm=np.mean([r['translation_delta_mm'] for r in rs]),rotation_improved_ratio=np.mean([r['rotation_delta']<0 for r in rs]),
            **{comp+'_pearson':correlation(np.concatenate([x[0] for x in vectors[(group,name,comp)]]),np.concatenate([x[1] for x in vectors[(group,name,comp)]])) for comp in ('translation','rotation','gripper')}))
    causes=[]
    for cause,name in sorted({(r['cause'],r['candidate']) for r in records}):
        rs=[r for r in records if r['cause']==cause and r['candidate']==name]
        causes.append(dict(cause=cause,candidate=name,n=len(rs),translation_proxy_mm=np.mean([r['translation_proxy_mm'] for r in rs]),
            translation_delta_mm=np.mean([r['translation_delta_mm'] for r in rs]),verify_nonzero_rate=np.mean([r['accepted_length']>0 for r in rs]),gripper_sign=np.mean([r['gripper_sign'] for r in rs])))
    write_csv(out/'E3_requests.csv',records); write_csv(out/'E3_routes.csv',routes); write_csv(out/'E3_summary.csv',summary); write_csv(out/'E3_costs.csv',costs); write_csv(out/'E3_by_cause.csv',causes)
    return dict(summary=summary,cause_summary=causes,selected_requests=len(costs),parent_requests=len(routes),verified_requests=sum(r['accepted'] is not None for r in routes),
        zero_requests=sum(r['accepted']==0 for r in routes),actual_selected_zero=sum(r['group']=='zero' for r in costs))


def plots(out,quality_rows,mech,layers):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    conditions=['full_d10','v14_d2','periodic_v14','periodic_old16']
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for i,name in enumerate(conditions):
        rs=[r for r in quality_rows if r['condition']==name and r['scope']=='all' and r['suite']!='all']
        if not rs: continue
        values={r['suite']:r for r in rs}
        axs[0].bar(np.arange(4)+(i-1.5)*.18,[values.get(s,{}).get('success_rate',np.nan)*100 for s in SUITES],width=.18,label=name)
        allr=next(r for r in quality_rows if r['condition']==name and r['scope']=='all' and r['suite']=='all')
        axs[1].bar(i,allr['policy_ms_per_action']); axs[1].text(i,allr['policy_ms_per_action'],f"n={allr['n']}",ha='center',va='bottom')
    axs[0].set_xticks(range(4),['Spatial','Object','Goal','Long']); axs[0].set_ylabel('Success (%)'); axs[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2)
    axs[1].set_xticks(range(4),conditions,rotation=25,ha='right'); axs[1].set_ylabel('Policy wall / executed action (ms)')
    fig.tight_layout(); fig.savefig(out/'closed_loop.png',dpi=180); plt.close(fig)
    if not mech: return
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    names=['v14',*[f'old_suffix_L{i}' for i in range(12,19)]]
    for d in ('2','10'):
        rs={(r['candidate'],r['suite']):r for r in mech['summary'] if r['D']==d and r['horizon']=='5' and r['component']=='translation'}
        axs[0].plot(range(len(names)),[rs.get((n,'all'),{}).get('controller_delta_proxy_mm',np.nan) for n in names],marker='o',label='D'+d)
    axs[0].set_xticks(range(len(names)),['V14']+[str(i) for i in range(12,19)]); axs[0].set_xlabel('Old suffix starts at layer'); axs[0].set_ylabel('Translation command delta proxy (mm)'); axs[0].legend()
    for suite in SUITES:
        rs={(r['candidate'],r['suite']):r for r in mech['summary'] if r['D']=='2' and r['horizon']=='5' and r['component']=='translation'}
        axs[1].plot(range(len(names)),[rs.get((n,suite),{}).get('controller_delta_proxy_mm',np.nan) for n in names],marker='o',label=suite)
    axs[1].set_xticks(range(len(names)),['V14']+[str(i) for i in range(12,19)]); axs[1].set_xlabel('Old suffix starts at layer'); axs[1].set_ylabel('D2 translation proxy (mm)'); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/'suffix_mechanism.png',dpi=180); plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,7))
    for col,key in enumerate(('K','V')):
        for candidate in ('v14','old'):
            values=[np.mean([float(r['relative_l2']) for r in layers if r['candidate']==candidate and r['kv']==key and r['region']=='all' and int(r['layer'])==layer]) for layer in range(1,19)]
            axs[0,col].plot(range(1,19),values,marker='.',label=candidate)
        for skipped in (3,4,9,11): axs[0,col].axvspan(skipped-.3,skipped+.3,color='grey',alpha=.15)
        axs[0,col].set_title(key+' relative L2 to current Full'); axs[0,col].set_xlabel('Original layer (grey = skipped hidden update)'); axs[0,col].legend()
    effects=read_csv(out/'mechanism_layer_effects.csv')
    for kind in ('single','suffix'):
        indices=sorted({int(r['layer']) for r in effects if r['kind']==kind})
        values=[np.mean([float(r['delta_mse']) for r in effects if r['kind']==kind and int(r['layer'])==layer and r['D']=='2']) for layer in indices]
        axs[1,0].plot(indices,values,marker='.',label=kind)
    axs[1,0].axhline(0,color='black',linewidth=.6); axs[1,0].set_xlabel('Layer / old suffix first layer'); axs[1,0].set_ylabel('D2 first5 normalized MSE change'); axs[1,0].legend()
    for candidate,positions in (('old_suffix_L',range(12,19)),('oracle_suffix_L',(12,13,16))):
        values=[next(r['controller_delta_proxy_mm'] for r in mech['summary'] if r['candidate']==candidate+str(layer) and r['D']=='2' and r['horizon']=='5' and r['component']=='translation' and r['suite']=='all') for layer in positions]
        axs[1,1].plot(list(positions),values,marker='o',label=candidate.replace('_L',''))
    axs[1,1].set_xlabel('Suffix first layer'); axs[1,1].set_ylabel('D2 translation command proxy (mm)'); axs[1,1].legend()
    fig.tight_layout(); fig.savefig(out/'layer_mechanism.png',dpi=180); plt.close(fig)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--plots',action='store_true'); a=p.parse_args()
    out=OUT/'analysis'; out.mkdir(exist_ok=True)
    summary,comparisons=quality(out); mech,layers=mechanism(out); e3result=e3(out)
    if a.plots and summary: plots(out,summary,mech,layers)
    atomic(out/'analysis_state.json',dict(time=time.time(),quality=summary,comparisons=comparisons,mechanism=mech,E3=e3result,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(quality_rows=len(summary),mechanism_pairs=mech.get('pairs',0),E3_selected=e3result['selected_requests'])))

if __name__=='__main__': main()
