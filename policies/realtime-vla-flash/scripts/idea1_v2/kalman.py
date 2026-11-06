from __future__ import annotations
import csv
import hashlib
import json
import numpy as np
from .core import Kalman, SUITES, write_json

GRID = (1e-6,1e-5,1e-4,1e-3,1e-2)
PREFIXES = (1,2,3,4,5,8,12)

def predictions(records,q,r):
    k = Kalman(q,r)
    last = 0
    for record in records:
        t = int(record['environment_action_step'])
        if t < last:
            raise ValueError('Nonmonotonic history cutoff')
        for action in record['_actions'][last:t]:
            k.observe(action)
        last = t
        mean,cov = k.forecast()
        hold = np.tile(np.clip(record['_actions'][t-1,:3],-1,1) if t else np.zeros(3),(50,1))
        yield record,mean,cov,hold

def equal_score(values):
    # Replan -> episode -> task -> suite; task keys do not mix suites.
    suite_means=[]
    for suite in SUITES:
        task_means=[]
        for task in sorted({key[1] for key in values if key[0]==suite}):
            task_means.append(np.mean([np.mean(v) for key,v in values.items() if key[:2]==(suite,task)]))
        suite_means.append(np.mean(task_means))
    return float(np.mean(suite_means))

def table(path,records,exclusive=True):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x' if exclusive else 'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

def run(groups,root):
    destination=root/'kalman'
    if destination.exists():
        raise FileExistsError('Kalman output exists; use a separate attempt, never recalibrate in place')
    destination.mkdir(parents=True)
    scores=[]
    calibration={key:value for key,value in groups.items() if key[1]==0}
    for q in GRID:
        for r in GRID:
            values={}
            for key,records in calibration.items():
                values[key]=[float(np.linalg.norm((np.clip(mean,-1,1)-np.clip(rec['full_actions'],-1,1)[:,:3])*50,axis=1)[:rec['_prefix']].mean())
                             for rec,mean,_,_ in predictions(records,q,r)]
            scores.append({'q0':q,'r0':r,'score_mm':equal_score(values)})
    minimum=min(s['score_mm'] for s in scores)
    winner=min((s for s in scores if s['score_mm']<=minimum+1e-12),key=lambda s:(s['q0'],s['r0']))
    parameters=dict(winner,dt=.05,selection='calibration_only_equal_suite_task_episode_replan',model_calls=0,environment_actions=0)
    parameter_hash=hashlib.sha256(json.dumps(parameters,sort_keys=True).encode()).hexdigest()
    # Freeze before any evaluation metrics are inspected.
    write_json(root/'protocol/kalman_parameters.json',dict(parameters,sha256=parameter_hash),exclusive=True)
    table(destination/'calibration_scores.csv',scores)
    means=[]; covs=[]; metadata=[]; errors=[]
    pairs=('Kalman-Full','HoldLast-Full','V14-Full','Kalman-V14')
    for key,records in sorted(groups.items()):
        for rec,mean,cov,hold in predictions(records,winner['q0'],winner['r0']):
            full=np.clip(rec['full_actions'],-1,1)[:,:3]
            variant=np.clip(rec['variant_actions'],-1,1)[:,:3]
            differences=np.asarray([np.clip(mean,-1,1)-full,hold-full,variant-full,np.clip(mean,-1,1)-variant])*50
            errors.append(differences)
            means.append(mean);covs.append(cov)
            t=rec['environment_action_step']
            metadata.append({'suite':key[0],'task':key[1],'init':key[2],'replan':rec['replan_id'],'step':t,
                             'history_count':t,'split':'calibration' if key[1]==0 else 'evaluation',
                             'phase':rec['semantic_phase'],'valid_executed_prefix':rec['_prefix'],
                             'last_assimilated_action':None if not t else rec['_actions'][t-1].tolist(),
                             'saturation_fraction':float(np.mean(np.abs(mean)>1)),
                             'parameter_sha256':parameter_hash,'source_input_sha256':rec['full_input_sha256']})
    error=np.asarray(errors)
    np.savez_compressed(destination/'predictions.npz',raw_mean=np.asarray(means),clipped_mean=np.clip(means,-1,1),covariance=np.asarray(covs),signed_error_mm=error)
    with (destination/'records.jsonl').open('x') as stream:
        for record in metadata:
            stream.write(json.dumps(record,allow_nan=False)+'\n')
    summaries=[];prefix_rows=[]
    for split in ('calibration','evaluation'):
        selected=[i for i,m in enumerate(metadata) if m['split']==split]
        for p,pair in enumerate(pairs):
            l2=np.linalg.norm(error[selected,p],axis=-1)
            for offset in range(50):
                values={}
                for j,i in enumerate(selected):
                    m=metadata[i];values.setdefault((m['suite'],m['task'],m['init']),[]).append(l2[j,offset])
                summaries.append({'split':split,'pair':pair,'offset':offset+1,'equal_mean_mm':equal_score(values),
                                  'sample_p50_mm':float(np.percentile(l2[:,offset],50)),'sample_p95_mm':float(np.percentile(l2[:,offset],95)),
                                  'max_mm':float(l2[:,offset].max()),'points':len(selected)})
            for h in PREFIXES:
                for executed in (False,True):
                    values={};n=0
                    for j,i in enumerate(selected):
                        m=metadata[i];count=min(h,m['valid_executed_prefix']) if executed else h
                        values.setdefault((m['suite'],m['task'],m['init']),[]).append(float(l2[j,:count].mean()));n+=count
                    prefix_rows.append({'split':split,'pair':pair,'h':h,'executed_prefix_only':executed,'equal_mean_mm':equal_score(values),'actions':n,'points':len(selected)})
    table(root/'analysis/kalman_offsets.csv',summaries)
    table(root/'analysis/kalman_prefixes.csv',prefix_rows)
    phases={'initial':'approach','approach':'approach','contact':'contact_grasp','grasp_or_manipulate':'contact_grasp',
            'transport':'transport','goal_alignment':'alignment_release','release_or_final_action':'alignment_release'}
    strata=[]
    for split in ('calibration','evaluation'):
        for grouping,levels in (('phase',('approach','contact_grasp','transport','alignment_release','unknown')),
                                ('history',('0','1-4','>=5'))):
            for level in levels:
                chosen=[]
                for i,m in enumerate(metadata):
                    actual=phases.get(m['phase'],'unknown') if grouping=='phase' else '0' if m['history_count']==0 else '1-4' if m['history_count']<5 else '>=5'
                    if m['split']==split and actual==level:chosen.append(i)
                for p,pair in enumerate(pairs):
                    values={}
                    for i in chosen:
                        m=metadata[i]
                        values.setdefault((m['suite'],m['task'],m['init']),[]).append(float(np.linalg.norm(error[i,p,:5],axis=-1).mean()))
                    # Missing strata are missing, never zero; averaging uses covered suites/tasks.
                    suite_scores=[]
                    for suite in SUITES:
                        task_scores=[]
                        for task in sorted({k[1] for k in values if k[0]==suite}):
                            task_scores.append(np.mean([np.mean(v) for k,v in values.items() if k[:2]==(suite,task)]))
                        if task_scores:suite_scores.append(np.mean(task_scores))
                    strata.append({'split':split,'grouping':grouping,'level':level,'pair':pair,'points':len(chosen),
                                   'episodes':len(values),'tasks':len({k[:2] for k in values}),
                                   'equal_covered_mean_mm':float(np.mean(suite_scores)) if suite_scores else None,
                                   'saturation_sample_fraction':float(np.mean([metadata[i]['saturation_fraction'] for i in chosen])) if chosen else None})
    table(root/'analysis/kalman_strata.csv',strata)
    axes=[]
    for split in ('calibration','evaluation'):
        selected=[i for i,m in enumerate(metadata) if m['split']==split]
        for p,pair in enumerate(pairs):
            for offset in range(50):
                for axis in range(3):
                    values={}
                    for i in selected:
                        m=metadata[i];values.setdefault((m['suite'],m['task'],m['init']),[]).append(error[i,p,offset,axis])
                    axes.append({'split':split,'pair':pair,'offset':offset+1,'axis':'xyz'[axis],
                                 'equal_signed_mean_mm':equal_score(values),'sample_p50_mm':float(np.percentile(error[selected,p,offset,axis],50)),
                                 'sample_p95_mm':float(np.percentile(error[selected,p,offset,axis],95))})
    table(root/'analysis/kalman_axes.csv',axes)
    # Bin boundaries are frozen from calibration, then applied without refitting.
    disagreement=np.linalg.norm(error[:,3],axis=-1)[:,:5].mean(axis=1)
    edges=np.quantile(disagreement[[m['split']=='calibration' for m in metadata]],[.2,.4,.6,.8])
    write_json(root/'protocol/divergence_bins.json',{'edges_mm':edges.tolist(),'source':'calibration'},exclusive=True)
    bins=[]
    for b in range(5):
        chosen=[i for i,m in enumerate(metadata) if m['split']=='evaluation' and np.searchsorted(edges,disagreement[i],side='right')==b]
        bins.append({'bin':b,'points':len(chosen),'tasks':len({(metadata[i]['suite'],metadata[i]['task']) for i in chosen}),
                     'v14_full_sample_mean_mm':float(np.linalg.norm(error[chosen,2,:5],axis=-1).mean()) if chosen else None,
                     'kalman_full_sample_mean_mm':float(np.linalg.norm(error[chosen,0,:5],axis=-1).mean()) if chosen else None})
    table(root/'analysis/divergence_bins.csv',bins)
    return {'status':'complete','points':len(metadata),'calibration':634,'evaluation':6345,'model_calls':0,'environment_actions':0,'parameter_sha256':parameter_hash}
