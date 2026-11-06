"""Four-variant full-trajectory prediction risk analysis; no model/environment calls."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiment11.protocol import RESULTS,ROOT,SUITES
from scripts.experiment10.common import PHASE_MAP,sha256,dump_json,json_lines
from scripts.experiment11.analyze_failure_events import write_csv,rigid_heights
from scripts.experiment11.shadow_risk_metrics import METRICS,differences,longest_run,task_cluster_summary

VARIANTS=('V16_A02','V15_A08','V14_A03','V13_Custom')
PREFIXES=(1,2,3,4,5,8,12)


def full_state_maps():
    result,hashes={},{}
    for suite in SUITES:
        base=RESULTS/'closed_loop/Full18_D3_R5_FORMAL'/suite/'run'
        episode_path=base/'episode_log.json'
        episodes=json.loads(episode_path.read_text());hashes[str(episode_path)]=sha256(episode_path)
        for episode in episodes:
            path=base/episode['telemetry_path'];rows=json_lines(path)
            rows=sorted([r for r in rows if r.get('when') in ('initial','post_action',None)],key=lambda r:int(r['step']))
            initial=rigid_heights(rows[0]);history=[];states={}
            for r in rows:
                step=int(r['step']);primary=r.get('primary_objects') or []
                flags=r.get('robot_grasps')
                known=isinstance(flags,dict) and bool(primary) and all(n in flags for n in primary)
                grasp=any(bool(flags[n]) for n in primary) if known else None
                history.append((step,grasp))
                stable=(len(history)>=3 and all(v is True for _,v in history[-3:]) and
                        [s for s,_ in history[-3:]]==list(range(step-2,step+1)))
                heights=rigid_heights(r)
                lifted=any(heights.get(n,z)-z>=.01 for n,z in initial.items()) if initial else None
                states[step]={'phase':PHASE_MAP.get(r.get('semantic_phase'),'unknown'),
                              'grasp':grasp,'stable':stable,'lifted':lifted}
            result[(suite,int(episode['task_id']),int(episode['episode_idx']))]=states
            hashes[str(path)]=sha256(path)
    return result,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-id',default='round2_20261011')
    args=parser.parse_args()
    output=RESULTS/'analysis/next_round'/args.attempt_id/'shadow_risk'
    output.mkdir(parents=True,exist_ok=True)
    v14status=RESULTS/f'experiment11_v14_shadow_supplement_status_{args.attempt_id}.json'
    status=json.loads(v14status.read_text())
    if not status.get('complete') or status.get('validated_jobs')!=4 or status.get('active') or status.get('failures'):
        raise RuntimeError('V14 must have complete validated 4/4 coverage before combined analysis')
    controller=ROOT/'examples/libero/.venv/lib/python3.8/site-packages/robosuite/controllers/config/osc_pose.json'
    config=json.loads(controller.read_text())
    if (config['input_min']!=-1 or config['input_max']!=1 or config['output_max']!=[.05,.05,.05,.5,.5,.5] or
            config['output_min']!=[-.05,-.05,-.05,-.5,-.5,-.5] or not config['control_delta']):
        raise RuntimeError('Controller scale does not match the frozen conversion')
    states,hashes=full_state_maps();hashes[str(controller)]=sha256(controller);hashes[str(v14status)]=sha256(v14status)
    reference={}
    expected_keys=None
    offsets,prefixes,contrasts,directional,windows,stage_errors=[],[],[],[],[],[]
    plots={}
    for variant in VARIANTS:
        values,task_keys,suite_keys,phases,actual_phases,stable_states,lifted_states,full_closed,valid_steps=[],[],[],[],[],[],[],[],[]
        grasp_states,unknown_states=[],[]
        sample_keys=set()
        for suite in SUITES:
            directory=(RESULTS/'shadow_supplement_v14'/args.attempt_id/variant/suite if variant=='V14_A03'
                       else RESULTS/'shadow'/variant/suite)
            prediction,marker=directory/'predictions.jsonl',directory/'validated.json'
            m=json.loads(marker.read_text())
            if (not m.get('complete') or m.get('environment_steps_executed_by_variant')!=0 or
                    m.get('predictions_sha256')!=sha256(prediction)):
                raise RuntimeError('Prediction marker/hash invalid')
            hashes[str(prediction)]=sha256(prediction);hashes[str(marker)]=sha256(marker)
            n=0
            for line in prediction.open():
                row=json.loads(line);n+=1
                key=(suite,int(row['task_id']),int(row['episode_idx']),int(row['replan_id']))
                if key in sample_keys or row['variant_id']!=variant or row['suite']!=suite:
                    raise RuntimeError('Duplicate/mismatched prediction identity')
                sample_keys.add(key)
                full=np.asarray(row['full_actions'],dtype=np.float32)
                signature=(row['full_input_sha256'],row['noise_sha256'],hashlib.sha256(full.tobytes()).hexdigest())
                if variant==VARIANTS[0]:reference[key]=signature
                elif reference.get(key)!=signature:raise RuntimeError('Full/input/noise not paired across variants')
                d=differences(full,row['variant_actions']);values.append(d)
                task_keys.append(f'{suite}::{int(row["task_id"]):02d}');suite_keys.append(suite)
                phase=PHASE_MAP.get(row.get('semantic_phase'),'unknown');phases.append(phase)
                maps=states[key[:3]];step=int(row['environment_action_step'])
                actual,stable,lifted,valid,grasp,unknown=[],[],[],[],[],[]
                for k in range(1,6):
                    after,before=maps.get(step+k),maps.get(step+k-1)
                    actual.append(after['phase'] if after else 'unknown')
                    stable.append(bool(before and before['stable']))
                    lifted.append(bool(before and before['lifted'] is True))
                    valid.append(after is not None)
                    grasp.append(bool(before and before['grasp'] is True))
                    unknown.append(before is None or before['grasp'] is None)
                actual_phases.append(actual);stable_states.append(stable);lifted_states.append(lifted)
                valid_steps.append(valid);full_closed.append(full[:5,6]>0)
                grasp_states.append(grasp);unknown_states.append(unknown)
                flags=d[:5,8].astype(bool)&np.array(valid)
                window={'variant_id':variant,'suite':suite,'task_id':row['task_id'],'episode_idx':row['episode_idx'],
                        'replan_id':row['replan_id'],'environment_step':step,'replan_phase':phase,
                        'full_input_sha256':row['full_input_sha256'],'noise_sha256':row['noise_sha256'],
                        'executed_full_offsets':int(sum(valid)),'any_nonzero_gripper_disagreement':bool(flags.any()),
                        'any_early_close':bool((d[:5,5].astype(bool)&valid).any()),
                        'any_early_open':bool((d[:5,6].astype(bool)&valid).any()),
                        'longest_disagreement_run_in_executed_full_chunk':longest_run(flags),
                        'longest_disagreement_run_in_predicted_50':longest_run(d[:,8]),
                        'first_disagreement_offset':int(np.flatnonzero(flags)[0]+1) if flags.any() else None}
                windows.append(window)
            if n!=m['replan_prediction_count']:raise RuntimeError('JSONL coverage does not match marker')
        if len(sample_keys)!=6979:raise RuntimeError('Expected 6979 keys per variant')
        if expected_keys is None:expected_keys=sample_keys
        elif expected_keys!=sample_keys:raise RuntimeError('Variants do not cover identical keys')
        array=np.stack(values);tasks=np.array(task_keys);suites=np.array(suite_keys);phase_array=np.array(phases)
        actual=np.array(actual_phases);stable=np.array(stable_states);lifted=np.array(lifted_states)
        closed=np.array(full_closed);valid=np.array(valid_steps)
        grasp=np.array(grasp_states);unknown=np.array(unknown_states)
        for suite in ('ALL',*SUITES):
            mask=np.ones(len(array),dtype=bool) if suite=='ALL' else suites==suite
            data=array[mask];t=tasks[mask];strata=suites[mask]
            mean,ci,nt=task_cluster_summary(data,t,strata=strata)
            if suite=='ALL':plots[variant]=(mean,ci)
            for offset in range(50):
                for metric_index,metric in enumerate(METRICS):
                    cell=data[:,offset,metric_index]
                    offsets.append({'variant_id':variant,'suite':suite,'offset':offset+1,'metric':metric,
                        'n_predictions':len(data),'n_tasks':nt,'task_equal_mean':float(mean[offset,metric_index]),
                        'task_ci95_low':float(ci[0,offset,metric_index]),'task_ci95_high':float(ci[1,offset,metric_index]),
                        'sample_mean':float(cell.mean()),'sample_p50':float(np.quantile(cell,.5)),
                        'sample_p95':float(np.quantile(cell,.95)),'sample_max':float(cell.max())})
            for start,end in ((1,5),(1,12),(1,50)):
                delta=data[:,end-1,:]-data[:,start-1,:]
                dm,dc,dnt=task_cluster_summary(delta,t,strata=strata)
                for i,name in enumerate(METRICS):
                    contrasts.append({'variant_id':variant,'suite':suite,'from_offset':start,'to_offset':end,
                        'metric':name,'paired_tasks':dnt,'task_equal_delta':float(dm[i]),
                        'task_ci95_low':float(dc[0,i]),'task_ci95_high':float(dc[1,i])})
            for h in PREFIXES:
                pm=np.stack([data[:,:h,3].mean(1),data[:,:h,4].mean(1),
                    np.sqrt(np.mean(data[:,:h,2]**2,axis=1)),data[:,:h,8].mean(1),
                    data[:,:h,8].max(1),data[:,:h,3].max(1)],axis=-1)
                names=('translation_target_difference_mm_mean','rotation_increment_vector_difference_deg_mean',
                       'continuous6_command_rmse','gripper_disagreement_rate','any_gripper_disagreement',
                       'translation_target_difference_mm_max')
                mm,cc,n_tasks=task_cluster_summary(pm,t,strata=strata)
                for i,name in enumerate(names):
                    prefixes.append({'variant_id':variant,'suite':suite,'horizon':h,'metric':name,
                        'n_predictions':len(data),'n_tasks':n_tasks,'task_equal_mean':float(mm[i]),
                        'task_ci95_low':float(cc[0,i]),'task_ci95_high':float(cc[1,i]),
                        'sample_p50':float(np.quantile(pm[:,i],.5)),'sample_p95':float(np.quantile(pm[:,i],.95)),
                        'sample_max':float(pm[:,i].max())})
            for threshold in (1,2,5,10):
                pm=(data[:,:5,3]>threshold).mean(1)
                mm,cc,n_tasks=task_cluster_summary(pm[:,None],t,strata=strata)
                prefixes.append({'variant_id':variant,'suite':suite,'horizon':5,'metric':'translation_threshold_exceedance',
                                 'threshold_mm':threshold,'task_equal_mean':float(mm[0]),
                                 'task_ci95_low':float(cc[0,0]),'task_ci95_high':float(cc[1,0]),'n_tasks':n_tasks})
            for threshold in (.5,1,2,5):
                pm=(data[:,:5,4]>threshold).mean(1)
                mm,cc,n_tasks=task_cluster_summary(pm[:,None],t,strata=strata)
                prefixes.append({'variant_id':variant,'suite':suite,'horizon':5,'metric':'rotation_vector_threshold_exceedance',
                                 'threshold_deg':threshold,'task_equal_mean':float(mm[0]),
                                 'task_ci95_low':float(cc[0,0]),'task_ci95_high':float(cc[1,0]),'n_tasks':n_tasks})
        # Stage labels here are realized Full offsets1-5 only, never future variant stages.
        for phase in ('all','approach','contact_grasp','transport','alignment_release','unknown'):
            for condition,condition_mask in [('all',valid),('grasp_now',valid&grasp),('unknown_grasp',valid&unknown),
                       ('stably_grasped',valid&stable),('lifted_10mm',valid&lifted),
                       ('phase_transition',valid&(actual!=phase_array[:,None])),
                       ('full_closed_and_stably_grasped',valid&stable&closed)]:
                selection=condition_mask.copy()
                if phase!='all':selection&=actual==phase
                for direction,i in [('full_open_variant_close',5),('full_close_variant_open',6),
                                    ('zero_command_in_pair',7),('nonzero_sign_disagreement',8)]:
                    rates,rate_tasks,rate_suites=[],[],[]
                    for task in sorted(set(tasks)):
                        tm=tasks==task;take=selection[tm]
                        if take.any():
                            rates.append(float(array[tm,:5,i][take].mean()))
                            rate_tasks.append(task);rate_suites.append(task.split('::')[0])
                    numerator=float(array[:,:5,i][selection].sum())
                    if rates:
                        rm,rc,_=task_cluster_summary(np.asarray(rates)[:,None],np.array(rate_tasks),strata=np.array(rate_suites))
                        directional.append({'variant_id':variant,'phase':phase,'condition':condition,'direction':direction,
                            'n_tasks':len(rates),'n_full_executed_offsets':int(selection.sum()),
                            'disagreement_count':int(numerator),'task_equal_rate':float(np.mean(rates)),
                            'task_ci95_low':float(rc[0,0]),'task_ci95_high':float(rc[1,0]),
                            'sample_rate':numerator/selection.sum(),'stage_basis':'realized_Full_post_action',
                            'state_basis':'Full_before_action;not_variant_execution'})
                if condition=='all' and selection.any():
                    task_means,task_names,task_suites=[],[],[]
                    for task in sorted(set(tasks)):
                        tm=tasks==task;take=selection[tm]
                        if take.any():
                            task_means.append(array[tm,:5,:][take].mean(0));task_names.append(task)
                            task_suites.append(task.split('::')[0])
                    mm,cc,nn=task_cluster_summary(np.stack(task_means),np.array(task_names),strata=np.array(task_suites))
                    for i,name in enumerate(METRICS):
                        stage_errors.append({'variant_id':variant,'phase':phase,'metric':name,'n_tasks':nn,
                            'n_full_executed_offsets':int(selection.sum()),'task_equal_mean':float(mm[i]),
                            'task_ci95_low':float(cc[0,i]),'task_ci95_high':float(cc[1,i]),
                            'stage_basis':'realized_Full_offsets1_5;not_future_variant_stage'})
        print(json.dumps({'variant_analyzed':variant,'matched_keys':len(sample_keys)}),flush=True)
    episode_rates=[]
    grouped=defaultdict(list)
    for r in windows:grouped[(r['variant_id'],r['suite'],r['task_id'],r['episode_idx'])].append(r)
    for (v,s,t,i),rs in grouped.items():
        episode_rates.append({'variant_id':v,'suite':s,'task_id':t,'episode_idx':i,'replan_windows':len(rs),
            'any_early_open':any(r['any_early_open'] for r in rs),'any_early_close':any(r['any_early_close'] for r in rs),
            'any_disagreement':any(r['any_nonzero_gripper_disagreement'] for r in rs)})
    write_csv(output/'offset1_50_controller_metrics.csv',offsets)
    write_csv(output/'prefix_controller_metrics.csv',prefixes)
    write_csv(output/'paired_offset_contrasts.csv',contrasts)
    write_csv(output/'gripper_direction_state_phase.csv',directional)
    write_csv(output/'executed_full_prefix_stage_errors.csv',stage_errors)
    write_csv(output/'gripper_window_events.csv',windows)
    write_csv(output/'gripper_episode_events.csv',episode_rates)
    aggregation=[]
    for scope,items in [('window',windows),('episode',episode_rates)]:
        for v in VARIANTS:
            subset=[r for r in items if r['variant_id']==v]
            t=np.array([f'{r["suite"]}::{r["task_id"]}' for r in subset]);ss=np.array([r['suite'] for r in subset])
            fields=('any_early_open','any_early_close','any_disagreement') if scope=='episode' else (
                   'any_early_open','any_early_close','any_nonzero_gripper_disagreement')
            for field in fields:
                mm,cc,nn=task_cluster_summary(np.array([r[field] for r in subset],dtype=float)[:,None],t,strata=ss)
                aggregation.append({'variant_id':v,'scope':scope,'metric':field,'n_observations':len(subset),
                    'n_tasks':nn,'task_equal_probability':float(mm[0]),'task_ci95_low':float(cc[0,0]),
                    'task_ci95_high':float(cc[1,0]),'sample_probability':float(np.mean([r[field] for r in subset]))})
    write_csv(output/'gripper_window_episode_probabilities.csv',aggregation)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for end in (12,50):
        fig,axes=plt.subplots(1,3,figsize=(15,4))
        for v,(mean,ci) in plots.items():
            for ax,k,title,factor in [(axes[0],3,'Translation target difference (mm)',1),
                         (axes[1],4,'Rotation increment-vector difference (deg)',1),
                         (axes[2],8,'Nonzero gripper sign disagreement (%)',100)]:
                x=np.arange(1,end+1);ax.plot(x,mean[:end,k]*factor,label=v)
                ax.fill_between(x,ci[0,:end,k]*factor,ci[1,:end,k]*factor,alpha=.12)
                ax.set_title(title);ax.set_xlabel('Predicted action offset');ax.axvspan(1,5,color='gray',alpha=.08)
        axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(output/f'offset1_{end}_controller_units.png',dpi=180);plt.close(fig)
    dump_json(output/'completion_audit.json',{'complete':True,'variants':list(VARIANTS),'paired_keys_per_variant':6979,
        'prediction_records':27916,'source_hashes':hashes,'offsets':list(range(1,51)),'prefixes':list(PREFIXES),
        'bootstrap_replicates':2000,'bootstrap_unit':'task;overall_suite_stratified','new_environment_steps':0,
        'control_units_warning':'target_increment_not_actual_EEF;rotation_vector_not_final_pose_angle',
        'zero_gripper_handling':'separate_from_open_close','outputs':{p.name:sha256(p) for p in output.iterdir()
        if p.is_file() and p.name!='completion_audit.json'}})
    print(json.dumps({'complete':True,'analysis':str(output),'prediction_records':27916}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
