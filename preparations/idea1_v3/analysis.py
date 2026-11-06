"""Incremental v3 summaries, true EEF XYZ artifacts and fixed-key media index.

Reads only small completion/result/inference/telemetry records in explicitly
named new attempts and their declared parents. No checkpoint/raw/video rehash.
"""
import argparse
import collections
import csv
import html
import json
import os
from pathlib import Path
import subprocess

import numpy as np

from preparations.idea1_v3.advance import completed
from preparations.idea1_v3.io import atomic


def checkpoint_identity(row):
    identity=row.get('checkpoint',{})
    return identity.get('base_sha256',identity.get('sha256',identity.get('path')))


def phase_label(state):
    phase=state.get('semantic_phase','unknown')
    contacts=any(value is True for value in state.get('robot_contacts',{}).values())
    grasps=any(value is True for value in state.get('robot_grasps',{}).values())
    distances=any(value is not None and np.isfinite(value) for value in state.get('eef_object_distances',{}).values())
    if phase=='initial' and not (contacts or grasps or distances): return 'unknown'
    if phase=='grasp_or_manipulate' and not (contacts or grasps): return 'unknown'
    if phase=='release_or_final_action' and state.get('min_object_target_distance') is None: return 'unknown'
    return phase


def write_csv(path,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def collect(roots):
    seen=set(); rows={}; attempts=[]
    def visit(root):
        root=Path(root).resolve()
        if root in seen: return
        seen.add(root); manifest=json.loads((root/'manifest.json').read_text()); attempts.append((root,manifest))
        for parent in manifest.get('source_attempts',[]): visit(parent)
        for result in completed(root):
            key=result['key']
            if key in rows:
                raise ValueError(f'Duplicate valid scientific key across explicit attempts: {key}')
            rows[key]=dict(result,source_attempt=str(root),analysis_stage=manifest['stage'])
    for root in roots: visit(root)
    return list(rows.values()),attempts


def historical_trajectories(index,rows,output):
    """Use only the eight preselected Full-D2 telemetry paths, never old raw scans."""
    index=Path(index).resolve()
    targets={(row['experiment'],row['suite'],row['task'],row['init']) for row in rows
        if row['experiment'] in ('E1','E4') and row['init']==6 and row['task'] in (0,5)}
    references=[]; controls=[]
    with index.open() as stream:
        for old in csv.DictReader(stream):
            if old['condition']!='pi0_full_u50' or int(old['init'])!=6: continue
            controls.append(old)
            if int(old['task']) not in (0,5): continue
            telemetry=(index.parent/old['telemetry']).resolve()
            if not telemetry.is_file(): raise FileNotFoundError(telemetry)
            for experiment,suite,task,init in sorted(targets):
                if (suite,task,init)!=(old['suite'],int(old['task']),int(old['init'])): continue
                references.append(dict(experiment=experiment,condition='historical_pi0_Full_D2_v2.2',
                    suite=suite,task=task,init=init,success=old['success'].lower()=='true',
                    directory=str(telemetry.parent),telemetry=str(telemetry),analysis_stage='historical_trajectory_reference',
                    historical_reference=True,config=dict(experiment=experiment),
                    source_index=str(index),source_condition=old['condition'],source_receipt=old.get('source_receipt'),
                    video=str((index.parent/old['video']).resolve()) if old.get('video') else None,
                    comparability='historical v2.2 Full-D2, not Full-D10; execution revision/timing differ; visual reference only'))
    write_csv(output/'historical_trajectory_references.csv',references)
    summaries=[]
    for suite in ['overall',*sorted({row['suite'] for row in controls})]:
        subset=controls if suite=='overall' else [row for row in controls if row['suite']==suite]
        summaries.append(dict(control='historical pi0 Full-D2',suite=suite,
            successes=sum(row['success'].lower()=='true' for row in subset),episodes=len(subset),
            source_index=str(index),filter='condition=pi0_full_u50, init=6, all listed tasks',
            matching_level='same suite/task/init; revision/limits/timing not fully matched; historical reference only',
            paired_CI=False,causal_latency_ratio=False))
    write_csv(output/'historical_controls.csv',summaries)
    historical_quality_overlap(index,rows,output)
    return references


def historical_quality_overlap(index,rows,output):
    """Descriptive same-key E4/history overlap; never a causal paired analysis."""
    index=Path(index).resolve(); old_by_key={}
    with index.open() as stream:
        for old in csv.DictReader(stream):
            if old['condition']!='pi0_full_u50' or int(old['init']) not in (6,7,8): continue
            key=(old['suite'],int(old['task']),int(old['init']))
            if key in old_by_key: raise ValueError(f'Duplicate historical suite/task/init: {key}')
            old_by_key[key]=old
    summaries=[]; details=[]
    conditions=sorted({row['condition'] for row in rows if row['experiment']=='E4' and row['analysis_stage']!='P0'})
    for condition in conditions:
        new=[row for row in rows if row['experiment']=='E4' and row['condition']==condition and row['analysis_stage']!='P0']
        new_by_key={(row['suite'],row['task'],row['init']):row for row in new}
        if len(new_by_key)!=len(new): raise ValueError('Duplicate new E4 suite/task/init in overlap analysis')
        common=sorted(set(new_by_key)&set(old_by_key))
        for key in common:
            old,new_row=old_by_key[key],new_by_key[key]
            details.append(dict(new_condition=condition,suite=key[0],task=key[1],init=key[2],
                historical_success=old['success'].lower()=='true',new_success=bool(new_row['success']),
                historical_episode_key=old.get('episode_key'),new_episode_key=new_row['key'],
                historical_index=str(index),new_source_attempt=new_row['source_attempt']))
        for suite in ['overall',*sorted({key[0] for key in common})]:
            keys=common if suite=='overall' else [key for key in common if key[0]==suite]
            pairs=[(old_by_key[key]['success'].lower()=='true',bool(new_by_key[key]['success'])) for key in keys]
            summaries.append(dict(new_condition=condition,historical_condition='pi0_full_u50 (Full-D2)',suite=suite,
                common_episodes=len(keys),historical_successes=sum(old for old,new_result in pairs),
                new_successes=sum(new_result for old,new_result in pairs),
                both_success=sum(old and new_result for old,new_result in pairs),
                historical_success_new_failure=sum(old and not new_result for old,new_result in pairs),
                historical_failure_new_success=sum(not old and new_result for old,new_result in pairs),
                both_failure=sum(not old and not new_result for old,new_result in pairs),
                new_full_successes=sum(row['success'] for row in new) if suite=='overall' else None,
                new_full_episodes=len(new) if suite=='overall' else None,
                source_index=str(index),historical_filter='condition=pi0_full_u50, init in [6,7,8]',
                overlap_rule='intersection by suite/task/init with new E4 valid episodes',
                interpretation='历史参照；混合执行revision，历史Full-D2不是Full-D10；四格仅描述共同key结果，不作因果解释',
                formal_causal_CI=False,causal_speed_ratio=False))
    write_csv(output/'historical_quality_overlap.csv',summaries)
    write_csv(output/'historical_quality_overlap_keys.csv',details)
    lines=['# E4 与历史 Full-D2 的共同样本质量参照','',
        '仅按 suite/task/init 取交集。历史条件为 v2.2 pi0_full_u50（Full-D2，不是 Full-D10），与新 E4 存在混合执行 revision/协议差异。四格计数仅描述结果，不提供正式因果置信区间或速度比；不并入新实验主表。','',
        '| 新条件 | suite | 共同分母 | 历史成功 | 新条件成功 | 双成功 | 历史成功/新失败 | 历史失败/新成功 | 双失败 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    lines.extend(f"| {row['new_condition']} | {row['suite']} | {row['common_episodes']} | {row['historical_successes']} | {row['new_successes']} | {row['both_success']} | {row['historical_success_new_failure']} | {row['historical_failure_new_success']} | {row['both_failure']} |" for row in summaries)
    lines.extend(['','[汇总及来源过滤条件](historical_quality_overlap.csv) · [共同key逐条四格依据](historical_quality_overlap_keys.csv)'])
    (output/'historical_quality_overlap.md').write_text('\n'.join(lines)+'\n')
    return summaries


class VerificationDetails:
    """Future-only samples; keep per-episode sufficient statistics for equal weights."""
    def __init__(self):
        self.error=collections.defaultdict(dict)
        self.gripper=collections.defaultdict(dict)

    def add(self,stage,condition,suite,phase,episode_key,route):
        base=(stage,condition,suite,phase)
        if route.get('signed_error') is not None:
            error=np.asarray(route['signed_error'],float)
            distance=np.asarray(route.get('distance'),float)
            if error.ndim!=3 or error.shape[0]!=2 or not 1<=error.shape[1]<=5 or error.shape[2]!=6 or distance.shape!=error.shape[:2]:
                raise ValueError('Expected future-only K2 x 1..5 x 6 signed error and matching distances')
            if not np.isfinite(error).all() or not np.isfinite(distance).all(): raise ValueError('Nonfinite verifier detail')
            if not np.allclose(distance,np.linalg.norm(error,axis=2)/np.sqrt(6),atol=1e-6,rtol=1e-6):
                raise ValueError('Saved distance does not match normalized six-axis signed error')
            for branch in range(2):
                for future in range(error.shape[1]):
                    group=self.error[(*base,future+1,branch+1)]
                    value=group.setdefault(episode_key,dict(n=0,total=np.zeros(6),square=np.zeros(6),distance=0.))
                    vector=error[branch,future]
                    value['n']+=1; value['total']+=vector; value['square']+=vector**2
                    value['distance']+=float(distance[branch,future])
        flags=('verify_gripper','post_verify_gripper')
        reason=route.get('full_reason')
        if any(name in route for name in flags) or reason in ('gripper','gripper_window'):
            value=self.gripper[base].setdefault(episode_key,collections.Counter())
            value['recorded_requests']+=1
            for name in flags:
                if name not in route: continue
                if not isinstance(route[name],bool): raise ValueError('Gripper check event must be a recorded boolean')
                value[name+'_checks']+=1; value[name+'_events']+=int(route[name])
            value['immediate_gripper_full_calls']+=reason=='gripper'
            value['scheduled_gripper_window_full_calls']+=reason=='gripper_window'

    @staticmethod
    def metrics(value):
        n=value['n']
        return dict(**{f'axis{i}_mean_signed':float(value['total'][i]/n) for i in range(6)},
            translation_rms=float(np.sqrt(value['square'][:3].sum()/(3*n))),
            rotation_rms=float(np.sqrt(value['square'][3:].sum()/(3*n))),distance_mean=value['distance']/n)

    def write(self,output):
        pooled=[]; equal=[]; per_episode=[]
        for key,episodes in sorted(self.error.items()):
            base=dict(zip(('stage','condition','suite','phase','future_index','verify_branch'),key))
            base.update(noise_t=(.10,.05)[key[-1]-1],observations=sum(v['n'] for v in episodes.values()),
                independent_episodes=len(episodes),history_prefix_included=False,
                phase_anchor='current replanning telemetry at inference.step; E3 verifier state can be earlier',
                units='model normalized coordinates; not mm or real EEF error')
            total=dict(n=base['observations'],total=sum((v['total'] for v in episodes.values()),np.zeros(6)),
                square=sum((v['square'] for v in episodes.values()),np.zeros(6)),distance=sum(v['distance'] for v in episodes.values()))
            pooled.append(dict(base,aggregation='observation_weighted',**self.metrics(total)))
            values=[]
            for episode_key,value in sorted(episodes.items()):
                metrics=self.metrics(value); values.append(metrics)
                per_episode.append(dict(base,episode_key=episode_key,observations=value['n'],
                    independent_episodes=1,aggregation='within_episode',**metrics))
            # Average each episode's signed means, RMS and distance mean equally.
            # In particular mean(episode RMS), not pooled RMS under a new label.
            equal.append(dict(base,aggregation='episode_equal_mean_of_episode_metrics',
                **{name:float(np.mean([value[name] for value in values])) for name in values[0]}))
        write_csv(output/'future_error_by_phase_step_branch.csv',pooled)
        write_csv(output/'future_error_by_phase_step_branch_episode_equal.csv',equal)
        write_csv(output/'future_error_by_phase_step_branch_per_episode.csv',per_episode)
        gripper=[]; gripper_equal=[]
        for key,episodes in sorted(self.gripper.items()):
            base=dict(zip(('stage','condition','suite','phase'),key))
            base.update(recorded_requests=sum(v['recorded_requests'] for v in episodes.values()),
                independent_episodes=len(episodes),event_definition='recorded boolean checks plus explicitly labeled gripper Full routes; not per-action gripper disagreement',
                per_future_gripper_disagreement='unavailable: raw per-step gripper comparisons were not saved')
            totals=sum(episodes.values(),collections.Counter())
            common=dict(base,**{name:totals[name] for name in ('immediate_gripper_full_calls','scheduled_gripper_window_full_calls')})
            rates={}; equal_rates={}
            for name in ('verify_gripper','post_verify_gripper'):
                checks=totals[name+'_checks']; event=totals[name+'_events']
                checked=[value for value in episodes.values() if value[name+'_checks']]
                common.update({name+'_checks':checks,name+'_events':event,name+'_check_episodes':len(checked)})
                rates[name+'_event_rate']=event/checks if checks else None
                equal_rates[name+'_event_rate']=float(np.mean([v[name+'_events']/v[name+'_checks'] for v in checked])) if checked else None
            gripper.append(dict(common,aggregation='check_observation_weighted',**rates))
            gripper_equal.append(dict(common,aggregation='episode_equal_mean_of_checked_episode_rates',**equal_rates))
        write_csv(output/'gripper_check_events_by_phase.csv',gripper)
        write_csv(output/'gripper_check_events_by_phase_episode_equal.csv',gripper_equal)


def summarize(rows,output):
    grouped=collections.defaultdict(list)
    for row in rows:
        # Keep calibration and P0 entirely separate from scientific init6–10.
        grouped[(row['analysis_stage'],row['condition'],row.get('execution','A'),row['suite'])].append(row)
    scores=[]; costs=[]; routing=[]; errors=[]; phases=[]; trajectories=[]; request_rows=[]; draft_artifacts={}
    phase_route=collections.defaultdict(lambda:dict(episodes=set(),requests=0,full=0,verified=0,rejected=0,contact_unknown=0,grasp_unknown=0))
    verifier_details=VerificationDetails()
    for (stage,condition,execution,suite),episodes in sorted(grouped.items()):
        scores.append(dict(stage=stage,condition=condition,execution=execution,suite=suite,
            D='N/A' if episodes[0]['experiment']=='E4' else episodes[0]['config'].get('draft_steps',10),
            successes=sum(e['success'] for e in episodes),episodes=len(episodes),
            success_rate=float(np.mean([e['success'] for e in episodes])),
            tasks=len({e['task'] for e in episodes}),inits=','.join(map(str,sorted({e['init'] for e in episodes})))))
        policy_ms=rpc_ms=actions=0; requests=[]; full_reasons=collections.Counter(); accepts=collections.Counter()
        full=verify=draft=0; ages=[]; axis_errors=[]
        for episode in episodes:
            directory=Path(episode['directory']); actions+=episode['actions']
            inferred=[json.loads(line) for line in (directory/'infer.jsonl').read_text().splitlines()]
            if sum(i['executed_length'] for i in inferred)!=episode['actions']: raise ValueError('Executed action count mismatch')
            previous_counts=dict(full=0,verify=0,draft=0)
            for inference in inferred:
                timing=inference['timing']; route=inference['route']; ms=timing['policy_wall_ms']
                delta={name:inference['counts'].get(name,0)-previous_counts.get(name,0) for name in ('full','verify','draft')}
                if any(value<0 for value in delta.values()): raise ValueError('Cumulative counters moved backwards within episode')
                previous_counts=inference['counts']
                if episode['experiment']=='E4':
                    for event in inference.get('startup_events',[]):
                        if event.get('kind')=='learned_draft_weight_load':
                            metadata=event['meta']
                            draft_artifacts[(suite,event['path'])]=dict(suite=suite,path=event['path'],
                                native_H=metadata.get('chunk_m'),out_dim=metadata.get('out_dim'),
                                use_last_actions=metadata.get('use_last_actions'),
                                declared_history_len=metadata.get('draft_history_len'),
                                runtime_history_consumed=False,actual_Rmax=5,
                                metadata=json.dumps(metadata,sort_keys=True))
                request_rows.append(dict(stage=stage,condition=condition,suite=suite,key=episode['key'],
                    replan=inference['replan'],route=route['path'],full_reason=route.get('full_reason'),
                    setup_or_first_request=bool(inference.get('startup_events') or inference['replan']==0),
                    policy_wall_ms=ms,rpc_roundtrip_ms=inference.get('rpc_roundtrip_ms'),
                    executed_length=inference['executed_length'],
                    mask=episode['config'].get('mask','N/A'),
                    native_policy_calls=delta['full'] if stage=='E2' else 'N/A',
                    full_calls='N/A' if stage=='E2' else delta['full'],
                    full_fallback_calls='N/A' if stage=='E2' else int(route.get('full_reason') in ('zero_accept','gripper','gripper_window')),
                    verify_calls=delta['verify'],draft_calls=delta['draft']))
                policy_ms+=ms; requests.append(ms); rpc_ms+=inference.get('rpc_roundtrip_ms',0)
                if route.get('full_reason'): full_reasons[route['full_reason']]+=1
                if 'accepted_length' in route: accepts[int(route['accepted_length'])]+=1
                if inference.get('full_cache_age'): ages.append(inference['full_cache_age']['env_steps'])
                if route.get('signed_error') is not None: axis_errors.extend(np.asarray(route['signed_error']).reshape(-1,6).tolist())
            if inferred:
                count=inferred[-1]['counts']; full+=count.get('full',0); verify+=count.get('verify',0); draft+=count.get('draft',0)
                if episode['experiment']=='E4' and (count.get('full',0) or count.get('verify',0)):
                    raise ValueError('E4 purity violation: AE/Full calls present')
            states=[json.loads(line) for line in (directory/'telemetry.jsonl').read_text().splitlines()]
            for inference in inferred:
                phase=phase_label(states[inference['step']])
                verifier_details.add(stage,condition,suite,phase,episode['key'],inference['route'])
                aggregate=phase_route[(stage,condition,suite,phase)]
                aggregate['episodes'].add(episode['key']); aggregate['requests']+=1
                aggregate['full']+=bool(inference['route'].get('full_reason'))
                aggregate['verified']+='accepted_length' in inference['route']
                aggregate['rejected']+=inference['route'].get('accepted_length')==0
                current_state=states[inference['step']]
                aggregate['contact_unknown']+=not any(value is True for value in current_state.get('robot_contacts',{}).values())
                aggregate['grasp_unknown']+=not any(value is True for value in current_state.get('robot_grasps',{}).values())
            xyz=np.asarray([state['eef_pos'] for state in states],float)
            if xyz.shape!=(episode['actions']+1,3) or not np.isfinite(xyz).all(): raise ValueError('True EEF XYZ dimensions/values invalid')
            order=['initial','approach','contact','grasp_or_manipulate','transport','goal_alignment','release_or_final_action','success']
            known=[phase_label(s) for s in states if phase_label(s) in order]
            phases.append(dict(stage=stage,condition=condition,suite=suite,task=episode['task'],init=episode['init'],
                success=episode['success'],furthest_phase=max(known,key=order.index) if known else 'unknown',
                final_phase=phase_label(states[-1]),
                unknown_steps=sum(phase_label(s)=='unknown' for s in states),
                telemetry_observations=len(states),contact_grasp_availability='positive observations supported; negative observations availability_unknown',
                contact_availability_unknown_steps=sum(not any(v is True for v in s.get('robot_contacts',{}).values()) for s in states),
                grasp_availability_unknown_steps=sum(not any(v is True for v in s.get('robot_grasps',{}).values()) for s in states),
                contact_grasp_negative_availability='legacy snapshot catches exceptions as False; false vs unavailable not disambiguated',
                interpretation='furthest observed stage; not a causal failure stage'))
            trajectories.append(dict(stage=stage,condition=condition,suite=suite,task=episode['task'],init=episode['init'],
                success=episode['success'],points=len(xyz),directory=str(directory),telemetry=str(directory/'telemetry.jsonl')))
        costs.append(dict(stage=stage,condition=condition,execution=execution,suite=suite,episodes=len(episodes),
            D='N/A' if episodes[0]['experiment']=='E4' else episodes[0]['config'].get('draft_steps',10),
            actual_actions=actions,requests=len(requests),policy_ms=policy_ms,policy_ms_per_actual_action=policy_ms/actions,
            request_p50_ms=float(np.percentile(requests,50)),request_p95_ms=float(np.percentile(requests,95)),
            rpc_roundtrip_ms=rpc_ms,episode_wall_s=sum(e['wall_s'] for e in episodes),
            environment_steps_per_s=actions/sum(e['wall_s'] for e in episodes),
            replans_per_s=len(requests)/sum(e['wall_s'] for e in episodes)))
        routing.append(dict(stage=stage,condition=condition,suite=suite,requests=len(requests),actual_actions=actions,
            mask=episodes[0]['config'].get('mask','N/A'),
            native_policy_calls=full if stage=='E2' else 'N/A',
            full_calls='N/A' if stage=='E2' else full,
            full_fallback_calls='N/A' if stage=='E2' else sum(full_reasons[reason] for reason in ('zero_accept','gripper','gripper_window')),
            verify_calls=verify,draft_calls=draft,full_per_request='N/A' if stage=='E2' else full/len(requests),
            full_per_action='N/A' if stage=='E2' else full/actions,full_reason_counts=json.dumps(dict(full_reasons)),
            accepted_length_counts=json.dumps(dict(accepts)),mean_kv_age_env_steps=float(np.mean(ages)) if ages else None))
        routing[-1]['counter_semantics']='E2累计counts.full表示该mask下原生策略调用，不是未剪Full或回退；Full回退统计不适用' if stage=='E2' else 'Full18-D10 path calls; E4 full/verify must be zero'
        if stage=='E2':
            implementation='native nn.scan' if episodes[0]['config']['mask']=='Full' else '18-layer Python-unrolled JAX trace, all AE layers retained'
            costs[-1]['ae_compilation_structure']=implementation
            costs[-1]['latency_attribution']='VLM mask plus changed AE compilation structure; not isolated VLM pruning speedup'
        if axis_errors:
            error=np.asarray(axis_errors)
            errors.append(dict(stage=stage,condition=condition,suite=suite,future_only_vectors=len(error),
                **{f'axis{i}_mean_signed':float(error[:,i].mean()) for i in range(6)},
                translation_rms=float(np.sqrt(np.mean(error[:,:3]**2))),rotation_rms=float(np.sqrt(np.mean(error[:,3:]**2))),
                units='model normalized coordinates; E3 previous-state anchor; not real EEF error'))
    write_csv(output/'success_by_condition_suite.csv',scores)
    write_csv(output/'latency_and_throughput.csv',costs)
    write_csv(output/'routing.csv',routing); write_csv(output/'future_reconstruction_error.csv',errors)
    write_csv(output/'episode_phases.csv',phases); write_csv(output/'trajectory_index.csv',trajectories)
    task_scores=[]
    for stage,condition,suite,task in sorted({(e['analysis_stage'],e['condition'],e['suite'],e['task']) for e in rows}):
        subset=[e for e in rows if (e['analysis_stage'],e['condition'],e['suite'],e['task'])==(stage,condition,suite,task)]
        task_scores.append(dict(stage=stage,condition=condition,suite=suite,task=task,successes=sum(e['success'] for e in subset),episodes=len(subset)))
    write_csv(output/'success_by_task.csv',task_scores)
    phase_rows=[]
    for (stage,condition,suite,phase),value in sorted(phase_route.items()):
        phase_rows.append(dict(stage=stage,condition=condition,suite=suite,phase=phase,
            episodes=len(value['episodes']),requests=value['requests'],
            native_policy_requests=value['requests'] if stage=='E2' else 'N/A',
            full_requests='N/A' if stage=='E2' else value['full'],
            verified_requests=value['verified'],zero_accept=value['rejected'],
            contact_availability_unknown_requests=value['contact_unknown'],grasp_availability_unknown_requests=value['grasp_unknown'],
            full_rate='N/A' if stage=='E2' else value['full']/value['requests'],rejection_rate=value['rejected']/value['verified'] if value['verified'] else None))
    write_csv(output/'phase_routing.csv',phase_rows)
    verifier_details.write(output)
    write_csv(output/'request_records.csv',request_rows)
    grouped_requests=collections.defaultdict(list)
    for row in request_rows: grouped_requests[(row['stage'],row['condition'],row['route'],row['setup_or_first_request'])].append(row)
    request_summary=[]
    for (stage,condition,route,setup),requests in sorted(grouped_requests.items()):
        values=[row['policy_wall_ms'] for row in requests]
        request_summary.append(dict(stage=stage,condition=condition,route=route,setup_or_first_request=setup,
            requests=len(requests),mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),
            mask=requests[0]['mask'],
            native_policy_calls=sum(row['native_policy_calls'] for row in requests) if stage=='E2' else 'N/A',
            full_calls='N/A' if stage=='E2' else sum(row['full_calls'] for row in requests),
            full_fallback_calls='N/A' if stage=='E2' else sum(row['full_fallback_calls'] for row in requests),
            verify_calls=sum(row['verify_calls'] for row in requests),
            draft_calls=sum(row['draft_calls'] for row in requests)))
    write_csv(output/'latency_by_route_setup.csv',request_summary)
    failure_counts=collections.Counter((row['stage'],row['condition'],row['suite'],row['furthest_phase']) for row in phases if not row['success'])
    write_csv(output/'failure_furthest_phase_counts.csv',[dict(stage=key[0],condition=key[1],suite=key[2],
        furthest_phase=key[3],failures=count,causal_stage=False) for key,count in sorted(failure_counts.items())])
    overall=[]
    for stage,condition,execution in sorted({(row['stage'],row['condition'],row['execution']) for row in scores}):
        subset=[row for row in scores if (row['stage'],row['condition'],row['execution'])==(stage,condition,execution)]
        overall.append(dict(stage=stage,condition=condition,execution=execution,D=subset[0]['D'],
            successes=sum(row['successes'] for row in subset),episodes=sum(row['episodes'] for row in subset),
            suites=len(subset),tasks=sum(row['tasks'] for row in subset)))
    write_csv(output/'success_by_condition.csv',overall)
    write_csv(output/'released_draft_protocol.csv',list(draft_artifacts.values()))
    return scores


def paired_new_conditions(rows,output):
    """Only newly measured conditions with exact init vector and checkpoint match."""
    scientific=[row for row in rows if row['analysis_stage'] not in ('P0','calibration')]
    configs={row['condition']:row['config'] for row in scientific}
    pairs=[]
    for condition,config in configs.items():
        reference=None
        if config['experiment']=='E2' and config['mask']!='Full': reference='E2_Full'
        elif config['experiment']=='E1':
            reference=f"E1_D{config['draft_steps']}_tau0.15_P2"
            if reference==condition: reference=None
        elif config['experiment']=='E3':
            from preparations.idea1_v3.cli import config_id
            reference=config_id(config['parent'])
        if reference in configs: pairs.append((reference,condition))
    result=[]; rng=np.random.default_rng(1701)
    for reference,variant in pairs:
        left={(r['suite'],r['task'],r['init']):r for r in scientific if r['condition']==reference}
        right={(r['suite'],r['task'],r['init']):r for r in scientific if r['condition']==variant}
        common=sorted(set(left)&set(right)); matched=[]
        for key in common:
            a,b=left[key],right[key]
            if a.get('init_vector_hash') and a['init_vector_hash']==b.get('init_vector_hash') and checkpoint_identity(a)==checkpoint_identity(b): matched.append(key)
        if not matched: continue
        diff={key:int(right[key]['success'])-int(left[key]['success']) for key in matched}
        suites=sorted({key[0] for key in matched}); replicates=[]
        for _ in range(1000):
            suite_means=[]
            for suite in suites:
                tasks=sorted({key[1] for key in matched if key[0]==suite})
                sampled=rng.choice(tasks,size=len(tasks),replace=True)
                suite_means.append(np.mean([diff[key] for task in sampled for key in matched if key[0]==suite and key[1]==task]))
            replicates.append(np.mean(suite_means))
        result.append(dict(reference=reference,variant=variant,common_exact_matched=len(matched),
            success_difference=float(np.mean(list(diff.values()))),ci_low=float(np.percentile(replicates,2.5)),
            ci_high=float(np.percentile(replicates,97.5)),recovered=sum(v==1 for v in diff.values()),new_failures=sum(v==-1 for v in diff.values()),
            bootstrap='1000 suite-stratified task-cluster draws; exploratory selected conditions; partial coverage reported',
            historical=False))
    write_csv(output/'paired_new_conditions.csv',result)


def media(rows,output,check=False,historical=None):
    lines=['# Fixed-key video index','', 'Video keys were fixed before outcomes: task0/init6 and task5/init6 per suite/condition.','']
    links=[]
    for row in rows:
        if not row.get('video'): continue
        directory=Path(row['directory']); video=directory/row['video']; verified=directory/'media_checked.json'
        if not video.is_file():
            lines.append(f"- Missing video: {row['condition']} {row['suite']} task{row['task']} init{row['init']} — `{video}`")
            links.append(f'<li>Missing video: <code>{html.escape(str(video))}</code></li>')
            continue
        if check:
            stat=video.stat(); identity=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
            previous=json.loads(verified.read_text()) if verified.exists() else {}
            if previous.get('identity')!=identity:
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                    '-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(video)],text=True))
                if int(probe['streams'][0]['nb_read_frames'])!=row['actions']+1: raise ValueError('video frame count mismatch')
                subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
                atomic(verified,dict(identity=identity,full_decode_checked=True,probe=probe))
        relative=os.path.relpath(video,output); label=f"{row['condition']} {row['suite']} task{row['task']} init{row['init']} success={row['success']}"
        lines.append(f'- [{label}]({relative}) — `{video}`; decode checked={verified.exists()}')
        links.append(f'<li><a href="{html.escape(relative)}">{html.escape(label)}</a></li>')
    paired_html=[]
    if historical is not None:
        lines.extend(['','## 同key历史视频参照','',
            '固定 task0/5、init6；历史 pi0 Full-D2（v2.2 pi0_full_u50），不是 Full-D10。仅链接既有视频，不重新解码历史媒体。','',
            '| suite/task/init | 新E4视频与路径 | 历史Full-D2视频与路径 | 历史来源 |',
            '|---|---|---|---|'])
        old_by_key={(row['experiment'],row['suite'],int(row['task']),int(row['init'])):row for row in historical}
        def video_cell(path,label):
            if path is None: return '缺失：索引未提供视频路径','缺失：索引未提供视频路径'
            path=Path(path); escaped=html.escape(str(path))
            if not path.is_file(): return f'缺失：`{path}`',f'缺失：<code>{escaped}</code>'
            relative=os.path.relpath(path,output)
            return f'[{label}](<{relative}>)<br>`{path}`',f'<a href="{html.escape(relative)}">{html.escape(label)}</a><br><code>{escaped}</code>'
        fixed=sorted((row for row in rows if row['experiment']=='E4' and row['init']==6 and row['task'] in (0,5)),
            key=lambda row:(row['condition'],row['suite'],row['task']))
        for row in fixed:
            key=('E4',row['suite'],row['task'],row['init']); old=old_by_key.get(key)
            new_path=Path(row['directory'])/row['video'] if row.get('video') else None
            old_path=old.get('video') if old else None
            new_md,new_html=video_cell(new_path,'新E4'); old_md,old_html=video_cell(old_path,'历史Full-D2，非D10')
            label=f"{row['suite']}/task{row['task']:02d}/init{row['init']:02d}"
            source=old.get('source_index','未提供') if old else '缺失：无同key历史索引记录'
            lines.append(f'| {label} | {new_md} | {old_md} | `{source}` |')
            paired_html.append(f'<tr><td>{html.escape(label)}</td><td>{new_html}</td><td>{old_html}</td><td><code>{html.escape(str(source))}</code></td></tr>')
    (output/'media_index.md').write_text('\n'.join(lines)+'\n')
    table='<h2>同key历史参照：v2.2 pi0 Full-D2，非Full-D10</h2><p>仅链接既有视频，不重新解码历史媒体。</p><table><tr><th>suite/task/init</th><th>新E4</th><th>历史Full-D2</th><th>历史索引来源</th></tr>'+''.join(paired_html)+'</table>' if historical is not None else ''
    (output/'media_index.html').write_text('<!doctype html><meta charset="utf-8"><title>Fixed videos</title><ul>'+''.join(links)+'</ul>'+table)


def xyz_plots(rows,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fixed=[e for e in rows if e['init']==6 and e['task'] in (0,5) and e['analysis_stage'] not in ('P0','calibration')]
    by=collections.defaultdict(list)
    for row in fixed: by[(row['experiment'],row['suite'],row['task'],row['init'])].append(row)
    plots=[]
    for key,episodes in sorted(by.items()):
        episodes=sorted(episodes,key=lambda e:e['condition'])
        # E3 uses its explicit frozen E1 parent, never an arbitrary pi0 condition.
        if key[0]=='E3':
            from preparations.idea1_v3.cli import config_id
            parent=config_id(episodes[0]['config']['parent'])
            candidates=[row for row in fixed if row['experiment']=='E1' and row['condition']==parent
                and (row['suite'],row['task'],row['init'])==key[1:]
                and row.get('init_vector_hash')==episodes[0].get('init_vector_hash')
                and checkpoint_identity(row)==checkpoint_identity(episodes[0])]
            if candidates: episodes=[candidates[0],*episodes]
        single=len(episodes)==1
        reference=next((e for e in episodes if e.get('historical_reference')),
            next((e for e in episodes if e['condition']=='E2_Full'),episodes[0]))
        for variant in episodes:
            if variant is reference and not single: continue
            figure=plt.figure(figsize=(12,8)); axes=[figure.add_subplot(2,3,1,projection='3d')]
            axes.extend(figure.add_subplot(2,3,i) for i in range(2,7))
            data=[]
            displayed=((variant,'#e68023','--'),) if single else ((reference,'#2864b4',':'),(variant,'#e68023','--'))
            for display_index,(row,color,style) in enumerate(displayed):
                states=[json.loads(line) for line in (Path(row['directory'])/'telemetry.jsonl').read_text().splitlines()]
                xyz=np.asarray([state['eef_pos'] for state in states]); data.append(xyz)
                marker_interval=max(1,len(xyz)//25)
                axes[0].plot(*xyz.T,color=color,ls=style,alpha=.65,label=row['condition'],marker='.',markersize=3,markevery=marker_interval)
                axes[0].scatter(*xyz[0],color=color,marker='o'); axes[0].scatter(*xyz[-1],color=color,marker='x')
                for axis,(x,y) in zip(axes[1:4],((0,1),(0,2),(1,2))):
                    axis.plot(xyz[:,x],xyz[:,y],color=color,ls=style,alpha=.65,marker='.',markersize=3,markevery=marker_interval)
                    axis.scatter(xyz[0,x],xyz[0,y],color=color,marker='o'); axis.scatter(xyz[-1,x],xyz[-1,y],color=color,marker='x')
                    axis.set_xlabel('XYZ'[x]+' (m)'); axis.set_ylabel('XYZ'[y]+' (m)')
                axes[4].plot(np.arange(len(xyz))*.05,xyz[:,2],color=color,ls=style,alpha=.65,
                    marker='.',markersize=3,markevery=marker_interval)
                # Annotate only the first stable entrance to a phase. Original
                # repeated transitions remain in telemetry and phase tables.
                short={'initial':'I','approach':'A','contact':'C','grasp_or_manipulate':'G',
                    'transport':'T','goal_alignment':'L','release_or_final_action':'R','success':'S'}
                seen_phases=set(); index=0
                while index<len(states):
                    phase=phase_label(states[index]); end=index+1
                    while end<len(states) and phase_label(states[end])==phase: end+=1
                    if end-index>=3 and phase not in seen_phases and phase in short:
                        seen_phases.add(phase)
                        axes[4].axvline(index*.05,color='grey',ls=style,alpha=.22)
                        prefix='V' if single or display_index else 'B'
                        axes[4].text(index*.05,.96-display_index*.11,f'{prefix}:{short[phase]}',
                            transform=axes[4].get_xaxis_transform(),fontsize=6,va='top',ha='left',clip_on=True)
                    index=end
            if single:
                axes[5].text(.05,.5,'No comparable baseline trajectory available.\nSingle measured trajectory only.',transform=axes[5].transAxes)
            else:
                length=min(map(len,data)); distance=np.linalg.norm(data[0][:length]-data[1][:length],axis=1)
                axes[5].plot(np.arange(length)*.05,distance,color='black'); axes[5].set_ylabel('same-step XYZ distance (m)')
            axes[5].set_xlabel('common simulated time (s), no padding'); axes[4].set_xlabel('simulated time (s)'); axes[4].set_ylabel('true EEF Z (m)')
            axes[4].set_title('First stable heuristic phase (≥3 steps); B reference / V variant\nI initial · A approach · C contact · G grasp · T transport · L align · R release · S success',fontsize=6)
            axes[0].set_xlabel('X'); axes[0].set_ylabel('Y'); axes[0].set_zlabel('Z'); axes[0].legend(fontsize=6)
            title=f'{key}; fixed-key case, native termination'
            if reference.get('historical_reference'): title+='; historical v2.2 Full-D2 (not Full-D10), visual reference only'
            figure.suptitle(title,fontsize=9); figure.tight_layout()
            label='no_baseline' if single else reference['condition']
            name=f"xyz_{key[0]}_{key[1]}_task{key[2]}_{label}_vs_{variant['condition']}.png"
            figure.savefig(output/name,dpi=150); plt.close(figure); plots.append(name)
    (output/'xyz_index.md').write_text('# True EEF XYZ\n\nFixed video keys only; circles=start, crosses=end. Blue=reference, orange=variant. Grey boundaries are first stable heuristic phases; negative contact/grasp availability remains unknown and is not fine-grained causal evidence. Same-time distance combines path and speed differences.\n\n'+
        ('\n'.join(f'![{name}]({name})' for name in plots) if plots else 'No within-experiment fixed-key comparison pair available. No historical trajectory has been invented.')+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--attempt',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--check-media',action='store_true')
    parser.add_argument('--historical-trajectory-index',type=Path)
    parser.add_argument('--xyz',action='store_true'); args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    rows,attempts=collect(args.attempt); scores=summarize(rows,args.output)
    paired_new_conditions(rows,args.output)
    historical=historical_trajectories(args.historical_trajectory_index,rows,args.output) if args.historical_trajectory_index else []
    media(rows,args.output,args.check_media,historical=historical if args.historical_trajectory_index else None)
    if args.xyz: xyz_plots(rows+historical,args.output)
    startup=[]
    for root,manifest in attempts:
        for journal in (root/'workers').glob('*/events.jsonl'):
            for line in journal.read_text().splitlines():
                event=json.loads(line)
                if event.get('kind')=='startup': startup.append(dict(attempt=str(root),**event))
    write_csv(args.output/'configuration_startup.csv',startup)
    status=[]
    for root,manifest in attempts:
        done=[e for e in rows if e['source_attempt']==str(root)]
        status.append(dict(attempt=str(root),stage=manifest['stage'],planned=len(manifest['tasks']),valid=len(done),
            status='complete' if len(done)==len(manifest['tasks']) else 'running_or_stopped'))
    atomic(args.output/'coverage.json',status)
    lines=['# idea1 v3 incremental results','',f'Valid episodes included: {len(rows)}. Technical invalid attempts are retained in worker journals.',
        'Calibration, P0 and scientific init6–10 rows are separate in the stage column. Failed tasks remain in denominators.',
        'Single-request mixed-policy latency and end-to-end throughput are separate columns. Startup is configuration_startup.csv; lazy first-use is included in episode policy wall.',
        'E2 scientific closed-loop records contain native total latency only, without stored VLM/AE partitions; independent stage benchmarks are available separately. CPU/GPU/disk-wait activity is not sampled.',
        'E2 的累计 counts.full 表示当前 mask 下的原生策略调用次数，不表示未剪 Full 或 Full 回退。E2 表中单列 native_policy_calls 与 mask，full_fallback_calls / Full 回退率记为 N/A。',
        'E2 性能边界：科学闭环中 Full 的 AE 使用原生 nn.scan；非空 mask 的 AE 虽保留完整18层，但由 Python 展开后交给 JAX 编译。该混杂保留在科学结果中，端到端延迟差不能全部归于 VLM 跳层。独立 v2-native-scanned-suffix 诊断的 suffix 使用原生 scan，并具有独立编译与同步边界；分项不替代科学 native 总量，以实际 stage_revision 为准，不改变已跑科学样本。',
        'E4 released converted draft has no history input; native draft.py explicitly discards last_actions. AE verification and Full call counts must both stay zero.',
        'Legacy contact/grasp helper returns False on exceptions; negative truth availability is not independently established. Conservative analysis maps unsupported phase claims to unknown; original telemetry is retained.',
        'History controls are references only until checkpoint/preprocessing/seed/initial-state/timing matches are established; no historical speed ratio or paired interval is fabricated.',
        '', '| stage | condition | suite | success / valid |','|---|---|---|---|']
    lines.extend(f"| {s['stage']} | {s['condition']} | {s['suite']} | {s['successes']} / {s['episodes']} |" for s in scores)
    lines.extend(['','[Video index](media_index.md) · [Playable links](media_index.html) · [Trajectory index](trajectory_index.csv)',
        '[Latency and throughput](latency_and_throughput.csv) · [Routing](routing.csv) · [Future-only reconstruction error](future_reconstruction_error.csv)',
        '[按阶段/未来1–5步/K分支误差](future_error_by_phase_step_branch.csv) · [episode等权误差](future_error_by_phase_step_branch_episode_equal.csv) · [逐episode依据](future_error_by_phase_step_branch_per_episode.csv)',
        '[分阶段夹爪检查事件](gripper_check_events_by_phase.csv) · [episode等权事件率](gripper_check_events_by_phase_episode_equal.csv)',
        '误差仅含已有future窗口，不含E3历史前缀；单位是模型归一化坐标，不是毫米或真实末端误差。episode等权误差为各episode内有符号均值/RMS/距离均值再等权平均；无该组观测的episode不补零。夹爪仅为保存的布尔检查事件及明确Full路由，未保存的逐步夹爪不一致率不可得。',
        'Phase table records furthest observed phase, not a proven cause of failure. Missing task telemetry remains unknown.'])
    if historical:
        controls=list(csv.DictReader((args.output/'historical_controls.csv').open()))
        lines.extend(['','Historical Full-D2 reference (separate from new results; no paired CI or causal latency ratio):',
            '| historical control | suite | success / listed init6 tasks |','|---|---|---|'])
        lines.extend(f"| {row['control']} | {row['suite']} | {row['successes']} / {row['episodes']} |" for row in controls)
        lines.extend(['','Source/filter/matching limits are in [historical_controls.csv](historical_controls.csv). Full-D2 is not Full-D10.'])
        lines.append('[E4 与历史 init6–8 共同样本质量及四格计数](historical_quality_overlap.md)：仅历史描述参照，不作正式因果CI或速度比。')
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(valid=len(rows),attempts=len(attempts),output=str(args.output))))

if __name__=='__main__': main()
