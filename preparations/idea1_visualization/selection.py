"""Trace-only phase inference and deterministic 12+8 request selection."""
from __future__ import annotations

from bisect import bisect_left
import copy
from pathlib import Path

import numpy as np

from .common import atomic_json, read_jsonl, write_jsonl

RULES = dict(version='trace-phase-selection-v2', approach_distance_m=.15,
             alignment_distance_m=.10, approach_progress_m=.02,
             transported_distance_m=.03, moving_distance_m=.001,
             direction_cosine=.5, window_control_steps=5,
             release_grasp_lookback_steps=10, event_debounce_steps=5,
             uniform_points=12, event_points=8, total_points=20)


def finite_values(mapping):
    return [float(x) for x in mapping.values() if x is not None and np.isfinite(x)]


def observed_any(mapping):
    values = list(mapping.values())
    if any(x is True for x in values):
        return True
    return False if values and all(x is not None for x in values) else None


def motion(row, previous, names=None):
    follow = False
    eef = np.asarray(row.get('eef_pos') or [],float)
    old_eef = np.asarray(previous.get('eef_pos') or [],float)
    if eef.shape!=(3,) or old_eef.shape!=(3,):
        return None
    de = eef-old_eef
    for name in (row.get('primary_objects',[]) if names is None else names):
        p = row.get('objects',{}).get(name,{}).get('pos')
        q = previous.get('objects',{}).get(name,{}).get('pos')
        if p is None or q is None:
            continue
        dp = np.asarray(p)-np.asarray(q)
        a,b = np.linalg.norm(dp),np.linalg.norm(de)
        if a>.001 and b>.001 and np.dot(dp,de)/(a*b)>.7:
            follow=True
    return follow


def annotate(rows):
    """Labels describe observed state at the request, with evidence, not GT."""
    rows=copy.deepcopy(rows)
    recent_grasp=-1000
    previous_command=None
    for index,row in enumerate(rows):
        prior=rows[max(0,index-RULES['window_control_steps'])]
        contact=observed_any(row.get('robot_contacts',{}))
        grasp=observed_any(row.get('robot_grasps',{}))
        if grasp:
            recent_grasp=row['step']
        action=row.get('executed_action',row.get('action'))
        command=None if action is None else float(action[6])
        held={name for name,value in row.get('robot_grasps',{}).items() if value is True}
        touched={name for name,value in row.get('robot_contacts',{}).items() if value is True}
        unfinished={p['predicate'][1] for p in row.get('goal_predicates',[])
                    if not p.get('satisfied') and len(p.get('predicate',[]))>=2}
        focus=held or touched or unfinished or set(row.get('primary_objects',[]))
        distances=finite_values({k:v for k,v in row.get('eef_object_distances',{}).items() if k in focus})
        old_distances=finite_values({k:v for k,v in prior.get('eef_object_distances',{}).items() if k in focus})
        targets=finite_values({k:v for k,v in row.get('object_target_distances',{}).items()
                              if len(k.split('->'))==2 and k.split('->')[0]!=k.split('->')[1]
                              and k.split('->')[0] in focus})
        displacement=finite_values({k:v for k,v in row.get('object_displacements',{}).items() if k in focus})
        follows=motion(row,prior,focus)
        sites={name for name in focus if row.get('objects',{}).get(name,{}).get('type')=='site'}
        site_follows=bool(sites) and motion(row,prior,sites)
        evidence=dict(contact=contact,grasp=grasp,object_follows_eef=follows,
                      active_objects=sorted(focus),goal_region_follows_eef=site_follows,
                      min_eef_object_distance=min(distances) if distances else None,
                      min_object_target_distance=min(targets) if targets else None,
                      goal_progress=row.get('goal_progress'),gripper_command=command)
        released=(previous_command is not None and previous_command>.2 and command is not None
                  and command<-.2 and row['step']-recent_grasp<=10)
        progress=(row.get('goal_progress',0)>prior.get('goal_progress',0))
        if row.get('success'):
            phase,reason='success','all observed goal predicates satisfied'
        elif released and ((targets and min(targets)<=.12) or progress):
            phase,reason='release_or_final_action','release command after recent grasp, near target or goal progress'
        elif progress or (grasp and targets and min(targets)<=.10):
            phase,reason='goal_alignment','recent goal predicate progress or currently grasped object near target'
        elif grasp and follows and displacement and max(displacement)>=.03:
            phase,reason='transport','current positive grasp, object follows EEF and moved >=3cm'
        elif grasp or (contact and command is not None and command>.2):
            phase,reason='grasp_or_manipulate','positive grasp or positive contact with closing command'
        elif site_follows and command is not None and command>.2:
            phase,reason='grasp_or_manipulate','goal region co-moves with EEF while closing; articulated manipulation inferred, grasp/contact unknown'
        elif contact:
            phase,reason='contact','positive observed robot-object contact'
        elif distances and (min(distances)<=.15 or
                            (old_distances and min(old_distances)-min(distances)>=.02)):
            phase,reason='approach','near a primary object or distance decreased in recent window'
        elif distances and (not sites or row['step']==0):
            phase,reason='initial','valid object distance, no approach/contact/manipulation evidence'
        else:
            phase,reason='unknown','insufficient reliable geometric/contact evidence'
        row.update(phase=phase,phase_evidence=evidence,phase_reason=reason,
                   phase_kind='offline heuristic, not ground truth')
        if command is not None:
            previous_command=command
    return rows


def events(rows):
    result=[]; last={}
    for index,row in enumerate(rows):
        if index==0:
            continue
        previous=rows[index-1]
        kinds=[]
        a,b=row.get('executed_action'),previous.get('executed_action')
        if a is not None and b is not None and (a[6]>.2 and b[6]<-.2 or a[6]<-.2 and b[6]>.2):
            kinds.append('gripper_switch')
        for name in ('robot_contacts','robot_grasps'):
            now,old=observed_any(row.get(name,{})),observed_any(previous.get(name,{}))
            if now is not None and old is not None and now!=old:
                kinds.append(name+'_change')
        if row.get('goal_progress')!=previous.get('goal_progress'):
            kinds.append('goal_progress_change')
        if index>=10:
            p5,p10=rows[index-5],rows[index-10]
            follow,oldfollow=motion(row,p5),motion(p5,p10)
            if follow is not None and oldfollow is not None and follow!=oldfollow:
                kinds.append('object_follow_change')
            xyz=[r.get('eef_pos') for r in (row,p5,p10)]
            if all(p is not None for p in xyz):
                d1,d0=np.asarray(xyz[0])-xyz[1],np.asarray(xyz[1])-xyz[2]
                n1,n0=np.linalg.norm(d1),np.linalg.norm(d0)
                if min(n1,n0)>.005 and np.dot(d1,d0)/(n1*n0)<.5:
                    kinds.append('motion_direction_change')
        for kind in kinds:
            if row['step']-last.get(kind,-1000)>=RULES['event_debounce_steps']:
                result.append(dict(step=row['step'],kind=kind,evidence=row['phase_evidence']))
                last[kind]=row['step']
    return result


def select_requests(infers,rows):
    if not infers:
        raise ValueError('Cannot select from empty rollout')
    steps=np.asarray([r['step'] for r in infers])
    if len(set(steps.tolist()))!=len(steps) or np.any(np.diff(steps)<=0):
        raise ValueError('Requests must have distinct increasing control steps')
    selected={}
    def add(index,source,event=None):
        item=selected.setdefault(index,dict(request=infers[index]['request'],step=int(steps[index]),
                                          path=infers[index]['path'],sources=[],events=[]))
        if source not in item['sources']:
            item['sources'].append(source)
        if event is not None and event not in item['events']:
            item['events'].append(event)
    for position in np.linspace(steps[0],steps[-1],min(12,len(infers))):
        add(int(np.argmin(np.abs(steps-position))),'uniform')
    candidates=events(rows)
    used_types={}; event_additions=0
    while candidates and event_additions<8:
        current=np.asarray([steps[i] for i in selected])
        # Cover event types first, then prefer events far from existing samples.
        event=min(candidates,key=lambda e:(used_types.get(e['kind'],0),
                        -float(np.min(np.abs(current-e['step']))),e['step'],e['kind']))
        candidates.remove(event)
        used_types[event['kind']]=used_types.get(event['kind'],0)+1
        right=bisect_left(steps.tolist(),event['step'])
        for index in dict.fromkeys((max(0,right-1),min(len(steps)-1,right))):
            is_new=index not in selected
            if is_new and (event_additions>=8 or len(selected)>=20):
                continue
            add(index,'event',event)
            event_additions+=int(is_new)
    target=min(20,len(infers))
    while len(selected)<target:
        remaining=[i for i in range(len(infers)) if i not in selected]
        ordered=sorted(selected)
        # Fill the widest remaining interval with its nearest midpoint request.
        gaps=[(float(steps[b]-steps[a]),a,b) for a,b in zip(ordered,ordered[1:]) if b-a>1]
        if gaps:
            _,a,b=max(gaps,key=lambda x:(x[0],-x[1]))
            index=min(range(a+1,b),key=lambda i:(abs(steps[i]-(steps[a]+steps[b])/2),i))
        else:
            index=remaining[0]
        add(index,'uniform_fill')
    by_step={r['step']:r for r in rows}
    output=[]
    for index,item in sorted(selected.items()):
        row=by_step[item['step']]
        output.append(dict(item,phase=row['phase'],phase_evidence=row['phase_evidence'],
                           phase_reason=row['phase_reason']))
    return output


def select_episode(directory):
    directory=Path(directory)
    rows=annotate(read_jsonl(directory/'trace.jsonl'))
    selected=select_requests(read_jsonl(directory/'infer.jsonl'),rows)
    write_jsonl(directory/'phase_trace.jsonl',rows)
    atomic_json(directory/'selection.json',dict(rules=RULES,samples=selected,
                selected=len(selected),available=len(read_jsonl(directory/'infer.jsonl')),
                label_kind='trace-derived offline heuristic'))
    return selected
