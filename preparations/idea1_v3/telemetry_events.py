"""Bounded post-hoc fixed-key telemetry descriptions; no causal classifier.

Supply final trajectory_index.csv files with repeated --trajectory-index and
historical_trajectory_references.csv with --historical-reference. No glob scans.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from preparations.idea1_v3.analysis import phase_label, write_csv

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
PARENT='E1_D2_tau0.1_P2'


def first_run(values, length=3):
    for i in range(len(values)-length+1):
        if all(values[i:i+length]): return i,i+length-1
    return None,None


def compare(a,b):
    # Same control step is same simulated elapsed time, not same wall time.
    left={int(s.get('step',i)):s for i,s in enumerate(a)}
    right={int(s.get('step',i)):s for i,s in enumerate(b)}
    steps=sorted(set(left)&set(right))
    timed=[k for k in steps if left[k].get('sim_time_s') is not None and right[k].get('sim_time_s') is not None]
    if timed:
        base=timed[0]
        if any(abs((left[k]['sim_time_s']-left[base]['sim_time_s'])-(right[k]['sim_time_s']-right[base]['sim_time_s']))>1e-6 for k in timed):
            raise ValueError('Same-step simulated elapsed times differ; cannot claim same-time comparison')
    distances=[float(np.linalg.norm(np.asarray(left[k]['eef_pos'])-np.asarray(right[k]['eef_pos']))) for k in steps]
    flags=[d>0.05 for d in distances]
    # Missing control steps cannot form a sustained run.
    start=confirm=None
    for i in range(len(steps)-2):
        if steps[i+2]==steps[i]+2 and all(flags[i:i+3]): start,confirm=i,i+2; break
    result=dict(common_points=len(steps),variant_points=len(a),reference_points=len(b),
        time_alignment='sim elapsed time validated' if len(timed)==len(steps) and timed else 'control-step alignment; sim-time unavailable for some/all points',
        max_distance_m=max(distances) if distances else None,
        divergence_status='observed' if start is not None else 'not_observed',
        onset_step=steps[start] if start is not None else None,
        confirmation_step=steps[confirm] if confirm is not None else None)
    for label,index in [('onset',start),('confirmation',confirm)]:
        for side,states in [('variant',left),('reference',right)]:
            state=states[steps[index]] if index is not None else {}
            result[f'{side}_{label}_phase']=phase_label(state) if state else None
            result[f'{side}_{label}_sim_time_s']=state.get('sim_time_s')
    return result


def grasp_events(states):
    names=sorted({name for s in states for name in s.get('robot_grasps',{})}) or ['__no_objects_recorded__']
    rows=[]
    for name in names:
        positive=[i for i,s in enumerate(states) if s.get('robot_grasps',{}).get(name) is True]
        first=positive[0] if positive else None
        # This is disappearance of positive evidence, NOT evidence of loss.
        after=next((i for i in range(first+1,len(states)) if states[i].get('robot_grasps',{}).get(name) is not True),None) if first is not None else None
        row=dict(object=name,positive_status='observed' if positive else 'not_observed',
            positive_observations=len(positive),observations=len(states),
            availability_unknown_observations=len(states)-len(positive),grasp_loss='undetermined')
        for label,index in [('first_positive',first),('first_subsequent_positive_not_observed',after)]:
            state=states[index] if index is not None else {}
            row[label+'_step']=state.get('step',index)
            row[label+'_sim_time_s']=state.get('sim_time_s')
        rows.append(row)
    return rows


def read_index(path,historical=False):
    path=Path(path).resolve(); rows=[]
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if int(row['init'])!=6 or int(row['task']) not in (0,5): continue
            stage=row.get('experiment') if historical else row.get('stage')
            if stage not in ('E1','E2','E3','E4'): continue
            raw=Path(row['telemetry']); row['telemetry']=str(raw if raw.is_absolute() else (path.parent/raw).resolve())
            row.update(stage=stage,task=int(row['task']),init=6,index_source=str(path),historical=historical)
            rows.append(row)
    return rows


def run(indices,historical,output):
    def deduplicate(values):
        unique={}
        for row in values:
            identity=tuple(row[k] for k in ('stage','condition','suite','task','init'))
            if identity in unique:
                prior=unique[identity]
                if prior['telemetry']!=row['telemetry']: raise ValueError(f'Conflicting telemetry paths: {identity}')
                if prior.get('success')!=row.get('success'): raise ValueError(f'Conflicting success: {identity}')
                prior['index_sources']=sorted(set(prior['index_sources']+[row['index_source']]))
            else:
                unique[identity]=dict(row,index_sources=[row['index_source']])
        return list(unique.values())
    rows=deduplicate(r for p in indices for r in read_index(p))
    refs=deduplicate(r for p in historical for r in read_index(p,True))
    cache={}; missing=set()
    def load(row):
        path=row['telemetry']
        if path not in cache:
            if not Path(path).is_file(): missing.add(path); cache[path]=None
            else: cache[path]=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cache[path]
    pairs=[]; grasps=[]; seen=set()
    for stage,condition in sorted({(r['stage'],r['condition']) for r in rows}):
        for suite in SUITES:
            for task in (0,5):
                key=dict(stage=stage,condition=condition,suite=suite,task=task,init=6)
                choices=[r for r in rows if all(r[k]==v for k,v in key.items())]
                if len(choices)>1: raise ValueError(f'Duplicate final trajectory: {key}')
                variant=choices[0] if choices else None
                pool=refs if stage in ('E1','E4') else rows
                refcondition='historical_pi0_Full_D2_v2.2' if stage in ('E1','E4') else ('E2_Full' if stage=='E2' else PARENT)
                candidates=[r for r in pool if r['condition']==refcondition and r['suite']==suite and r['task']==task and r['init']==6 and (r['stage']==stage if stage in ('E1','E4') else r['stage']==('E2' if stage=='E2' else 'E1'))]
                if len(candidates)>1: raise ValueError(f'Duplicate reference: {key}')
                reference=candidates[0] if candidates else None
                result=dict(key,reference_condition=refcondition,rule='post-hoc fixed >0.05m for 3 consecutive control steps; descriptive, not statistical significance or causation',reference_boundary='historical Full-D2, not D10; mixed execution visual reference' if stage in ('E1','E4') else 'current Full' if stage=='E2' else 'frozen E1 parent')
                a=load(variant) if variant else None; b=load(reference) if reference else None
                result.update(pair_status='available' if a and b else 'missing_variant_or_reference',variant_telemetry=variant['telemetry'] if variant else None,reference_telemetry=reference['telemetry'] if reference else None,variant_success=variant.get('success') if variant else None,reference_success=reference.get('success') if reference else None)
                result.update(variant_index_sources=json.dumps(variant['index_sources']) if variant else None,reference_index_sources=json.dumps(reference['index_sources']) if reference else None)
                baseline_self=stage=='E2' and condition=='E2_Full'
                result['variant_case']=not baseline_self
                if baseline_self: result.update(pair_status='baseline_self',divergence_status='no_variant')
                elif a and b: result.update(compare(a,b))
                pairs.append(result)
                for row,states in [(variant,a),(reference,b)]:
                    if not row or states is None or row['telemetry'] in seen: continue
                    seen.add(row['telemetry'])
                    grasps.extend(dict(stage=row['stage'],condition=row['condition'],suite=suite,task=task,init=6,telemetry=row['telemetry'],**event) for event in grasp_events(states))
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    write_csv(output/'fixed_key_divergence_events.csv',pairs)
    write_csv(output/'fixed_key_grasp_evidence.csv',grasps)
    (output/'telemetry_events_receipt.json').write_text(json.dumps(dict(trajectory_indices=list(map(str,indices)),historical_indices=list(map(str,historical)),unique_paths_read=len(cache)-len(missing),missing_paths=sorted(missing),fixed_keys='init6/task0,5 across four suites',alignment='same control step; common intersection only, no padding/interpolation; sim_time recorded when available',grasp_boundary='False/missing is availability_unknown; intentional release cannot be excluded; grasp_loss always undetermined'),indent=2))


def self_test():
    a=[dict(step=i,eef_pos=[0,0,0],robot_grasps={'x':i==1}) for i in range(7)]
    b=[dict(step=i,eef_pos=[.06 if i>=1 else .05,0,0]) for i in range(5)]
    r=compare(a,b); assert (r['onset_step'],r['confirmation_step'],r['common_points'])==(1,3,5)
    assert compare(a,b[:3])['divergence_status']=='not_observed'
    g=grasp_events(a)[0]; assert g['first_positive_step']==1 and g['first_subsequent_positive_not_observed_step']==2 and g['availability_unknown_observations']==6 and g['grasp_loss']=='undetermined'
    assert grasp_events(b)[0]['positive_status']=='not_observed'
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'index.csv'; write_csv(path,[dict(stage='E4',condition='draft',suite='libero_spatial',task=0,init=6,telemetry='missing.jsonl',success=False)])
        run([path],[],Path(directory)/'out')
        with (Path(directory)/'out/fixed_key_divergence_events.csv').open() as stream: assert len(list(csv.DictReader(stream)))==8
        # Repeated recursive views are not repeated episodes; conflicting paths fail.
        run([path,path],[],Path(directory)/'dedup')
        write_csv(path,[dict(stage='E2',condition='E2_Full',suite='libero_spatial',task=0,init=6,telemetry='missing.jsonl',success=False)])
        run([path],[],Path(directory)/'self')
        with (Path(directory)/'self/fixed_key_divergence_events.csv').open() as stream:
            assert all(r['pair_status']=='baseline_self' and r['variant_case']=='False' for r in csv.DictReader(stream))
    print('PASS: sustained threshold, common truncation, false unknown, missing pairs/fixed eight keys')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trajectory-index',action='append',default=[])
    p.add_argument('--historical-reference',action='append',default=[])
    p.add_argument('--output',type=Path); p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test: self_test(); return
    if not args.trajectory_index or not args.output: p.error('explicit --trajectory-index and --output required')
    run(args.trajectory_index,args.historical_reference,args.output)


if __name__=='__main__': main()
