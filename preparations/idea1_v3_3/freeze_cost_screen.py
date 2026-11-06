"""Immutable preliminary E2 matched-cost screen; final Xiaomi gate is separate."""
import hashlib
import json
from pathlib import Path
import time
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records
from .cost_analysis import profile,load_nested


def main():
    path=OUT/'cost_gate_pi0_screen.json'
    if path.exists():
        raise ValueError('Preliminary receipt already frozen; never overwrite')
    groups=records();root=OUT/'engineering_cost_rebaseline'
    parents={str(g):profile(load_nested(root/f'pi0_parent_cost_gpu{g}')) for g in (0,2,3)}
    if any(p['n']!=80 for p in parents.values()):raise ValueError('Matched parent80 incomplete')
    conditions={};profiles={};sensitivity={}
    baseline={tuple(r['case'][k] for k in ('suite','task','init')):r for r in groups['pi0_parent']}
    for cid,rows in groups.items():
        if not cid.startswith('E2_') or cid.endswith('_no_c'):continue
        if len(rows)!=80:raise ValueError('E2 preliminary lacks complete80: '+cid)
        p=profile(rows);profiles[cid]=p
        gpu_counts=p['physical_gpu_case_counts']
        if len(gpu_counts)!=1:raise ValueError('Mixed GPU E2 screen needs stratification')
        gpu=next(iter(gpu_counts));base=parents[gpu];ratio=p['raw_ms_per_action']/base['raw_ms_per_action']
        confirmed=gpu!='0';control=cid=='E2_alternate'
        conditions[cid]=dict(physical_gpu=int(gpu),matched_parent_engineering_id=f'pi0_parent_cost_gpu{gpu}',matched_cost_ratio=ratio,
            verification='confirmed' if confirmed else 'inconclusive',eligible_cost_gate=confirmed and ratio<=.9 and not control,
            reason=('GPU0 complete-wall ratio is inconclusive because parent input preparation outliers conflict with hot/startup sensitivities.' if not confirmed else
                'Alternate is a preregistered control, excluded from main candidate promotion.' if control else
                f'Matched complete-wall ratio {ratio:.9f} '+('meets' if ratio<=.9 else 'exceeds')+' the 0.9 maximum for a >=10% reduction.'),
            raw_ms_per_action=p['raw_ms_per_action'],matched_parent_ms_per_action=base['raw_ms_per_action'],
            parent_first_second_init=base['initialization_subcohorts'],candidate_first_second_init=p['initialization_subcohorts'])
        changes=[]
        for r in rows:
            key=tuple(r['case'][k] for k in ('suite','task','init'))
            if key[0]=='libero_spatial' and key[1]==0 and key[2] in (6,7):continue
            b=baseline[key]
            changes.append((bool(b['success']),bool(r['success'])))
        sensitivity[cid]=dict(diagnostic_only=True,excluded_fixed_cases=[['libero_spatial',0,6],['libero_spatial',0,7]],n=len(changes),
            recovered=sum(not a and b for a,b in changes),new_failures=sum(a and not b for a,b in changes),
            formal80_unchanged=True,warning='First-service native-probe runtime history correlates with two numeric trajectory differences; responsible operator unknown. Never remove these from formal quality denominator.')
    evidence=OUT/'cost_gate_pi0_screen_profiles.json'
    atomic(evidence,dict(parents=parents,candidates=profiles,quality_history_sensitivity=sensitivity))
    paths=[evidence,Path(__file__),Path(__file__).with_name('costs.py'),root/'initial_task_numeric_history_matrix.json',root/'legacy_timing_confound.json']
    paths += [root/f'gpu{g}_{kind}.json' for g in (0,2,3) for kind in ('comparison','boundary_receipt','manifest')]
    atomic(path,dict(status='frozen',scope='pi0_screen_only',time=time.time(),conditions=conditions,
        main_metric='Complete closed-loop policy wall milliseconds / actually executed actions; no old cross-GPU fallback',
        scientific_denominator_increment=0,engineering_episodes=240,final_receipt_still_pending=True,
        no_c_rule='Quality first if no credible >=10% cost candidate; ties use confirmed same-GPU normalized cost ratio then conservative parameters. No-c is a diagnostic, not promotion.',
        quality_history_sensitivity=sensitivity,hashed_inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    gate=OUT/'cost_selection_gate.json';data=json.loads(gate.read_text());data.update(pi0_screen_ready=True,pi0_screen_receipt=str(path),pending=True,final_pending_reason='Xiaomi pure matched80 and complete E1/E3 cost decisions remain pending')
    atomic(gate,data)
    print(path)


if __name__=='__main__':main()
