"""Freeze final matched-cost decisions after all bounded and engineering80."""
import hashlib
import json
from pathlib import Path
import time
from collections import Counter
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records
from .cost_analysis import profile,load_nested
from .costs import receipt


def equivalent(a,b,groups):
    def key(r):return tuple(r['case'][k] for k in ('suite','task','init'))
    left={key(r):r for r in groups[a]};right={key(r):r for r in groups[b]};checks=[];counts={a:Counter(),b:Counter()}
    if len(left)!=80 or left.keys()!=right.keys():raise ValueError('Equivalence requires identical complete80')
    for k,x in sorted(left.items()):
        y=right[k];p=Path(x['_directory']);q=Path(y['_directory'])
        with np.load(p/'arrays.npz') as aa,np.load(q/'arrays.npz') as bb:
            fields=sorted(f for f in aa.files if f.startswith('actions_'));other=sorted(f for f in bb.files if f.startswith('actions_'))
            exact=fields==other and all(np.array_equal(aa[f],bb[f]) for f in fields)
        i=[json.loads(l) for l in (p/'infer.jsonl').read_text().splitlines()];j=[json.loads(l) for l in (q/'infer.jsonl').read_text().splitlines()]
        for name,logs in ((a,i),(b,j)):
            for r in logs:
                counts[name][r['route']['path']]+=1
                if 'distance' in r['route']:
                    counts[name]['verified_draft_proposals']+=1
                    length=r['route'].get('accepted_length');counts[name]['accepted_full10']+=length==10;counts[name]['accepted_zero']+=length==0;counts[name]['accepted_partial']+=length not in (0,10)
        rhythm=lambda logs:[(r['route']['path'],r['executed_length']) for r in logs]
        checks.append(dict(case=list(k),decoded_exact=exact,request_path_and_execution_exact=rhythm(i)==rhythm(j),success_equal=x['success']==y['success'],
            left_arrays_sha256=hashlib.sha256((p/'arrays.npz').read_bytes()).hexdigest(),right_arrays_sha256=hashlib.sha256((q/'arrays.npz').read_bytes()).hexdigest()))
    return dict(left=a,right=b,n=80,all_exact=all(c['decoded_exact'] and c['request_path_and_execution_exact'] and c['success_equal'] for c in checks),route_counts={name:dict(values) for name,values in counts.items()},checks=checks)


def main():
    target=OUT/'cost_gate_final.json'
    if target.exists():raise ValueError('Final receipt immutable')
    groups=records();parent=json.loads((OUT/'manifest.json').read_text());stage=OUT/'stages/bounded_followups/manifest.json'
    cfgs=parent['conditions']+json.loads(stage.read_text())['conditions']
    if any(len(groups[c['id']])!=80 for c in cfgs):raise ValueError('Scientific first/bounded80 not complete')
    prelim=receipt('pi0_screen');conditions=dict(prelim['conditions']);root=OUT/'engineering_cost_rebaseline'
    matches=list(root.glob('xiaomi_gpu*_comparison.json'))
    xparents={json.loads(p.read_text())['gpu']:json.loads(p.read_text()) for p in matches}
    if set(xparents)!={2,3}:raise ValueError('Require complete Xiaomi matched engineering80 on BOTH physicalGPU2 and3')
    if any(x['profile']['n']!=80 for x in xparents.values()):raise ValueError('Xiaomi engineering80 count mismatch')
    finished=root/'xiaomi_gpu2_fresh_finished.json'
    if not finished.exists() or not json.loads(finished.read_text()).get('all_own_C_G_exited'):raise ValueError('Fresh GPU2 engineering service must release all resources before promotion')
    parents={g:profile(load_nested(root/f'pi0_parent_cost_gpu{g}')) for g in (0,2,3)}
    profiles={};pending=[]
    for c in cfgs:
        cid=c['id'];p=profile(groups[cid]);profiles[cid]=p
        if c['experiment'] not in ('E1','E2','E3','E4') or cid in conditions:continue
        quality_base=sum(r['success'] for r in groups['pi0_parent' if c['model']=='pi0' else 'xiaomi_pure_d2'])
        quality=sum(r['success'] for r in groups[cid]);counts=p['physical_gpu_case_counts']
        gpu=int(next(iter(counts))) if len(counts)==1 else None
        bindings=Counter();binding_sources=[]
        for r in groups[cid]:
            side=Path(r['_directory'])/'engineering_revision.json'
            binding=json.loads(side.read_text()) if side.exists() else r.get('engineering_binding',{})
            bindings[(binding.get('model_gpu'),binding.get('render_gpu'))]+=1
            if side.exists():binding_sources.append(dict(path=str(side),sha256=hashlib.sha256(side.read_bytes()).hexdigest()))
        p['engineering_binding_sources']=binding_sources
        corrected_binding=gpu is not None and set(bindings)=={(gpu,gpu)}
        base=(parents.get(gpu) if c['model']=='pi0' else xparents[gpu]['profile'] if gpu in xparents else None) if corrected_binding else None
        reason='No matched same-model/render physical-GPU baseline; legacy/mixed EGL or missing same-card evidence prohibits cost qualification';verification='inconclusive';ratio=None;eligible=False
        if quality<quality_base-2:
            reason='Formal complete80 quality failed; no cost confirmation needed';verification='quality_failed'
        elif base is not None:
            ratio=p['raw_ms_per_action']/base['raw_ms_per_action']
            hot='7' if c['model']=='pi0' else '1';hot_ratio=p['initialization_subcohorts'][hot]['policy_ms_per_action']/base['initialization_subcohorts'][hot]['policy_ms_per_action']
            verification='confirmed';reason=f'Matched complete-wall ratio {ratio:.9f}; second-init sensitivity ratio {hot_ratio:.9f}'
            if c['model']=='pi0' and gpu==0:
                verification='inconclusive';reason+='; GPU0 preparation outliers prohibit a raw-wall speed qualification'
            elif ratio<=.9 and hot_ratio>.9:
                verification='inconclusive';reason+='; threshold depends on startup imbalance, cost inconclusive without paired confirmation'
            eligible=verification=='confirmed' and ratio<=.9
        conditions[cid]=dict(physical_gpu=gpu,verification=verification,matched_cost_ratio=ratio,eligible_cost_gate=eligible,reason=reason,
            candidate_model_render_binding_counts={str(k):v for k,v in bindings.items()},
            matched_parent_engineering_id=(f'pi0_parent_cost_gpu{gpu}' if c['model']=='pi0' and gpu is not None else f'xiaomi_pure_cost_gpu{gpu}' if c['model']=='xiaomi' and gpu in xparents else None),raw_ms_per_action=p['raw_ms_per_action'])
        if verification=='inconclusive':pending.append(cid)
    equivalences=[];comparison=None
    if 'E4_tau1.6' in groups:
        comparison=equivalent('E4_tau0.8','E4_tau1.6',groups)
        if comparison['all_exact']:equivalences.append(dict(members=['E4_tau0.8','E4_tau1.6'],representative='E4_tau0.8',rule='Exact all80 decoded actions, path/execution rhythm and outcomes; choose preregistered smaller tau, not wall-time noise'))
    evidence=OUT/'cost_gate_final_profiles.json';atomic(evidence,dict(scientific_profiles=profiles,matched_parent=parents,matched_Xiaomi_by_physical_gpu=xparents,E4_boundary_comparison=comparison))
    if comparison is not None:atomic(OUT/'E4_tau08_tau16_exact_receipt.json',comparison)
    technical=[]
    for p in root.glob('*/episodes/*/task*/init*.invalid.*/technical_error.json'):
        technical.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),error=json.loads(p.read_text())))
    paths=[evidence,OUT/'cost_gate_pi0_screen.json',*matches,finished,Path(__file__),stage,Path(__file__).with_name('cost_analysis.py')]
    paths += [root/f'xiaomi_gpu{g}_{kind}.json' for g in (2,3) for kind in ('manifest','boundary_receipt')]
    paths += [OUT/name for name in ['bounded_probe_history_h1.json','bounded_probe_history_h1_execution.json','D1_native_euler_probe.json','E4_timing_device_scope_receipt.json']]
    paths += [Path(__file__).with_name(name) for name in ['cross_restart_history_h1.py','server_c1_h1.py','stage_h1.py','xiaomi_cost_fresh.py']]
    paths += [root/'xiaomi_fresh_gpu2_service'/name for name in ['native_probe.json','initial_input_rgb_parity.json']]
    atomic(target,dict(status='frozen',time=time.time(),scope='final_confirmation_selection',conditions=conditions,behavioral_equivalence_classes=equivalences,
        main_metric='Full closed-loop policy wall / actual executed actions, same physical GPU and corrected EGL; no unmatched fallback.',
        inconclusive_cost_no_speed_promotion=pending,quality_first_E4_remains_eligible=True,scientific_denominator_increment=0,
        engineering_valid_budget=400,matched_Xiaomi_baseline_rule='Candidate physicalGPU2 uses onlyGPU2 pure80; physicalGPU3 uses onlyGPU3 pure80. Legacy cross-EGL/mixed bindings do not receive a speed qualification.',
        engineering_technical_attempts=technical,engineering_history_warning='Only valid complete80 enters each engineering main denominator. GPU2 parent Long task04 init6 had one retained OOM/retry; prior failed preparation may change cache/cold history. Its21.5986ms reference and borderline9.11% E2 reduction are not precise causal speed estimates.',
        hashed_inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]))
    gate=OUT/'cost_selection_gate.json';data=json.loads(gate.read_text());data.update(pending=False,final_receipt=str(target),final_frozen=time.time());atomic(gate,data)
    print(target)


if __name__=='__main__':main()
