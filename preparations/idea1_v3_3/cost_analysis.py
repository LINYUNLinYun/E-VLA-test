"""Audit raw closed-loop costs and uniform cold/warm diagnostics.

This module does not authorize promotion; frozen decision receipts do that.
No request wall or scientific outcome is rewritten.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records
from .manifest import cases


def load_nested(directory):
    rows=[]
    for p in Path(directory).glob('episodes/*/task*/init*/result.json'):
        if '.invalid.' not in str(p) and p.with_name('complete.json').exists():
            row=json.loads(p.read_text());row['_directory']=str(p.parent);rows.append(row)
    return rows


def profile(rows):
    if not rows:return dict(n=0)
    model=rows[0]['config']['model'];first=0 if model=='xiaomi' else 6;second=first+1
    total_ms=sum(r['policy_wall_s']*1000 for r in rows);actions=sum(r['actions'] for r in rows)
    by_init={i:dict(ms=0.,actions=0,first_policy=[],first_input=[],nonstartup_ms=0.,nonstartup_actions=0) for i in (first,second)}
    pairs=defaultdict(dict);modules=defaultdict(float);sources=[];gpu_rows=defaultdict(list)
    for episode in rows:
        directory=Path(episode['_directory']);path=directory/'infer.jsonl';requests=[json.loads(l) for l in path.read_text().splitlines()];init=episode['case']['init'];item=by_init.get(init)
        if item is None:continue
        item['ms']+=episode['policy_wall_s']*1000;item['actions']+=episode['actions'];item['first_policy'].append(requests[0]['timing']['policy_wall_ms'])
        first_input=sum(v for k,v in requests[0]['timing'].items() if k.endswith('input_processing_ms'))
        item['first_input'].append(first_input);item['nonstartup_ms']+=sum(r['timing']['policy_wall_ms'] for r in requests[1:]);item['nonstartup_actions']+=sum(r['executed_length'] for r in requests[1:])
        pairs[(episode['case']['suite'],episode['case']['task'])][init]=requests[0]['timing']['policy_wall_ms']
        for r in requests:
            for k,v in r['timing'].items():
                if isinstance(v,(int,float)) and k.endswith('_ms'):modules[k]+=v
        sources.append(dict(case=episode['case'],result_path=str(directory/'result.json'),result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest(),infer_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        side=json.loads((directory/'engineering_revision.json').read_text()) if (directory/'engineering_revision.json').exists() else episode.get('engineering_binding',{})
        gpu_rows[str(side.get('model_gpu','unknown'))].append(episode)
    init_stats={}
    for i,v in by_init.items():
        init_stats[str(i)]=dict(n=len(v['first_policy']),policy_ms_per_action=v['ms']/v['actions'] if v['actions'] else None,first_policy_ms_median=float(np.median(v['first_policy'])) if v['first_policy'] else None,
            first_input_processing_ms_median=float(np.median(v['first_input'])) if v['first_input'] else None,nonstartup_ms_per_action=v['nonstartup_ms']/v['nonstartup_actions'] if v['nonstartup_actions'] else None)
    paired_adjustment=sum(v[first]-v[second] for v in pairs.values() if first in v and second in v)
    return dict(n=len(rows),actions=actions,raw_ms_per_action=total_ms/actions,initialization_subcohorts=init_stats,
        paired_startup_adjusted_sensitivity_ms_per_action=(total_ms-paired_adjustment)/actions,
        first_step_capped_300ms_sensitivity_ms_per_action=(total_ms-sum(max(0,x-300) for v in by_init.values() for x in v['first_policy']))/actions,
        sensitivity_warning='Diagnostics only; complete80 raw closed-loop wall remains the main cost. No effective episode was removed or altered.',
        modules_ms_per_action={k:v/actions for k,v in modules.items()},physical_gpu_case_counts={k:len(v) for k,v in gpu_rows.items()},sources=sources)


def main():
    groups=records();parent={};root=OUT/'engineering_cost_rebaseline'
    for gpu in (0,2,3):
        path=root/f'gpu{gpu}_comparison.json';rows=load_nested(root/f'pi0_parent_cost_gpu{gpu}')
        if len(rows)==80 and path.exists():parent[str(gpu)]=dict(profile=profile(rows),numerical_comparison=json.loads(path.read_text()),comparison_path=str(path),comparison_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    candidates={cid:profile(rows) for cid,rows in groups.items() if len(rows)==80}
    atomic(OUT/'cost_calibration_progress.json',dict(time=time.time(),status='descriptive_profiles_only; selection_pending',matched_parent_by_gpu=parent,scientific_conditions=candidates,
        method='Same formulas for every condition: raw complete policy/actual-actions; first/second init; full step0 input preparation; nonstartup; paired startup and300ms cap diagnostics. No old cross-GPU baseline fallback.',
        numeric_scope='Old-parent first two SpatialTask0 init6/7 differ from later-runtime engineering; profile and exact comparison retain that fact. GPU0 preparation outlier is not treated as algorithm speed.'))
    print({gpu:round(data['profile']['raw_ms_per_action'],3) for gpu,data in parent.items()})


if __name__=='__main__':main()
