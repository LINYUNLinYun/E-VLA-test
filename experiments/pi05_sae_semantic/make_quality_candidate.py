"""Freeze a nontrivial relative-impact prior before any heldout rollout SR.

The strict delta=0.1 outcome is preserved. This separate exploratory candidate
chooses the least measured calibration-cost pair and the smallest gate budget
that yields replay coverage; it does not claim latent invariance or good SR.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',required=True)
    p.add_argument('--selection',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    study,selection,output=map(lambda x:Path(x).resolve(),(args.study,args.selection,args.output))
    if json.loads((study/'study_status.json').read_text())['state']!='offline_mask_study_complete':
        raise ValueError('Complete actual combination/calibration results first')
    base=json.loads((study/'skip_rules.json').read_text())
    with (study/'combination_calibration/concept_ffn_matrix.csv').open() as stream:measurements=list(csv.DictReader(stream))
    rows=json.loads((selection/'score_observations.json').read_text());scores=np.load(selection/'semantic_scores.npz')
    definitions=json.loads((selection/'concept_features.json').read_text())
    pool=json.loads((selection/'semantic_selection_summary.json').read_text()).get('token_pool','all')
    early={c:d for c,d in definitions.items() if d['layers']['5']['supported']}
    by_mask={}
    for row in measurements:
        if row['routing_supported']=='True' and row['D']:
            by_mask.setdefault(row['mask'],{})[row['concept']]={k:float(row[k]) for k in ('D','E')}
    possibilities=[];cal=[]
    for i,row in enumerate(rows):
        if row['split']!='calibration':continue
        active={c for c,d in early.items() if scores[f"L5_{d['category']}_{d['concept']}_{pool}"][i]>=d['layers']['5']['theta']}
        call=row['request']+1;refresh=call<=2 or call%10==0
        cal.append((row,active,refresh))
        if not active or refresh:continue
        for mask,support in by_mask.items():
            if active<=set(support):possibilities.append((max(support[c]['D'] for c in active),mask))
    if not possibilities:raise ValueError('No combination has conditional support and nontrivial replay coverage')
    delta,mask=min(possibilities,key=lambda r:(r[0],r[1]))
    supported={c:v for c,v in by_mask[mask].items() if v['D']<=delta}
    eligible=sum(bool(active and active<=set(supported) and not refresh) for row,active,refresh in cal)
    opportunities=sum(not refresh for row,active,refresh in cal)
    target_period=opportunities/eligible
    schedules=[]
    for period in range(1,17):
        for offset in range(period):
            count=sum(not refresh and (row['request']+offset)%period==0 for row,active,refresh in cal)
            schedules.append((abs(count-eligible),abs(period-target_period),offset,period,count))
    _,_,offset,period,static_count=min(schedules)
    rule={**base,'state':'exploratory_relative_prior_needs_quality_test','delta_semantic':delta,
        'masks':[dict(mask=[int(s) for s in mask.split(',')],supported_concepts=supported)],
        'static_mask':[int(s) for s in mask.split(',')],'static_period':period,'static_offset':offset,
        'selection':str(selection),'token_pool':pool,
        'early_concepts':{c:d['layers']['5'] for c,d in early.items()},
        'strict_default_result':str(study/'skip_rules.json'),
        'candidate_selection':'Lowest measured conditional calibration cost with at least one non-refresh replay opportunity; no SR used',
        'replay_eligible_after_refresh':eligible,'replay_static_after_refresh':static_count,
        'latent_invariance_at_delta_0_1':False,'closed_loop_evaluated':False,'gpu_policy_latency_measured':False,
        'interpretation':'Relative-impact prior for a nontrivial experiment; not a claim that semantic activations or actions are preserved'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(rule,indent=2))
    print(json.dumps({k:rule[k] for k in ['state','delta_semantic','static_mask','static_period','static_offset','replay_eligible_after_refresh','replay_static_after_refresh']},indent=2))


if __name__=='__main__':
    main()
