"""Finite single-module -> actual combination -> calibration study.

Single-module costs only propose masks. Actual combination effects are always
measured. Unsupported or absent late evidence is never interpreted as zero
semantic impact. This script does not launch rollouts or claim policy speedup.
"""
import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def propose_masks(rows,max_masks,diversify=False):
    concepts=sorted({r['concept'] for r in rows})
    costs={(int(r['mask']),r['concept']):float(r['D']) for r in rows
           if r['supported']=='True' and r['D'] and math.isfinite(float(r['D'])) and ',' not in r['mask']}
    candidates=[]
    for pair in itertools.combinations((6,8,10,12,14,15),2):
        if pair[1]-pair[0]==1:continue
        if not all((m,c) in costs for m in pair for c in concepts):continue
        values={c:sum(costs[(m,c)] for m in pair) for c in concepts}
        candidates.append(dict(mask=list(pair),proposal_cost=max(values.values()),per_concept_proposal_cost=values))
    candidates.sort(key=lambda r:r['proposal_cost'])
    if not candidates:return [],[]
    selected=[candidates[0]]
    for concept in concepts:
        best=min(candidates,key=lambda r:r['per_concept_proposal_cost'][concept])
        if best not in selected and len(selected)<max_masks:selected.append(best)
    if diversify:
        # If every concept proposes the same pair, test a bounded alternative
        # without each component. Single-module additivity is not assumed.
        for excluded in candidates[0]['mask']:
            options=[r for r in candidates if excluded not in r['mask'] and r not in selected]
            if options and len(selected)<max_masks:selected.append(options[0])
    return selected,candidates


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collection',required=True)
    p.add_argument('--selection',required=True)
    p.add_argument('--assets',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--wait-matrix',required=True)
    p.add_argument('--wait-pid',type=int,required=True)
    p.add_argument('--max-masks',type=int,default=3)
    p.add_argument('--delta-semantic',type=float,default=.1)
    p.add_argument('--diversify-proposals',action='store_true')
    args=p.parse_args()
    if not 1<=args.max_masks<=3 or not math.isfinite(args.delta_semantic) or args.delta_semantic<0:
        raise ValueError('Use one to three masks and a finite nonnegative semantic threshold')
    collection,selection,assets,output,previous=map(lambda x:Path(x).resolve(),
        (args.collection,args.selection,args.assets,args.output,args.wait_matrix))
    output.mkdir(parents=True,exist_ok=True)
    def status(value,**kwargs):
        data=dict(state=value,pid=os.getpid(),updated_unix=time.time(),**kwargs)
        (output/'study_status.json').write_text(json.dumps(data,indent=2));print(json.dumps(data),flush=True)
    status('waiting_for_current_matrix')
    while True:
        try:finished=(previous/'matrix_status.json').exists() and json.loads((previous/'matrix_status.json').read_text()).get('state')=='complete'
        except json.JSONDecodeError:finished=False
        if finished:break
        try:os.kill(args.wait_pid,0)
        except ProcessLookupError:raise RuntimeError('The matrix process exited without a complete result; inspect before restarting')
        time.sleep(20)
    cfg=json.loads((collection/'collection_config.json').read_text())
    summary=json.loads((selection/'semantic_selection_summary.json').read_text())
    if summary['partial']:raise ValueError('Use complete-pilot scores, even when feature parameters came from an earlier frozen snapshot')
    common=[sys.executable,'-u','-m','experiments.pi05_sae_semantic.map_concept_ffn',
        '--collection',str(collection),'--selection',str(selection),'--assets',str(assets),
        '--checkpoint',cfg['checkpoint'],'--device',cfg['device'],'--threads','32','--evidence-level','l5_calibrated']
    single=output/'single_frozen'
    status('frozen_single_module_matrix_running')
    subprocess.run([*common,'--output',str(single),'--max-observations','12'],check=True)
    single_status=json.loads((single/'matrix_status.json').read_text())
    if single_status['state']!='complete':
        status('no_supported_offline_concepts',reason=single_status);return
    with (single/'concept_ffn_matrix.csv').open() as stream:rows=list(csv.DictReader(stream))
    selected,ranked=propose_masks(rows,args.max_masks,args.diversify_proposals)
    proposals=dict(selection=str(selection),single_matrix=str(single),selected=selected,ranked_pairs=ranked,
        diversified=args.diversify_proposals,
        interpretation='Sums of single-module Q90 values are a proposal heuristic, not the measured combination effect or a bound')
    (output/'candidate_masks.json').write_text(json.dumps(proposals,indent=2))
    if not selected:status('no_supported_pair_proposals');return
    masks=[value for candidate in selected for value in ('--mask',','.join(map(str,candidate['mask'])))]
    for split,n in [('discovery',12),('calibration',128)]:
        status(f'{split}_combination_matrix_running',masks=[c['mask'] for c in selected])
        subprocess.run([*common,'--output',str(output/f'combination_{split}'),'--split',split,
                        '--max-observations',str(n),*masks],check=True)
    with (output/'combination_calibration/concept_ffn_matrix.csv').open() as stream:calibration=list(csv.DictReader(stream))
    definitions=json.loads((selection/'concept_features.json').read_text())
    def mask_table(delta):
        catalog=[]
        for candidate in selected:
            key=','.join(map(str,candidate['mask']))
            evidence=[r for r in calibration if r['mask']==key]
            support={r['concept']:dict(D=float(r['D']),E=float(r['E']),observations=int(r['observations']),
                episodes=int(r['episodes']),late_baseline_evidence_episodes=int(r['late_baseline_evidence_episodes']))
                for r in evidence if r['routing_supported']=='True' and r['D'] and float(r['D'])<=delta}
            if support:catalog.append(dict(mask=candidate['mask'],supported_concepts=support))
        return catalog
    catalog=mask_table(args.delta_semantic)
    import numpy as np
    observation_rows=json.loads((selection/'score_observations.json').read_text())
    scores=np.load(selection/'semantic_scores.npz',allow_pickle=False)
    token_pool=summary.get('token_pool','all')
    early={name:item for name,item in definitions.items() if item['layers']['5']['supported']}
    def replay_coverage(catalog):
        eligible=after_refresh=0;total=0
        for i,row in enumerate(observation_rows):
            if row['split']!='calibration':continue
            total+=1
            active={name for name,item in early.items() if scores[f"L5_{item['category']}_{item['concept']}_{token_pool}"][i]
                    >=item['layers']['5']['theta']}
            can_skip=bool(active and any(active<=set(mask['supported_concepts']) for mask in catalog))
            eligible+=can_skip
            request=row.get('request',0)
            after_refresh+=bool(can_skip and request>=2 and (request+1)%10!=0)
        return dict(calibration_observations=total,eligible_before_refresh_rules=eligible,
                    eligible_after_refresh_rules=after_refresh,scope='Fixed first/last observation replay, not free-running policy coverage')
    rules=dict(state='candidate_rules_need_quality_evaluation' if catalog else 'no_supported_rule_at_default_delta',
        delta_semantic=args.delta_semantic,masks=catalog,fallback='FULL',first_full_calls=2,periodic_full_every_calls=10,
        early_concepts={name:item['layers']['5'] for name,item in definitions.items() if item['layers']['5']['supported']},
        selection=str(selection),calibration_matrix=str(output/'combination_calibration'),
        closed_loop_evaluated=False,gpu_policy_latency_measured=False,
        threshold_sweep=[dict(delta=d,eligible_masks=len(mask_table(d)),**replay_coverage(mask_table(d))) for d in (.1,.25,.5,1.)],
        replay_coverage=replay_coverage(catalog),
        interpretation='Conditional L5-cohort support and measured combination change are necessary priors, not action-preservation guarantees')
    (output/'skip_rules.json').write_text(json.dumps(rules,indent=2))
    status('offline_mask_study_complete',candidate_masks=len(selected),eligible_masks=len(catalog))


if __name__=='__main__':
    main()
