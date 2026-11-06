"""Task-paired suite differences and within-budget rank exchanges from existing D3 data."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json

import numpy as np

from scripts.experiment11.protocol import RESULTS,SUITES
from scripts.experiment11.analyze_failure_events import write_csv
from scripts.experiment11.shadow_risk_metrics import task_cluster_summary
from scripts.experiment10.common import dump_json,sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-id',default='round2_20261011')
    args=parser.parse_args()
    output=RESULTS/'analysis/next_round'/args.attempt_id/'task_type_results';output.mkdir(parents=True,exist_ok=True)
    source=RESULTS/'analysis/closed_loop_d3/episode_results.csv'
    candidates_path=RESULTS/'protocol/candidate_manifest.json'
    candidates=json.loads(candidates_path.read_text())['candidates']
    with source.open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=6200:raise RuntimeError('Expected complete original D3 matrix')
    full={(r['suite'],r['task_id'],r['episode_idx']):r for r in rows if r['variant_id']=='Full18'}
    if len(full)!=200:raise RuntimeError('Full coverage mismatch')
    candidates_by_id={r['variant_id']:r for r in candidates}
    ids=list(candidates_by_id)+['V13_Custom']
    task_rows,summary=[],[]
    for v in ids:
        data=[r for r in rows if r['variant_id']==v]
        if len(data)!=200 or {(r['suite'],r['task_id'],r['episode_idx']) for r in data}!=set(full):
            raise RuntimeError(f'Candidate paired coverage invalid: {v}')
        family=candidates_by_id[v]['family'] if v in candidates_by_id else 'V13-reference'
        deleted=candidates_by_id[v]['deleted_layer_count'] if v in candidates_by_id else 5
        for suite in ('ALL',*SUITES):
            rs=[r for r in data if suite=='ALL' or r['suite']==suite]
            delta=np.array([int(r['success'])-int(full[(r['suite'],r['task_id'],r['episode_idx'])]['success']) for r in rs])
            tasks=np.array([f'{r["suite"]}::{r["task_id"]}' for r in rs]);strata=np.array([r['suite'] for r in rs])
            mean,ci,nt=task_cluster_summary(delta[:,None],tasks,strata=strata)
            summary.append({'variant_id':v,'family':family,'deleted_layers':deleted,'suite':suite,
                'episodes':len(rs),'successes':sum(int(r['success']) for r in rs),
                'success_rate':float(np.mean([int(r['success']) for r in rs])),
                'full_success_rate':float(np.mean([int(full[(r['suite'],r['task_id'],r['episode_idx'])]['success']) for r in rs])),
                'task_equal_success_delta_pp':float(mean[0]*100),'task_ci95_low_pp':float(ci[0,0]*100),
                'task_ci95_high_pp':float(ci[1,0]*100),'new_failures':int((delta<0).sum()),
                'recovered_full_failures':int((delta>0).sum()),'n_tasks':nt,'exploratory':True})
        for suite in SUITES:
            for task in range(10):
                rs=[r for r in data if r['suite']==suite and int(r['task_id'])==task]
                d=[int(r['success'])-int(full[(suite,r['task_id'],r['episode_idx'])]['success']) for r in rs]
                task_rows.append({'variant_id':v,'family':family,'deleted_layers':deleted,'suite':suite,'task_id':task,
                                  'episodes':5,'successes':sum(int(r['success']) for r in rs),
                                  'paired_delta_pp':float(np.mean(d)*100),'new_failures':sum(x<0 for x in d),
                                  'recovered_full_failures':sum(x>0 for x in d)})
    ranks=[]
    for family in ('V16','V15','V14','V13'):
        for suite in ('ALL',*SUITES):
            rs=sorted([r for r in summary if r['family']==family and r['suite']==suite],
                      key=lambda r:(-r['success_rate'],r['variant_id']))
            for r in rs:
                ranks.append({**r,'rank_within_deletion_budget':1+sum(x['success_rate']>r['success_rate'] for x in rs),
                              'tie_rule':'equal_rate_equal_rank;id_only_for_display'})
    exchanges=[]
    for v in candidates_by_id:
        vr=[r for r in ranks if r['variant_id']==v]
        record={'variant_id':v,'family':candidates_by_id[v]['family']}
        record.update({r['suite']+'_rank':r['rank_within_deletion_budget'] for r in vr})
        record['suite_rank_range']=max(record[s+'_rank'] for s in SUITES)-min(record[s+'_rank'] for s in SUITES)
        exchanges.append(record)
    write_csv(output/'paired_task_outcomes.csv',task_rows)
    write_csv(output/'suite_paired_success.csv',summary)
    write_csv(output/'within_budget_ranks.csv',ranks)
    write_csv(output/'suite_rank_exchanges.csv',exchanges)
    write_csv(output/'suite_at_least_90_percent.csv',[r for r in summary if r['suite']!='ALL' and r['success_rate']>=.9])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=[r['variant_id'] for r in candidates]
    lookup={(r['variant_id'],r['suite']):r for r in summary}
    matrix=np.array([[lookup[v,s]['task_equal_success_delta_pp'] for s in SUITES] for v in ordered])
    fig,ax=plt.subplots(figsize=(9,11));im=ax.imshow(matrix,cmap='RdBu',vmin=-80,vmax=10,aspect='auto')
    ax.set_yticks(range(len(ordered)),ordered,fontsize=8);ax.set_xticks(range(4),SUITES,rotation=15)
    for i in range(len(ordered)):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]:+.0f}',ha='center',va='center',fontsize=7)
    ax.set_title('Paired success difference from Full18-D3 (percentage points)')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(output/'mask_suite_paired_delta.png',dpi=180);plt.close(fig)
    dump_json(output/'completion_audit.json',{'complete':True,'candidate_masks':29,'diagnostic_references':1,
        'source_episodes':6200,'candidate_task_rows':len(task_rows),'new_model_predictions':0,'new_environment_steps':0,
        'source_hashes':{str(source):sha256(source),str(candidates_path):sha256(candidates_path)},
        'bootstrap_unit':'task;overall_suite_stratified','bootstrap_replicates':2000,
        'warning':'same_exploratory_tasks;rank_exchange_not_single_layer_function_or_routing_validation',
        'outputs':{p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='completion_audit.json'}})
    print(json.dumps({'complete':True,'analysis':str(output),'candidate_masks':29}),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
