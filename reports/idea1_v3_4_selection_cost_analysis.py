"""Read-only aggregation of the frozen single-card experiment outputs."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/idea1_v3_4_selection_cost'


def read(name):
    with (OUT/name).open() as stream:return list(csv.DictReader(stream))


def stats(values):
    a=np.array(list(values),dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),min=float(a.min()),max=float(a.max()))


def main():
    timing=read('benchmark/timings.csv');memory=read('benchmark/memory.csv')
    setup=read('benchmark/preparation.csv');score=read('benchmark/scoring.csv')
    result=dict(batch={},scoring={},scope='8 fixed real inputs; raw paired measurements preserved')
    for batch in (1,2,4):
        rows=[r for r in timing if int(r['batch_size'])==batch]
        if not rows:continue
        peaks=[r for r in memory if int(r['batch_size'])==batch]
        first=[r for r in setup if int(r['batch_size'])==batch]
        paired=[]
        for r in rows:
            baseline=next(b for b in timing if b['batch_size']=='1' and (b['sample'],b['repeat'])==(r['sample'],r['repeat']))
            paired.append(float(baseline['measured_wall_ms'])-float(r['measured_wall_ms']))
        result['batch'][str(batch)]=dict(total_ms=stats(float(r['measured_wall_ms']) for r in rows),
            forward_ms=stats(float(r['selector_forward_ms']) for r in rows),
            backward_ms=stats(float(r['selector_backward_ms']) for r in rows),
            paired_saving_ms=stats(paired),
            first_call_s=stats(float(r['first_selector_call_s']) for r in first),
            setup_ms=stats(float(r['selector_setup_ms']) for r in first),
            peak_allocated_mib=max(float(r['peak_allocated_mib']) for r in peaks),
            peak_reserved_mib=max(float(r['peak_reserved_mib']) for r in peaks),
            any_oom=any(r['oom']=='True' for r in peaks))
    for k in (4,6):
        result['scoring'][str(k)]={aggregation:stats(float(r['scoring_ms']) for r in score
            if int(r['k'])==k and r['aggregation']==aggregation)
            for aggregation in ('layer_rms_sum','signed_sum_rms')}
    result['equivalence']=json.loads((OUT/'benchmark/complete.json').read_text())['equivalence']
    if (OUT/'summary.csv').exists():
        summary=read('summary.csv');comparisons=[]
        for row in summary:
            if row['variant']=='baseline':continue
            baseline=next(r for r in summary if r['base_id']==row['base_id'] and r['variant']=='baseline')
            comparisons.append(dict(condition=row['condition'],variant=row['variant'],
                success=int(row['success']),baseline_success=int(baseline['success']),n=int(row['n']),
                hot_policy_reduction_percent=100*(1-float(row['hot_policy_ms_per_request'])/float(baseline['hot_policy_ms_per_request'])),
                amortized_selector_reduction_percent=100*(1-float(row['selector_ms_per_request'])/float(baseline['selector_ms_per_request'])),
                trigger_full_ms=float(row['trigger_full_mean_ms']),
                trigger_request_p95_ms=float(row['trigger_request_p95_ms'])))
        result['closed_loop_comparisons']=comparisons
    if (OUT/'batch200_graph/complete.json').exists():
        result['graph_batch4']=json.loads((OUT/'batch200_graph/complete.json').read_text())
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'analysis_receipt.json').write_text(json.dumps(dict(script=str(Path(__file__).resolve()),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
