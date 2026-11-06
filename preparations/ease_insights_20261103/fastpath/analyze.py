"""Paired, predeclared suite comparisons; exploratory task summaries stay descriptive."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from preparations.idea1_v3_1.common import atomic_json, read_manifest
from .runner import CONDITIONS, cases, sha
from .worker import completed, episode_dir

A, B = CONDITIONS[1:]


def exact_mcnemar_greater(wins, losses):
    """One-sided exact conditional McNemar: desired wins exceed losses."""
    n=wins+losses
    return sum(math.comb(n,k) for k in range(wins,n+1))/(2**n) if n else 1.


def paired_statistics(records, better, worse, bootstrap=20000, seed=1103):
    """One record per matched official task/init; bootstrap preserves pairs."""
    if not records: return dict(n=0)
    delta=np.array([int(r[better])-int(r[worse]) for r in records],dtype=float)
    wins=int((delta>0).sum()); losses=int((delta<0).sum())
    tasks=sorted({r['task'] for r in records})
    task_rows=[]
    for task in tasks:
        rows=[r for r in records if r['task']==task]
        ds=[int(r[better])-int(r[worse]) for r in rows]
        task_rows.append(dict(task=task,n=len(rows),better_successes=sum(r[better] for r in rows),
            worse_successes=sum(r[worse] for r in rows),difference=float(np.mean(ds)),
            wins=sum(x>0 for x in ds),losses=sum(x<0 for x in ds)))
    rng=np.random.default_rng(seed)
    # This interval treats paired episodes as independent; task clustering below
    # is mandatory sensitivity reporting because each task has repeated inits.
    sampled=delta[rng.integers(len(delta),size=(bootstrap,len(delta)))].mean(axis=1)
    pair_ci=np.quantile(sampled,[.025,.975]).tolist()
    td=np.array([r['difference'] for r in task_rows])
    cluster_sample=td[rng.integers(len(td),size=(bootstrap,len(td)))].mean(axis=1)
    cluster_ci=np.quantile(cluster_sample,[.025,.975]).tolist()
    # Ten task labels are the randomization unit. With unequal incomplete
    # task counts this sensitivity test concerns the equal-task mean only.
    sign_values=np.array([np.dot(signs,td)/len(td) for signs in itertools.product((-1.,1.),repeat=len(td))])
    task_p=float(np.mean(sign_values>=td.mean()-1e-12))
    init_rows=[]
    for init in sorted({r['init'] for r in records}):
        rows=[r for r in records if r['init']==init]
        init_rows.append(dict(init=init,n=len(rows),better_successes=sum(r[better] for r in rows),
            worse_successes=sum(r[worse] for r in rows),difference=float(np.mean([int(r[better])-int(r[worse]) for r in rows]))))
    return dict(n=len(records),better=better,worse=worse,better_successes=sum(r[better] for r in records),
        worse_successes=sum(r[worse] for r in records),difference=float(delta.mean()),wins=wins,losses=losses,
        both_success=sum(r[better] and r[worse] for r in records),both_failure=sum(not r[better] and not r[worse] for r in records),
        exact_one_sided_mcnemar_p=exact_mcnemar_greater(wins,losses),paired_episode_bootstrap_ci95=pair_ci,
        equal_task_difference=float(td.mean()),task_cluster_bootstrap_ci95=cluster_ci,task_sign_flip_one_sided_p=task_p,
        tasks_better=sum(r['difference']>0 for r in task_rows),tasks_worse=sum(r['difference']<0 for r in task_rows),
        tasks_tied=sum(r['difference']==0 for r in task_rows),tasks=task_rows,inits=init_rows,
        bootstrap_samples=bootstrap,bootstrap_seed=seed,
        caveat="Episode McNemar and paired bootstrap assume independent episodes; task-cluster bootstrap and task sign-flip show sensitivity to repeated initial states within a task. Bootstrap intervals are percentile intervals, not exact intervals.")


def load_records(output,manifest,stage,verify):
    expected=cases(manifest,stage)
    records={cid:{} for cid in CONDITIONS}
    for cid in CONDITIONS:
        for scene in expected:
            d=episode_dir(output,cid,scene)
            if not (d/'episode_receipt.json').exists(): continue
            receipt=json.loads((d/'episode_receipt.json').read_text())
            if receipt['manifest_hash']!=manifest['manifest_hash'] or receipt['condition']!=cid or receipt['case']!=scene:
                raise ValueError(f'Wrong episode receipt identity: {d}')
            if verify and not completed(output,cid,scene,manifest['manifest_hash']): continue
            result=json.loads((d/'result.json').read_text())
            if not result.get('valid') or not receipt.get('valid'): continue
            if result['success']!=receipt['success']: raise ValueError(f'Outcome differs from receipt: {d}')
            records[cid][(scene['suite'],scene['task'],scene['init'])]=dict(receipt=receipt,result=result)
    return expected,records


def analyze(output,stage='confirm',verify=False,bootstrap=20000):
    output=Path(output).resolve(); manifest=read_manifest(output/'manifest.json')
    expected,records=load_records(output,manifest,stage,verify)
    summaries={}
    for cid,items in records.items():
        rows=list(items.values()); n=len(rows); predictions=sum(r['result']['predictions'] for r in rows)
        summaries[cid]=dict(n=n,expected=len(expected),successes=sum(r['result']['success'] for r in rows),
            success_rate=sum(r['result']['success'] for r in rows)/n if n else None,
            actions=sum(r['result']['actions'] for r in rows),predictions=predictions,
            mean_model_request_ms=sum(r['receipt']['latency']['model_request_ms_total'] for r in rows)/predictions if predictions else None,
            sum_episode_wall_s=sum(r['result']['wall_s'] for r in rows),
            caveat="Model latency uses each condition's own closed-loop observations and excludes service preparation; it is an operational comparison, not same-input speedup.")
    matched=[]
    for key in sorted(set(records[A])&set(records[B])):
        left,right=records[A][key],records[B][key]
        left_gate,right_gate=left['receipt']['initial_gate'],right['receipt']['initial_gate']
        if any(left_gate[k]!=right_gate[k] for k in ('manifest_hash','case','input_hash','seed')):
            raise ValueError(f'Unpaired input/noise identity: {key}')
        if left['result']['init_vector_sha256']!=right['result']['init_vector_sha256']:
            raise ValueError(f'Unpaired official init vector: {key}')
        row=dict(suite=key[0],task=key[1],init=key[2],**{A:left['result']['success'],B:right['result']['success']})
        if key in records['pi0_full']:
            full=records['pi0_full'][key]
            if any(full['receipt']['initial_gate'][k]!=left_gate[k] for k in ('manifest_hash','case','input_hash','seed')):
                raise ValueError(f'Full reference initial identity mismatch: {key}')
            row['pi0_full']=full['result']['success']
        matched.append(row)
    suites={}; hypotheses=[]
    for index,suite in enumerate(manifest['suites']):
        rows=[r for r in matched if r['suite']==suite]
        declared=next((h for h in manifest['preregistered_hypotheses'] if h['suite']==suite),None)
        better,worse=(declared['better'],declared['worse']) if declared else (A,B)
        stats=paired_statistics(rows,better,worse,bootstrap,1103+index)
        stats.update(suite=suite,predeclared=bool(declared),complete=len(rows)==sum(s['suite']==suite for s in expected),
            direction='predeclared' if declared else 'descriptive A minus B')
        suites[suite]=stats
        if declared: hypotheses.append(stats)
    complete=all(len(records[c])==len(expected) for c in CONDITIONS)
    directional=all(h.get('difference',0)>0 for h in hypotheses)
    stable=all(h.get('paired_episode_bootstrap_ci95',[-1])[0]>0 and h.get('task_cluster_bootstrap_ci95',[-1])[0]>0 for h in hypotheses)
    if not complete: interpretation='Incomplete matrix: provisional descriptive evidence; no confirmation claim.'
    elif stage=='pilot': interpretation='Pilot complete; execute the frozen full confirmation regardless of observed direction.'
    elif directional and stable: interpretation='Both predeclared directions reproduced, with positive episode and task-cluster intervals; this supports task-dependent complementary paths within this exploratory matrix.'
    elif directional: interpretation='Both predeclared directions reproduced, but uncertainty or task clustering prevents a strong complementarity claim.'
    else: interpretation='The predeclared two-way preference did not reproduce. Do not infer a need for adaptive selection from this matrix.'
    full_comparisons={}
    for cid in (A,B):
        keys=sorted(set(records[cid])&set(records['pi0_full']))
        fast=sum(records[cid][k]['result']['success'] for k in keys); full=sum(records['pi0_full'][k]['result']['success'] for k in keys)
        full_comparisons[cid]=dict(n=len(keys),fast_successes=fast,full_successes=full,
            relative_success_retention=fast/full if full else None,
            success_difference=(fast-full)/len(keys) if keys else None)
    runs=[]
    for path in output.glob('*/runs/*/result.json'):
        row=json.loads(path.read_text())
        if row.get('manifest_hash')==manifest['manifest_hash']:
            row=dict(row,path=str(path)); prep=path.parent/'preparation.json'
            if prep.exists(): row['preparation']=json.loads(prep.read_text())
            runs.append(row)
    return dict(manifest_hash=manifest['manifest_hash'],stage=stage,complete=complete,artifact_hashes_verified=verify,
        conditions=summaries,paired_suites=suites,full_comparisons=full_comparisons,paired_cases=matched,
        interpretation=interpretation,runs=runs,analysis_source_sha256=sha(__file__),
        method_note='Directions/candidates/cases were frozen before validation. One-sided exact McNemar and two-sided 95% paired percentile intervals report effect and uncertainty; equal-task cluster bootstrap and exact task sign-flip are sensitivity analyses. Strong wording additionally requires complete data and both positive cluster intervals. This wording rule is an analysis convention, not an additional preregistered statistical test.',
        throughput_note='Per-run wall time includes workers and serial model queue; sum of concurrent run walls is resource time, not end-to-end makespan. Failed/incomplete run wall time must be counted separately for total experiment cost.')


def markdown(result):
    lines=['# EASE fast-path independent confirmation','',result['interpretation'],'',
        f"Stage: `{result['stage']}`. Full matrix complete: **{result['complete']}**. Artifact hashes verified: **{result['artifact_hashes_verified']}**.",'',
        '| Condition | Successes | Mean completed model request (ms) |','|---|---:|---:|']
    for cid,row in result['conditions'].items():
        latency=f"{row['mean_model_request_ms']:.2f}" if row['mean_model_request_ms'] is not None else '—'
        lines.append(f"| {cid} | {row['successes']}/{row['n']} (expected {row['expected']}) | {latency} |")
    lines+=['','| Suite and direction | N | Difference (pp) | Paired 95% CI (pp) | Task-cluster 95% CI (pp) | Wins/losses | One-sided McNemar p | Tasks +/−/= |','|---|---:|---:|---|---|---:|---:|---|']
    for suite,row in result['paired_suites'].items():
        if not row['n']: continue
        ci=lambda key: ', '.join(f'{100*x:.1f}' for x in row[key])
        direction=('A > B' if row['better']==A else 'B > A')+(' (predeclared)' if row['predeclared'] else ' (descriptive)')
        lines.append(f"| {suite}: {direction} | {row['n']} | {100*row['difference']:+.1f} | [{ci('paired_episode_bootstrap_ci95')}] | [{ci('task_cluster_bootstrap_ci95')}] | {row['wins']}/{row['losses']} | {row['exact_one_sided_mcnemar_p']:.4g} | {row['tasks_better']}/{row['tasks_worse']}/{row['tasks_tied']} |")
    lines+=['',f'A = `{A}`; B = `{B}`. Positive differences follow each row’s stated direction.','',result['method_note'],'',result['throughput_note'],'',
        'Request latency compares each policy’s own closed-loop observations. Service preparation, cold compilation and 20 warmup requests are separate in per-run records. A paired same-input benchmark is needed to isolate compute speedup.','',
        'Per-task and per-init effect counts, paired case outcomes, Full retention, and complete per-run timing are included in the JSON.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',required=True,help='Run directory containing frozen manifest')
    parser.add_argument('--stage',choices=('pilot','confirm'),default='confirm')
    parser.add_argument('--report',required=True,help='Output JSON path; sibling Markdown and paired-case CSV are also written')
    parser.add_argument('--verify',action='store_true'); parser.add_argument('--bootstrap',type=int,default=20000)
    args=parser.parse_args()
    if args.bootstrap<1000: parser.error('Use at least 1000 paired bootstrap samples')
    result=analyze(args.output,args.stage,args.verify,args.bootstrap)
    path=Path(args.report); atomic_json(path,result); path.with_suffix('.md').write_text(markdown(result))
    with path.with_suffix('.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['suite','task','init',A,B,'pi0_full']); writer.writeheader(); writer.writerows(result['paired_cases'])
    print(json.dumps(dict(complete=result['complete'],interpretation=result['interpretation'],report=str(path)),indent=2))


if __name__=='__main__': main()
