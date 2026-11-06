"""CPU-only prospective statistics for continuous online Encoder closed-loop.

Fixed300-cell grid; no reuse of previous synchronous/lag experiments. JSON
input is a list or {"cases": [...]}, with configuration/suite/task/init/status/
success. One primary contrast, separate Encoder ablation; no noninferiority
margin. No Torch/model/GPU imports. Run --self-test before terminal analysis.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random

SUITES = ("libero_object", "libero_spatial", "libero_goal", "libero_10")
ARMS = ("fresh18_no_encoder", "sm16_no_encoder", "sm16_encoder")
SEED = 20261026
DRAWS = 10000
COMPARISONS = (
    ("fresh18_no_encoder", "sm16_encoder", "primary"),
    ("sm16_no_encoder", "sm16_encoder", "encoder_ablation"),
    ("fresh18_no_encoder", "sm16_no_encoder", "exploratory_mixed_KV_resource_control"),
)


def case_grid():
    return {(suite,task,init) for suite in SUITES for task in range(10)
            for init in range(3 if task % 2 == 0 else 2)}


def weighted_cluster_mean(units):
    denominator = sum(count for total,count in units)
    if denominator <= 0:
        raise ValueError("empty cluster sample")
    return sum(total for total,count in units) / denominator


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return None
    if not 0 <= successes <= n:
        raise ValueError("successes outside [0,n]")
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def exact_mcnemar(b, c):
    """Two-sided conditional exact binomial p, doubled smaller tail, not mid-p.

    b=candidate success/reference failure; c=reverse. n=b+c discordant pairs.
    """
    if not isinstance(b, int) or not isinstance(c, int) or min(b, c) < 0:
        raise ValueError("discordance counts must be nonnegative integers")
    n = b + c
    if not n:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / (2 ** n))


def quantile(values, p):
    values = sorted(values)
    if not values:
        raise ValueError("empty quantile input")
    index = (len(values) - 1) * p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def mean_bootstrap_ci(unit_values, draws=DRAWS, seed=SEED):
    """Resample equal-sized paired units; percentile95%, linear quantiles."""
    if draws <= 0 or not unit_values:
        raise ValueError("positive draws and at least one unit required")
    # Exact shortcut; explicitly flagged as degenerate by paired_summary.
    if min(unit_values) == max(unit_values):
        return [float(unit_values[0]), float(unit_values[0])]
    rng = random.Random(seed)
    n = len(unit_values)
    replicates = [sum(unit_values[rng.randrange(n)] for _ in range(n)) / n
                  for _ in range(draws)]
    return [quantile(replicates, .025), quantile(replicates, .975)]


def task_cluster_bootstrap_ci(tasks, draws=DRAWS, seed=SEED):
    """Resample tasks retaining2/3 paired inits; ratio preserves episode mean."""
    if draws <= 0 or not tasks or any(not values for values in tasks.values()):
        raise ValueError("positive draws and nonempty clusters required")
    units = [(sum(tasks[key]), len(tasks[key])) for key in sorted(tasks)]
    means = [total / n for total, n in units]
    if min(means) == max(means):
        return [float(means[0]), float(means[0])]
    rng = random.Random(seed)
    count = len(units)
    replicates = []
    for _ in range(draws):
        chosen = [units[rng.randrange(count)] for _ in range(count)]
        replicates.append(weighted_cluster_mean(chosen))
    return [quantile(replicates,.025), quantile(replicates,.975)]


def paired_summary(reference, candidate, keys, *, draws=DRAWS, seed=SEED):
    differences = {key:int(candidate[key]['success'])-int(reference[key]['success']) for key in keys}
    tasks = {}
    for key,value in differences.items():tasks.setdefault(key[:2],[]).append(value)
    for (suite,task),values in tasks.items():
        if len(values) != (3 if task % 2 == 0 else 2):
            raise ValueError('Incomplete task cluster')
    b=sum(v==1 for v in differences.values());c=sum(v==-1 for v in differences.values())
    cluster=task_cluster_bootstrap_ci(tasks,draws,seed)
    episode=mean_bootstrap_ci(list(differences.values()),draws,seed+1)
    n=len(keys)
    return dict(n=n,task_clusters=len(tasks),task_cluster_sizes=sorted(map(len,tasks.values())),
        candidate_only_success_b=b,reference_only_success_c=c,
        both_success=sum(reference[key]['success'] and candidate[key]['success'] for key in keys),
        both_failure=sum(not reference[key]['success'] and not candidate[key]['success'] for key in keys),
        paired_success_rate_difference=(b-c)/n,exact_mcnemar_two_sided_p=exact_mcnemar(b,c),
        task_cluster_percentile95=cluster,episode_percentile95_sensitivity=episode,
        bootstrap_degenerate_task=cluster[0]==cluster[1],bootstrap_degenerate_episode=episode[0]==episode[1],
        no_discordance=b+c==0,
        caution=('No discordance gives a degenerate empirical bootstrap, not proof of equivalence.' if b+c==0
                 else 'Conditional exact McNemar assumes independent episode discordances; task dependence limits population interpretation.'))


def analyze(rows, *, draws=DRAWS, seed=SEED):
    if draws<=0:raise ValueError('bootstrap draws must be positive')
    expected=case_grid();selected={arm:{} for arm in ARMS}
    unknown=[];duplicates=[];invalid=[]
    for index,row in enumerate(rows):
        names=[row[k] for k in ('configuration','label','arm') if k in row]
        arm=names[0] if names else None
        if len(set(names))>1:
            unknown.append(dict(row=index,reason='conflicting configuration aliases'));continue
        key=(row.get('suite'),row.get('task'),row.get('init'))
        if arm not in ARMS or key not in expected or type(key[1]) is not int or type(key[2]) is not int:
            unknown.append(dict(row=index,configuration=arm,key=key));continue
        if key in selected[arm]:
            duplicates.append(dict(row=index,configuration=arm,key=key));continue
        selected[arm][key]=row
        if row.get('status')!='valid' or type(row.get('success')) is not bool:
            invalid.append(dict(configuration=arm,key=key,status=row.get('status')))
    missing={arm:[list(key) for key in sorted(expected-set(selected[arm]))] for arm in ARMS}
    complete=not(unknown or duplicates or invalid or any(missing.values()))
    configurations=[];comparisons=[];casewise=[]
    for arm in ARMS:
        for scope in ('overall',*SUITES):
            keys=expected if scope=='overall' else {key for key in expected if key[0]==scope}
            valid=[row for key,row in selected[arm].items() if key in keys and row.get('status')=='valid' and type(row.get('success')) is bool]
            n=len(valid);k=sum(row['success'] for row in valid)
            configurations.append(dict(configuration=arm,scope=scope,expected_cases=len(keys),valid_cases=n,
                successes=k,task_failures=n-k,success_rate=k/n if n else None,wilson95=wilson(k,n),
                provisional=not complete,interval_scope='Nominal episode-binomial description; task-correlated outcomes need cluster sensitivity.'))
    if complete:
        for reference,candidate,role in COMPARISONS:
            for scope in ('overall',*SUITES):
                keys=sorted(expected if scope=='overall' else (key for key in expected if key[0]==scope))
                comparisons.append(dict(reference=reference,candidate=candidate,role=role,scope=scope,
                    confirmatory=(role=='primary' and scope=='overall'),
                    **paired_summary(selected[reference],selected[candidate],keys,draws=draws,seed=seed)))
            for suite,task,init in sorted(expected):
                ref=selected[reference][(suite,task,init)];cand=selected[candidate][(suite,task,init)]
                casewise.append(dict(reference=reference,candidate=candidate,role=role,suite=suite,task=task,init=init,
                    reference_success=ref['success'],candidate_success=cand['success'],
                    difference=int(cand['success'])-int(ref['success']),
                    outcome=('candidate_only_success' if cand['success'] and not ref['success'] else
                             'reference_only_success' if ref['success'] and not cand['success'] else
                             'both_success' if ref['success'] else 'both_failure'),
                    reference_steps=ref.get('steps'),candidate_steps=cand.get('steps'),
                    reference_directory=ref.get('directory'),candidate_directory=cand.get('directory')))
    return dict(status='complete' if complete else 'incomplete_or_invalid',plan_version=1,
        expected_total_cases=300,received_rows=len(rows),missing=missing,duplicates=duplicates,unknown=unknown,invalid=invalid,
        configurations=configurations,comparisons=comparisons,casewise_comparisons=casewise,
        cases=[dict(row,configuration=arm) for arm in ARMS for key,row in sorted(selected[arm].items())],
        method=dict(seed=seed,bootstrap_draws=draws,
            cluster='Resample40 (suite,task) clusters,20 with3 inits and20 with2; replicate=sum paired deltas/sum sampled episode counts. Within-suite resample10 clusters.',
            episode_sensitivity='Resample100 paired episodes overall or25 within-suite; preserve pairing.',
            confidence='95% percentile, linear interpolation at (draws-1)*p; Wilson z=1.959963984540054.',
            exact_test='Two-sided conditional binomial McNemar, doubled smaller tail; no mid-p.',
            multiplicity='One overall primary contrast: sm16_encoder vs fresh18_no_encoder. Other contrasts/suite results exploratory; not extra confirmatory tests.',
            design='New same-online-protocol baseline; all gate episodes count if technically valid; no reuse of previous synchronous or250mslag runs.'),
        limitations=['p>0.05 does not establish no effect, equivalence or noninferiority.',
            'No noninferiority margin or SLO pass/fail threshold is defined.',
            '100 episodes cannot guarantee1pp safety; only40 task clusters,10 per-suite, and sparse discordances can produce degenerate intervals.',
            'Conditional exact McNemar and Wilson depend on episode-independence assumptions; task clustering is the main uncertainty sensitivity.',
            'Valid task failures remain outcomes. Technical retries require a new directory, at most once; provenance audit is separate from statistics.',
            'Pairing matches initial conditions, not later free observations/actions; wall-time is not matched-input GPU speed.',
            'Outcome-based case selection, gate selection or repeated runs until success is forbidden.'])


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else value
                          for key,value in row.items()} for row in rows)


def self_test():
    checks=[]
    def check(value,label):
        if not value:raise AssertionError(label)
        checks.append(label)
    for b,c,answer in [(0,0,1),(1,0,1),(2,0,.5),(3,1,.625),(5,0,.0625),(6,0,.03125),(5,5,1)]:
        check(exact_mcnemar(b,c)==answer and exact_mcnemar(c,b)==answer,f'Exact symmetric McNemar {b}/{c}')
    check(abs(wilson(100,100)[0]-.9630065017930143)<1e-12,'Wilson100/100 lower96.30065%')
    check(wilson(0,0) is None,'No invented interval for zero cases')
    check(quantile([0,10],.25)==2.5,'Linear percentile')
    check(weighted_cluster_mean([(-2,2),(0,3)])==-.4,'Unequal2/3 task sizes use ratio, not mean(-1,0)')
    check(mean_bootstrap_ci([0]*100,10)==[0.,0.],'Zero-discordance bootstrap explicitly degenerate')
    grid=case_grid();check(len(grid)==100,'100-case grid')
    for suite in SUITES:
        check(sum(key[0]==suite for key in grid)==25,f'{suite}:25 cases')
    rows=[dict(configuration=arm,suite=suite,task=task,init=init,status='valid',success=True)
          for arm in ARMS for suite,task,init in sorted(grid)]
    result=analyze(rows,draws=64)
    check(result['status']=='complete' and len(result['comparisons'])==15 and len(result['casewise_comparisons'])==300,'Complete300 cells/15 statistics/300 paired outcomes')
    primary=next(r for r in result['comparisons'] if r['confirmatory'])
    check(sum(r['confirmatory'] for r in result['comparisons'])==1,'Exactly one overall primary test')
    check(primary['n']==100 and primary['task_clusters']==40 and primary['task_cluster_sizes'].count(2)==20 and primary['task_cluster_sizes'].count(3)==20,'40 ratio-weighted task clusters')
    changed=[dict(row) for row in rows];changed[-1]['success']=False
    result=analyze(changed,draws=64);primary=next(r for r in result['comparisons'] if r['confirmatory'])
    check(primary['candidate_only_success_b']==0 and primary['reference_only_success_c']==1 and primary['paired_success_rate_difference']==-.01,'Task failure retained and delta sign correct')
    check(analyze(rows[:-1],draws=16)['comparisons']==[],'Incomplete gate suppresses paired inference')
    check(analyze(rows+[rows[0]],draws=16)['status']=='incomplete_or_invalid','No duplicate cherry-picking')
    bad=[dict(row) for row in rows];bad[-1]['status']='technical_failure'
    check(analyze(bad,draws=16)['status']=='incomplete_or_invalid','Technical failure not silently scored')
    bad[-1]['status']='valid';bad[-1]['success']=1
    check(analyze(bad,draws=16)['status']=='incomplete_or_invalid','Success must be genuine JSON boolean')
    bad=[dict(row) for row in rows];bad[-1]['init']=2;bad[-1]['task']=9
    check(analyze(bad,draws=16)['status']=='incomplete_or_invalid','Odd-task init2 forbidden')
    check(all(not r['confirmatory'] for r in result['comparisons'] if r['role']=='encoder_ablation'),'Ablation not promoted to extra primary test')
    return dict(status='passed',gpu_launched=False,checks=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--csv-dir',type=Path);parser.add_argument('--draws',type=int,default=DRAWS)
    parser.add_argument('--seed',type=int,default=SEED);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:result=self_test()
    else:
        if args.input is None:parser.error('--input required unless --self-test')
        data=json.loads(args.input.read_text());result=analyze(data['cases'] if isinstance(data,dict) else data,draws=args.draws,seed=args.seed)
        result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    result['statistics_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')
    if args.csv_dir and not args.self_test:
        args.csv_dir.mkdir(parents=True,exist_ok=True)
        for name,rows in [('success_rates',result['configurations']),('paired_comparisons',result['comparisons']),
            ('casewise_comparisons',result['casewise_comparisons']),('cases',result['cases']),
            ('discordant_cases',[r for r in result['casewise_comparisons'] if r['difference']])]:
            write_csv(args.csv_dir/(name+'.csv'),rows)


if __name__=='__main__':main()
