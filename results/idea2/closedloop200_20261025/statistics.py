"""Prospective CPU-only paired statistics for the fixed 4x10x5 LIBERO grid.

Input: JSON list, or {"cases": [...]}; fields configuration (or label/arm), suite,
task, init, status="valid", success (JSON bool). Optional provenance is retained
by the runner; statistics does not turn unverified receipts into valid cases.
No Torch/GPU imports. Task failures remain observations; missing/technical cases
prevent final inference. --self-test runs deterministic hand-checkable tests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
ARMS = ("fresh18", "sm16_tail0", "sm32_tail0", "sm32_tail2")
SEED = 20261025
DRAWS = 10000


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
    """Resample tasks, retaining all paired inits; ratio handles4/5-init units."""
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
        replicates.append(sum(total for total,n in chosen) / sum(n for total,n in chosen))
    return [quantile(replicates,.025), quantile(replicates,.975)]


def holm(pvalues):
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    result, running = {}, 0.0
    for i, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * pvalues[key]))
        result[key] = running
    return result


def paired_summary(reference, candidate, keys, *, draws=DRAWS, seed=SEED, allowed_task_sizes=(5,)):
    pairs = [(key, int(candidate[key]["success"]) - int(reference[key]["success"])) for key in keys]
    b, c = sum(v == 1 for _, v in pairs), sum(v == -1 for _, v in pairs)
    n = len(pairs)
    tasks = {}
    for key, value in pairs:
        tasks.setdefault(key[:2], []).append(value)
    sizes = {len(values) for values in tasks.values()}
    if not sizes.issubset(set(allowed_task_sizes)):
        raise ValueError("unexpected number of paired inits in a task")
    episode_values = [value for _, value in pairs]
    cluster = task_cluster_bootstrap_ci(tasks, draws, seed)
    episode = mean_bootstrap_ci(episode_values, draws, seed + 1)
    return dict(n=n, tasks=len(tasks), task_sizes=sorted(len(v) for v in tasks.values()), candidate_only_success_b=b, reference_only_success_c=c,
                both_success=sum(candidate[key]["success"] and reference[key]["success"] for key in keys),
                both_failure=sum(not candidate[key]["success"] and not reference[key]["success"] for key in keys),
                paired_success_rate_difference=(b-c)/n,
                exact_mcnemar_two_sided_p=exact_mcnemar(b,c),
                task_cluster_percentile95=cluster, episode_percentile95_sensitivity=episode,
                bootstrap_degenerate_task=(cluster[0] == cluster[1]),
                no_discordance=(b+c == 0),
                no_discordance_warning=("Zero discordances give degenerate empirical bootstrap; this is not proof of equivalence."
                                        if b+c == 0 else None),
                zero_discordance_iid_binomial_two_sided95_upper=(1 - .025 ** (1/n) if b+c == 0 else None),
                descriptive_harm_thresholds={str(pp): dict(
                    margin_percentage_points=pp,
                    task_ci_lower_above_minus_margin=(cluster[0] > -pp/100 if cluster[0] != cluster[1] else None),
                    episode_ci_lower_above_minus_margin=(episode[0] > -pp/100 if episode[0] != episode[1] else None),
                    caution="Descriptive interval comparison only; no user-approved noninferiority margin or SLO. Degenerate bootstrap is not a guarantee."
                ) for pp in (1,2,5)})



def _tables(selected, expected, complete, *, draws, seed, new_only=False):
    configurations=[]
    for arm in ARMS:
        for scope in ("overall",*SUITES):
            targets = expected if scope == "overall" else {key for key in expected if key[0]==scope}
            valid=[row for key,row in selected[arm].items() if key in targets and row.get("status")=="valid" and type(row.get("success")) is bool]
            k=sum(row["success"] for row in valid);n=len(valid)
            configurations.append(dict(configuration=arm,scope=scope,expected=len(targets),valid_cases=n,
                successes=k,task_failures=n-k,success_rate=k/n if n else None,wilson95=wilson(k,n),
                interval_scope="Nominal episode-binomial descriptive interval; inits within a task may be correlated.",
                provisional=not complete))
    comparisons=[]
    if complete:
        for j,candidate in enumerate(ARMS):
            for reference in ARMS[:j]:
                family="primary_vs_fresh" if reference=="fresh18" else "exploratory_direct"
                for scope in ("overall",*SUITES):
                    keys=sorted(expected if scope=="overall" else (key for key in expected if key[0]==scope))
                    result=paired_summary(selected[reference],selected[candidate],keys,draws=draws,seed=seed,allowed_task_sizes=(4,5) if new_only else (5,))
                    comparisons.append(dict(reference=reference,candidate=candidate,scope=scope,family=family,**result))
        primary={r["candidate"]:r["exact_mcnemar_two_sided_p"] for r in comparisons if r["family"]=="primary_vs_fresh" and r["scope"]=="overall"}
        adjusted=holm(primary)
        for row in comparisons:
            row["holm_p_primary_three_overall"]=adjusted[row["candidate"]] if row["family"]=="primary_vs_fresh" and row["scope"]=="overall" else None
    return configurations,comparisons

def analyze(rows, *, draws=DRAWS, seed=SEED):
    expected = {(suite,task,init) for suite in SUITES for task in range(10) for init in range(5)}
    selected = {arm:{} for arm in ARMS}
    duplicate, unknown, invalid = [], [], []
    for index, row in enumerate(rows):
        arm = row.get("configuration", row.get("label", row.get("arm")))
        aliases = [row[name] for name in ("configuration","label","arm") if name in row]
        if len(set(aliases)) > 1:
            unknown.append(dict(index=index,reason="contradictory arm aliases")); continue
        key = (row.get("suite"),row.get("task"),row.get("init"))
        if arm not in ARMS or key not in expected:
            unknown.append(dict(index=index,configuration=arm,key=key)); continue
        if key in selected[arm]:
            duplicate.append(dict(index=index,configuration=arm,key=key)); continue
        selected[arm][key] = row
        if row.get("status") != "valid" or type(row.get("success")) is not bool:
            invalid.append(dict(configuration=arm,key=key,status=row.get("status")))
    missing = {arm:[list(key) for key in sorted(expected-set(selected[arm]))] for arm in ARMS}
    complete = not (duplicate or unknown or invalid or any(missing.values()))
    configurations,comparisons = _tables(selected,expected,complete,draws=draws,seed=seed)
    pilot = {key for key in expected if key[1] in (0,5) and key[2]==0}
    fresh = expected-pilot
    secondary = dict(status="pending_complete800",expected_cases_per_arm=192,
        excluded_pilot_keys=[list(key) for key in sorted(pilot)],
        role="Secondary sensitivity excluding the8 known pilot cases per arm; not a second independent trial.",
        configurations=[],comparisons=[])
    if complete:
        conf,comp = _tables(selected,fresh,True,draws=draws,seed=seed,new_only=True)
        secondary.update(status="complete",configurations=conf,comparisons=comp,
            excluded_pilot_successes={arm:sum(selected[arm][key]["success"] for key in pilot) for arm in ARMS},
            cluster_rule="Resample40 tasks with all retained4/5 init pairs; replicate difference=sum(deltas)/sum(episodes). Per-suite10 tasks/48 episodes.",
            episode_rule="Resample192 paired episodes (48 per suite).")
    return dict(status="complete" if complete else "incomplete_or_invalid",statistical_plan_version=2,
        expected_total_cases=800,received_rows=len(rows),missing=missing,duplicates=duplicate,unknown=unknown,invalid=invalid,
        configurations=configurations,comparisons=comparisons,new192_sensitivity=secondary,
        method=dict(seed=seed,bootstrap_draws=draws,cluster="Globally resample40 (suite,task) units; keep all five paired inits. Per-suite resamples10 tasks.",
            episode_sensitivity="Resample200 paired episodes (50 per-suite); do not resample arms independently.",
            confidence="95% percentile bootstrap, linear interpolation at (B-1)*p",wilson_z=1.959963984540054,
            exact_test="Two-sided conditional binomial McNemar; double smaller tail, capped1; no mid-p.",
            multiplicity="Holm correction for three overall candidate-vs-fresh tests. Direct and per-suite analyses exploratory.",
            pairing="suite/task/init; seed/model/checkpoint/protocol compatibility must be audited separately."),
        limitations=["p>0.05 does not establish no effect or noninferiority.",
            "No user-approved noninferiority margin;1/2/5pp flags are descriptive, not pass/fail SLOs.",
            "Exact McNemar is conditional under independent episode discordances; within-task dependence limits population interpretation. Task-cluster bootstrap is the main uncertainty sensitivity.",
            "Only40 task clusters; sparse discordances and ceiling success can give degenerate/unreliable bootstrap intervals.",
            "Fixed selected tasks/inits and one model seed do not establish overall LIBERO or robot-population quality.",
            "Valid task failures remain outcomes. Technical failures are separate and prevent completion. Repeated attempts cannot replace a valid failed episode."])


def self_test():
    checks=[]
    def check(value,label):
        if not value:raise AssertionError(label)
        checks.append(label)
    for b,c,expected in [(0,0,1),(1,0,1),(2,0,.5),(5,0,.0625),(6,0,.03125),(3,1,.625),(5,5,1)]:
        check(exact_mcnemar(b,c)==expected,f"exact McNemar {b}/{c}={expected}")
        check(exact_mcnemar(c,b)==expected,f"McNemar symmetry {b}/{c}")
    lo,hi=wilson(200,200);check(abs(lo-.9811546736227335)<1e-12 and abs(hi-1)<1e-12,"Wilson200/200")
    check(wilson(0,0) is None,"Wilson empty")
    check(quantile([0,10],.25)==2.5,"linear percentile")
    check(mean_bootstrap_ci([0]*40,100)==[0.,0.],"zero discordance bootstrap explicit")
    check(holm({'a':.01,'b':.04,'c':.03})=={'a':.03,'c':.06,'b':.06},"Holm monotonic")
    rows=[dict(configuration=arm,suite=suite,task=task,init=init,status='valid',success=True)
          for arm in ARMS for suite in SUITES for task in range(10) for init in range(5)]
    result=analyze(rows,draws=64);check(result['status']=='complete' and len(result['comparisons'])==30,"800-case complete grid")
    secondary=result['new192_sensitivity']
    check(secondary['status']=='complete' and len(secondary['comparisons'])==30,"new192 only after full800")
    subset=secondary['comparisons'][0]
    check(subset['n']==192 and subset['tasks']==40 and subset['task_sizes'].count(4)==8 and subset['task_sizes'].count(5)==32,"new192 unequal task sizes")
    check(task_cluster_bootstrap_ci({('s',0):[1]*4,('s',1):[0]*5},2000,7)==[0.0,1.0],"unequal cluster bootstrap exact support")
    changed=[dict(row) for row in rows];changed[-1]['success']=False
    result=analyze(changed,draws=64);pair=next(x for x in result['comparisons'] if x['candidate']=='sm32_tail2' and x['reference']=='fresh18' and x['scope']=='overall')
    check(pair['candidate_only_success_b']==0 and pair['reference_only_success_c']==1 and pair['paired_success_rate_difference']==-.005,"real failure retained; sign convention")
    sens=next(x for x in result['new192_sensitivity']['comparisons'] if x['candidate']=='sm32_tail2' and x['reference']=='fresh18' and x['scope']=='overall')
    check(sens['paired_success_rate_difference']==-1/192 and math.isclose(pair['paired_success_rate_difference'],sens['paired_success_rate_difference']*.96,abs_tol=1e-15),"pilot successes dilute delta by4percent")
    check(analyze(rows[:-1],draws=64)['comparisons']==[],"missing case suppresses final inference")
    check(analyze(rows+[rows[0]],draws=64)['status']=='incomplete_or_invalid',"duplicate cannot replace outcome")
    invalid=[dict(row) for row in rows];invalid[-1]['status']='technical_failure'
    check(analyze(invalid,draws=64)['status']=='incomplete_or_invalid',"technical failure is not task failure or success")
    invalid[-1]['status']='valid';invalid[-1]['success']=1
    check(analyze(invalid,draws=64)['status']=='incomplete_or_invalid',"success must be JSON bool")
    return dict(status='passed',checks=checks,gpu_launched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--draws',type=int,default=DRAWS)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:result=self_test()
    else:
        if args.input is None:parser.error('--input required unless --self-test')
        data=json.loads(args.input.read_text());rows=data['cases'] if isinstance(data,dict) else data
        result=analyze(rows,draws=args.draws,seed=args.seed)
        result['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
        reuse=Path(__file__).with_name('reuse32.json')
        result['pilot_reuse_evidence']=dict(path=str(reuse),exists=reuse.exists())
        if reuse.exists():
            proof=json.loads(reuse.read_text())
            result['pilot_reuse_evidence'].update(sha256=hashlib.sha256(reuse.read_bytes()).hexdigest(),
                recorded_status=proof.get('status'),selected_episodes=proof.get('selected_episodes'),
                scope='Receipt provenance reference; full compatibility audit is separate from statistical calculations.')
    result['statistics_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    text=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
    else:print(text,end='')


if __name__=='__main__':main()
