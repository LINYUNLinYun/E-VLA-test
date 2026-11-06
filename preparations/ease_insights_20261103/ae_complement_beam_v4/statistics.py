"""Frozen ten-task inference; policy-noise replicates stay inside each task."""
import itertools
import math
import numpy as np

SEED=20261111
REPLICATES=20000


def task_counts(values,task_ids):
    raw=np.asarray(values);tasks=sorted(set(task_ids))
    if (len(values)!=60 or len(task_ids)!=60 or tasks!=list(range(10))
        or any(task_ids.count(t)!=6 for t in tasks) or not np.isin(raw,[-1,0,1]).all()):
        raise ValueError('Need ten tasks with six paired binary differences per task')
    return [int(raw[[i for i,t in enumerate(task_ids) if t==task]].sum()) for task in tasks]


def task_signflip(counts):
    if len(counts)!=10 or any(type(v) is not int or abs(v)>6 for v in counts):
        raise ValueError('Need ten integer six-observation task sums')
    observed=sum(counts)
    extreme=sum(sum(s*v for s,v in zip(signs,counts))>=observed
        for signs in itertools.product((-1,1),repeat=10))
    return dict(p_one_sided=extreme/1024,extreme_assignments=extreme,assignments=1024,
        observed_mean_difference=observed/60,task_difference_counts=counts,
        task_mean_differences=[v/6 for v in counts],ties_included=True,
        null='Ten whole-task paired differences are sign-exchangeable; six init/noise observations stay within each task')


def bootstrap(values,task_ids,replicates=REPLICATES):
    counts=task_counts(values,task_ids);raw=np.asarray(values,np.int64)
    cases=np.random.default_rng(SEED).choice(raw,size=(replicates,60),replace=True).sum(axis=1)/60
    clusters=np.random.default_rng(SEED).choice(np.asarray(counts,np.int64),size=(replicates,10),replace=True).sum(axis=1)/60
    def interval(draws,unit):
        return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')),
            seed=SEED,replicates=replicates,unit=unit,method='percentile',quantile_method='linear')
    return dict(case=interval(cases,'paired rollout,descriptive sensitivity only'),
        task_cluster=dict(interval(clusters,'whole task with3 official states and2 policy-noise streams'),clusters=10))


def holm(values):
    if len(values)!=2 or any(not 0<=v<=1 for v in values):raise ValueError('Exactly two frozen hypotheses required')
    order=sorted(range(2),key=lambda i:(values[i],i));out=[0.,0.];previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(2-rank)*values[i]));out[i]=previous
    return out


def case_p(values):
    a=sum(v==1 for v in values);b=sum(v==-1 for v in values);n=a+b
    return sum(math.comb(n,k) for k in range(a,n+1))/2**n if n else 1.


def direction_inference(values,task_ids):
    counts=task_counts(values,task_ids);intervals=bootstrap(values,task_ids)
    loo=[dict(task=t,difference=(sum(counts)-v)/54) for t,v in enumerate(counts)]
    return dict(primary_task_block=task_signflip(counts),primary_holm2_p=None,
        sensitivity_case_mcnemar_p=case_p(values),sensitivity_holm2_p=None,
        confidence_intervals=intervals,leave_one_task_out=loo,positive_direction=sum(values)>0,
        every_leave_one_task_out_positive=all(x['difference']>0 for x in loo),
        task_cluster_lower_bound_positive=intervals['task_cluster']['low']>0,direction_reproduced=None)


def apply_verdict(directions,eligibility):
    if len(directions)!=2 or set(eligibility)!={'vlm_c','ae14_d3'}:raise ValueError('Fixed pair and two directions required')
    primary=holm([d['inference']['primary_task_block']['p_one_sided'] for d in directions])
    sensitivity=holm([d['inference']['sensitivity_case_mcnemar_p'] for d in directions])
    for d,p,sp in zip(directions,primary,sensitivity):
        info=d['inference'];info.update(primary_holm2_p=p,sensitivity_holm2_p=sp)
        info['direction_reproduced']=bool(info['positive_direction'] and p<.05
            and info['task_cluster_lower_bound_positive'] and info['every_leave_one_task_out_positive'])
    return 'complementarity_reproduced' if all(d['inference']['direction_reproduced'] for d in directions) and all(eligibility.values()) else 'not_reproduced'
