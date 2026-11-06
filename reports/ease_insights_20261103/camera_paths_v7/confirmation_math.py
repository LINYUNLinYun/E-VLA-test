"""Prospective five-state task-cluster arithmetic;never loads real outcomes."""
import itertools
import numpy as np
SEED=20261121
REPLICATES=20000


def task_signflip(counts):
    if len(counts)!=10 or any(type(v) not in (int,np.int64,np.int32) or abs(v)>5 for v in counts):
        raise ValueError('Need ten integer task differences,each containing five paired states')
    observed=sum(counts)
    extreme=sum(sum(s*v for s,v in zip(signs,counts))>=observed for signs in itertools.product((-1,1),repeat=10))
    return dict(p_one_sided=extreme/1024,extreme_assignments=extreme,assignments=1024,task_difference_counts=list(map(int,counts)),
        observed_mean_difference=observed/50,task_mean_differences=[v/5 for v in counts],ties_included=True,
        null='Task-level paired differences are sign-exchangeable;ten tasks are the inference units')


def bootstrap(values,task_ids):
    raw=np.asarray(values);tasks=sorted(set(task_ids))
    if len(values)!=50 or len(task_ids)!=50 or len(tasks)!=10 or any(task_ids.count(t)!=5 for t in tasks) or not np.isin(raw,[-1,0,1]).all():
        raise ValueError('Need ten tasks,five paired binary differences each')
    raw=raw.astype(np.int64)
    counts=np.asarray([raw[[i for i,t in enumerate(task_ids) if t==task]].sum() for task in tasks],np.int64)
    cases=np.random.default_rng(SEED).choice(raw,size=(REPLICATES,50),replace=True).sum(axis=1)/50
    clusters=np.random.default_rng(SEED).choice(counts,size=(REPLICATES,10),replace=True).sum(axis=1)/50
    def interval(draws,unit):
        return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')),
            seed=SEED,replicates=REPLICATES,unit=unit,method='percentile',quantile_method='linear')
    return dict(case=interval(cases,'paired case;descriptive sensitivity'),
        task_cluster=dict(interval(clusters,'whole task containing five initial states'),clusters=10))
