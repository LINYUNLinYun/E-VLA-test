"""Paired bootstrap: integer sums before division preserve the exact zero boundary."""
import numpy as np


def bootstrap(values,task_ids,seed_case,seed_task,replicates=20000):
    tasks=sorted(set(task_ids));raw=np.asarray(values)
    if (len(values)!=30 or len(task_ids)!=30 or len(tasks)!=10 or any(task_ids.count(t)!=3 for t in tasks)
        or not np.isin(raw,[-1,0,1]).all()):raise ValueError('Need ten tasks,three paired binary-outcome differences per task')
    values=raw.astype(np.int64)
    cases=np.random.default_rng(seed_case).choice(values,size=(replicates,30),replace=True).sum(axis=1)/30
    counts=np.asarray([values[[i for i,t in enumerate(task_ids) if t==task]].sum() for task in tasks],np.int64)
    clusters=np.random.default_rng(seed_task).choice(counts,size=(replicates,10),replace=True).sum(axis=1)/30
    def interval(draws,seed,unit):
        return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')),
            seed=seed,replicates=replicates,unit=unit,method='percentile',quantile_method='linear')
    return dict(case=interval(cases,seed_case,'paired case'),
        task_cluster=dict(interval(clusters,seed_task,'whole task with three initial states'),clusters=10))
