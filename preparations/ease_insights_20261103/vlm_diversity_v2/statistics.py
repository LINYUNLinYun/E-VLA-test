"""Prospective three-init confirmation bootstrap; tasks are the cluster unit."""
import numpy as np


def bootstrap(values,task_ids,seed_case,seed_task,replicates=20000):
    values=np.asarray(values,dtype=np.float64);tasks=sorted(set(task_ids))
    if len(values)!=30 or len(task_ids)!=30 or len(tasks)!=10 or any(task_ids.count(t)!=3 for t in tasks):
        raise ValueError('Confirmation needs ten tasks with three matched initial states each')
    rng=np.random.default_rng(seed_case)
    cases=rng.choice(values,size=(replicates,30),replace=True).mean(axis=1)
    means=np.asarray([values[[i for i,t in enumerate(task_ids) if t==task]].mean() for task in tasks])
    clusters=np.random.default_rng(seed_task).choice(means,size=(replicates,10),replace=True).mean(axis=1)
    def interval(draws,seed,unit):
        return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')),
            seed=seed,replicates=replicates,unit=unit,method='percentile',quantile_method='linear')
    return dict(case=interval(cases,seed_case,'paired case'),
                task_cluster=dict(interval(clusters,seed_task,'whole task with three initial states'),clusters=10))
