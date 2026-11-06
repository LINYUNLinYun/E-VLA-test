"""Task-balanced offline error only; it cannot establish rollout quality."""
import numpy as np


def errors(actions):
    values = np.asarray(actions, dtype=np.float64)
    if values.shape != (3, 154, 50, 7) or not np.isfinite(values).all():
        raise ValueError('Need all three seeds, dense reference and153 full action arrays')
    ref = values[:, :1, :5]
    mse = np.mean((values[:, 1:, :5] - ref) ** 2, axis=(2, 3))
    scale = np.maximum(np.mean(ref ** 2, axis=(2, 3)), 1e-4)
    return dict(mse=mse, nmse=mse / scale)


def aggregate(rows, suites):
    """Seeds within snapshot, snapshots within task, tasks within suite."""
    groups = {(suite, task): [] for suite in suites for task in range(10)}
    seen = set()
    for row in rows:
        if row['sample_id'] in seen:
            raise ValueError('Duplicate observation')
        seen.add(row['sample_id'])
        value = np.asarray(row['nmse'], dtype=np.float64)
        if value.shape != (3, 153) or not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Incomplete or invalid error matrix')
        groups[row['suite'], row['task']].append(value.mean(axis=0))
    if any(not items for items in groups.values()):
        raise ValueError('All40 task identities are required')
    tasks = {suite: np.stack([np.mean(groups[suite, task], axis=0) for task in range(10)]) for suite in suites}
    means = {suite: value.mean(axis=0) for suite, value in tasks.items()}
    overall = np.stack(list(means.values())).mean(axis=0)
    return tasks, means, overall
