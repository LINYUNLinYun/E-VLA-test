"""Task-bootstrap/LOO stability of search-only within-layer activity associations.

Intervals are exploratory, not multiplicity-adjusted tests or pruning guarantees.
Undefined constant bootstrap resamples are retained in the validity denominator.
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scripts.experiment10.common import RESULTS, SUITES, dump_json, sha256


def correlations(x, y):
    a, b = rankdata(x, axis=-1), rankdata(y, axis=-1)
    a -= a.mean(axis=-1, keepdims=True)
    b -= b.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((a*a).sum(axis=-1)*(b*b).sum(axis=-1))
    return np.divide((a*b).sum(axis=-1), denominator,
                     out=np.full_like(denominator, np.nan, dtype=float), where=denominator > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    directory = RESULTS/'numerical_updates'/args.suite
    source = directory/'task_activity_intervention_links.csv'
    data = pd.read_csv(source)
    data = data[data['split'] == 'search']
    features = ['relative_attn_update', 'relative_ffn_update', 'cancellation_ratio', 'attn_ffn_cosine_derived']
    rng = np.random.default_rng(20261009)
    rows = []
    for identity, group in data.groupby(['layer', 'semantic_phase', 'modality'], sort=True):
        for feature in features:
            paired = group[['task_id', feature, 'normalized_continuous_rmse']].dropna().sort_values('task_id')
            if paired.task_id.duplicated().any():
                raise RuntimeError('Expected one value per task in each layer/phase/modality cell')
            x, y = paired[feature].to_numpy(), paired.normalized_continuous_rmse.to_numpy()
            row = dict(zip(('layer', 'semantic_phase', 'modality'), identity))
            row.update(feature=feature, task_count=len(x), bootstrap_draws=10000,
                       rho=None, ci025=None, ci975=None, valid_bootstrap_draws=0,
                       loo_min=None, loo_max=None, valid_loo_count=0)
            if len(x) >= 3:
                rho = float(correlations(x, y))
                indices = rng.integers(0, len(x), size=(10000, len(x)))
                bootstrap = correlations(x[indices], y[indices])
                finite = bootstrap[np.isfinite(bootstrap)]
                loo = np.array([correlations(np.delete(x, i), np.delete(y, i)) for i in range(len(x))])
                valid_loo = loo[np.isfinite(loo)]
                row.update(rho=rho if np.isfinite(rho) else None,
                           valid_bootstrap_draws=len(finite), valid_loo_count=len(valid_loo))
                if len(finite):
                    row.update(ci025=float(np.quantile(finite, .025)), ci975=float(np.quantile(finite, .975)))
                if len(valid_loo):
                    row.update(loo_min=float(valid_loo.min()), loo_max=float(valid_loo.max()))
            rows.append(row)
    pd.DataFrame(rows).to_csv(directory/'activity_association_stability.csv', index=False)
    dump_json(directory/'activity_association_stability_manifest.json', {
        'complete_search_task_bootstrap_loo': True, 'source_sha256': sha256(source),
        'code_sha256': sha256(__file__), 'seed': 20261009, 'draws': 10000,
        'cells': len(rows), 'unit': 'task; within layer/phase/modality',
        'validation_frozen_relationship_test_complete': False,
        'warning': 'Exploratory percentile intervals conditional on nonconstant resamples; tiny n; no multiplicity adjustment or safety claim'})


if __name__ == '__main__':
    main()
