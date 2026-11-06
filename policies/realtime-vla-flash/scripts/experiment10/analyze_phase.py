"""Task-clustered phase contrasts and search-only numerical priors; no GPU initialization."""
from __future__ import annotations

import argparse
from itertools import combinations, product
import json

import numpy as np
import pandas as pd

from scripts.experiment10.common import RESULTS, SUITES, PHASES, dump_json, sha256


def contrasts(task_cells, episode_cells=None):
    rows = []
    for (suite, role), chosen in task_cells.groupby(['suite', 'split']):
        primary = chosen[chosen.layer < 18]
        pivot = primary.pivot_table(index=['task_id', 'layer'], columns='semantic_phase',
                                    values='normalized_continuous_rmse')
        common = primary.groupby(['task_id', 'semantic_phase']).normalized_continuous_rmse.mean()
        for phase_a, phase_b in combinations(PHASES, 2):
            if phase_a not in pivot or phase_b not in pivot:
                continue
            for layer in sorted(primary.layer.unique()):
                pairs = pivot.xs(layer, level='layer')[[phase_a, phase_b]].dropna()
                if len(pairs) < 3:
                    continue
                difference = (pairs[phase_b] - pairs[phase_a]).to_numpy()
                centered = difference - np.array([common.loc[(t, phase_b)] - common.loc[(t, phase_a)]
                                                 for t in pairs.index])
                paired_episode_count = None
                if episode_cells is not None:
                    ep = episode_cells[(episode_cells.suite == suite) &
                                       (episode_cells['split'] == role) & (episode_cells.layer < 18)]
                    ep_pivot = ep.pivot_table(index=['task_id', 'episode_idx', 'layer'],
                                             columns='semantic_phase', values='normalized_continuous_rmse')
                    if phase_a not in ep_pivot or phase_b not in ep_pivot:
                        continue
                    matched = ep_pivot[[phase_a, phase_b]].dropna()
                    if layer not in matched.index.get_level_values('layer'):
                        continue
                    ep_difference = matched[phase_b] - matched[phase_a]
                    common_difference = ep_difference.groupby(['task_id', 'episode_idx']).mean()
                    layer_difference = ep_difference.xs(layer, level='layer')
                    relative = layer_difference - common_difference.reindex(layer_difference.index)
                    paired_episode_count = len(layer_difference)
                    raw_tasks = layer_difference.groupby('task_id').mean()
                    relative_tasks = relative.groupby('task_id').mean().dropna()
                    if len(relative_tasks) < 3:
                        continue
                    difference = raw_tasks.reindex(relative_tasks.index).to_numpy()
                    centered = relative_tasks.to_numpy()
                    pairs = relative_tasks  # task count reflects matched episodes only
                rng = np.random.default_rng(20261009 + int(layer))
                draws = rng.integers(0, len(difference), (10000, len(difference)))
                interval = np.quantile(centered[draws].mean(axis=1), [.025, .975])
                signs = np.asarray(list(product((-1, 1), repeat=len(centered))))
                p = float(np.mean(np.abs((signs * centered).mean(axis=1)) >= abs(centered.mean()) - 1e-15))
                loo = [(np.delete(centered, i)).mean() for i in range(len(centered))]
                rows.append({'suite': suite, 'split': role, 'layer': int(layer), 'phase_a': phase_a,
                             'phase_b': phase_b, 'task_count': len(pairs), 'raw_phase_difference': float(difference.mean()),
                             'paired_episode_count': paired_episode_count,
                             'pairing': 'same_task_same_init' if episode_cells is not None else 'task_cells_only',
                             'relative_layer_difference': float(centered.mean()), 'ci95_low': float(interval[0]),
                             'ci95_high': float(interval[1]), 'p_signflip': p,
                             'leave_one_task_out_sign_stable': all(np.sign(v) == np.sign(centered.mean()) for v in loo),
                             'p_holm': None})
    # One multiplicity family per suite/split, covering layers and phase pairs.
    for suite in SUITES:
        for role in ('search', 'validation', 'evaluation'):
            family = sorted([r for r in rows if r['suite'] == suite and r['split'] == role], key=lambda r: r['p_signflip'])
            previous = 0.
            for index, row in enumerate(family):
                previous = max(previous, min(1., row['p_signflip'] * (len(family) - index)))
                row['p_holm'] = previous
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    metrics, statistics, sources = [], [], []
    for suite in SUITES:
        directory = RESULTS / 'probe' / suite
        if not (directory / 'summary.json').exists():
            if args.allow_partial:
                continue
            raise RuntimeError('Missing completed suite: ' + suite)
        summary = json.loads((directory / 'summary.json').read_text())
        if not summary['complete'] or summary.get('partial_sample_limit'):
            raise RuntimeError('Only fully covered probes may generate frozen priors')
        sources.append({'suite': suite, 'summary_sha256': sha256(directory / 'summary.json')})
        metrics.append(pd.read_csv(directory / 'layer_action_sensitivity.csv'))
        statistics.append(pd.read_csv(directory / 'activation_statistics.csv', low_memory=False))
    if not metrics:
        raise RuntimeError('No complete probes')
    output = RESULTS / 'phase_analysis'
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(metrics, ignore_index=True)
    primary = frame[frame.horizon == 4]
    # First average observation/noise within episode, then episodes within task.
    episode = primary.groupby(['suite', 'split', 'task_id', 'episode_idx', 'semantic_phase', 'layer'], as_index=False)[
                    ['normalized_continuous_rmse', 'gripper_flip_rate']].mean()
    tasks = episode.groupby(['suite', 'split', 'task_id', 'semantic_phase', 'layer'], as_index=False)[
                    ['normalized_continuous_rmse', 'gripper_flip_rate']].mean()
    tasks.to_csv(output / 'task_phase_sensitivity.csv', index=False)
    comparisons = contrasts(tasks, episode)
    comparisons.to_csv(output / 'phase_interaction.csv', index=False)
    activity = pd.concat(statistics, ignore_index=True)
    search = activity[(activity['split'] == 'search') & (activity.tensor == 'delta_block') & (activity.layer < 18)].copy()
    cells = search.groupby(['suite', 'task_id', 'semantic_phase', 'modality', 'layer'], as_index=False).cosine.median()
    priors = []
    search_actions = tasks[tasks['split'] == 'search']
    for layer in range(1, 19):
        observed = cells[cells.layer == layer].cosine.dropna()
        robust = float(np.quantile(observed, .1)) if len(observed) else None
        actions = search_actions[search_actions.layer == layer].groupby(['suite', 'task_id']).normalized_continuous_rmse.mean()
        priors.append({'layer': layer, 'robust_cosine_p10': robust,
                       'cosine_cell_count': len(observed), 'protected_structural': layer in (1, 2, 17, 18),
                       'protected_low_cosine': robust is None or robust < .98,
                       'single_layer_task_mean_rmse': float(actions.mean()) if len(actions) else None})
    coverage = search_actions.groupby(['suite', 'semantic_phase']).task_id.nunique()
    phase_layers = {}
    for phase in PHASES:
        phase_layers[phase] = []
        for layer in range(1, 19):
            observed = cells[(cells.layer == layer) & (cells.semantic_phase == phase)].cosine.dropna()
            robust = float(np.quantile(observed, .1)) if len(observed) else None
            values = search_actions[(search_actions.layer == layer) & (search_actions.semantic_phase == phase)]
            actions = values.groupby(['suite', 'task_id']).normalized_continuous_rmse.mean()
            phase_layers[phase].append({'layer': layer, 'robust_cosine_p10': robust,
                       'cosine_cell_count': len(observed), 'protected_structural': layer in (1, 2, 17, 18),
                       'protected_low_cosine': robust is None or robust < .98,
                       'single_layer_task_mean_rmse': float(actions.mean()) if len(actions) else None})
    dump_json(output / 'layer_prior.json', {'complete_four_suites': len(sources) == 4,
              'cosine_threshold': .98, 'source_summaries': sources, 'selection_split': 'search',
              'layers': priors, 'phase_layers': phase_layers, 'phase_task_coverage': [{'suite': s, 'phase': p, 'tasks': int(v)}
                         for (s, p), v in coverage.items()],
              'limitations': ['Only five inits; historical task exposure', 'Phase grouping is not a causal phase intervention']})
    dump_json(output / 'summary.json', {'suites': len(sources), 'task_cells': len(tasks),
              'phase_comparisons': len(comparisons), 'primary_horizon': 4, 'task_bootstrap_repeats': 10000})


if __name__ == '__main__':
    main()
