"""Choose at most four diagnostic layers from search-only measured quadrants."""
import pandas as pd

from scripts.experiment10.common import RESULTS, SUITES, dump_json, sha256


def main():
    activities, sensitivities, sources = [], [], []
    keys = ['suite', 'task_id', 'episode_idx', 'semantic_phase', 'layer']
    for suite in SUITES:
        root = RESULTS / 'probe' / suite
        activity_path = root / 'activation_statistics.csv'
        action_path = root / 'layer_action_sensitivity.csv'
        a = pd.read_csv(activity_path, low_memory=False)
        s = pd.read_csv(action_path)
        a = a[(a['split'] == 'search') & (a.tensor == 'delta_block') & (a.layer < 18)]
        s = s[(s['split'] == 'search') & (s.horizon == 4) & (s.layer < 18)]
        # Equal observations within episode; modalities then phases/tasks equal.
        a = a.groupby(keys + ['modality']).relative_residual.mean().reset_index()
        a = a.groupby(['suite', 'task_id', 'semantic_phase', 'layer', 'modality']).relative_residual.mean().reset_index()
        a = a.groupby(['suite', 'task_id', 'layer']).relative_residual.mean().reset_index()
        s = s.groupby(keys).normalized_continuous_rmse.mean().reset_index()
        s = s.groupby(['suite', 'task_id', 'semantic_phase', 'layer']).normalized_continuous_rmse.mean().reset_index()
        s = s.groupby(['suite', 'task_id', 'layer']).normalized_continuous_rmse.mean().reset_index()
        activities.append(a)
        sensitivities.append(s)
        sources.append({'suite': suite, 'activity_sha256': sha256(activity_path), 'sensitivity_sha256': sha256(action_path)})
    joined = pd.concat(activities).merge(pd.concat(sensitivities), on=['suite', 'task_id', 'layer'], validate='one_to_one')
    means = joined.groupby('layer')[['relative_residual', 'normalized_continuous_rmse']].mean()
    thresholds = means.median()
    means['update_high'] = means.relative_residual >= thresholds.relative_residual
    means['effect_high'] = means.normalized_continuous_rmse >= thresholds.normalized_continuous_rmse
    # Most extreme rank distance from median, not hand-picked layers.
    ranks = means[['relative_residual', 'normalized_continuous_rmse']].rank(pct=True)
    means['diagnostic_rank_distance'] = (ranks-.5).abs().sum(axis=1)
    selected, missing = [], []
    for update in (False, True):
        for effect in (False, True):
            group = means[(means.update_high == update) & (means.effect_high == effect)]
            if group.empty:
                missing.append({'update_high': update, 'effect_high': effect})
                continue
            layer = int(group.sort_values(['diagnostic_rank_distance'], ascending=False, kind='stable').index[0])
            selected.append({'layer': layer, 'update_high': update, 'effect_high': effect,
                             'relative_block_update': float(means.loc[layer, 'relative_residual']),
                             'h4_single_layer_rmse': float(means.loc[layer, 'normalized_continuous_rmse'])})
    output = RESULTS / 'submodule'
    output.mkdir(parents=True, exist_ok=True)
    means.to_csv(output / 'selection_layer_statistics.csv')
    dump_json(output / 'selected_layers.json', {'selection_split': 'search', 'selected': selected,
              'missing_quadrants': missing, 'thresholds': thresholds.to_dict(), 'sources': sources,
              'rule': 'one most extreme rank-distance layer per median quadrant; ties lower layer',
              'scope': 'diagnostic selection, not safe pruning recommendation',
              'submodule_interventions_complete': False})
    print(selected, flush=True)


if __name__ == '__main__':
    main()
