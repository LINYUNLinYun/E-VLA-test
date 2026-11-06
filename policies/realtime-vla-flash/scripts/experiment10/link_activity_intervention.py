"""Join measured activation features to paired h4 skip effects; no safety labels."""
import argparse
import pandas as pd

from scripts.experiment10.common import RESULTS, SUITES, sha256, dump_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    root = RESULTS / 'probe' / args.suite
    features_path = RESULTS / 'numerical_updates' / args.suite / 'module_cancellation.csv'
    actions_path = root / 'layer_action_sensitivity.csv'
    features = pd.read_csv(features_path)
    actions = pd.read_csv(actions_path)
    keys = ['sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase', 'split', 'layer']
    metrics = ['normalized_continuous_rmse', 'gripper_flip_rate', 'translation_command_rmse',
               'rotation_command_rmse']
    h4 = actions[actions.horizon == 4].groupby(keys, as_index=False)[metrics].mean()
    linked = features.merge(h4, on=keys, how='left', validate='many_to_one', indicator=True)
    if not (linked['_merge'] == 'both').all():
        raise RuntimeError('Activity/action coverage mismatch')
    linked = linked.drop(columns='_merge')
    feature_names = ['relative_attn_update', 'relative_ffn_update', 'cancellation_ratio',
                     'attn_ffn_cosine_derived']
    values = feature_names + metrics
    ep_keys = ['suite', 'split', 'task_id', 'episode_idx', 'semantic_phase', 'layer', 'modality']
    episodes = linked.groupby(ep_keys)[values].mean()
    task_keys = [k for k in ep_keys if k != 'episode_idx']
    tasks = episodes.groupby(task_keys)[values].mean().reset_index()
    output = RESULTS / 'numerical_updates' / args.suite
    linked.to_csv(output / 'activation_intervention_links.csv', index=False)
    tasks.to_csv(output / 'task_activity_intervention_links.csv', index=False)
    rows = []
    # Search only: validation/evaluation are saved, never used to discover links.
    for identity, chosen in tasks[tasks['split'] == 'search'].groupby(['layer', 'semantic_phase', 'modality']):
        for feature in feature_names:
            paired = chosen[[feature, 'normalized_continuous_rmse']].dropna()
            count = len(paired)
            correlation = None
            if count >= 3 and (paired.nunique() > 1).all():
                correlation = float(paired.rank().corr().iloc[0, 1])
            rows.append({'layer': identity[0], 'semantic_phase': identity[1], 'modality': identity[2],
                         'feature': feature, 'task_count': count, 'spearman_descriptive': correlation,
                         'interpretation': 'exploratory association; task-bootstrap/LOO validation still required'})
    pd.DataFrame(rows).to_csv(output / 'descriptive_activity_correlations.csv', index=False)
    dump_json(output / 'links_manifest.json', {'complete_join': True, 'primary_horizon': 4,
             'sample_layer_modality_rows': len(linked), 'task_cells': len(tasks),
             'actions_sha256': sha256(actions_path), 'features_sha256': sha256(features_path),
             'association_validation_complete': False, 'not_a_pruning_safety_test': True})
    print(args.suite, len(linked), len(tasks), flush=True)


if __name__ == '__main__':
    main()
