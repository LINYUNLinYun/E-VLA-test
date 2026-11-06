"""Task-equal descriptive summaries for focused Attention/FFN interventions."""
import argparse
import json
import pandas as pd
from scripts.experiment10.common import RESULTS, SUITES, dump_json, sha256


ACTION_METRICS = ['normalized_continuous_rmse', 'translation_command_rmse',
                  'rotation_command_rmse', 'gripper_command_rmse', 'gripper_flip_rate']
DETAIL_METRICS = ['token_norm_p50', 'token_norm_p90', 'token_norm_p99',
                  'token_norm_p90_over_p50', 'top10pct_token_energy_fraction',
                  'top10pct_channel_energy_fraction']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    directory = RESULTS/'submodule'/args.suite
    summary = json.loads((directory/'summary.json').read_text())
    if not summary.get('complete') or summary.get('parity_passed') is not True:
        raise RuntimeError('Submodule suite collection is incomplete or failed parity')
    samples = set(summary['samples'])
    sample_names = {p.rsplit('/', 1)[-1] for p in samples}
    actions_path = directory/'submodule_action_metrics.csv'
    actions = pd.read_csv(actions_path)
    actions = actions[(actions['horizon'] == 4) & actions['sample_id'].map(
        lambda name: name+'.npz' in sample_names)]
    if actions.empty or not set(actions['mode']).issubset({'attn', 'ffn', 'both'}):
        raise RuntimeError('Missing focused module-action records')
    # First average repeated noise draws/observations within each task-phase;
    # then average tasks equally. Frames and noise seeds never count as tasks.
    task_keys = ['suite', 'task_id', 'semantic_phase', 'layer', 'mode']
    task = actions.groupby(task_keys)[ACTION_METRICS].mean().reset_index()
    grouped = task.groupby(['suite', 'semantic_phase', 'layer', 'mode'])
    action_summary = grouped[ACTION_METRICS].mean().reset_index()
    action_summary['task_count'] = grouped['task_id'].nunique().to_numpy()
    action_summary.to_csv(directory/'submodule_task_equal_action_summary.csv', index=False)

    detail_paths = [p for p in directory.glob('*.update_details.csv')
                    if p.name[:-len('.update_details.csv')]+'.npz' in sample_names]
    if len(detail_paths) != len(samples):
        raise RuntimeError('Focused numerical-detail files do not cover all selected samples')
    details = pd.concat([pd.read_csv(p) for p in detail_paths], ignore_index=True)
    if details.empty or not set(details['module']).issubset({'attn', 'ffn', 'block'}):
        raise RuntimeError('Invalid focused numerical-detail rows')
    sample_info = actions[['sample_id', 'suite', 'task_id', 'semantic_phase']].drop_duplicates()
    if sample_info['sample_id'].duplicated().any():
        raise RuntimeError('A sample maps to multiple task/phase identities')
    details = details.merge(sample_info, on='sample_id', how='left',
                            validate='many_to_one', indicator=True)
    if not (details['_merge'] == 'both').all():
        raise RuntimeError('Numerical details lack a matched intervention observation')
    detail_keys = ['suite', 'task_id', 'semantic_phase', 'layer', 'modality', 'module']
    per_task = details.groupby(detail_keys)[DETAIL_METRICS].mean().reset_index()
    dg = per_task.groupby(['suite', 'semantic_phase', 'layer', 'modality', 'module'])
    detail_summary = dg[DETAIL_METRICS].mean().reset_index()
    detail_summary['task_count'] = dg['task_id'].nunique().to_numpy()
    detail_summary.to_csv(directory/'submodule_task_equal_numerical_summary.csv', index=False)
    dump_json(directory/'submodule_analysis_manifest.json', {
        'complete': True, 'search_only_representative_samples': True,
        'sample_count': len(samples), 'action_rows_h4': len(actions),
        'task_phase_action_cells': len(task), 'numerical_detail_rows': len(details),
        'action_source_sha256': sha256(actions_path),
        'numeric_source_files': len(detail_paths),
        'reduction': 'noise/observations within task-phase first; task-equal suite means second',
        'not_closed_loop_evidence': True, 'confirmatory_tests': False,
        'suite_source_complete': summary['complete']})
    print(args.suite, 'samples', len(samples), 'task_cells', len(task), flush=True)


if __name__ == '__main__':
    main()
