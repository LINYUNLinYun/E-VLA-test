"""Paired actual-EEF branch deviations; partial collection is explicitly marked."""
import argparse
import json
import numpy as np
import pandas as pd
from scripts.experiment10.common import RESULTS, SUITES, dump_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    directory = RESULTS/'branches'/args.suite
    source_manifest = RESULTS/'observations'/args.suite/'manifest.json'
    observations = {e['sha256']: e for e in json.loads(source_manifest.read_text())['observations']}
    records = []
    for path in sorted(directory.glob('*_h*.json')):
        data = json.loads(path.read_text())
        if not all(data.get(k) for k in ('video_full_decode_verified', 'anchor_observation_bitwise_equal', 'controller_next_step_parity_passed')):
            raise RuntimeError('Unvalidated branch '+str(path))
        identity = data['provenance']
        records.append((path, data, (identity['observation_sha256'], identity['horizon'], identity['noise_seed'])))
    baselines = {key: (path, data) for path, data, key in records if data['provenance']['condition'] == 'Full18_D10'}
    rows, missing = [], []
    for path, data, key in records:
        if data['provenance']['condition'] == 'Full18_D10':
            continue
        if key not in baselines:
            missing.append(str(path)); continue
        baseline_path, baseline = baselines[key]
        observation = observations[key[0]]
        if observation['split'] != 'validation' or observation['episode_idx'] != 0:
            raise RuntimeError('Unexpected branch sampling role/init')
        if data['provenance']['shadow_sha256'] != baseline['provenance']['shadow_sha256']:
            raise RuntimeError('Paired branch shadow sources differ')
        count = min(len(baseline['states']), len(data['states']))
        xyz = [np.asarray([s['eef_pos'] for s in r['states'][:count]], dtype=float) for r in (baseline, data)]
        if not all(np.isfinite(v).all() for v in xyz) or not np.array_equal(xyz[0][0], xyz[1][0]):
            raise RuntimeError('Branch initial-state or finite XYZ mismatch')
        error = np.linalg.norm(xyz[0]-xyz[1], axis=1)
        quaternions = [np.asarray([s['eef_quat'] for s in r['states'][:count]], dtype=float)
                       for r in (baseline, data)]
        for q in quaternions:
            norms = np.linalg.norm(q, axis=1, keepdims=True)
            if q.shape != (count, 4) or not np.isfinite(q).all() or np.any(norms == 0):
                raise RuntimeError('Invalid branch quaternion')
            q /= norms
        # q and -q encode the same orientation; never compare components directly.
        angle = np.degrees(2*np.arccos(np.clip(np.abs((quaternions[0]*quaternions[1]).sum(axis=1)), 0, 1)))
        rows.append({'variant': data['provenance']['condition'], 'horizon': key[1],
                     'suite': args.suite, 'task_id': observation['task_id'],
                     'episode_idx': observation['episode_idx'], 'semantic_phase': observation['semantic_phase'],
                     'sample_id': observation['sample_id'], 'split': observation['split'],
                     'observation_sha256': key[0], 'shared_states': count,
                     'baseline_actions': baseline['actions_executed'], 'variant_actions': data['actions_executed'],
                     'mean_xyz_error_m': float(error.mean()), 'max_xyz_error_m': float(error.max()),
                     'last_shared_xyz_error_m': float(error[-1]),
                     'max_orientation_error_deg': float(angle.max()),
                     'last_shared_orientation_error_deg': float(angle[-1]),
                     'both_reached_requested_horizon': count == key[1]+1,
                     'baseline_terminal_success': baseline['terminal_success'],
                     'variant_terminal_success': data['terminal_success'],
                     'baseline_record_sha256': sha256(baseline_path), 'variant_record_sha256': sha256(path)})
    output = RESULTS/'branch_analysis'/args.suite
    output.mkdir(parents=True, exist_ok=True)
    paired = pd.DataFrame(rows)
    paired.to_csv(output/'paired_prefix_xyz.csv', index=False)
    if not paired.empty:
        task_keys = ['suite', 'task_id', 'semantic_phase', 'variant', 'horizon']
        per_task = paired.groupby(task_keys, as_index=False).agg(
            anchor_pairs=('sample_id', 'nunique'),
            mean_xyz_error_m=('mean_xyz_error_m', 'mean'),
            mean_anchor_max_xyz_error_m=('max_xyz_error_m', 'mean'),
            max_anchor_xyz_error_m=('max_xyz_error_m', 'max'),
            mean_orientation_error_deg=('max_orientation_error_deg', 'mean'),
            max_orientation_error_deg=('max_orientation_error_deg', 'max'),
            early_terminated_pairs=('both_reached_requested_horizon', lambda x: int((~x).sum())))
        group_keys = ['suite', 'semantic_phase', 'variant', 'horizon']
        phase_summary = per_task.groupby(group_keys).agg(
            task_count=('task_id', 'nunique'), anchor_pairs=('anchor_pairs', 'sum'),
            mean_task_xyz_error_m=('mean_xyz_error_m', 'mean'),
            mean_task_max_xyz_error_m=('mean_anchor_max_xyz_error_m', 'mean'),
            worst_anchor_xyz_error_m=('max_anchor_xyz_error_m', 'max'),
            mean_task_orientation_error_deg=('mean_orientation_error_deg', 'mean'),
            worst_anchor_orientation_error_deg=('max_orientation_error_deg', 'max'),
            early_terminated_pairs=('early_terminated_pairs', 'sum')).reset_index()
        phase_summary.to_csv(output/'phase_task_equal_paired_prefix_xyz.csv', index=False)
    summary = directory/'summary.json'
    complete = summary.exists() and json.loads(summary.read_text()).get('complete_prefix_branches', False)
    dump_json(output/'manifest.json', {'complete_source_collection': complete, 'paired_records': len(rows),
        'observation_manifest_sha256': sha256(source_manifest), 'code_sha256': sha256(__file__),
        'phase_summary': 'task equal within phase/variant/horizon; descriptive only',
        'unpaired_pending_records': missing, 'shared_step_alignment': True,
        'recovery_evaluated': False, 'warning': 'Overlap-only actual EEF error, not full-horizon endpoint after early termination; terminal success is not recovery rate'})


if __name__ == '__main__':
    main()
