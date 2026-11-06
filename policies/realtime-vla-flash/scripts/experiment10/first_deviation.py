"""Same-step positional/pose deviation onset, distinct from failure stage and causality."""
import json
import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, json_lines, dump_json


def first_persistent(values, threshold, consecutive=3):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or consecutive < 1:
        raise ValueError('Need finite one-dimensional errors and positive window')
    above = values > threshold
    for index in range(len(values)-consecutive+1):
        if above[index:index+consecutive].all():
            return index
    return None


def pose_error_degrees(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Need paired finite quaternion arrays')
    norms = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
    if np.any(norms == 0):
        raise ValueError('Zero quaternion')
    return np.degrees(2*np.arccos(np.clip(np.abs(np.sum(a*b, axis=1))/norms, 0, 1)))


def main():
    summary = json.loads((RESULTS/'closed_loop_analysis/validated_results.json').read_text())
    complete = {(r['condition'], r['suite']) for r in summary['suite_rows'] if r['status'] == 'validated_complete'}
    rows = []
    for suite in SUITES:
        if ('Full18_D10', suite) not in complete:
            continue
        root = RESULTS/'closed_loop/Full18_D10'/suite/'run'
        base = {(e['task_id'], e['episode_idx']): e for e in json.loads((root/'episode_log.json').read_text())}
        for condition in ('Full18_D1', 'Full18_D2', 'Full18_D4'):
            if (condition, suite) not in complete:
                continue
            variant_root = RESULTS/'closed_loop'/condition/suite/'run'
            variants = json.loads((variant_root/'episode_log.json').read_text())
            for variant in variants:
                key = variant['task_id'], variant['episode_idx']
                baseline = base[key]
                blue = json_lines(root/baseline['telemetry_path'])
                orange = json_lines(variant_root/variant['telemetry_path'])
                count = min(len(blue), len(orange))
                if [r['step'] for r in blue[:count]] != [r['step'] for r in orange[:count]]:
                    raise RuntimeError('Same-step alignment mismatch')
                xyz = np.linalg.norm(np.asarray([r['eef_pos'] for r in blue[:count]])-
                                     np.asarray([r['eef_pos'] for r in orange[:count]]), axis=1)
                pose = pose_error_degrees([r['eef_quat'] for r in blue[:count]], [r['eef_quat'] for r in orange[:count]])
                position_first, pose_first = first_persistent(xyz, .02), first_persistent(pose, 5.)
                row = {'suite': suite, 'condition': condition, 'task': key[0], 'init': key[1],
                       'baseline_success': baseline['success'], 'variant_success': variant['success'],
                       'overlap_states': count, 'baseline_states': len(blue), 'variant_states': len(orange),
                       'max_xyz_m': float(xyz.max()), 'max_pose_degrees': float(pose.max()),
                       'first_persistent_xyz_step': position_first, 'first_persistent_pose_step': pose_first,
                       'baseline_phase_at_xyz_onset': blue[position_first]['semantic_phase'] if position_first is not None else None,
                       'baseline_phase_at_pose_onset': blue[pose_first]['semantic_phase'] if pose_first is not None else None,
                       'variant_highest_phase': variant['furthest_semantic_phase'],
                       'unknown_after_overlap': len(blue) != len(orange)}
                rows.append(row)
    dump_json(RESULTS/'closed_loop_analysis/first_deviation.json', {'rows': rows,
             'criteria': {'xyz_m': .02, 'pose_degrees': 5., 'consecutive_states': 3},
             'interpretation': 'Trajectory discrepancy, not task error or causal failure onset; overlap only'})
    print(len(rows), 'paired deviation records', flush=True)


if __name__ == '__main__':
    main()
