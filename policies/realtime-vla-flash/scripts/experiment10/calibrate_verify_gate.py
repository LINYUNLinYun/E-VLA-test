"""Fit a D3-vs-Full18 contiguous-prefix gate on validation tasks only."""
from __future__ import annotations

import json

import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, sha256
from scripts.experiment10.probe_schedules import CONDITIONS


HORIZON = 8
QUANTILE = 0.95


def main():
    candidate_index = next(i for i, (name, _, _) in enumerate(CONDITIONS) if name == 'D3')
    cells = {}
    sources = []
    for suite in SUITES:
        for path in sorted((RESULTS / 'shadow' / suite).glob('*.npz')):
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata_json'].item()))
                if metadata.get('split') != 'validation' or metadata.get('task_id') not in (6, 7):
                    continue
                reference = np.asarray(data['baseline_commands'][:, :HORIZON], dtype=float)
                candidate = np.asarray(data['candidate_commands'][:, candidate_index, :HORIZON], dtype=float)
            if reference.shape != (3, HORIZON, 7) or candidate.shape != reference.shape:
                raise RuntimeError('Unexpected paired-command shape in %s' % path)
            if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
                raise RuntimeError('Non-finite paired command in %s' % path)
            translation_max = np.linalg.norm(candidate[:, :, :3] - reference[:, :, :3], axis=2).max(axis=1)
            rotation_max = np.linalg.norm(candidate[:, :, 3:6] - reference[:, :, 3:6], axis=2).max(axis=1)
            gripper_mismatch = np.not_equal(candidate[:, :, 6] > 0, reference[:, :, 6] > 0).any(axis=1)
            key = (suite, metadata['task_id'], metadata['episode_idx'], metadata['semantic_phase'])
            cells[key] = {
                'translation_max_m': float(np.median(translation_max)),
                'rotation_max_rad': float(np.median(rotation_max)),
                'gripper_mismatch_rate': float(np.mean(gripper_mismatch)),
            }
            sources.append({'path': str(path), 'sha256': sha256(path), 'sample_id': metadata['sample_id'],
                            'suite': suite, 'task': metadata['task_id'], 'episode': metadata['episode_idx'],
                            'phase': metadata['semantic_phase']})
    if not cells:
        raise RuntimeError('No validation task 6-7 paired D3/Full18 shadow samples')

    by_task = {}
    for (suite, task, _, _), cell in cells.items():
        by_task.setdefault((suite, task), []).append(cell)
    task_values = []
    for (suite, task), values in sorted(by_task.items()):
        task_values.append({
            'suite': suite,
            'task': task,
            'translation_max_m': float(np.median([v['translation_max_m'] for v in values])),
            'rotation_max_rad': float(np.median([v['rotation_max_rad'] for v in values])),
            'gripper_mismatch_rate': float(np.mean([v['gripper_mismatch_rate'] for v in values])),
            'phase_cells': len(values),
        })
    translation = np.asarray([r['translation_max_m'] for r in task_values])
    rotation = np.asarray([r['rotation_max_rad'] for r in task_values])
    output = RESULTS / 'verify_gate'
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / 'calibration.json', {
        'schema': 1,
        'candidate': 'Full18_D3',
        'reference': 'Full18_D10',
        'split': 'validation_tasks_6_7_only',
        'horizon': HORIZON,
        'quantile': QUANTILE,
        'translation_threshold_m': float(np.quantile(translation, QUANTILE)),
        'rotation_threshold_rad': float(np.quantile(rotation, QUANTILE)),
        'gripper_rule': 'strict sign equality at every accepted action; mismatch stops the prefix',
        'aggregation': 'median across fixed noise seeds per observation; median phase cells within task; task-equal quantile',
        'task_count': len(task_values),
        'cell_count': len(cells),
        'task_values': task_values,
        'sources': sources,
        'not_a_safety_or_success_guarantee': True,
        'limitations': ['Only validation tasks 6-7 are used',
                        'Agreement with Full18 is not task correctness',
                        'The gate costs an additional Full18 inference and may not accelerate execution'],
    })
    print(json.dumps({'event': 'verify_gate_calibrated', 'tasks': len(task_values),
                      'cells': len(cells), 'translation_threshold_m': float(np.quantile(translation, QUANTILE)),
                      'rotation_threshold_rad': float(np.quantile(rotation, QUANTILE))}), flush=True)


if __name__ == '__main__':
    main()
