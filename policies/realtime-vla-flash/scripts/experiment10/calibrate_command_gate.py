"""Validation-only command-predictor calibration; no evaluation task fitting."""
import json
import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, HORIZONS, dump_json, json_lines, sha256
from scripts.experiment10.command_predictor import CommandPredictor


def main():
    histories, sources = [], []
    for suite in SUITES:
        root = RESULTS/'closed_loop/Full18_D10'/suite/'run'
        validated = json.loads((root/'validated.json').read_text())
        if not validated['complete']:
            raise RuntimeError('Calibration baseline unvalidated')
        for row in json.loads((root/'episode_log.json').read_text()):
            if row['task_id'] not in (6, 7):
                continue
            trace_path, infer_path = root/row['trace_path'], root/row['infer_path']
            actions = np.asarray([r['executed_action'] for r in json_lines(trace_path)], dtype=float)
            if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
                raise RuntimeError('Invalid executed command history')
            histories.append((suite, row['task_id'], row['episode_idx'], actions, json_lines(infer_path)))
            sources.append({'suite': suite, 'task': row['task_id'], 'init': row['episode_idx'],
                            'trace_sha256': sha256(trace_path), 'infer_sha256': sha256(infer_path)})
    # One scale estimate per episode then task, rather than overweighting long tasks.
    task_scales = {}
    for suite, task, init, actions, _ in histories:
        if len(actions) < 6:
            continue
        second_difference = np.diff(actions[:, :6], n=2, axis=0)
        estimate = np.median(second_difference**2, axis=0)
        task_scales.setdefault((suite, task), []).append(estimate)
    scale = np.median([np.mean(v, axis=0) for v in task_scales.values()], axis=0)
    floor = 1e-6
    q = np.maximum(scale*.5, floor)
    r = np.maximum(q*.1, floor)
    cells = {}
    for suite, task, init, actions, inference in histories:
        predictor = CommandPredictor(q, r)
        consumed = 0
        for infer in inference:
            frame = infer['frame_idx']
            if frame < consumed or frame > len(actions):
                raise RuntimeError('Inference/execution alignment error')
            predictor.observe_executed(actions[consumed:frame])
            consumed = frame
            if predictor.history_count < predictor.minimum_history:
                continue
            # Proposed Full18 commands are references, not physical action truth.
            proposal = np.asarray(infer['chunk_actions'], dtype=float)
            if len(proposal) < 12 or proposal.shape[1] != 7 or not np.isfinite(proposal).all():
                raise RuntimeError('Incomplete baseline proposal')
            means, variances = predictor.predict(12)
            residual = (proposal[:12, :6]-means)**2/variances
            for horizon in HORIZONS:
                for component, selected in (('translation', slice(0, 3)), ('rotation', slice(3, 6))):
                    value = float(residual[:horizon, selected].sum(axis=1).max())
                    cells.setdefault((suite, task, init, horizon, component), []).append(value)
    thresholds = {}
    for horizon in HORIZONS:
        thresholds[str(horizon)] = {}
        for component in ('translation', 'rotation'):
            task_values = {}
            for (suite, task, init, h, c), values in cells.items():
                if h == horizon and c == component:
                    task_values.setdefault((suite, task), []).append(values)
            if not task_values:
                raise RuntimeError('No calibration coverage')
            observations, weights = [], []
            for episodes in task_values.values():
                for values in episodes:
                    observations.extend(values)
                    weights.extend([1/(len(task_values)*len(episodes)*len(values))]*len(values))
            observations, weights = np.asarray(observations), np.asarray(weights)
            order = np.argsort(observations, kind='stable')
            cumulative = np.cumsum(weights[order])/weights.sum()
            indices = np.searchsorted(cumulative, [.5,.75,.9,.95], side='left')
            pooled = observations[order][np.minimum(indices, len(observations)-1)]
            thresholds[str(horizon)][component] = {str(p): float(v) for p, v in zip((50,75,90,95), pooled)}
    output = RESULTS/'command_gate'
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output/'calibration.json', {'selection_split': 'validation', 'tasks': [6,7], 'sources': sources,
              'process_variance': q.tolist(), 'measurement_variance': r.tolist(),
              'variance_floor': floor, 'floor_used_channels': (scale*.5 < floor).tolist(),
              'variance_rule': 'task-balanced median squared command second differences; q=0.5*scale, r=0.1*q; explicit floor',
              'thresholds': thresholds, 'threshold_rule': 'weighted empirical inverse-CDF of max-prefix residuals; tasks and episodes equal weight',
              'calibration_task_count': len(task_scales), 'minimum_executed_history': 4,
              'not_a_physical_dynamics_model': True, 'online_evaluation_complete': False,
              'limitations': ['Only two validation tasks per suite', 'Q/R are heuristic, not identifiable noise estimates',
                             'Floors and thresholds need sensitivity checks; gate agreement is not task correctness']})
    print('Calibrated', len(histories), 'validation episodes', flush=True)


if __name__ == '__main__':
    main()
