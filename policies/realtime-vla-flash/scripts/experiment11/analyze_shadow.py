"""Analyze full-trajectory shadow predictions; variants never execute in LIBERO."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.experiment11.protocol import RESULTS, SUITES
from scripts.experiment10.common import PHASE_MAP, dump_json, json_lines, sha256
from scripts.experiment11.launch_shadow import frozen_jobs, validate_job


PREFIXES = (1, 2, 3, 4, 5, 8, 12)
OFFSETS = tuple(range(1, 51))
SCALE_FACTORS = (0.01, 0.05, 0.10)
K_THRESHOLDS = (1, 2, 3, 4, 5, 8, 12)
BOOTSTRAP_REPLICATES = 2000
ANALYSIS = RESULTS / 'analysis/shadow_d3'
SHADOW_ROOT = RESULTS / 'shadow'
SHADOW_STATUS = RESULTS / 'experiment11_shadow_status.json'


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def coarse_phase(raw: str | None) -> str:
    return PHASE_MAP.get(str(raw or ''), 'unknown')


def task_key(suite: str, task_id: int) -> str:
    return f'{suite}::task{task_id:02d}'


def action_differences(full_actions, variant_actions) -> dict[str, np.ndarray]:
    full = np.asarray(full_actions, dtype=np.float32)
    variant = np.asarray(variant_actions, dtype=np.float32)
    if full.shape != (50, 7) or variant.shape != (50, 7):
        raise ValueError(f'Expected two 50x7 action chunks; got {full.shape} and {variant.shape}')
    if not np.isfinite(full).all() or not np.isfinite(variant).all():
        raise ValueError('Shadow action chunk contains NaN/Inf')
    delta = variant - full
    return {
        'translation_command_l2': np.linalg.norm(delta[:, :3], axis=1),
        'rotation_command_l2': np.linalg.norm(delta[:, 3:6], axis=1),
        'continuous6_component_rmse': np.sqrt(np.mean(np.square(delta[:, :6]), axis=1)),
        'gripper_abs_error': np.abs(delta[:, 6]),
        'gripper_disagreement': (variant[:, 6] >= 0) != (full[:, 6] >= 0),
    }


def prefix_metrics(diff: dict[str, np.ndarray], horizon: int) -> dict[str, float]:
    t = diff['translation_command_l2'][:horizon]
    r = diff['rotation_command_l2'][:horizon]
    g = diff['gripper_disagreement'][:horizon]
    return {
        'translation_command_l2_mean': float(np.mean(t)),
        'translation_command_l2_p95_within_prefix': float(np.quantile(t, 0.95)),
        'translation_command_l2_max_within_prefix': float(np.max(t)),
        'rotation_command_l2_mean': float(np.mean(r)),
        'rotation_command_l2_p95_within_prefix': float(np.quantile(r, 0.95)),
        'rotation_command_l2_max_within_prefix': float(np.max(r)),
        'continuous6_component_rmse': float(np.sqrt(np.mean(np.square(
            diff['continuous6_component_rmse'][:horizon])))),
        'gripper_disagreement_rate': float(np.mean(g)),
        'any_gripper_disagreement': float(np.any(g)),
    }


def add_values(store: dict, task_store: dict, key: tuple, task: str, values: dict[str, float]) -> None:
    for metric, value in values.items():
        store[key][metric].append(float(value))
        task_store[key][metric][task].append(float(value))


def add_prefix_metrics(store: dict, task_store: dict, variant: str, suite: str,
                       replan_phase: str, horizon: int, task: str,
                       values: dict[str, float]) -> None:
    for suite_group in (suite, 'ALL'):
        for phase in ('all', replan_phase):
            key = (variant, suite_group, phase, horizon)
            add_values(store, task_store, key, task, values)


def consistency_k(diff: dict[str, np.ndarray], translation_threshold: float,
                  rotation_threshold: float) -> int:
    passes = ((diff['translation_command_l2'] <= translation_threshold) &
              (diff['rotation_command_l2'] <= rotation_threshold) &
              (~diff['gripper_disagreement']))
    failed = np.flatnonzero(~passes)
    return int(failed[0]) if failed.size else 50


def bootstrap_mean_ci(task_means: list[float], seed: int) -> tuple[float | None, float | None]:
    if len(task_means) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    values = np.asarray(task_means, dtype=np.float64)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_cells(store: dict, task_store: dict, key_columns: tuple[str, ...],
                    metric_names: tuple[str, ...], ci: bool = False) -> list[dict]:
    rows = []
    for key in sorted(store):
        key = key if isinstance(key, tuple) else (key,)
        by_metric = task_store[key]
        for metric in metric_names:
            samples = np.asarray(store[key][metric], dtype=np.float64)
            task_means = [float(np.mean(values)) for _, values in sorted(by_metric[metric].items())]
            seed_text = repr((key, metric)).encode('utf-8')
            seed = int.from_bytes(hashlib.sha256(seed_text).digest()[:4], 'little')
            ci_for_cell = ci and len(key) >= 3 and key[2] == 'all'
            low, high = bootstrap_mean_ci(task_means, seed=seed) if ci_for_cell else (None, None)
            row = dict(zip(key_columns, key))
            row.update({
                'metric': metric, 'n_predictions': int(samples.size), 'n_tasks': len(task_means),
                'sample_weighted_mean': float(np.mean(samples)),
                'sample_median': float(np.quantile(samples, 0.5)),
                'sample_p95': float(np.quantile(samples, 0.95)),
                'task_equal_mean': float(np.mean(task_means)) if task_means else None,
                'task_bootstrap_ci95_low': low, 'task_bootstrap_ci95_high': high,
            })
            rows.append(row)
    return rows


def paired_phase_contrasts(stage_task_samples: dict) -> list[dict]:
    """Compare Full realized phases within the same suite/task cluster."""
    phase_task_values = defaultdict(lambda: defaultdict(list))
    for (variant, suite, phase, _offset), metrics in stage_task_samples.items():
        for metric, task_values in metrics.items():
            if metric not in ('translation_command_l2', 'rotation_command_l2',
                              'gripper_disagreement'):
                continue
            for task, values in task_values.items():
                phase_task_values[(variant, suite, phase, metric)][task].extend(
                    float(value) for value in values)

    rows = []
    phase_pairs = (('approach', 'contact_grasp'), ('approach', 'transport'),
                   ('approach', 'alignment_release'))
    variants = sorted({key[0] for key in phase_task_values})
    suites = sorted({key[1] for key in phase_task_values})
    metrics = ('translation_command_l2', 'rotation_command_l2', 'gripper_disagreement')
    for variant in variants:
        for suite in suites:
            for phase_a, phase_b in phase_pairs:
                for metric in metrics:
                    values_a = phase_task_values.get((variant, suite, phase_a, metric), {})
                    values_b = phase_task_values.get((variant, suite, phase_b, metric), {})
                    common_tasks = sorted(set(values_a) & set(values_b))
                    deltas = [float(np.mean(values_b[task]) - np.mean(values_a[task]))
                              for task in common_tasks]
                    seed_key = repr((variant, suite, phase_a, phase_b, metric)).encode('utf-8')
                    seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:4], 'little')
                    low, high = bootstrap_mean_ci(deltas, seed) if len(deltas) >= 3 else (None, None)
                    rows.append({
                        'variant_id': variant, 'suite': suite,
                        'phase_a': phase_a, 'phase_b': phase_b, 'metric': metric,
                        'delta_definition': 'phase_b minus phase_a; positive means larger difference in phase_b',
                        'n_paired_tasks': len(common_tasks),
                        'phase_a_predictions': sum(len(values_a[task]) for task in common_tasks),
                        'phase_b_predictions': sum(len(values_b[task]) for task in common_tasks),
                        'task_equal_delta_mean': float(np.mean(deltas)) if deltas else None,
                        'task_equal_delta_median': float(np.median(deltas)) if deltas else None,
                        'task_bootstrap_ci95_low': low, 'task_bootstrap_ci95_high': high,
                        'bootstrap_replicates': BOOTSTRAP_REPLICATES if low is not None else 0,
                        'bootstrap_unit': 'suite::task; all episodes/inits for a task kept together',
                    })
    return rows


def load_full_phase_maps() -> dict[tuple[str, int, int], dict[int, str]]:
    maps = {}
    for suite in SUITES:
        run_dir = RESULTS / 'closed_loop/Full18_D3_R5_FORMAL' / suite / 'run'
        episodes = json.loads((run_dir / 'episode_log.json').read_text())
        if len(episodes) != 50:
            raise RuntimeError(f'{suite}: Full episode coverage is {len(episodes)}, expected 50')
        for episode in episodes:
            telemetry = json_lines(run_dir / episode['telemetry_path'])
            step_map = {}
            for item in telemetry:
                if item.get('when') not in ('post_action', None):
                    continue
                step_map[int(item['step'])] = coarse_phase(item.get('semantic_phase'))
            maps[(suite, int(episode['task_id']), int(episode['episode_idx']))] = step_map
    return maps


def load_expected_selection():
    jobs, selection_hash, audit_hash = frozen_jobs()
    status_path = SHADOW_STATUS
    status = json.loads(status_path.read_text())
    if (not status.get('complete') or status.get('active') or status.get('pending') or
            status.get('failures') or status.get('selection_sha256') != selection_hash or
            status.get('d3_audit_sha256') != audit_hash or status.get('validated_jobs') != len(jobs)):
        raise RuntimeError('Shadow scheduler status is not a complete, clean run for the frozen selection')
    for job in jobs:
        if not validate_job(job):
            raise RuntimeError(f"Shadow prediction validation failed for {job['job_id']}")
    return jobs, selection_hash, status


def read_prediction_records(job: dict):
    path = SHADOW_ROOT / job['variant_id'] / job['suite'] / 'predictions.jsonl'
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f'Malformed shadow record at {path}:{line_number}') from exc
            if row.get('variant_id') != job['variant_id'] or row.get('suite') != job['suite']:
                raise RuntimeError(f'Shadow record identity mismatch at {path}:{line_number}')
            if row.get('environment_action_source') != 'Full18 only; this worker never calls env.step':
                raise RuntimeError(f'Unexpected variant execution provenance at {path}:{line_number}')
            yield row


def command_scales(jobs: list[dict]) -> tuple[float, float, dict]:
    # Calibrate the illustrative threshold grid from Full actions, task0-5 only.
    first_variant = jobs[0]['variant_id']
    per_task_t: dict[str, list[float]] = defaultdict(list)
    per_task_r: dict[str, list[float]] = defaultdict(list)
    for job in jobs:
        if job['variant_id'] != first_variant:
            continue
        for row in read_prediction_records(job):
            if int(row['task_id']) > 5:
                continue
            full = np.asarray(row['full_actions'], dtype=np.float32)
            task = task_key(row['suite'], int(row['task_id']))
            per_task_t[task].extend(np.linalg.norm(full[:5, :3], axis=1).tolist())
            per_task_r[task].extend(np.linalg.norm(full[:5, 3:6], axis=1).tolist())
    task_t = {task: float(np.quantile(values, 0.5)) for task, values in per_task_t.items() if values}
    task_r = {task: float(np.quantile(values, 0.5)) for task, values in per_task_r.items() if values}
    if len(task_t) < 12 or len(task_r) < 12:
        raise RuntimeError(f'Only {min(len(task_t), len(task_r))} search tasks available for threshold calibration')
    scale_t = float(np.median(list(task_t.values())))
    scale_r = float(np.median(list(task_r.values())))
    if not math.isfinite(scale_t) or not math.isfinite(scale_r) or scale_t <= 0 or scale_r <= 0:
        raise RuntimeError('Cannot calibrate a positive finite action-command scale')
    return scale_t, scale_r, {'task_translation_medians': task_t, 'task_rotation_medians': task_r}


def render_figures(offset_rows: list[dict], stage_rows: list[dict], variants: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    colors = plt.get_cmap('tab10')
    for suite in SUITES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        for index, variant in enumerate(variants):
            for ax, metric, title in ((axes[0], 'translation_command_l2', 'Translation command Δ (L2)'),
                                      (axes[1], 'rotation_command_l2', 'Rotation command Δ (L2)')):
                rows = [row for row in offset_rows if row['suite'] == suite and
                        row['variant_id'] == variant and row['replan_phase'] == 'all' and
                        row['metric'] == metric]
                rows.sort(key=lambda row: int(row['offset']))
                if not rows:
                    continue
                x = np.asarray([int(row['offset']) for row in rows])
                mean = np.asarray([float(row['sample_weighted_mean']) for row in rows])
                p95 = np.asarray([float(row['sample_p95']) for row in rows])
                ax.plot(x, mean, color=colors(index), label=variant, linewidth=1.8)
                ax.fill_between(x, mean, p95, color=colors(index), alpha=0.12)
                ax.set_title(title)
                ax.set_xlabel('Action offset (1–50; command space)')
                ax.grid(alpha=0.22)
        axes[0].set_ylabel('Difference; not physical XYZ error')
        axes[0].legend(fontsize=8, loc='best')
        fig.suptitle(f'{suite}: Full vs layer-skip predictions at shared Full observations')
        fig.tight_layout()
        fig.savefig(out / f'offset50_{suite}.png', dpi=180)
        plt.close(fig)

    phases = ('approach', 'contact_grasp', 'transport', 'alignment_release')
    for variant in variants:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
        for ax, metric, title in ((axes[0], 'translation_command_l2', 'Translation command Δ'),
                                  (axes[1], 'rotation_command_l2', 'Rotation command Δ')):
            for phase_index, phase in enumerate(phases):
                rows = [row for row in stage_rows if row['variant_id'] == variant and
                        row['suite'] != 'ALL' and row['actual_full_phase'] == phase and
                        row['metric'] == metric]
                rows.sort(key=lambda row: int(row['offset']))
                if not rows:
                    continue
                x = np.asarray([int(row['offset']) for row in rows])
                y = np.asarray([float(row['sample_weighted_mean']) for row in rows])
                ax.plot(x, y, marker='o', label=phase, color=colors(phase_index))
            ax.set_xticks(range(1, 6))
            ax.set_xlabel('Executed action offset within Full 5-step chunk')
            ax.set_title(title)
            ax.grid(alpha=0.22)
        axes[0].set_ylabel('Difference; command units')
        axes[0].legend(fontsize=8)
        fig.suptitle(f'{variant}: prediction difference by realized Full phase (offsets 1–5 only)')
        fig.tight_layout()
        fig.savefig(out / f'prefix5_actual_phase_{variant}.png', dpi=180)
        plt.close(fig)


def render_k_figure(k_rows: list[dict], variants: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    colors = plt.get_cmap('tab10')
    horizons = np.asarray(K_THRESHOLDS)
    for ax, suite in zip(axes.flat, SUITES):
        for index, variant in enumerate(variants):
            rows = [row for row in k_rows if row['variant_id'] == variant and row['suite'] == suite and
                    row['replan_phase'] == 'all' and
                    math.isclose(float(row['translation_scale_factor']), 0.05) and
                    math.isclose(float(row['rotation_scale_factor']), 0.05)]
            rows.sort(key=lambda row: int(row['minimum_prefix_k']))
            if rows:
                y = [float(row['task_equal_probability']) for row in rows]
                ax.plot(horizons, y, marker='o', label=variant, color=colors(index))
        ax.set_title(suite)
        ax.set_xticks(horizons)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.22)
        ax.set_xlabel('Minimum consistent prefix K')
        ax.set_ylabel('Task-equal P(K ≥ threshold)')
    axes.flat[0].legend(fontsize=8, loc='best')
    fig.suptitle('Reference-consistent prefix diagnostic (5% action-scale grid; not a safety limit)')
    fig.tight_layout()
    fig.savefig(out / 'consistent_prefix_k_5pct.png', dpi=180)
    plt.close(fig)


def main() -> int:
    jobs, selection_hash, scheduler_status = load_expected_selection()
    variants = list(dict.fromkeys(job['variant_id'] for job in jobs))
    phase_maps = load_full_phase_maps()

    offset_samples = defaultdict(lambda: defaultdict(list))
    offset_task_samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    stage_samples = defaultdict(lambda: defaultdict(list))
    stage_task_samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    prefix_samples = defaultdict(lambda: defaultdict(list))
    prefix_task_samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scale_t, scale_r, scale_details = command_scales(jobs)
    threshold_pairs = [(a, b, scale_t * a, scale_r * b)
                       for a in SCALE_FACTORS for b in SCALE_FACTORS]

    for job in jobs:
        for row in read_prediction_records(job):
            variant = job['variant_id']
            suite = job['suite']
            task_id, episode_idx = int(row['task_id']), int(row['episode_idx'])
            task = task_key(suite, task_id)
            replan_phase = coarse_phase(row.get('semantic_phase'))
            diff = action_differences(row['full_actions'], row['variant_actions'])
            phase_map = phase_maps[(suite, task_id, episode_idx)]
            env_step = int(row['environment_action_step'])

            for offset in OFFSETS:
                index = offset - 1
                values = {metric: float(metric_values[index])
                          for metric, metric_values in diff.items()}
                for suite_group in (suite, 'ALL'):
                    for phase in ('all', replan_phase):
                        key = (variant, suite_group, phase, offset)
                        add_values(offset_samples, offset_task_samples, key, task, values)
                if offset <= 5:
                    actual_phase = phase_map.get(env_step + offset, 'unknown')
                    if actual_phase != 'unknown':
                        for suite_group in (suite, 'ALL'):
                            key = (variant, suite_group, actual_phase, offset)
                            add_values(stage_samples, stage_task_samples, key, task, values)

            for horizon in PREFIXES:
                values = prefix_metrics(diff, horizon)
                add_prefix_metrics(prefix_samples, prefix_task_samples, variant, suite,
                                   replan_phase, horizon, task, values)

    offset_metric_names = ('translation_command_l2', 'rotation_command_l2',
                           'continuous6_component_rmse', 'gripper_abs_error',
                           'gripper_disagreement')
    prefix_metric_names = ('translation_command_l2_mean', 'translation_command_l2_p95_within_prefix',
                           'translation_command_l2_max_within_prefix', 'rotation_command_l2_mean',
                           'rotation_command_l2_p95_within_prefix', 'rotation_command_l2_max_within_prefix',
                           'continuous6_component_rmse', 'gripper_disagreement_rate',
                           'any_gripper_disagreement')
    offset_rows = summarize_cells(offset_samples, offset_task_samples,
                                  ('variant_id', 'suite', 'replan_phase', 'offset'), offset_metric_names)
    stage_rows = summarize_cells(stage_samples, stage_task_samples,
                                 ('variant_id', 'suite', 'actual_full_phase', 'offset'), offset_metric_names)
    prefix_rows = summarize_cells(prefix_samples, prefix_task_samples,
                                  ('variant_id', 'suite', 'replan_phase', 'horizon'),
                                  prefix_metric_names, ci=True)
    phase_contrast_rows = paired_phase_contrasts(stage_task_samples)

    # Re-scan compact JSONL files for prefix-consistency K; no physical safety threshold is asserted.
    k_samples = defaultdict(lambda: defaultdict(list))
    k_task_samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for job in jobs:
        for row in read_prediction_records(job):
            diff = action_differences(row['full_actions'], row['variant_actions'])
            phase = coarse_phase(row.get('semantic_phase'))
            task = task_key(job['suite'], int(row['task_id']))
            for ft, fr, tau_t, tau_r in threshold_pairs:
                k_value = consistency_k(diff, tau_t, tau_r)
                for suite_group in (job['suite'], 'ALL'):
                    for phase_group in ('all', phase):
                        key = (job['variant_id'], suite_group, phase_group, ft, fr)
                        k_samples[key]['K'].append(float(k_value))
                        k_task_samples[key]['K'][task].append(float(k_value))

    k_rows = []
    for key in sorted(k_samples):
        variant, suite, phase, factor_t, factor_r = key
        task_values = k_task_samples[key]['K']
        for threshold in K_THRESHOLDS:
            sample_probability = float(np.mean(np.asarray(k_samples[key]['K']) >= threshold))
            task_probability = [float(np.mean(np.asarray(values) >= threshold))
                                for _, values in sorted(task_values.items())]
            k_rows.append({
                'variant_id': variant, 'suite': suite, 'replan_phase': phase,
                'translation_scale_factor': factor_t, 'rotation_scale_factor': factor_r,
                'translation_threshold_command_units': scale_t * factor_t,
                'rotation_threshold_command_units': scale_r * factor_r,
                'minimum_prefix_k': threshold, 'n_predictions': len(k_samples[key]['K']),
                'n_tasks': len(task_probability), 'sample_weighted_probability': sample_probability,
                'task_equal_probability': float(np.mean(task_probability)) if task_probability else None,
            })

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    write_csv(ANALYSIS / 'shadow_offset_summary.csv', offset_rows,
              ['variant_id', 'suite', 'replan_phase', 'offset', 'metric', 'n_predictions', 'n_tasks',
               'sample_weighted_mean', 'sample_median', 'sample_p95', 'task_equal_mean',
               'task_bootstrap_ci95_low', 'task_bootstrap_ci95_high'])
    write_csv(ANALYSIS / 'shadow_executed_prefix_phase_summary.csv', stage_rows,
              ['variant_id', 'suite', 'actual_full_phase', 'offset', 'metric', 'n_predictions', 'n_tasks',
               'sample_weighted_mean', 'sample_median', 'sample_p95', 'task_equal_mean',
               'task_bootstrap_ci95_low', 'task_bootstrap_ci95_high'])
    write_csv(ANALYSIS / 'shadow_prefix_summary.csv', prefix_rows,
              ['variant_id', 'suite', 'replan_phase', 'horizon', 'metric', 'n_predictions', 'n_tasks',
               'sample_weighted_mean', 'sample_median', 'sample_p95', 'task_equal_mean',
               'task_bootstrap_ci95_low', 'task_bootstrap_ci95_high'])
    write_csv(ANALYSIS / 'shadow_paired_phase_contrasts.csv', phase_contrast_rows,
              ['variant_id', 'suite', 'phase_a', 'phase_b', 'metric', 'delta_definition',
               'n_paired_tasks', 'phase_a_predictions', 'phase_b_predictions',
               'task_equal_delta_mean', 'task_equal_delta_median',
               'task_bootstrap_ci95_low', 'task_bootstrap_ci95_high', 'bootstrap_replicates',
               'bootstrap_unit'])
    write_csv(ANALYSIS / 'shadow_consistent_prefix_k.csv', k_rows,
              ['variant_id', 'suite', 'replan_phase', 'translation_scale_factor',
               'rotation_scale_factor', 'translation_threshold_command_units',
               'rotation_threshold_command_units', 'minimum_prefix_k', 'n_predictions', 'n_tasks',
               'sample_weighted_probability', 'task_equal_probability'])

    thresholds = {
        'definition': 'K is the longest consecutive prefix with translation/rotation command L2 errors below thresholds and gripper sign agreement.',
        'translation_action_scale_task_equal_median': scale_t,
        'rotation_action_scale_task_equal_median': scale_r,
        'calibration_tasks': sorted(scale_details['task_translation_medians']),
        'calibration_source': 'Full18 D3, tasks 0-5, offsets 1-5 only; per-task median then equal-task median',
        'factors': list(SCALE_FACTORS), 'threshold_pairs': len(threshold_pairs),
        'warning': 'Diagnostic reference-consistency thresholds in policy command units; not a physical safety boundary or deployable gate.',
    }
    dump_json(ANALYSIS / 'shadow_threshold_reference.json', thresholds)
    render_figures(offset_rows, stage_rows, variants, ANALYSIS / 'figures')
    render_k_figure(k_rows, variants, ANALYSIS / 'figures')

    total_records = sum(int(json.loads((SHADOW_ROOT / job['variant_id'] / job['suite'] /
                                       'validated.json').read_text())['replan_prediction_count'])
                        for job in jobs)
    audit = {
        'complete': True, 'experiment_id': 'experiment11_shadow_prefix_d3',
        'validated_jobs': len(jobs), 'expected_jobs': len(jobs),
        'variants': variants, 'suites': list(SUITES),
        'replan_prediction_records': total_records, 'action_horizon': 50,
        'primary_prefix_offsets': [1, 2, 3, 4, 5], 'diagnostic_offsets': list(OFFSETS),
        'environment_steps_executed_by_variants': 0,
        'selection_sha256': selection_hash,
        'scheduler_status_sha256': sha256(SHADOW_STATUS),
        'source_hashes': {job['job_id']: sha256(
            SHADOW_ROOT / job['variant_id'] / job['suite'] / 'predictions.jsonl') for job in jobs},
    }
    report = [
        '# Experiment 4 Shadow Replay — D3', '',
        f"Validated {len(jobs)} variant×suite jobs and {total_records} matched replan prediction pairs.",
        'The simulator executed only Full18-D3 actions in 5-step chunks. Variants were inference-only; this analysis does not report variant success rate or actual variant XYZ.',
        '', '## Interpretation limits', '',
        '- Differences are in policy action-command space, not physical EEF displacement or pose error.',
        '- Full18 is a reference policy, not ground-truth optimal action.',
        '- Offsets 1–5 correspond to the executed Full chunk; offsets 6–50 are prediction diagnostics after which Full would replan.',
        '- K thresholds are an illustrative consistency grid calibrated on Full tasks 0–5; they are not physical safety limits.',
        '- Episodes and replanning observations within a task are correlated; summaries include task-equal aggregation.',
        '', '## Primary outputs', '',
        '- `shadow_prefix_summary.csv`: h=1/2/3/4/5/8/12 prefix errors, gripper disagreement and task-cluster intervals.',
        '- `shadow_executed_prefix_phase_summary.csv`: actual Full phase for offsets 1–5 only.',
        '- `shadow_paired_phase_contrasts.csv`: same-suite/task phase contrasts with task-cluster bootstrap intervals.',
        '- `shadow_offset_summary.csv`: every predicted offset 1–50, stratified by phase at the replan point.',
        '- `shadow_consistent_prefix_k.csv` and `shadow_threshold_reference.json`: diagnostic K curves and frozen scale source.',
        '- `figures/offset50_*.png`, `prefix5_actual_phase_*.png`, and `consistent_prefix_k_5pct.png`: command-difference visualizations; no fabricated variant trajectory.',
        '', '## Overall equal-task summary (five-step prefix)', '',
        '| Variant | Translation command L2 | Rotation command L2 | Gripper disagreement | P(K≥5), 5%/5% grid |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    prefix_lookup = {(row['variant_id'], row['suite'], row['replan_phase'], int(row['horizon']), row['metric']): row
                     for row in prefix_rows}
    k_lookup = {(row['variant_id'], row['suite'], row['replan_phase'],
                 float(row['translation_scale_factor']), float(row['rotation_scale_factor']),
                 int(row['minimum_prefix_k'])): row for row in k_rows}
    for variant in variants:
        t = prefix_lookup.get((variant, 'ALL', 'all', 5, 'translation_command_l2_mean'))
        r = prefix_lookup.get((variant, 'ALL', 'all', 5, 'rotation_command_l2_mean'))
        g = prefix_lookup.get((variant, 'ALL', 'all', 5, 'gripper_disagreement_rate'))
        probability = k_lookup.get((variant, 'ALL', 'all', 0.05, 0.05, 5))
        fmt = lambda item: 'n/a' if item is None or item.get('task_equal_mean') is None else f"{item['task_equal_mean']:.5f}"
        probability_text = 'n/a' if probability is None else f"{probability['task_equal_probability']:.3f}"
        report.append(f'| {variant} | {fmt(t)} | {fmt(r)} | {fmt(g)} | {probability_text} |')
    report.extend(['', '## Same-task phase contrasts (Full executed offsets 1–5)', '',
                   'Positive values mean the listed phase has a larger Full–variant command difference than approach. Each task is one cluster; all init episodes within that suite/task stay together.', '',
                   '| Variant | Comparison | Metric | Paired tasks | Task-equal Δ (95% task bootstrap CI) |',
                   '| --- | --- | --- | ---: | ---: |'])
    contrast_lookup = {(row['variant_id'], row['suite'], row['phase_a'], row['phase_b'], row['metric']): row
                       for row in phase_contrast_rows}
    contrast_specs = (('approach', 'alignment_release', 'translation_command_l2'),
                      ('approach', 'alignment_release', 'rotation_command_l2'),
                      ('approach', 'contact_grasp', 'gripper_disagreement'))
    for variant in variants:
        for phase_a, phase_b, metric in contrast_specs:
            row = contrast_lookup.get((variant, 'ALL', phase_a, phase_b, metric))
            if row is None or row['task_equal_delta_mean'] is None:
                formatted = 'n/a'
                n_tasks = 0 if row is None else row['n_paired_tasks']
            else:
                low, high = row['task_bootstrap_ci95_low'], row['task_bootstrap_ci95_high']
                interval = 'n/a' if low is None or high is None else f'[{low:.5f}, {high:.5f}]'
                formatted = f"{row['task_equal_delta_mean']:.5f} {interval}"
                n_tasks = row['n_paired_tasks']
            report.append(f'| {variant} | {phase_a} → {phase_b} | {metric} | {n_tasks} | {formatted} |')
    report.extend(['', 'These are reference-policy command differences, not variant task-success rates or physical error bounds.'])
    (ANALYSIS / 'report.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    audit['outputs'] = [str(path) for path in sorted(ANALYSIS.rglob('*'))
                        if path.is_file() and path.name != 'completion_audit.json']
    dump_json(ANALYSIS / 'completion_audit.json', audit)
    print(json.dumps({'analysis': str(ANALYSIS), 'jobs': len(jobs),
                      'replan_prediction_records': total_records,
                      'variants': variants, 'audit': str(ANALYSIS / 'completion_audit.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
