"""Audit and summarize the completed Experiment 11 D3 closed-loop matrix."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiment11.protocol import RESULTS, ROOT, SUITES
from scripts.experiment10.common import dump_json, json_lines, sha256


BASELINE = 'Full18_D3_R5_FORMAL'
CUSTOM_V13 = 'V13_Custom_D3_R5_FORMAL'
EXPECTED_EPISODES_PER_SUITE = 50
EXPECTED_TASKS = tuple(range(10))
EXPECTED_INITS = tuple(range(5))
PHASES = ('initial', 'approach', 'contact', 'grasp_or_manipulate', 'transport',
          'goal_alignment', 'release_or_final_action', 'success', 'unknown')
PHASE_COLORS = ('#dce9f5', '#a6c8e2', '#76b7b2', '#59a14f', '#edc948',
                '#f28e2b', '#e15759', '#b07aa1', '#bab0ac')
OUTCOME = ('both_success', 'baseline_only_success', 'variant_only_success', 'both_failure')
PERSISTENT_DISTANCE_M = 0.02
PERSISTENT_STEPS = 3


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def condition_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    config = read_json(RESULTS / 'protocol/closed_loop_config_formal.json')
    manifest = read_json(RESULTS / 'protocol/candidate_manifest.json')
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in manifest['candidates']:
        condition = candidate['condition']['name'] + '_FORMAL'
        by_id[candidate['variant_id']] = {
            **candidate,
            'condition_name': condition,
            'family': candidate['family'],
        }
    expected_names = {BASELINE, CUSTOM_V13} | {row['condition_name'] for row in by_id.values()}
    actual_names = {row['name'] for row in config['conditions']}
    if actual_names != expected_names or len(config['conditions']) != 31:
        raise RuntimeError('Frozen candidate manifest and formal config differ')
    return config['conditions'], by_id


def load_suite(condition: dict[str, Any], suite: str, *, require_validated: bool) -> dict[tuple[int, int], dict[str, Any]]:
    run_dir = RESULTS / 'closed_loop' / condition['name'] / suite / 'run'
    log_path = run_dir / 'episode_log.json'
    if not log_path.is_file():
        raise RuntimeError(f'Missing episode log: {log_path}')
    if require_validated:
        valid_path = run_dir / 'validated.json'
        if not valid_path.is_file():
            raise RuntimeError(f'Missing validated.json: {valid_path}')
        valid = read_json(valid_path)
        if not valid.get('complete') or valid.get('episodes') != EXPECTED_EPISODES_PER_SUITE:
            raise RuntimeError(f'Invalid validation record: {valid_path}')
    rows = read_json(log_path)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        key = (int(row['task_id']), int(row['episode_idx']))
        if key in result:
            raise RuntimeError(f'Duplicate task/init {key} in {log_path}')
        result[key] = row
    expected = {(task, init) for task in EXPECTED_TASKS for init in EXPECTED_INITS}
    if set(result) != expected:
        raise RuntimeError(f'Coverage mismatch in {log_path}: {len(result)}/{len(expected)}')
    if any(row.get('failure_reason') or row.get('episode_error') for row in rows):
        raise RuntimeError(f'Infrastructure/episode exception in {log_path}')
    return result


def input_artifact_coverage(run_dir: Path, episodes: dict[tuple[int, int], dict[str, Any]]) -> dict[str, int]:
    expected = present = 0
    for row in episodes.values():
        infer_path = run_dir / row['infer_path']
        for infer in json_lines(infer_path):
            expected += 1
            metadata = infer.get('experiment11', {})
            artifact = Path(metadata.get('full_input_path', ''))
            if metadata.get('condition') == BASELINE and artifact.is_file():
                present += 1
    return {'expected': expected, 'present': present, 'missing': expected - present}


def outcome_label(base_success: bool, variant_success: bool) -> str:
    if base_success and variant_success:
        return 'both_success'
    if base_success:
        return 'baseline_only_success'
    if variant_success:
        return 'variant_only_success'
    return 'both_failure'


def paired_summary(
    base: dict[str, dict[tuple[int, int], dict[str, Any]]],
    variant: dict[str, dict[tuple[int, int], dict[str, Any]]],
    suites: tuple[str, ...], seed_offset: int,
) -> dict[str, Any]:
    suite_effects: dict[str, list[float]] = {}
    discordance = Counter()
    n = 0
    for suite in suites:
        values_by_task: dict[int, list[float]] = defaultdict(list)
        for key, base_row in base[suite].items():
            var_row = variant[suite][key]
            values_by_task[key[0]].append(float(bool(var_row['success'])) - float(bool(base_row['success'])))
            discordance[outcome_label(bool(base_row['success']), bool(var_row['success']))] += 1
            n += 1
        suite_effects[suite] = [float(np.mean(values_by_task[t])) for t in sorted(values_by_task)]
    if len(suites) == 1:
        clustered = np.asarray([suite_effects[suites[0]]], dtype=float)
    else:
        clustered = np.asarray([suite_effects[suite] for suite in suites], dtype=float)
    rng = np.random.default_rng(20261010 + seed_offset)
    draws = rng.integers(0, clustered.shape[1], size=(10000, clustered.shape[0], clustered.shape[1]))
    sampled = clustered[np.arange(clustered.shape[0])[None, :, None], draws]
    bootstrap = sampled.mean(axis=(1, 2))
    estimate = float(clustered.mean())
    return {
        'episodes': n,
        'variant_minus_baseline_success_rate': estimate,
        'task_cluster_bootstrap_95_low': float(np.quantile(bootstrap, 0.025)),
        'task_cluster_bootstrap_95_high': float(np.quantile(bootstrap, 0.975)),
        'bootstrap_replicates': 10000,
        'bootstrap_unit': 'task, with all five inits kept together; suite-stratified when pooled',
        **{name: int(discordance[name]) for name in OUTCOME},
    }


def telemetry_metrics(base_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]]) -> dict[str, Any]:
    base_by_step = {int(row['step']): row for row in base_rows}
    variant_by_step = {int(row['step']): row for row in variant_rows}
    shared = sorted(base_by_step.keys() & variant_by_step.keys())
    if not shared:
        raise RuntimeError('Paired telemetry has no common environment steps')
    distances = np.asarray([
        np.linalg.norm(np.asarray(base_by_step[step]['eef_pos'], dtype=float) -
                       np.asarray(variant_by_step[step]['eef_pos'], dtype=float))
        for step in shared
    ], dtype=float)
    first_sustained = None
    for start in range(len(shared) - PERSISTENT_STEPS + 1):
        window = shared[start:start + PERSISTENT_STEPS]
        if window == list(range(window[0], window[0] + PERSISTENT_STEPS)) and np.all(
                distances[start:start + PERSISTENT_STEPS] >= PERSISTENT_DISTANCE_M):
            first_sustained = int(window[0])
            break
    base_contacts = []
    for row in base_rows:
        contacts = row.get('robot_contacts') or {}
        primary = row.get('primary_objects') or list(contacts)
        if any(bool(contacts.get(name, False)) for name in primary):
            base_contacts.append(int(row['step']))
    first_contact = min(base_contacts) if base_contacts else None
    phase_at_divergence = None
    if first_sustained is not None:
        phase_at_divergence = base_by_step.get(first_sustained, {}).get('semantic_phase')
    return {
        'shared_step_count': len(shared),
        'baseline_step_count': len(base_rows),
        'variant_step_count': len(variant_rows),
        'mean_xyz_deviation_m': float(distances.mean()),
        'p95_xyz_deviation_m': float(np.quantile(distances, 0.95)),
        'max_xyz_deviation_m': float(distances.max()),
        'max_deviation_step': int(shared[int(np.argmax(distances))]),
        'first_persistent_2cm_deviation_step': first_sustained,
        'persistent_rule': f'{PERSISTENT_STEPS} consecutive shared steps at >= {PERSISTENT_DISTANCE_M:.3f} m',
        'baseline_first_primary_object_contact_step': first_contact,
        'divergence_before_baseline_contact': (
            None if first_sustained is None or first_contact is None else first_sustained < first_contact
        ),
        'phase_at_first_persistent_deviation': phase_at_divergence,
    }


def write_heatmap(rows: list[dict[str, Any]], suites: tuple[str, ...], path: Path) -> None:
    ordered = sorted(rows, key=lambda r: (0 if r['family'] == 'Full' else 1,
                                          {'V16': 0, 'V15': 1, 'V14': 2, 'V13': 3}.get(r['family'], 9),
                                          r['variant_id']))
    values = np.asarray([[next(r['success_rate'] for r in ordered
                               if r['variant_id'] == identifier and r['suite'] == suite)
                          for suite in suites]
                         for identifier in dict.fromkeys(r['variant_id'] for r in ordered)], dtype=float)
    identifiers = list(dict.fromkeys(r['variant_id'] for r in ordered))
    fig, ax = plt.subplots(figsize=(9, max(8, len(identifiers) * 0.34)), layout='constrained')
    image = ax.imshow(values * 100, vmin=0, vmax=100, aspect='auto', cmap='YlGnBu')
    ax.set_xticks(range(len(suites)), suites, rotation=20, ha='right')
    ax.set_yticks(range(len(identifiers)), identifiers)
    for y, identifier in enumerate(identifiers):
        for x, suite in enumerate(suites):
            value = values[y, x] * 100
            color = 'white' if value < 35 or value > 78 else '#222222'
            marker = '*' if value > 90 else ''
            ax.text(x, y, f'{value:.0f}%{marker}', ha='center', va='center', color=color, fontsize=8)
    ax.set_title('D3 closed-loop success by variant and LIBERO suite (* > 90%)')
    fig.colorbar(image, ax=ax, label='Success rate (%)')
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_stage_plot(stage_rows: list[dict[str, Any]], path: Path) -> None:
    conditions = sorted({r['variant_id'] for r in stage_rows}, key=lambda x: x)
    fig, ax = plt.subplots(figsize=(13, max(7, len(conditions) * 0.38)), layout='constrained')
    left = np.zeros(len(conditions), dtype=float)
    for phase, color in zip(PHASES, PHASE_COLORS):
        counts = np.asarray([sum(r['count'] for r in stage_rows
                                 if r['variant_id'] == condition and r['cohort'] == 'new_failure' and
                                 r['furthest_phase'] == phase) for condition in conditions], dtype=float)
        if not counts.any():
            continue
        ax.barh(conditions, counts, left=left, color=color, label=phase,
                edgecolor='white', linewidth=0.7, height=0.7)
        left += counts
    ax.set_title('Furthest phase among paired new failures (not causal onset)')
    ax.set_xlabel('Episodes (Full success, variant failure)')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.055), ncol=4, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def write_divergence_plot(rows: list[dict[str, Any]], path: Path) -> list[dict[str, Any]]:
    variants = sorted({row['variant_id'] for row in rows})
    categories = ('before_baseline_contact', 'at_or_after_contact', 'not_observed_or_unknown')
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row['cohort'] != 'baseline_only_success':
            continue
        if row['divergence_before_baseline_contact'] is True:
            category = categories[0]
        elif row['divergence_before_baseline_contact'] is False:
            category = categories[1]
        else:
            category = categories[2]
        counts[(row['variant_id'], category)] += 1
    colors = ('#4e79a7', '#f28e2b', '#bab0ac')
    fig, ax = plt.subplots(figsize=(12, max(7, len(variants) * 0.38)), layout='constrained')
    left = np.zeros(len(variants), dtype=float)
    for category, color in zip(categories, colors):
        values = np.asarray([counts[(variant, category)] for variant in variants], dtype=float)
        if not values.any():
            continue
        ax.barh(variants, values, left=left, color=color, label=category,
                edgecolor='white', linewidth=0.7, height=0.7)
        left += values
    ax.set_title('First persistent 2 cm trajectory divergence in paired new failures')
    ax.set_xlabel('Episodes (Full success, variant failure); trajectory diagnostic, not causal proof')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.055), ncol=3, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return [{'variant_id': variant, 'category': category, 'count': int(counts[(variant, category)])}
            for variant in variants for category in categories]


def load_trajectory(run_dir: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    return json_lines(run_dir / row['telemetry_path'])


def choose_example(records: list[dict[str, Any]]) -> dict[str, Any]:
    new_failures = [r for r in records if r['cohort'] == 'baseline_only_success']
    if new_failures:
        return max(new_failures, key=lambda r: (r['max_xyz_deviation_m'], -r['task_id'], -r['episode_idx']))
    mutual_success = [r for r in records if r['cohort'] == 'both_success']
    pool = mutual_success or records
    ranked = sorted(pool, key=lambda r: (r['max_xyz_deviation_m'], r['task_id'], r['episode_idx']))
    return ranked[len(ranked) // 2]


def plot_trajectory_examples(
    candidate_records: dict[tuple[str, str], list[dict[str, Any]]],
    episodes: dict[str, dict[str, dict[tuple[int, int], dict[str, Any]]]],
    output_root: Path,
    *, render_videos: bool,
) -> list[dict[str, Any]]:
    from scripts.experiment8.plot_paired_xyz import read_trajectory, compare_trajectories, plot_comparison

    examples: list[dict[str, Any]] = []
    video_renderer = Path(__file__).resolve().parents[1] / 'experiment8/render_xyz_video.py'
    video_python = Path(__file__).resolve().parents[2] / 'examples/libero/.venv/bin/python'
    if not video_python.is_file():
        video_python = Path(sys.executable)
    video_candidates: dict[str, dict[str, Any]] = {}
    for (variant_id, suite), records in sorted(candidate_records.items()):
        chosen = choose_example(records)
        condition_name = chosen['condition_name']
        key = (chosen['task_id'], chosen['episode_idx'])
        baseline_row = episodes[BASELINE][suite][key]
        variant_row = episodes[condition_name][suite][key]
        baseline_run = RESULTS / 'closed_loop' / BASELINE / suite / 'run'
        variant_run = RESULTS / 'closed_loop' / condition_name / suite / 'run'
        baseline_telemetry = baseline_run / baseline_row['telemetry_path']
        variant_telemetry = variant_run / variant_row['telemetry_path']
        b_rows, v_rows = read_trajectory(baseline_telemetry), read_trajectory(variant_telemetry)
        comparison = compare_trajectories(b_rows, v_rows, baseline_label='Full18-D3',
                                          variant_label=variant_id,
                                          baseline_path=str(baseline_telemetry),
                                          variant_path=str(variant_telemetry))
        prefix = output_root / 'figures/trajectory_pairs' / variant_id / f'{suite}_task{key[0]:02d}_init{key[1]:02d}'
        plot_comparison(comparison, baseline_label='Full18-D3', variant_label=variant_id,
                        png_path=prefix.with_suffix('.png'), pdf_path=prefix.with_suffix('.pdf'))
        example = {
            'variant_id': variant_id, 'condition_name': condition_name, 'suite': suite,
            'task_id': key[0], 'episode_idx': key[1], 'cohort': chosen['cohort'],
            'selection_rule': 'largest XYZ deviation among paired new failures; otherwise median deviation among both-success pairs',
            'baseline_video': str(baseline_run / baseline_row['video_path']),
            'variant_video': str(variant_run / variant_row['video_path']),
            'baseline_telemetry': str(baseline_telemetry), 'variant_telemetry': str(variant_telemetry),
            'trajectory_png': str(prefix.with_suffix('.png')), 'trajectory_pdf': str(prefix.with_suffix('.pdf')),
            'max_xyz_deviation_m': chosen['max_xyz_deviation_m'],
        }
        examples.append(example)
        previous = video_candidates.get(variant_id)
        priority = (chosen['cohort'] == 'baseline_only_success', chosen['max_xyz_deviation_m'])
        previous_priority = ((previous['cohort'] == 'baseline_only_success', previous['max_xyz_deviation_m'])
                             if previous else (False, -math.inf))
        if priority > previous_priority:
            video_candidates[variant_id] = example

    rendered_video_paths: dict[str, str] = {}
    if render_videos:
        for variant_id, example in sorted(video_candidates.items()):
            output = output_root / 'videos/paired_xyz' / f'{variant_id}.mp4'
            sidecar = output.with_suffix('.json')
            if output.is_file() and sidecar.is_file():
                rendered_video_paths[variant_id] = str(output)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(video_python), str(video_renderer),
                       '--baseline-video', example['baseline_video'],
                       '--variant-video', example['variant_video'],
                       '--baseline-telemetry', example['baseline_telemetry'],
                       '--variant-telemetry', example['variant_telemetry'],
                       '--output', str(output), '--alignment', 'state-synchronous',
                       '--baseline-label', 'Full18-D3', '--variant-label', variant_id]
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode:
                raise RuntimeError(f'Video rendering failed for {variant_id}: {completed.stderr[-4000:]}')
            rendered_video_paths[variant_id] = str(output)
    for example in examples:
        example['paired_xyz_video'] = rendered_video_paths.get(example['variant_id'], '')
    return examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-videos', action='store_true', help='Generate trajectory figures but not paired MP4s')
    args = parser.parse_args()

    status_path = RESULTS / 'experiment11_closed_loop_status.json'
    status = read_json(status_path)
    if not status.get('complete') or status.get('active') or status.get('pending') or status.get('failures'):
        raise RuntimeError('Formal matrix is not terminal/clean; do not summarize incomplete results')
    conditions, candidates = condition_rows()
    episode_data: dict[str, dict[str, dict[tuple[int, int], dict[str, Any]]]] = {}
    run_validation: list[dict[str, Any]] = []
    for condition in conditions:
        condition_name = condition['name']
        episode_data[condition_name] = {}
        for suite in SUITES:
            rows = load_suite(condition, suite, require_validated=True)
            episode_data[condition_name][suite] = rows
            run_dir = RESULTS / 'closed_loop' / condition_name / suite / 'run'
            validation = read_json(run_dir / 'validated.json')
            run_validation.append({
                'condition': condition_name, 'suite': suite, 'episodes': len(rows),
                'validated': bool(validation.get('complete')),
                'validated_video_count': validation.get('full_video_decodes'),
                'episode_log_sha256': sha256(run_dir / 'episode_log.json'),
                'validated_json_sha256': sha256(run_dir / 'validated.json'),
            })
    analysis_root = RESULTS / 'analysis/closed_loop_d3'
    analysis_root.mkdir(parents=True, exist_ok=True)

    candidate_for_condition = {v['condition_name']: key for key, v in candidates.items()}
    summary_rows, task_rows, episode_rows = [], [], []
    condition_suite: dict[tuple[str, str], dict[str, Any]] = {}
    for condition in conditions:
        name = condition['name']
        family = 'Full' if name == BASELINE else 'V13-reference' if name == CUSTOM_V13 else name.split('_')[0]
        variant_id = 'Full18' if name == BASELINE else 'V13_Custom' if name == CUSTOM_V13 else candidate_for_condition[name]
        for suite in SUITES:
            rows = episode_data[name][suite]
            successes = sum(bool(row['success']) for row in rows.values())
            latency_values = [float(row['infer_latency_mean_ms']) for row in rows.values()
                              if row.get('infer_latency_mean_ms') is not None]
            policy_values = [float(row['policy_time_mean_ms']) for row in rows.values()
                             if row.get('policy_time_mean_ms') is not None]
            run_dir = RESULTS / 'closed_loop' / name / suite / 'run'
            input_counts = input_artifact_coverage(run_dir, rows) if name == BASELINE else {'expected': 0, 'present': 0, 'missing': 0}
            record = {
                'condition': name, 'variant_id': variant_id, 'family': family, 'suite': suite,
                'successes': successes, 'episodes': len(rows), 'success_rate': successes / len(rows),
                'over_90_percent': successes / len(rows) > 0.90,
                'mean_episode_infer_latency_ms_concurrent': float(np.mean(latency_values)) if latency_values else None,
                'p50_episode_infer_latency_ms_concurrent': float(np.median(latency_values)) if latency_values else None,
                'mean_episode_policy_time_ms_concurrent': float(np.mean(policy_values)) if policy_values else None,
                'full_input_artifacts_expected': input_counts['expected'],
                'full_input_artifacts_present': input_counts['present'],
                'full_input_artifacts_missing': input_counts['missing'],
            }
            condition_suite[(name, suite)] = record
            summary_rows.append(record)
            for task in EXPECTED_TASKS:
                task_successes = sum(bool(rows[(task, init)]['success']) for init in EXPECTED_INITS)
                task_rows.append({'condition': name, 'variant_id': variant_id, 'family': family,
                                  'suite': suite, 'task_id': task, 'successes': task_successes,
                                  'episodes': len(EXPECTED_INITS), 'success_rate': task_successes / len(EXPECTED_INITS)})
            for (task, init), row in sorted(rows.items()):
                episode_rows.append({'condition': name, 'variant_id': variant_id, 'family': family,
                                     'suite': suite, 'task_id': task, 'episode_idx': init,
                                     'success': int(bool(row['success'])), 'env_steps_taken': row.get('env_steps_taken'),
                                     'furthest_semantic_phase': row.get('furthest_semantic_phase') or 'unknown',
                                     'infer_latency_mean_ms': row.get('infer_latency_mean_ms'),
                                     'policy_time_mean_ms': row.get('policy_time_mean_ms'),
                                     'trace_path': row['trace_path'], 'telemetry_path': row['telemetry_path'],
                                     'video_path': row['video_path']})

    write_csv(analysis_root / 'suite_results.csv', summary_rows, list(summary_rows[0]))
    write_csv(analysis_root / 'task_results.csv', task_rows, list(task_rows[0]))
    write_csv(analysis_root / 'episode_results.csv', episode_rows, list(episode_rows[0]))
    write_csv(analysis_root / 'run_validation.csv', run_validation, list(run_validation[0]))
    over_90 = [row for row in summary_rows if row['over_90_percent'] and row['family'] not in ('Full', 'V13-reference')]
    write_csv(analysis_root / 'suite_over_90_percent.csv', over_90, list(summary_rows[0]))

    base = episode_data[BASELINE]
    custom = episode_data[CUSTOM_V13]
    paired_rows = []
    for index, (variant_id, candidate) in enumerate(sorted(candidates.items())):
        condition_name = candidate['condition_name']
        for compare_name, reference in [('Full18-D3', base)]:
            for suite in SUITES:
                values = paired_summary(base, episode_data[condition_name], (suite,), index * 8 + SUITES.index(suite))
                paired_rows.append({'variant_id': variant_id, 'family': candidate['family'],
                                    'comparison': compare_name, 'suite': suite, **values})
            values = paired_summary(base, episode_data[condition_name], SUITES, index * 8 + 7)
            paired_rows.append({'variant_id': variant_id, 'family': candidate['family'],
                                'comparison': compare_name, 'suite': 'ALL_EQUAL_SUITE', **values})
        if candidate['family'] == 'V13':
            for suite in SUITES:
                values = paired_summary(custom, episode_data[condition_name], (suite,), index * 8 + SUITES.index(suite) + 1000)
                paired_rows.append({'variant_id': variant_id, 'family': candidate['family'],
                                    'comparison': 'V13-Custom-D3', 'suite': suite, **values})
            values = paired_summary(custom, episode_data[condition_name], SUITES, index * 8 + 1007)
            paired_rows.append({'variant_id': variant_id, 'family': candidate['family'],
                                'comparison': 'V13-Custom-D3', 'suite': 'ALL_EQUAL_SUITE', **values})
    write_csv(analysis_root / 'paired_success.csv', paired_rows, list(paired_rows[0]))

    phase_counts = Counter()
    trajectory_rows: list[dict[str, Any]] = []
    candidate_records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    baseline_telemetry_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for index, (variant_id, candidate) in enumerate(sorted(candidates.items())):
        condition_name = candidate['condition_name']
        for suite in SUITES:
            baseline_run = RESULTS / 'closed_loop' / BASELINE / suite / 'run'
            variant_run = RESULTS / 'closed_loop' / condition_name / suite / 'run'
            for key in sorted(base[suite]):
                base_row = base[suite][key]
                variant_row = episode_data[condition_name][suite][key]
                cohort = outcome_label(bool(base_row['success']), bool(variant_row['success']))
                furthest = variant_row.get('furthest_semantic_phase') or 'unknown'
                if furthest not in PHASES:
                    furthest = 'unknown'
                if cohort == 'baseline_only_success':
                    phase_counts[(variant_id, 'new_failure', furthest)] += 1
                if cohort == 'both_failure':
                    phase_counts[(variant_id, 'both_failure', furthest)] += 1
                cache_key = (suite, key[0], key[1])
                if cache_key not in baseline_telemetry_cache:
                    baseline_telemetry_cache[cache_key] = load_trajectory(baseline_run, base_row)
                metrics = telemetry_metrics(baseline_telemetry_cache[cache_key], load_trajectory(variant_run, variant_row))
                record = {'variant_id': variant_id, 'condition_name': condition_name, 'suite': suite,
                          'task_id': key[0], 'episode_idx': key[1], 'cohort': cohort,
                          'variant_furthest_phase': furthest, **metrics}
                trajectory_rows.append(record)
                candidate_records[(variant_id, suite)].append(record)

    stage_rows = [{'variant_id': variant, 'cohort': cohort, 'furthest_phase': phase, 'count': count}
                  for (variant, cohort, phase), count in sorted(phase_counts.items())]
    write_csv(analysis_root / 'failure_stage_counts.csv', stage_rows,
              ['variant_id', 'cohort', 'furthest_phase', 'count'])
    write_csv(analysis_root / 'trajectory_deviation.csv', trajectory_rows, list(trajectory_rows[0]))
    divergence_rows = write_divergence_plot(
        trajectory_rows, analysis_root / 'figures/first_deviation_relative_to_baseline_contact.png')
    write_csv(analysis_root / 'first_deviation_counts.csv', divergence_rows,
              ['variant_id', 'category', 'count'])

    heatmap_rows = [row for row in summary_rows if row['family'] in ('Full', 'V13-reference', 'V16', 'V15', 'V14', 'V13')]
    write_heatmap(heatmap_rows, SUITES, analysis_root / 'figures/success_heatmap.png')
    write_stage_plot(stage_rows, analysis_root / 'figures/new_failure_furthest_phase.png')
    examples = plot_trajectory_examples(candidate_records, episode_data, analysis_root,
                                        render_videos=not args.skip_videos)
    write_csv(analysis_root / 'trajectory_example_index.csv', examples, list(examples[0]))
    audit = {
        'status_path': str(status_path), 'status_sha256': sha256(status_path),
        'protocol_path': str(RESULTS / 'protocol/closed_loop_config_formal.json'),
        'protocol_sha256': sha256(RESULTS / 'protocol/closed_loop_config_formal.json'),
        'candidate_manifest_sha256': sha256(RESULTS / 'protocol/candidate_manifest.json'),
        'checkpoint_sha256': read_json(RESULTS / 'protocol/candidate_manifest.json')['checkpoint_sha256'],
        'preflight_path': str(RESULTS / 'provenance/preflight.json'),
        'preflight_sha256': sha256(RESULTS / 'provenance/preflight.json'),
        'analysis_code_sha256': sha256(Path(__file__).resolve()),
        'execution_code_sha256': {
            relative: sha256(ROOT / relative) for relative in (
                'scripts/experiment10/run_closed_loop.py',
                'scripts/experiment10/serve_policy.py',
                'scripts/experiment10/backend.py',
                'scripts/experiment10/client_libero.py',
                'scripts/spec/spec_client_libero.py',
            )
        },
        'condition_suite_runs_expected': 124, 'condition_suite_runs_validated': len(run_validation),
        'episodes_expected': 6200, 'episodes_validated': len(episode_rows),
        'full_observation_input_artifacts_missing': sum(row['full_input_artifacts_missing']
                                                        for row in summary_rows),
        'validated_video_files': sum(int(row['validated_video_count'] or 0) for row in run_validation),
        'video_pairs_rendered': len({row['variant_id'] for row in examples}) if not args.skip_videos else 0,
        'result_interpretation': 'Exploratory five-init closed-loop results; no held-out task claim; latency is observed under concurrent rollout load.',
        'trajectory_metric_warning': 'Same-environment-step EEF differences compare separately executed trajectories and are not same-state causal action error.',
        'stage_metric_warning': 'Furthest reached phase is not causal error onset; new failure means Full succeeded and variant failed.',
    }
    if audit['condition_suite_runs_validated'] != 124 or audit['episodes_validated'] != 6200 or audit['full_observation_input_artifacts_missing']:
        raise RuntimeError(f'Completion audit found incomplete coverage: {audit}')
    dump_json(analysis_root / 'completion_audit.json', audit)
    print(json.dumps({'analysis_root': str(analysis_root), **audit}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
