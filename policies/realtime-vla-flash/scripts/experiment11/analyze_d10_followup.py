"""Validate and analyze the frozen D10-vs-D3 denoising interaction diagnostic."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiment11.protocol import RESULTS, SUITES
from scripts.experiment10.common import dump_json, sha256


CONFIG = RESULTS / 'protocol/closed_loop_config_d10_followup.json'
D3_CONFIG = RESULTS / 'protocol/closed_loop_config_formal.json'
D10_STATUS = RESULTS / 'experiment11_d10_closed_loop_status.json'
D3_STATUS = RESULTS / 'experiment11_closed_loop_status.json'
D10_ANALYSIS = RESULTS / 'analysis/d10_followup_final'
EXPECTED_KEYS = {(task, init) for task in range(10) for init in range(5)}
BOOTSTRAP_REPLICATES = 2000


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_episode_map(condition: dict, suite: str, *, denoise_steps: int) -> dict[tuple[int, int], dict]:
    condition_name = condition['name']
    run_dir = RESULTS / 'closed_loop' / condition_name / suite / 'run'
    validation_path = run_dir / 'validated.json'
    episode_path = run_dir / 'episode_log.json'
    if not validation_path.is_file() or not episode_path.is_file():
        raise RuntimeError(f'Missing validated D{denoise_steps} artifacts for {condition_name}/{suite}')
    validation = json.loads(validation_path.read_text())
    expected_config = validation.get('condition') or {}
    if (validation.get('complete') is not True or validation.get('episodes') != 50 or
            validation.get('full_video_decodes') != 50 or validation.get('suite') != suite or
            expected_config.get('name') != condition_name or
            expected_config.get('steps') != denoise_steps or
            expected_config.get('horizon') != 5 or expected_config.get('replan_steps') != 5):
        raise RuntimeError(f'Protocol/video validation failed for {condition_name}/{suite}')
    episodes = json.loads(episode_path.read_text())
    rows = {}
    for row in episodes:
        key = int(row['task_id']), int(row['episode_idx'])
        if key in rows or type(row.get('success')) is not bool:
            raise RuntimeError(f'Duplicate or malformed episode record for {condition_name}/{suite}: {key}')
        rows[key] = row
    if set(rows) != EXPECTED_KEYS:
        raise RuntimeError(f'{condition_name}/{suite}: expected exact task0–9/init0–4 coverage')
    return rows


def stratified_task_ci(groups: dict[str, list[float]], seed: int) -> tuple[float | None, float | None]:
    nonempty = {suite: np.asarray(values, dtype=np.float64) for suite, values in groups.items() if values}
    if sum(values.size for values in nonempty.values()) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(BOOTSTRAP_REPLICATES):
        suite_means = []
        for values in nonempty.values():
            indices = rng.integers(0, values.size, size=values.size)
            suite_means.append(float(np.mean(values[indices])))
        samples[replicate] = np.mean(suite_means)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _metric_rows(d10_id: str, d10_name: str, d10_maps: dict,
                 d3_name: str, d3_maps: dict, full_d10_maps: dict,
                 full_d3_maps: dict) -> list[dict]:
    per_task = []
    for suite in SUITES:
        for task in range(10):
            d10_values = [bool(d10_maps[suite][(task, init)]['success']) for init in range(5)]
            d3_values = [bool(d3_maps[suite][(task, init)]['success']) for init in range(5)]
            full_d10 = [bool(full_d10_maps[suite][(task, init)]['success']) for init in range(5)]
            full_d3 = [bool(full_d3_maps[suite][(task, init)]['success']) for init in range(5)]
            d10_rate, d3_rate = float(np.mean(d10_values)), float(np.mean(d3_values))
            full_delta = float(np.mean(full_d10) - np.mean(full_d3))
            per_task.append({
                'suite': suite, 'task_id': task,
                'd10_successes': sum(d10_values), 'd3_successes': sum(d3_values),
                'both_success': sum(a and b for a, b in zip(d10_values, d3_values)),
                'd10_only_success': sum(a and not b for a, b in zip(d10_values, d3_values)),
                'd3_only_success': sum(b and not a for a, b in zip(d10_values, d3_values)),
                'both_failure': sum(not a and not b for a, b in zip(d10_values, d3_values)),
                'd10_minus_d3': d10_rate - d3_rate,
                'interaction_vs_full': (d10_rate - d3_rate) - full_delta,
            })

    summaries = []
    for suite_group in (*SUITES, 'ALL_EQUAL_SUITE'):
        rows = per_task if suite_group == 'ALL_EQUAL_SUITE' else [r for r in per_task if r['suite'] == suite_group]
        suite_groups = ({suite: [r['d10_minus_d3'] for r in rows if r['suite'] == suite]
                         for suite in SUITES} if suite_group == 'ALL_EQUAL_SUITE' else
                        {suite_group: [r['d10_minus_d3'] for r in rows]})
        interaction_groups = ({suite: [r['interaction_vs_full'] for r in rows if r['suite'] == suite]
                               for suite in SUITES} if suite_group == 'ALL_EQUAL_SUITE' else
                              {suite_group: [r['interaction_vs_full'] for r in rows]})
        seed_prefix = f'{d10_id}|{suite_group}'.encode('utf-8')
        delta_low, delta_high = stratified_task_ci(
            suite_groups, int.from_bytes(hashlib.sha256(seed_prefix + b'|delta').digest()[:4], 'little'))
        interaction_low, interaction_high = stratified_task_ci(
            interaction_groups, int.from_bytes(hashlib.sha256(seed_prefix + b'|interaction').digest()[:4], 'little'))
        total = len(rows) * 5
        d10_successes = sum(r['d10_successes'] for r in rows)
        d3_successes = sum(r['d3_successes'] for r in rows)
        summaries.append({
            'variant_id': d10_id, 'd10_condition': d10_name, 'd3_condition': d3_name,
            'suite': suite_group, 'task_clusters': len(rows), 'episodes': total,
            'd10_successes': d10_successes, 'd10_success_rate': d10_successes / total,
            'd3_successes': d3_successes, 'd3_success_rate': d3_successes / total,
            'd10_minus_d3_success_rate': float(np.mean(suite_groups[suite_group])) if suite_group != 'ALL_EQUAL_SUITE'
            else float(np.mean([np.mean(values) for values in suite_groups.values()])),
            'd10_minus_d3_ci95_low': delta_low, 'd10_minus_d3_ci95_high': delta_high,
            'interaction_vs_full': float(np.mean(interaction_groups[suite_group])) if suite_group != 'ALL_EQUAL_SUITE'
            else float(np.mean([np.mean(values) for values in interaction_groups.values()])),
            'interaction_ci95_low': interaction_low, 'interaction_ci95_high': interaction_high,
            'both_success': sum(r['both_success'] for r in rows),
            'd10_only_success': sum(r['d10_only_success'] for r in rows),
            'd3_only_success': sum(r['d3_only_success'] for r in rows),
            'both_failure': sum(r['both_failure'] for r in rows),
            'bootstrap_replicates': BOOTSTRAP_REPLICATES,
            'bootstrap_unit': 'task within suite; five init episodes kept together',
        })
    return summaries


def render_figures(summary_rows: list[dict], output: Path) -> list[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    variants = list(dict.fromkeys(row['variant_id'] for row in summary_rows if row['suite'] == SUITES[0]))
    colors = {'D3': '#2878B5', 'D10': '#E07A24'}
    figure = output / 'd10_vs_d3_success_by_suite.png'
    fig, axes = plt.subplots(2, 2, figsize=(17, 10), sharey=True)
    x = np.arange(len(variants))
    width = 0.36
    for ax, suite in zip(axes.flat, SUITES):
        rows = {row['variant_id']: row for row in summary_rows if row['suite'] == suite}
        d3 = [rows[v]['d3_success_rate'] for v in variants]
        d10 = [rows[v]['d10_success_rate'] for v in variants]
        ax.bar(x - width / 2, d3, width, color=colors['D3'], label='D3')
        ax.bar(x + width / 2, d10, width, color=colors['D10'], label='D10')
        ax.set_title(suite)
        ax.set_ylim(0, 1)
        ax.set_xticks(x, variants, rotation=35, ha='right')
        ax.grid(axis='y', alpha=0.22)
        ax.set_ylabel('Success rate (50 episodes)')
    axes.flat[0].legend()
    fig.suptitle('Matched D3 vs D10 closed-loop success; D10 is post-hoc diagnostic')
    fig.tight_layout()
    fig.savefig(figure, dpi=180)
    plt.close(fig)

    interaction = [row for row in summary_rows if row['suite'] == 'ALL_EQUAL_SUITE' and row['variant_id'] != 'Full18']
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(interaction))
    means = np.asarray([row['interaction_vs_full'] for row in interaction]) * 100
    low = np.asarray([row['interaction_ci95_low'] for row in interaction]) * 100
    high = np.asarray([row['interaction_ci95_high'] for row in interaction]) * 100
    ax.errorbar(means, y, xerr=np.vstack([means - low, high - means]), fmt='o', capsize=3, color='#4D7C0F')
    ax.axvline(0, color='black', linewidth=1, alpha=0.65)
    ax.set_yticks(y, [row['variant_id'] for row in interaction])
    ax.set_xlabel('Interaction: (variant D10−D3) − (Full D10−D3), percentage points')
    ax.set_title('Task-clustered, equal-suite Denoising-step interaction (95% bootstrap CI)')
    ax.grid(axis='x', alpha=0.22)
    fig.tight_layout()
    second = output / 'd10_interaction_equal_suite.png'
    fig.savefig(second, dpi=180)
    plt.close(fig)
    return [str(figure), str(second)]


def analyze() -> dict:
    if D10_ANALYSIS.exists() and (D10_ANALYSIS / 'completion_audit.json').exists():
        raise FileExistsError(f'D10 analysis already exists; refusing to overwrite: {D10_ANALYSIS}')
    d10_status = json.loads(D10_STATUS.read_text())
    d3_status = json.loads(D3_STATUS.read_text())
    if (d10_status.get('complete') is not True or d10_status.get('active') or
            d10_status.get('pending') or d10_status.get('failures')):
        raise RuntimeError('D10 scheduler status is not complete and clean')
    if (d3_status.get('complete') is not True or d3_status.get('active') or
            d3_status.get('pending') or d3_status.get('failures')):
        raise RuntimeError('D3 source matrix is not complete and clean')

    config = json.loads(CONFIG.read_text())
    d3_config = json.loads(D3_CONFIG.read_text())
    if len(config['conditions']) != 7 or config.get('episodes_per_condition') != 200:
        raise RuntimeError('D10 config must contain exactly seven frozen 200-episode conditions')
    if any(c.get('steps') != 10 or c.get('horizon') != 5 or c.get('replan_steps') != 5
           for c in config['conditions']):
        raise RuntimeError('D10 config changed the frozen 10-step, predict-50/execute-5 protocol')
    d3_names = {condition['name'] for condition in d3_config['conditions']}
    d3_conditions_by_variant = {'Full18': 'Full18_D3_R5_FORMAL',
                                'V13_Custom': 'V13_Custom_D3_R5_FORMAL'}
    candidate_manifest = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
    d3_conditions_by_variant.update({row['variant_id']: row['condition']['name'] + '_FORMAL'
                                     for row in candidate_manifest['candidates']})
    if not set(d3_conditions_by_variant.values()).issubset(d3_names):
        raise RuntimeError('Cannot map D10 diagnostics back to their D3 source conditions')

    d10_maps_by_variant = {}
    d3_maps_by_variant = {}
    summaries = []
    run_rows = []
    source_artifact_hashes = {}
    failure_stage_counts = Counter()
    full_d10_maps = {suite: load_episode_map(
        {'name': 'Full18_D10_R5_DIAG'}, suite, denoise_steps=10) for suite in SUITES}
    full_d3_maps = {suite: load_episode_map(
        {'name': 'Full18_D3_R5_FORMAL'}, suite, denoise_steps=3) for suite in SUITES}
    for condition in config['conditions']:
        d10_name = condition['name']
        variant = ('Full18' if condition.get('d10_role') == 'reference_full' else
                   'V13_Custom' if condition.get('d10_role') == 'reference_custom_v13' else
                   condition['variant_id'])
        d3_name = d3_conditions_by_variant[variant]
        d10_maps = {}
        d3_maps = {}
        for suite in SUITES:
            d10_maps[suite] = load_episode_map(condition, suite, denoise_steps=10)
            d3_maps[suite] = load_episode_map({'name': d3_name}, suite, denoise_steps=3)
            for label, condition_name in (('d10', d10_name), ('d3', d3_name)):
                run_dir = RESULTS / 'closed_loop' / condition_name / suite / 'run'
                for artifact_name in ('validated.json', 'episode_log.json'):
                    artifact = run_dir / artifact_name
                    source_artifact_hashes[f'{label}/{condition_name}/{suite}/{artifact_name}'] = sha256(artifact)
            episodes = list(d10_maps[suite].values())
            successes = sum(bool(row['success']) for row in episodes)
            for row in episodes:
                if not bool(row['success']):
                    failure_stage_counts[(d10_name, variant, suite,
                                          str(row.get('furthest_semantic_phase') or 'unknown'))] += 1
            run_rows.append({'condition': d10_name, 'variant_id': variant, 'suite': suite,
                             'successes': successes, 'episodes': len(episodes),
                             'success_rate': successes / len(episodes),
                             'failed_episodes': len(episodes) - successes})
        d10_maps_by_variant[variant] = d10_maps
        d3_maps_by_variant[variant] = d3_maps
        summaries.extend(_metric_rows(variant, d10_name, d10_maps, d3_name, d3_maps,
                                      full_d10_maps, full_d3_maps))

    if len(run_rows) != 28 or len(summaries) != 7 * (len(SUITES) + 1):
        raise RuntimeError('D10 analysis did not cover all 28 condition-suite jobs and summaries')

    D10_ANALYSIS.mkdir(parents=True, exist_ok=True)
    write_csv(D10_ANALYSIS / 'd10_suite_success.csv', run_rows,
              ['condition', 'variant_id', 'suite', 'successes', 'episodes', 'success_rate',
               'failed_episodes'])
    write_csv(D10_ANALYSIS / 'd10_failure_stage_counts.csv', [
        {'condition': condition, 'variant_id': variant, 'suite': suite,
         'furthest_semantic_phase': phase, 'failed_episodes': count}
        for (condition, variant, suite, phase), count in sorted(failure_stage_counts.items())
    ], ['condition', 'variant_id', 'suite', 'furthest_semantic_phase', 'failed_episodes'])
    write_csv(D10_ANALYSIS / 'd10_vs_d3_paired_summary.csv', summaries,
              ['variant_id', 'd10_condition', 'd3_condition', 'suite', 'task_clusters', 'episodes',
               'd10_successes', 'd10_success_rate', 'd3_successes', 'd3_success_rate',
               'd10_minus_d3_success_rate', 'd10_minus_d3_ci95_low', 'd10_minus_d3_ci95_high',
               'interaction_vs_full', 'interaction_ci95_low', 'interaction_ci95_high',
               'both_success', 'd10_only_success', 'd3_only_success', 'both_failure',
               'bootstrap_replicates', 'bootstrap_unit'])
    figures = render_figures(summaries, D10_ANALYSIS)

    report = [
        '# Experiment 11 — D10 denoising interaction diagnostic', '',
        'This is a post-hoc diagnostic on the seven frozen conditions. Each condition ran all four LIBERO suites, task0–9, init0–4 (200 episodes); the model predicts 50 actions and the environment executes the first 5. D10 changes the number of denoising evaluations over the full integration interval; it is not “the first three steps of D10.”',
        '',
        'The selected masks were chosen using D3 outcomes, so D10 is not an independent test set. D3/D10 outcomes share task/init starts, but their closed-loop states can diverge after actions; paired success differences are matched-start diagnostics, not same-state trajectory comparisons. Confidence intervals resample tasks within suite and keep the five init episodes of a task together.',
        '',
        '## Equal-suite D10 vs D3 outcomes', '',
        '| Variant | D3 successes / 200 | D10 successes / 200 | ΔD10−D3 (95% task CI) | Interaction vs Full (95% task CI) |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    for row in summaries:
        if row['suite'] != 'ALL_EQUAL_SUITE':
            continue
        delta_ci = f"[{row['d10_minus_d3_ci95_low']:.3f}, {row['d10_minus_d3_ci95_high']:.3f}]"
        interaction_ci = f"[{row['interaction_ci95_low']:.3f}, {row['interaction_ci95_high']:.3f}]"
        report.append(f"| {row['variant_id']} | {row['d3_successes']}/200 ({row['d3_success_rate']:.1%}) | "
                      f"{row['d10_successes']}/200 ({row['d10_success_rate']:.1%}) | "
                      f"{row['d10_minus_d3_success_rate']:+.1%} {delta_ci} | "
                      f"{row['interaction_vs_full']:+.1%} {interaction_ci} |")
    report.extend([
        '',
        'The interaction is `(variant D10 − variant D3) − (Full D10 − Full D3)`; positive values mean the variant’s success-rate change under D10 is more favorable than Full’s change. This is a descriptive matched-start interaction, not a universal denoising law.',
        '', '## Per-suite and visualization artifacts', '',
        '- `d10_suite_success.csv`: all 28 condition×suite success rates.',
        '- `d10_failure_stage_counts.csv`: furthest semantic phase among D10 failures; not causal error-onset labels.',
        '- `d10_vs_d3_paired_summary.csv`: suite and equal-suite task-clustered paired deltas/interactions.',
        '- `d10_vs_d3_success_by_suite.png` and `d10_interaction_equal_suite.png`.',
        '', 'Ordinary task failures are counted in success rates. Invalid/missing runs are rejected by the audit and are never recoded as task failures.',
    ])
    report_path = D10_ANALYSIS / 'report.md'
    report_path.write_text('\n'.join(report) + '\n', encoding='utf-8')

    video_decodes = sum(50 for _ in run_rows)
    audit = {
        'complete': True, 'experiment_id': 'experiment11_d10_interaction_diagnostic',
        'conditions_expected': 7, 'conditions_validated': len(config['conditions']),
        'condition_suite_runs_expected': 28, 'condition_suite_runs_validated': len(run_rows),
        'episodes_expected': 1400, 'episodes_validated': sum(row['episodes'] for row in run_rows),
        'video_decodes_validated': video_decodes,
        'config_sha256': sha256(CONFIG), 'd3_config_sha256': sha256(D3_CONFIG),
        'candidate_manifest_sha256': sha256(RESULTS / 'protocol/candidate_manifest.json'),
        'd10_status_sha256': sha256(D10_STATUS), 'd3_status_sha256': sha256(D3_STATUS),
        'analysis_code_sha256': sha256(Path(__file__).resolve()),
        'source_artifact_hashes': source_artifact_hashes,
        'result_interpretation': 'Post-hoc seven-condition interaction diagnostic; not held-out validation.',
        'outputs': [str(path) for path in sorted(D10_ANALYSIS.rglob('*')) if path.is_file()],
    }
    if audit['conditions_validated'] != 7 or audit['condition_suite_runs_validated'] != 28 or audit['episodes_validated'] != 1400:
        raise RuntimeError(f'D10 completion audit incomplete: {audit}')
    dump_json(D10_ANALYSIS / 'completion_audit.json', audit)
    print(json.dumps({'analysis': str(D10_ANALYSIS), 'audit': audit}, indent=2))
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    analyze()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
