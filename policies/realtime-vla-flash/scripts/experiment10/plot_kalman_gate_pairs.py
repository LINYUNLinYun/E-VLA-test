"""Paired measured-EEF plots and synchronized videos for a validated Kalman gate."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess

from scripts.experiment10.common import ROOT, RESULTS, SUITES, dump_json, sha256
from scripts.experiment8.plot_paired_xyz import (
    read_trajectory, compare_trajectories, plot_comparison, _jsonable_summary,
)
from scripts.experiment8.render_xyz_video import validate_video


def process_suite(condition, suite):
    name = condition['name']
    baseline_root = RESULTS / 'closed_loop' / 'Full18_D10' / suite / 'run'
    variant_root = RESULTS / 'closed_loop' / name / suite / 'run'
    for root in (baseline_root, variant_root):
        validation = json.loads((root / 'validated.json').read_text())
        if not validation.get('complete'):
            raise RuntimeError('Trajectory source is not validated: %s' % root)

    baseline_rows = json.loads((baseline_root / 'episode_log.json').read_text())
    variant_rows = json.loads((variant_root / 'episode_log.json').read_text())
    expected = {(task, init) for task in condition['task_ids'] for init in condition['init_indices']}
    baseline = {(r['task_id'], r['episode_idx']): r for r in baseline_rows if r['task_id'] in condition['task_ids']}
    variant = {(r['task_id'], r['episode_idx']): r for r in variant_rows}
    if set(baseline) != expected or set(variant) != expected:
        raise RuntimeError('Expected paired task/init coverage does not match condition')

    output = RESULTS / 'closed_loop_analysis' / 'xyz' / name / suite
    output.mkdir(parents=True, exist_ok=True)
    comparisons, table = {}, []
    for key in sorted(expected):
        base, draft = baseline[key], variant[key]
        base_telemetry, draft_telemetry = baseline_root / base['telemetry_path'], variant_root / draft['telemetry_path']
        comparison = compare_trajectories(
            read_trajectory(base_telemetry), read_trajectory(draft_telemetry),
            baseline_label='Full18-D10', variant_label=name,
            baseline_path=str(base_telemetry), variant_path=str(draft_telemetry))
        comparison['baseline_trust'] = {
            'validated_by_script': True,
            'note': 'Same-suite, same-task, same-official-init Full18-D10 paired rollout',
        }
        comparisons[key] = comparison
        table.append({
            'task_id': key[0], 'episode_idx': key[1],
            'baseline_success': int(bool(base['success'])),
            'variant_success': int(bool(draft['success'])),
            'paired_mean_xyz_deviation_m': comparison['paired_deviation_m']['mean'],
            'paired_median_xyz_deviation_m': comparison['paired_deviation_m']['median'],
            'paired_p95_xyz_deviation_m': comparison['paired_deviation_m']['p95'],
            'paired_max_xyz_deviation_m': comparison['paired_deviation_m']['maximum'],
            'paired_max_step': comparison['paired_deviation_m']['maximum_step'],
            'shared_step_count': comparison['step_coverage']['shared_step_count'],
        })

    # One worst-overlap pair per suite/condition is a diagnostic example, not an average.
    chosen = max(expected, key=lambda key: (comparisons[key]['paired_deviation_m']['p95'], key))
    key = chosen
    prefix = output / ('task%02d_init%02d' % key)
    png, pdf = prefix.with_suffix('.png'), prefix.with_suffix('.pdf')
    comparison = comparisons[key]
    plot_comparison(comparison, baseline_label='Full18-D10', variant_label=name,
                    png_path=png, pdf_path=pdf)
    plot_summary = _jsonable_summary(comparison, png_path=png, pdf_path=pdf)
    plot_summary['baseline_trust'] = comparison['baseline_trust']
    dump_json(prefix.with_suffix('.summary.json'), plot_summary)
    base, draft = baseline[key], variant[key]
    example = {
        'task_id': key[0], 'episode_idx': key[1], 'selection': 'largest p95 deviation on shared environment steps',
        'plot_png': str(png), 'plot_pdf': str(pdf),
        'baseline_video': str(baseline_root / base['video_path']),
        'variant_video': str(variant_root / draft['video_path']),
        'baseline_telemetry': str(baseline_root / base['telemetry_path']),
        'variant_telemetry': str(variant_root / draft['telemetry_path']),
    }
    with (output / 'paired_xyz.csv.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    (output / 'paired_xyz.csv.tmp').replace(output / 'paired_xyz.csv')
    dump_json(output / 'paired_xyz_index.json', {
        'condition': condition, 'suite': suite, 'pairs': table, 'examples': [example],
        'selection': 'one worst-p95 shared-step trajectory pair; diagnostic, not representative average',
        'alignment': 'same simulator step; only overlap is compared; measured EEF XYZ in world meters',
        'trajectory_colors': {'baseline': 'blue', 'variant': 'orange'},
    })
    return output, example


def render_example(condition, suite, output, example):
    name = condition['name']
    key = example['task_id'], example['episode_idx']
    video = output / ('task%02d_init%02d_synchronized.mp4' % key)
    provenance = video.with_suffix('.provenance.json')
    sources = [Path(example[k]) for k in (
        'baseline_video', 'baseline_telemetry', 'variant_video', 'variant_telemetry')]
    evidence = {str(path): sha256(path) for path in sources}
    renderer = ROOT / 'scripts/experiment8/render_xyz_video.py'
    if video.exists():
        if not provenance.exists():
            raise RuntimeError('Existing synchronized video has no provenance: %s' % video)
        saved = json.loads(provenance.read_text())
        if saved.get('source_sha256') != evidence or saved.get('renderer_sha256') != sha256(renderer):
            raise RuntimeError('Existing synchronized video provenance differs: %s' % video)
        report = json.loads(video.with_suffix('.json').read_text())
        validate_video(video, len(report['frame_index']))
        return {'task_id': key[0], 'episode_idx': key[1], 'path': str(video), 'reused': True}

    command = [
        str(ROOT / '.venv/bin/python'), str(renderer),
        '--baseline-video', str(sources[0]), '--baseline-telemetry', str(sources[1]),
        '--variant-video', str(sources[2]), '--variant-telemetry', str(sources[3]),
        '--baseline-label', 'Full18-D10', '--variant-label', name,
        '--alignment', 'state-synchronous', '--output', str(video),
    ]
    with video.with_suffix('.log').open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = json.loads(video.with_suffix('.json').read_text())
    validate_video(video, len(report['frame_index']))
    dump_json(provenance, {
        'source_sha256': evidence, 'video_sha256': sha256(video),
        'renderer_sha256': sha256(renderer),
        'alignment': 'state-synchronous N+1, actual measured EEF XYZ',
    })
    return {'task_id': key[0], 'episode_idx': key[1], 'path': str(video), 'reused': False}


def run(config_path, condition_name):
    config = json.loads(Path(config_path).read_text())
    condition = next((c for c in config['conditions'] if c['name'] == condition_name), None)
    if condition is None:
        raise ValueError('Unknown Kalman-gate condition: %s' % condition_name)
    suites = [process_suite(condition, suite) for suite in SUITES]
    rendered = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(render_example, condition, suite, output, example)
                   for suite, (output, example) in zip(SUITES, suites)]
        for future in futures:
            rendered.append(future.result())
    result = {
        'complete': True, 'condition': condition_name,
        'suites': {suite: {'index': str(output / 'paired_xyz_index.json'),
                           'csv': str(output / 'paired_xyz.csv')}
                   for suite, (output, _) in zip(SUITES, suites)},
        'synchronized_videos': rendered,
        'video_protocol': 'H.264/yuv420p, full frame-count and ffmpeg decode validation, state-synchronous N+1',
    }
    dump_json(RESULTS / 'closed_loop_analysis' / 'xyz' / condition_name / 'analysis_index.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--conditions', type=Path,
                        default=Path('scripts/experiment10/kalman_gate_closed_loop_config.json'))
    args = parser.parse_args()
    print(json.dumps(run(args.conditions, args.condition), indent=2), flush=True)


if __name__ == '__main__':
    main()
