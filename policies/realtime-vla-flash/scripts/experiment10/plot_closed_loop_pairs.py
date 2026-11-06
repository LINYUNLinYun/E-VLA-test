"""Actual EEF comparisons, with deterministic outcome-stratified diagnostic examples."""
import argparse
import json

from scripts.experiment10.common import RESULTS, SUITES, dump_json
from scripts.experiment8.plot_paired_xyz import read_trajectory, compare_trajectories, plot_comparison, _jsonable_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--variant', choices=('Full18_D1', 'Full18_D2', 'Full18_D4'), required=True)
    args = parser.parse_args()
    baseline = RESULTS / 'closed_loop/Full18_D10' / args.suite / 'run'
    variant = RESULTS / 'closed_loop' / args.variant / args.suite / 'run'
    for root in (baseline, variant):
        if not json.loads((root / 'validated.json').read_text())['complete']:
            raise RuntimeError('Unvalidated suite')
    maps = [{(r['task_id'], r['episode_idx']): r for r in json.loads((root / 'episode_log.json').read_text())}
            for root in (baseline, variant)]
    if maps[0].keys() != maps[1].keys() or len(maps[0]) != 50:
        raise RuntimeError('Paired episode coverage mismatch')
    comparisons, groups, table = {}, {}, []
    for key in sorted(maps[0]):
        b, v = maps[0][key], maps[1][key]
        bp, vp = baseline / b['telemetry_path'], variant / v['telemetry_path']
        comparison = compare_trajectories(read_trajectory(bp), read_trajectory(vp),
                     baseline_label='Full18-D10', variant_label=args.variant,
                     baseline_path=str(bp), variant_path=str(vp))
        comparisons[key] = comparison
        outcome = (bool(b['success']), bool(v['success']))
        groups.setdefault(outcome, []).append(key)
        table.append({'task': key[0], 'init': key[1], 'baseline_success': outcome[0],
                      'variant_success': outcome[1], **comparison['paired_deviation_m'],
                      'shared_steps': comparison['step_coverage']['shared_step_count']})
    # Include task0/init0 plus largest observed overlap p95 in each outcome group.
    chosen = {(0, 0)}
    for keys in groups.values():
        chosen.add(max(keys, key=lambda k: (comparisons[k]['paired_deviation_m']['p95'], k)))
    output = RESULTS / 'closed_loop_analysis/xyz' / args.variant / args.suite
    output.mkdir(parents=True, exist_ok=True)
    examples = []
    for key in sorted(chosen):
        prefix = output / ('task%02d_init%02d' % key)
        png, pdf = prefix.with_suffix('.png'), prefix.with_suffix('.pdf')
        comparison = comparisons[key]
        plot_comparison(comparison, baseline_label='Full18-D10', variant_label=args.variant,
                        png_path=png, pdf_path=pdf)
        summary = _jsonable_summary(comparison, png_path=png, pdf_path=pdf)
        summary['baseline_trust'] = {'validated_by_script': True,
                                   'note': 'Full18-D10 same-init validated Experiment10 condition'}
        dump_json(prefix.with_suffix('.summary.json'), summary)
        b, v = maps[0][key], maps[1][key]
        examples.append({'task': key[0], 'init': key[1], 'png': str(png),
                         'baseline_video': str(baseline / b['video_path']),
                         'variant_video': str(variant / v['video_path'])})
    dump_json(output / 'paired_xyz_index.json', {'pairs': table, 'examples': examples,
              'selection': 'task0/init0 plus maximum p95 per outcome group; diagnostic, not representative average',
              'alignment': 'same environment step on overlap; terminated tails excluded from deviations',
              'synchronized_composite_video_complete': False})


if __name__ == '__main__':
    main()
