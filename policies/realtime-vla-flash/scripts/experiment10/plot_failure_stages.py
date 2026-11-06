"""Validated highest-stage failure counts; never present these as first-error causes."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, PHASE_MAP, dump_json


def main():
    source = json.loads((RESULTS/'closed_loop_analysis/validated_results.json').read_text())
    rows = {(r['condition'], r['suite']): r for r in source['suite_rows']}
    conditions = ['Full18_D10', 'Full18_D1', 'Full18_D2', 'Full18_D4']
    stages = ['approach', 'contact_grasp', 'transport', 'alignment_release', 'unknown']
    colors = ['#74add1', '#fdae61', '#66c2a5', '#b58ac9', '#aaaaaa']
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    normalized = []
    ymax = max([10]+[r['episodes']-r['successes']+4 for r in source['suite_rows']
                    if r['status'] == 'validated_complete'])
    for suite, axis in zip(SUITES, axes.flat):
        for position, condition in enumerate(conditions):
            row = rows.get((condition, suite))
            if not row or row['status'] != 'validated_complete':
                axis.text(position, .8, 'pending', rotation=90, ha='center', color='#888888')
                continue
            counts = dict.fromkeys(stages, 0)
            for stage, count in row['failure_highest_stage_counts'].items():
                grouped = PHASE_MAP.get(stage, stage if stage in stages else 'unknown')
                counts[grouped] += count
            if sum(counts.values()) != row['episodes']-row['successes']:
                raise RuntimeError('Stage counts do not cover all validated failures')
            bottom = 0
            for stage, color in zip(stages, colors):
                count = counts[stage]
                axis.bar(position, count, bottom=bottom, color=color, width=.65,
                         label=stage if position == 0 else None)
                if count:
                    axis.text(position, bottom+count/2, str(count), ha='center', va='center', fontsize=9)
                bottom += count
            axis.text(position, bottom+.6, str(bottom)+'/50 failed', ha='center', fontsize=8)
            normalized.append({'suite': suite, 'condition': condition, 'counts': counts})
        axis.set_title(suite)
        axis.set_xticks(range(4), ['D10', 'D1', 'D2', 'D4'])
        axis.set_xlim(-.6, 3.6)
        axis.set_ylim(0, ymax)
        axis.set_ylabel('Failed episodes (50 evaluated per bar)')
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
    fig.suptitle('Highest reached stage in failed episodes\nNot the first deviation, failure onset, or causal attribution')
    output = RESULTS/'closed_loop_analysis'
    fig.savefig(output/'failure_highest_stage_counts.png', dpi=180)
    plt.close(fig)
    dump_json(output/'failure_highest_stage_counts.json', {'rows': normalized,
              'interpretation': 'highest stage only; first sustained XYZ/pose deviation analysis still required'})


if __name__ == '__main__':
    main()
