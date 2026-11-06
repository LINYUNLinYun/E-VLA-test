"""Static scientific figure of the completed offline proxy screen only."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/root/nfs/embodied/benchmark')
HERE = Path(__file__).resolve().parent
ANALYSIS = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v1/analysis/analysis.json'
SELECTION = ANALYSIS.with_name('selection.json')
AUDIT = ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v1/independent_audit.json'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if sha(ANALYSIS) != 'ccdfcae893d988daa1562850aafee068cbdf9bd3dfee4f2c8ccf68b9766de1ae':
        raise ValueError('Completed proxy analysis changed')
    if sha(SELECTION) != '8bdd13b5c091387ecba67c721df8452543cccc03a9305e08fde5646cceb42c11':
        raise ValueError('Frozen proxy choice changed')
    value, audit = json.loads(ANALYSIS.read_text()), json.loads(AUDIT.read_text())
    if not value['complete'] or audit['status'] != 'pass' or audit['selected']['masks'] != [[2, 13], [7, 14]]:
        raise ValueError('Need complete independently reconstructed proxy results')
    a, b = value['decision']['selected']['mask_indices']
    suites = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
    rows = []
    colors = ['#137c83', '#d67828']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1), gridspec_kw={'width_ratios': [1.25, 1, 1]})
    x = np.arange(4)
    for mi, label, color, offset in [(a, 'A: skip [2,13]', colors[0], -.18), (b, 'B: skip [7,14]', colors[1], .18)]:
        data = [value['suite_NMSE'][s][mi] for s in suites]
        axes[0].bar(x + offset, data, width=.34, color=color, label=label)
        for s, y in zip(suites, data):
            rows.append(dict(level='suite', suite=s, task='', metric='task_equal_NMSE', path=label, value=y))
    axes[0].set_xticks(x, ['Spatial', 'Object', 'Goal', 'Long'])
    axes[0].set_ylabel('Normalized command MSE (lower is better)')
    axes[0].set_title('Equal Fast budget: AE16 / D2')
    axes[0].legend(frameon=False, fontsize=9)
    for ax, suite, better, worse, title, color in [
        (axes[1], 'libero_object', a, b, 'Object: error(B) − error(A)', colors[0]),
        (axes[2], 'libero_spatial', b, a, 'Spatial: error(A) − error(B)', colors[1]),
    ]:
        diffs = [task[worse] - task[better] for task in value['task_NMSE'][suite]]
        ax.bar(np.arange(10), diffs, color=[color if d > 0 else '#999999' for d in diffs], width=.72)
        ax.axhline(0, color='#444444', lw=.8)
        ax.set_xticks(range(10))
        ax.set_xlabel('Task ID')
        ax.set_ylabel('Paired task-mean NMSE difference')
        ax.set_title(title)
        ax.text(.03, .97, f'{sum(d > 0 for d in diffs)}/10 tasks in frozen direction', transform=ax.transAxes, va='top', fontsize=9)
        for task, d in enumerate(diffs):
            rows.append(dict(level='task', suite=suite, task=task, metric='worse_minus_better_NMSE', path=title, value=d))
    fig.suptitle('Offline same-input proxy preferences — closed-loop success remains unverified', fontsize=12)
    fig.text(.5, .015, '110 archived Full-trajectory snapshots · 40 tasks · 3 repeated noise seeds · 153 masks screened\nFirst 5 decoded commands, all 7 dimensions; equal task weights. Initial-only sensitivity selected different masks.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .11, 1, .92))
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(HERE / ('AE16_joint_proxy.' + extension), dpi=180, bbox_inches='tight')
    plt.close(fig)
    with (HERE / 'proxy_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['level', 'suite', 'task', 'metric', 'path', 'value'])
        writer.writeheader()
        writer.writerows(rows)
    provenance = dict(inputs={str(p): sha(p) for p in [ANALYSIS, SELECTION, AUDIT]},
                      source=dict(path=str(Path(__file__).resolve()), sha256=sha(__file__)),
                      physical_episodes=0, quality_or_speed_claim=False, closed_loop_results_read=False,
                      metric='First5/all7 decoded NMSE, dense reference mean-square scale floor1e-4; seeds within snapshot, snapshots within task, tasks equal.',
                      limitations='Offline selected pair on archived Full-trajectory states, not independent rollout evidence; repeated seeds are not independent tasks; initial-only winner differs.')
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(status='pass', output=str(HERE), rows=len(rows), closed_loop_results_read=False)))


if __name__ == '__main__': main()
