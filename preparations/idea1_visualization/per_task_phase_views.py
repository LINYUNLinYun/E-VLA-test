"""Render all ten independent task projections, two/three stages side by side.

Only reads previously validated coordinates and labels. Does not fit any model.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .common import atomic_json
from .goal_h9_merged_phases import SCHEMES, record, scatter

STAGES = [SCHEMES[2], SCHEMES[1]]


def panels(axes, source, catalog, task, *, legend):
    prior = None
    detail = []
    for ax, scheme in zip(axes, STAGES):
        row = record(catalog, task, scheme['id'], 'raw')
        with np.load(source/'projections'/f"{row['slug']}.npz") as a:
            z, known, train, phase = (a[k] for k in ('coordinates', 'known', 'train', 'phases'))
            if prior is not None:
                assert np.array_equal(z, prior[0])
                assert np.array_equal(known, prior[1]) and np.array_equal(train, prior[2])
            prior = z.copy(), known.copy(), train.copy()
            scatter(ax, a, scheme)
            counts = np.bincount(phase[known], minlength=len(scheme['names']))
            observed = int(np.count_nonzero(counts))
            ax.set_title(f"Task{task} | {scheme['name']} | {int(known.sum())} points", fontsize=10)
            ax.set_xlabel(f"PC1 ({row['variance'][0]:.1%})", fontsize=8)
            ax.set_ylabel(f"PC2 ({row['variance'][1]:.1%})", fontsize=8)
            if observed < len(scheme['names']):
                ax.text(.025, .025, f"Observed stages: {observed}/{len(scheme['names'])}",
                        transform=ax.transAxes, fontsize=8, color='#475569')
            if legend:
                handles = [Line2D([], [], marker='o', linestyle='', color=color,
                                  label=f'{name} ({counts[i]})')
                           for i, (name, color) in enumerate(zip(scheme['names'], scheme['colors']))]
                ax.legend(handles=handles, fontsize=7.5, loc='best')
            detail.append(dict(scheme=scheme['id'], n_train=int(np.sum(train & known)),
                               n_test=int(np.sum(~train & known)), unknown=int(np.sum(~known)),
                               observed_stages=observed, counts=counts.tolist()))
    return detail


def render(source, output):
    catalog = json.loads((source/'catalog.json').read_text())
    details = []
    for task in range(10):
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
        detail = panels(axes, source, catalog, task, legend=True)
        details.append(dict(task=task, stages=detail))
        fig.suptitle(f'Goal task{task} | VLM H9 language features | task-specific training PCA', fontsize=14)
        fig.text(.5, .035, 'Colors represent merged stage labels. Circles: train init0/1. Triangles: held-out init2.\nBoth panels use identical coordinates. Unknown-stage points excluded. No new model fitting.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .115, 1, .94))
        fig.savefig(output/'figures'/f'task{task:02d}_two_three_stages.png', dpi=160)
        plt.close(fig)
    for first in (0, 5):
        fig, axes = plt.subplots(5, 2, figsize=(12.5, 18))
        for i, task in enumerate(range(first, first+5)):
            panels(axes[i], source, catalog, task, legend=False)
        fig.suptitle(f'Goal tasks {first}-{first+4} | H9 | Two stages (left), three stages (right)', fontsize=14)
        handles = [Line2D([], [], marker='o', linestyle='', color='#2563eb', label='Both: prepare / approach'),
                   Line2D([], [], marker='o', linestyle='', color='#dc2626', label='Left red: all other operation'),
                   Line2D([], [], marker='o', linestyle='', color='#d97706', label='Right orange: contact / manipulate'),
                   Line2D([], [], marker='o', linestyle='', color='#16a34a', label='Right green: transport / place / finish'),
                   Line2D([], [], marker='o', linestyle='', color='#475569', label='Circles: train init0/1'),
                   Line2D([], [], marker='^', linestyle='', color='#475569', label='Triangles: held-out init2')]
        fig.legend(handles=handles, ncol=2, loc='lower center', bbox_to_anchor=(.5, .012), fontsize=9)
        fig.text(.5, .075, 'Each task has its own training-only PCA. Same point coordinates within each row; no cross-task distance comparison.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .095, 1, .965))
        fig.savefig(output/'figures'/f'tasks{first:02d}_{first+4:02d}_two_three_stages.png', dpi=150)
        plt.close(fig)
    write_report(source, output, catalog, details)
    atomic_json(output/'render_validation.json', dict(passed=True, tasks=10, individual_figures=10,
        overview_figures=2, points=sum(d['stages'][0]['n_train']+d['stages'][0]['n_test'] for d in details),
        stage_coordinate_pairs_identical=10, no_refit=True, gpu_used=False, source=str(source),
        details=details, source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def write_report(source, output, catalog, details):
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Goal十个任务各自的H9阶段分布</title><style>
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}@page{size:A4 landscape;margin:12mm;@bottom-center{content:counter(page);font-size:8pt}}body{font:9.5pt/1.55 "DejaVu Sans",CJK,sans-serif;color:#203047}h1{font-size:18pt;margin:0 0 3mm}p{margin:1.5mm 0}.task{break-before:page}.task:first-of-type{break-before:auto}img{width:100%;display:block;max-height:128mm;object-fit:contain}.note{color:#52647a;font-size:9pt}</style>'''
    for d in details:
        task = d['task']; r = record(catalog, task, 'two', 'raw'); counts = d['stages'][0]
        body += f'<section class="task"><h1>Goal task{task}：H9 的两阶段／三阶段</h1>'
        body += f'<p>指令：{html.escape(r["instruction"])}</p>'
        body += f'<p>训练 {counts["n_train"]} 点（init0/1），留出 {counts["n_test"]} 点（init2）；排除 {counts["unknown"]} 个阶段未知点。左两阶段，右三阶段；本任务左右点位完全相同。</p>'
        body += f'<img src="{(output/"figures"/f"task{task:02d}_two_three_stages.png").as_uri()}">'
        body += '<p>左：蓝＝准备／接近，红＝其余操作。右：蓝＝准备／接近，橙＝接触／操作，绿＝搬运／就位／完成。圆点为训练观测，三角为留出的新轨迹。</p>'
        body += '<p class="note">各 task 独立使用其训练轨迹拟合的 PCA；直接复用已有坐标，没有重新拟合。跨任务的坐标轴不能直接比较。颜色为阶段标签，不是 k-means 簇。阶段来自 trace 规则推断；每 task 仅一条留出轨迹，相邻点相关。</p>'
        if counts['observed_stages'] == 1:
            body += '<p>本任务合并后仅剩“准备／接近”一类，无法据此评价阶段区分效果。</p>'
        elif d['stages'][1]['observed_stages'] < 3:
            body += '<p>三阶段规则下，本任务没有出现全部三类；缺失的阶段不会补点。</p>'
        body += '</section>'
    (output/'report_zh.html').write_text(body+'</html>')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True); (output/'figures').mkdir(exist_ok=True)
    render(args.source.resolve(), output)
    print(json.dumps(dict(passed=True, output=str(output))), flush=True)


if __name__ == '__main__':
    main()
