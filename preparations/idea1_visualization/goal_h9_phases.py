"""Within-task Goal H9 phase scatters, train-only PCA and held-out clustering.

Uses only saved pooled activations. No policy inference or GPU work is performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .analyze import PHASES, fit_projection, project, write_csv
from .common import atomic_json

FEATURE = 'vlm/language/hidden'
PHASE_NAMES = ['Initial', 'Approach', 'Contact', 'Grasp / manipulate', 'Transport',
               'Goal alignment', 'Release / final', 'Success', 'Unknown']
PHASE_COLORS = ['#64748b', '#2563eb', '#d97706', '#dc2626', '#16a34a',
                '#9333ea', '#db2777', '#0891b2', '#cbd5e1']
CLUSTER_COLORS = ['#636efa', '#ef553b', '#00a978', '#ab63fa', '#e09522',
                  '#19b5d3', '#ff6692', '#8c6947']


def predictions(x, y, train):
    model = make_pipeline(StandardScaler(), RidgeClassifier(alpha=10., solver='lsqr', tol=1e-3))
    model.fit(x[train], y[train])
    return model.predict(x)


def load_task(root, task):
    samples, values = [], []
    for init in range(3):
        episode = root/'episodes/libero_goal'/f'task{task:02d}'/f'init{init:02d}'
        identity = json.loads((episode/'identity.json').read_text())
        for s in json.loads((episode/'selection.json').read_text())['samples']:
            directory = root/'features/libero_goal'/f'task{task:02d}'/f'init{init:02d}'/f"request_{s['request']:04d}"
            assert json.loads((directory/'complete.json').read_text())['passed']
            with np.load(directory/'pooled.npz', allow_pickle=False) as archive:
                values.append(archive[FEATURE][9].copy())
            samples.append(dict(task=task, init=init, request=s['request'], step=s['step'],
                                phase=s['phase'], sources=s['sources'],
                                instruction=identity['instruction'], feature_directory=str(directory)))
    return samples, np.stack(values)


def compute(root, output):
    metrics, support, catalog = [], [], []
    old_rows = list(csv.DictReader((root/'analysis/conditional_probes.csv').open()))
    for task in range(10):
        samples, original = load_task(root, task)
        phase = np.array([s['phase'] for s in samples])
        y = np.array([PHASES.index(p) for p in phase])
        train = np.array([s['init'] < 2 for s in samples])
        known = phase != 'unknown'
        fit, test = train & known, (~train) & known
        k_phase = len(np.unique(y[fit]))
        atomic_json(output/f'task{task:02d}_samples.json', samples)
        for p, name in enumerate(PHASES):
            support.append(dict(task=task, phase=name, n_train=int(np.sum(fit & (y == p))),
                                n_test=int(np.sum(test & (y == p))),
                                n_unknown_train=int(np.sum(train & ~known)) if name == 'unknown' else 0,
                                n_unknown_test=int(np.sum(~train & ~known)) if name == 'unknown' else 0))
        for mode in ('raw', 'l2'):
            # Preserve float32 for the raw original-space probe to reproduce its
            # existing result exactly; PCA is fit in float64 by fit_projection.
            x = original if mode == 'raw' else original.astype(np.float64) / np.maximum(
                np.linalg.norm(original.astype(np.float64), axis=1, keepdims=True), 1e-12)
            pca, active, tolerance = fit_projection(x[fit])
            z = project(pca, active, x)
            pred = predictions(x, y, fit)
            pred2 = predictions(z[:, :2], y, fit)
            pred3 = predictions(z[:, :3], y, fit)
            ba = float(balanced_accuracy_score(y[test], pred[test]))
            if mode == 'raw':
                old = next(r for r in old_rows if r['suite'] == 'libero_goal' and r['feature'] == FEATURE
                           and r['layer'] == '9' and r['condition'] == 'within_task'
                           and r['condition_value'] == str(task))
                assert int(old['n_train']) == int(fit.sum()) and int(old['n_test']) == int(test.sum())
                assert abs(ba - float(old['balanced_accuracy'])) < 1e-12, (task, ba, old)
            slug = f'task{task:02d}_H09_{mode}'
            arrays = dict(coordinates=z, phases=y, train=train, known=known,
                          pca_components=pca.components_, pca_mean=pca.mean_, pca_active=active,
                          pca_variance=pca.explained_variance_ratio_, predictions=pred,
                          predictions_2d=pred2, predictions_3d=pred3)
            rows = []
            for k in sorted({2, 4, 8, k_phase}):
                for seed in (0, 1, 2):
                    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z[fit])
                    labels = model.predict(z)
                    arrays[f'clusters_k{k}_seed{seed}'] = labels
                    arrays[f'centers_k{k}_seed{seed}'] = model.cluster_centers_
                    # Verify predictions are nearest-center assignments for all
                    # displayed points, including the held-out episode.
                    d2 = ((z[:, None, :] - model.cluster_centers_[None, :, :])**2).sum(-1)
                    assert np.array_equal(labels, d2.argmin(1))
                    row = dict(task=task, preprocessing=mode, k=k, seed=seed, phase_count_train=k_phase,
                               phase_count_test=len(np.unique(y[test])), n_train=int(fit.sum()), n_test=int(test.sum()),
                               test_ari=float(adjusted_rand_score(y[test], labels[test])),
                               test_nmi=float(normalized_mutual_info_score(y[test], labels[test])),
                               original_linear_ba=ba,
                               pc12_linear_ba=float(balanced_accuracy_score(y[test], pred2[test])),
                               pc123_linear_ba=float(balanced_accuracy_score(y[test], pred3[test])),
                               phase_silhouette=float(silhouette_score(z[test], y[test])),
                               pca_dimensions=z.shape[1], pca_retained_variance=float(pca.explained_variance_ratio_.sum()))
                    rows.append(row); metrics.append(row)
            np.savez(output/'projections'/f'{slug}.npz', **arrays)
            catalog.append(dict(task=task, preprocessing=mode, slug=slug, phase_k=k_phase,
                                instruction=samples[0]['instruction'], metrics=rows,
                                variance=pca.explained_variance_ratio_[:4].tolist(),
                                numerical_rank_tolerance=tolerance))
        print(json.dumps(dict(task=task, known_train=int(fit.sum()), known_test=int(test.sum()),
                              unknown=int((~known).sum())), ensure_ascii=False), flush=True)
    write_csv(output/'metrics.csv', metrics)
    write_csv(output/'phase_support.csv', support)
    atomic_json(output/'catalog.json', catalog)
    atomic_json(output/'validation.json', dict(passed=True, tasks=10, feature=FEATURE, layer=9,
                configurations=len(catalog), cluster_fits=len(metrics), original_probe_reproduced_tasks=10,
                all_cluster_assignments_nearest_center=True,
                split='init0/1 fit, init2 held out; unknown excluded from all fitting and metrics',
                gpu_used=False, source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def draw(ax, z, labels, train, known, *, clusters=False, k=None):
    colors = CLUSTER_COLORS if clusters else PHASE_COLORS
    names = [f'C{i}' for i in range(k)] if clusters else PHASE_NAMES
    for p in range(k if clusters else len(PHASES)-1):
        for training in (True, False):
            use = known & (labels == p) & (train == training)
            if not use.any():
                continue
            ax.scatter(z[use, 0], z[use, 1], s=16 if training else 48,
                       c=colors[p], marker='o' if training else '^',
                       alpha=.4 if training else .98, linewidths=.45,
                       edgecolors='none' if training else '#182434', label=names[p])
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.13)


def render(output):
    catalog = json.loads((output/'catalog.json').read_text())
    for kind in ('phases', 'clusters'):
        fig, axes = plt.subplots(2, 5, figsize=(21, 9))
        for task, ax in enumerate(axes.flat):
            row = next(r for r in catalog if r['task'] == task and r['preprocessing'] == 'raw')
            with np.load(output/'projections'/f"{row['slug']}.npz") as a:
                z, y, train, known = (a[key] for key in ('coordinates', 'phases', 'train', 'known'))
                cluster = a[f"clusters_k{row['phase_k']}_seed0"]
                draw(ax, z, cluster if kind == 'clusters' else y, train, known,
                     clusters=kind == 'clusters', k=row['phase_k'])
            m = next(m for m in row['metrics'] if m['k'] == row['phase_k'] and m['seed'] == 0)
            line = (f"linear BA {m['original_linear_ba']:.0%}; 2D BA {m['pc12_linear_ba']:.0%}"
                    if kind == 'phases' else f"k={row['phase_k']}; held-out ARI {m['test_ari']:.2f}")
            ax.set_title(f'Goal task{task} | {line}', fontsize=9)
            ax.set_xlabel(f"PC1 ({row['variance'][0]:.1%})", fontsize=8)
            ax.set_ylabel(f"PC2 ({row['variance'][1]:.1%})", fontsize=8)
        names, colors = (PHASE_NAMES[:-1], PHASE_COLORS[:-1]) if kind == 'phases' else (
            [f'Cluster {i}' for i in range(8)], CLUSTER_COLORS)
        handles = [Line2D([], [], marker='o', linestyle='', color=c, label=n) for n, c in zip(names, colors)]
        handles += [Line2D([], [], marker=m, linestyle='', color='#475569', label=n)
                    for m, n in [('o', 'Train init0/1'), ('^', 'Held-out init2')]]
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=10, bbox_to_anchor=(.5, .015))
        label = 'Inferred phase colors' if kind == 'phases' else 'Unsupervised k-means colors (seed 0)'
        fig.suptitle('Goal | VLM H9 language-token features | '+label, fontsize=16, y=.99)
        fig.text(.5, .925, 'Each task has its own train-fitted PCA. Clustering uses up to 64 PCs; the plot shows PC1/PC2. Unknown labels excluded.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .125, 1, .91))
        fig.savefig(output/'figures'/f'goal_H09_{kind}_overview.png', dpi=150)
        plt.close(fig)
    build_explorer(output, catalog)
    build_report(output, catalog)
    render_examples(output, catalog)


def render_examples(output, catalog):
    """Two illustrative tasks; the overview retains every task without selection."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row_index, task in enumerate((5, 8)):
        r = next(r for r in catalog if r['task'] == task and r['preprocessing'] == 'raw')
        with np.load(output/'projections'/f"{r['slug']}.npz") as a:
            z, y, train, known = (a[key] for key in ('coordinates', 'phases', 'train', 'known'))
            cluster = a[f"clusters_k{r['phase_k']}_seed0"]
            for col in (0, 1):
                draw(axes[row_index, col], z, y if col == 0 else cluster, train, known,
                     clusters=col == 1, k=r['phase_k'])
                axes[row_index, col].set_xlabel(f"PC1 ({r['variance'][0]:.1%})")
                axes[row_index, col].set_ylabel(f"PC2 ({r['variance'][1]:.1%})")
        m = next(m for m in r['metrics'] if m['k'] == r['phase_k'] and m['seed'] == 0)
        axes[row_index, 0].set_title(f'Goal task{task}: inferred phase colors')
        axes[row_index, 1].set_title(f"Same coordinates: k-means k={r['phase_k']} | held-out ARI {m['test_ari']:.2f}")
    handles = [Line2D([], [], marker='o', linestyle='', color=c, label=n)
               for n, c in zip(PHASE_NAMES[:6], PHASE_COLORS[:6])]
    handles += [Line2D([], [], marker=m, linestyle='', color='#475569', label=n)
                for m, n in [('o', 'Train init0/1'), ('^', 'Held-out init2')]]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=10, bbox_to_anchor=(.5, .015))
    fig.suptitle('VLM H9 language features | two illustrative tasks', fontsize=15)
    fig.text(.5, .943, 'Phase colors on the left; independent cluster colors on the right. All 10 tasks are in the overview / PDF.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .105, 1, .925))
    fig.savefig(output/'figures'/'goal_H09_phase_cluster_examples.png', dpi=160)
    plt.close(fig)


def build_explorer(output, catalog):
    sys.path.insert(0, str(Path(__file__).parent/'_deps'))
    from plotly.offline import get_plotlyjs
    records = []
    for row in catalog:
        with np.load(output/'projections'/f"{row['slug']}.npz") as a:
            clusters = {f"{m['k']}_{m['seed']}": a[f"clusters_k{m['k']}_seed{m['seed']}"].tolist()
                        for m in row['metrics']}
            records.append(dict(row, coordinates=a['coordinates'][:, :4].tolist(), phases=a['phases'].tolist(),
                                clusters=clusters))
    payload = json.dumps(dict(records=records, phase_names=PHASE_NAMES, phase_colors=PHASE_COLORS,
                              cluster_colors=CLUSTER_COLORS,
                              samples={str(t): json.loads((output/f'task{t:02d}_samples.json').read_text()) for t in range(10)}),
                         ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Goal H9 阶段散点与聚类</title>
<style>body{font:15px system-ui;margin:22px;background:#f8fafc;color:#203047}select{padding:6px;margin:6px}.panels{display:grid;grid-template-columns:1fr 1fr}.plot{height:630px}p{line-height:1.65;max-width:1300px}#metrics{padding:12px;background:#eaf0f8;margin:12px 0}@media(max-width:900px){.panels{grid-template-columns:1fr}}</style>
<h1>Goal：H9 特征是否按阶段成簇？</h1><p>每次推理一个点。左侧按轨迹规则推断的阶段着色，右侧按 k-means 簇着色，左右坐标一致。每个 task 单独分析；圆点为 init0/1，三角（3D 中为菱形）为留出 init2。拟合不使用留出观测；阶段未知的点默认隐藏，也不参与拟合。</p>
<label>任务<select id="task"></select></label><label>预处理<select id="mode"><option value="raw">Raw 原始特征</option><option value="l2">每点 L2 归一化</option></select></label>
<label>簇数<select id="k"></select></label><label>种子<select id="seed"><option>0</option><option>1</option><option>2</option></select></label>
<label>坐标<select id="dims"><option value="0,1">PC1 / PC2</option><option value="0,2">PC1 / PC3</option><option value="1,2">PC2 / PC3</option><option value="0,1,2">PC1 / PC2 / PC3</option></select></label>
<label>显示<select id="subset"><option value="all">全部已标阶段</option><option value="test">仅留出轨迹</option><option value="uniform">原始12个均匀选点中的已标阶段</option><option value="unknown">包含阶段未知的点</option></select></label>
<div id="instruction"></div><div id="metrics"></div><div class="panels"><div id="truth" class="plot"></div><div id="cluster" class="plot"></div></div>
<p>聚类在训练 PCA 的至多64维中拟合，阶段标签不参与分簇；默认 k 等于训练集出现的阶段数，这个数量使用了标签，其余 k=2/4/8 也保留。簇编号不等于阶段编号，不同任务之间也不对应。显示筛选只隐藏点，不重新拟合。</p>
<p>ARI 衡量簇和阶段标签的对应：1为一致，0附近为机会水平。阶段可能沿连续轨迹变化，不一定成为球形簇。每个 task 只有一条留出轨迹，相邻点相关；不把点数当独立试验数，不据此宣称统计显著或跳层安全。</p>
<script>__PLOTLY__</script><script>const p=__DATA__,$=id=>document.getElementById(id);
for(let t=0;t<10;t++)$('task').add(new Option('Goal task'+t,String(t)));
function record(){return p.records.find(r=>String(r.task)===$('task').value&&r.preprocessing===$('mode').value)}
function setup(){const r=record();$('k').innerHTML='';[...new Set(r.metrics.map(m=>m.k))].forEach(k=>$('k').add(new Option('k='+k+(k===r.phase_k?'（训练阶段数）':''),String(k))));$('k').value=String(r.phase_k);draw()}
function draw(){const r=record(),samples=p.samples[String(r.task)],k=Number($('k').value),seed=Number($('seed').value),dims=$('dims').value.split(',').map(Number),subset=$('subset').value;
const ids=samples.map((s,i)=>i).filter(i=>(subset==='unknown'||samples[i].phase!=='unknown')&&(subset!=='test'||samples[i].init===2)&&(subset!=='uniform'||samples[i].sources.includes('uniform')));
const axis=a=>'PC'+(a+1)+' ('+(100*r.variance[a]).toFixed(1)+'%)';const ranges=dims.map(a=>{const v=ids.map(i=>r.coordinates[i][a]);if(!v.length)return[-1,1];const lo=Math.min(...v),hi=Math.max(...v),pad=(hi-lo||1)*.08;return[lo-pad,hi+pad]});
function panel(id,labels,colors,names,title){const traces=[];for(let c=0;c<names.length;c++)for(const training of [true,false]){const chosen=ids.filter(i=>labels[i]===c&&(samples[i].init<2)===training);if(!chosen.length)continue;const trace={type:dims.length===3?'scatter3d':'scattergl',mode:'markers',name:names[c]+(training?' train':' held-out'),x:chosen.map(i=>r.coordinates[i][dims[0]]),y:chosen.map(i=>r.coordinates[i][dims[1]]),text:chosen.map(i=>'task'+r.task+' init'+samples[i].init+' request'+samples[i].request+' step'+samples[i].step+'<br>'+p.phase_names[r.phases[i]]),hovertemplate:'%{text}<extra></extra>',marker:{color:colors[c],size:training?7:11,opacity:training?.45:1,symbol:dims.length===3?(training?'circle':'diamond'):(training?'circle':'triangle-up')}};if(dims.length===3)trace.z=chosen.map(i=>r.coordinates[i][dims[2]]);traces.push(trace)}
Plotly.react(id,traces,{title:{text:title},xaxis:{title:{text:axis(dims[0])},range:ranges[0]},yaxis:{title:{text:axis(dims[1])},range:ranges[1]},scene:{xaxis:{title:{text:axis(dims[0])},range:ranges[0]},yaxis:{title:{text:axis(dims[1])},range:ranges[1]},zaxis:{title:{text:axis(dims[2]||2)},range:ranges[2]}},legend:{orientation:'h'},margin:{t:60,l:65,b:90,r:20}},{responsive:true})}
panel('truth',r.phases,p.phase_colors,p.phase_names,'H9 语言特征：阶段颜色');panel('cluster',r.clusters[k+'_'+seed],p.cluster_colors,Array.from({length:k},(_,c)=>'Cluster '+c),'相同坐标：k-means 颜色');
const m=r.metrics.find(m=>m.k===k&&m.seed===seed);$('instruction').textContent='指令：'+r.instruction;$('metrics').textContent='训练已标阶段 '+m.n_train+' 点 / 2条轨迹；留出 '+m.n_test+' 点 / 1条轨迹 · 原始维度线性 BA '+(100*m.original_linear_ba).toFixed(1)+'% · PC1/PC2 线性 BA '+(100*m.pc12_linear_ba).toFixed(1)+'% · 前3PC线性 BA '+(100*m.pc123_linear_ba).toFixed(1)+'% · 聚类留出 ARI '+m.test_ari.toFixed(3)+'，NMI '+m.test_nmi.toFixed(3)+' · 显示 '+ids.length+' 点';}
['task','mode'].forEach(id=>$(id).onchange=setup);['k','seed','dims','subset'].forEach(id=>$(id).onchange=draw);setup();</script></html>'''
    (output/'goal_H09_phase_explorer.html').write_text(page.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', payload))


def build_report(output, catalog):
    raw = [r for r in catalog if r['preprocessing'] == 'raw']
    table = []
    for r in raw:
        rows = [m for m in r['metrics'] if m['k'] == r['phase_k']]
        m = rows[0]
        table.append(f"<tr><td>{r['task']}</td><td>{m['phase_count_train']}/{m['phase_count_test']}</td>"
                     f"<td>{m['n_train']}/{m['n_test']}</td><td>{m['original_linear_ba']:.1%}</td>"
                     f"<td>{m['pc12_linear_ba']:.1%}</td><td>{m['pc123_linear_ba']:.1%}</td>"
                     f"<td>{min(s['test_ari'] for s in rows):.3f}–{max(s['test_ari'] for s in rows):.3f}</td>"
                     f"<td>{m['phase_silhouette']:.3f}</td></tr>")
    mean_ba = float(np.mean([r['metrics'][0]['original_linear_ba'] for r in raw]))
    summary = dict(feature=FEATURE, layer=9, suite='libero_goal', mean_original_linear_ba=mean_ba,
                   tasks=[dict(task=r['task'], k=r['phase_k'], metrics=[m for m in r['metrics'] if m['k'] == r['phase_k']]) for r in raw])
    atomic_json(output/'summary.json', summary)
    text = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Goal H9 阶段可视化</title><style>
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}
@page{size:A4 landscape;margin:14mm;@bottom-center{content:counter(page);font-size:8pt}}@page plot{size:A3 landscape;margin:12mm}
body{font:10pt/1.6 "DejaVu Sans",CJK,sans-serif;color:#203047}h1{font-size:21pt}h2{font-size:14pt}table{border-collapse:collapse;width:100%;font-size:9pt}th,td{padding:1.6mm;border-bottom:1px solid #dce4ef;text-align:left}th{background:#edf3f8}.plot{page:plot;break-before:page}.plot img{max-width:390mm;max-height:245mm;width:100%}.plot p{font-size:9pt}a{color:#2563eb}</style>
<h1>Goal suite：VLM H9 的阶段特征能否自然成簇？</h1>
<p>复用原实验30条轨迹中每次推理的语言 token 均值特征（2048维）。每个 task 的 init0/1 拟合，init2 留出；分别绘制10个 task，避免把任务身份差异混入阶段差异。原始维度的阶段线性读出平均为 __BA__，与原报告逐任务分数完全一致。</p>
<p><b>观察：</b>阶段信息可以读出，但多数任务没有自然分成与阶段逐一对应的清晰簇。Raw、k=训练阶段数时，task1/2/3/5/6/7 的三种子留出 ARI 均低于0.41；task4 为0.479–0.754，task8 为0.387–0.465，task9 为0.472–0.713，存在部分分离。task0 的 Raw ARI=1，但只有两类已知阶段，且 L2 后降至0.165，不能作为稳健多阶段聚类证据。前2PC上的固定线性读出整体弱于原维度，说明二维图不能完整呈现原始读出结果。</p>
<p>每个 task 独立拟合 PCA，仅使用训练集已标阶段的点。k-means 同样只用训练集，使用至多64个PC（本轮训练样本少于64，保留所有非零训练方向），图上展示前两个PC。默认 k 为训练阶段数；也保存 k=2/4/8 和3个固定种子，另有逐点 L2 归一化对照。阶段标签不参与聚类目标，只用于事后评价；默认簇数使用了阶段数量。没有根据留出结果挑选种子或投影。</p>
<table><tr><th>Goal task</th><th>阶段数 训练/留出</th><th>点数 训练/留出</th><th>原维度线性 BA</th><th>前2PC BA</th><th>前3PC BA</th><th>k-means ARI：3种子</th><th>阶段 silhouette</th></tr>__TABLE__</table>
<p>BA 是每个实际留出阶段的召回率平均。ARI=1 表示簇与阶段完全对应，0附近为机会水平，可为负数。silhouette 衡量高维中同阶段点是否紧凑、与其他阶段分离；接近0说明重叠，负数提示部分点更靠近别的阶段。高线性 BA 不保证自然聚类或二维显著分离；阶段也可能表现为连续轨迹。</p>
<p>阶段由 trace 规则推断，并非人工真值。unknown 不参与拟合与评分；例如 task0 排除了28个未知点，留出已知点只有初始1点、接近11点，100%的 BA 不能代表完整操作阶段识别。每 task 仅一条留出轨迹，相邻观测相关；不报告把帧当独立样本的显著性检验，也不把这些结果解释为跳层安全性。</p>
<p>交互入口：<a href="goal_H09_phase_explorer.html">goal_H09_phase_explorer.html</a>，可切换 task、2D/3D、PC组合、簇数、种子、Raw/L2、仅留出与原始均匀选点。CSV 保留全部配置分数，NPZ 保留PCA、坐标、簇中心和逐点预测。</p>
'''.replace('__BA__', f'{mean_ba:.1%}').replace('__TABLE__', ''.join(table))
    for kind, title in [('phases', '相同 task 内：按阶段着色'), ('clusters', '同一组坐标：按无监督聚类着色')]:
        text += f'<section class="plot"><h2>{title}</h2><p>圆点：训练 init0/1；三角：新轨迹 init2。不同 task 使用不同 PCA；同一 task 的两页坐标完全相同。聚类页固定展示 seed0、k=训练阶段数，簇颜色与阶段颜色不对应。</p><img src="{(output/"figures"/f"goal_H09_{kind}_overview.png").as_uri()}"></section>'
    (output/'report_zh.html').write_text(text+'</html>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ('projections', 'figures'):
        (output/name).mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        if not args.render_only:
            compute(args.root.resolve(), output)
        render(output)
    print(json.dumps(dict(passed=True, output=str(output))), flush=True)


if __name__ == '__main__':
    main()
