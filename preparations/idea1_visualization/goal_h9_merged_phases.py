"""Compare fixed semantic phase mergers on the existing Goal H9 feature bank."""
from __future__ import annotations

import argparse
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
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, silhouette_score
from threadpoolctl import threadpool_limits

from .analyze import PHASES, write_csv
from .common import atomic_json
from .goal_h9_phases import PHASE_COLORS, PHASE_NAMES, load_task, predictions

SCHEMES = [
    dict(id='fine', name='Original phases', zh='原阶段',
         mapping=list(range(8))+[-1], names=PHASE_NAMES[:8], colors=PHASE_COLORS[:8]),
    dict(id='three', name='Three broad stages', zh='三大阶段',
         mapping=[0, 0, 1, 1, 2, 2, 2, 2, -1],
         names=['Prepare / approach', 'Contact / manipulate', 'Transport / place / finish'],
         colors=['#2563eb', '#d97706', '#16a34a']),
    dict(id='two', name='Two broad stages', zh='两大阶段',
         mapping=[0, 0, 1, 1, 1, 1, 1, 1, -1],
         names=['Prepare / approach', 'Interaction / transport / finish'],
         colors=['#2563eb', '#dc2626']),
]


def align_clusters(cluster, stages, fit, k):
    """One-to-one color/label matching maximizes TRAIN balanced accuracy only."""
    classes = np.unique(stages[fit])
    assert len(classes) == k
    counts = np.array([[np.sum(fit & (stages == c) & (cluster == j)) for j in range(k)] for c in classes])
    weights = counts / counts.sum(axis=1, keepdims=True)
    rows, cols = linear_sum_assignment(-weights)
    mapping = np.empty(k, int)
    mapping[cols] = classes[rows]
    return mapping[cluster], mapping


def compute(root, source, output):
    catalog, metrics, support = [], [], []
    source_catalog = json.loads((source/'catalog.json').read_text())
    provenance = []
    reused, new_fits = 0, 0
    for task in range(10):
        samples, original = load_task(root, task)
        prior_samples = json.loads((source/f'task{task:02d}_samples.json').read_text())
        assert samples == prior_samples
        atomic_json(output/f'task{task:02d}_samples.json', samples)
        for mode in ('raw', 'l2'):
            source_path = source/'projections'/f'task{task:02d}_H09_{mode}.npz'
            provenance.append(dict(path=str(source_path), sha256=hashlib.sha256(source_path.read_bytes()).hexdigest()))
            with np.load(source_path) as a:
                z, fine, train, known = (a[key] for key in ('coordinates', 'phases', 'train', 'known'))
                fit, test = train & known, (~train) & known
                assert np.array_equal(train, np.array([s['init'] < 2 for s in samples]))
                assert np.array_equal(fine, np.array([PHASES.index(s['phase']) for s in samples]))
                x = original if mode == 'raw' else original.astype(np.float64) / np.maximum(
                    np.linalg.norm(original.astype(np.float64), axis=1, keepdims=True), 1e-12)
                reconstructed = (np.asarray(x, np.float64) - a['pca_mean']) @ a['pca_components'].T
                reconstructed[:, ~a['pca_active']] = 0.
                assert np.allclose(reconstructed, z, rtol=1e-10, atol=1e-8)
                base = next(r for r in source_catalog if r['task'] == task and r['preprocessing'] == mode)
                # Each unique k is fitted at most once per task/mode; existing
                # k=2 and fine-k results are reused byte-for-byte from the source.
                partitions = {}
                for scheme in SCHEMES:
                    y = np.asarray(scheme['mapping'])[fine]
                    k = len(np.unique(y[fit]))
                    eligible = k >= 2 and len(np.unique(y[test])) >= 2
                    slug = f"task{task:02d}_{mode}_{scheme['id']}"
                    if mode == 'raw':
                        for label, name in enumerate(scheme['names']):
                            support.append(dict(task=task, scheme=scheme['id'], stage=name,
                                                train=int(np.sum(fit & (y == label))),
                                                test=int(np.sum(test & (y == label)))))
                    saved = dict(coordinates=z, phases=y, fine_phases=fine, train=train, known=known)
                    score = dict(task=task, preprocessing=mode, scheme=scheme['id'],
                                 k=k, eligible=eligible, n_train=int(fit.sum()), n_test=int(test.sum()),
                                 test_stage_count=len(np.unique(y[test])),
                                 original_linear_ba=None, pc12_linear_ba=None, pc123_linear_ba=None,
                                 stage_silhouette=None, stage_silhouette_2d=None)
                    if eligible:
                        if scheme['id'] == 'fine':
                            pred, pred2, pred3 = (a[key] for key in ('predictions', 'predictions_2d', 'predictions_3d'))
                        else:
                            pred, pred2, pred3 = (predictions(xx, y, fit) for xx in (x, z[:, :2], z[:, :3]))
                        score.update(original_linear_ba=float(balanced_accuracy_score(y[test], pred[test])),
                                     pc12_linear_ba=float(balanced_accuracy_score(y[test], pred2[test])),
                                     pc123_linear_ba=float(balanced_accuracy_score(y[test], pred3[test])),
                                     stage_silhouette=float(silhouette_score(z[test], y[test])),
                                     stage_silhouette_2d=float(silhouette_score(z[test, :2], y[test])))
                        saved.update(predictions=pred, predictions_2d=pred2, predictions_3d=pred3)
                    rows = []
                    for seed in (0, 1, 2):
                        row = dict(score, seed=seed, test_ari=None, kmeans_mapped_ba=None,
                                   ari_with_unchanged_fine_partition=None)
                        if eligible:
                            key = f'clusters_k{k}_seed{seed}'
                            if key not in partitions:
                                if key in a:
                                    partitions[key] = (a[key].copy(), a[f'centers_k{k}_seed{seed}'].copy())
                                    reused += 1
                                else:
                                    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z[fit])
                                    partitions[key] = (model.predict(z), model.cluster_centers_)
                                    new_fits += 1
                            cluster, centers = partitions[key]
                            assert np.array_equal(cluster, ((z[:, None, :] - centers[None, :, :])**2).sum(-1).argmin(1))
                            aligned, mapping = align_clusters(cluster, y, fit, k)
                            row.update(test_ari=float(adjusted_rand_score(y[test], cluster[test])),
                                       kmeans_mapped_ba=float(balanced_accuracy_score(y[test], aligned[test])),
                                       ari_with_unchanged_fine_partition=float(adjusted_rand_score(
                                           y[test], a[f"clusters_k{base['phase_k']}_seed{seed}"][test])))
                            saved[f'clusters_seed{seed}'] = cluster
                            saved[f'aligned_seed{seed}'] = aligned
                            saved[f'centers_seed{seed}'] = centers
                            saved[f'cluster_stage_mapping_seed{seed}'] = mapping
                        rows.append(row); metrics.append(row)
                    np.savez(output/'projections'/f'{slug}.npz', **saved)
                    catalog.append(dict(task=task, preprocessing=mode, scheme=scheme['id'], slug=slug,
                                        instruction=samples[0]['instruction'], eligible=eligible,
                                        variance=base['variance'], metrics=rows))
        print(json.dumps(dict(task=task, passed=True)), flush=True)
    write_csv(output/'metrics.csv', metrics)
    write_csv(output/'stage_support.csv', support)
    atomic_json(output/'catalog.json', catalog)
    atomic_json(output/'methods.json', dict(schemes=SCHEMES, source=str(source),
                split='init0/1 fit, init2 held out', pca='Exact reuse; no refit or coordinate changes',
                unknown='Excluded identically to the source analysis', clustering='k=number of observed training stages; n_init10; seeds0/1/2',
                label_matching='One-to-one Hungarian mapping maximizing training balanced accuracy; no held-out labels',
                source_projections=provenance, reused_partitions=reused, new_partition_fits=new_fits,
                gpu_used=False, source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def record(catalog, task, scheme, mode='raw'):
    return next(r for r in catalog if (r['task'], r['scheme'], r['preprocessing']) == (task, scheme, mode))


def scatter(ax, a, scheme, *, aligned=False, seed=0):
    z, train, known = a['coordinates'], a['train'], a['known']
    y = a[f'aligned_seed{seed}'] if aligned else a['phases']
    for label, color in enumerate(scheme['colors']):
        for training in (True, False):
            use = known & (y == label) & (train == training)
            ax.scatter(z[use, 0], z[use, 1], s=14 if training else 44,
                       color=color, marker='o' if training else '^', alpha=.35 if training else .98,
                       edgecolors='none' if training else '#182434', linewidths=.4)
    ax.tick_params(labelsize=7); ax.grid(alpha=.13)


def render(output):
    catalog = json.loads((output/'catalog.json').read_text())
    for scheme in SCHEMES[1:]:
        fig, axes = plt.subplots(2, 5, figsize=(21, 9))
        for task, ax in enumerate(axes.flat):
            r = record(catalog, task, scheme['id'])
            with np.load(output/'projections'/f"{r['slug']}.npz") as a:
                scatter(ax, a, scheme)
            m = r['metrics'][0]
            text = (f"linear BA {m['original_linear_ba']:.0%}; k-means ARI {m['test_ari']:.2f}"
                    if r['eligible'] else 'Only one observed stage; not evaluable')
            ax.set_title(f'Goal task{task} | '+text, fontsize=9)
            ax.set_xlabel(f"PC1 ({r['variance'][0]:.1%})", fontsize=8)
            ax.set_ylabel(f"PC2 ({r['variance'][1]:.1%})", fontsize=8)
        handles = [Line2D([], [], marker='o', linestyle='', color=c, label=n)
                   for n, c in zip(scheme['names'], scheme['colors'])]
        handles += [Line2D([], [], marker=m, linestyle='', color='#475569', label=n)
                    for m, n in [('o', 'Train init0/1'), ('^', 'Held-out init2')]]
        fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=10, bbox_to_anchor=(.5, .025))
        fig.suptitle('Goal | VLM H9 | '+scheme['name'], fontsize=16, y=.99)
        fig.text(.5, .925, 'Identical PCA coordinates and points as before merging; no label-driven projection. Unknown excluded.',
                 ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .13, 1, .91))
        fig.savefig(output/'figures'/f"{scheme['id']}_stage_overview.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for row, task in enumerate((5, 8)):
        for col, scheme in enumerate(SCHEMES):
            r = record(catalog, task, scheme['id'])
            with np.load(output/'projections'/f"{r['slug']}.npz") as a:
                scatter(axes[row, col], a, scheme)
            axes[row, col].set_title(f"Goal task{task} | {scheme['name']}", fontsize=10)
            axes[row, col].set_xlabel('PC1'); axes[row, col].set_ylabel('PC2')
    fig.suptitle('Same H9 features, same points, same PCA | only stage definitions change', fontsize=14)
    fig.text(.5, .025, 'Fine colors: gray initial, blue approach, orange contact, red grasp, green transport, purple alignment.\nThree stages: blue prepare/approach, orange contact/manipulate, green transport/place. Two stages: blue prepare/approach, red subsequent operation.\nCircles: train init0/1. Triangles: held-out init2. All 10 tasks are included in the overview and report.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .125, 1, .945)); fig.savefig(output/'figures'/'merge_comparison_examples.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    scheme = SCHEMES[2]
    for row, task in enumerate((5, 8)):
        r = record(catalog, task, 'two')
        with np.load(output/'projections'/f"{r['slug']}.npz") as a:
            scatter(axes[row, 0], a, scheme)
            scatter(axes[row, 1], a, scheme, aligned=True)
        m = r['metrics'][0]
        axes[row, 0].set_title(f'Goal task{task}: two merged stages')
        axes[row, 1].set_title(f"k-means k=2 | ARI {m['test_ari']:.2f} | BA {m['kmeans_mapped_ba']:.0%}")
        for ax in axes[row]: ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    fig.suptitle('Actual merged stages vs unsupervised clusters | colors matched using training only', fontsize=12)
    fig.text(.5, .025, 'Blue: prepare/approach. Red: subsequent operation. Circles: train init0/1; triangles: held-out init2.\nBoth panels use exactly the same coordinates. Clustering uses all retained training PCs; the plot shows PC1/PC2.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, .945)); fig.savefig(output/'figures'/'two_stage_cluster_examples.png', dpi=160); plt.close(fig)
    report(output, catalog)
    explorer(output, catalog)


def report(output, catalog):
    table = []
    for task in range(10):
        cells = [str(task)]
        for s in SCHEMES:
            r = record(catalog, task, s['id']); rows = r['metrics']; m = rows[0]
            cells.extend([f"{m['original_linear_ba']:.1%}" if r['eligible'] else '不可评估',
                          f"{min(x['test_ari'] for x in rows):.3f}–{max(x['test_ari'] for x in rows):.3f}" if r['eligible'] else '仅一类'])
        table.append('<tr>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
    # Use identical tasks for every macro comparison; task0 loses all stage
    # contrasts after merging and is explicitly excluded from all three means.
    eligible = [t for t in range(10) if all(record(catalog, t, s['id'])['eligible'] for s in SCHEMES)]
    summary = dict(common_tasks=eligible, excluded_tasks=[t for t in range(10) if t not in eligible], schemes=[])
    for s in SCHEMES:
        rows = [record(catalog, t, s['id'])['metrics'] for t in eligible]
        summary['schemes'].append(dict(scheme=s['id'],
            linear_ba=float(np.mean([rr[0]['original_linear_ba'] for rr in rows])),
            pc12_linear_ba=float(np.mean([rr[0]['pc12_linear_ba'] for rr in rows])),
            mean_ari_over_tasks_and_seeds=float(np.mean([r['test_ari'] for rr in rows for r in rr])),
            mean_silhouette=float(np.mean([rr[0]['stage_silhouette'] for rr in rows]))))
    atomic_json(output/'summary.json', summary)
    macro = ''.join(f"<li>{s['zh']}：原维度线性 BA {m['linear_ba']:.1%}，前2PC线性 BA {m['pc12_linear_ba']:.1%}，任务×种子平均 ARI {m['mean_ari_over_tasks_and_seeds']:.3f}。</li>" for s, m in zip(SCHEMES, summary['schemes']))
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H9 合并阶段对照</title><style>
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}@page{size:A4 landscape;margin:14mm;@bottom-center{content:counter(page);font-size:8pt}}@page plot{size:A3 landscape;margin:12mm}body{font:10pt/1.6 "DejaVu Sans",CJK,sans-serif;color:#203047}h1{font-size:20pt}h2{font-size:14pt}table{border-collapse:collapse;width:100%;font-size:9pt}td,th{padding:1.4mm;border-bottom:1px solid #dce4ef;text-align:left}th{background:#edf3f8}.plot{page:plot;break-before:page}.plot img{width:100%;max-height:235mm;object-fit:contain}</style>
<h1>阶段合并后，Goal H9 特征能否分隔？</h1>
<p>结论：部分任务改善，但“原阶段过细”不能解释全部重叠。task5 合并为“准备／接近”与“接触／后续操作”后，三种子的留出 ARI 均从0.316升至0.621；task9 的两阶段 ARI 均为0.606。task8 的三阶段 ARI 为0.357–0.531，但两阶段仅0.047；task2 两阶段 ARI 为0。因此不能宣称合并后所有任务都能自然分成对应的簇。</p>
<p>归一化也会影响自然簇：两阶段的 task7，Raw 三种子 ARI 为0.200–0.350，L2 后三个种子均为1，训练匹配后的留出阶段 BA 均为100%（16个点）；task9 的 L2 三种子 ARI 均为0.799、聚类阶段 BA 均为96.4%。task5 的 Raw 三种子聚类阶段 BA 均为90%（18/20点对应正确），L2 为90%–100%。这些是具体配置下的探索观察，不能把每个 task 最好的预处理拼成总体结果。</p>
<p>固定合并规则：三阶段①initial＋approach（准备／接近）；②contact＋grasp_or_manipulate（交互／操作）；③transport＋goal_alignment＋release_or_final_action＋success（搬运／就位／完成）。两阶段①initial＋approach；②其余已知操作阶段。它们合并的是原 trace 规则标签，不代表通过人工核实的真实接触前后。unknown 仍排除。</p>
<p>数据、选点、PCA坐标和训练／留出划分完全沿用原分析。每 task 的 init0/1 拟合、init2 留出。PCA未重拟合；已有 k=2 聚类原样复用，只为原来没有的 k=3 补拟合。默认 k 为该合并规则在训练中出现的阶段数；标签不参与 k-means 目标。用训练标签匹配簇颜色，留出标签只评分。高维聚类使用全部保留的训练PC；图仍展示前两个PC。原维度和二维读出均使用固定 StandardScaler＋Ridge(alpha10)，没有为合并规则调参。</p>
<table><thead><tr><th>Task</th><th>原阶段 BA</th><th>原阶段 ARI</th><th>三阶段 BA</th><th>三阶段 ARI</th><th>两阶段 BA</th><th>两阶段 ARI</th></tr></thead><tbody>__TABLE__</tbody></table>
<p>表内均为 Raw 特征留出结果；ARI 为三个固定种子的范围，1表示分簇与阶段一致，0附近表示机会水平。不同阶段数量改变了分类难度，BA提高本身不能证明表征增强；须同时看新轨迹、阶段支持量及聚类。每 task 仅一条留出轨迹，不能把相邻观测当独立重复，也不据此宣称统计显著或跳层安全。</p>
<h2>相同9个可比较任务：task1–9</h2><ul>__MACRO__</ul>
<p>task0 合并后只剩“准备／接近”一类，不具备区分条件，三种口径的宏平均均排除它。task5、task7 在三阶段规则下也只有两类真实出现的阶段，未人为补出第三类。Raw／逐点L2、三种子、全部任务结果均保留在 metrics.csv；stage_support.csv 保留各阶段训练／留出数量。NPZ 含相同PCA坐标、合并标签及逐点聚类／预测。交互图可切换规则、任务、预处理、种子和2D/3D。</p>
'''.replace('__TABLE__', ''.join(table)).replace('__MACRO__', macro)
    for name, title in [('merge_comparison_examples', '同一坐标下直接比较合并前后：task5 与 task8'),
                        ('two_stage_cluster_examples', '两阶段标签与真正的 k-means 分簇'),
                        ('three_stage_overview', '三阶段合并：完整十个任务'),
                        ('two_stage_overview', '两阶段合并：完整十个任务')]:
        body += f'<section class="plot"><h2>{title}</h2><p>半透明圆点：训练 init0/1；实心三角：新轨迹 init2。没有移动点来改善分离；不同任务各有固定的PCA坐标。</p><img src="{(output/"figures"/(name+".png")).as_uri()}"></section>'
    (output/'report_zh.html').write_text(body+'</html>')


def explorer(output, catalog):
    sys.path.insert(0, str(Path(__file__).parent/'_deps'))
    from plotly.offline import get_plotlyjs
    records = []
    for r in catalog:
        with np.load(output/'projections'/f"{r['slug']}.npz") as a:
            records.append(dict(r, coordinates=a['coordinates'][:, :4].tolist(), phases=a['phases'].tolist(),
                                aligned=[a[f'aligned_seed{s}'].tolist() for s in (0, 1, 2)] if r['eligible'] else []))
    payload = json.dumps(dict(records=records, schemes=SCHEMES,
        samples={str(t):json.loads((output/f'task{t:02d}_samples.json').read_text()) for t in range(10)}), ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>H9 阶段合并对照</title>
<style>body{font:15px system-ui;margin:22px;background:#f8fafc;color:#203047}select{padding:6px;margin:6px}.panels{display:grid;grid-template-columns:1fr 1fr}.plot{height:630px}p{line-height:1.65;max-width:1300px}#metrics{padding:12px;background:#eaf0f8;margin:12px 0}@media(max-width:900px){.panels{grid-template-columns:1fr}}</style>
<h1>H9 阶段合并：点不动，只改变阶段定义</h1><p>左侧阶段颜色，右侧无监督聚类（仅用训练标签对齐颜色）。圆点是训练轨迹，三角／3D菱形是新轨迹。阶段未知点排除；PCA坐标与原分析完全相同。k为训练中实际出现的阶段数；只有一个阶段时不评估聚类。</p>
<label>任务<select id="task"></select></label><label>阶段规则<select id="scheme"></select></label><label>特征<select id="mode"><option value="raw">Raw</option><option value="l2">L2</option></select></label><label>种子<select id="seed"><option>0</option><option>1</option><option>2</option></select></label><label>坐标<select id="dims"><option value="0,1">PC1 / PC2</option><option value="0,2">PC1 / PC3</option><option value="1,2">PC2 / PC3</option><option value="0,1,2">PC1 / PC2 / PC3</option></select></label><label>显示<select id="subset"><option value="all">全部</option><option value="test">仅新轨迹 init2</option></select></label>
<div id="metrics"></div><div class="panels"><div id="truth" class="plot"></div><div id="cluster" class="plot"></div></div><p>三阶段：准备／接近 → 交互／操作 → 搬运／就位。两阶段：准备／接近 vs 其余操作。每 task 只有一条留出轨迹，不能把点数当独立试验数。ARI衡量簇与阶段的对应，不是动作成功率或跳层依据。</p>
<script>__PLOTLY__</script><script>const p=__DATA__,$=id=>document.getElementById(id);for(let t=0;t<10;t++)$('task').add(new Option('Goal task'+t,String(t)));p.schemes.forEach(s=>$('scheme').add(new Option(s.zh,s.id)));
function draw(){const r=p.records.find(r=>String(r.task)===$('task').value&&r.scheme===$('scheme').value&&r.preprocessing===$('mode').value),s=p.schemes.find(s=>s.id===r.scheme),samples=p.samples[String(r.task)],seed=Number($('seed').value),dims=$('dims').value.split(',').map(Number);const ids=samples.map((v,i)=>i).filter(i=>r.phases[i]>=0&&($('subset').value!=='test'||samples[i].init===2));const axis=a=>'PC'+(a+1)+' ('+(100*r.variance[a]).toFixed(1)+'%)',ranges=dims.map(a=>{const v=ids.map(i=>r.coordinates[i][a]),lo=Math.min(...v),hi=Math.max(...v),d=(hi-lo||1)*.08;return[lo-d,hi+d]});
function panel(id,labels,title){const traces=[];for(let c=0;c<s.names.length;c++)for(const training of [true,false]){const use=ids.filter(i=>labels[i]===c&&(samples[i].init<2)===training);if(!use.length)continue;const t={type:dims.length===3?'scatter3d':'scattergl',mode:'markers',name:s.names[c]+(training?' train':' held-out'),x:use.map(i=>r.coordinates[i][dims[0]]),y:use.map(i=>r.coordinates[i][dims[1]]),text:use.map(i=>'init'+samples[i].init+' request'+samples[i].request+'<br>original '+samples[i].phase),hovertemplate:'%{text}<extra></extra>',marker:{color:s.colors[c],size:training?7:11,opacity:training?.4:1,symbol:dims.length===3?(training?'circle':'diamond'):(training?'circle':'triangle-up')}};if(dims.length===3)t.z=use.map(i=>r.coordinates[i][dims[2]]);traces.push(t)}Plotly.react(id,traces,{title:{text:title},xaxis:{title:{text:axis(dims[0])},range:ranges[0]},yaxis:{title:{text:axis(dims[1])},range:ranges[1]},scene:{xaxis:{title:{text:axis(dims[0])},range:ranges[0]},yaxis:{title:{text:axis(dims[1])},range:ranges[1]},zaxis:{title:{text:axis(dims[2]||2)},range:ranges[2]}},legend:{orientation:'h'},margin:{t:60,l:65,b:100,r:20}},{responsive:true})}
panel('truth',r.phases,'实际阶段：'+s.zh);if(r.eligible)panel('cluster',r.aligned[seed],'k-means：训练标签匹配颜色');else Plotly.react('cluster',[],{title:{text:'只有一个已标阶段，无法评估区分效果'}});const m=r.metrics[seed],pct=v=>(100*v).toFixed(1)+'%';$('metrics').textContent=r.instruction+' · '+m.n_train+'训练点 / '+m.n_test+'留出点 · '+(r.eligible?'原维度线性 BA '+pct(m.original_linear_ba)+' · 前2PC线性 BA '+pct(m.pc12_linear_ba)+' · k='+m.k+' · 留出 ARI '+m.test_ari.toFixed(3)+' · 聚类对应阶段 BA '+pct(m.kmeans_mapped_ba):'合并后仅一类：不计算分离指标');}
['task','scheme','mode','seed','dims','subset'].forEach(id=>$(id).onchange=draw);$('task').value='5';$('scheme').value='two';draw();</script></html>'''
    (output/'merged_phase_explorer.html').write_text(page.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', payload))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    for name in ('projections', 'figures'): (output/name).mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        if not args.render_only: compute(args.root.resolve(), args.source.resolve(), output)
        render(output)
    print(json.dumps(dict(passed=True, output=str(output))), flush=True)


if __name__ == '__main__':
    main()
