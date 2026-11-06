"""All ten Goal tasks in shared H9 PCA coordinates, with two/three stages."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from threadpoolctl import threadpool_limits

from .analyze import PHASES, fit_projection, project, write_csv
from .common import atomic_json
from .goal_h9_phases import load_task, predictions
from .goal_h9_merged_phases import SCHEMES, align_clusters, scatter

STAGES = [SCHEMES[2], SCHEMES[1]]
TASK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#6b7280', '#bcbd22', '#17becf']


def compute(root, output):
    samples, values = [], []
    for task in range(10):
        ss, xx = load_task(root, task)
        samples.extend(ss); values.append(xx)
    x0 = np.concatenate(values)
    task = np.array([s['task'] for s in samples])
    train = np.array([s['init'] < 2 for s in samples])
    fine = np.array([PHASES.index(s['phase']) for s in samples])
    known = fine != PHASES.index('unknown')
    fit, test = train & known, (~train) & known
    assert len(samples) == 549 and fit.sum() == 347 and test.sum() == 174
    assert len({(s['task'], s['init']) for s in samples if s['init'] < 2}) == 20
    assert len({(s['task'], s['init']) for s in samples if s['init'] == 2}) == 10
    atomic_json(output/'samples.json', samples)
    np.savez(output/'feature_bank.npz', h9_language=x0, tasks=task, train=train, known=known, fine_phases=fine)
    support = []
    for s in STAGES:
        y = np.asarray(s['mapping'])[fine]
        for t in range(10):
            for c, name in enumerate(s['names']):
                support.append(dict(scheme=s['id'], task=t, stage=name,
                                    train=int(np.sum(fit & (task == t) & (y == c))),
                                    test=int(np.sum(test & (task == t) & (y == c)))))
    write_csv(output/'stage_support.csv', support)
    catalog, metrics, within_task = [], [], []
    for mode in ('raw', 'l2'):
        x = x0 if mode == 'raw' else x0.astype(np.float64) / np.maximum(
            np.linalg.norm(x0.astype(np.float64), axis=1, keepdims=True), 1e-12)
        # ONE PCA for the whole ten-task training set. This is not a
        # concatenation of unrelated per-task PCA coordinates.
        pca, active, tolerance = fit_projection(x[fit])
        z = project(pca, active, x)
        arrays = dict(coordinates=z, train=train, known=known, tasks=task,
                      pca_mean=pca.mean_, pca_components=pca.components_, pca_active=active,
                      pca_variance=pca.explained_variance_ratio_)
        mode_rows = []
        for s in STAGES:
            y = np.asarray(s['mapping'])[fine]
            k = len(s['names'])
            assert len(np.unique(y[fit])) == k and len(np.unique(y[test])) == k
            pred, pred2, pred3 = (predictions(xx, y, fit) for xx in (x, z[:, :2], z[:, :3]))
            # Task-only baseline uses training majority stage for each task.
            task_baseline = np.zeros(len(samples), int)
            for t in range(10):
                classes, counts = np.unique(y[fit & (task == t)], return_counts=True)
                task_baseline[task == t] = classes[counts.argmax()]
                use = test & (task == t)
                evaluable = len(np.unique(y[use])) >= 2
                within_task.append(dict(preprocessing=mode, scheme=s['id'], task=t,
                    n_test=int(use.sum()), test_stage_count=len(np.unique(y[use])), eligible=evaluable,
                    pooled_classifier_ba=float(balanced_accuracy_score(y[use], pred[use])) if evaluable else None,
                    pooled_pc12_classifier_ba=float(balanced_accuracy_score(y[use], pred2[use])) if evaluable else None,
                    task_only_baseline_ba=float(balanced_accuracy_score(y[use], task_baseline[use])) if evaluable else None))
            arrays[f'phases_{s["id"]}'] = y
            arrays[f'predictions_{s["id"]}'] = pred
            arrays[f'predictions_2d_{s["id"]}'] = pred2
            arrays[f'predictions_3d_{s["id"]}'] = pred3
            arrays[f'task_baseline_{s["id"]}'] = task_baseline
            for seed in (0, 1, 2):
                model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z[fit])
                labels = model.predict(z)
                assert np.array_equal(labels, ((z[:, None] - model.cluster_centers_[None])**2).sum(-1).argmin(1))
                aligned, mapping = align_clusters(labels, y, fit, k)
                arrays[f'clusters_{s["id"]}_seed{seed}'] = labels
                arrays[f'aligned_{s["id"]}_seed{seed}'] = aligned
                arrays[f'centers_{s["id"]}_seed{seed}'] = model.cluster_centers_
                arrays[f'mapping_{s["id"]}_seed{seed}'] = mapping
                row = dict(preprocessing=mode, scheme=s['id'], k=k, seed=seed,
                    n_train=int(fit.sum()), n_test=int(test.sum()),
                    phase_ari=float(adjusted_rand_score(y[test], labels[test])),
                    task_ari=float(adjusted_rand_score(task[test], labels[test])),
                    kmeans_stage_ba=float(balanced_accuracy_score(y[test], aligned[test])),
                    pooled_linear_ba=float(balanced_accuracy_score(y[test], pred[test])),
                    pc12_linear_ba=float(balanced_accuracy_score(y[test], pred2[test])),
                    pc123_linear_ba=float(balanced_accuracy_score(y[test], pred3[test])),
                    task_only_baseline_ba=float(balanced_accuracy_score(y[test], task_baseline[test])),
                    retained_pca_variance=float(pca.explained_variance_ratio_.sum()))
                metrics.append(row); mode_rows.append(row)
        np.savez(output/'projections'/f'joint_H09_{mode}.npz', **arrays)
        catalog.append(dict(preprocessing=mode, projection=f'projections/joint_H09_{mode}.npz',
            variance=pca.explained_variance_ratio_[:4].tolist(), numerical_rank_tolerance=tolerance,
            metrics=mode_rows))
        print(json.dumps(dict(preprocessing=mode, passed=True, metrics=mode_rows)), flush=True)
    write_csv(output/'metrics.csv', metrics); write_csv(output/'within_task_pooled_probes.csv', within_task)
    atomic_json(output/'catalog.json', catalog)
    atomic_json(output/'methods.json', dict(feature='vlm/language/hidden', layer=9, suite='libero_goal',
        schemes=STAGES, task_count=10, all_points=549, known_points=521, unknown_excluded=28,
        train_points=347, test_points=174, training_episodes=20, heldout_episodes=10,
        pca='one training-only shared PCA per raw/L2 mode, maximum 64 PCs; no per-task centering or whitening',
        split='init0/1 training, init2 held out for all tasks; known tasks, new episodes, not unseen-task generalization',
        classifier='ONE StandardScaler + RidgeClassifier(alpha10, LSQR) across tasks per stage scheme; not ten separate classifiers',
        cluster_matching='Hungarian matching maximizes training balanced accuracy only',
        gpu_used=False, source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def labels_for(a, s):
    return dict(coordinates=a['coordinates'], phases=a[f'phases_{s["id"]}'],
                known=a['known'], train=a['train'], aligned_seed0=a[f'aligned_{s["id"]}_seed0'])


def label_axes(ax, record):
    ax.set_xlabel(f"PC1 ({record['variance'][0]:.1%})")
    ax.set_ylabel(f"PC2 ({record['variance'][1]:.1%})")


def render(output):
    catalog = json.loads((output/'catalog.json').read_text())
    for r in catalog:
        with np.load(output/r['projection']) as a:
            # Both stage schemes use the identical joint coordinates.
            fig, axes = plt.subplots(1, 2, figsize=(13, 6))
            for ax, s in zip(axes, STAGES):
                scatter(ax, labels_for(a, s), s); label_axes(ax, r)
                y = a[f'phases_{s["id"]}']; counts = np.bincount(y[a['known']])
                ax.set_title(s['name']+f" | {int(a['known'].sum())} points", fontsize=12)
                handles = [Line2D([], [], marker='o', linestyle='', color=c, label=f'{n} ({counts[i]})')
                           for i, (n, c) in enumerate(zip(s['names'], s['colors']))]
                ax.legend(handles=handles, loc='best', fontsize=8)
            fig.suptitle(f"All 10 Goal tasks | VLM H9 language features | {r['preprocessing'].upper()}", fontsize=15)
            fig.text(.5, .025, 'Circles: 347 training observations (init0/1). Triangles: 174 held-out observations (init2).\nSame shared training PCA on both sides; 28 phase-unknown points excluded. Each point is one inference.', ha='center', fontsize=10)
            fig.tight_layout(rect=(0, .12, 1, .94))
            fig.savefig(output/'figures'/f"joint_stages_{r['preprocessing']}.png", dpi=160); plt.close(fig)
            fig, axes = plt.subplots(2, 3, figsize=(17, 10))
            for row, s in enumerate(STAGES):
                arrays = labels_for(a, s)
                scatter(axes[row, 0], arrays, s)
                scatter(axes[row, 1], arrays, s, aligned=True)
                task_arrays = dict(arrays, phases=a['tasks'])
                scatter(axes[row, 2], task_arrays, dict(names=[str(i) for i in range(10)], colors=TASK_COLORS))
                m = next(m for m in r['metrics'] if m['scheme'] == s['id'] and m['seed'] == 0)
                axes[row, 0].set_title(s['name']+': stage colors', fontsize=11)
                axes[row, 1].set_title(f"k={m['k']} clusters | phase ARI {m['phase_ari']:.3f} | stage BA {m['kmeans_stage_ba']:.0%}", fontsize=10)
                axes[row, 2].set_title(f"Same points: task colors | task ARI {m['task_ari']:.3f}", fontsize=10)
                for ax in axes[row]: label_axes(ax, r)
            handles = [Line2D([], [], marker='o', linestyle='', color=c, label=f'task{i}') for i, c in enumerate(TASK_COLORS)]
            fig.legend(handles=handles, loc='lower center', ncol=10, fontsize=9, bbox_to_anchor=(.5, .025))
            fig.suptitle(f"All ten tasks in shared coordinates | {r['preprocessing'].upper()} | seed0", fontsize=15)
            fig.text(.5, .075, 'Stage, cluster and task panels contain identical points. Cluster colors match stages using training labels only.\nClustering uses up to 64 PCs; plots show PC1/PC2. Task colors provide an identity-confound check.', ha='center', fontsize=9)
            fig.tight_layout(rect=(0, .15, 1, .95))
            fig.savefig(output/'figures'/f"joint_stage_cluster_task_{r['preprocessing']}.png", dpi=150); plt.close(fig)
    build_report(output, catalog); build_explorer(output, catalog)


def build_report(output, catalog):
    rows = []
    for r in catalog:
        for s in STAGES:
            mm = [m for m in r['metrics'] if m['scheme'] == s['id']]; m = mm[0]
            rows.append(f"<tr><td>{r['preprocessing']}</td><td>{s['zh']}</td>"
                f"<td>{m['pooled_linear_ba']:.1%}</td><td>{m['pc12_linear_ba']:.1%}</td>"
                f"<td>{m['task_only_baseline_ba']:.1%}</td>"
                f"<td>{min(v['phase_ari'] for v in mm):.3f}–{max(v['phase_ari'] for v in mm):.3f}</td>"
                f"<td>{min(v['kmeans_stage_ba'] for v in mm):.1%}–{max(v['kmeans_stage_ba'] for v in mm):.1%}</td></tr>")
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>十任务联合阶段可视化</title><style>
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}@page{size:A4 landscape;margin:14mm;@bottom-center{content:counter(page);font-size:8pt}}@page plot{size:A3 landscape;margin:12mm}body{font:10pt/1.65 "DejaVu Sans",CJK,sans-serif;color:#203047}h1{font-size:21pt}h2{font-size:14pt}table{width:100%;border-collapse:collapse;font-size:9pt}th,td{padding:2mm;border-bottom:1px solid #dce4ef;text-align:left}th{background:#edf3f8}.plot{page:plot;break-before:page}.plot img{width:100%;max-height:235mm;object-fit:contain}</style>
<h1>十个 Goal task 合并：H9 的两阶段／三阶段分布</h1>
<p>这次把十个任务的实际特征点放在同一坐标系里，不是拼接十套不同的PCA。共549次推理，排除28个阶段未知点后，显示521点：347个训练点来自20条init0/1轨迹；174个留出点来自10条init2轨迹。复用既有特征，不运行策略或GPU。</p>
<p><b>观察：</b>两／三阶段在联合散点中明显交错，没有形成对应的大团。Raw、k=2的留出阶段ARI为−0.006至0.003，聚类阶段BA为49.2%–50.6%；k=3的阶段ARI为0.032，聚类阶段BA为44.2%。Raw与L2的seed0两簇均把task7全部49个已知阶段点独立成簇，另外472点全是其余task；这组自然簇对应task身份差异，不能当成阶段分簇证据。</p>
<p>原2048维的统一阶段分类器仍能在留出轨迹上读出两阶段96.6%、三阶段92.5%的BA，前2PC上则分别为53.7%、33.3%。因此“这里没有清晰阶段簇”不能推成“表征没有阶段信息”。统一分类器在各task内部、排除只有一类的task0后，BA宏平均分别为96.4%、92.8%；它优于仅task多数阶段基线，仍只检验已知task的新轨迹。Raw前两PC占训练方差47.3%，用于聚类的前64PC占91.8%。</p>
<p>两阶段：准备／接近（initial＋approach）vs其余已知操作阶段。三阶段：准备／接近；接触／操作（contact＋grasp_or_manipulate）；搬运／就位／完成（transport＋goal_alignment＋release_or_final_action＋success）。两种规则使用相同样本和相同PCA坐标。task0合并后虽然只有一类，但仍保留在联合分布中；不能将其单独看成区分证据。</p>
<p>Raw和每点L2各自仅在全部任务的已知阶段训练点上拟合一套共享PCA，最多64维；没有按任务减均值或使用阶段标签选投影。k-means分别固定k=2/3，三个种子、n_init10，使用高维PCA；图展示前两维。簇颜色只用训练标签匹配。另画同坐标的task颜色，辅助检查自然簇是否更对应任务身份。</p>
<table><thead><tr><th>特征</th><th>阶段</th><th>统一分类器原维度BA</th><th>前2PC分类BA</th><th>仅task多数阶段基线</th><th>聚类阶段ARI（三种子）</th><th>聚类阶段BA（三种子）</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>分类器现在是跨十个任务统一训练的一个阶段分类器，仍只检验已见任务的新轨迹，不是新任务泛化。BA为阶段召回率平均。ARI=1表示分簇与阶段一致，0附近为机会水平；颜色一一对应关系由训练集确定。task和阶段组成相关，仅凭混合后的高读出不能证明通用阶段感知；within_task_pooled_probes.csv另存该统一分类器在各task内部的评分。</p>
<p>每点一次推理，相邻点相关，521个点来自30条轨迹，不是521次独立实验。阶段来自trace规则推断，少数标签支持不足。混合不同任务增加点数，也增加了指令／场景差异；分成大团不自动意味着分出了阶段。交互图可按阶段、簇、task着色，切换任务筛选、仅留出、Raw/L2、种子及2D/3D；筛选不重新拟合。</p>
'''.replace('__ROWS__', ''.join(rows))
    for name, title in [('joint_stages_raw', '全部十个task：两阶段与三阶段（Raw）'),
                        ('joint_stage_cluster_task_raw', '阶段、聚类与task颜色对照（Raw）'),
                        ('joint_stages_l2', '全部十个task：两阶段与三阶段（L2）'),
                        ('joint_stage_cluster_task_l2', '阶段、聚类与task颜色对照（L2）')]:
        body += f'<section class="plot"><h2>{title}</h2><p>半透明圆点：训练init0/1；实心三角：留出init2。每种预处理下，两／三阶段及task颜色使用完全相同的联合坐标。</p><img src="{(output/"figures"/(name+".png")).as_uri()}"></section>'
    (output/'report_zh.html').write_text(body+'</html>')


def build_explorer(output, catalog):
    sys.path.insert(0, str(Path(__file__).parent/'_deps'))
    from plotly.offline import get_plotlyjs
    records = []
    for r in catalog:
        with np.load(output/r['projection']) as a:
            records.append(dict(r, coordinates=a['coordinates'][:, :4].tolist(),
                phases={s['id']:a[f'phases_{s["id"]}'].tolist() for s in STAGES},
                aligned={s['id']:[a[f'aligned_{s["id"]}_seed{i}'].tolist() for i in (0,1,2)] for s in STAGES}))
    payload = json.dumps(dict(records=records, schemes=STAGES, task_colors=TASK_COLORS,
                              samples=json.loads((output/'samples.json').read_text())), ensure_ascii=False).replace('</', '<\\/')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>十任务联合阶段分布</title><style>body{font:15px system-ui;margin:22px;background:#f8fafc;color:#203047}select{padding:6px;margin:6px}.panels{display:grid;grid-template-columns:repeat(3,1fr)}.plot{height:620px}p{line-height:1.65;max-width:1400px}#metrics{padding:12px;background:#eaf0f8;margin:12px 0}@media(max-width:1100px){.panels{grid-template-columns:1fr}}</style>
<h1>十个 Goal task 放在一起：两／三阶段</h1><p>共521个已标阶段的点：347训练圆点，174留出三角（3D为菱形）。全部task共享训练集PCA。左阶段、中无监督簇、右task颜色，三个面板点位一致；簇与阶段的颜色只用训练标签匹配。</p>
<label>阶段<select id="scheme"></select></label><label>特征<select id="mode"><option value="raw">Raw</option><option value="l2">L2</option></select></label><label>种子<select id="seed"><option>0</option><option>1</option><option>2</option></select></label><label>坐标<select id="dims"><option value="0,1">PC1 / PC2</option><option value="0,2">PC1 / PC3</option><option value="1,2">PC2 / PC3</option><option value="0,1,2">PC1 / PC2 / PC3</option></select></label><label>显示<select id="subset"><option value="all">全部</option><option value="test">仅留出</option></select></label><label>任务筛选<select id="task"><option value="all">全部十个task</option></select></label>
<div id="metrics"></div><div class="panels"><div id="truth" class="plot"></div><div id="cluster" class="plot"></div><div id="identity" class="plot"></div></div><p>筛选只隐藏点，不重新拟合。阶段未知的28个点未参加拟合／评分。任务身份与阶段组成有关；分成团不自动意味着阶段可分。验证是已见任务的新轨迹，并非新任务泛化。</p>
<script>__PLOTLY__</script><script>const p=__DATA__,$=id=>document.getElementById(id);p.schemes.forEach(s=>$('scheme').add(new Option(s.zh,s.id)));for(let t=0;t<10;t++)$('task').add(new Option('Goal task'+t,String(t)));
function draw(){const r=p.records.find(r=>r.preprocessing===$('mode').value),s=p.schemes.find(s=>s.id===$('scheme').value),y=r.phases[s.id],seed=Number($('seed').value),dims=$('dims').value.split(',').map(Number),samples=p.samples;const ids=samples.map((v,i)=>i).filter(i=>y[i]>=0&&($('subset').value!=='test'||samples[i].init===2)&&($('task').value==='all'||String(samples[i].task)===$('task').value));const axis=a=>'PC'+(a+1)+' ('+(100*r.variance[a]).toFixed(1)+'%)',ranges=dims.map(a=>{const v=ids.map(i=>r.coordinates[i][a]);if(!v.length)return[-1,1];const lo=Math.min(...v),hi=Math.max(...v),pad=(hi-lo||1)*.08;return[lo-pad,hi+pad]});
function panel(id,labels,names,colors,title){const traces=[];for(let c=0;c<names.length;c++)for(const training of [true,false]){const use=ids.filter(i=>labels[i]===c&&(samples[i].init<2)===training);if(!use.length)continue;const t={type:dims.length===3?'scatter3d':'scattergl',mode:'markers',name:names[c]+(training?' train':' held-out'),x:use.map(i=>r.coordinates[i][dims[0]]),y:use.map(i=>r.coordinates[i][dims[1]]),text:use.map(i=>'task'+samples[i].task+' init'+samples[i].init+' request'+samples[i].request+'<br>'+samples[i].phase+'<br>'+samples[i].instruction),hovertemplate:'%{text}<extra></extra>',marker:{color:colors[c],size:training?5:8,opacity:training?.35:.95,symbol:dims.length===3?(training?'circle':'diamond'):(training?'circle':'triangle-up')}};if(dims.length===3)t.z=use.map(i=>r.coordinates[i][dims[2]]);traces.push(t)}Plotly.react(id,traces,{title:{text:title},xaxis:{title:{text:axis(dims[0])},range:ranges[0]},yaxis:{title:{text:axis(dims[1])},range:ranges[1]},scene:{xaxis:{title:{text:axis(dims[0])},range:ranges[0]},yaxis:{title:{text:axis(dims[1])},range:ranges[1]},zaxis:{title:{text:axis(dims[2]||2)},range:ranges[2]}},legend:{orientation:'h'},margin:{t:60,l:60,b:100,r:15}},{responsive:true})}
panel('truth',y,s.names,s.colors,'阶段颜色：'+s.zh);panel('cluster',r.aligned[s.id][seed],s.names,s.colors,'无监督k-means：训练标签匹配颜色');panel('identity',samples.map(v=>v.task),Array.from({length:10},(_,t)=>'task'+t),p.task_colors,'相同点位：task颜色');const m=r.metrics.find(m=>m.scheme===s.id&&m.seed===seed),pct=v=>(100*v).toFixed(1)+'%';$('metrics').textContent='显示 '+ids.length+' 点 · 全体留出集指标（不随筛选改变）：统一原维度阶段 BA '+pct(m.pooled_linear_ba)+' · 前2PC阶段 BA '+pct(m.pc12_linear_ba)+' · 仅task基线 '+pct(m.task_only_baseline_ba)+' · 聚类阶段ARI '+m.phase_ari.toFixed(3)+' · 聚类阶段BA '+pct(m.kmeans_stage_ba)+' · 聚类task ARI '+m.task_ari.toFixed(3);}
['scheme','mode','seed','dims','subset','task'].forEach(id=>$(id).onchange=draw);$('scheme').value='two';draw();</script></html>'''
    (output/'joint_phase_explorer.html').write_text(page.replace('__PLOTLY__', get_plotlyjs()).replace('__DATA__', payload))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--render-only', action='store_true')
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    for name in ('projections', 'figures'): (output/name).mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        if not args.render_only: compute(args.root.resolve(), output)
        render(output)
    print(json.dumps(dict(passed=True, output=str(output))), flush=True)


if __name__ == '__main__':
    main()
