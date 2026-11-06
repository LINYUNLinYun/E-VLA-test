"""Episode-held-out probes, PCA views, high-dimensional clustering and residuals."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import ROOT,SUITES,atomic_json,episode_path,read_jsonl

PHASES=['initial','approach','contact','grasp_or_manipulate','transport','goal_alignment',
        'release_or_final_action','success','unknown']
VLM_LAYERS=[0,3,6,9,12,18]
VISION_PLOT_LAYERS=[0,3,6,9,12,18,24,27]


def write_csv(path,rows):
    if not rows:
        return
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def classifier(x,labels,train,test):
    classes=np.unique(labels[train])
    test_classes=np.unique(labels[test])
    if len(classes)<2 or test.sum()==0:
        return dict(accuracy=None,balanced_accuracy=None,n_train=int(train.sum()),n_test=int(test.sum()),
                    train_classes=len(classes),test_classes=len(test_classes),test_unseen_class_fraction=None,
                    majority_accuracy=None,majority_balanced_accuracy=None)
    fit=make_pipeline(StandardScaler(),RidgeClassifier(alpha=10.,solver='lsqr',tol=1e-3))
    fit.fit(x[train],labels[train]);pred=fit.predict(x[test])
    values,counts=np.unique(labels[train],return_counts=True)
    majority=np.repeat(values[int(np.argmax(counts))],int(test.sum()))
    return dict(accuracy=float(accuracy_score(labels[test],pred)),
                balanced_accuracy=float(balanced_accuracy_score(labels[test],pred)),
                n_train=int(train.sum()),n_test=int(test.sum()),train_classes=len(classes),test_classes=len(test_classes),
                test_unseen_class_fraction=float(np.mean(~np.isin(labels[test],classes))),
                majority_accuracy=float(accuracy_score(labels[test],majority)),
                majority_balanced_accuracy=float(balanced_accuracy_score(labels[test],majority)))


def conditional_probes(x,tasks,phase,train,test,*,feature,layer,suite):
    """Keep the same episode split while conditioning on task or inferred phase."""
    result=[]
    for task in range(10):
        chosen=(tasks==task)&(phase!='unknown')
        result.append(dict(feature=feature,layer=layer,suite=suite,target='phase',
            condition='within_task',condition_value=str(task),
            **classifier(x,phase,train&chosen,test&chosen)))
    for label in PHASES:
        if label=='unknown':
            continue
        chosen=phase==label
        result.append(dict(feature=feature,layer=layer,suite=suite,target='task',
            condition='within_phase',condition_value=label,
            **classifier(x,tasks,train&chosen,test&chosen)))
    return result


def scatter(ax,z,labels,phase=False):
    names=PHASES if phase else list(range(10))
    colors=plt.get_cmap('tab10')
    for index,label in enumerate(names):
        chosen=labels==label
        if chosen.any() or phase:
            ax.scatter(z[chosen,0],z[chosen,1],s=9,alpha=.55,color=colors(index%10),label=str(label))


def fit_projection(train_x,maximum_dimensions=64):
    """Fit training-only float64 PCA and identify numerically undefined axes.

    The matrix-rank tolerance is machine precision times matrix size, rather
    than a percentage-variance cutoff. Legitimate low-variance PCs are retained.
    Raw-dimensional probes never use this projection or this rank mask.
    """
    train_x=np.asarray(train_x,np.float64)
    n=min(maximum_dimensions,train_x.shape[0]-1,train_x.shape[1])
    pca=PCA(n_components=n,svd_solver='randomized',random_state=7)
    pca.fit(train_x)
    tolerance=max(train_x.shape)*np.finfo(np.float64).eps*pca.singular_values_[0]
    active=pca.singular_values_>tolerance
    return pca,active,float(tolerance)


def project(pca,active,x):
    result=pca.transform(np.asarray(x,np.float64))
    # A zero-variance training PC has no defined orientation. Zeroing only these
    # coordinates avoids magnifying roundoff or arbitrary null-space directions.
    result[:,~active]=0.
    return result


def load(root):
    samples=[];episodes=[];data={};residual={}
    for suite in SUITES:
        for task in range(10):
            for init in range(3):
                directory=episode_path(root,suite,task,init)
                identity=json.loads((directory/'identity.json').read_text())
                episodes.append(json.loads((directory/'result.json').read_text()))
                selected=json.loads((directory/'selection.json').read_text())['samples']
                for item in selected:
                    feature=root/'features'/suite/f'task{task:02d}'/f'init{init:02d}'/f"request_{item['request']:04d}"
                    if not json.loads((feature/'complete.json').read_text()).get('passed'):
                        raise ValueError(f'Unverified feature: {feature}')
                    samples.append(dict(item,suite=suite,task=task,init=init,feature=str(feature),
                                        instruction=identity['instruction'],
                                        service_group=f"gpu{identity['service']['cuda_visible_devices']}_pid{identity['service']['pid']}"))
                    with np.load(feature/'pooled.npz',allow_pickle=False) as arrays:
                        for key in arrays.files:
                            relevant=(key.startswith('vision/') and key.endswith('/hidden') or
                                      key.startswith(('vlm/vision/','vlm/language/')))
                            if relevant:
                                data.setdefault(key,[]).append(arrays[key].copy())
                    with np.load(feature/'residual_stats.npz',allow_pickle=False) as arrays:
                        for key in arrays.files:
                            if key.startswith(('vlm/vision/','vlm/language/')):
                                residual.setdefault(key,[]).append(arrays[key].copy())
    for key,arrays in data.items():
        if len(arrays)!=len(samples):
            raise ValueError(f'Feature missing from some samples: {key}')
    return samples,episodes,{k:np.stack(v) for k,v in data.items()},{k:np.stack(v) for k,v in residual.items()}


def analyze(root):
    output=root/'analysis';output.mkdir(exist_ok=True)
    figures=output/'figures';figures.mkdir(exist_ok=True)
    samples,episodes,data,residual=load(root)
    suite=np.asarray([s['suite'] for s in samples]);tasks=np.asarray([s['task'] for s in samples])
    phase=np.asarray([s['phase'] for s in samples]);init=np.asarray([s['init'] for s in samples])
    service=np.asarray([s['service_group'] for s in samples])
    uniform=np.asarray(['uniform' in s['sources'] for s in samples])
    probes=[];conditional=[];backend_controls=[];clusters=[];projections=[];images=[]
    for key,x_all in data.items():
        vision=key.startswith('vision/')
        kind=key.rsplit('/',1)[1]
        layers=(list(range(x_all.shape[1])) if vision else
                VLM_LAYERS if kind=='hidden' else [l for l in VLM_LAYERS if l>0])
        plot_layers=VISION_PLOT_LAYERS if vision else (VLM_LAYERS if kind=='hidden' else [3,9,18])
        for layer in layers:
            index=layer if kind=='hidden' else layer-1
            x=x_all[:,index,:]
            should_plot=layer in plot_layers
            if should_plot:
                fig,axes=plt.subplots(4,2,figsize=(11,17))
            for si,name in enumerate(SUITES):
                use=suite==name;train=use&(init<2);test=use&(init==2)
                for target,labels,known in (('task',tasks,np.ones(len(samples),bool)),('phase',phase,phase!='unknown')):
                    score=classifier(x,labels,train&known,test&known)
                    probes.append(dict(feature=key,layer=layer,suite=name,target=target,subset='all_selected',**score))
                    score=classifier(x,labels,train&known&uniform,test&known&uniform)
                    probes.append(dict(feature=key,layer=layer,suite=name,target=target,subset='uniform12',**score))
                if kind=='hidden' and layer in plot_layers:
                    conditional.extend(conditional_probes(x,tasks,phase,train,test,
                        feature=key,layer=layer,suite=name))
                for group in np.unique(service[use]):
                    chosen=service==group
                    backend_controls.append(dict(feature=key,layer=layer,suite=name,
                        service_group=group,target='task',
                        **classifier(x,tasks,train&chosen,test&chosen)))
                if not should_plot:
                    continue
                pca,active,rank_tolerance=fit_projection(x[train])
                n=len(active)
                z=project(pca,active,x[use]);zt=project(pca,active,x[test])
                ztrain=project(pca,active,x[train])
                distinct_train_inputs=len(np.unique(x[train],axis=0))
                captured=float(pca.explained_variance_ratio_.sum())
                title=f'{name} | PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}'
                scatter(axes[si,0],z,tasks[use]);scatter(axes[si,1],z,phase[use],True)
                for ax in axes[si]:
                    ax.set_title(title,fontsize=9);ax.set_xlabel('PC1');ax.set_ylabel('PC2')
                if si==0:
                    axes[si,0].legend(fontsize=6,ncol=5);axes[si,1].legend(fontsize=6,ncol=2)
                rows=[samples[i] for i in np.where(use)[0]]
                projection=dict(feature=key,layer=layer,suite=name,
                    non_null_retained_components=int(active.sum()),numerical_rank_tolerance=rank_tolerance,
                    variance=pca.explained_variance_ratio_[:4].tolist(),coordinates=z[:,:4].tolist(),
                    task=tasks[use].tolist(),phase=phase[use].tolist(),
                    episode=[f"{s['suite']}/{s['task']}/{s['init']}" for s in rows],
                    request=[s['request'] for s in rows],sources=[s['sources'] for s in rows],
                    phase_reason=[s['phase_reason'] for s in rows],
                    active_objects=[s['phase_evidence'].get('active_objects',[]) for s in rows],
                    task_instructions={str(s['task']):s['instruction'] for s in rows})
                projections.append(projection)
                for k in (5,10,15):
                    for seed in (0,1,2):
                        condition=dict(feature=key,layer=layer,suite=name,k=k,seed=seed,
                            pca_dimensions=n,pca_explained_variance=captured,n_train=int(train.sum()),n_test=int(test.sum()),
                            non_null_retained_components=int(active.sum()),distinct_train_inputs=distinct_train_inputs,
                            phase_test_samples=int(np.sum(phase[test]!='unknown')),
                            phase_test_classes=len(np.unique(phase[test][phase[test]!='unknown'])))
                        if distinct_train_inputs<k:
                            clusters.append(dict(condition,status='not_estimable:k_exceeds_distinct_training_inputs',
                                fitted_clusters=None,task_ari=None,task_nmi=None,phase_ari=None,phase_nmi=None))
                            continue
                        km=KMeans(n_clusters=k,n_init=5,random_state=seed)
                        km.fit(ztrain);pred=km.predict(zt)
                        if k==10 and seed==0:
                            projection.update(cluster=km.predict(z).tolist(),cluster_spec=dict(
                                k=k,seed=seed,fitted_clusters=len(np.unique(km.labels_)),
                                fit='init0/1, high-dimensional PCA space'))
                        known=phase[test]!='unknown'
                        clusters.append(dict(condition,status='estimated',fitted_clusters=len(np.unique(km.labels_)),
                            task_ari=float(adjusted_rand_score(tasks[test],pred)),
                            task_nmi=float(normalized_mutual_info_score(tasks[test],pred)),
                            phase_ari=float(adjusted_rand_score(phase[test][known],pred[known])) if known.sum()>1 else None,
                            phase_nmi=float(normalized_mutual_info_score(phase[test][known],pred[known])) if known.sum()>1 else None))
            if should_plot:
                fig.suptitle(f'{key}, H/block {layer} | PCA fit on init0/1; all selected points shown',fontsize=11)
                fig.tight_layout(rect=(0,0,1,.98))
                name=f"pca_{key.replace('/','_')}_layer{layer:02d}.png"
                fig.savefig(figures/name,dpi=130);plt.close(fig);images.append(name)
            print(json.dumps(dict(feature=key,layer=layer,probes=len(probes))),flush=True)
    write_csv(output/'linear_probes.csv',probes);write_csv(output/'conditional_probes.csv',conditional)
    write_csv(output/'backend_control_probes.csv',backend_controls)
    write_csv(output/'clustering.csv',clusters)
    atomic_json(output/'pca_projections.json',projections)
    residual_figures=[]
    for key,values in residual.items():
        fig,axes=plt.subplots(1,4,figsize=(19,5),sharey=True)
        for ax,name in zip(axes,SUITES):
            matrix=[]
            for label in PHASES:
                chosen=(suite==name)&(phase==label)
                matrix.append(values[chosen].mean(axis=0) if chosen.any() else np.full(values.shape[1],np.nan))
            shown=ax.imshow(np.asarray(matrix),aspect='auto',interpolation='nearest')
            ax.set_title(name);ax.set_xticks(range(0,18,3),range(1,19,3));ax.set_xlabel('Block (1-based)')
            ax.set_yticks(range(len(PHASES)),PHASES);fig.colorbar(shown,ax=ax,shrink=.6)
        fig.suptitle(key+' | descriptive sample mean, inferred phase labels');fig.tight_layout()
        name=f"residual_{key.replace('/','_')}.png";fig.savefig(figures/name,dpi=130);plt.close(fig)
        residual_figures.append(name)
    curve_figures=[]
    for key in data:
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,target in zip(axes,('task','phase')):
            for name in SUITES:
                rows=[r for r in probes if r['feature']==key and r['suite']==name and r['target']==target and r['subset']=='all_selected']
                ax.plot([r['layer'] for r in rows],[r['balanced_accuracy'] if r['balanced_accuracy'] is not None else np.nan for r in rows],'-o',label=name,markersize=3)
            ax.set_title(target+' | init2 held out');ax.set_xlabel('H/block index');ax.set_ylabel('Balanced accuracy');ax.set_ylim(0,1.03);ax.legend(fontsize=7)
        fig.suptitle(key);fig.tight_layout()
        name=f"probe_{key.replace('/','_')}.png";fig.savefig(figures/name,dpi=130);plt.close(fig);curve_figures.append(name)
    conditional_figures=[]
    for key in data:
        if not key.endswith('/hidden'):
            continue
        rows=[r for r in conditional if r['feature']==key and r['condition']=='within_task']
        layers=sorted({r['layer'] for r in rows});matrix=np.full((40,len(layers)),np.nan)
        support={}
        for row in rows:
            index=SUITES.index(row['suite'])*10+int(row['condition_value'])
            support[index]=(row['n_test'],row['test_classes'])
            if row['balanced_accuracy'] is not None and row['test_classes']>=2:
                matrix[index,layers.index(row['layer'])]=row['balanced_accuracy']-row['majority_balanced_accuracy']
        cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#333333')
        fig,ax=plt.subplots(figsize=(10,15))
        image=ax.imshow(matrix,aspect='auto',cmap=cmap,vmin=-1,vmax=1)
        names=[f'{s}/task{t:02d} (n={support.get(si*10+t,(0,0))[0]}, K={support.get(si*10+t,(0,0))[1]})'
               for si,s in enumerate(SUITES) for t in range(10)]
        ax.set_yticks(range(40),names,fontsize=7);ax.set_xticks(range(len(layers)),[f'H{l}' for l in layers])
        for border in (9.5,19.5,29.5):ax.axhline(border,color='black',linewidth=.6)
        ax.set_xlabel('Hidden-state index');ax.set_title(key+'\nWithin-task phase: held-out balanced accuracy minus training-majority baseline',fontsize=10)
        fig.colorbar(image,ax=ax,label='Score difference; pale grey = baseline, dark = insufficient coverage',shrink=.6)
        fig.tight_layout();name=f"conditional_{key.replace('/','_')}.png"
        fig.savefig(figures/name,dpi=130);plt.close(fig);conditional_figures.append(name)
    counts=Counter(phase.tolist())
    summary=dict(episodes=len(episodes),samples=len(samples),successes=sum(e['success'] for e in episodes),
        per_suite={s:dict(episodes=sum(e['suite']==s for e in episodes),successes=sum(e['success'] for e in episodes if e['suite']==s),
                          samples=int(np.sum(suite==s))) for s in SUITES},
        phase_counts=dict(counts),uniform12_samples=int(uniform.sum()),
        phase_counts_by_suite={name:dict(Counter(phase[suite==name].tolist())) for name in SUITES},
        phase_counts_by_split={name:{split:dict(Counter(phase[(suite==name)&mask].tolist()))
            for split,mask in (('train_init01',init<2),('test_init2',init==2))} for name in SUITES},
        split='Each task: init0/1 train; init2 held out; scalers/PCA fit on train only',
        probe='Fixed RidgeClassifier alpha10 on original pooled dimensions; no model fine-tuning',
        conditional_probe='Selected hidden layers: infer phase within each task and task within each phase, same held-out init2; majority baseline and class coverage reported',
        backend_control='Task is assigned to one persistent model service; additionally fit/evaluate task probes within each service (five tasks per suite), preserving episode split. This is not a cross-service generalization test.',
        uniform_probe='Refit the diagnostic on original uniform12 train samples and score original uniform12 held-out samples; train and test budgets both differ from all-selected results',
        clustering='KMeans in up to 64 PCA dimensions; k5/10/15 x seed0/1/2; fit train, score held-out init2',
        clustering_degeneracy='Requested k above the number of distinct original training vectors is recorded as not estimable; actual fitted cluster count also reported',
        pca_numerics='Fit/transform in float64; zero only numerically undefined training-null coordinates at standard machine-epsilon matrix-rank tolerance; original-dimensional probes unaffected',
        caveats=['Trace phases are inferred labels, not ground truth; phase scores measure consistency with these rules.',
                 'Selected official pi05_libero disables discrete state input; language pool contains instruction prompt tokens.',
                 'Task identity may reflect prompt, scene, object and state differences.',
                 'Task IDs are correlated with persistent service assignment; cross-service BF16 differences were observed. Within-service task probes are supplied as a control; broad semantic or backend-invariance claims are unsupported.',
                 'PCA and probes operate on the defined mean-pooled vectors, not the entire token tensor; poor readout does not establish absence of information in full token states.',
                 'Two-dimensional overlap does not establish absence of higher-dimensional information.',
                 'PCA axes and cluster IDs are fitted independently for each feature/layer/suite and are not aligned across plots.',
                 'Readability, residual magnitude and causal action importance are distinct.',
                 'This native sampler consumes per-layer prefix KV caches and discards the final prefix hidden output; H18 geometry is not a direct action-sensitivity measurement.',
                 'Three initializations per task are exploratory, not the 50-init benchmark.'])
    atomic_json(output/'summary.json',summary)
    build_explorer(output,projections)
    lines=['# π0.5 LIBERO 表征探索实验报告','',
           f"完成 {len(episodes)} 条轨迹、{len(samples)} 个所选真实观测。成功 {summary['successes']}/{len(episodes)}，该成绩只描述本轮采集轨迹。",'',
           '每 task 使用 init0/1 训练诊断分类器，init2 留出评价。分类器不修改策略模型。PCA／标准化仅使用训练轨迹拟合。','',
           '## 结果入口','',
           '- [交互 PCA：二维／三维、前四维配对、task／phase 着色](pca_explorer.html)',
           '- [逐层线性 probe 数值](linear_probes.csv)',
           '- [同 task 内 phase／同 phase 内 task 的条件对照](conditional_probes.csv)',
           '- [同一模型服务内的 5-task 读出对照](backend_control_probes.csv)',
           '- [高维聚类及 k／seed 敏感性](clustering.csv)',
           '- [样本规模、阶段分布及分析口径](summary.json)','',
           'phase 来自 trace 的事后规则推断，不是人工或仿真阶段真值。本轮官方 pi05_libero 配置关闭离散状态输入，语言 token 池为有效指令 prompt token。','',
           '本轮描述可读出信息和残差更新，未通过受控跳层干预检验动作重要性；二维不可分也不等于高维不可分。','',
           '结构口径：本轮原生采样器将 VLM prefill 的逐层 KV 传给 AE，并忽略 prefill 的最终 hidden 输出。因此，末层 H18 的几何变化不能直接解释为动作影响；这是代码依赖关系的边界，不是可视化得出的因果结论。','',
           'PCA 与 probe 针对预先定义的均值池化向量，不代表完整 token 张量的信息量。各特征／层／suite 分别拟合 PCA，坐标轴与 cluster 编号不跨图对齐。uniform12 对照会在均匀训练子集上重新拟合分类器，因此训练和留出样本量均与完整所选集不同。','',
           '本轮 task 与持续模型服务的分片绑定，跨服务曾观察到 BF16 数值差。因此另报同一服务内的 5-task 读出，不能把合并后的 10-task 读出直接当作跨后端语义泛化证明。同 task 内的 phase 对照也保持在同一服务中。','',
           '## 所选观测的阶段支持量','',
           '| 推断阶段 | 训练 init0/1 | 留出 init2 | 合计 |',
           '| --- | ---: | ---: | ---: |',
           *[f"| {label} | {int(np.sum((phase==label)&(init<2)))} | {int(np.sum((phase==label)&(init==2)))} | {counts.get(label,0)} |" for label in PHASES],'',
           '表内比例描述所选观测，不是 rollout 的自然阶段时长比例。unknown 不参与 phase probe；样本很少的类别需结合留出类别覆盖读取，规则标签的可标注比例不代表正确率。','',
           '## 逐层读出曲线','']
    for name in curve_figures:
        lines.extend([f'![{name}](figures/{name})',''])
    lines.extend(['## 同 task 内 phase 的层间差异','',
                  '每行仅一条 init2 留出轨迹；n 为有效留出观测数，K 为留出阶段数。颜色是相对训练多数类基线的诊断得分差，不是动作重要性；不同 task 的阶段组成不同。浅灰色表示接近基线，深灰色表示不足两个训练或留出阶段，需结合 conditional_probes.csv 的支持量读取。',''])
    for name in conditional_figures:
        lines.extend([f'![{name}](figures/{name})',''])
    lines.extend(['## 残差更新与阶段',''])
    for name in residual_figures:
        lines.extend([f'![{name}](figures/{name})',''])
    lines.extend(['## PCA 静态图',''])
    for name in images:
        lines.append(f'- [{name}](figures/{name})')
    (output/'report_zh.md').write_text('\n'.join(lines)+'\n')
    atomic_json(output/'analysis_complete.json',dict(passed=True,episodes=len(episodes),samples=len(samples),
                probes=len(probes),conditional_probes=len(conditional),backend_control_probes=len(backend_controls),cluster_conditions=len(clusters),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def build_explorer(output,projections):
    sys.path.insert(0,str(Path(__file__).parent/'_deps'))
    from plotly.offline import get_plotlyjs
    payload=json.dumps(projections,ensure_ascii=False).replace('</','<\/')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>π0.5 表征探索</title>
<style>body{font:15px system-ui;margin:24px;background:#f8fafc;color:#172033}select{margin:8px;padding:6px}#plot{height:78vh;background:white}p{max-width:1050px}</style>
<h1>π0.5 LIBERO 表征探索</h1><p>PCA 和参考聚类均在 init0/1 拟合，图中包含全部所选观测。各特征／层／suite 单独拟合，坐标轴与 cluster 编号不跨图对齐。聚类使用高维 PCA 空间。phase 为 trace 推断；事件补点改变了样本阶段比例。二维重叠不代表高维不可分。</p>
<label>特征与层<select id="feature"></select></label><label>Suite<select id="suite"></select></label>
<label>着色<select id="color"><option value="task">task</option><option value="phase">phase</option><option value="cluster">参考聚类</option></select></label>
<label>维度<select id="dims"><option value="0,1">PC1 / PC2</option><option value="0,2">PC1 / PC3</option><option value="0,3">PC1 / PC4</option><option value="1,2">PC2 / PC3</option><option value="1,3">PC2 / PC4</option><option value="2,3">PC3 / PC4</option><option value="0,1,2">PC1 / PC2 / PC3</option></select></label>
<label>样本<select id="subset"><option value="all">全部所选点</option><option value="uniform">最初 12 个均匀点</option></select></label><div id="plot"></div>
<script>__PLOTLY__</script><script>const data=__DATA__;const el=id=>document.getElementById(id);
const names=[...new Set(data.map(d=>d.feature+' | '+d.layer))];names.forEach(n=>el('feature').add(new Option(n,n)));
[...new Set(data.map(d=>d.suite))].forEach(n=>el('suite').add(new Option(n,n)));
function draw(){const d=data.find(d=>d.feature+' | '+d.layer===el('feature').value&&d.suite===el('suite').value);if(!d)return;
const dims=el('dims').value.split(',').map(Number),labels=d[el('color').value],isPhase=el('color').value==='phase',phaseOrder=['initial','approach','contact','grasp_or_manipulate','transport','goal_alignment','release_or_final_action','success','unknown'];
if(!labels){Plotly.react('plot',[],{title:{text:'当前特征没有可估计的参考聚类；原因见 clustering.csv'}},{responsive:true});return;}
const palette=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
const levels=isPhase?phaseOrder.filter(x=>labels.includes(x)):[...new Set(labels)].sort(),traces=[];
for(const level of levels){const ids=labels.map((x,i)=>x===level?i:-1).filter(i=>i>=0&&(el('subset').value==='all'||d.sources[i].includes('uniform')));
const hover=i=>d.episode[i]+' request '+d.request[i]+'<br>'+(d.task_instructions?d.task_instructions[String(d.task[i])]:'')+'<br>phase: '+d.phase[i]+'<br>'+(d.phase_reason?d.phase_reason[i]:'')+'<br>active: '+(d.active_objects?d.active_objects[i].join(', '):'')+'<br>'+d.sources[i].join(', ');
const t={type:dims.length===3?'scatter3d':'scattergl',mode:'markers',name:String(level),x:ids.map(i=>d.coordinates[i][dims[0]]),y:ids.map(i=>d.coordinates[i][dims[1]]),text:ids.map(hover),marker:{size:dims.length===3?3:5,opacity:.65,color:palette[isPhase?phaseOrder.indexOf(level):Number(level)]}};
if(dims.length===3)t.z=ids.map(i=>d.coordinates[i][dims[2]]);traces.push(t);}
const title=axis=>'PC'+(axis+1)+' ('+(100*d.variance[axis]).toFixed(1)+'%'+(Number.isInteger(d.non_null_retained_components)&&axis>=d.non_null_retained_components?', numerical null':'')+')';
const clusterTitle=el('color').value==='cluster'&&d.cluster_spec?' | k'+d.cluster_spec.k+' seed'+d.cluster_spec.seed:'';
Plotly.react('plot',traces,{title:{text:d.feature+' | layer '+d.layer+' | '+d.suite+clusterTitle},xaxis:{title:{text:title(dims[0])}},yaxis:{title:{text:title(dims[1])}},scene:{xaxis:{title:{text:title(dims[0])}},yaxis:{title:{text:title(dims[1])}},zaxis:{title:{text:title(dims[2]||2)}}},margin:{t:60},uirevision:el('feature').value+el('suite').value+el('dims').value},{responsive:true});}
document.querySelectorAll('select').forEach(e=>e.onchange=draw);draw();</script></html>'''
    (output/'pca_explorer.html').write_text(page.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    with threadpool_limits(limits=2):
        analyze(Path(args.root))


if __name__=='__main__':
    main()
