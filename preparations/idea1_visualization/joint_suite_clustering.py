"""Joint k-means/PCA of actual pooled hidden states from all four suites."""
from __future__ import annotations

import argparse
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
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_samples
from threadpoolctl import threadpool_limits

from .analyze import fit_projection,project
from .common import SUITES,atomic_json,episode_path

NAMES=['Spatial','Object','Goal','LIBERO-10']
FEATURES={
    'vision/base_0_rgb/hidden':dict(name='Base camera Vision',layers=[0,3,6,9,12,18,24,27],dimension=1152),
    'vision/left_wrist_0_rgb/hidden':dict(name='Wrist camera Vision',layers=[0,3,6,9,12,18,24,27],dimension=1152),
    'vlm/vision/hidden':dict(name='VLM visual tokens',layers=[0,3,6,9,12,18],dimension=2048),
    'vlm/language/hidden':dict(name='VLM language tokens',layers=[0,3,6,9,12,18],dimension=2048),
}
KS=(2,4,8)
SEEDS=(0,1,2)
SUITE_COLORS=['#1f77b4','#ff7f0e','#2ca02c','#9467bd']
CLUSTER_COLORS=['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#FF6692','#B6E880']


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def inventory(root):
    samples=[]
    for s,suite in enumerate(SUITES):
        for task in range(10):
            for init in range(3):
                episode=episode_path(root,suite,task,init)
                identity=json.loads((episode/'identity.json').read_text())
                for row in json.loads((episode/'selection.json').read_text())['samples']:
                    directory=root/'features'/suite/f'task{task:02d}'/f'init{init:02d}'/f"request_{row['request']:04d}"
                    samples.append(dict(suite=s,suite_name=suite,task=task,init=init,request=row['request'],
                        phase=row['phase'],sources=row['sources'],step=row['step'],
                        instruction=identity['instruction'],feature_directory=str(directory)))
    assert len(samples)==2326
    assert len({(x['suite'],x['task'],x['init']) for x in samples})==120
    return samples


def prepare_vectors(x,mode):
    value=np.asarray(x,np.float64)
    if mode=='raw':return value
    if mode=='l2':return value/np.maximum(np.linalg.norm(value,axis=1,keepdims=True),1e-12)
    raise ValueError(mode)


def fit_joint_projection(x,train):
    pca,active,tolerance=fit_projection(x[train])
    return pca,active,tolerance,project(pca,active,x)


def fit_partition(z,train,k,seed):
    model=KMeans(n_clusters=k,n_init=5,random_state=seed)
    model.fit(z[train])
    return model.predict(z).astype(np.uint8),model.cluster_centers_,len(np.unique(model.labels_))


def composition(labels,suites,k):
    counts=np.zeros((k,4),int)
    np.add.at(counts,(labels,suites),1)
    by_suite=np.divide(counts,counts.sum(0,keepdims=True),out=np.zeros_like(counts,dtype=float),where=counts.sum(0,keepdims=True)>0)
    by_cluster=np.divide(counts,counts.sum(1,keepdims=True),out=np.zeros_like(counts,dtype=float),where=counts.sum(1,keepdims=True)>0)
    return counts,by_suite,by_cluster


def compute(root,output):
    started=time.monotonic();samples=inventory(root);atomic_json(output/'samples.json',samples)
    suite=np.array([s['suite'] for s in samples]);train=np.array([s['init']<2 for s in samples]);test=~train
    arrays={key:np.empty((len(samples),len(spec['layers']),spec['dimension']),np.float32) for key,spec in FEATURES.items()}
    for i,sample in enumerate(samples):
        directory=Path(sample['feature_directory'])
        assert json.loads((directory/'complete.json').read_text())['passed']
        with np.load(directory/'pooled.npz',allow_pickle=False) as archive:
            for key,spec in FEATURES.items():arrays[key][i]=archive[key][spec['layers']]
        if (i+1)%200==0:
            progress=dict(stage='load_features',completed=i+1,total=len(samples),elapsed_s=time.monotonic()-started)
            atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    catalog=[];metrics=[];compositions=[]
    for key,spec in FEATURES.items():
        for index,layer in enumerate(spec['layers']):
            for mode in ('raw','l2'):
                slug=key.replace('/','_')+f'_H{layer:02d}_{mode}'
                x=prepare_vectors(arrays[key][:,index],mode)
                assert np.isfinite(x).all()
                pca,active,tolerance,z=fit_joint_projection(x,train)
                predictions=np.empty((len(KS),len(SEEDS),len(samples)),np.uint8)
                payload=dict(coordinates=z,components=pca.components_,mean=pca.mean_,active=active,
                             explained_variance_ratio=pca.explained_variance_ratio_)
                row_metrics=[]
                for ki,k in enumerate(KS):
                    for si,seed in enumerate(SEEDS):
                        labels,centers,fitted=fit_partition(z,train,k,seed)
                        predictions[ki,si]=labels;payload[f'centers_k{k}_seed{seed}']=centers
                        record=dict(feature=key,layer=layer,preprocessing=mode,k=k,seed=seed,
                            n_train=int(train.sum()),n_test=int(test.sum()),fitted_clusters=fitted,
                            suite_ari=float(adjusted_rand_score(suite[test],labels[test])),
                            suite_nmi=float(normalized_mutual_info_score(suite[test],labels[test])),
                            pca_dimensions=z.shape[1],non_null_components=int(active.sum()),
                            retained_variance=float(pca.explained_variance_ratio_.sum()))
                        row_metrics.append(record);metrics.append(record)
                        if seed==0:
                            counts,by_suite,by_cluster=composition(labels[test],suite[test],k)
                            for c,s in np.ndindex(counts.shape):
                                compositions.append(dict(feature=key,layer=layer,preprocessing=mode,k=k,seed=0,
                                    cluster=c,suite=SUITES[s],test_count=int(counts[c,s]),
                                    fraction_within_suite=float(by_suite[c,s]),fraction_within_cluster=float(by_cluster[c,s])))
                payload['cluster_labels']=predictions
                np.savez(output/'projections'/f'{slug}.npz',**payload)
                silhouettes=silhouette_samples(z[test],suite[test],metric='euclidean')
                catalog.append(dict(feature=key,module=spec['name'],layer=layer,preprocessing=mode,slug=slug,
                    projection=f'projections/{slug}.npz',variance=pca.explained_variance_ratio_[:4].tolist(),
                    retained_variance=float(pca.explained_variance_ratio_.sum()),
                    non_null_components=int(active.sum()),numerical_rank_tolerance=tolerance,
                    silhouette_by_suite=[float(silhouettes[suite[test]==s].mean()) for s in range(4)],
                    seed_stability={str(k):[float(adjusted_rand_score(predictions[ki,0,test],predictions[ki,si,test])) for si in (1,2)] for ki,k in enumerate(KS)},
                    metrics=row_metrics))
                progress=dict(stage='joint_pca_kmeans',completed=len(catalog),total=56,last=slug,elapsed_s=time.monotonic()-started)
                atomic_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    atomic_json(output/'catalog.json',catalog)
    write_csv(output/'clustering_metrics.csv',metrics);write_csv(output/'cluster_suite_composition.csv',compositions)
    atomic_json(output/'compute_complete.json',dict(passed=True,samples=len(samples),train=int(train.sum()),test=int(test.sum()),
        configurations=len(catalog),clustering_fits=len(metrics),elapsed_s=time.monotonic()-started,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def scatter(ax,z,labels,colors,names):
    for i,name in enumerate(names):
        mask=labels==i
        ax.scatter(z[mask,0],z[mask,1],s=5,alpha=.42,color=colors[i],edgecolors='none',label=name,rasterized=True)


def render_static(output,catalog,samples):
    lookup={(r['feature'],r['layer'],r['preprocessing']):r for r in catalog}
    suite=np.array([s['suite'] for s in samples]);paths=[]
    for feature,spec in FEATURES.items():
        layers=(0,9,spec['layers'][-1])
        fig,axes=plt.subplots(3,4,figsize=(15,11))
        for row,layer in enumerate(layers):
            config=lookup[feature,layer,'raw']
            with np.load(output/config['projection']) as archive:
                z=archive['coordinates'];pred=archive['cluster_labels'][:,0]
            scatter(axes[row,0],z,suite,SUITE_COLORS,NAMES)
            axes[row,0].set_title(f'H{layer} | color = suite',fontsize=10)
            for col,k in enumerate(KS,1):
                scatter(axes[row,col],z,pred[col-1],CLUSTER_COLORS,[f'C{i}' for i in range(k)])
                metric=next(r for r in config['metrics'] if r['k']==k and r['seed']==0)
                axes[row,col].set_title(f'k={k} | color = cluster | held-out ARI {metric["suite_ari"]:.2f}',fontsize=9)
            for ax in axes[row]:
                ax.set_xlabel(f'PC1 ({config["variance"][0]:.1%})',fontsize=8)
                ax.set_ylabel(f'PC2 ({config["variance"][1]:.1%})',fontsize=8)
                ax.tick_params(labelsize=7)
        handles=[Line2D([],[],marker='o',linestyle='',color=c,label=n) for c,n in zip(SUITE_COLORS,NAMES)]
        fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.5,.965),fontsize=10)
        cluster_handles=[Line2D([],[],marker='o',linestyle='',color=CLUSTER_COLORS[i],label=f'cluster C{i}') for i in range(8)]
        fig.legend(handles=cluster_handles,loc='lower center',ncol=8,bbox_to_anchor=(.5,.025),fontsize=8)
        fig.suptitle(spec['name']+' | all four suites jointly fitted | raw pooled features',fontsize=14,y=.995)
        fig.text(.5,.012,'Same coordinates within each row. K-means in up to 64 PCA dimensions; PCA/centers fit on init0/1. All 2326 selected observations shown. Cluster IDs are local to each fit.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.075,1,.94));path=output/'figures'/f'{feature.replace("/","_")}_joint_scatter.png'
        fig.savefig(path,dpi=155);plt.close(fig);paths.append(path)
    fig,axes=plt.subplots(4,4,figsize=(15,13))
    for row,(feature,spec) in enumerate(FEATURES.items()):
        layer=spec['layers'][-1]
        for mi,mode in enumerate(('raw','l2')):
            config=lookup[feature,layer,mode]
            with np.load(output/config['projection']) as archive:
                z=archive['coordinates'];pred=archive['cluster_labels'][1,0]
            scatter(axes[row,mi*2],z,suite,SUITE_COLORS,NAMES)
            scatter(axes[row,mi*2+1],z,pred,CLUSTER_COLORS,[f'C{i}' for i in range(4)])
            for col in (mi*2,mi*2+1):
                axes[row,col].set_title(f'{spec["name"]} H{layer} | {mode} | '+('suite' if col%2==0 else 'k=4'),fontsize=8)
                axes[row,col].set_xlabel(f'PC1 ({config["variance"][0]:.1%})',fontsize=7)
                axes[row,col].set_ylabel(f'PC2 ({config["variance"][1]:.1%})',fontsize=7)
                axes[row,col].tick_params(labelsize=6)
    fig.suptitle('Final hidden states: raw versus per-observation L2 normalization | all suites together',fontsize=13)
    handles=[Line2D([],[],marker='o',linestyle='',color=c,label='suite '+n) for c,n in zip(SUITE_COLORS,NAMES)]
    handles.extend(Line2D([],[],marker='o',linestyle='',color=CLUSTER_COLORS[i],label=f'cluster C{i}') for i in range(4))
    fig.legend(handles=handles,loc='lower center',ncol=8,bbox_to_anchor=(.5,.01),fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.97));path=output/'figures/raw_vs_l2_final.png';fig.savefig(path,dpi=155);plt.close(fig);paths.append(path)
    return paths


def build_explorer(output,catalog,samples):
    sys.path.insert(0,str(Path(__file__).parent/'_deps'))
    from plotly.offline import get_plotlyjs
    records=[]
    for row in catalog:
        with np.load(output/row['projection']) as archive:
            records.append(dict(row,coordinates=archive['coordinates'][:,:4].tolist(),
                clusters={str(k):{str(seed):archive['cluster_labels'][ki,si].tolist() for si,seed in enumerate(SEEDS)} for ki,k in enumerate(KS)}))
    data=json.dumps(dict(samples=samples,records=records,suite_names=NAMES,suite_colors=SUITE_COLORS,cluster_colors=CLUSTER_COLORS),ensure_ascii=False).replace('</','<\/')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四套件联合散点与 k-means</title>
<style>body{font:15px system-ui;margin:22px;background:#f8fafc;color:#203047}h1{font-size:25px}.controls{display:flex;gap:12px;flex-wrap:wrap}select{padding:6px;margin:5px}p{max-width:1300px;line-height:1.6}.pair{display:grid;grid-template-columns:1fr 1fr}.plot{height:560px;background:white}#composition{height:350px}#metrics{margin:15px 0;padding:12px;background:#eaf0f8}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>
<h1>四个 suite 的真实 hidden state：联合 PCA 与 k-means</h1>
<p>左右是同一批点、同一套坐标：左图按 suite 着色，右图按 k-means 簇着色。四个 suite 一起拟合 PCA 和聚类，suite 标签仅用于着色与事后统计。切换 k 值时坐标不变。聚类使用至多 64 维 PCA，散点展示其中两维或三维。</p>
<div class="controls"><label>模块<select id="feature"></select></label><label>层<select id="layer"></select></label><label>预处理<select id="mode"><option value="raw">原始 pooled 特征</option><option value="l2">逐观测 L2 归一化</option></select></label><label>k<select id="k"><option>2</option><option selected>4</option><option>8</option></select></label><label>种子<select id="seed"><option>0</option><option>1</option><option>2</option></select></label><label>坐标<select id="dims"><option value="0,1">PC1 / PC2</option><option value="0,2">PC1 / PC3</option><option value="1,2">PC2 / PC3</option><option value="2,3">PC3 / PC4</option><option value="0,1,2">PC1 / PC2 / PC3</option></select></label><label>显示样本<select id="subset"><option value="all">全部所选点</option><option value="test">留出 init2</option><option value="train">训练 init0/1</option><option value="uniform">原始均匀点</option></select></label></div>
<div id="metrics"></div><div class="pair"><div id="suiteplot" class="plot"></div><div id="clusterplot" class="plot"></div></div><div id="composition"></div>
<p>聚类与 PCA 始终在 init0/1 拟合；筛选样本只改变显示，不重新拟合。簇编号不跨层、k、种子或预处理对齐。不同层的 PCA 坐标轴也不对齐。suite 分簇说明这些观测的特征组织方式，不能单独判断动作重要性。</p>
<script>__PLOTLY__</script><script>const payload=__DATA__;const $=id=>document.getElementById(id);const records=payload.records,samples=payload.samples;
const featureNames=[...new Set(records.map(r=>r.feature))];featureNames.forEach(f=>$('feature').add(new Option(records.find(r=>r.feature===f).module,f)));
function layers(){const old=$('layer').value;$('layer').innerHTML='';records.filter(r=>r.feature===$('feature').value&&r.preprocessing==='raw').forEach(r=>$('layer').add(new Option('H'+r.layer,String(r.layer))));if([...$('layer').options].some(o=>o.value===old))$('layer').value=old;else $('layer').value='9';}
function draw(){const d=records.find(r=>r.feature===$('feature').value&&String(r.layer)===$('layer').value&&r.preprocessing===$('mode').value);if(!d)return;
const k=Number($('k').value),seed=Number($('seed').value),cluster=d.clusters[k][seed],dims=$('dims').value.split(',').map(Number),subset=$('subset').value;
const ids=samples.map((s,i)=>i).filter(i=>subset==='all'||(subset==='test'&&samples[i].init===2)||(subset==='train'&&samples[i].init<2)||(subset==='uniform'&&samples[i].sources.includes('uniform')));
const axis=a=>'PC'+(a+1)+' ('+(d.variance[a]*100).toFixed(1)+'%)';
const hover=i=>payload.suite_names[samples[i].suite]+' task '+samples[i].task+' init '+samples[i].init+' request '+samples[i].request+'<br>'+samples[i].instruction+'<br>phase: '+samples[i].phase+'<br>cluster: C'+cluster[i];
function panel(target,labels,names,colors,title){const traces=names.map((name,label)=>{const chosen=ids.filter(i=>labels[i]===label);const t={type:dims.length===3?'scatter3d':'scattergl',mode:'markers',name:name,x:chosen.map(i=>d.coordinates[i][dims[0]]),y:chosen.map(i=>d.coordinates[i][dims[1]]),text:chosen.map(hover),hovertemplate:'%{text}<extra>%{fullData.name}</extra>',marker:{size:dims.length===3?3:5,opacity:.55,color:colors[label]}};if(dims.length===3)t.z=chosen.map(i=>d.coordinates[i][dims[2]]);return t;});
const range=a=>{const v=ids.map(i=>d.coordinates[i][a]),lo=Math.min(...v),hi=Math.max(...v),pad=(hi-lo||1)*.05;return[lo-pad,hi+pad]};
Plotly.react(target,traces,{title:{text:title},xaxis:{title:{text:axis(dims[0])},range:range(dims[0])},yaxis:{title:{text:axis(dims[1])},range:range(dims[1])},scene:{xaxis:{title:{text:axis(dims[0])}},yaxis:{title:{text:axis(dims[1])}},zaxis:{title:{text:axis(dims[2]||2)}}},margin:{t:65,l:70,b:65,r:20},legend:{orientation:'h'},uirevision:d.slug+$('dims').value+subset},{responsive:true});}
panel('suiteplot',samples.map(s=>s.suite),payload.suite_names,payload.suite_colors,d.module+' H'+d.layer+' · suite');panel('clusterplot',cluster,Array.from({length:k},(_,i)=>'C'+i),payload.cluster_colors,'k-means k='+k+' seed='+seed+' · 簇');
const counts=Array.from({length:k},()=>Array(4).fill(0));ids.forEach(i=>counts[cluster[i]][samples[i].suite]++);const totals=Array.from({length:4},(_,s)=>ids.filter(i=>samples[i].suite===s).length);const percentages=counts.map(row=>row.map((n,s)=>totals[s]?n/totals[s]*100:0));
Plotly.react('composition',[{type:'heatmap',z:percentages,x:payload.suite_names,y:Array.from({length:k},(_,i)=>'C'+i),zmin:0,zmax:100,colorscale:'Blues',customdata:counts,text:percentages.map(row=>row.map(n=>n.toFixed(1)+'%')),texttemplate:'%{text}',hovertemplate:'%{x}, %{y}<br>样本数 %{customdata}<br>占该 suite 的 %{z:.1f}%<extra></extra>'}],{title:{text:'各 suite 的点分到了哪些簇（每列合计 100%，随显示筛选更新）'},margin:{t:55,b:45,l:65,r:30}},{responsive:true});
const metric=d.metrics.find(r=>r.k===k&&r.seed===seed);$('metrics').textContent='显示 '+ids.length+' 个点 · PCA/聚类训练 '+metric.n_train+' 点，留出 '+metric.n_test+' 点 · 留出 suite ARI '+metric.suite_ari.toFixed(3)+'，NMI '+metric.suite_nmi.toFixed(3)+' · 前两 PC 方差 '+((d.variance[0]+d.variance[1])*100).toFixed(1)+'% · 聚类 PCA 保留方差 '+(d.retained_variance*100).toFixed(1)+'%';}
$('feature').onchange=()=>{layers();draw()};['layer','mode','k','seed','dims','subset'].forEach(id=>$(id).onchange=draw);layers();draw();</script></html>'''
    (output/'joint_suite_explorer.html').write_text(template.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',data))


def report(output,catalog,samples,paths):
    lookup={(r['feature'],r['layer'],r['preprocessing']):r for r in catalog}
    rows=[]
    for feature,spec in FEATURES.items():
        values=[]
        for layer in (0,9,spec['layers'][-1]):
            values.append(next(m['suite_ari'] for m in lookup[feature,layer,'raw']['metrics'] if m['k']==4 and m['seed']==0))
        l2=next(m['suite_ari'] for m in lookup[feature,spec['layers'][-1],'l2']['metrics'] if m['k']==4 and m['seed']==0)
        rows.append('<tr><td>'+html.escape(spec['name'])+'</td>'+''.join(f'<td>{value:.3f}</td>' for value in values+[l2])+'</tr>')
    source='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四个 suite 联合散点与 k-means</title><style>
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}
@page{size:A4;margin:17mm;@bottom-center{content:counter(page);font-size:8pt}}
@page plot{size:A3 landscape;margin:15mm}body{font:10pt/1.65 "DejaVu Sans",CJK,sans-serif;color:#193047}h1{font-size:21pt}h2{font-size:14pt;bookmark-level:1}table{border-collapse:collapse;width:100%;font-size:8.5pt}th,td{padding:2mm;border-bottom:1px solid #dbe3ee}th{background:#edf3fa;text-align:left}.plot{page:plot;break-before:page;break-after:page}.plot h2{margin:0 0 3mm}.plot p{font-size:8.5pt;margin:0 0 2mm}.plot img{display:block;max-width:388mm;max-height:237mm;margin:auto}</style>
<h1>四个 suite 的真实特征：联合散点与 k-means</h1>
<p>本报告将四个 suite 的真实 pooled hidden state 放在同一空间：2326 个观测，120 条轨迹。PCA 与 k-means 在 init0/1 的 1556 点拟合，init2 的 770 点留出。suite 标签不参与拟合，只用于着色和事后评价。</p>
<h2>如何看图</h2><p>每个模块一页：三行分别为输入、中层、末层；每行第一张按 suite 着色，后面三张分别按 k=2、4、8 的簇着色。同一行使用完全相同的点和坐标，因此能直接看到各 suite 是集中在某个簇，还是混合分布。</p>
<p>聚类在至多 64 维 PCA 坐标中进行，散点显示 PC1–PC2。图中包含全部所选点；图题 ARI 仅用留出点计算。k-means n_init=5，静态图使用 seed0；交互 HTML 还可切换 seed1/2、全部已检查层、二维／三维、训练／留出样本和 L2 归一化。</p>
<h2>k=4 与 suite 标签的一致性（留出 ARI）</h2>
<table><tr><th>模块</th><th>Raw H0</th><th>Raw H9</th><th>Raw 末层</th><th>L2 末层</th></tr>__ROWS__</table>
<p>最后一页比较原始特征和逐观测 L2 归一化，帮助检查幅值差异是否主导分簇。不同层和预处理分别拟合 PCA，坐标轴不跨图对齐；簇编号也不跨层、k 或种子对齐。suite 分簇可能来自任务、指令、场景或操作状态差异，不能单独证明某层对动作更重要。</p>
<p>所有原始图、交互页面、每簇的 suite 组成及三种种子的 ARI／NMI 数值均保存在同目录。AE 本轮未分析。</p>'''.replace('__ROWS__',''.join(rows))
    for i,path in enumerate(paths):
        title=list(FEATURES.values())[i]['name'] if i<4 else '末层：原始特征与 L2 归一化对照'
        source+=f'<section class="plot"><h2>{html.escape(title)}</h2><p>四个 suite 联合拟合；每行中 suite 着色图和聚类着色图使用同一批点、同一套坐标。</p><img src="{path.as_uri()}"></section>'
    source+='</html>';(output/'report_zh.html').write_text(source)


def render(output):
    catalog=json.loads((output/'catalog.json').read_text());samples=json.loads((output/'samples.json').read_text())
    paths=render_static(output,catalog,samples);build_explorer(output,catalog,samples);report(output,catalog,samples,paths)
    atomic_json(output/'render_complete.json',dict(passed=True,configurations=len(catalog),figures=len(paths),samples=len(samples),interactive=True,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    atomic_json(output/'progress.json',dict(stage='complete',status='complete',configurations=len(catalog),samples=len(samples)))
    print(json.dumps(dict(stage='complete',configurations=len(catalog),samples=len(samples),figures=len(paths))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'projections').mkdir(exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        if not args.render_only:compute(args.root.resolve(),output)
        render(output)


if __name__=='__main__':main()
