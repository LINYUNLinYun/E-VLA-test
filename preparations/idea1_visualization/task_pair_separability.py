"""Direct task-pair scatter, unsupervised k=2 and episode-held-out readout."""
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
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import adjusted_rand_score,balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .analyze import write_csv
from .common import atomic_json
from .joint_suite_clustering import (FEATURES,fit_joint_projection,fit_partition,
                                     inventory,prepare_vectors)

PAIRS=[
    dict(id='spatial0_vs_spatial5',name='Spatial task0 vs Spatial task5',tasks=[(0,0),(0,5)],requested=True),
    dict(id='spatial0_vs_goal0',name='Spatial task0 vs Goal task0',tasks=[(0,0),(2,0)],requested=True),
    dict(id='spatial0_vs_spatial4_control',name='Spatial task0 vs Spatial task4 (same-service control)',tasks=[(0,0),(0,4)],requested=False),
]
COLORS=['#2878b5','#ed8b23']


def linear_predictions(x,y,train):
    model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=10.,solver='lsqr',tol=1e-3))
    model.fit(x[train],y[train])
    return model.predict(x)


def align_binary_clusters(clusters,y,train):
    identity=float(balanced_accuracy_score(y[train],clusters[train]))
    reverse=float(balanced_accuracy_score(y[train],1-clusters[train]))
    flip=reverse>identity
    return (1-clusters if flip else clusters),bool(flip)


def phase_controls(x,y,train,phases):
    result=[]
    for phase in ('initial','approach'):
        use=phases==phase;fit=train&use;test=(~train)&use
        row=dict(phase=phase,n_train=int(fit.sum()),n_test=int(test.sum()),balanced_accuracy=None,
                 train_classes=len(np.unique(y[fit])),test_classes=len(np.unique(y[test])))
        if row['train_classes']==row['test_classes']==2:
            predicted=linear_predictions(x,y,fit)
            row['balanced_accuracy']=float(balanced_accuracy_score(y[test],predicted[test]))
        result.append(row)
    return result


def compute(root,output):
    all_samples=inventory(root)
    wanted={task for pair in PAIRS for task in pair['tasks']}
    samples=[s for s in all_samples if (s['suite'],s['task']) in wanted]
    data={feature:np.empty((len(samples),len(spec['layers']),spec['dimension']),np.float32) for feature,spec in FEATURES.items()}
    for i,sample in enumerate(samples):
        with np.load(Path(sample['feature_directory'])/'pooled.npz',allow_pickle=False) as archive:
            for feature,spec in FEATURES.items():data[feature][i]=archive[feature][spec['layers']]
    catalog=[];metrics=[];phase_rows=[]
    for pair in PAIRS:
        selected=[i for i,s in enumerate(samples) if (s['suite'],s['task']) in pair['tasks']]
        pair_samples=[samples[i] for i in selected]
        y=np.array([pair['tasks'].index((s['suite'],s['task'])) for s in pair_samples])
        train=np.array([s['init']<2 for s in pair_samples]);test=~train
        phases=np.array([s['phase'] for s in pair_samples])
        assert len({(s['suite'],s['task'],s['init']) for s in pair_samples})==6
        atomic_json(output/f'{pair["id"]}_samples.json',pair_samples)
        for feature,spec in FEATURES.items():
            for li,layer in enumerate(spec['layers']):
                for mode in ('raw','l2'):
                    x=prepare_vectors(data[feature][selected,li],mode)
                    pca,active,tolerance,z=fit_joint_projection(x,train)
                    linear=linear_predictions(x,y,train)
                    linear2d=linear_predictions(z[:,:2],y,train)
                    labels=[];mapped=[];seed_metrics=[];centers=[]
                    for seed in (0,1,2):
                        cluster,center,fitted=fit_partition(z,train,2,seed)
                        aligned,flip=align_binary_clusters(cluster,y,train)
                        labels.append(cluster);mapped.append(aligned);centers.append(center)
                        row=dict(pair=pair['id'],feature=feature,layer=layer,preprocessing=mode,seed=seed,
                            n_train=int(train.sum()),n_test=int(test.sum()),
                            test_a=int(np.sum(test&(y==0))),test_b=int(np.sum(test&(y==1))),
                            kmeans_balanced_accuracy=float(balanced_accuracy_score(y[test],aligned[test])),
                            kmeans_ari=float(adjusted_rand_score(y[test],cluster[test])),
                            linear_balanced_accuracy=float(balanced_accuracy_score(y[test],linear[test])),
                            linear_2d_balanced_accuracy=float(balanced_accuracy_score(y[test],linear2d[test])),
                            kmeans_recall_a=float(np.mean(aligned[test&(y==0)]==0)),
                            kmeans_recall_b=float(np.mean(aligned[test&(y==1)]==1)),
                            linear_correct_a=int(np.sum(linear[test&(y==0)]==0)),
                            linear_correct_b=int(np.sum(linear[test&(y==1)]==1)),
                            mapping_flipped=flip,fitted_clusters=fitted,pca_dimensions=z.shape[1],
                            non_null_components=int(active.sum()))
                        metrics.append(row);seed_metrics.append(row)
                    controls=phase_controls(x,y,train,phases)
                    for control in controls:phase_rows.append(dict(pair=pair['id'],feature=feature,layer=layer,preprocessing=mode,**control))
                    slug=f'{pair["id"]}_{feature.replace("/","_")}_H{layer:02d}_{mode}'
                    np.savez(output/'projections'/f'{slug}.npz',coordinates=z,active=active,labels=y,train=train,
                        cluster_labels=np.array(labels),mapped_cluster_labels=np.array(mapped),centers=np.array(centers),
                        linear_predictions=linear,linear_2d_predictions=linear2d,
                        pca_components=pca.components_,pca_mean=pca.mean_,pca_variance=pca.explained_variance_ratio_)
                    catalog.append(dict(pair=pair['id'],pair_name=pair['name'],feature=feature,module=spec['name'],
                        layer=layer,preprocessing=mode,slug=slug,projection=f'projections/{slug}.npz',
                        variance=pca.explained_variance_ratio_[:4].tolist(),metrics=seed_metrics,phase_controls=controls))
        print(json.dumps(dict(pair=pair['id'],configurations=len(catalog)),ensure_ascii=False),flush=True)
    write_csv(output/'pair_metrics.csv',metrics);write_csv(output/'phase_control_probes.csv',phase_rows)
    atomic_json(output/'catalog.json',catalog)
    atomic_json(output/'compute_complete.json',dict(passed=True,pairs=3,configurations=len(catalog),cluster_fits=len(metrics),
        classifier='StandardScaler + RidgeClassifier alpha10, LSQR; original pooled dimensions',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def draw_points(ax,z,labels,train):
    for label in (0,1):
        for training in (True,False):
            use=(labels==label)&(train==training)
            ax.scatter(z[use,0],z[use,1],c=COLORS[label],s=14 if training else 40,
                marker='o' if training else '^',alpha=.45 if training else .95,
                edgecolors='none' if training else '#283345',linewidths=.35)


def render(output):
    catalog=json.loads((output/'catalog.json').read_text());paths=[]
    lookup={(r['pair'],r['feature'],r['layer'],r['preprocessing']):r for r in catalog}
    for pair in PAIRS:
        fig,axes=plt.subplots(4,4,figsize=(16,13))
        for row,layer in enumerate((0,3,12,18)):
            for mi,feature in enumerate(('vlm/vision/hidden','vlm/language/hidden')):
                record=lookup[pair['id'],feature,layer,'raw']
                with np.load(output/record['projection']) as archive:
                    z=archive['coordinates'];truth=archive['labels'];train=archive['train'];mapped=archive['mapped_cluster_labels'][0]
                score=record['metrics'][0]
                draw_points(axes[row,mi*2],z,truth,train);draw_points(axes[row,mi*2+1],z,mapped,train)
                module='Visual tokens' if mi==0 else 'Language tokens'
                axes[row,mi*2].set_title(f'{module} H{layer}: task colors\noriginal-space linear BA {score["linear_balanced_accuracy"]:.1%}',fontsize=9)
                axes[row,mi*2+1].set_title(f'H{layer}: k-means k=2, train-mapped colors\nheld-out BA {score["kmeans_balanced_accuracy"]:.1%}',fontsize=9)
                for col in (mi*2,mi*2+1):
                    ax=axes[row,col];ax.set_xlabel(f'PC1 ({record["variance"][0]:.1%})',fontsize=7)
                    ax.set_ylabel(f'PC2 ({record["variance"][1]:.1%})',fontsize=7);ax.tick_params(labelsize=7)
        handles=[Line2D([],[],marker='o',linestyle='',color=COLORS[i],label=f'{"A" if i==0 else "B"}: '+('Spatial' if task[0]==0 else 'Goal')+f' task{task[1]}') for i,task in enumerate(pair['tasks'])]
        handles.extend([Line2D([],[],marker='o',linestyle='',color='#777777',label='Train init0/1'),Line2D([],[],marker='^',linestyle='',color='#777777',label='Held-out init2')])
        fig.legend(handles=handles,loc='upper center',ncol=4,bbox_to_anchor=(.5,.965),fontsize=10)
        fig.suptitle(pair['name']+' | actual pooled hidden states | raw features',fontsize=14,y=.995)
        fig.text(.5,.012,'Each task has 2 training episodes + 1 held-out episode. Pair PCA/k-means fit only on training. Cluster color mapping also uses training labels only.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.035,1,.94));path=output/'figures'/f'{pair["id"]}_vlm_scatter.png'
        fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)
    build_report(output,catalog,paths);build_explorer(output,catalog)
    atomic_json(output/'render_complete.json',dict(passed=True,figures=len(paths),interactive=True,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


def build_report(output,catalog,paths):
    table=[]
    for pair in PAIRS:
        for feature in ('vlm/vision/hidden','vlm/language/hidden'):
            for layer in (0,3,12,18):
                row=next(r for r in catalog if r['pair']==pair['id'] and r['feature']==feature and r['layer']==layer and r['preprocessing']=='raw')
                m=row['metrics'][0];k=[v['kmeans_balanced_accuracy'] for v in row['metrics']]
                table.append(f'<tr><td>{html.escape(pair["name"])}</td><td>{"视觉池" if "vision" in feature else "语言池"} H{layer}</td><td>{m["linear_balanced_accuracy"]:.1%}</td><td>{min(k):.1%}–{max(k):.1%}</td><td>{m["test_a"]}+{m["test_b"]}</td></tr>')
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实任务对的特征可分性</title><style>
@font-face{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}@page{size:A4;margin:16mm;@bottom-center{content:counter(page);font-size:8pt}}@page plot{size:A3 landscape;margin:14mm}body{font:9.5pt/1.6 "DejaVu Sans",CJK,sans-serif;color:#1e3349}h1{font-size:21pt}h2{font-size:14pt;bookmark-level:1}table{border-collapse:collapse;width:100%;font-size:7.5pt}th,td{padding:1.5mm;border-bottom:1px solid #dce4ef}th{background:#edf3f8;text-align:left}tr{break-inside:avoid}.plot{page:plot;break-before:page;break-after:page}.plot h2{margin:0 0 3mm}.plot p{font-size:8pt}.plot img{display:block;max-width:390mm;max-height:239mm;margin:auto}</style>
<h1>Spatial／Goal 具体 task 对：是否可分？</h1><p>直接使用每次观测的 pooled hidden state，而非 probe 分数曲线。检验用户指定的 Spatial task0 vs task5、Spatial task0 vs Goal task0，并补充 Spatial task0 vs task4 的同服务对照。</p>
<p>每个 task 的 init0/1 训练，init2 留出；每对仅有两条留出轨迹，不能把几十个相邻观测当成几十次独立实验。PCA、k-means、标准化和线性分类器均不使用留出数据拟合。k-means 的两簇与任务名称的对应关系仅根据训练标签确定。</p>
<p>表格区分两种证据：k-means 高，说明不看任务标签也容易自然分成两簇；线性读出高，说明任务信息可被简单分类器读取，即使二维散点重叠或自然簇不对应任务。聚类在至多64维的“这两个任务联合 PCA”中执行；二维图仅展示前两个方向。</p>
<p>Spatial task0／task5 原始采集服务不同，且本机此前观察到服务间 BF16 差异，不能单凭这一对把所有差异归因于任务。Spatial task0／Goal task0 以及补充的 Spatial task0／task4 绑定同一个服务。语言 H0 本就包含不同任务指令，可分不等于任务语义在深层才形成。</p>
<h2>固定 raw 特征的留出结果</h2><table><tr><th>任务对</th><th>位置</th><th>原始维度线性 BA</th><th>k-means BA：3个种子</th><th>留出点 A+B</th></tr>__TABLE__</table>
<p>BA 为 balanced accuracy，二分类多数类基线为50%。训练分类器固定为 StandardScaler + RidgeClassifier(alpha=10)。另保存了逐观测 L2 归一化、二维线性读出和相同 initial／approach phase 内的对照；不能将点数或高分解释为跨新任务泛化或动作因果重要性。</p>'''.replace('__TABLE__',''.join(table))
    document+='<p>静态图展示 H0／H3／H12／H18，以呈现此次观察到的分簇差异；全体已检查层均在交互图和 CSV 中保留。这些是探索性展示位置，不是通过独立验证确定的最优层。</p>'
    for pair,path in zip(PAIRS,paths):document+=f'<section class="plot"><h2>{html.escape(pair["name"])}</h2><p>蓝／橙表示两个任务；圆点为训练观测，三角形为留出观测。每个模块、每层独立拟合二任务共享 PCA，同行内左右图坐标一致。</p><img src="{path.as_uri()}"></section>'
    (output/'report_zh.html').write_text(document+'</html>')


def build_explorer(output,catalog):
    sys.path.insert(0,str(Path(__file__).parent/'_deps'))
    from plotly.offline import get_plotlyjs
    records=[]
    samples={pair['id']:json.loads((output/f'{pair["id"]}_samples.json').read_text()) for pair in PAIRS}
    for row in catalog:
        with np.load(output/row['projection']) as a:
            records.append(dict(row,coordinates=a['coordinates'][:,:4].tolist(),labels=a['labels'].tolist(),
                mapped=a['mapped_cluster_labels'].tolist(),linear=a['linear_predictions'].tolist()))
    payload=json.dumps(dict(records=records,pairs=PAIRS,samples=samples),ensure_ascii=False).replace('</','<\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>具体 task 对的特征散点</title><style>body{font:15px system-ui;margin:24px;background:#f8fafc;color:#203047}select{padding:6px;margin:6px}.pair{display:grid;grid-template-columns:1fr 1fr}.plot{height:580px}p{max-width:1300px;line-height:1.6}#metrics{padding:12px;background:#eaf0f8;margin:12px 0}@media(max-width:850px){.pair{grid-template-columns:1fr}}</style>
<h1>具体 task 对：真实特征能不能分开？</h1><p>左图是两个任务的实际标签，右图是无监督 k-means k=2 的结果。聚类颜色仅用训练标签对齐，未使用留出标签。圆点为训练 init0/1；留出 init2 在二维中用三角、三维中用菱形表示。每次推理一个点。</p>
<label>任务对<select id="pair"></select></label><label>模块<select id="feature"></select></label><label>层<select id="layer"></select></label><label>预处理<select id="mode"><option value="raw">Raw</option><option value="l2">L2</option></select></label><label>种子<select id="seed"><option>0</option><option>1</option><option>2</option></select></label><label>坐标<select id="dims"><option value="0,1">PC1 / PC2</option><option value="0,2">PC1 / PC3</option><option value="0,1,2">PC1 / PC2 / PC3</option></select></label><label>显示<select id="subset"><option value="all">全部</option><option value="test">仅留出</option><option value="initial">initial phase</option><option value="approach">approach phase</option></select></label>
<div id="metrics"></div><div class="pair"><div id="truth" class="plot"></div><div id="cluster" class="plot"></div></div>
<p>同一任务内的观测高度相关；每个 task 只有一条留出轨迹。仅改变显示筛选，不重新拟合。语言输入 H0 含指令差异；Spatial0／5 的原采集服务不同，另两对为同服务。所有层均可在下拉菜单选择。</p>
<script>__PLOTLY__</script><script>const payload=__DATA__,records=payload.records,$=id=>document.getElementById(id);
payload.pairs.forEach(p=>$('pair').add(new Option(p.name,p.id)));[...new Set(records.map(r=>r.feature))].forEach(f=>$('feature').add(new Option(records.find(r=>r.feature===f).module,f)));
function layers(){const old=$('layer').value;$('layer').innerHTML='';records.filter(r=>r.pair===$('pair').value&&r.feature===$('feature').value&&r.preprocessing==='raw').forEach(r=>$('layer').add(new Option('H'+r.layer,String(r.layer))));if([...$('layer').options].some(o=>o.value===old))$('layer').value=old;else $('layer').value='12';}
function draw(){const d=records.find(r=>r.pair===$('pair').value&&r.feature===$('feature').value&&String(r.layer)===$('layer').value&&r.preprocessing===$('mode').value);if(!d)return;const samples=payload.samples[d.pair],seed=Number($('seed').value),dims=$('dims').value.split(',').map(Number),subset=$('subset').value;
const ids=samples.map((s,i)=>i).filter(i=>subset==='all'||(subset==='test'&&samples[i].init===2)||samples[i].phase===subset),colors=['#2878b5','#ed8b23'];const pair=payload.pairs.find(p=>p.id===d.pair);const names=pair.tasks.map(t=>(t[0]===0?'Spatial':'Goal')+' task'+t[1]);const axis=a=>'PC'+(a+1)+' ('+(100*d.variance[a]).toFixed(1)+'%)';
function panel(target,labels,title){const traces=[];for(let label=0;label<2;label++)for(const training of [true,false]){const chosen=ids.filter(i=>labels[i]===label&&(samples[i].init<2)===training),t={type:dims.length===3?'scatter3d':'scattergl',mode:'markers',name:names[label]+(training?' train':' held-out'),x:chosen.map(i=>d.coordinates[i][dims[0]]),y:chosen.map(i=>d.coordinates[i][dims[1]]),text:chosen.map(i=>names[d.labels[i]]+' init '+samples[i].init+' request '+samples[i].request+'<br>'+samples[i].instruction+'<br>phase '+samples[i].phase),marker:{color:colors[label],size:training?5:9,opacity:training?.45:.95,symbol:dims.length===3?(training?'circle':'diamond'):(training?'circle':'triangle-up')}};if(dims.length===3)t.z=chosen.map(i=>d.coordinates[i][dims[2]]);traces.push(t);}const range=a=>{const values=ids.map(i=>d.coordinates[i][a]);if(!values.length)return[-1,1];const lo=Math.min(...values),hi=Math.max(...values),pad=(hi-lo||1)*.08;return[lo-pad,hi+pad]};Plotly.react(target,traces,{title:{text:title},xaxis:{title:{text:axis(dims[0])},range:range(dims[0])},yaxis:{title:{text:axis(dims[1])},range:range(dims[1])},scene:{xaxis:{title:{text:axis(dims[0])}},yaxis:{title:{text:axis(dims[1])}},zaxis:{title:{text:axis(dims[2]||2)}}},legend:{orientation:'h'},margin:{t:65,l:65,b:65,r:20}},{responsive:true});}
panel('truth',d.labels,d.module+' H'+d.layer+' | 实际 task');panel('cluster',d.mapped[seed],'k-means k=2 | 训练标签对齐颜色');const m=d.metrics[seed];$('metrics').textContent='显示 '+ids.length+' 点 · 训练 '+m.n_train+' 点，留出 '+m.test_a+'+'+m.test_b+' 点（每任务一条留出轨迹） · 原始维度线性 BA '+(100*m.linear_balanced_accuracy).toFixed(1)+'% · PC1/PC2 线性 BA '+(100*m.linear_2d_balanced_accuracy).toFixed(1)+'% · k-means 留出 BA '+(100*m.kmeans_balanced_accuracy).toFixed(1)+'%，ARI '+m.kmeans_ari.toFixed(3);}
['pair','feature'].forEach(id=>$(id).onchange=()=>{layers();draw()});['layer','mode','seed','dims','subset'].forEach(id=>$(id).onchange=draw);$('feature').value='vlm/language/hidden';layers();draw();</script></html>'''
    (output/'task_pair_explorer.html').write_text(page.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',payload))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true');args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    (output/'projections').mkdir(exist_ok=True);(output/'figures').mkdir(exist_ok=True)
    with threadpool_limits(limits=2):
        if not args.render_only:compute(args.root.resolve(),output)
        render(output)
    print(json.dumps(dict(passed=True,output=str(output))),flush=True)


if __name__=='__main__':main()
