"""Compare suite-specific module/depth readout profiles from frozen probe CSVs.

No model inference, classifier refitting, activation loading or policy ablation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram,fcluster,leaves_list,linkage
from scipy.spatial.distance import pdist,squareform

SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
NAMES=['Spatial','Object','Goal','LIBERO-10']
FEATURES=['vision/base_0_rgb/hidden','vision/left_wrist_0_rgb/hidden',
          'vlm/vision/hidden','vlm/language/hidden']
MODULES=['Vision: base camera','Vision: wrist camera','VLM: visual tokens','VLM: language tokens']
SHORT=['Base Vision','Wrist Vision','VLM visual','VLM language']
COLORS=['#2878b5','#ed8b23','#32965a','#be4c73']


def read_csv(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def baseline_gain(score,baseline):
    score,baseline=np.asarray(score,float),np.asarray(baseline,float)
    if np.any(baseline>=1):raise ValueError('Baseline leaves no score headroom')
    return (score-baseline)/(1-baseline)


def depth_projection(keys,uniform=False):
    """Equal module weight; integrate over each module's observed H>0 span."""
    projection=np.zeros((len(FEATURES),len(keys)))
    for m,feature in enumerate(FEATURES):
        indices=[i for i,(f,_) in enumerate(keys) if f==feature]
        x=np.asarray([keys[i][1] for i in indices],float)
        if len(x)<2 or np.any(np.diff(x)<=0):raise ValueError('Unordered/incomplete module')
        if uniform:weights=np.full(len(x),1/len(x))
        else:
            x=(x-x[0])/(x[-1]-x[0]);delta=np.diff(x)
            weights=np.zeros(len(x));weights[:-1]+=delta/2;weights[1:]+=delta/2
        projection[m,indices]=weights
    np.testing.assert_allclose(projection.sum(1),1)
    return projection


def decompose(gain,projection,module_index):
    """Decompose overall interaction into module and within-module depth terms."""
    auc=np.einsum('mr,...rs->...ms',projection,gain)
    module=(auc-auc.mean(-2,keepdims=True)-auc.mean(-1,keepdims=True)
            +auc.mean((-2,-1),keepdims=True))
    depth=gain-auc[...,module_index,:]
    depth=depth-depth.mean(-1,keepdims=True)
    overall=depth+module[...,module_index,:]
    return dict(auc=auc,module=module,depth=depth,overall=overall)


def suite_tree(gain,projection):
    auc=projection@gain
    interaction=gain-gain.mean(1,keepdims=True)-auc.mean(0,keepdims=True)+auc.mean()
    vectors=(interaction*np.sqrt(projection.mean(0))[:,None]).T
    if np.any(np.linalg.norm(vectors,axis=1)<1e-12):raise ValueError('Undefined zero suite fingerprint')
    distance=np.clip(pdist(vectors,'cosine'),0,2)
    tree=linkage(distance,method='average',optimal_ordering=True)
    return tree,squareform(distance)


def task_bootstrap(gain,draws=5000,seed=7):
    """Resample whole tasks, paired across all layers/modules within each suite."""
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,gain.shape[-1],size=(draws,gain.shape[1],gain.shape[-1]))
    result=np.empty((draws,gain.shape[0],gain.shape[1]))
    for s in range(gain.shape[1]):result[:,:,s]=np.take(gain[:,s,:],indices[:,s,:],axis=1).mean(-1).T
    return result


def simultaneous_intervals(observed,replicates):
    """One max-standardized-error family across module and depth cells."""
    names=('module','depth')
    center=np.concatenate([observed[n].ravel() for n in names])
    samples=np.concatenate([replicates[n].reshape(len(replicates[n]),-1) for n in names],axis=1)
    se=samples.std(0,ddof=1);active=se>1e-12
    maximum=np.max(np.abs((samples[:,active]-center[active])/se[active]),axis=1)
    critical=float(np.quantile(maximum,.95))
    result={};begin=0
    for name in names:
        size=observed[name].size;spread=se[begin:begin+size].reshape(observed[name].shape)*critical
        point=np.quantile(replicates[name],[.025,.975],axis=0)
        result[name]=dict(low=observed[name]-spread,high=observed[name]+spread,
                          point_low=point[0],point_high=point[1])
        begin+=size
    return result,critical


def load_scores(root):
    conditional=read_csv(root/'conditional_probes.csv')
    backend=read_csv(root/'backend_control_probes.csv')
    # H0 is an input control, excluded from the block-output preference analysis.
    keys=[(feature,layer) for feature in FEATURES for layer in sorted({int(r['layer'])
        for r in conditional if r['feature']==feature and r['condition']=='within_task'}) if layer>0]
    phase=np.empty((len(keys),4,10));phase_baseline=np.empty_like(phase)
    task=np.empty((len(keys),4,2));task_baseline=np.empty_like(task)
    supports=[]
    for i,(feature,layer) in enumerate(keys):
        for s,suite in enumerate(SUITES):
            rows=[r for r in conditional if r['feature']==feature and int(r['layer'])==layer
                  and r['suite']==suite and r['condition']=='within_task']
            assert len(rows)==10 and {int(r['condition_value']) for r in rows}==set(range(10))
            for row in rows:
                assert row['balanced_accuracy'] and int(row['train_classes'])>=2 and int(row['test_classes'])>=2
                t=int(row['condition_value']);phase[i,s,t]=float(row['balanced_accuracy'])
                phase_baseline[i,s,t]=float(row['majority_balanced_accuracy'])
                supports.append(dict(feature=feature,layer=layer,suite=suite,task=t,
                    n_train=int(row['n_train']),n_test=int(row['n_test']),
                    train_classes=int(row['train_classes']),test_classes=int(row['test_classes']),
                    score=phase[i,s,t],baseline=phase_baseline[i,s,t]))
            rows=sorted([r for r in backend if r['feature']==feature and int(r['layer'])==layer
                         and r['suite']==suite],key=lambda r:r['service_group'])
            assert len(rows)==2
            for k,row in enumerate(rows):
                assert int(row['train_classes'])==int(row['test_classes'])==5
                task[i,s,k]=float(row['balanced_accuracy']);task_baseline[i,s,k]=float(row['majority_balanced_accuracy'])
    assert len(keys)==24
    return keys,dict(phase=phase,task=task),dict(phase=phase_baseline,task=task_baseline),supports


def heatmap(ax,matrix,rows,columns,*,title,limit=None,absolute=False,interval=None):
    scale=100;shown=matrix*scale
    kwargs=dict(vmin=0,vmax=100,cmap='YlGnBu') if absolute else dict(vmin=-limit*scale,vmax=limit*scale,cmap='RdBu_r')
    im=ax.imshow(shown,aspect='auto',**kwargs)
    ax.set_xticks(range(len(columns)),columns,rotation=20,ha='right')
    ax.set_yticks(range(len(rows)),rows,fontsize=9)
    ax.set_title(title,fontsize=11,pad=12)
    for i,j in np.ndindex(matrix.shape):
        star=''
        if interval is not None and (interval['low'][i,j]>0 or interval['high'][i,j]<0):star='*'
        text=f'{shown[i,j]:.1f}{star}' if absolute else f'{shown[i,j]:+.1f}{star}'
        dark=shown[i,j]>65 if absolute else abs(matrix[i,j])>.65*limit
        ax.text(j,i,text,ha='center',va='center',fontsize=8,color='white' if dark else '#172333')
    return im


def figures(output,keys,raw,gains,projection,indices,stats,intervals,cluster_info):
    paths=[]
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for row,target in enumerate(('task','phase')):
        absolute=projection@raw[target].mean(-1)
        im=heatmap(axes[row,0],absolute,SHORT,NAMES,title=f'{target}: average readout over sampled block outputs (%)',absolute=True)
        fig.colorbar(im,ax=axes[row,0],shrink=.7)
        im=heatmap(axes[row,1],stats[target]['module'],SHORT,NAMES,
            title=f'{target}: relative module preference (normalized points)',limit=.25,
            interval=intervals['module'] if target=='phase' else None)
        fig.colorbar(im,ax=axes[row,1],shrink=.7)
    fig.suptitle('Module preference | equal module weight; fixed input controls H0 excluded',fontsize=14)
    fig.text(.5,.01,'Red: stronger comparative advantage for this suite. Blue: weaker. *: phase simultaneous bootstrap interval excludes zero.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.96));path=output/'module_suite_preferences.png';fig.savefig(path,dpi=160);plt.close(fig);paths.append(path)

    fig,axes=plt.subplots(4,2,figsize=(14,15))
    for m,feature in enumerate(FEATURES):
        ii=np.where(indices==m)[0];x=np.array([keys[i][1] for i in ii])
        for s,name in enumerate(NAMES):
            axes[m,0].plot(x,raw['phase'][ii,s].mean(-1)*100,'-o',label=name,color=COLORS[s],markersize=4)
            axes[m,1].plot(x,stats['phase']['depth'][ii,s]*100,'-o',label=name,color=COLORS[s],markersize=4)
        axes[m,0].set_title(MODULES[m]+' | phase readout')
        axes[m,0].set_ylabel('Within-task balanced accuracy (%)');axes[m,0].set_ylim(0,100)
        axes[m,1].set_title(MODULES[m]+' | relative depth preference')
        axes[m,1].axhline(0,color='#999999',linewidth=.8)
        axes[m,1].set_ylabel('Baseline-normalized preference points');axes[m,1].set_ylim(-18,18)
        for ax in axes[m]:ax.set_xticks(x);ax.set_xlabel('Native hidden-state index H');ax.grid(alpha=.18);ax.legend(fontsize=8,ncol=2)
    fig.suptitle('Phase: different suites favor different depths | relative depth removes whole-module suite offsets',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97));path=output/'phase_depth_profiles.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)

    fig,axes=plt.subplots(1,2,figsize=(15,13))
    row_clusters={}
    for ax,target in zip(axes,('task','phase')):
        depth=stats[target]['depth']
        tree=linkage(depth,method='average',metric='euclidean',optimal_ordering=True)
        order=leaves_list(tree);labels=fcluster(tree,4,criterion='maxclust')
        column_order=cluster_info[target]['order']
        selected=depth[np.ix_(order,column_order)]
        row_names=[f'C{labels[i]} | {SHORT[indices[i]]} H{keys[i][1]}' for i in order]
        im=heatmap(ax,selected,row_names,[NAMES[s] for s in column_order],
            title=target+': layer profile groups (up to 4 display groups)',limit=.18)
        for i in range(1,len(order)):
            if labels[order[i]]!=labels[order[i-1]]:ax.axhline(i-.5,color='black',linewidth=1)
        fig.colorbar(im,ax=ax,shrink=.65,label='Relative depth preference points')
        row_clusters[target]=[dict(feature=keys[i][0],layer=keys[i][1],cluster=int(labels[i])) for i in range(len(keys))]
    fig.suptitle('Clustering layer preference signatures | groups describe readout patterns, not causal importance',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.97));path=output/'layer_preference_clusters.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(path)

    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for row,target in enumerate(('task','phase')):
        info=cluster_info[target]
        dendrogram(np.array(info['tree']),labels=NAMES,ax=axes[row,0],color_threshold=0,above_threshold_color='#315b83')
        axes[row,0].set_title(target+': suite fingerprint hierarchy');axes[row,0].set_ylabel('Cosine distance / average linkage')
        if target=='task':
            matrix=np.array(info['distance']);im=axes[row,1].imshow(matrix,vmin=0,vmax=2,cmap='Blues')
            axes[row,1].set_xticks(range(4),NAMES,rotation=20,ha='right');axes[row,1].set_yticks(range(4),NAMES)
            axes[row,1].set_title('task: cosine distances (lower = more similar)')
            for i,j in np.ndindex(matrix.shape):axes[row,1].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>1.3 else 'black')
            fig.colorbar(im,ax=axes[row,1],shrink=.7)
        else:
            im=heatmap(axes[row,1],np.array(info['coassignment']),NAMES,NAMES,
                title='phase: same group in task bootstrap, fixed k=2 (%)',absolute=True)
            fig.colorbar(im,ax=axes[row,1],shrink=.7)
    fig.suptitle('Suite clustering uses complete module/depth fingerprints; only four suites, no claim of natural cluster count',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));path=output/'suite_fingerprint_clusters.png';fig.savefig(path,dpi=155);plt.close(fig);paths.append(path)
    return paths,row_clusters


def build_report(output,paths,summary,stats,intervals):
    task_pair=summary['clusters']['task']['first_pair'];phase_pair=summary['clusters']['phase']['first_pair']
    frequencies=summary['clusters']['phase']['first_pair_bootstrap_frequency']
    task_module_rows=''.join('<tr><td>'+html.escape(SHORT[m])+'</td>'+''.join(f'<td>{100*stats["task"]["module"][m,s]:+.1f}</td>' for s in range(4))+'</tr>' for m in range(4))
    module_rows=''.join('<tr><td>'+html.escape(SHORT[m])+'</td>'+''.join(f'<td>{100*stats["phase"]["module"][m,s]:+.1f}</td>' for s in range(4))+'</tr>' for m in range(4))
    candidates=[]
    for m,s in np.ndindex(stats['phase']['module'].shape):
        lo,hi=intervals['module']['low'][m,s],intervals['module']['high'][m,s]
        if lo>0 or hi<0:candidates.append((SHORT[m],NAMES[s],100*stats['phase']['module'][m,s],100*lo,100*hi))
    layer_examples=[]
    keys=summary['keys']
    for i,s in sorted(np.ndindex(stats['phase']['depth'].shape),key=lambda pair:abs(stats['phase']['depth'][pair]),reverse=True):
        lo,hi=intervals['depth']['low'][i,s],intervals['depth']['high'][i,s]
        if lo>0 or hi<0:
            layer_examples.append((keys[i]['label'],NAMES[s],100*stats['phase']['depth'][i,s],100*lo,100*hi))
    def candidates_table(rows):
        if not rows:return '<p>在本轮 task 重采样的同时区间口径下，没有方向足够稳定的单元；图中点估计仍可用于提出候选假设。</p>'
        return '<table><tr><th>模块／层</th><th>Suite</th><th>偏好点数</th><th>95% 同时区间</th></tr>'+''.join(f'<tr><td>{html.escape(a)}</td><td>{b}</td><td>{v:+.1f}</td><td>[{lo:+.1f}, {hi:+.1f}]</td></tr>' for a,b,v,lo,hi in rows)+'</table>'
    html_text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>四个 suite 的模块与层偏好</title>
<style>@font-face{{font-family:CJK;src:url(file:///usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf)}}
@page{{size:A4;margin:17mm;@bottom-center{{content:counter(page);font-size:8pt;color:#64748b}}}}
@page figure{{size:A3 portrait;margin:16mm}}body{{font:10pt/1.65 "DejaVu Sans",CJK,sans-serif;color:#183047}}h1{{font-size:22pt}}h2{{font-size:15pt;break-after:avoid;bookmark-level:1}}h3{{font-size:12pt;break-after:avoid}}table{{border-collapse:collapse;width:100%;font-size:9pt;margin:4mm 0}}th,td{{padding:2mm;border-bottom:1px solid #d5dfeb}}th{{background:#edf3f8;text-align:left}}tr{{break-inside:avoid}}p{{orphans:3;widows:3}}.figure{{page:figure;break-before:page;break-after:page}}.figure img{{display:block;max-width:261mm;max-height:350mm;margin:auto}}.small{{font-size:8.5pt;color:#53667b}}</style>
<h1>四个 suite 的模块与层偏好</h1><p>基于首轮 120 条轨迹的既有留出 probe 结果；无需重新运行模型。本补充研究的是<strong>表征可读出的相对偏好</strong>，没有测量屏蔽某层造成的动作或成功率变化。</p>
<h2>读图方法</h2><p>左侧原始分数回答“能读出多少”；红蓝偏好图回答“相对于该 suite 的整体水平、以及该模块或深度的一般表现，哪里更有比较优势”。红色为相对偏好，蓝色为相对劣势；蓝色不表示没有信息或可以删除。</p>
<p>模块偏好与层偏好分开计算：模块偏好比较四个模块的整段曲线；层偏好先扣除该模块在该 suite 的平均水平，再比较深浅层的相对变化。输入 H0 不参与聚类，避免指令输入直接区分 task 的平凡效应。</p>
<h2>四个 suite 如何分组</h2><p>task 指纹最先合并 <strong>{html.escape(task_pair)}</strong>；phase 指纹最先合并 <strong>{html.escape(phase_pair)}</strong>。二者聚类不同，说明“任务身份”和“操作阶段”的模块／层分布不是同一种倾向。</p>
<p>每个 suite 用四个模块沿已观察深度的读出曲线构成一个指纹。task 的首次合并在逐一去掉一个模块的四次检查、以及改用各观察层等权的检查中均保持 Object + LIBERO-10；这属于对分析选择的敏感性检查，尚不是对新增轨迹或 seed 的验证。</p>
<p>phase 的首次合并在 5000 次 task 重采样中：{'; '.join(f'{html.escape(k)} {v:.1%}' for k,v in sorted(frequencies.items(),key=lambda pair:pair[1],reverse=True))}。只有四个 suite，层级树用于描述相似性，不据此认定存在某个自然类别数。</p>
<h2>模块的 task 相对偏好</h2><p>Object 与 LIBERO-10 在主相机 Vision 的相对读出优势较大；Goal 在主相机方向为 {100*stats['task']['module'][0,2]:+.1f} 点，在 VLM 视觉池方向为 {100*stats['task']['module'][2,2]:+.1f} 点。该对比显示不同 suite 的模块比较优势不同；它没有测量策略对模块的因果依赖。</p>
<table><tr><th>模块</th>{''.join('<th>'+name+'</th>' for name in NAMES)}</tr>{task_module_rows}</table>
<h2>模块的 phase 相对偏好</h2><p>单位为基线归一化后的偏好点数，不是准确率差。正值表示比较优势；原始准确率同时保留在图中。</p>
<table><tr><th>模块</th>{''.join('<th>'+name+'</th>' for name in NAMES)}</tr>{module_rows}</table>
<h3>同时区间支持的模块方向</h3>{candidates_table(candidates)}
<h3>同时区间支持的层方向</h3>{candidates_table(layer_examples[:12])}
<h2>统计与解释边界</h2><p>phase：每个 task 内使用原固定分类器结果，init0/1 训练、init2 留出，十个 task 等权。先逐 task 计算 (BA−多数类基线)/(1−多数类基线)，再平均。task：使用同一持续模型服务内的 5-task 读出，两个服务等权；不是原报告的合并 10-task 分数。</p>
<p>模块曲线采用各自已观察 H&gt;0 深度区间上的梯形面积平均，每个模块等权；点间线性连接仅用于曲线摘要，不视为未测层的实际观测。Vision 与 VLM 使用实际层号，跨模块不把同号层当成同一深度。聚类使用基线归一化、扣除通用模块／深度与 suite 平均效应后的指纹；suite 采用加权余弦距离和 average linkage，层采用四维偏好向量的欧氏距离和 average linkage，最多四组仅用于展示。</p>
<p>phase 重采样以完整 task 为单位，每个 suite 内十个 task 有放回抽样，跨层／模块保持同一抽样索引；95% 同时区间覆盖全部 16 个模块单元和 96 个层单元的一个 family，使用最大标准化 bootstrap 偏差。它描述这些已观察 task 的异质性，不能代替更多 episode、seed 或跨服务重复实验。task 只有聚合分数，不虚构 task 级置信区间。</p>
<p>suite 的阶段类别数与样本组成不同，基线归一化和中心化只能控制其中一部分差异。所有 phase 是 trace 推断标签；模型服务与 task 分片仍有关联。即使排除 H0，语言池后续仍保留指令，高 task 读出不等于任务语义理解或动作需要。最高层或最高分不等于动作最重要层；AE 本轮没有分析。</p>
<p class="small">数据来源：conditional_probes.csv、backend_control_probes.csv。CSV 和 summary.json 保存全部分数、区间、聚类和敏感性分析；原实验报告与输入表保持不变。</p>'''
    captions=[
        ('模块 × suite：绝对读出和相对偏好','task 为同服务 5-task 平均；phase 为同 task 内读出平均。星号仅表示 phase 的同时区间不跨零，不表示动作因果重要。'),
        ('phase 的深度倾向','左：实际 balanced accuracy。右：扣除整个模块在各 suite 的平均水平后，某个深度相对更适合哪个 suite。原生层号不跨模块等同。'),
        ('把层的偏好模式聚成组','聚类对象是 24 个模块／层的四维偏好向量；同组表示 suite 倾向相似，不表示可以互相替代或删除。'),
        ('把四个 suite 按模块／层指纹聚类','task 和 phase 分别聚类。phase 右下角为 task 重采样下固定分成两组的同组频率，不是自然簇数或泛化概率。')]
    for path,(title,caption) in zip(paths,captions):
        html_text+=f'<section class="figure"><h2>{title}</h2><p class="small">{caption}</p><img src="{path.as_uri()}"></section>'
    html_text+='</html>'
    (output/'report_zh.html').write_text(html_text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args();root=args.analysis.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    sources={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('conditional_probes.csv','backend_control_probes.csv')}
    keys,raw,baselines,supports=load_scores(root)
    projection=depth_projection(keys);indices=np.asarray([FEATURES.index(f) for f,_ in keys])
    gain_cases={target:baseline_gain(raw[target],baselines[target]) for target in raw}
    gains={target:value.mean(-1) for target,value in gain_cases.items()}
    stats={target:decompose(value,projection,indices) for target,value in gains.items()}
    boot_gain=task_bootstrap(gain_cases['phase'],args.bootstrap)
    boot=decompose(boot_gain,projection,indices)
    intervals,critical=simultaneous_intervals(stats['phase'],boot)
    cluster_info={}
    for target in ('task','phase'):
        tree,distance=suite_tree(gains[target],projection)
        pair=' + '.join(NAMES[i] for i in sorted(tree[0,:2].astype(int)))
        sensitivity=[]
        for omitted in range(4):
            keep=indices!=omitted
            alternate,_=suite_tree(gains[target][keep],np.delete(projection,omitted,axis=0)[:,keep])
            sensitivity.append(dict(omitted_module=SHORT[omitted],first_pair=[NAMES[i] for i in sorted(alternate[0,:2].astype(int))]))
        uniform_tree,_=suite_tree(gains[target],depth_projection(keys,uniform=True))
        info=dict(tree=tree.tolist(),distance=distance.tolist(),order=leaves_list(tree).tolist(),
                  first_pair=pair,leave_one_module_out=sensitivity,
                  equal_layer_weight_first_pair=[NAMES[i] for i in sorted(uniform_tree[0,:2].astype(int))])
        if target=='phase':
            coassignment=np.zeros((4,4));first=Counter()
            for draw in boot_gain:
                alternate,_=suite_tree(draw,projection)
                first[' + '.join(NAMES[i] for i in sorted(alternate[0,:2].astype(int)))]+=1
                labels=fcluster(alternate,2,criterion='maxclust')
                coassignment+=labels[:,None]==labels[None,:]
            info.update(coassignment=(coassignment/len(boot_gain)).tolist(),
                        first_pair_bootstrap_frequency={key:value/len(boot_gain) for key,value in first.items()})
        cluster_info[target]=info
    rows=[];module_rows=[]
    for target in ('task','phase'):
        for i,(feature,layer) in enumerate(keys):
            for s,suite in enumerate(SUITES):
                row=dict(target=target,feature=feature,layer=layer,suite=suite,
                    balanced_accuracy=float(raw[target][i,s].mean()),normalized_gain=float(gains[target][i,s]),
                    module_preference=float(stats[target]['module'][indices[i],s]),
                    depth_preference=float(stats[target]['depth'][i,s]),overall_preference=float(stats[target]['overall'][i,s]))
                for key in ('low','high','point_low','point_high'):
                    row['phase_depth_'+key]=float(intervals['depth'][key][i,s]) if target=='phase' else None
                rows.append(row)
        raw_auc=projection@raw[target].mean(-1)
        for m,feature in enumerate(FEATURES):
            for s,suite in enumerate(SUITES):
                row=dict(target=target,feature=feature,suite=suite,average_balanced_accuracy=float(raw_auc[m,s]),
                         normalized_auc=float(stats[target]['auc'][m,s]),preference=float(stats[target]['module'][m,s]))
                for key in ('low','high','point_low','point_high'):
                    row['phase_'+key]=float(intervals['module'][key][m,s]) if target=='phase' else None
                module_rows.append(row)
    write_csv(output/'layer_suite_preferences.csv',rows);write_csv(output/'module_suite_preferences.csv',module_rows)
    write_csv(output/'phase_task_support.csv',supports)
    paths,row_clusters=figures(output,keys,raw,gains,projection,indices,stats,intervals,cluster_info)
    summary=dict(suites=SUITES,modules=FEATURES,keys=[dict(feature=f,layer=l,label=f'{SHORT[FEATURES.index(f)]} H{l}') for f,l in keys],
        source_sha256=sources,bootstrap_draws=args.bootstrap,bootstrap_seed=7,
        simultaneous_family_cells=112,simultaneous_critical=critical,clusters=cluster_info,
        row_clusters=row_clusters,module_preference={target:stats[target]['module'].tolist() for target in stats},
        raw_module_auc={target:(projection@raw[target].mean(-1)).tolist() for target in raw},
        supported_phase_module_cells=int(np.sum((intervals['module']['low']>0)|(intervals['module']['high']<0))),
        supported_phase_depth_cells=int(np.sum((intervals['depth']['low']>0)|(intervals['depth']['high']<0))),
        caveat='Readout preference only; no causal layer/module importance or ablation result',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    build_report(output,paths,summary,stats,intervals)
    (output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    for name,digest in sources.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    print(json.dumps(dict(passed=True,layer_rows=len(rows),module_rows=len(module_rows),figures=len(paths),
                         task_first_pair=cluster_info['task']['first_pair'],phase_first_pair=cluster_info['phase']['first_pair'],
                         phase_pair_bootstrap=cluster_info['phase']['first_pair_bootstrap_frequency'],output=str(output)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
