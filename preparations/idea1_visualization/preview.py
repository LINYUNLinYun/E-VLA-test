"""Fixed first-two-task preview from completed real samples; not the final report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .analyze import build_explorer,classifier,scatter,write_csv,fit_projection,project
from .common import ROOT,atomic_json,episode_path


def main():
    root=ROOT/'results/idea1_visualization/run_20261026'
    output=root/'preview_first2_spatial';output.mkdir(exist_ok=True)
    keys=['vision/base_0_rgb/hidden','vision/left_wrist_0_rgb/hidden',
          'vlm/vision/hidden','vlm/language/hidden']
    samples=[];data={key:[] for key in keys}
    for task in (0,1):
        for init in range(3):
            directory=episode_path(root,'libero_spatial',task,init)
            identity=json.loads((directory/'identity.json').read_text())
            selected=json.loads((directory/'selection.json').read_text())['samples']
            for item in selected:
                path=root/'features/libero_spatial'/f'task{task:02d}'/f'init{init:02d}'/f"request_{item['request']:04d}"
                if not (path/'complete.json').exists() or not (path/'sample.json').exists():
                    raise RuntimeError(f'Fixed preview cases still incomplete: {path}')
                samples.append(dict(item,task=task,init=init,instruction=identity['instruction']))
                with np.load(path/'pooled.npz',allow_pickle=False) as arrays:
                    for key in keys:data[key].append(arrays[key])
    data={key:np.stack(values) for key,values in data.items()}
    tasks=np.asarray([s['task'] for s in samples]);phase=np.asarray([s['phase'] for s in samples])
    init=np.asarray([s['init'] for s in samples]);train=init<2;test=init==2
    probes=[];projections=[];fig,axes=plt.subplots(4,4,figsize=(19,16))
    for row,key in enumerate(keys):
        for col,layer in enumerate((0,data[key].shape[1]-1)):
            x=data[key][:,layer,:]
            pca,active,tolerance=fit_projection(x[train]);z=project(pca,active,x)
            scatter(axes[row,col*2],z,tasks)
            scatter(axes[row,col*2+1],z,phase,True)
            for ax,role in ((axes[row,col*2],'task'),(axes[row,col*2+1],'phase')):
                ax.set_title(f'{key}\nH{layer} / {role} | PC1 {pca.explained_variance_ratio_[0]:.1%}, PC2 {pca.explained_variance_ratio_[1]:.1%}',fontsize=9)
                ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.legend(fontsize=6,ncol=2)
            km=KMeans(n_clusters=2,n_init=5,random_state=0).fit(project(pca,active,x[train]))
            projections.append(dict(feature=key,layer=layer,suite='libero_spatial',
                non_null_retained_components=int(active.sum()),numerical_rank_tolerance=tolerance,
                cluster=km.predict(z).tolist(),cluster_spec=dict(k=2,seed=0,fit='init0/1, high-dimensional PCA space; two fixed tasks preview'),
                variance=pca.explained_variance_ratio_[:4].tolist(),coordinates=z[:,:4].tolist(),
                task=tasks.tolist(),phase=phase.tolist(),
                episode=[f"libero_spatial/{s['task']}/{s['init']}" for s in samples],
                request=[s['request'] for s in samples],sources=[s['sources'] for s in samples],
                phase_reason=[s['phase_reason'] for s in samples],
                active_objects=[s['phase_evidence'].get('active_objects',[]) for s in samples],
                task_instructions={str(s['task']):s['instruction'] for s in samples}))
            probes.append(dict(feature=key,layer=layer,target='task',condition='two_fixed_tasks',
                               **classifier(x,tasks,train,test)))
            for task in (0,1):
                known=(phase!='unknown')&(tasks==task)
                probes.append(dict(feature=key,layer=layer,target='phase',condition=f'within_task{task}',
                                   **classifier(x,phase,train&known,test&known)))
    fig.suptitle('PREVIEW ONLY: LIBERO Spatial tasks0/1, init0/1 train, init2 held out; trace-inferred phases',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output/'pca_preview.png',dpi=140);plt.close(fig)
    write_csv(output/'probe_preview.csv',probes)
    atomic_json(output/'projections.json',projections);build_explorer(output,projections)
    summary=dict(scope='Fixed first Spatial tasks0/1, three inits each; incomplete experiment preview only',
        episodes=6,samples=len(samples),train_samples=int(train.sum()),test_samples=int(test.sum()),
        task_service_assignment={'0':'GPU1','1':'GPU2'},
        language_H0_max_within_task_difference={str(t):float(np.max(np.abs(data['vlm/language/hidden'][tasks==t,0]-data['vlm/language/hidden'][np.where(tasks==t)[0][0],0]))) for t in (0,1)},
        production_parameters_changed=False)
    atomic_json(output/'summary.json',summary)
    (output/'report_zh.md').write_text('\n'.join([
        '# 固定前两个 Spatial task 的预览','',
        f"使用 task0/1 各 init0–2，共 6 条真实轨迹、{len(samples)} 个观测。init0/1 训练，init2 留出。完整实验仍在提取 2326 个观测，本预览不替代全四套件结果。",'',
        '两个预览 task 分别在 GPU1／GPU2 的持续服务上采集，task 与服务绑定，任务分离仅供流程检查。正式报告另提供同一服务内 5-task 读出对照；每个 task 的 phase 对照始终位于同一服务。','',
        '- [二维／三维与前四维配对交互图](pca_explorer.html)',
        '- [H0 与最后一层的 PCA 静态图](pca_preview.png)',
        '- [task 与同 task 内 phase 的留出读出数值](probe_preview.csv)',
        '- [样本量与输入基线核对](summary.json)','',
        'phase 是 trace 推断标签。本配置关闭离散机器人状态输入，H0 语言 token 的同 task 内特征应固定。PCA 使用 float64；仅将训练数据中数值上未确定的零空间坐标置零，保留有效低方差方向；原始特征 probe 不受影响。',
        '此预览不用于挑选最终分析参数、样本或最终结论。','']) )
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
