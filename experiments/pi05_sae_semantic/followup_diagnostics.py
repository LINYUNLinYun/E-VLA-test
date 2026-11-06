"""Bounded diagnostics on the same completed pilot, then one module matrix.

The nominal published-candidate analysis runs first in collect_local. This
follow-up keeps dictionary re-association and instruction-only readout in
separate directories; it serializes physical interventions with the first run.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--collection',required=True)
    p.add_argument('--assets',required=True)
    p.add_argument('--run-root',required=True)
    p.add_argument('--primary-pid',type=int,required=True)
    args=p.parse_args()
    root,collection,assets=map(lambda x:Path(x).resolve(),(args.run_root,args.collection,args.assets))
    state_path=root/'followup_status.json'
    def state(value,**kwargs):
        record=dict(state=value,pid=os.getpid(),updated_unix=time.time(),**kwargs)
        state_path.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
    def primary_live():
        try:os.kill(args.primary_pid,0)
        except ProcessLookupError:raise RuntimeError('Primary process ended before its expected artifact; inspect status instead of restarting')
    state('waiting_for_complete_pilot')
    cfg=json.loads((collection/'collection_config.json').read_text())
    while not all((collection/'observations'/identity/'complete.json').exists() for identity in cfg['observations']):
        primary_live();time.sleep(20)
    variants=[('dictionary_all','all',root/'semantic_dictionary_all_complete'),
              ('dictionary_instruction','instruction',root/'semantic_dictionary_instruction_complete')]
    for name,pool,output in variants:
        state('semantic_diagnostic_running',variant=name)
        subprocess.run([sys.executable,'-u','-m','experiments.pi05_sae_semantic.select_sae_concepts',
            '--collection',str(collection),'--assets',str(assets),'--output',str(output),
            '--candidate-pool','full_dictionary','--token-pool',pool],check=True)
    paths=[('released_all',root/'semantic_selection_cpu'),*[(name,path) for name,pool,path in variants]]
    state('waiting_for_primary_interventions')
    while True:
        status=collection/'postprocessing_status.json'
        try:finished=status.exists() and json.loads(status.read_text()).get('state')=='postprocessing_complete'
        except json.JSONDecodeError:finished=False
        if finished:break
        primary_live();time.sleep(20)
    rows=[]
    for name,path in paths:
        summary=json.loads((path/'semantic_selection_summary.json').read_text())
        definitions=json.loads((path/'concept_features.json').read_text())
        for concept,layers in summary['concepts'].items():
            for layer,entry in layers.items():
                metric=entry.get('calibration_metrics',{})
                rows.append(dict(variant=name,observations=summary['completed_observations'],concept=concept,layer=layer,
                    selected_features=len(definitions[concept]['layers'][layer]['features']),theta=entry.get('theta'),
                    precision=metric.get('precision'),recall=metric.get('recall'),false_positive_rate=metric.get('false_positive_rate'),
                    positive_episodes=metric.get('positive_episodes'),supported=entry['supported']))
    with (root/'semantic_variant_comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=['# 同一 64 观测上的语义读出对照','',
        '三种读出都使用冻结的 π0.5、同一对发布 SAE 和同一发现/校准划分。特征关联仅由发现集计算，校准集只检查分数阈值；这些是探索性结果，不是独立测试或成功率保证。','',
        '| 读出 | 概念 | 层 | 特征数 | 校准 precision | recall | FPR | 支持 |','|---|---|---:|---:|---:|---:|---:|---|']
    def show(v):return '—' if v is None else f'{v:.3f}'
    report += [f"| {r['variant']} | {r['concept']} | {r['layer']} | {r['selected_features']} | {show(r['precision'])} | {show(r['recall'])} | {show(r['false_positive_rate'])} | {r['supported']} |" for r in rows]
    report += ['', 'released_all 直接使用公开候选池；dictionary_all 在同一 SAE 字典中重新关联特征；dictionary_instruction 只用已保存的指令 token 读出。没有新增 VLM 前向或 SAE 训练。',
        '', '指令 token 含有视觉上下文，但当前标签是静态任务语义；不能把分类表现直接解释为操作阶段识别。Prompt-only 的额外价值对照留待策略阶段。',
        '', '后续 FFN 矩阵使用冻结 L5 触发选样，L16 不做逐样本筛选；物理干预按观测/mask 缓存，各种语义读出可复用同一次前向。', '']
    (root/'semantic_variant_comparison.md').write_text('\n'.join(report))
    selected=root/'semantic_dictionary_instruction_complete'
    state('instruction_module_matrix_running')
    subprocess.run([sys.executable,'-u','-m','experiments.pi05_sae_semantic.map_concept_ffn',
        '--collection',str(collection),'--selection',str(selected),'--assets',str(assets),
        '--output',str(root/'module_matrix_instruction_initial'),'--checkpoint',cfg['checkpoint'],
        '--device',cfg['device'],'--threads','32','--max-observations','12'],check=True)
    state('followup_complete')


if __name__=='__main__':
    main()
