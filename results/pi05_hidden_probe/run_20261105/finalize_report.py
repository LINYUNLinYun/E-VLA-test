"""Audit complete matrices and report calibration-selected operating points."""
from __future__ import annotations
import hashlib,json,subprocess,os
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.pi05_hidden_probe.common import load_config,read_jsonl,select_rows,digest,config_hash,write_json,observation_hash
from experiments.pi05_hidden_probe.workflow import expected_result_keys,row_key
from experiments.pi05_hidden_probe.data import load_observation,read_manifest
from experiments.pi05_hidden_probe.metrics import balanced_accuracy

BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[2]


def bootstrap_point(real,nopatch,shuffle,seed=42):
    key=lambda r:(r['sample_id'],r['control_id'])
    rmap={key(r):r for r in real};nmap={key(r):r for r in nopatch};smap={key(r):r for r in shuffle}
    if len(rmap)!=len(real) or rmap.keys()!=nmap.keys() or rmap.keys()!=smap.keys():raise ValueError('Operating-point controls are not paired')
    groups=sorted({r['group_id'] for r in real});labels=real[0]['labels']
    indexes={g:i for i,g in enumerate(groups)}
    counts=np.zeros((len(groups),len(labels)))
    correct={condition:np.zeros_like(counts) for condition in ['real','nopatch','shuffle','free']}
    donor=np.zeros(len(groups));totals=np.zeros(len(groups))
    cf=defaultdict(list)
    for key_,row in rmap.items():
        g=indexes[row['group_id']];y=labels.index(row['label']);counts[g,y]+=1;totals[g]+=1
        for condition,mapping in [('real',rmap),('nopatch',nmap),('shuffle',smap)]:
            correct[condition][g,y]+=mapping[key_]['score_prediction']==row['label']
        correct['free'][g,y]+=row['prediction']==row['label']
        donor[g]+=smap[key_]['score_prediction']==smap[key_]['donor_label']
        cf[(row['group_id'],row['counterfactual_group_id'],row['control_id'])].append(row)
    rng=np.random.default_rng(seed);draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=1000)
    def values(success,weights):
        return np.mean((weights@success)/(weights@counts),axis=1)
    point={name:float(values(success,np.ones((1,len(groups))))[0]) for name,success in correct.items()}
    distributions={name:values(success,draws) for name,success in correct.items()}
    ci=lambda a:[float(x) for x in np.quantile(a,[.025,.975])]
    cf_ok=np.zeros(len(groups));cf_n=np.zeros(len(groups))
    for (group,_,_),rows in cf.items():
        if len(rows) not in [2,4]:raise ValueError('Incomplete counterfactual group at operating point')
        cf_n[indexes[group]]+=1;cf_ok[indexes[group]]+=all(r['score_prediction']==r['label'] for r in rows)
    return {'candidate_bacc':point['real'],'candidate_bacc_ci':ci(distributions['real']),
        'generation_bacc':point['free'],'generation_bacc_ci':ci(distributions['free']),
        'nopatch_candidate_bacc':point['nopatch'],'shuffle_candidate_bacc':point['shuffle'],
        'delta_nopatch':point['real']-point['nopatch'],'delta_nopatch_ci':ci(distributions['real']-distributions['nopatch']),
        'delta_shuffle':point['real']-point['shuffle'],'delta_shuffle_ci':ci(distributions['real']-distributions['shuffle']),
        'donor_follow_rate':float(donor.sum()/totals.sum()),'donor_follow_ci':ci((draws@donor)/(draws@totals)),
        'counterfactual_all_correct':float(cf_ok.sum()/cf_n.sum()),'counterfactual_all_correct_ci':ci((draws@cf_ok)/(draws@cf_n)),
        'observations':len({r['sample_id'] for r in real}),'readouts':len(real),'independent_groups':len(groups),
        'per_control_candidate_bacc':{c:balanced_accuracy([r for r in real if r['control_id']==c],'score_prediction') for c in sorted({r['control_id'] for r in real})}}


def audit_matrix(config,rows,result_path):
    identity=json.loads((result_path.parent/'run_identity.json').read_text())
    completion=json.loads((result_path.parent/'completion.json').read_text())
    if hashlib.sha256(result_path.read_bytes()).hexdigest()!=completion['results_sha256']:raise ValueError('Completed results changed')
    if not completion['completed'] or identity['config_hash']!=config_hash(config):raise ValueError('Run identity mismatch')
    if digest(rows)!=identity['manifest_digest']:raise ValueError('Manifest changed')
    records=read_jsonl(result_path)
    layers=list(range(identity['source_identity']['audit']['depth']+1)) if config['source_layers']=='all' else config['source_layers']
    keys=[row_key(r) for r in records]
    if len(keys)!=len(set(keys)) or set(keys)!=expected_result_keys(config,rows,layers):raise ValueError('Missing/duplicate result matrix cells')
    for row in rows:
        if observation_hash(load_observation(row))!=identity['source_input_digests'][row['sample_id']]:raise ValueError('Observation contents changed')
    for r in records:
        if not all(np.isfinite(v) for v in r['candidate_scores'].values()):raise ValueError('Nonfinite candidate scores')
    return records,identity,{'rows':len(records),'samples':len(rows),'source_layers':layers,'full_coverage':True,
                            'results_sha256':completion['results_sha256'],'input_contents_match':True}


def main():
    supervisor=json.loads((BASE/'experiment_supervisor.json').read_text())
    if supervisor['stage']!='all_runs_and_analysis_complete':raise RuntimeError('The full experiment has not finished')
    report=[];audits={};identities={}
    for channel in ['image','text']:
        frozen=load_config(BASE/channel/'frozen_probe_config.yaml')
        cal=load_config(BASE/(channel+'_calibration.yaml'))
        calibration,_,cal_audit=audit_matrix(cal,select_rows(cal),BASE/channel/'calibration/probe_results.jsonl')
        test,identity,test_audit=audit_matrix(frozen,select_rows(frozen),BASE/channel/'evaluation/probe_results.jsonl')
        identities[channel]=identity
        for stage in ['calibration','evaluation']:
            gate=json.loads((BASE/channel/stage/'patch_integrity.json').read_text())
            if not gate['passed']:raise ValueError('A scientific run did not pass its engineering gate')
        if hashlib.sha256((BASE/channel/'calibration/probe_results.jsonl').read_bytes()).hexdigest()!=frozen['calibration_provenance']['sha256']:raise ValueError('Frozen selection provenance mismatch')
        audits[channel]={'calibration':cal_audit,'independent_test':test_audit,'frozen_config_intact':True,'engineering_gates_passed':True}
        for probe in frozen['probes']:
            q=probe['target_layers'][0]
            candidates={layer:balanced_accuracy([r for r in calibration if r['probe_id']==probe['id'] and r['condition']=='real' and r['source_layer']==layer and r['target_layer']==q],'score_prediction') for layer in cal['source_layers']}
            layer=min(candidates,key=lambda l:(-candidates[l],l))
            relevant=[r for r in test if r['probe_id']==probe['id'] and r['target_layer']==q]
            real=[r for r in relevant if r['condition']=='real' and r['source_layer']==layer]
            nopatch=[r for r in relevant if r['condition']=='nopatch']
            shuffle=[r for r in relevant if r['condition']=='shuffle' and r['source_layer']==layer]
            metrics=bootstrap_point(real,nopatch,shuffle)
            direct=[r for r in relevant if r['condition']=='direct']
            report.append({'channel':channel,'task_type':probe['task_type'],'probe_id':probe['id'],
                'target_layer':q,'summary_source_layer_selected_on_calibration':layer,
                'source_token_count':len(real[0]['source_positions']),'calibration_source_candidates':candidates,
                'direct_target_candidate_bacc':balanced_accuracy(direct,'score_prediction'),**metrics})
        # Calibration heatmaps have the full observed 7x4 grid; the test curves
        # contain all 19 Source layers under the one frozen Target layer.
        analysis=BASE/channel/'calibration/analysis';analysis.mkdir(exist_ok=True)
        expected_analysis={'input_sha256':hashlib.sha256((BASE/channel/'calibration/probe_results.jsonl').read_bytes()).hexdigest(),
            'bootstrap_samples':1000,'seed':42,
            'metrics_code_sha256':hashlib.sha256((ROOT/'experiments/pi05_hidden_probe/metrics.py').read_bytes()).hexdigest(),
            'analysis_code_sha256':hashlib.sha256((ROOT/'experiments/pi05_hidden_probe/analyze_results.py').read_bytes()).hexdigest()}
        identity_path=analysis/'analysis_identity.json'
        reusable=identity_path.exists() and json.loads(identity_path.read_text())==expected_analysis and (analysis/'layer_metrics.csv').exists() and (analysis/'summary.md').exists()
        if not reusable:
            with (analysis/'analysis.log').open('w') as log:
                process=subprocess.run([str(ROOT/'policies/openpi/.venv/bin/python'),'-m','experiments.pi05_hidden_probe.analyze_results',
                    '--input',str(BASE/channel/'calibration/probe_results.jsonl'),'--output-dir',str(analysis),
                    '--bootstrap-samples','1000','--seed','42'],cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1'},stdout=log,stderr=subprocess.STDOUT)
            if process.returncode:raise RuntimeError('Calibration analysis failed')
            write_json(identity_path,expected_analysis)
    manifest=read_manifest(BASE/'counterfactual_data_v2/manifest.jsonl')
    cal_groups={r['group_id'] for r in manifest if r['split']=='calibration'}
    test_groups={r['group_id'] for r in manifest if r['split']=='test'}
    if cal_groups&test_groups:raise ValueError('Calibration/test leakage')
    write_json(BASE/'operating_point_results.json',{'selection_rule':'Source layer chosen only from calibration after q frozen; both controls pooled with grouped paired bootstrap','results':report})
    write_json(BASE/'completion_audit.json',{'all_scientific_runs_complete':True,'audits':audits,
        'calibration_independent_groups':len(cal_groups),'test_independent_groups':len(test_groups),'group_disjoint':True,
        'dataset_samples':len(manifest),'readout_scope':'closed-set primary; generation secondary; no action-policy causal claim',
        'final_interpretation_review_required':True})
    write_json(BASE/'model_audit.json',{'source':identities['image']['source_identity'],'targets':{c:i['target_identity'] for c,i in identities.items()}})
    pct=lambda x:f'{100*x:.1f}%'
    ci=lambda x:f'[{100*x[0]:.1f}, {100*x[1]:.1f}]'
    names={'visual_attribute':'颜色','spatial_relation':'左右关系','instruction_conditioned':'指令条件化定位'}
    lines=['# π0.5 隐藏表征实验结果','',
      'Source 使用实际机器人微调后的 π0.5-LIBERO；Target 是冻结的官方 PaliGemma-3B-mix-224。两者之间仅使用恒等表征注入，无训练映射。Target 用来读出表示，不负责机器人动作。','',
      '320 条样本来自 40 个官方初态派生的合成反事实，覆盖颜色、左右关系和指令×布局 2×2；按场景分为 96 条校准与 224 条独立测试。所有源边界 0…18 均完成独立测试。','',
      '主要指标是候选短答案似然的 balanced accuracy。自由生成单独统计。表中目标层及摘要源层均仅由校准数据选择，两个固定控制图一起按完整场景进行 1,000 次配对 bootstrap。','',
      '| 路径 | 属性 | Source层 / Target层 | Token数 | 测试候选 BAcc（95% CI） | Δ打乱源（95% CI） | 成对／四组全对 | Donor跟随 | 自由生成 BAcc |',
      '|---|---|---|---:|---|---|---:|---:|---:|']
    for r in report:
        lines.append(f"| {'视觉块' if r['channel']=='image' else '单文本token'} | {names[r['task_type']]} | {r['summary_source_layer_selected_on_calibration']} / {r['target_layer']} | {r['source_token_count']} | {pct(r['candidate_bacc'])} {ci(r['candidate_bacc_ci'])} | {pct(r['delta_shuffle'])} {ci(r['delta_shuffle_ci'])} | {pct(r['counterfactual_all_correct'])} | {pct(r['donor_follow_rate'])} | {pct(r['generation_bacc'])} |")
    lines+=['','灰／黑两种控制图分别的结果、正常看图参考和全部 Source 层曲线见各路径分析文件。上表的成对指标用于颜色与左右任务，四组全对用于指令条件化任务。','',
      '## 结果文件','',
      '- `operating_point_results.json`：完整摘要指标及每种控制图结果。',
      '- `completion_audit.json`：原始矩阵覆盖、输入内容、工程门槛、冻结配置和划分核验。',
      '- `image/evaluation/analysis/`、`text/evaluation/analysis/`：逐层指标、置信区间、曲线和随机生成案例。',
      '- `image/calibration/analysis/`、`text/calibration/analysis/`：实际测得的 Source×Target 校准热图。',
      '- 各路径 `frozen_probe_config.yaml` 和 `probe_results.jsonl` 保存完整可复核配置与输出。','',
      '## 结论边界','',
      '这里只评估这组冻结检查点、提示、层及 token 配置下的可解码性。H0 已经过视觉编码器；不能把曲线解释为语义首次出现，不能由读出失败推断信息不存在，也不能据此断言机器人动作使用了该属性。','',
      '多视觉 token 与单文本 token 的信息容量不同。任务条件化成功也可能包含 Target 后续计算完成的目标绑定。所有数据均来自同一 LIBERO 场景模板的反事实，尚不证明跨任务泛化。封闭候选读出也不等同开放式自然语言自解释。','']
    (BASE/'summary.md').write_text('\n'.join(lines))
    print(BASE/'summary.md',flush=True)


if __name__=='__main__':main()
