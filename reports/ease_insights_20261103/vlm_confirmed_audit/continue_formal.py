"""Read-only 30-second completion monitor followed by the authorized CPU audit.

Only this audit directory and the supplementary analysis output are written.
The original pipeline owns its completion markers and primary analysis.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from preparations.ease_insights_20261103.vlm.common import atomic_json, sha

WORKSPACE = Path('/root/nfs/embodied/benchmark')
RESULTS = WORKSPACE/'results/ease_insights_20261103'
ROOT = RESULTS/'vlm_confirmation'
PRIMARY = WORKSPACE/'reports/ease_insights_20261103/vlm_confirmation'
OUTPUT = RESULTS/'vlm_confirmation_supplementary_analysis'
AUDIT = Path(__file__).resolve().parent
READY = RESULTS/'pipeline/vlm_report_ready.json'
JOB = 'vlm_confirm_full_d10_all200_retry2'
CONDITIONS = ('full_d10','fixed_v12_jacobian','fixed_v12_action_mse')


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def snapshot():
    jobs = read(RESULTS/'scheduler/jobs_state.json', {})
    job = jobs.get(JOB, {})
    pipeline = read(RESULTS/'pipeline/state.json', {}).get('branches', {}).get('vlm', {})
    markers = {cid: read(ROOT/cid/'confirm_shard00of01_complete.json', {}) for cid in CONDITIONS}
    main = read(PRIMARY/'analysis.json', {}) if READY.exists() else {}
    counts = {cid: sum(1 for _ in (ROOT/cid/'episodes').glob('*/task*/init[0-9][0-9]/episode_receipt.json')) for cid in CONDITIONS}
    return dict(observed_unix=time.time(), status='waiting', full_job_status=job.get('status'), full_job_pid=job.get('pid'),
        full_job_pid_exists=bool(job.get('pid') and Path('/proc',str(job['pid'])).exists()),
        pipeline_phase=pipeline.get('phase'), pipeline_error=pipeline.get('error'),
        receipt_counts_unvalidated=counts, completed_markers={cid: bool(m.get('complete')) for cid,m in markers.items()},
        report_ready=READY.exists(), primary_complete=bool(main.get('complete')),primary_valid_episodes=main.get('valid_episodes'),
        primary_errors=dict(missing=len(main.get('missing',[])),invalid=len(main.get('invalid',[])),unexpected=len(main.get('unexpected_completed_cases',[]))))


def finalize():
    env=dict(os.environ, CUDA_VISIBLE_DEVICES='')
    command=[sys.executable,'-m','preparations.ease_insights_20261103.vlm.analyze_supplementary',
        '--root',str(ROOT),'--output',str(OUTPUT),'--contention-root',str(RESULTS/'resource_contention')]
    with (AUDIT/'supplementary_analysis.log').open('w') as log:
        subprocess.run(command,cwd=WORKSPACE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    main=read(PRIMARY/'analysis.json');supp=read(OUTPUT/'supplementary_analysis.json')
    for report in (main,supp):
        assert report['complete'] and report['status']=='complete' and report['valid_episodes']==600
        assert not report['missing'] and not report['invalid'] and not report['unexpected_completed_cases']
    assert main['manifest_hash']==supp['manifest_hash'] and main['selection_hash']==supp['selection_hash']
    assert main['conclusion']==supp['original_primary_analysis']['conclusion']
    for before,after in zip(main['primary_directions'],supp['original_primary_analysis']['primary_directions']):
        for field in ('frozen_direction','n_paired','a_success','b_success','a_only','b_only','success_difference','inference'):
            assert before[field]==after[field], field
    all_conditions={r['condition']:r for r in main['condition_summaries'] if r['level']=='all'}
    by_suite=[r for r in main['condition_summaries'] if r['level']=='suite']
    audit=dict(status='complete',audited_unix=time.time(),valid_episodes=600,expected_episodes=600,
        manifest_hash=main['manifest_hash'],selection_hash=main['selection_hash'],
        original_primary_conclusion=main['conclusion'],
        supplementary_goal_long_conclusion=supp['supplementary_goal_long_analysis']['conclusion'],
        original_primary_matches_supplementary_copy=True,
        prospective_supplementary_registration=supp['prospective_registration'],
        condition_summaries=all_conditions,suite_condition_summaries=by_suite,
        three_fixed_contrasts=supp['contrasts'],
        separate_primary_holm2=main['primary_directions'],
        scientific_results_modified=False,new_gpu_episodes=0,runtime_manifest_selection_or_queue_modified=False,
        limits=['Original Spatial/Long and supplementary Goal/Long conclusions remain separate.',
            'Negative or uncertain results remain unchanged by any later exploratory grouping or new experiment.',
            'Non-significance does not establish equivalence; ten task clusters and five initial states limit precision.',
            'Results concern fixed paths, not runtime selection or phase switching.',
            'Policy/RPC/wall timing and fixed-input ratios can be affected by external contention; no exclusive-GPU speed claim.'],
        report_evidence=[dict(path=str(path),sha256=sha(path)) for path in (READY,PRIMARY/'analysis.json',PRIMARY/'analysis.md',
            OUTPUT/'supplementary_analysis.json',OUTPUT/'supplementary_analysis.md',ROOT/'manifest.json',ROOT/'supplementary_goal_contrast.json',Path(__file__))])
    atomic_json(AUDIT/'audit.json',audit)
    labels={'full_d10':'Full','fixed_v12_jacobian':'Jac','fixed_v12_action_mse':'MSE'}
    lines=['本轮VLM独立复验600/600条通过完整审计，正式主分析和补充分析均已完成。',
        '原Spatial/Long结论：'+main['conclusion']+'。',
        '预先登记的补充Goal/Long结论：'+audit['supplementary_goal_long_conclusion']+'。','',
        '补充报告中的原主分析逐项与原正式报告一致，保留原Holm2；以下三方向另统一给出Holm3。','',
        '| 冻结方向 | A/B成功数 | 差值 | 单侧精确p | Holm3 | Case 95%CI | Task-cluster 95%CI |',
        '|---|---:|---:|---:|---:|---|---|']
    for r in supp['contrasts']:
        inf=r['inference'];ci=inf['confidence_intervals'];d=r['frozen_direction']
        lines.append(f"| {d['suite']}: {labels[d['better']]}>{labels[d['worse']]} | {r['a_success']}/{r['b_success']}（各50） | {100*r['success_difference']:+.1f}pp | {inf['exact_mcnemar_one_sided']:.6g} | {inf['holm_adjusted_p']:.6g} | [{100*ci['case']['low']:+.1f}, {100*ci['case']['high']:+.1f}]pp | [{100*ci['task_cluster']['low']:+.1f}, {100*ci['task_cluster']['high']:+.1f}]pp |")
    lines += ['', '| 条件 | 全部成功率 |', '|---|---:|']
    for cid,row in all_conditions.items():
        # Shared analysis summary schema is recorded in JSON; format its keys directly.
        successes=row.get('successes',row.get('success_count',row.get('success')))
        lines.append(f"| {labels[cid]} | {successes}/{row.get('n',row.get('episodes',200))} |")
    lines += ['', '补充注册审计：'+supp['prospective_registration']['classification']+'。原identity没有内嵌Unix时间，先后关系使用包括技术重试在内的原始文件mtime；这一限制保留在审计中。',
        '这些是固定路径在init31–35上的结果。不能将非显著解释为等价，也不能据此声称运行时选路、阶段切换或独占GPU加速。外部资源争用不会改写质量成功/失败；成本证据及其限制保存在正式报告中。',
        '后续若使用这些数据重新选择任务或语义类别，只能作为新的discovery，不改写本轮阴性或不确定结论。',
        '完整逐task/init、leave-one-out、Full参照及证据SHA见audit.json和原正式主/补报告。']
    (AUDIT/'README_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(event='formal_audit_complete',audit=str(AUDIT/'audit.json'),
        primary=audit['original_primary_conclusion'],supplementary=audit['supplementary_goal_long_conclusion'],
        condition_summaries=all_conditions)),flush=True)


def main():
    absent_pid_checks=0
    last_output=0.
    while True:
        current=snapshot()
        atomic_json(AUDIT/'monitor_status.json',current)
        if current['full_job_status'] in ('failed','interrupted_or_launch_error'):
            raise RuntimeError('Unexpected Full retry technical failure: '+json.dumps(current))
        absent_pid_checks=absent_pid_checks+1 if current['full_job_status']=='running' and not current['full_job_pid_exists'] else 0
        if absent_pid_checks>=2:
            raise RuntimeError('Full retry recorded running but PID absent for two polls: '+json.dumps(current))
        if current['pipeline_phase']=='needs_attention':
            raise RuntimeError('Original pipeline requires attention: '+json.dumps(current))
        if current['report_ready'] and current['primary_complete'] and current['primary_valid_episodes']==600 and all(current['completed_markers'].values()):
            assert sum(current['receipt_counts_unvalidated'].values())==600
            print(json.dumps(dict(event='all_markers_and_formal_primary_ready',**current)),flush=True)
            finalize()
            atomic_json(AUDIT/'monitor_status.json',dict(current,status='complete',finished_unix=time.time()))
            return
        if time.time()-last_output>=60:
            print(json.dumps(current),flush=True);last_output=time.time()
        time.sleep(30)


if __name__=='__main__':
    main()
