"""Export the completed original VLM confirmation without rerunning statistics.

Inputs are the existing formal JSON reports. This script only converts their
reported values to CSV, figure coordinates and an aggregate cost table.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

WORKSPACE=Path(__file__).resolve().parents[3]
CONDITIONS=('full_d10','fixed_v12_jacobian','fixed_v12_action_mse')
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
NAMES={'full_d10':'Full','fixed_v12_jacobian':'Jac','fixed_v12_action_mse':'MSE'}
SUITE_NAMES={'libero_spatial':'Spatial','libero_object':'Object','libero_goal':'Goal','libero_10':'Long'}
COLORS={'full_d10':'#52606D','fixed_v12_jacobian':'#D88221','fixed_v12_action_mse':'#008A8A'}
CONTRAST_IDS=('primary_spatial','primary_long','supplementary_goal')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def load(primary_path,supplementary_path):
    main=json.loads(primary_path.read_text());supp=json.loads(supplementary_path.read_text())
    for report in (main,supp):
        assert report['status']=='complete' and report['complete'] and report['valid_episodes']==600
        assert not report['missing'] and not report['invalid'] and not report['unexpected_completed_cases']
    assert main['manifest_hash']==supp['manifest_hash']
    assert main['selection_hash']==supp['selection_hash']
    assert main['conclusion']==supp['original_primary_analysis']['conclusion']
    assert [r['contrast_id'] for r in supp['contrasts']]==list(CONTRAST_IDS)
    for original,copy in zip(main['primary_directions'],supp['original_primary_analysis']['primary_directions']):
        for field in ('frozen_direction','n_paired','success_difference','inference'):
            assert original[field]==copy[field]
    return main,supp


def export(main,supp,output,inputs):
    suite_lookup={(r['suite'],r['condition']):r for r in main['condition_summaries'] if r['level']=='suite'}
    suite_rows=[]
    for suite in SUITES:
        for cid in CONDITIONS:
            r=suite_lookup[suite,cid]
            assert r['n']==r['expected_n']==50
            suite_rows.append(dict(suite=suite,suite_label=SUITE_NAMES[suite],condition=cid,condition_label=NAMES[cid],
                successes=r['successes'],n=r['n'],success_rate=r['success_rate'],
                source_field='condition_summaries[level=suite]'))
    contrast_rows=[]
    for r in supp['contrasts']:
        ci=r['inference']['confidence_intervals']['task_cluster'];d=r['frozen_direction']
        assert r['n_paired']==50 and ci['clusters']==10
        contrast_rows.append(dict(contrast_id=r['contrast_id'],
            analysis_role='Original primary' if r['contrast_id'].startswith('primary_') else 'Prospective supplementary',
            suite=d['suite'],better=d['better'],worse=d['worse'],n_paired=r['n_paired'],
            better_successes=r['a_success'],worse_successes=r['b_success'],difference=r['success_difference'],
            difference_pp=100*r['success_difference'],task_cluster_ci_low=ci['low'],task_cluster_ci_high=ci['high'],
            task_cluster_ci_low_pp=100*ci['low'],task_cluster_ci_high_pp=100*ci['high'],
            exact_p_one_sided=r['inference']['exact_mcnemar_one_sided'],holm3_p=r['inference']['holm_adjusted_p'],
            interval_method=ci['method'],bootstrap_replicates=ci['replicates'],task_clusters=ci['clusters'],
            source_field='contrasts[].inference.confidence_intervals.task_cluster'))
    all_lookup={r['condition']:r for r in main['condition_summaries'] if r['level']=='all'}
    costs=[]
    for cid in CONDITIONS:
        r=all_lookup[cid]
        assert r['n']==200 and r['successes']==sum(s['successes'] for s in suite_rows if s['condition']==cid)
        costs.append(dict(condition=cid,label=NAMES[cid],successes=r['successes'],n=r['n'],success_rate=r['success_rate'],
            executed_actions=r['actions'],policy_requests=r['predictions'],policy_total_ms=r['policy']['total_ms'],
            rpc_total_ms=r['rpc']['total_ms'],policy_ms_per_executed_action=r['policy']['total_ms']/r['actions'],
            rpc_ms_per_executed_action=r['rpc']['total_ms']/r['actions'],
            summed_episode_wall_s=r['summed_episode_wall_s'],
            source_field='condition_summaries[level=all]; ratio of cumulative time to cumulative executed actions'))
    write_csv(output/'suite_success.csv',suite_rows)
    write_csv(output/'registered_contrasts.csv',contrast_rows)
    write_csv(output/'condition_quality_and_cost.csv',costs)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,
        'axes.linewidth':.8,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42,
        'svg.fonttype':'none','svg.hashsalt':'ease-vlm-original-confirmation-v1','savefig.facecolor':'white'})
    fig=plt.figure(figsize=(11.8,4.6),facecolor='white')
    ax=fig.add_axes([.065,.20,.44,.65])
    effect=fig.add_axes([.695,.20,.28,.65])
    width=.27
    for condition_index,cid in enumerate(CONDITIONS):
        x=[i+(condition_index-1)*width for i in range(4)]
        heights=[100*suite_lookup[suite,cid]['success_rate'] for suite in SUITES]
        bars=ax.bar(x,heights,width=width*.92,color=COLORS[cid],label=NAMES[cid],zorder=3)
        for bar,suite in zip(bars,SUITES):
            row=suite_lookup[suite,cid]
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.8,f"{row['successes']}/{row['n']}",
                ha='center',va='bottom',fontsize=8,color='#202833')
    ax.set_xlim(-.55,3.55);ax.set_ylim(0,112)
    ax.set_xticks(range(4),[SUITE_NAMES[s] for s in SUITES])
    ax.set_yticks([0,20,40,60,80,100]);ax.set_ylabel('Success (%)')
    ax.grid(axis='y',color='#E5E8EB',lw=.7,zorder=0)
    ax.set_title('A   Success by suite',loc='left',fontweight='bold',pad=25)
    ax.legend(loc='upper left',bbox_to_anchor=(-.015,1.09),frameon=False,ncol=3,
        handlelength=1.3,columnspacing=1.8,fontsize=9)
    for spine in ('top','right'):ax.spines[spine].set_visible(False)
    ax.tick_params(axis='both',length=3,color='#68737D')

    positions=[2,1,0]
    for y,row in zip(positions,contrast_rows):
        estimate=row['difference_pp'];low=row['task_cluster_ci_low_pp'];high=row['task_cluster_ci_high_pp']
        marker='s' if row['contrast_id']=='supplementary_goal' else 'o'
        color=COLORS[row['better']]
        effect.errorbar(estimate,y,xerr=[[estimate-low],[high-estimate]],fmt=marker,
            color=color,ecolor=color,markersize=7,elinewidth=2,capsize=4,capthick=1.4,zorder=3)
        effect.text((low+high)/2,y+.22,f"{estimate:+.0f} pp  [{low:+.0f}, {high:+.0f}]",
            ha='center',va='bottom',fontsize=9,color='#27313D')
    effect.axvline(0,color='#6C7782',lw=1,ls=(0,(3,3)),zorder=1)
    effect.axhline(.5,color='#D8DDE2',lw=.8,zorder=0)
    effect.set_ylim(-.55,2.72);effect.set_xlim(-25,20)
    effect.set_yticks(positions,['Spatial: Jac − MSE\nOriginal primary',
        'Long: MSE − Jac\nOriginal primary','Goal: Jac − MSE\nProspective supplement'])
    effect.tick_params(axis='y',length=0,pad=10,labelsize=9)
    effect.set_xticks([-20,-10,0,10,20])
    effect.xaxis.set_major_formatter(FuncFormatter(lambda value,_:'0' if value==0 else f'{value:+.0f}'))
    effect.set_xlabel('Success difference (percentage points)',labelpad=9)
    effect.set_title('B   Registered direction checks',loc='left',fontweight='bold',pad=25)
    effect.text(0,1.035,'Task-cluster 95% CI  ·  all Holm-3 p = 1.00',transform=effect.transAxes,
        ha='left',va='bottom',fontsize=8.8,color='#4D5966')
    for spine in ('top','right','left'):effect.spines[spine].set_visible(False)
    effect.tick_params(axis='x',length=3,color='#68737D')
    fig.text(.065,.078,'Independent confirmation: 600 episodes; each suite and condition has 10 tasks × 5 initial states (31–35).',
        fontsize=9,color='#45515E')
    fig.text(.065,.032,'Counts in A have no error bars. Intervals in B are copied from the formal report; non-reproduction does not imply equivalence.',
        fontsize=8.7,color='#45515E')
    basename=output/'vlm_independent_confirmation'
    for suffix in ('png','pdf','svg'):
        metadata={'CreationDate':None,'ModDate':None} if suffix=='pdf' else {'Date':None} if suffix=='svg' else None
        fig.savefig(basename.with_suffix('.'+suffix),dpi=300,metadata=metadata)
    plt.close(fig)

    caption=('Independent confirmation of the frozen VLM paths on LIBERO. (A) Success proportions for Full, Jac and MSE '
        'on four suites, with exact success counts shown out of 50. Every suite-condition group contains 10 tasks and five '
        'paired initial states (31–35); no independently sampled case error bars are assumed. (B) Success differences in '
        'the two original primary directions and the Goal direction registered prospectively after discovery. Positive '
        'values follow the stated registered direction. The displayed 95% percentile task-cluster bootstrap intervals '
        'were copied from the completed formal report (10,000 replicates; whole tasks retain their five initial states). '
        'All three one-sided exact paired tests have Holm-three adjusted p = 1.00. The original Spatial/Long claim and '
        'the supplementary Goal/Long claim were not reproduced; this does not establish path equivalence. No statistics '
        'were rerun or directions reselected to make this figure.')
    (output/'caption.txt').write_text(caption+'\n')
    lines=['本图仅使用本轮已完成的600条正式VLM报告，不重新运行统计或筛选方向。', '',
        'A：四个suite的Full/Jac/MSE成功率，直接标注成功数/50，不附假定案例独立的误差条。',
        'B：原主分析Spatial和Long，以及发现后、独立复验前登记的Goal补充方向；展示正式报告已有的任务聚类95%区间。正值顺着预定方向。三方向Holm3均为1；原主假设及补充假设未复现，不意味着路径等价。', '',
        '可导出文件：`vlm_independent_confirmation.png`（300 dpi）、`.pdf`和`.svg`；CSV为绘图的全部数据。英文完整图注见`caption.txt`。', '',
        '| 条件 | 成功数 | 执行动作数 | 策略请求数 | policy ms/执行动作 | RPC ms/执行动作 |',
        '|---|---:|---:|---:|---:|---:|']
    for row in costs:
        lines.append(f"| {row['label']} | {row['successes']}/{row['n']} | {row['executed_actions']} | {row['policy_requests']} | {row['policy_ms_per_executed_action']:.3f} | {row['rpc_ms_per_executed_action']:.3f} |")
    lines += ['',
        '该成本表统一使用正式报告全部200个episode的累计policy/RPC毫秒数，除以累计实际执行动作数，保留成功与失败案例；不是逐case比值的均值。policy沿用本轮VLM报告的策略调用墙钟字段；RPC为客户端往返。没有对历史native计时作跨实验等同。',
        '不同路径产生的轨迹、动作数、冷启动和外部争用可能不同。数值不是独占GPU速度、质量非劣性或闭环加速的证明；episode墙钟之和也不是并行总耗时。', '',
        '复现（在benchmark根目录运行）：', '',
        '```bash', 'python3 reports/ease_insights_20261103/vlm_confirmation_figures/generate_figures.py', '```', '',
        '输入报告、manifest/selection身份及输出SHA见`provenance.json`。原主分析的Holm2没有改变；图B统一呈现另行登记的三检验家族Holm3。']
    (output/'README_zh.md').write_text('\n'.join(lines)+'\n')
    for path,old_sha in inputs.items():
        assert sha(path)==old_sha, 'Input changed during plotting: '+str(path)
    outputs=sorted(p for p in output.iterdir() if p.is_file() and p.name!='provenance.json')
    provenance=dict(status='complete',purpose='Static export of existing formal statistics; no inferential recomputation',
        source_reports=[dict(path=str(p),sha256=value) for p,value in inputs.items()],
        manifest_hash=main['manifest_hash'],selection_hash=main['selection_hash'],
        original_primary_conclusion=main['conclusion'],
        supplementary_goal_long_conclusion=supp['supplementary_goal_long_analysis']['conclusion'],
        valid_episodes=600,paired_initial_states=list(range(31,36)),matplotlib_version=matplotlib.__version__,
        confidence_intervals_recomputed=False,directions_reselected=False,error_bars_in_panel_a=False,
        panel_b_interval='Previously reported paired task-cluster percentile 95% CI, 10 tasks with five initial states each',
        cost_aggregation='Ratio of cumulative policy/RPC time to cumulative actual executed actions across all 200 cases per condition',
        source_field_mapping=dict(panel_a='primary.condition_summaries[level=suite].success_rate and successes/n',
            panel_b='supplementary.contrasts[].success_difference and inference.confidence_intervals.task_cluster',
            cost_table='primary.condition_summaries[level=all].{policy.total_ms,rpc.total_ms,actions}'),
        outputs=[dict(path=str(p),sha256=sha(p)) for p in outputs])
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(status='complete',output=str(output),files=[p.name for p in outputs],
        policy_ms_per_action={r['label']:r['policy_ms_per_executed_action'] for r in costs},
        rpc_ms_per_action={r['label']:r['rpc_ms_per_executed_action'] for r in costs})))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--primary',type=Path,default=WORKSPACE/'reports/ease_insights_20261103/vlm_confirmation/analysis.json')
    parser.add_argument('--supplementary',type=Path,default=WORKSPACE/'results/ease_insights_20261103/vlm_confirmation_supplementary_analysis/supplementary_analysis.json')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    primary_path=args.primary.resolve();supplementary_path=args.supplementary.resolve()
    inputs={path:sha(path) for path in (primary_path,supplementary_path)}
    main_report,supp=load(primary_path,supplementary_path)
    export(main_report,supp,output,inputs)


if __name__=='__main__':main()
