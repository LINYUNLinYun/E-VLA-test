"""Final Chinese reports and case/request tables; no policy or GPU mutation."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records,summary,pairing
from .manifest import SUITES,cases


def table(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def requests(groups):
    rows=[];routes={}
    for cid,episodes in groups.items():
        stats=Counter(); reasons=Counter(); blocked=Counter(); lengths=Counter(); first=Counter(); runs=Counter();modules=Counter();diagnostics=[];gpu_cost=defaultdict(lambda:dict(policy_ms=0.,actions=0,episodes=0))
        for episode in episodes:
            directory=Path(episode['_directory']);side=directory/'engineering_revision.json'
            binding=json.loads(side.read_text()) if side.exists() else episode.get('engineering_binding',{})
            gpu=str(binding.get('model_gpu','unknown'));gpu_cost[gpu]['episodes']+=1;gpu_cost[gpu]['policy_ms']+=episode['policy_wall_s']*1000;gpu_cost[gpu]['actions']+=episode['actions']
            previous=Counter()
            for line in (Path(episode['_directory'])/'infer.jsonl').read_text().splitlines():
                r=json.loads(line); route=r['route'];delta=Counter(r['counts']);delta.subtract(previous);previous=Counter(r['counts'])
                modules.update({k:v for k,v in r['timing'].items() if k.endswith('_ms') and isinstance(v,(int,float))})
                stats['requests']+=1;stats['full_calls']+=delta['full'];stats['verify_calls']+=delta['verify'];stats['draft_calls']+=delta['draft'];stats['kalman_proposals']+=delta['kalman']
                stats['kalman_executed_requests']+=route['path']=='kalman_verified';stats['kalman_executed_actions']+=r['executed_length'] if route['path']=='kalman_verified' else 0
                stats['diagnostic_verify_calls']+='padding_diagnostic' in route
                if 'padding_diagnostic' in route:diagnostics.append(dict(case=episode['case'],step=r['step'],replan=r['replan'],details=route['padding_diagnostic'],excluded_ms=r['timing'].get('diagnostic_excluded_ms',0),infer_path=str(directory/'infer.jsonl'),infer_sha256=hashlib.sha256((directory/'infer.jsonl').read_bytes()).hexdigest()))
                stats['spatial_zero']+=route.get('spatial_length',-1)==0;stats['verify_gripper']+=bool(route.get('verify_gripper'));stats['draft_gripper_truncation']+=bool(route.get('post_verify_gripper'))
                reasons[str(route.get('full_reason'))]+=1;blocked.update(route.get('gate_blocked',[]));lengths[r['executed_length']]+=1
                if route.get('first_exceed') is not None:first[str(route['first_exceed'])]+=1
                if 'longest_exceed' in route:runs[str(route['longest_exceed'])]+=1
                rows.append(dict(condition=cid,suite=episode['case']['suite'],task=episode['case']['task'],init=episode['case']['init'],replan=r['replan'],step=r['step'],path=route['path'],full_reason=route.get('full_reason'),
                    executed_length=r['executed_length'],accepted_length=route.get('accepted_length'),spatial_length=route.get('spatial_length'),full_calls=delta['full'],verify_calls=delta['verify'],draft_calls=delta['draft'],kalman_proposals=delta['kalman'],
                    first_exceed=route.get('first_exceed'),exceed_count=route.get('exceed_count'),longest_exceed=route.get('longest_exceed'),verify_gripper=route.get('verify_gripper'),draft_gripper_truncation=route.get('post_verify_gripper'),gate_blocks='|'.join(route.get('gate_blocked',[])),
                    policy_ms=r['timing']['policy_wall_ms'],diagnostic_excluded_ms=r['timing'].get('diagnostic_excluded_ms',0),rpc_ms=r['rpc_ms'],seed=r['seed'],verification_seed=r['verification_seed'],
                    draft_noise_hash=r.get('draft_noise_hash'),full_noise_hash=r.get('full_noise_hash'),verify_noise_hash=r.get('verify_noise_hash'),infer_path=str(Path(episode['_directory'])/'infer.jsonl')))
        routes[cid]=dict(stats,full_reasons=dict(reasons),gate_blocks=dict(blocked),executed_lengths=dict(lengths),first_exceed=dict(first),longest_exceed=dict(runs),modules_ms_sum=dict(modules),physical_gpu_cost=dict(gpu_cost),padding_diagnostics=diagnostics)
    return rows,routes


def split(rows,which):
    if not rows:return []
    first={tuple(c[k] for k in ('suite','task','init')) for c in cases(rows[0]['config']['model'])}
    return [r for r in rows if (tuple(r['case'][k] for k in ('suite','task','init')) in first)==(which=='first80')]


def paths_figure(exp,cids,groups,base,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(18,9))
    model='xiaomi' if exp=='E4' else 'pi0';init=0 if model=='xiaomi' else 6
    for index,(task,suite) in enumerate((t,s) for t in (0,5) for s in SUITES):
        ax=fig.add_subplot(2,4,index+1,projection='3d')
        for cid in [base]+cids:
            row=next((r for r in groups[cid] if r['case']==dict(suite=suite,task=task,init=init)),None)
            if row is None:continue
            states=[json.loads(line) for line in (Path(row['_directory'])/'telemetry.jsonl').read_text().splitlines()]
            xyz=np.asarray([r['eef_pos'] for r in states]);ax.plot(*xyz.T,label=cid,linewidth=1)
            ax.scatter(*xyz[0],marker='o',s=8);ax.scatter(*xyz[-1],marker='x',s=15)
        ax.set_title(f'{suite} task{task} init{init}');ax.set_xlabel('World X / m');ax.set_ylabel('World Y / m');ax.set_zlabel('World Z / m')
    handles,labels=fig.axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',fontsize=7,ncol=2)
    fig.suptitle(exp+' actual EEF XYZ; fixed case keys; no temporal padding/interpolation');fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(destination,dpi=150);plt.close(fig)


def main():
    groups=records();manifest=json.loads((OUT/'manifest.json').read_text());configs={c['id']:c for c in manifest['conditions']}
    for name in ('bounded_followups','confirm300'):
        path=OUT/'stages'/name/'manifest.json'
        if path.exists():configs.update({c['id']:c for c in json.loads(path.read_text())['conditions']})
    dest=OUT/'reports';dest.mkdir(exist_ok=True)
    from .campaign_wall import run as campaign_wall
    campaign_timing=campaign_wall(groups)
    request_rows,routing=requests(groups);table(dest/'requests.csv',request_rows);atomic(dest/'routing.json',routing)
    episode_rows=[]
    for cid,rows in groups.items():
        for r in rows:episode_rows.append(dict(condition=cid,**r['case'],success=r['success'],actions=r['actions'],predictions=r['predictions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s'],manifest_hash=r['manifest_hash'],result_path=str(Path(r['_directory'])/'result.json'),telemetry_path=str(Path(r['_directory'])/'telemetry.jsonl')))
    table(dest/'episodes.csv',episode_rows)
    task_rows=[];gpu_rows=[];module_rows=[]
    for cid,rows in groups.items():
        for suite in SUITES:
            for task in range(10):
                own=[r for r in rows if r['case']['suite']==suite and r['case']['task']==task]
                if own:
                    s=summary(own);task_rows.append(dict(condition=cid,suite=suite,task=task,n=s['n'],successes=s['successes'],actions=s['actions'],policy_wall_s=s['policy_wall_s'],observed_ms_per_action=s['ms_per_action'],timeouts=sum(r['termination']=='timeout' for r in own)))
        for gpu,values in routing[cid]['physical_gpu_cost'].items():gpu_rows.append(dict(condition=cid,physical_gpu=gpu,**values,observed_ms_per_action=values['policy_ms']/values['actions']))
        actions=sum(r['actions'] for r in rows)
        for module,ms in routing[cid]['modules_ms_sum'].items():module_rows.append(dict(condition=cid,module=module,sum_ms=ms,ms_per_actual_action=ms/actions,scope='Overlapping timing fields are not additive; descriptive science cost, not matched promotion ratio'))
    table(dest/'tasks.csv',task_rows);table(dest/'cost_by_physical_gpu.csv',gpu_rows);table(dest/'module_times.csv',module_rows)
    technical=[]
    for p in OUT.glob('*/episodes/*/task*/init*/technical_error.json'):
        technical.append(dict(path=str(p),**json.loads(p.read_text()),scientific_complete=(p.parent/'complete.json').exists()))
    atomic(dest/'execution_status_inventory.json',dict(time=time.time(),valid_science=len(episode_rows),technical_attempts=technical,
        planned_conditions_without_valid_cases=[cid for cid in configs if not groups[cid]],
        archived_supervisor_alerts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),receipt=json.loads(p.read_text())) for p in OUT.glob('campaign_alert*.json') if p.name!='campaign_alert.json'],
        original_service_logs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for pattern in ['workers/gpu*/service.log','stages/*/gpu*/service.log'] for p in OUT.glob(pattern)],
        service_log_scope='Raw startup/dependency/readiness and service recovery logs are retained; these are technical execution history, not scientific quality failures.',
        semantics='Technical attempts are retained, not scientific failures. Every valid timeout remains a scientific failure. Engineering nested directories are excluded from all scientific denominators.'))
    selection_path=OUT/'stages/confirm300/manifest.json';selection=json.loads(selection_path.read_text()).get('decision',{}) if selection_path.exists() else {}
    cp=OUT/'cost_gate_final.json';cost=json.loads(cp.read_text()) if cp.exists() else {'conditions':{}}
    from .compare_paths import run as compare_e4_paths
    e4_comparison=compare_e4_paths()
    from .probe_history import run as trace_probes
    probe_history=trace_probes()
    from .matched_stage_costs import run as matched_confirmation_costs
    matched_costs=matched_confirmation_costs()
    confirmation={}
    for exp,decision in selection.items():
        base='xiaomi_pure_d2' if exp=='E4' else 'pi0_parent'
        for cid in decision.get('selected',[]):
            a,b=summary(groups[cid]),summary(groups[base]);confirmation[cid]=dict(experiment=exp,baseline=base,n=a['n'],baseline_n=b['n'],successes=a['successes'],baseline_successes=b['successes'],
                net_successes=a['successes']-b['successes'],formal300_complete=a['n']==b['n']==300,
                within_7_net_loss_count=a['n']==b['n']==300 and a['successes']>=b['successes']-7,
                improves_success_count=a['n']==b['n']==300 and a['successes']>b['successes'],
                statistical_scope='Predeclared count threshold only; not a population noninferiority proof. No unmatched300 speed claim.')
    atomic(dest/'confirmation_decision.json',confirmation)
    ablations=[]
    pairs=[(cid,'E1_pure_d1') for cid,c in configs.items() if c['experiment']=='E1' and c.get('mode')!='pure']
    pairs += [(cid,cid.removesuffix('_no_c')) for cid in configs if cid.endswith('_no_c')]
    pairs += [('E3_alpha1.25','E3_hard0.125'),('E3_alpha1.5','E3_hard0.15')]
    pairs += [(cid,reference) for cid in selection.get('E4',{}).get('selected',[]) for reference in ('E4_periodic','xiaomi_full_d2')]
    for left,right in pairs:
        if groups[left] and groups[right]:ablations.append(dict(experiment=configs[left]['experiment'],candidate=left,reference=right,**pairing(groups[left],groups[right])))
    atomic(dest/'ablation_pairing.json',ablations)
    for exp in ('E1','E2','E3','E4'):
        cids=[cid for cid,c in configs.items() if c['experiment']==exp];base='xiaomi_pure_d2' if exp=='E4' else 'pi0_parent'
        lead={'E1':'四个D1+verify候选均未达到预登记质量门槛，按协议停于各80条；单步Euler实现已通过原生数值等价门禁。',
              'E2':'九个a/b/c主门控候选没有同时满足质量与可信至少10%降本门槛，未扩300；最终部署姿态阈值仍未确定。No-c消融的单例恢复与运行历史限制分别报告。',
              'E3':'两软规则均73/80，处于允许净少2条的质量边界；α1.25降本不足，α1.5受GPU0冷态混杂而成本不确定，两者均未扩300。'}
        if exp=='E4':
            ss=summary(groups['E4_tau0.8']);bs=summary(groups[base]);ps=summary(groups['E4_periodic']);fs=summary(groups['xiaomi_full_d2'])
            lead[exp]=f"质量分支τ=.8为{ss['successes']}/{ss['n']}，pure为{bs['successes']}/{bs['n']}，Full-D2也为{fs['successes']}/{fs['n']}，无verify周期对照为{ps['successes']}/{ps['n']}。与周期对照实际执行前缀逐位一致{e4_comparison['executed_all_exact']}/{e4_comparison['n']}；未证实verifier额外成功增益，也不能把相对pure的改善唯一归因P2。成本不使用混卡300整体速度比。"
        text=[f'# idea1 v3.3 {exp} 实验报告','',lead[exp],'',f'有效科学回合按原始官方案例键计数；探索初筛80条，入选后首80与新增220合成300。该结果不是官方50初态完整LIBERO基准。',
            '',f"整个研究从首真实episode到末科学complete的观察wall为{campaign_timing['elapsed_s']/3600:.3f}小时，聚合科学吞吐{campaign_timing['observed_aggregate_science_episodes_per_hour']:.2f} episode/小时，期间含400工程回合、校准等待及GPU租约变化；不构成配置加速比。初始准备/加载和最终CPU报告另计，完整范围见campaign_wall_receipt.json。",
            '',f'共同参考 `{base}`：{summary(groups[base])["successes"]}/{len(groups[base])}。下表的每动作成本为完整策略推理wall累计/实际动作数，含拒绝与Full接管，排除单列padding诊断；它是不同闭环轨迹上的观测成本，不能充作同输入请求加速比。',
            '', '| 条件 | 首80 | 新增220 | 全部 | Spatial/Object/Goal/Long | 每动作ms | task/suite等权成功率 |', '|---|---:|---:|---:|---|---:|---:|']
        for cid in cids:
            rows=groups[cid];s=summary(rows);a=summary(split(rows,'first80'));b=summary(split(rows,'new220'));suites='/'.join(f"{s['suites'][suite]['successes']}/{s['suites'][suite]['n']}" for suite in SUITES)
            text.append(f"| {cid} | {a['successes']}/{a['n']} | {b['successes']}/{b['n']} | {s['successes']}/{s['n']} | {suites} | {s['ms_per_action']:.3f} | {100*s['suite_task_macro_success']:.2f}% |" if s['n'] else f'| {cid} | 未启动 | — | 0/0 | — | — | — |')
        text+=['','共同基线分别列示（不同D不混作同一Full）：','', '| 基线 | 成功 | 观测ms/action | episode/worker小时 |','|---|---:|---:|---:|']
        for bid in (['xiaomi_full_d2','xiaomi_pure_d2'] if exp=='E4' else ['pi0_full_d10','pi0_full_d2','pi0_pure_d2','pi0_parent']):
            s=summary(groups[bid]);text.append(f"| {bid} | {s['successes']}/{s['n']} | {s['ms_per_action']:.3f} | {s['episodes_per_worker_hour']:.2f} |")
        text+=['','## 同卡成本晋级门禁','', '以上科学wall是原始观察值；原父基线存在旧EGL跨卡渲染重叠和首任务运行历史差异，不能直接作为同卡提速证据。独立工程对照不增加科学样本，保留完整wall主口径及统一首/次init、step0封顶和配对启动敏感性。GPU0父输入准备异常与热态结果冲突时成本不确定，禁止凭原始比率晋级。',
            '', '额外工程预算为三卡π0父80×3加两卡Xiaomi pure80×2，共400有效工程回合；全部嵌套目录，不增加科学分母。Xiaomi候选逐物理卡匹配：τ=.8使用GPU2 pure，τ=1.6/其他GPU3条件使用GPU3 pure；legacy跨EGL或混合绑定不给速度资格。',
            '', 'GPU2工程父Long task04 init6曾一次技术OOM后按预声明上限重试，有效80才计主分母；invalid attempt完整保留。该次失败准备可能改变缓存/冷热历史，因此GPU2的21.5986ms/action参考与E2临界9.11%下降不作精确因果速度解释。',
            '', '收据：[preliminary Pi screen](../cost_gate_pi0_screen.json)、[final cost gate](../cost_gate_final.json)、[工程逐GPU证据](../engineering_cost_rebaseline/)、[原始重叠时间线](../engineering_cost_rebaseline/legacy_timing_confound.json)。',
            '', '| 条件 | 物理GPU | 匹配candidate/parent成本比 | 验证状态 | 速度门槛合格 |','|---|---:|---:|---|---|']
        for cid in cids:
            entry=cost['conditions'].get(cid,{});ratio=entry.get('matched_cost_ratio')
            text.append(f"| {cid} | {entry.get('physical_gpu','—')} | {ratio:.5f} | {entry.get('verification','未决')} | {entry.get('eligible_cost_gate',False)} |" if ratio is not None else f"| {cid} | {entry.get('physical_gpu','—')} | — | {entry.get('verification','未决')} | False |")
        text+=['','## 选择与配对统计','',f'冻结选择：`{json.dumps(selection.get(exp,{}),ensure_ascii=False)}`。无合格候选时按规则停止扩样，不能把未扩样说成协议遗漏。',
            '', '| 条件 vs 共同参考 | 配对n | 恢复 | 新失败 | task/suite等权Δ | 分层task簇bootstrap95% |', '|---|---:|---:|---:|---:|---|']
        for cid in cids:
            p=pairing(groups[cid],groups[base])
            if p['n']:text.append(f"| {cid} | {p['n']} | {p['recoveries']} | {p['new_failures']} | {100*p['delta']:+.2f}pp | [{100*p['ci95'][0]:+.2f},{100*p['ci95'][1]:+.2f}]pp |")
        for cid,item in confirmation.items():
            if item['experiment']==exp:
                text+=['',f"300条确认 `{cid}`：{item['successes']}/{item['n']} vs `{item['baseline']}` {item['baseline_successes']}/{item['baseline_n']}，净成功{item['net_successes']:+d}；完整300={item['formal300_complete']}，净少≤7计数门槛={item['within_7_net_loss_count']}，成功数高于参考={item['improves_success_count']}。这仍不是总体非劣或verifier因果收益的统计证明。"]
        own_ablations=[r for r in ablations if r['experiment']==exp]
        if own_ablations:
            text+=['','必要消融直接配对（candidate减reference）：','', '| candidate / reference | 配对n | 恢复 | 新失败 | task/suite等权Δ | bootstrap95% |','|---|---:|---:|---:|---:|---|']
            for p in own_ablations:text.append(f"| {p['candidate']} / {p['reference']} | {p['n']} | {p['recoveries']} | {p['new_failures']} | {100*p['delta']:+.2f}pp | [{100*p['ci95'][0]:+.2f},{100*p['ci95'][1]:+.2f}]pp |")
        text+=['','bootstrap按suite分层重采样task簇，保留init，2000次seed3301；初筛和扩样共享首80，候选选择及多重比较未作校正。300条的偶数task多一个预定init，因此同时报告原始成功数和task/suite等权率。',
            '', '## 路由与成本','', '| 条件 | 请求 | Full调用 | verify调用 | Kalman提案/实际执行请求 | 空间零接受 | 执行长度均值 |', '|---|---:|---:|---:|---:|---:|---:|']
        for cid in cids:
            r=routing.get(cid,{})
            if not r:continue
            mean=sum(int(k)*v for k,v in r['executed_lengths'].items())/r['requests']
            text.append(f"| {cid} | {r['requests']} | {r['full_calls']} | {r['verify_calls']} | {r['kalman_proposals']}/{r['kalman_executed_requests']} | {r['spatial_zero']} | {mean:.3f} |")
        text+=['','逐模块累计毫秒和按物理GPU分层的wall/动作分母见routing.json；多卡300条不混成跨GPU速度比。每请求成本见requests.csv；完整评测吞吐按包含reset、仿真、RPC与落盘的episode wall报告。模块耗时、固定请求测速、闭环每动作成本和完整评测吞吐是不同指标，不互相替代。',
            '', '| 条件 | requests/episode | wall ms/request | episode/worker小时 | mean actions | 有效超时失败 |','|---|---:|---:|---:|---:|---:|']
        for cid in cids:
            s=summary(groups[cid])
            if s['n']:text.append(f"| {cid} | {s['requests']/s['n']:.2f} | {s['policy_wall_s']*1000/s['requests']:.3f} | {s['episodes_per_worker_hour']:.2f} | {s['mean_steps']:.2f} | {sum(r['termination']=='timeout' for r in groups[cid])} |")
        bankmodel='xiaomi' if exp=='E4' else 'pi0';banks=list((OUT/'request_bank').glob(bankmodel+'_gpu*.json'))
        if banks:
            bank=json.loads(banks[0].read_text());text+=['','## 辅助固定输入请求延迟','',f"物理GPU{bank['physical_gpu']}，B1，共{bank['n']}个保存的真实输入；同输入与seed，先暖各prompt/路径，模型加载{bank['model_load_s']:.2f}s、暖机{bank['warmup_s']:.2f}s另列。设备UUID、环境、bank/source及100项输入hash见[{banks[0].name}](../request_bank/{banks[0].name})。",'', '| 后端路径 | mean wall ms/input |','|---|---:|']
            for name,item in bank['summary'].items():text.append(f"| {name} | {item['mean_ms']:.3f} |")
            text+=['',f"每输入先统一建立prompt/runtime及Full输入，平均共同setup为{bank['common_input_setup_mean_ms']:.3f}ms，单列且未算入任一路径，以免首路径独占准备成本。随后每条prepare+sample路径仍包含自身图像/状态变换、vision/VLM/sample；verify行使用该输入另建的Full KV及其draft，不模拟历史缓存年龄。该表没有环境、RPC、门控或Full接管，属于辅助同输入后端延迟；不能替代闭环主成本晋级，也不能称完整评测吞吐。首轮固定顺序混入准备偏差的结果保留在attempt1_fixed_order，已被本次口径替代。"]
        if exp!='E4':
            text+=['','原父与后续resident工程父的成功标签80/80相同，但完整raw/decoded/路由只有78/80逐位一致；Spatial task0 init6/7存在可观差异，同输入hash、seed/noise仍同。三GPU工程结果彼此一致，差异与首服务native probe/runtime构造历史分组相关，具体算子原因未知，不归因为硬件数值漂移。以下固定两例排除仅作质量敏感性，正式80/300均不删除：',
                '', '| 条件（首80） | 固定排除后配对n | 恢复 | 新失败 | 正式80质量门槛 | 固定78诊断门槛 |','|---|---:|---:|---:|---|---|']
            for cid in cids:
                rows=[r for r in split(groups[cid],'first80') if not (r['case']['suite']=='libero_spatial' and r['case']['task']==0 and r['case']['init'] in (6,7))]
                p=pairing(rows,split(groups[base],'first80'))
                full=sum(r['success'] for r in split(groups[cid],'first80'))>=sum(r['success'] for r in split(groups[base],'first80'))-2
                diagnostic=p['recoveries']-p['new_failures']>=-2
                if p['n']:text.append(f"| {cid} | {p['n']} | {p['recoveries']} | {p['new_failures']} | {full} | {diagnostic} |")
            text+=['','固定78诊断沿用净少≤2的计数参照，仅展示结论敏感性；它不是新选型规则，不推翻正式80资格，也不将差异归因于算法。']
        if exp=='E1':text+=['','D1为原生一次Euler更新1→0；额外真实输入数值门禁D1_native_euler_probe.json将adapter D1与released native.record(V14,num_steps=1)同obs/seed对比，raw与decoded逐位一致（最大差0），输入/源码/输出数组hash及既定atol/rtol均保存。E1质量不达门槛不能归因于未正确实现单步原语。verify时间仍为0.10/0.05，tau为normalized前六维RMS空间半径。Full原因、逐动作首次超限位置、超限数和连续超限段见routing.json/requests.csv。这些描述路由触发因素，不是任务失败的因果诊断。']
        if exp=='E2':text+=['','c使用真实EEF世界xyzw四元数最近五个已完成控制步的最大两两最短弧角，与a/b及共同约束AND，严格小于候选θ_c才允许尝试Kalman。2°/5°/10°在父80状态/命令时点分布上冻结为探索候选，最终推荐阈值由闭环质量/成本决定；不称已知部署阈值。b为已解码环境控制命令的平移变化，既非模型normalized action也非真实EEF位移。',
            '', 'Kalman仅预测XYZ，旋转/夹爪保持最后实际命令，经当前decoder逆映射到raw32，padded25维零。首真实对照包含零padding/Full尾维诊断并将其额外verify时间与调用单列（E2_padding_diagnostic_receipt.json）；初筛3次诊断的前六维RMS最大变化约0.00150/0.00150/0.000764，包括后续NoC在内的全部诊断与排除wall见routing.json的padding_diagnostics。非模型metadata准备的vision/VLM调用实测为零。门控只是允许尝试，不证明可忽略视觉。']
        if exp=='E2':
            text+=['','交替/移除c是预登记的控制与必要消融，不能因其出现成本阳性而升格为a/b/c主门控候选；未扩300不是遗漏。最终部署θ_c仍未确定，无主候选满足可信质量—成本联合门槛时不推荐部署阈值。']
            gate_audit=json.loads((OUT/'E2_gate_audit.json').read_text())
            text+=['','实际历史重放的门控审计如下；a/b/c总阻止可以重叠，不能把c总次数解释为其独立贡献。“仅c”只表示其他a/b/共同历史条件通过，尚未扣除禁连续Kalman，仍不是任务成败因果效应。',
                '', '| 条件 | 决策点 | a阻止 | b阻止 | c阻止 | 仅c | gate全过 | 实际KF动作/全部动作 |','|---|---:|---:|---:|---:|---:|---:|---:|']
            for cid in cids:
                g=gate_audit['conditions'].get(cid)
                if not g:continue
                blocks=g['individual_blocks'];joint=g['joint_stats'];k=joint.get('Kalman_executed_actions',0);total=g['total_executed_actions']
                text.append(f"| {cid} | {g['decisions']} | {blocks.get('a_turn',0)} | {blocks.get('b_change',0)} | {blocks.get('c_pose',0)} | {g['sole_blocker'].get('c_pose',0)} | {joint.get('gate_all_pass',0)} | {k}/{total} ({100*k/total:.2f}%) |")
            if (OUT/'bounded_probe_history_h1.json').exists():text+=['','No-c shard1首次探针的Full/V14原生与metadata等价均通过，但其Task1→Task0重建历史与原父Task0-first历史不同，旧参考门禁阻止了两次技术attempt，尚无科学动作。恢复H1事先固定已冻结GPU2工程NEW参考，对同一次实际输出要求raw/decoded严格bitwise；旧父差异保留非fatal历史诊断，不放宽容差、不运行时选参考、不改变NoC科学配置。注册与执行收据为bounded_probe_history_h1*.json。']
            for item in probe_history['no_c_comparisons']:
                p=item['probe_prompt_sensitivity'];text+=['',f"No-c原始完整80比较见必要消融表；其各shard首服务probe对应prompt案例固定并列在probe_prompt_history_receipt.json。本轮排除清单 `{json.dumps(item['excluded_fixed_probe_prompt_cases'])}` 仅作敏感性，剩余n={p['n']}，恢复{p['recoveries']}、新失败{p['new_failures']}，task/suite等权Δ={100*p['delta']:+.2f}pp。两条件初始Full10 raw/decoded逐位一致{item['initial_Full10_exact_count']}/{len(item['initial_Full10_comparison'])}例，其余案例逐键保存，不扩大为全80一致。正式80不删，若差异集中于首prompt，不能把结果解释为c的独立因果作用。"]
            if (OUT/'E2_no_c_case_mechanism_receipt.json').exists():text+=['','唯一配对结局差异是Goal task9 init7：前7次整proposal与step0..35真实telemetry均逐位一致；replan7/step35输入hash、采样与验证seed相同，世界姿态窗角10.71237°。c10仅因c_pose走V14，NoC允许Kalman；后者159步成功，前者300步超时。E2_no_c_case_mechanism_receipt.json保存该事后挑出的单案例机制，不是预定代表曲线，不将单例外推为总体c因果或部署阈值。']
        if exp=='E3':text+=['','软规则允许最多一个轻微超限，所有动作仍为原序连续前缀；每步≤ατ、整段Σe²≤Lτ²。增加接受率本身不作为质量成功判据；硬τ=.125/.15是相应半径放宽消融。夹爪规则保持父系统strict。']
        if exp=='E4':text+=['','所有Xiaomi条件均不启用夹爪保护；startup/periodic/空间零接受之外不因夹爪接管。P2只有累计两轮draft后才周期Full；拒绝后立即Full会重置draft轮数。',
            '', '首80六阈值网格仅最高边界τ=.8位于观察前沿，按预登记规则追加一次τ=1.6的80条；最低边界未触发τ=.0125，未继续无界扩网格。此为有界探索触发，不把原始跨GPU前沿当成速度资格。',
            '', 'τ=.025首80与Full-D2全部80案例的每轮decoded动作及执行长度逐位相等（E4_tau0025_full_exact_receipt.json），构成Full动作路径加额外draft/verify开销。低τ的质量不能归因于充分混合使用；必须结合实际正接受与周期Full次数解读。Xiaomi normalized空间阈值不能与π0直接比较。',
            '', '首80对照：τ=.8与无verify的periodic均78/80，80/80案例的动作、Full/draft节奏与成败完全相同（460 Full、823 draft）；823/823 verified draft均完整接受10。该批数据未显示verifier过滤的额外质量贡献，不把与pure的差异唯一归因P2或Full容量。τ=.8在GPU2、periodic在GPU3，wall差2.796ms/action（18.48%）仅跨卡观察，不能称同卡verifier净开销或外推300。τ=.8自身verify AE CUDA-event共35,904.8ms/12,468动作=2.880ms/action，已计policy wall；它不含全部CPU/同步开销，亦非总净策略开销。设备与范围见E4_timing_device_scope_receipt.json；首80动作事实见E4_tau08_periodic_exact_receipt.json，其早期因果措辞由E4_causal_interpretation_receipt.json更正。τ=.8/1.6边界等价见E4_tau08_tau16_exact_receipt.json。',
            '', f"全部已完成共有案例重新比较：n={e4_comparison['n']}，完整proposal数组逐位相等{e4_comparison['proposal_all_exact']}、实际执行前缀相等{e4_comparison['executed_all_exact']}、Full/draft节奏相等{e4_comparison['rhythm_all_exact']}、成功标签相等{e4_comparison['outcomes_equal']}；{e4_comparison['verified_full10_acceptances']}/{e4_comparison['verified_draft_proposals']}个verified draft接受完整10动作。完整逐案hash与差异见E4_tau08_periodic_all_shared_comparison.json，扩样结论使用此实测结果，不从首80推断。"]
        if exp=='E4':text+=['','唯一差异为Goal task3 init6，同物理GPU0、同模型服务109338：input0与request0输入hash/seed/noise/raw/actions全相同，telemetry step0..10也逐位一致。但request1/step10的完整观测hash已经不同，seed/noise仍同，draft raw最大差0.015625；这是该episode第一次verify之前已生成的draft，之后仍全接受10。哪项未保存观测不同及其原因未知，不能把该例失败归因verifier接受机制，也不宣称300条完全等价或普遍无模型副作用。E4_single_case_difference_receipt.json保存首次分歧与服务/RNG来源。',
            '', '全300中Full-D2与τ=.8均293成功，两者配对为5恢复/5新失败；periodic294、pure282。τ=.8相对pure净+11不能唯一归因P2，Full独立也达到同一总数，且有首80选型及多重比较偏差。当前没有显示verifier独立质量增益的证据；这不是普遍因果定理。完整边界见E4_causal_interpretation_receipt.json。']
        if exp=='E4':
            text+=['','新增220的同初态、同物理model/render GPU闭环成本配对如下（原始首80跨卡/legacy绑定不混入；固定条件顺序与冷态历史仍是限制，不称同输入请求加速或完全识别的净开销）：',
                '', '| τ=.8 vs reference | GPU | matched新案例n | candidate ms/action | reference ms/action | 比率 |','|---|---:|---:|---:|---:|---:|']
            for p in matched_costs['rows']:
                if p['cohort']=='new220':text.append(f"| {p['reference']} | {p['physical_gpu']} | {p['n']} | {p['candidate_policy_ms_per_action']:.3f} | {p['reference_policy_ms_per_action']:.3f} | {p['ratio']:.4f} |")
            text+=['','每组实际动作分母、逐case结果hash、排除绑定键及init≥3热态敏感性见matched_confirmation_costs.json；敏感性不改变正式300质量计数。']
        representative=selection.get(exp,{}).get('selected',[])
        if exp=='E1' and not representative:representative=[cid for cid in ['E1_D1_tau0.1','E1_pure_d1'] if groups[cid]]
        if exp=='E2' and not representative:
            bounded=OUT/'stages/bounded_followups/manifest.json'
            source=json.loads(bounded.read_text()).get('decision',{}).get('E2_no_c_parent') if bounded.exists() else None
            representative=[cid for cid in [source,source+'_no_c' if source else None] if cid and groups[cid]]
        if exp=='E3' and not representative:representative=[cid for cid in ['E3_alpha1.25','E3_alpha1.5'] if groups[cid]]
        if exp=='E4' and representative:representative=representative[:1]+['E4_periodic']
        if representative:
            figure=dest/(exp+'_fixed_xyz.png');paths_figure(exp,representative[:2],groups,base,figure)
            text+=['','## 真实XYZ与复现','',f'![预定案例的真实EEF XYZ]({figure.name})','', '固定每suite task0/5、π0 init6/Xiaomi init0；未按成败换样。曲线直接来自实际EEF观测，不是命令累加或预测动作误差，未补零/时间拉齐。所有逐步XYZ/姿态与动作可按episodes.csv进入原始telemetry。']
        text+=['','复现入口：原始manifest.json、E2_manifest.json/E2_calibration.json、metadata_revision_r2.json、egl_revision.json及阶段manifest。GPU1于用户授权idea2租约后排除；EGL物理绑定修正通过初始RGB/状态逐位门禁，GPU2迁移后22个零draft案例与Full动作序列逐位一致，其余2个接受draft案例不作该比较。',
            '', '服务模型加载与native probe在策略wall之外，分别保留worker协议和native probe收据；请求内prompt runtime重建仍计入原始策略wall，并在input_processing模块/冷热敏感性单列，不假称全程warm。环境创建、reset、RPC、仿真与落盘保留在bounded_profile.json及逐回合timing中。无全量视频。技术invalid/中断保留并不计成科学失败；有效超时按失败计入，不重试到成功。', '']
        (dest/(exp+'_report_zh.md')).write_text('\n'.join(text))
    inputs=[OUT/name for name in ['manifest.json','E2_manifest.json','E2_calibration.json','metadata_revision_r2.json','egl_revision.json','cost_gate_pi0_screen.json','cost_gate_final.json','E2_padding_diagnostic_receipt.json','D1_native_euler_probe.json','E4_tau08_periodic_all_shared_comparison.json','E4_timing_device_scope_receipt.json','E4_single_case_difference_receipt.json','E4_causal_interpretation_receipt.json','E2_no_c_case_mechanism_receipt.json','probe_prompt_history_receipt.json','bounded_probe_history_h1.json','bounded_probe_history_h1_execution.json','request_bank/timing_order_receipt.json']]
    atomic(dest/'report_receipt.json',dict(time=time.time(),scientific_cases=len(episode_rows),requests=len(request_rows),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ['analyze.py','campaign_wall.py','compare_paths.py','probe_history.py','matched_stage_costs.py','audit.py','costs.py']},
        hashed_inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs if p.exists()],
        reports={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*') if p.is_file() and p.name!='report_receipt.json'}))


if __name__=='__main__':main()
