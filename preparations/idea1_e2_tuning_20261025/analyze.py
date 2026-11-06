"""Small paired6-arm report; no launcher or adaptive parameter selection."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .design import OUT


def records():
    groups=defaultdict(list)
    for p in OUT.glob('*/episodes/*/task*/init08/result.json'):
        if '.invalid.' in str(p) or not p.with_name('complete.json').exists():continue
        r=json.loads(p.read_text());r['_directory']=str(p.parent);groups[r['condition']].append(r)
    return groups


def summary(rows):
    n=len(rows);actions=sum(r['actions'] for r in rows);wall=sum(r['wall_s'] for r in rows)
    return dict(n=n,successes=sum(r['success'] for r in rows),actions=actions,policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in rows)/actions if actions else None,
        predict_only_ms_per_action=1000*sum(r['predict_policy_wall_s'] for r in rows)/actions if actions else None,ack_ms_per_action=1000*sum(r['ack_policy_wall_s'] for r in rows)/actions if actions else None,
        active_episode_wall_s=wall,episodes_per_active_worker_hour=3600*n/wall if wall else None,common_warm_s=sum(r['common_warm_s'] for r in rows),
        suites={s:dict(n=sum(r['case']['suite']==s for r in rows),successes=sum(r['success'] for r in rows if r['case']['suite']==s)) for s in ['libero_spatial','libero_object','libero_goal','libero_10']})


def pairing(rows,ref):
    key=lambda r:tuple(r['case'][k] for k in ['suite','task','init']);right={key(r):r for r in ref};pairs=[(r,right[key(r)]) for r in rows if key(r) in right]
    if not pairs:return dict(n=0)
    for a,b in pairs:
        if a['engineering_binding']['model_gpu']!=b['engineering_binding']['model_gpu']:raise ValueError('Cross-card paired cost')
    a,b=summary([x for x,y in pairs]),summary([y for x,y in pairs]);delta={s:[float(x['success'])-float(y['success']) for x,y in pairs if x['case']['suite']==s] for s in a['suites']}
    rng=np.random.default_rng(20261025);boot=[np.mean([np.mean(rng.choice(v,len(v),replace=True)) for v in delta.values() if v]) for _ in range(2000)]
    gpu={}
    for g in [2,3]:
        own=[(x,y) for x,y in pairs if x['engineering_binding']['model_gpu']==g]
        if own:
            aa,bb=summary([x for x,y in own]),summary([y for x,y in own]);gpu[str(g)]=dict(n=len(own),candidate_ms_per_action=aa['policy_ms_per_action'],reference_ms_per_action=bb['policy_ms_per_action'],ratio=aa['policy_ms_per_action']/bb['policy_ms_per_action'])
    return dict(n=len(pairs),recoveries=sum(x['success'] and not y['success'] for x,y in pairs),new_failures=sum(not x['success'] and y['success'] for x,y in pairs),
        delta_successes=a['successes']-b['successes'],ci95_delta=np.percentile(boot,[2.5,97.5]).tolist(),matched_policy_ratio=a['policy_ms_per_action']/b['policy_ms_per_action'],by_physical_gpu=gpu)


def profile(groups,m):
    existing=OUT/'profiling_receipt.json'
    if existing.exists():return json.loads(existing.read_text())
    selected={tuple(c[k] for k in ['suite','task','init']) for c in m['cases'][:8]};rows=[r for own in groups.values() for r in own if tuple(r['case'][k] for k in ['suite','task','init']) in selected]
    if len(rows)!=48:return None
    by_gpu={}
    for g in [2,3]:
        own=[r for r in rows if r['engineering_binding']['model_gpu']==g];ends=[(Path(r['_directory'])/'complete.json').stat().st_mtime for r in own]
        span=max(ends)-min(end-r['wall_s'] for r,end in zip(own,ends));s=summary(own)
        by_gpu[str(g)]=dict(summary=s,observed_profile_window_s=span,observed_episodes_per_hour=len(own)*3600/span,
            timings_s={k:sum(r['timing'].get(k,0) for r in own) for k in ['environment_acquire_s','reset_s','rpc_s','step_s','ack_s','save_s']})
    data=dict(time=time.time(),profile_episodes=48,profile_cases=8,included_in240=True,by_physical_gpu=by_gpu,
        decision='Continue frozen B1/single environment per service, fresh environment per arm; no infrastructure optimization or threshold changes. Profile ended when predefined8 cases completed; no artificial delay to10minutes.')
    atomic(OUT/'profiling_receipt.json',data);return data


def main():
    m=json.loads((OUT/'manifest.json').read_text());groups=records();stats={c['id']:summary(groups[c['id']]) for c in m['conditions']};profile(groups,m)
    atomic(OUT/'progress.json',dict(time=time.time(),valid=sum(len(v) for v in groups.values()),target=240,conditions=stats))
    if any(s['n']!=40 for s in stats.values()):print('Partial',sum(len(v) for v in groups.values()));return
    pairs={cid:{ref:pairing(rows,groups[ref]) for ref in ['old_c5','alternate','parent'] if cid!=ref} for cid,rows in groups.items()}
    atomic(OUT/'paired_results.json',dict(summary=stats,paired=pairs,seed=20261025,bootstrap=2000,scope='40 paired initial-case screening only, no multiplicity adjustment or deployment claim'))
    with (OUT/'episodes.csv').open('w',newline='') as stream:
        fields=['condition','suite','task','init','success','actions','policy_wall_s','predict_policy_wall_s','ack_policy_wall_s','common_warm_s','wall_s','gpu','directory'];writer=csv.DictWriter(stream,fields);writer.writeheader()
        for cid,rows in groups.items():
            for r in rows:writer.writerow(dict(condition=cid,**r['case'],**{k:r[k] for k in fields if k in r and k not in ['condition']},gpu=r['engineering_binding']['model_gpu'],directory=r['_directory']))
    text=['# E2决策机小规模调参结果','', '固定init8四suite×10task，六臂各40，总240。仅两个残差臂采集新指标；主成本统一predict+ack服务端wall，公用预热单列。该40例仅探索筛选，不证明部署优势。','', '| 条件 | 成功 | policy ms/action | predict-only | ack | warm s |','|---|---:|---:|---:|---:|---:|']
    for cid,s in stats.items():text.append(f"| {cid} | {s['successes']}/40 | {s['policy_ms_per_action']:.3f} | {s['predict_only_ms_per_action']:.3f} | {s['ack_ms_per_action']:.4f} | {s['common_warm_s']:.2f} |")
    text+=['', 'old_c5为a30°/b.10/c<5°；relaxed_abc仅放宽为a45°/b.15/c<10°。error_abs用最近5个已执行命令的一步先验预测RMS残差最大值≤.15替代a/b；error_relative用该最大值除以max(.02, 最近5步实际XYZ命令RMS)，阈值≤.50替代a/b；两个残差臂均保留c<5°。误差目标为实际送入环境的XYZ控制命令（按原filter规则clip至[-1,1]），不是世界EEF位移。所有主门控保留共同历史/夹爪/旋转约束、禁连续Kalman、verify τ=.10、P2、冻结Kalman参数及数学实现；alternate保留原简单交替定义。',
        '', '每case六臂同物理卡、同resident服务，各臂使用新环境并检查初始RGB/state/seed和startup Full输出相等。Williams顺序轮换；GPU2/3各20case、每suite各5case，位置计数各3或4次。公用prompt/model准备在每臂闭环前执行并单列，不把首次条件的冷准备计入其策略成本。']
    text+=['', '本轮三个新候选均未显示优于简单交替的明确证据。alternate为38/40，old_c5为37/40；放宽阈值35/40，绝对残差37/40，相对残差36/40。相对old_c5，绝对残差是2恢复/2新失败，且成本在GPU2下降2.4%、GPU3上升1.0%，没有稳定降本。放宽阈值两卡略快但质量下降2例。当前保留原对照，不推荐据这40例替换为新候选。',
        '', '下列汇总wall与成本比仅描述这批两卡样本；速度判断以逐卡同case分层为准。',
        '', '## 各套件成功数', '', '| 条件 | Spatial | Object | Goal | Long |', '|---|---:|---:|---:|---:|']
    for cid,s in stats.items():text.append('| '+cid+' | '+' | '.join(f"{v['successes']}/10" for v in s['suites'].values())+' |')
    text+=['','## 配对对照','', '| 候选 / 参考 | 恢复 | 新失败 | 描述性汇总成本比 | 分层bootstrap95%成功率差 |','|---|---:|---:|---:|---|']
    for cid,refs in pairs.items():
        for ref,p in refs.items():text.append(f"| {cid} / {ref} | {p['recoveries']} | {p['new_failures']} | {p['matched_policy_ratio']:.4f} | [{100*p['ci95_delta'][0]:+.2f},{100*p['ci95_delta'][1]:+.2f}]pp |")
    text+=['','## 同卡成本分层','', '| 候选 / 参考 | GPU | 同case n | candidate ms/action | reference ms/action | 比率 |','|---|---:|---:|---:|---:|---:|']
    for cid in ['alternate','old_c5','relaxed_abc','error_abs','error_relative']:
        for ref in ['old_c5','alternate','parent']:
            if cid==ref:continue
            for g,s in pairs[cid][ref]['by_physical_gpu'].items():text.append(f"| {cid} / {ref} | {g} | {s['n']} | {s['candidate_ms_per_action']:.3f} | {s['reference_ms_per_action']:.3f} | {s['ratio']:.4f} |")
    routing=json.loads((OUT/'routing_summary.json').read_text())
    text+=['','## 实际路由与门控','', '| 条件 | 请求 | KF提案 | KF执行请求 | KF实际动作占比 | Full |','|---|---:|---:|---:|---:|---:|']
    for cid,r in routing.items():text.append(f"| {cid} | {r['requests']} | {r.get('KF_proposals',0)} | {r.get('KF_executed_requests',0)} | {100*r.get('KF_actions',0)/r['actions']:.2f}% | {r.get('full',0)} |")
    text+=['','requests_and_causal_features.csv逐请求重放所有已执行历史，包含a/b/c与残差数值；非残差臂的新增残差列是离线诊断，不伪装成其运行时工作。execution_audit.json核对全240的门控、连续约束、真实动作前缀、旋转/夹爪保持、初态/Full签名和predict+ack成本。',
        '', '旧parent轨迹2012个候选时点，原c5允许1138，放宽阈值1561，绝对残差1338，相对残差1263；均未纳入新路由会改变历史的反馈，不能当实际Kalman使用率。NIS仅诊断，不将冻结任意命令协方差当作已校准χ²分布。参数在新40观测前冻结，old_coverage_receipt.json记录重放来源。']
    text+=['','原v3.3≥10%降本仅作参考，即成本比≤0.90；应同时看成功计数与两物理卡的分层比值。未校正候选筛选/多重比较，不能把小样本相等成功数说成总体非劣。新成本含ack，因此不直接与v3.3旧predict-only表比速度。',
        '', '父对照沿用现有共享Policy：即便不选择Kalman，它的ack仍按旧实现调用Kalman.observe。本轮计入此真实计算，不冒称最简/优化过的no-Kalman实现，也不做未经实测的扣时。调参收益主要对照old_c5与alternate；predict-only与ack分别列示。',
        '', '## 同卡成本与复现','', '逐卡、逐case配对和置信区间见paired_results.json；原始种子/噪声、初态/Full签名、动作、实际EEF XYZ/姿态和时延见每episode目录。共同warm包含固定真实初态的模型准备，动作未进入环境；与实际闭环policy wall分开保存。',
        '', '旧init6/7仅用于参数/覆盖诊断，新40固定后未改阈值；任务/初态可能曾用于更早研究，不称外部未见测试集。所有有效超时失败保留，技术attempt另列。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(20,10));ids=[c['id'] for c in m['conditions']]
    for i,case in enumerate(m['cases'][:8]):
        ax=fig.add_subplot(2,4,i+1,projection='3d')
        for cid in ids:
            r=next(x for x in groups[cid] if x['case']==case);points=np.asarray([json.loads(l)['eef_pos'] for l in (Path(r['_directory'])/'telemetry.jsonl').read_text().splitlines()]);ax.plot(*points.T,label=cid,linewidth=1)
        ax.set_title(f"{case['suite']} task{case['task']} init8");ax.set_xlabel('World X / m');ax.set_ylabel('World Y / m');ax.set_zlabel('World Z / m')
    h,l=fig.axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=6,fontsize=8);fig.tight_layout(rect=(0,.04,1,.98));fig.savefig(OUT/'fixed_profile_cases_xyz.png',dpi=150);plt.close(fig)
    text+=['','## 固定真实XYZ','', '![首8个预定profile案例的实际EEF XYZ](fixed_profile_cases_xyz.png)','', '只使用预先选定的task0/5四suite，不按结果换案例；不累加预测命令、不补零或时间拉齐。']
    (OUT/'report_zh.md').write_text('\n'.join(text))
    print({cid:s['successes'] for cid,s in stats.items()})


if __name__=='__main__':main()
