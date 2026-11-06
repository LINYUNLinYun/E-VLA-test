"""Post-run comparison of independent changes, their combination and new160."""
import argparse
import hashlib,json,subprocess,time,re
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci,write_csv
from .common import ROOT,OUT,OLD
from .controller import results


def live(pid):
    p=subprocess.run(['ps','-p',str(pid),'-o','args='],capture_output=True,text=True)
    return p.returncode==0 and 'preparations.idea1_v3_4_followup.controller' in p.stdout


def main(wait=False):
    if wait:
        pid=json.loads((OUT/'process.json').read_text())['pid']
        atomic(OUT/'review_process.json',dict(pid=__import__('os').getpid(),wait_pid=pid,started=time.time()))
        while live(pid): time.sleep(10)
    audit=json.loads((OUT/'audit.json').read_text()); complete=json.loads((OUT/'complete.json').read_text())
    assert audit['passed']
    report=Path(complete['report']); assert hashlib.sha256(report.read_bytes()).hexdigest()==complete['report_sha256']
    decision=json.loads((OUT/'decision.json').read_text()); m=json.loads((OUT/'manifest.json').read_text())
    data={}
    for name,old in (('periodic_v14',True),('periodic_old16',True),('A17_all',False),('B16_first',False),('C17_first',False)):
        data[name]={tuple(r['case'][k] for k in ('suite','task','init')):r for d,r in results(name,old=old)}
    checks=[]
    def check(name,value,**detail):
        checks.append(dict(name=name,passed=bool(value),**detail)); assert value,(name,detail)
    check('old_manifest_identity',hashlib.sha256((OLD/'manifest.json').read_bytes()).hexdigest()==m['historical_manifest_sha256'])
    old40=sum(r['success'] for k,r in data['periodic_old16'].items() if k[2]==14)
    independent={name:sum(r['success'] for k,r in data[name].items() if k[2]==14)>=old40+2 and decision['cycle_ms'][name]<decision['cycle_ms']['baseline'] for name in ('A17_all','B16_first')}
    check('independent_effect_decision',independent==decision['effects'])
    check('combination_trigger',decision['combo']==all(independent.values()))
    ran=['A17_all','B16_first']+(['C17_first'] if decision['combo'] else [])
    expected_expanded=[name for name in ran if sum(r['success'] for k,r in data[name].items() if k[2]==14)>=35 and decision['cycle_ms'][name]<=decision['cycle_ms']['baseline']]
    check('expansion_decision',expected_expanded==decision['expanded'])
    check('canonical_episode_total',sum(len(data[n]) for n in ran)==complete['episodes'])
    pairrows=[]
    comparisons=[(base,name) for name in ran for base in ('periodic_v14','periodic_old16')]
    if 'C17_first' in ran: comparisons += [('A17_all','C17_first'),('B16_first','C17_first')]
    for baseline,candidate in comparisons:
        shared=sorted(set(data[baseline])&set(data[candidate]))
        for scope in ('first40','new160','all_shared'):
            keys=[k for k in shared if scope=='all_shared' or (k[2]==14)==(scope=='first40')]
            if not keys: continue
            pairs=[dict(suite=k[0],task=k[1],delta=int(data[candidate][k]['success'])-int(data[baseline][k]['success'])) for k in keys]
            ci=paired_ci(pairs)
            pairrows.append(dict(baseline=baseline,candidate=candidate,scope=scope,n=len(keys),baseline_success=sum(data[baseline][k]['success'] for k in keys),
                candidate_success=sum(data[candidate][k]['success'] for k in keys),recovered=sum(not data[baseline][k]['success'] and data[candidate][k]['success'] for k in keys),
                lost=sum(data[baseline][k]['success'] and not data[candidate][k]['success'] for k in keys),delta_pp=np.mean([p['delta'] for p in pairs])*100,
                ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    write_csv(OUT/'independent_combined_comparison.csv',pairrows)
    interaction=None
    if decision['combo']:
        keys=[k for k in data['A17_all'] if k[2]==14]
        a,b,c,d=[sum(data[n][k]['success'] for k in keys) for n in ('A17_all','B16_first','C17_first','periodic_old16')]
        delta=[dict(suite=k[0],task=k[1],delta=int(data['C17_first'][k]['success'])-int(data['A17_all'][k]['success'])-int(data['B16_first'][k]['success'])+int(data['periodic_old16'][k]['success'])) for k in keys]
        ci=paired_ci(delta)
        ar={k for k in keys if data['A17_all'][k]['success'] and not data['periodic_old16'][k]['success']}
        br={k for k in keys if data['B16_first'][k]['success'] and not data['periodic_old16'][k]['success']}
        interaction=dict(n=len(keys),A=a,B=b,C=c,old16=d,additive_count=a+b-d,observed_count=c,interaction_pp=(c-a-b+d)*100/len(keys),
            ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100,A_recovered=len(ar),B_recovered=len(br),shared_recovered=len(ar&br))
        atomic(OUT/'screen_interaction.json',interaction)
    exits=json.loads((OUT/'service_exits.json').read_text()); resources=[]
    for gpu,item in exits.items():
        p=subprocess.run(['ps','-p',str(item['pid']),'-o','args='],capture_output=True,text=True)
        own_live=p.returncode==0 and 'preparations.idea1_v3_4_followup.server' in p.stdout
        check('gpu'+gpu+'_model_exited',not own_live)
        resources.append(dict(gpu=int(gpu),pid=item['pid'],live=own_live))
    technical=list(OUT.glob('*/episodes/*/task*/init*.invalid.*'))
    rows=[]
    for name in ran:
        for scope in ('first40','new160','all'):
            selected=[r for k,r in data[name].items() if scope=='all' or (k[2]==14)==(scope=='first40')]
            if not selected: continue
            rows.append(dict(condition=name,scope=scope,n=len(selected),success=sum(r['success'] for r in selected),
                action_cost_ms=1000*sum(r['policy_wall_s'] for r in selected)/sum(r['actions'] for r in selected),
                episode_per_hour=len(selected)*3600/sum(r['wall_s'] for r in selected),actions_mean=np.mean([r['actions'] for r in selected])))
    write_csv(OUT/'stage_summary.csv',rows)
    labels=dict(A17_all='A：L17–18两轮复用',B16_first='B：L16–18仅首轮复用',C17_first='C：L17–18仅首轮复用')
    lines=['# v3.4 E1补充实验结论','',
        '两项独立改动已完成，P=2保持Full→draft→draft；没有把第二轮draft改成Full。历史对照通过数值复用门。',
        '', '|条件|样本口径|成功|策略ms/执行动作|回合/小时/worker|', '|---|---|---|---|---|']
    for r in rows: lines.append(f"|{labels[r['condition']]}|{r['scope']}|{r['success']}/{r['n']}|{r['action_cost_ms']:.3f}|{r['episode_per_hour']:.1f}|")
    lines+=['','原对照：旧L16–18两轮复用31/40、164/200；无复用周期V14为37/40、182/200。初筛决定扩样，不将40条与200条直接比较；new160单独给出，仍属探索评测。',
        '', '## 组合相对独立改动','', '|组合对照|样本口径|n|对照成功|组合成功|恢复/损失|差值pp|task聚类探索区间pp|','|---|---|---|---|---|---|---|---|']
    for r in pairrows:
        if r['baseline'] not in ('A17_all','B16_first'): continue
        lo=0 if abs(r['ci_low_pp'])<1e-10 else r['ci_low_pp']; hi=0 if abs(r['ci_high_pp'])<1e-10 else r['ci_high_pp']
        lines.append(f"|{labels[r['baseline']]}|{r['scope']}|{r['n']}|{r['baseline_success']}|{r['candidate_success']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|")
    if interaction:
        lines += ['',f"初筛四臂中，旧方案{interaction['old16']}/40，A {interaction['A']}/40，B {interaction['B']}/40，组合C {interaction['C']}/40。若直接相加净收益，预测C为{interaction['additive_count']}/40；实际为{interaction['observed_count']}/40。描述性加性交互为{interaction['interaction_pp']:+.2f}个百分点，不把成功数改善当作可简单相加。A/B分别救回{interaction['A_recovered']}/{interaction['B_recovered']}个旧失败，其中重合{interaction['shared_recovered']}个；这仍不是内部机制的唯一因果解释。"]
    t=decision['request_ms']; old_cycle=(t['full10']+2*t['old16'])/3
    lines += ['', '## 成本与适用边界','',f"同输入无复用周期V14为{decision['cycle_ms']['baseline']:.3f}ms/请求，原L16–18两轮复用为{old_cycle:.3f}ms/请求。以下是模型路径周期估算（prefill/AE/解码），包含Full刷新，不含RPC、仿真和模型加载。",
        '', '|条件|周期平均请求ms|相对无复用V14加速|相对原两轮old16耗时变化|','|---|---|---|---|']
    for name in ran:
        cost=decision['cycle_ms'][name]; lines.append(f"|{labels[name]}|{cost:.3f}|{decision['cycle_ms']['baseline']/cost:.4f}×|{(cost/old_cycle-1)*100:+.2f}%|")
    if len(data['B16_first'])==len(data['C17_first'])==200:
        bsuccess=sum(r['success'] for r in data['B16_first'].values()); csuccess=sum(r['success'] for r in data['C17_first'].values())
        c_vs_b=next(r for r in pairrows if r['baseline']=='B16_first' and r['candidate']=='C17_first' and r['scope']=='all_shared')
        spatial={name:sum(r['success'] for k,r in data[name].items() if k[0]=='libero_spatial') for name in ('B16_first','C17_first')}
        lead=['',f"主要发现：限制第二轮复用的B恢复到{bsuccess}/200，组合C为{csuccess}/200；原两轮old16为164/200，无复用周期V14为182/200。组合相对B恢复{c_vs_b['recovered']}例、损失{c_vs_b['lost']}例，净增{csuccess-bsuccess}例，探索区间[{c_vs_b['ci_low_pp']:+.2f},{c_vs_b['ci_high_pp']:+.2f}]个百分点，未显示稳定的组合增量。",
            '',f"Spatial中B为{spatial['B16_first']}/50，C为{spatial['C17_first']}/50；但C的Goal/Long并未同步提高，不能用Spatial的恢复代表所有任务。A只完成40条，不能据此给出其200条质量结论。",
            '',f"质量补救有计算代价：B/C周期模型耗时相对原两轮old16分别增加{(decision['cycle_ms']['B16_first']/old_cycle-1)*100:.2f}%/{(decision['cycle_ms']['C17_first']/old_cycle-1)*100:.2f}%，相对无复用V14仍分别节省{(1-decision['cycle_ms']['B16_first']/decision['cycle_ms']['baseline'])*100:.2f}%/{(1-decision['cycle_ms']['C17_first']/decision['cycle_ms']['baseline'])*100:.2f}%。组合没有在这批数据中同时明显胜过B的质量与成本。"]
        lines[3:3]=lead
    lines+=['', '周期估算包括Full刷新，用相同输入、同卡、同计时边界的实测路径组合；自由闭环每动作成本与回合吞吐另列，状态分叉和回合长度不同，二者不能合成一个速度结论。原始matched input检查、cutoff17截断与完整计算后替换KV一致性、F-H-H/F-H-V实际路由与模型退出已核验。',
        '', '[详细报告](idea1_v3_4_followup_report.md) · [冻结方案](idea1_v3_4_followup_plan.md) · [完整执行验收](../../../results/idea1_v3_4_followup/audit.json) · [独立与组合逐案例比较](../../../results/idea1_v3_4_followup/independent_combined_comparison.csv)']
    summary=ROOT/'docs/experiments/idea1/idea1_v3_4_followup_summary.md'; summary.write_text('\n'.join(lines)+'\n')
    for path in (report,summary):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if not target.startswith(('http','#')): check('link:'+target,(path.parent/target).resolve().exists())
    atomic(OUT/'completion_review.json',dict(passed=True,time=time.time(),checks=checks,resources=resources,technical_invalid=sorted(map(str,technical)),
        episodes=complete['episodes'],interaction=interaction,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary=str(summary),summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),report_sha256=complete['report_sha256']))
    print(json.dumps(dict(passed=True,episodes=complete['episodes'],summary=str(summary))))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--wait',action='store_true'); args=parser.parse_args(); main(args.wait)
