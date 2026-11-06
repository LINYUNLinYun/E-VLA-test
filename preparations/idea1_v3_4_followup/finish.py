import hashlib,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4.analyze import write_csv,paired_ci
from preparations.idea1_v3_4.manifest import cases,SUITES
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3_1.common import LIMITS
from .common import ROOT,OUT,OLD
from .controller import results


def main():
    m=json.loads((OUT/'manifest.json').read_text()); decision=json.loads((OUT/'decision.json').read_text()); failures=[]; checks=[]
    def check(name,value,**detail):
        checks.append(dict(name=name,passed=bool(value),**detail))
        if not value: failures.append(checks[-1])
    body=dict(m); body.pop('manifest_hash'); check('manifest',digest(body)==m['manifest_hash'])
    check('sources',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in m['source_files'].items()))
    check('plan',hashlib.sha256((ROOT/'docs/experiments/idea1/idea1_v3_4_followup_plan.md').read_bytes()).hexdigest()==m['plan_sha256'])
    summary=[]; comparisons=[]; allcases=[]
    for cfg in m['conditions']:
        name=cfg['id']; target=200 if name in decision['expanded'] else 40 if name!='C17_first' or decision['combo'] else 0
        data=results(name); expected={(x['suite'],x['task'],x['init']) for x in cases(target)}
        found={tuple(r['case'][k] for k in ('suite','task','init')) for d,r in data}
        check(name+'_case_keys',found==expected,completed=len(data),target=target)
        errors=[]
        for d,r in data:
            try:
                assert r['manifest_hash']==m['manifest_hash'] and r['config']==cfg
                cs=r['case']; assert 0<r['actions']<=LIMITS[cs['suite']]
                if not r['success']: assert r['actions']==LIMITS[cs['suite']]
                rows=[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()]
                trace=[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()]
                assert len(trace)==r['actions']+1 and len(rows)==r['predictions']
                step=0
                with np.load(d/'arrays.npz') as arrays:
                    for i,row in enumerate(rows):
                        assert row['step']==step and row['seed']==seed(cs['suite'],cs['task'],cs['init'],i)
                        a=arrays[f'actions_{i}']; raw=arrays[f'raw_{i}']; assert a.shape==(50,7) and raw.shape==(50,32)
                        assert np.isfinite(a).all() and np.isfinite(raw).all()
                        isfull=i%3==0; assert (row['route']['path']=='full')==isfull
                        if not isfull:
                            active=not cfg.get('first_only') or i%3==1
                            assert row['route']['old_kv_used']==active
                            assert row['route']['path']==('hybrid' if active else 'v14')
                            assert row['full_cache_age']==dict(rounds=i%3,env_steps=(i%3)*5)
                            if active: assert row['timing']['old_kv_layers']==19-cfg['cutoff']
                            else: assert row['timing']['old_kv_layers']==0
                        n=10 if isfull else 2
                        assert np.allclose(row['timing']['nodes'],[1-j/n for j in range(n+1)],atol=1e-8,rtol=0)
                        executed=row['executed_length']; assert 0<executed<=5
                        for j in range(executed): assert np.array_equal(a[j],np.asarray(trace[step+j+1]['action'],np.float32))
                        step+=executed
                assert step==r['actions']
                allcases.append(dict(condition=name,**cs,success=r['success'],actions=r['actions'],policy_wall_s=r['policy_wall_s'],wall_s=r['wall_s']))
            except Exception as exc: errors.append(dict(directory=str(d),error=repr(exc)))
        check(name+'_routing_actions',not errors,errors=errors)
        if not target: continue
        hist=json.loads((OUT/f'{name}_historical_gate.json').read_text())
        check(name+'_historical_reuse',hist['passed'] and len(hist['checks'])==target)
        for scope in ('first40','all'):
            rs=[r for d,r in data if scope=='all' or r['case']['init']==14]
            for suite in ('all',*SUITES):
                chosen=[r for r in rs if suite=='all' or r['case']['suite']==suite]
                summary.append(dict(condition=name,scope=scope,suite=suite,n=len(chosen),success=sum(r['success'] for r in chosen),
                    policy_ms_per_action=sum(r['policy_wall_s'] for r in chosen)*1000/sum(r['actions'] for r in chosen),
                    episode_per_hour=len(chosen)*3600/sum(r['wall_s'] for r in chosen)))
        candidate={tuple(r['case'][k] for k in ('suite','task','init')):r for d,r in data}
        for base_name in ('periodic_v14','periodic_old16'):
            baseline={tuple(r['case'][k] for k in ('suite','task','init')):r for d,r in results(base_name,old=True)}
            for scope in ('first40','all'):
                shared=[k for k in sorted(candidate) if scope=='all' or k[2]==14]
                pairs=[dict(suite=k[0],task=k[1],delta=int(candidate[k]['success'])-int(baseline[k]['success'])) for k in shared]
                ci=paired_ci(pairs)
                comparisons.append(dict(candidate=name,baseline=base_name,scope=scope,n=len(shared),
                    recovered=sum(candidate[k]['success'] and not baseline[k]['success'] for k in shared),
                    lost=sum(not candidate[k]['success'] and baseline[k]['success'] for k in shared),
                    delta_pp=np.mean([x['delta'] for x in pairs])*100,ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    for gpu in (0,3):
        wd=OUT/'workers'/f'gpu{gpu}'
        check(f'gpu{gpu}_gate',json.loads((wd/'gate.json').read_text())['passed'])
        check(f'gpu{gpu}_historical_service',json.loads((wd/'historical_service_gate.json').read_text())['passed'])
        bench=json.loads((wd/'timing.json').read_text()); check(f'gpu{gpu}_timing_complete',bench['completed_pairs']==bench['total']==40 and len(bench['rows'])==480)
    check('model_processes_terminal',all(r['exit'] is not None for r in json.loads((OUT/'service_exits.json').read_text()).values()))
    atomic(OUT/'audit.json',dict(passed=not failures,checks=checks,failures=failures,time=time.time()))
    write_csv(OUT/'episodes.csv',allcases); write_csv(OUT/'quality_summary.csv',summary); write_csv(OUT/'paired_quality.csv',comparisons)
    if failures: raise RuntimeError('Final audit failed')
    lines=['# v3.4 E1 两项独立补充实验','',
        '原P=2、Full→draft→draft不变。A仅复用L17–18两轮；B保留L16–18但只首轮复用；C为两者组合（条件触发）。',
        '',f"初筛有效性：{decision['effects']}；组合触发：{decision['combo']}；扩样条件：{decision['expanded']}。",'',
        '|条件|口径|Spatial|Object|Goal|Long|总成功|策略ms/动作|', '|---|---|---|---|---|---|---|---|']
    for name in ('A17_all','B16_first','C17_first'):
        for scope in ('first40','all'):
            group=[r for r in summary if r['condition']==name and r['scope']==scope]
            if not group: continue
            s={r['suite']:r for r in group}; lines.append('|'+ '|'.join([name,scope,*[f"{s[k]['success']}/{s[k]['n']}" for k in SUITES],f"{s['all']['success']}/{s['all']['n']}",f"{s['all']['policy_ms_per_action']:.3f}"])+'|')
    lines+=['','原同case对照：periodic_v14首40为37/40、全200为182/200；periodic_old16首40为31/40、全200为164/200。以上历史质量通过当前输入/Full原始动作检查后复用；跨运行绝对wall不直接作为加速证据。',
        '', '## 同输入周期成本','', '|条件|周期平均请求ms|相对无复用周期V14加速|','|---|---|---|']
    for name,cost in decision['cycle_ms'].items(): lines.append(f"|{name}|{cost:.3f}|{decision['cycle_ms']['baseline']/cost:.4f}×|")
    lines+=['','周期估算计入一次Full及两次draft；单次路径实测见decision.json与workers/gpu*/timing.json，准备加载与仿真不属于该微基准。自由闭环每动作策略成本和episode/小时另列quality_summary.csv。',
        '', '## 配对结果','', '|候选|对照|口径|n|恢复|损失|成功率差pp|task聚类探索区间pp|','|---|---|---|---|---|---|---|---|']
    for r in comparisons:
        lo=0. if abs(r['ci_low_pp'])<1e-10 else r['ci_low_pp']; hi=0. if abs(r['ci_high_pp'])<1e-10 else r['ci_high_pp']
        lines.append(f"|{r['candidate']}|{r['baseline']}|{r['scope']}|{r['n']}|{r['recovered']}|{r['lost']}|{r['delta_pp']:+.2f}|[{lo:+.2f},{hi:+.2f}]|")
    lines+=['','这是探索评测，非官方50-init全量基准；初筛/扩样与组合条件均在运行前冻结，不以接受率或单输入误差代替闭环质量。原始数据保存在results/idea1_v3_4_followup，逐案例、逐请求、真实EEF姿态、动作、路由、seed和年龄均保留。',
        '', '[冻结方案](idea1_v3_4_followup_plan.md) · [最终验收](../../../results/idea1_v3_4_followup/audit.json) · [触发决策与测速](../../../results/idea1_v3_4_followup/decision.json)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_followup_report.md'; report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(time=time.time(),report=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),episodes=len(allcases)))

if __name__=='__main__': main()
