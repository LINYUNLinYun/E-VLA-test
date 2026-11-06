import hashlib,json,time
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4.analyze import write_csv,paired_ci
from preparations.idea1_v3_4.manifest import cases,SUITES
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3_1.common import LIMITS
from .common import ROOT,OUT,OLD,FOLLOWUP,NAME,PLAN,CONFIG,results


def main():
    m=json.loads((OUT/'manifest.json').read_text());decision=json.loads((OUT/'decision.json').read_text())
    checks=[]
    def check(name,value,**details):checks.append(dict(name=name,passed=bool(value),**details))
    body=dict(m);body.pop('manifest_hash');check('manifest_digest',digest(body)==m['manifest_hash'])
    check('frozen_sources',all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in m['source_files'].items()))
    check('frozen_plan',hashlib.sha256(PLAN.read_bytes()).hexdigest()==m['plan_sha256'])
    target=200 if decision['expand'] else 40;data=results()
    key=lambda r:tuple(r['case'][k] for k in ('suite','task','init'))
    expected={tuple(c[k] for k in ('suite','task','init')) for c in cases(target)}
    check('case_keys',{key(r) for d,r in data}==expected and len(data)==target,completed=len(data),target=target)
    routing_errors=[];episodes=[];tracepoints=0
    for d,r in data:
        try:
            assert r['config']==CONFIG and r['manifest_hash']==m['manifest_hash']
            cs=r['case'];assert 0<r['actions']<=LIMITS[cs['suite']]
            if not r['success']:assert r['actions']==LIMITS[cs['suite']]
            rows=[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()]
            trace=[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()]
            assert len(rows)==r['predictions'] and len(trace)==r['actions']+1
            assert all(np.isfinite(t['eef_pos']).all() and np.isfinite(t['eef_quat']).all() for t in trace)
            step=0
            with np.load(d/'arrays.npz') as z:
                for i,row in enumerate(rows):
                    a=z[f'actions_{i}'];raw=z[f'raw_{i}'];assert a.shape==(50,7) and raw.shape==(50,32)
                    assert np.isfinite(a).all() and np.isfinite(raw).all()
                    assert row['step']==step and row['seed']==seed(cs['suite'],cs['task'],cs['init'],i)
                    mod=i%3;assert row['route']['path']==('full','hybrid','v14')[mod]
                    if mod:
                        assert row['route']['old_kv_used']==(mod==1)
                        assert row['full_cache_age']==dict(rounds=mod,env_steps=mod*5)
                        assert row['timing']['old_kv_layers']==(4 if mod==1 else 0)
                        assert row['timing']['current_kv_layers']==(14 if mod==1 else 18)
                        assert row['route']['cutoff']==(15 if mod==1 else None)
                    n=10 if mod==0 else 2
                    assert np.allclose(row['timing']['nodes'],[1-j/n for j in range(n+1)],atol=1e-8,rtol=0)
                    executed=row['executed_length'];assert 0<executed<=5
                    for j in range(executed):assert np.array_equal(a[j],np.asarray(trace[step+j+1]['action'],np.float32))
                    step+=executed
            assert step==r['actions'];tracepoints+=len(trace)
            episodes.append(dict(condition=NAME,**cs,success=r['success'],actions=r['actions'],wall_s=r['wall_s'],policy_wall_s=r['policy_wall_s']))
        except Exception as exc:routing_errors.append(dict(path=str(d),error=repr(exc)))
    check('routing_noise_outputs_executed_actions',not routing_errors,errors=routing_errors,tracepoints=tracepoints)
    hist=json.loads((OUT/'historical_case_gate.json').read_text())
    check('historical_case_equivalence',hist['passed'] and len(hist['checks'])==target)
    timing_pairs=0;timing_rows=0
    for wd in sorted((OUT/'workers').glob('gpu*')):
        if not (wd/'process.json').exists():continue
        gate=json.loads((wd/'gate.json').read_text())
        check(wd.name+'_native_cutoff15_gate',gate['passed'] and sum(c.get('cutoff')==15 and c.get('passed',False) for c in gate['checks'])==4)
        check(wd.name+'_historical_service',json.loads((wd/'historical_service_gate.json').read_text())['passed'])
        if (wd/'timing.json').exists():
            timing=json.loads((wd/'timing.json').read_text());timing_pairs+=timing['completed_pairs'];timing_rows+=len(timing['rows'])
            check(wd.name+'_timing_complete',timing['completed_pairs']==timing['total'] and len(timing['rows'])==timing['total']*15)
    check('timing_bank_80_pairs',timing_pairs==80 and timing_rows==1200,pairs=timing_pairs,rows=timing_rows)
    exits=json.loads((OUT/'service_exits.json').read_text());check('model_processes_terminal',bool(exits) and all(x['exit'] is not None for x in exits.values()))
    invalids=list((OUT/NAME/'episodes').glob('*/task*/init*.invalid.*'))
    atomic(OUT/'technical_attempts.json',dict(invalid_directories=[str(x) for x in invalids]))
    summary=[];comparison=[]
    baselines=[(OLD,'periodic_v14'),(OLD,'periodic_old16'),(FOLLOWUP,'B16_first'),(FOLLOWUP,'C17_first')]
    for scope in ('first40','new160','all'):
        chosen=[r for d,r in data if scope=='all' or (r['case']['init']==14)==(scope=='first40')]
        if not chosen:continue
        for suite in ('all',*SUITES):
            rs=[r for r in chosen if suite=='all' or r['case']['suite']==suite]
            summary.append(dict(condition=NAME,scope=scope,suite=suite,n=len(rs),success=sum(r['success'] for r in rs),
                policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in rs)/sum(r['actions'] for r in rs),
                episodes_per_hour=3600*len(rs)/sum(r['wall_s'] for r in rs)))
        candidate={key(r):r for r in chosen}
        for root,name in baselines:
            base={key(r):r for d,r in results(root,name)}
            assert all(k in base for k in candidate)
            keys=sorted(candidate);pairs=[dict(suite=k[0],task=k[1],delta=int(candidate[k]['success'])-int(base[k]['success'])) for k in keys]
            ci=paired_ci(pairs)
            comparison.append(dict(candidate=NAME,baseline=name,scope=scope,n=len(keys),candidate_success=sum(candidate[k]['success'] for k in keys),baseline_success=sum(base[k]['success'] for k in keys),
                recovered=sum(candidate[k]['success'] and not base[k]['success'] for k in keys),lost=sum(not candidate[k]['success'] and base[k]['success'] for k in keys),
                delta_pp=np.mean([p['delta'] for p in pairs])*100,ci_low_pp=ci[0]*100,ci_high_pp=ci[1]*100))
    atomic(OUT/'audit.json',dict(passed=all(c['passed'] for c in checks),checks=checks,time=time.time()))
    write_csv(OUT/'episodes.csv',episodes);write_csv(OUT/'quality_summary.csv',summary);write_csv(OUT/'paired_quality.csv',comparison)
    if not all(c['passed'] for c in checks):raise RuntimeError('Final audit failed')
    lines=['# E1：L15–18仅首轮复用结果','',
        '周期固定Full-D10 → old-L15–18 draft-D2 → 当前V14 draft-D2，H50/R5，无verify/夹爪保护。对照历史结果通过服务和初始输入/Full输出核验后复用。',
        '',f"初筛{decision['screen_success']}/40；按预定质量/成本门{'扩至200条' if decision['expand'] else '未扩样，最终40条'}。",'',
        '|口径|Spatial|Object|Goal|Long|总成功|策略ms/执行动作|','|---|---|---|---|---|---|---|']
    for scope in ('first40','new160','all'):
        group={r['suite']:r for r in summary if r['scope']==scope}
        if group:lines.append('|'+ '|'.join([scope,*[f"{group[s]['success']}/{group[s]['n']}" for s in SUITES],f"{group['all']['success']}/{group['all']['n']}",f"{group['all']['policy_ms_per_action']:.3f}"])+'|')
    lines+=['','## 同case配对结果','','|口径|对照|本方案成功|对照成功|恢复/损失|差值pp|task聚类探索区间pp|','|---|---|---|---|---|---|---|']
    for r in comparison:
        lo=0 if abs(r['ci_low_pp'])<1e-10 else r['ci_low_pp'];hi=0 if abs(r['ci_high_pp'])<1e-10 else r['ci_high_pp']
        lines.append(f"|{r['scope']}|{r['baseline']}|{r['candidate_success']}/{r['n']}|{r['baseline_success']}/{r['n']}|{r['recovered']}/{r['lost']}|{r['delta_pp']:+.2f}|[{lo:+.2f}, {hi:+.2f}]|")
    lines+=['','## 周期模型路径成本','','同卡同输入80对，预热后AB/BA三次；计入Full刷新，不含RPC、仿真和加载。','','|条件|周期ms/请求|相对无复用周期加速|','|---|---|---|']
    for name,cost in decision['cycle_ms'].items():lines.append(f"|{name}|{cost:.3f}|{decision['cycle_ms']['baseline']/cost:.4f}×|")
    lines+=['','成功率为探索评测；初筛/扩样口径分别报告，非官方50-init全量基准。历史绝对wall与此次运行受资源/环境启动分片影响，不直接作为同输入加速证据。模型路径成本、自由闭环策略成本、评测吞吐单列。',
        '',f"最终验收{sum(c['passed'] for c in checks)}/{len(checks)}项通过，{tracepoints}个实际EEF记录点；技术中断归档{len(invalids)}个目录，有效失败不重试。",'',
        '[冻结方案](idea1_v3_4_l15_first_plan.md) · [最终验收](../../../results/idea1_v3_4_l15_first/audit.json) · [逐case配对](../../../results/idea1_v3_4_l15_first/paired_quality.csv) · [触发决策](../../../results/idea1_v3_4_l15_first/decision.json)']
    report=ROOT/'docs/experiments/idea1/idea1_v3_4_l15_first_report.md';report.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'complete.json',dict(episodes=len(data),report=str(report),report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',completed=len(data),success=sum(r['success'] for d,r in data),time=time.time()))


if __name__=='__main__':main()
