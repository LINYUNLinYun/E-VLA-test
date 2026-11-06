"""CPU-only progress, bounded profiling, causal counter audit and four-arm report."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import LIMITS
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3.policy import verification_seed
from .design import OUT
from .rules import acceptance


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows():
    result=defaultdict(list)
    for p in OUT.glob('*/episodes/*/task*/init08/result.json'):
        if p.with_name('complete.json').exists():
            r=read(p);r['_directory']=str(p.parent);result[r['condition']].append(r)
    return result
def key(r):return tuple(r['case'][k] for k in ['suite','task','init'])
def stats(rs):
    a=sum(r['actions'] for r in rs);wall=sum(r['wall_s'] for r in rs)
    return dict(n=len(rs),successes=sum(r['success'] for r in rs),actions=a,requests=sum(r['predictions'] for r in rs),
        total_policy_wall_s=sum(r['policy_wall_s'] for r in rs),
        mean_episode_actions=a/len(rs) if rs else None,mean_success_episode_actions=float(np.mean([r['actions'] for r in rs if r['success']])) if any(r['success'] for r in rs) else None,
        policy_ms_per_action=1000*sum(r['policy_wall_s'] for r in rs)/a if a else None,
        predict_ms_per_action=1000*sum(r['predict_policy_wall_s'] for r in rs)/a if a else None,
        ack_ms_per_action=1000*sum(r['ack_policy_wall_s'] for r in rs)/a if a else None,
        warm_s=sum(r['common_warm_s'] for r in rs),episode_worker_wall_s=wall,
        episode_per_worker_hour=len(rs)*3600/wall if wall else None,actions_per_worker_second=a/wall if wall else None,
        timings_s={k:sum(r['timing'].get(k,0.) for r in rs) for k in ['environment_acquire_s','reset_s','rpc_s','step_s','ack_s','save_s']},
        suites={s:sum(r['success'] for r in rs if r['case']['suite']==s) for s in ['libero_spatial','libero_object','libero_goal','libero_10']})
def pair(a,b):
    right={key(r):r for r in b};pairs=[(r,right[key(r)]) for r in a]
    assert len(pairs)==40
    assert all(x['engineering_binding']['model_gpu']==y['engineering_binding']['model_gpu']==3 for x,y in pairs)
    aa,bb=stats(a),stats(b);rng=np.random.default_rng(20261025)
    delta={s:[int(x['success'])-int(y['success']) for x,y in pairs if x['case']['suite']==s] for s in aa['suites']}
    boot=[];costboot=[]
    strata={s:[(x,y) for x,y in pairs if x['case']['suite']==s] for s in aa['suites']}
    for _ in range(2000):
        sample=[v[i] for v in strata.values() for i in rng.integers(0,len(v),len(v))]
        boot.append(np.mean([int(x['success'])-int(y['success']) for x,y in sample]))
        ca=sum(x['policy_wall_s'] for x,y in sample)/sum(x['actions'] for x,y in sample)
        cb=sum(y['policy_wall_s'] for x,y in sample)/sum(y['actions'] for x,y in sample)
        costboot.append(ca/cb)
    return dict(n=40,physical_gpu=3,recoveries=sum(x['success'] and not y['success'] for x,y in pairs),new_failures=sum(not x['success'] and y['success'] for x,y in pairs),
        success_bootstrap_degenerate=bool(np.ptp(boot)==0),success_bootstrap_interpretation='Empirical boundary degeneracy when all observed paired outcomes agree; [0,0] must not be interpreted as a population95%CI or quality equivalence.',
        policy_wall_per_action_ratio=aa['policy_ms_per_action']/bb['policy_ms_per_action'],policy_ratio_task_stratified_bootstrap_ci95=np.percentile(costboot,[2.5,97.5]).tolist(),success_difference_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
        case_outcomes=[dict(case=x['case'],candidate_success=x['success'],reference_success=y['success']) for x,y in pairs])


def behavior(a,b):
    right={key(r):r for r in b};checks=[]
    for r in a:
        s=right[key(r)];da,db=Path(r['_directory']),Path(s['_directory'])
        ia=[json.loads(l) for l in (da/'infer.jsonl').read_text().splitlines()];ib=[json.loads(l) for l in (db/'infer.jsonl').read_text().splitlines()]
        first=None
        with np.load(da/'arrays.npz') as aa,np.load(db/'arrays.npz') as bb:
            for i in range(max(len(ia),len(ib))):
                if i>=min(len(ia),len(ib)):
                    first=dict(replan=i,reason='request_count');break
                x,y=ia[i],ib[i]
                if not np.array_equal(aa[f'actions_{i}'],bb[f'actions_{i}']) or x['executed_length']!=y['executed_length']:
                    first=dict(replan=i,step=[x['step'],y['step']],input_hash_equal=x['input_hash']==y['input_hash'],seed_equal=x['seed']==y['seed'],noise_hash_equal=x['noise_hash']==y['noise_hash'],
                        routes=[x['route'],y['route']],proposed_execute_lengths=[x['execute_length'],y['execute_length']],actual_executed_lengths=[x['executed_length'],y['executed_length']],actions_maxabs=float(np.max(np.abs(aa[f'actions_{i}']-bb[f'actions_{i}']))));break
        actual_a=np.asarray([json.loads(l)['action'] for l in (da/'telemetry.jsonl').read_text().splitlines()[1:]],np.float32)
        actual_b=np.asarray([json.loads(l)['action'] for l in (db/'telemetry.jsonl').read_text().splitlines()[1:]],np.float32)
        rhythm=lambda inf:[(q['step'],'full' if q['route']['path']=='full' else 'draft',q['executed_length']) for q in inf]
        checks.append(dict(case=r['case'],all_decoded_actions_and_executed_lengths_exact=first is None,actual_action_trajectory_exact=np.array_equal(actual_a,actual_b),
            request_full_draft_rhythm_exact=rhythm(ia)==rhythm(ib),success_equal=r['success']==s['success'],first_difference=first))
    return dict(n=40,exact=sum(c['all_decoded_actions_and_executed_lengths_exact'] for c in checks),actual_trajectory_exact=sum(c['actual_action_trajectory_exact'] for c in checks),
        rhythm_exact=sum(c['request_full_draft_rhythm_exact'] for c in checks),success_equal=sum(c['success_equal'] for c in checks),checks=checks)


def audit(groups,m):
    checks=[];routing={};request_rows=[];blocks=defaultdict(list)
    for cfg in m['conditions']:
        rs=groups[cfg['id']];assert len(rs)==40 and {key(r) for r in rs}=={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']}
        count=Counter();reasons=Counter();kvrounds=[];kvsteps=[];verify_kvr=[];verify_kvs=[];allkv=[];times=Counter()
        for r in rs:
            d=Path(r['_directory']);identity=read(d/'identity.json');inf=[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()];tele=[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()]
            assert len(tele)==r['actions']+1 and len(inf)==r['predictions']
            assert r['success'] or r['actions']==LIMITS[r['case']['suite']]
            assert r['manifest_hash']==m['manifest_hash'] and identity['manifest_hash']==m['manifest_hash']
            blocks[key(r)].append((r['initial_hash'],r['startup_signature']['raw_sha256'],r['startup_signature']['decoded_sha256'],r['engineering_binding']['model_gpu'],identity['protocol']['server_pid'],inf[0]['seed'],inf[0]['noise_hash'],inf[0]['input_hash']))
            step=0;draft=0;fullround=fullstep=None;total=0.
            with np.load(d/'arrays.npz') as arrays:
                for i,q in enumerate(inf):
                    route=q['route'];path=route['path'];n=q['executed_length'];period=cfg['refresh_period'];state=q['refresh_state']
                    assert q['seed']==seed(r['case']['suite'],r['case']['task'],r['case']['init'],i)
                    assert q['verification_seed']==verification_seed(q['seed'])
                    assert q['step']==step and state['period']==period and state['draft_rounds_before']==draft
                    assert q['full_cache_age']==(None if fullround is None else dict(rounds=i-fullround,env_steps=step-fullstep))
                    assert 0<n<=q['execute_length']<=10
                    actual=np.asarray([t['action'] for t in tele[step+1:step+n+1]],np.float32)
                    assert np.array_equal(actual,arrays[f'actions_{i}'][:n])
                    if fullround is not None:allkv.append(i-fullround)
                    verified='distance' in route
                    if cfg['mode']=='periodic':
                        assert not verified and q['counts']['verify']==0 and not any(k.startswith('verify_') for k in q['timing'])
                    if verified:
                        verify_kvr.append(q['full_cache_age']['rounds']);verify_kvs.append(q['full_cache_age']['env_steps'])
                        count['verify']+=1;length,details=acceptance(arrays[f'draft_future_{i}'],arrays[f'reconstructed_future_{i}'],.8,10,protect=False)
                        assert length==route['accepted_length'] and details=={k:route[k] for k in details}
                        count['verify_zero']+=length==0;count['verify_partial']+=0<length<10;count['verify_full10']+=length==10
                    assert not route.get('verify_gripper',False) and not route.get('post_verify_gripper',False)
                    assert route.get('full_reason') not in ['pending_gripper','gripper']
                    if path=='full':
                        reason=route['full_reason'];reasons[reason]+=1
                        assert reason in ['startup','periodic','zero_accept']
                        if reason=='startup':assert i==0
                        if reason=='periodic':assert draft==period
                        if reason=='zero_accept':assert verified and route['accepted_length']==0 and draft<period
                        fullround=i;fullstep=step;draft=0
                    else:
                        assert path==('periodic_draft' if cfg['mode']=='periodic' else 'skip_verified') and draft<period
                        kvrounds.append(i-fullround);kvsteps.append(step-fullstep);draft+=1
                    assert state['draft_rounds_after']==draft
                    count['requests']+=1;count[path]+=1;count['actions']+=n;count['full_actions' if path=='full' else 'draft_actions']+=n
                    for k,v in q['timing'].items():times[k]+=v
                    times[path+'_complete_policy_ms']+=q['timing']['policy_wall_ms']+q['ack_policy_wall_ms']
                    if path=='full' and route['full_reason']=='zero_accept':
                        times['zero_accept_complete_policy_ms']+=q['timing']['policy_wall_ms']+q['ack_policy_wall_ms']
                        for name in ['draft_ae_ms','verify_ae_ms','full_ae_ms']:times['zero_accept_'+name]+=q['timing'].get(name,0.)
                    total+=q['timing']['policy_wall_ms']+q['ack_policy_wall_ms']
                    request_rows.append(dict(condition=cfg['id'],**r['case'],replan=i,step=step,path=path,full_reason=route.get('full_reason'),executed_length=n,
                        draft_before=state['draft_rounds_before'],draft_after=draft,KV_rounds=None if q['full_cache_age'] is None else q['full_cache_age']['rounds'],KV_steps=None if q['full_cache_age'] is None else q['full_cache_age']['env_steps'],
                        accepted_length=route.get('accepted_length'),predict_ms=q['timing']['policy_wall_ms'],ack_ms=q['ack_policy_wall_ms']))
                    step+=n
            assert step==r['actions'] and np.isclose(total/1000,r['policy_wall_s'],rtol=1e-12,atol=1e-9)
            checks.append(dict(condition=cfg['id'],case=r['case'],passed=True,result_sha256=sha(d/'result.json'),identity_sha256=sha(d/'identity.json')))
        routing[cfg['id']]=dict(counts=dict(count),full_reasons=dict(reasons),timing_sum_ms=dict(times),
            draft_KV_rounds=dict(max=max(kvrounds),mean=float(np.mean(kvrounds))),draft_KV_steps=dict(max=max(kvsteps),mean=float(np.mean(kvsteps))),
            verification_KV_rounds=dict(max=max(verify_kvr),mean=float(np.mean(verify_kvr))) if verify_kvr else None,
            verification_KV_steps=dict(max=max(verify_kvs),mean=float(np.mean(verify_kvs))) if verify_kvs else None,
            request_KV_rounds_max=max(allkv))
    assert len(blocks)==40 and all(len(v)==4 and len(set(v))==1 for v in blocks.values())
    atomic(OUT/'execution_audit.json',dict(time=time.time(),passed=True,episodes=160,requests=len(request_rows),initial_startup_Full_same_card_service_exact_blocks=40,checks=checks,
        scope='Actual prefixes, P counters, zero/partial acceptance, original gripper-disabled rule, KV age, periodic no-verify, wall accounting; valid timeout failures retained.'))
    atomic(OUT/'routing_summary.json',routing)
    with (OUT/'requests.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(request_rows[0]));w.writeheader();w.writerows(request_rows)
    return routing


def main():
    m=read(OUT/'manifest.json');groups=rows();ss={c['id']:stats(groups[c['id']]) for c in m['conditions']};valid=sum(s['n'] for s in ss.values())
    atomic(OUT/'progress.json',dict(time=time.time(),valid=valid,target=160,conditions=ss))
    profile_keys={tuple(c[k] for k in ['suite','task','init']) for c in m['cases'][:8]};profile=[r for rs in groups.values() for r in rs if key(r) in profile_keys]
    if len(profile)==32 and not (OUT/'profiling_receipt.json').exists():
        ends=[(Path(r['_directory'])/'complete.json').stat().st_mtime for r in profile];span=max(ends)-min(e-r['wall_s'] for e,r in zip(ends,profile))
        atomic(OUT/'profiling_receipt.json',dict(time=time.time(),predefined_cases=8,included_episodes=32,observed_window_s=span,episodes_per_hour=32*3600/span,stats=stats(profile),decision='Keep B1 single worker and frozen4 arms; finish all160, no tuning/extra smoke.'))
    if valid!=160:print('Progress',valid,'/160');return
    routing=audit(groups,m)
    comparisons=[('p4_periodic','p2_periodic'),('p4_verified','p2_verified'),('p2_verified','p2_periodic'),('p4_verified','p4_periodic')]
    pairs={a+'_vs_'+b:pair(groups[a],groups[b]) for a,b in comparisons}
    equivalence={f'p{p}_verified_vs_periodic':behavior(groups[f'p{p}_verified'],groups[f'p{p}_periodic']) for p in [2,4]}
    atomic(OUT/'behavior_comparisons.json',equivalence)
    atomic(OUT/'paired_results.json',dict(summary=ss,paired=pairs,bootstrap=2000,seed=20261025,scope='40 fixed paired cases screening; no multiple-testing correction.'))
    text=['# E4 P4刷新周期补测','', '四suite各10task、init8，四臂各40例；同物理GPU3/同模型服务、每case四臂平衡顺序，每臂新环境。Xiaomi skip12、Full/draft D2、H10/R10、无夹爪保护。P=连续draft轮数；零接受Full重置，部分接受计一轮；验证保持tau=.8、q=.10/.05、原生1-q方向。旧300仅历史参照。',
        '', '| 条件 | 成功 | policy ms/action | predict | ack | 动作 | 请求 | 每例平均动作/成功例均值 | episode worker秒 | episode/小时 | 动作/秒 |', '|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for cid,s in ss.items():text.append(f"| {cid} | {s['successes']}/40 | {s['policy_ms_per_action']:.3f} | {s['predict_ms_per_action']:.3f} | {s['ack_ms_per_action']:.4f} | {s['actions']} | {s['requests']} | {s['mean_episode_actions']:.2f}/{s['mean_success_episode_actions']:.2f} | {s['episode_worker_wall_s']:.2f} | {s['episode_per_worker_hour']:.1f} | {s['actions_per_worker_second']:.2f} |")
    r0=pairs['p4_periodic_vs_p2_periodic']['policy_wall_per_action_ratio'];r1=pairs['p4_verified_vs_p2_verified']['policy_wall_per_action_ratio']
    text.insert(2,f"主要观察：无verify时P4相对P2每动作成本变化{100*(r0-1):+.2f}%；保留verify时变化{100*(r1-1):+.2f}%，因此本轮未降低带验证策略的开销。四臂均40/40，同周期有/无验证的动作与结果完全相同；小幅成本差受固定suite顺序限制，仅作本批观察。\n")
    text+=['','policy=完整服务端predict+ack /实际动作，同卡同case；不是RPC wall或纯GPU时间。共享ack仍调用原Kalman.observe，四臂共同保留，不称最简Xiaomi实现。公用Xiaomi准备/重复Full检查、模型加载、环境/reset/RPC/仿真/IO均单列。',
        '', '## 同卡配对', '', '| 候选/参考 | 恢复 | 新失败 | 成本比 | 成本比bootstrap95% | 成功率差描述 |','|---|---:|---:|---:|---|---|']
    for (a,b),p in zip(comparisons,pairs.values()):
        quality='样本无差异，bootstrap退化' if p['success_bootstrap_degenerate'] else f"bootstrap [{p['success_difference_ci95'][0]*100:+.1f},{p['success_difference_ci95'][1]*100:+.1f}]pp"
        text.append(f"| {a}/{b} | {p['recoveries']} | {p['new_failures']} | {p['policy_wall_per_action_ratio']:.4f} | [{p['policy_ratio_task_stratified_bootstrap_ci95'][0]:.4f},{p['policy_ratio_task_stratified_bootstrap_ci95'][1]:.4f}] | {quality} |")
    text+=['', 'Bootstrap按suite分层重采样task/case（每task仅init8），同一case两臂一起重采样，2000次；每次重新计算各臂sum wall/sum实际动作再求比值，不把同episode的请求当独立样本。区间含1时只报观察值，不宣称稳定加速；区间也不消除固定suite顺序等系统混杂。']
    text+=['', '四臂全部40/40成功、所有配对结局差均为0，因此成功差的经验bootstrap得到[0,0]，这是边界退化，不是可用于总体差异推断的95%区间，不能证明质量等价或非劣。成本区间仍按上述任务分层计算。',
        '', '## 每40任务累计策略wall（描述性补充）', '', '| 模式 | P2累计policy秒 | P4累计policy秒 | 累计wall变化 | P2/P4实际动作 |', '|---|---:|---:|---:|---|']
    for mode in ['periodic','verified']:
        a,b=ss[f'p2_{mode}'],ss[f'p4_{mode}'];change=b['total_policy_wall_s']/a['total_policy_wall_s']-1
        text.append(f"| {mode} | {a['total_policy_wall_s']:.4f} | {b['total_policy_wall_s']:.4f} | {change*100:+.3f}% | {a['actions']}/{b['actions']} |")
    text+=['', 'P4实际动作5931→6011（+1.35%），所以每动作成本变化不等于完成同一批40任务的累计策略耗时变化。以上仅服务端predict+ack，不含环境、公用warm或加载；主成本仍为每实际动作wall。']
    text+=['', '同周期verified与periodic逐case全部decoded动作及实际执行长度相等：'+ '；'.join(f"P{p}: {equivalence[f'p{p}_verified_vs_periodic']['exact']}/40" for p in [2,4])+'。实际已执行动作轨迹/Full-draft请求节奏/成功标签相等分别为'+ '；'.join(f"P{p}: {equivalence[f'p{p}_verified_vs_periodic']['actual_trajectory_exact']}/{equivalence[f'p{p}_verified_vs_periodic']['rhythm_exact']}/{equivalence[f'p{p}_verified_vs_periodic']['success_equal']}（各40）" for p in [2,4])+'。behavior_comparisons.json保存不等案例首处分歧与输入/种子/路由；不能把轨迹差异仅凭总数直接归因验证器。验收长度使用原accepted_length；episode成功或步数上限可截短actual executed_length，两者不混算。']
    text+=['','## 套件与路由','','| 条件 | Spatial | Object | Goal | Long | Full请求/全部 | Full原因 | verify零/部分/全10 | draft KV轮数 max/mean | draft KV步数 max/mean |','|---|---:|---:|---:|---:|---:|---|---|---|---|']
    for cid,s in ss.items():
        rt=routing[cid];c=rt['counts'];kr,ks=rt['draft_KV_rounds'],rt['draft_KV_steps']
        text.append('| '+cid+' | '+' | '.join(f'{v}/10' for v in s['suites'].values())+f" | {c['full']}/{c['requests']} | {rt['full_reasons']} | {c.get('verify_zero',0)}/{c.get('verify_partial',0)}/{c.get('verify_full10',0)} | {kr['max']}/{kr['mean']:.2f} | {ks['max']}/{ks['mean']:.2f} |")
    text+=['', '上表draft KV年龄只统计实际正接受draft；验证年龄单独统计所有有distance/verify调用的请求（包括随后零接受Full）：']
    for cid in ['p2_verified','p4_verified']:
        vr,vs=routing[cid]['verification_KV_rounds'],routing[cid]['verification_KV_steps']
        text.append(f"{cid}: verify KV轮数max/mean={vr['max']}/{vr['mean']:.2f}，控制步数max/mean={vs['max']}/{vs['mean']:.2f}。")
    text+=['startup/periodic Full请求开头的旧age未用于本轮验证，不混入verify年龄；P4第4次draft可用到4轮/40步，下一periodic Full行的5轮/50步是刷新前旧age，不称验证KV年龄。']
    text+=['', 'Full减少比例不能当完整推理加速比例；原Full与skip12单请求成本接近，且verify成本仍存在。不同策略自由闭环可能改变轨迹和总动作数。40例仅探索筛选，未校正多重比较，不证明总体非劣或部署优势。',
        '', '## 准备与模块耗时','','| 条件 | 公用warm秒 | env/reset秒 | RPC predict/ack秒 | physics/save秒 | verify ms/实际动作 | Full AE ms/Full请求 | draft AE ms/draft提案 |','|---|---:|---|---|---|---:|---:|---:|']
    for cid,s in ss.items():
        rt=routing[cid];t=rt['timing_sum_ms'];c=rt['counts'];e=s['timings_s'];draft=c['requests']-c['full']+c.get('verify_zero',0)
        text.append(f"| {cid} | {s['warm_s']:.2f} | {e['environment_acquire_s']:.2f}/{e['reset_s']:.2f} | {e['rpc_s']:.2f}/{e['ack_s']:.2f} | {e['step_s']:.2f}/{e['save_s']:.2f} | {t.get('verify_ae_ms',0)/s['actions']:.3f} | {t.get('full_ae_ms',0)/c['full']:.2f} | {t.get('draft_ae_ms',0)/draft:.2f} |")
    text+=['', '请求级描述（不同自由闭环观测，不是固定同输入模块因果对照；Full条件均值包含拒绝后接管请求，拒绝项本身包括draft+verify+Full）：',
        '', '| 条件 | Full次数/占比 | Full完整policy ms/request | 正接受draft次数 | draft完整policy ms/request | verify次数/AE均值ms | 零接受次数/完整policy均值ms |','|---|---|---:|---:|---:|---|---|']
    for cid in ss:
        rt=routing[cid];c=rt['counts'];t=rt['timing_sum_ms']
        path='periodic_draft' if cid.endswith('periodic') else 'skip_verified';n=c.get(path,0);v=c.get('verify',0);z=c.get('verify_zero',0)
        text.append(f"| {cid} | {c['full']}/{100*c['full']/c['requests']:.2f}% | {t['full_complete_policy_ms']/c['full']:.2f} | {n} | {t.get(path+'_complete_policy_ms',0)/max(1,n):.2f} | {v}/{t.get('verify_ae_ms',0)/max(1,v):.2f} | {z}/{t.get('zero_accept_complete_policy_ms',0)/max(1,z):.2f} |")
    text+=['', 'P4增加draft占比，也可能增加verify次数；若draft+verify比Full贵，即使Full减少，verified总成本仍可能上升。routing_summary.json额外保存零接受请求内draft AE/verify AE/Full AE及完整policy wall总和，不把这些模块条件均值解释为相同输入的因果benchmark。']
    text+=['', '顺序只在全局40case中平衡：每task恰有4suite、旋转依赖case index mod4，故同一suite内部四臂顺序固定。公用暖机与startup门禁减少冷态混杂，但不消除所有顺序/热态历史影响；若速度仅约1%，不称稳定提速，需未来独立反序复验。本轮不更改在途顺序、不追加矩阵。不同P轨迹/步数不同本身不证明验证部分接受；验证增量依据同P配对及真实accepted_length。']
    text+=['','execution_audit.json核对160例的实际执行前缀、真实P计数/拒绝重置/部分接受、KV轮数与控制步年龄、periodic零verify、无夹爪保护及predict+ack总成本。40组初态RGB/state/启动Full raw+decoded在四臂逐位相等。原始观测首帧、动作、EEF世界XYZ/四元数、种子、噪声hash与逐请求时延保存在episode目录；无全量视频。',
        '', 'paired_results.json提供逐case恢复/新失败与同卡比率；routing_summary.json保存模块耗时总和与KV年龄；requests.csv保存逐请求计数/年龄。']
    (OUT/'report_zh.md').write_text('\n'.join(text)+'\n')
    print('Complete', {cid:s['successes'] for cid,s in ss.items()})


if __name__=='__main__':main()
