import json
import time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import write_csv, paired_ci
from .common import ROOT, OUT, REPORT, SUITES, MASK, directory, key, sha


def finish():
    manifest = json.loads((OUT/'manifest.json').read_text())
    assert all(sha(ROOT/p)==expected for p,expected in manifest['source_files'].items())
    assert all(sha(p)==expected for p,expected in manifest['reference_files'].items())
    data, flat, summary, requests = {}, [], [], []
    checked = 0
    for cfg in manifest['conditions']:
        records = []
        for case in manifest['cases']:
            d = directory(cfg['id'],case)
            assert (d/'complete.json').exists()
            result = json.loads((d/'result.json').read_text())
            assert result['valid'] and result['manifest_hash']==manifest['manifest_hash'] and result['config']==cfg
            rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
            refresh_steps, total_steps = [], 0
            for i, row in enumerate(rows):
                route, timing = row['route'], row['timing']
                assert row['round']==i and row['step']==total_steps and 0<row['executed_length']<=5
                total_steps += row['executed_length']
                refresh = i%3==0
                assert route['residual_refresh']==refresh
                if refresh:
                    refresh_steps.append(row['step'])
                assert route['history_steps']==refresh_steps[-2:]
                assert route['D']==1 and route['skip']==list(MASK) and route['action_source']=='v12_d1'
                assert not route['full_ae_executed'] and not route['full_action_executed'] and not route['verifier']
                assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
                assert row['counts']['full']==row['counts']['verify']==0
                assert row['counts']['draft']==i+1 and row['counts']['residual_refresh']==len(refresh_steps)
                ratio = 0. if len(refresh_steps)<2 else (row['step']-refresh_steps[-1])/(refresh_steps[-1]-refresh_steps[-2])
                assert np.isclose(route['extrapolation_ratio'],ratio)
                use_kf = cfg['initial']=='kalman' and i>0
                assert route['initial_state']==('kalman6' if use_kf else 'gaussian')
                assert route['kalman_bootstrap']==(cfg['initial']=='kalman' and i==0)
                assert route['kalman_observations']==route['kalman_rotation_observations']==(row['step'] if cfg['initial']=='kalman' else 0)
                assert route['kalman_source_last_step']==(row['step']-1 if use_kf else None)
                assert row['counts']['kalman']==(i if cfg['initial']=='kalman' else 0)
                actual_start = .75 if use_kf else 1.
                assert route['actual_start_time']==actual_start
                assert timing['nodes']['actual_NFE']==timing['nodes']['NFE']==1 and timing['time_embedding']=='openpi_linspace'
                assert timing['nodes']['nodes']==[actual_start,0.]
                requests.append(dict(condition=cfg['id'],**case,request=i,step=row['step'],executed=row['executed_length'],refresh=refresh,
                    initial_state=route['initial_state'],policy_ms=timing['policy_wall_ms']+timing['policy_ack_ms'],
                    rpc_ms=row['rpc_ms'],ae_ms=timing['ae_ms'],kalman_predict_ms=timing.get('kalman_predict_ms',0.),
                    kalman_encode_ms=timing.get('kalman_encode_ms',0.),ack_ms=timing['policy_ack_ms'],
                    refresh_vlm_ms=timing.get('refresh_vlm_ms',0.),refresh_vision_ms=timing.get('refresh_vision_ms',0.)))
                checked += 1
            assert total_steps==result['actions']
            hot = rows[3:]
            cost = lambda r:r['timing']['policy_wall_ms']+r['timing']['policy_ack_ms']
            row = dict(condition=cfg['id'],**case,success=result['success'],actions=result['actions'],requests=len(rows),
                residual_refreshes=result['counts']['residual_refresh'],wall_s=result['wall_s'],policy_ms=sum(cost(r) for r in rows),
                hot_ms=sum(cost(r) for r in hot),hot_actions=sum(r['executed_length'] for r in hot),
                first_input_hash=rows[0]['input_hash'],source=str(d))
            records.append(row)
            flat.append(row)
        data[cfg['id']] = {key(r):r for r in records}
        for suite in ('all',*SUITES):
            group = [r for r in records if suite=='all' or r['suite']==suite]
            total = lambda field:sum(r[field] for r in group)
            reqs = [r for r in requests if r['condition']==cfg['id'] and (suite=='all' or r['suite']==suite)]
            summary.append(dict(condition=cfg['id'],suite=suite,n=len(group),success=sum(r['success'] for r in group),
                policy_ms_per_action=total('policy_ms')/total('actions'),hot_ms_per_action=total('hot_ms')/total('hot_actions'),
                policy_ms_per_request=total('policy_ms')/total('requests'),mean_ae_ms=float(np.mean([r['ae_ms'] for r in reqs if r['request']>=3])),
                mean_actions=total('actions')/len(group),wall_s=total('wall_s'),wall_s_per_episode=total('wall_s')/len(group),
                episodes_per_hour=3600*len(group)/total('wall_s'),actions_per_second=total('actions')/total('wall_s'),
                residual_refreshes=total('residual_refreshes'),full_actions=0,full_ae_calls=0,verifier_calls=0))
    baseline,candidate = [cfg['id'] for cfg in manifest['conditions']]
    a,b = data[baseline],data[candidate]
    assert set(a)==set(b) and len(a)==40
    assert all(a[k]['first_input_hash']==b[k]['first_input_hash'] for k in a)
    initial_raw_checks = []
    for case in manifest['cases']:
        with np.load(directory(baseline,case)/'arrays.npz') as aa, np.load(directory(candidate,case)/'arrays.npz') as bb:
            passed = bool(np.allclose(aa['raw_0'],bb['raw_0'],atol=1e-5,rtol=1e-5))
            initial_raw_checks.append(dict(case=case,passed=passed,max_abs=float(np.abs(aa['raw_0']-bb['raw_0']).max())))
            assert passed
    pairs = [dict(suite=a[k]['suite'],task=a[k]['task'],delta=int(b[k]['success'])-int(a[k]['success'])) for k in a]
    ci = paired_ci(pairs)
    paired = dict(baseline=baseline,candidate=candidate,n=40,
        recovered=sum(not a[k]['success'] and b[k]['success'] for k in a),lost=sum(a[k]['success'] and not b[k]['success'] for k in a),
        delta_pp=100*np.mean([p['delta'] for p in pairs]),ci_low_pp=100*ci[0],ci_high_pp=100*ci[1])
    old_records = []
    for case in manifest['cases']:
        d = Path(manifest['reference_root'])/manifest['reference_id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        r = json.loads((d/'result.json').read_text())
        old_records.append(dict(**case,success=r['success']))
    reference = dict(id=manifest['reference_id'],n=40,success=sum(r['success'] for r in old_records),
        suites={s:sum(r['success'] for r in old_records if r['suite']==s) for s in SUITES},historical=True,
        timing_comparison=False,manifest_hash=manifest['reference_manifest_hash'])
    workers = sorted((OUT/'workers').glob('gpu*'))
    assert workers and all(json.loads((w/'exits.json').read_text())['released'] for w in workers)
    assert all(json.loads((w/'preflight.json').read_text())['passed'] and json.loads((w/'cross_service.json').read_text())['passed'] for w in workers)
    preparation = [dict(worker=w.name,**json.loads((w/'protocol.json').read_text())) for w in workers]
    for name, rows in [('episodes.csv',flat),('summary.csv',summary),('requests.csv',requests),('paired.csv',[paired])]:
        write_csv(OUT/name,rows)
    lines = ['# Kalman起点与纯一步AE：Full只刷新残差', '',
        '新跑两组各40条，四个LIBERO套件各10任务、官方init19。固定V12跳层[2,3,5,14,16,17]，沿用v4.2提取子路径；没有动态选层或反传。两组模型、mask、残差日程、噪声身份、环境初始化一致。', '',
        '所有实际动作均为V12+D1，包括首轮和残差刷新轮；Full只运行完整VLM提取残差，没有Full AE、Full动作、验证器、夹爪保护或拒绝回退。每三轮完整VLM刷新后重跑V12，用自身最新两份真残差按实际age/gap线性外推，首次周期hold。H50/R5、20Hz仿真、seed7、settling10，修正OpenPI时间编码，encoder/AE均无CUDA Graph。', '',
        '高斯组t=1→0一步Euler。Kalman组冻结v4.3六维冠军tau=.75、rho=1，x_tau=.25*KF_raw+.75*epsilon，一步.75→0；位移和旋转增量只来自ack的已执行命令，夹爪保持。首轮无历史，两组使用完全相同的高斯D1，之后没有任何Full引导动作。', '',
        '|条件|Spatial|Object|Goal|LIBERO-10|合计|热态策略ms/动作|热态AE ms/请求|平均执行步数|',
        '|---|---|---|---|---|---|---|---|---|']
    for cfg in manifest['conditions']:
        group = {r['suite']:r for r in summary if r['condition']==cfg['id']}
        all_row = group['all']
        values = [cfg['id'],*[f"{group[s]['success']}/10" for s in SUITES],f"{all_row['success']}/40",
            f"{all_row['hot_ms_per_action']:.3f}",f"{all_row['mean_ae_ms']:.3f}",f"{all_row['mean_actions']:.1f}"]
        lines.append('|'+'|'.join(values)+'|')
    lines += ['', f"Kalman相对高斯：恢复{paired['recovered']}例、新增失败{paired['lost']}例；成功率差{paired['delta_pp']:+.1f}个百分点，suite分层task聚类探索95%区间[{paired['ci_low_pp']:+.1f},{paired['ci_high_pp']:+.1f}]。", '',
        f"历史同固定子路径、仅刷新残差的高斯D2：{reference['success']}/40（四套件分别{', '.join(str(reference['suites'][s])+'/10' for s in SUITES)}）。该结果只读复用v4.2，未冒充同轮延迟对照。此前带验证与Full接管的Kalman38/40属于另一种策略，不能混入本轮纯一步成功率。", '',
        '这是40例探索评测；Kalman参数此前曾使用相关任务选择，本轮没有重新调参，不能当作未见任务独立统计确认。该实验评价固定V12子路径的一步动作质量，不能直接外推为完整VLM起点的结论。', '',
        '热态从每回合第4次请求开始，包含实际刷新、视觉/VLM、残差处理、AE、解码以及ack滤波更新。预测和坐标编码在本实现串行计入；可与视觉计算重叠的CPU部分未做工程优化，不能作为算法不可消除的成本。与v4.2一样20Hz是仿真控制频率，不人为以墙钟sleep；各自自由轨迹计时不构成相同输入的受控部署加速比。', '',
        '|条件|闭环总墙钟s|episode/h|动作/s|', '|---|---|---|---|']
    for r in summary:
        if r['suite']=='all':
            lines.append(f"|{r['condition']}|{r['wall_s']:.2f}|{r['episodes_per_hour']:.2f}|{r['actions_per_second']:.2f}|")
    lines += ['', '服务加载及GPU机制检查：'+ '; '.join(f"{p['worker']} {p['load_and_preflight_s']:.2f}s" for p in preparation)+'.', '',
        f"已审查全部80条、{checked}次请求，实际NFE均为1，Full AE/Full动作/验证均为0。40组初态和首轮高斯输出在原atol=rtol=1e-5内一致；Kalman计数只对应已执行步，所有自有服务与GPU锁均已释放。", '',
        '[逐回合](../../../results/idea1_v4_4_one_step/episodes.csv) · [逐请求](../../../results/idea1_v4_4_one_step/requests.csv) · [配对](../../../results/idea1_v4_4_one_step/paired.csv)']
    REPORT.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'reference.json',reference)
    atomic(OUT/'preparation.json',preparation)
    atomic(OUT/'audit.json',dict(passed=True,episodes=80,route_checks=checked,initial_pairs=40,
        initial_raw_checks=initial_raw_checks,full_actions=0,full_ae_calls=0,verifier_calls=0,
        sources_match=True,reference_unchanged=True,time=time.time()))
    atomic(OUT/'complete.json',dict(passed=True,report=str(REPORT),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',completed=80,target=80,summary=summary,paired=paired,time=time.time()))


if __name__=='__main__': finish()
