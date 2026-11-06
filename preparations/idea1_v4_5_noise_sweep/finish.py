"""Audit the complete one-step sweep and summarize immutable historical controls."""
import json
import time
from pathlib import Path

import numpy as np

from preparations.idea1_v2_1.runtime_client import input_digest
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4.analyze import paired_ci, write_csv
from .common import ROOT, OUT, REPORT, SUITES, MASK, directory, key
from .common import all_conditions, sha


ATOL = RTOL = 1e-5


def read(path):
    return json.loads(Path(path).read_text())


def reference_directory(reference, case):
    return (Path(reference['root'])/reference['source_id']/'episodes'/case['suite']/
            f"task{case['task']:02d}"/f"init{case['init']:02d}")


def cost(row):
    return row['timing']['policy_wall_ms'] + row['timing'].get('policy_ack_ms', 0.)


def episode_record(condition, case, result, rows, identity, source, historical, kind, tau):
    assert rows and result['predictions'] == len(rows)
    assert result['case'] == identity['case'] == case
    hot = rows[3:]
    return dict(condition=condition, **case, historical=historical, kind=kind, tau=tau,
        success=bool(result['success']), actions=result['actions'], requests=len(rows),
        residual_refreshes=result['counts']['residual_refresh'], wall_s=result['wall_s'],
        policy_ms=sum(cost(r) for r in rows), hot_ms=sum(cost(r) for r in hot),
        hot_actions=sum(r['executed_length'] for r in hot),
        hot_ae_ms=sum(r['timing']['ae_ms'] for r in hot), hot_requests=len(hot),
        first_input_hash=rows[0]['input_hash'], first_noise_hash=rows[0]['model_noise_sha256'],
        first_seed=rows[0]['seed'], init_vector_sha256=identity['init_vector_sha256'],
        bddl_sha256=identity['bddl_sha256'], server_pid=identity['protocol']['server_pid'],
        source=str(source))


def audit_request(row, i, step, cfg, refresh_steps):
    """Keep the original numerical and causal boundaries, with tau from config."""
    route, timing = row['route'], row['timing']
    assert row['round'] == row['replan'] == i and row['step'] == step
    assert 0 < row['executed_length'] <= row['execute_length'] == 5
    refresh = i % cfg['refresh_interval'] == 0
    assert route['residual_refresh'] == refresh
    if refresh:
        refresh_steps.append(step)
    assert route['history_steps'] == refresh_steps[-2:]
    assert route['has_second_history'] == (len(refresh_steps) >= 2)
    assert route['D'] == 1 and route['skip'] == list(MASK)
    assert route['path'] == route['action_source'] == 'v12_d1'
    assert route['compensation'] == 'extrapolate'
    assert not route['full_ae_executed'] and not route['full_action_executed'] and not route['verifier']
    assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
    assert not timing['native_ae_graph']
    assert row['counts']['full'] == row['counts']['verify'] == 0
    assert row['counts']['draft'] == i+1 and row['counts']['residual_refresh'] == len(refresh_steps)
    ratio = 0. if len(refresh_steps) < 2 else (step-refresh_steps[-1])/(refresh_steps[-1]-refresh_steps[-2])
    assert np.isclose(route['extrapolation_ratio'], ratio)
    use_kf = cfg['initial'] == 'kalman' and i > 0
    assert route['initial_state'] == ('kalman6' if use_kf else 'gaussian')
    assert route['kalman_bootstrap'] == (cfg['initial'] == 'kalman' and i == 0)
    assert route['kalman_dimensions'] == (6 if cfg['initial'] == 'kalman' else 0)
    observations = step if cfg['initial'] == 'kalman' else 0
    assert route['kalman_observations'] == route['kalman_rotation_observations'] == observations
    assert route['kalman_source_last_step'] == (step-1 if use_kf else None)
    assert row['counts']['kalman'] == (i if cfg['initial'] == 'kalman' else 0)
    actual_start = cfg['start_time'] if use_kf else 1.
    assert route['actual_start_time'] == actual_start
    assert route['warmstart_time'] == actual_start
    assert route['warmstart_noise_scale'] == cfg['noise_scale']
    assert route['actual_noise_std'] == actual_start*cfg['noise_scale']
    assert route['kalman_prior_weight'] == (1-actual_start if use_kf else 0.)
    assert route['fast_time_embedding'] == timing['time_embedding'] == 'openpi_linspace'
    nodes = timing['nodes']
    assert nodes['actual_NFE'] == nodes['NFE'] == 1
    assert nodes['nodes'] == [actual_start, 0.]
    assert nodes['timesteps'] == [actual_start] and nodes['dt'] == [-actual_start]
    return refresh


def request_record(condition, case, row, historical):
    timing = row['timing']
    return dict(condition=condition, **case, historical=historical, request=row['round'],
        step=row['step'], executed=row['executed_length'], refresh=row['route']['residual_refresh'],
        initial_state=row['route'].get('initial_state', 'gaussian'),
        actual_start_time=row['route'].get('actual_start_time', 1.),
        actual_NFE=timing['nodes']['actual_NFE'], policy_ms=cost(row), rpc_ms=row['rpc_ms'],
        ae_ms=timing['ae_ms'], kalman_predict_ms=timing.get('kalman_predict_ms', 0.),
        kalman_encode_ms=timing.get('kalman_encode_ms', 0.), ack_ms=timing.get('policy_ack_ms', 0.),
        refresh_vlm_ms=timing.get('refresh_vlm_ms', 0.),
        refresh_vision_ms=timing.get('refresh_vision_ms', 0.))


def summarize(records, metadata):
    summary = []
    for condition, group_records in records.items():
        for suite in ('all', *SUITES):
            group = [r for r in group_records.values() if suite == 'all' or r['suite'] == suite]
            assert group
            total = lambda field: sum(r[field] for r in group)
            n, actions, wall = len(group), total('actions'), total('wall_s')
            hot_requests, hot_actions = total('hot_requests'), total('hot_actions')
            summary.append(dict(condition=condition, suite=suite, n=n,
                historical=metadata[condition]['historical'], kind=metadata[condition]['kind'],
                tau=metadata[condition]['tau'], success=total('success'),
                success_rate=total('success')/n, policy_ms_per_action=total('policy_ms')/actions,
                hot_ms_per_action=total('hot_ms')/hot_actions if hot_actions else None,
                policy_ms_per_request=total('policy_ms')/total('requests'),
                mean_ae_ms=total('hot_ae_ms')/hot_requests if hot_requests else None,
                mean_actions=actions/n, wall_s=wall, wall_s_per_episode=wall/n,
                episodes_per_hour=3600*n/wall, actions_per_second=actions/wall,
                residual_refreshes=total('residual_refreshes'), full_actions=0,
                full_ae_calls=0, verifier_calls=0))
    return summary


def comparison(baseline, candidate, a, b, keys, scope):
    pairs = [dict(suite=a[k]['suite'], task=a[k]['task'],
                  delta=int(b[k]['success'])-int(a[k]['success'])) for k in keys]
    ci = paired_ci(pairs) if pairs else None
    return dict(baseline=baseline, candidate=candidate, scope=scope, n=len(keys),
        baseline_success=sum(a[k]['success'] for k in keys),
        candidate_success=sum(b[k]['success'] for k in keys),
        recovered=sum(not a[k]['success'] and b[k]['success'] for k in keys),
        lost=sum(a[k]['success'] and not b[k]['success'] for k in keys),
        delta_pp=100*float(np.mean([p['delta'] for p in pairs])) if pairs else None,
        ci_low_pp=100*ci[0] if ci is not None else None,
        ci_high_pp=100*ci[1] if ci is not None else None,
        historical_comparison=True, timing_comparison=False)


def first_check(baseline, candidate, a, b, raw_first, case, kind):
    k = key(case)
    aa, bb = a[k], b[k]
    # An initial observation mismatch is a protocol fault; an output mismatch is
    # retained and disclosed rather than widening tolerances or rerunning outcomes.
    assert aa['first_input_hash'] == bb['first_input_hash']
    assert aa['init_vector_sha256'] == bb['init_vector_sha256']
    assert aa['bddl_sha256'] == bb['bddl_sha256']
    assert aa['first_seed'] == bb['first_seed']
    assert aa['first_noise_hash'] == bb['first_noise_hash']
    x, y = raw_first[baseline, k], raw_first[candidate, k]
    checks = {}
    for field, shape in (('raw', (50, 32)), ('actions', (50, 7))):
        assert x[field].shape == y[field].shape == shape
        assert np.isfinite(x[field]).all() and np.isfinite(y[field]).all()
        checks[field+'_passed'] = bool(np.allclose(x[field], y[field], atol=ATOL, rtol=RTOL))
        checks[field+'_max_abs'] = float(np.abs(x[field]-y[field]).max())
    return dict(baseline=baseline, candidate=candidate, case=case, kind=kind,
        passed=all(checks[field+'_passed'] for field in ('raw', 'actions')),
        **checks, atol=ATOL, rtol=RTOL)


def finish():
    manifest = read(OUT/'manifest.json')
    assert all(sha(ROOT/p) == expected for p, expected in manifest['source_files'].items()), 'Frozen source changed'
    assert all(sha(p) == expected for p, expected in manifest['reference_files'].items()), 'Historical artifact changed'
    cases = manifest['cases']
    assert len(cases) == 40 and len({key(c) for c in cases}) == 40
    assert manifest['target_episodes'] == len(cases)*len(manifest['conditions']) == 200
    records, metadata, raw_first, flat, requests = {}, {}, {}, [], []
    checked, noise_checks = 0, {}
    used_service_pids = set()
    for cfg in manifest['conditions']:
        name = cfg['id']
        assert cfg['D'] == 1 and cfg['initial'] == 'kalman' and cfg['noise_scale'] == 1.
        metadata[name] = dict(historical=False, kind='kalman_d1', tau=cfg['start_time'])
        records[name] = {}
        for case in cases:
            d = directory(name, case)
            complete, result, identity = [read(d/f) for f in ('complete.json', 'result.json', 'identity.json')]
            assert complete['valid'] and complete['manifest_hash'] == manifest['manifest_hash']
            assert result['valid'] and result['manifest_hash'] == manifest['manifest_hash'] and result['config'] == cfg
            assert identity['manifest_hash'] == manifest['manifest_hash'] and identity['config'] == cfg
            assert result['termination'] in ('success', 'timeout') and (result['termination'] == 'success') == result['success']
            assert identity['env_seed'] == 7 and identity['settling'] == 10
            rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
            refresh_steps, steps = [], 0
            for i, row in enumerate(rows):
                audit_request(row, i, steps, cfg, refresh_steps)
                steps += row['executed_length']
                noise_key = (key(case), i)
                noise = (row['seed'], row['model_noise_sha256'])
                assert noise_checks.setdefault(noise_key, noise) == noise, 'Paired base noise changed'
                requests.append(request_record(name, case, row, False))
                checked += 1
            assert steps == result['actions']
            first_input = d/'input_0000.npz'
            assert sha(first_input) == rows[0]['input_file_sha256']
            with np.load(first_input) as saved_input:
                obs = {name:saved_input[name] for name in saved_input.files if name not in ('instruction', 'seed')}
                assert input_digest(obs, str(saved_input['instruction'])) == rows[0]['input_hash']
                assert int(saved_input['seed']) == rows[0]['seed']
            assert result['counts'] == rows[-1]['counts']
            record = episode_record(name, case, result, rows, identity, d, False, 'kalman_d1', cfg['start_time'])
            records[name][key(case)] = record
            flat.append(record)
            used_service_pids.add(record['server_pid'])
            with np.load(d/'arrays.npz') as saved:
                raw_first[name, key(case)] = dict(raw=saved['raw_0'].copy(), actions=saved['actions_0'].copy())
    assert len(flat) == manifest['target_episodes']
    references = []
    for reference in manifest['references']:
        name = reference['id']
        assert name not in records
        cfg = reference['config']
        tau = cfg.get('start_time', 1.) if reference['kind'] == 'kalman_d1' else None
        metadata[name] = dict(historical=True, kind=reference['kind'], tau=tau)
        records[name] = {}
        for case in cases:
            d = reference_directory(reference, case)
            complete, result, identity = [read(d/f) for f in ('complete.json', 'result.json', 'identity.json')]
            assert complete['valid'] and result['valid']
            assert complete['manifest_hash'] == result['manifest_hash'] == identity['manifest_hash'] == reference['manifest_hash']
            assert result['config'] == identity['config'] == cfg
            rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
            record = episode_record(name, case, result, rows, identity, d, True, reference['kind'], tau)
            records[name][key(case)] = record
            flat.append(record)
            requests.extend(request_record(name, case, row, True) for row in rows)
            if reference['kind'] != 'gaussian_d2':
                with np.load(d/'arrays.npz') as saved:
                    raw_first[name, key(case)] = dict(raw=saved['raw_0'].copy(), actions=saved['actions_0'].copy())
        references.append(dict(reference, n=len(records[name]),
            success=sum(r['success'] for r in records[name].values()), historical=True,
            timing_comparison=False, suites={s:sum(r['success'] for r in records[name].values() if r['suite'] == s) for s in SUITES}))
    first_checks, recoveries = [], []
    for case in cases:
        case_key = key(case)
        pids = sorted({records[cfg['id']][case_key]['server_pid'] for cfg in manifest['conditions']})
        done = read(OUT/'case_done'/f'{case_key}.json')
        assert done['passed'] and done['case'] == case
        assert done['atol'] == ATOL and done['rtol'] == RTOL and done['server_pids'] == pids
        assert len(done['checks']) == 2*len(manifest['conditions'])
        assert {(c['condition'], c['array']) for c in done['checks']} == {(cfg['id'], field) for cfg in manifest['conditions'] for field in ('raw_0', 'actions_0')}
        assert all(check['passed'] for check in done['checks'])
        if len(pids) != 1:
            recovery = read(OUT/'case_recovery'/f'{case_key}.json')
            assert recovery == done['technical_recovery']
            assert recovery['reason'] == 'resource_oom' and recovery['attempt'] == 1
            assert recovery['case'] == case and recovery['valid_results_preserved']
            assert len(pids) == 2 and recovery['previous_server_pid'] in pids
            assert (OUT/recovery['archive']).is_dir()
            recoveries.append(dict(case=case, server_pids=pids, **{k:v for k,v in recovery.items() if k != 'case'}))
    anchor = manifest['conditions'][0]['id']
    for cfg in manifest['conditions'][1:]:
        for case in cases:
            check = first_check(anchor, cfg['id'], records[anchor], records[cfg['id']], raw_first, case, 'new_same_service')
            check['same_service'] = records[anchor][key(case)]['server_pid'] == records[cfg['id']][key(case)]['server_pid']
            assert check['passed'], 'New bootstrap output mismatch'
            first_checks.append(check)
    gaussian = next(r['id'] for r in manifest['references'] if r['kind'] == 'gaussian_d1')
    kf75 = next(r['id'] for r in manifest['references'] if r['kind'] == 'kalman_d1')
    tau_configs = all_conditions()
    assert len(tau_configs) == 6 and all(cfg['id'] in records for cfg in tau_configs)
    paired = []
    for cfg in tau_configs:
        candidate = cfg['id']
        assert metadata[candidate]['tau'] == cfg['start_time']
        for baseline in (gaussian, kf75):
            checks = [first_check(baseline, candidate, records[baseline], records[candidate], raw_first, case,
                                  'historical_bootstrap') for case in cases]
            first_checks.extend(checks)
            eligible = [key(c['case']) for c in checks if c['passed']]
            all_keys = [key(c) for c in cases]
            paired.append(comparison(baseline, candidate, records[baseline], records[candidate], all_keys, 'all_cases_descriptive'))
            paired.append(comparison(baseline, candidate, records[baseline], records[candidate], eligible, 'initial_raw_matched'))
    workers = sorted((OUT/'workers').glob('gpu*')) + sorted((OUT/'attempts').glob('resource_conflict_*/gpu*'))
    assert workers
    preparation = []
    seen_service_pids = set()
    for worker in workers:
        exits = read(worker/'exits.json')
        assert exits['released'], f'Worker resource not released: {worker}'
        protocol = read(worker/'protocol.json') if (worker/'protocol.json').exists() else {}
        used = protocol.get('server_pid') in used_service_pids
        preflight = read(worker/'preflight.json') if (worker/'preflight.json').exists() else {}
        if used:
            assert preflight['passed'] and read(worker/'cross_service.json')['passed']
            seen_service_pids.add(protocol['server_pid'])
        preparation.append(dict(worker=str(worker.relative_to(OUT)), used_for_evaluation=used, released=True,
            historical_gate_passed=preflight.get('historical_gate_passed'),
            historical_gate_checks=len(preflight.get('historical_checks', [])),
            historical_gate_checks_passed=sum(c['passed'] for c in preflight.get('historical_checks', [])), **protocol))
    assert seen_service_pids == used_service_pids
    assert all(sha(ROOT/p) == expected for p, expected in manifest['source_files'].items())
    assert all(sha(p) == expected for p, expected in manifest['reference_files'].items())
    summary = summarize(records, metadata)
    for name, rows in [('episodes.csv', flat), ('summary.csv', summary), ('requests.csv', requests),
                       ('paired.csv', paired), ('initial_checks.csv', [dict(c, case=key(c['case'])) for c in first_checks])]:
        write_csv(OUT/name, rows)
    group = {(r['condition'], r['suite']):r for r in summary}
    total = lambda name: group[name, 'all']
    descriptions = {gaussian:'高斯＋一步 AE（历史 v4.4）', kf75:'Kalman τ=0.75（历史 v4.4）'}
    d2 = next(r['id'] for r in manifest['references'] if r['kind'] == 'gaussian_d2')
    descriptions[d2] = '高斯＋两步 AE（历史 v4.2）'
    lines = ['# Kalman 起点：六种起始时间的纯一步闭环', '',
        '新测 τ=0.05、0.10、0.20、0.35、0.50，各40条，共200条；τ=0.75复用v4.4的40条。高斯D1复用v4.4、高斯D2复用v4.2各40条，只读核验历史结果及数组哈希。四个LIBERO套件各10任务，官方init19、seed7、settling10。', '',
        '固定V12跳层[2,3,5,14,16,17]、自身残差外推与每三轮刷新保持不变。Full仅运行完整VLM提取真残差，再重跑V12生成动作；没有Full AE、Full动作、验证、夹爪保护或回退。每次请求H50/R5，一步AE，首轮因无执行历史，所有条件使用相同高斯D1启动。encoder/AE均不使用CUDA Graph。', '',
        '所有Kalman条件ρ=1，模型归一化动作空间中 xτ=(1−τ)×KF_raw+τ×ε，ε~N(0,I)，一步Euler从τ走到0。τ同时改变Kalman权重、噪声标准差和积分步长；这不是只改变噪声的独立因果实验。位移及旋转增量只根据ack的实际执行命令更新，夹爪保持；没有测试干净Kalman草稿或零噪声一步修正。', '',
        '|条件|Spatial|Object|Goal|Long|总成功率|来源|',
        '|---|---|---|---|---|---|---|']
    for name in [gaussian, *[cfg['id'] for cfg in tau_configs], d2]:
        row = total(name)
        label = descriptions[name] if name in descriptions else f"Kalman τ={metadata[name]['tau']:.2f}"
        values = [label, *[f"{group[name,s]['success']}/10" for s in SUITES],
                  f"{row['success']}/40（{100*row['success_rate']:.1f}%）", '历史复用' if row['historical'] else '本轮新测']
        lines.append('|'+'|'.join(values)+'|')
    lines += ['', '下表保留全部40个任务的描述性配对；恢复/新增失败按同任务、同官方初态计算。严格配对另要求首轮高斯输出通过原atol=rtol=1e-5；历史跨服务输出超容差时保留有效闭环，但从严格配对剔除，未改变容差或依据结果重跑。区间为suite分层、task聚类bootstrap的探索95%区间，没有多重比较校正。', '',
        '|τ|参照|恢复|新增失败|全40例差值pp|探索95%区间pp|严格匹配n|严格匹配差值pp|',
        '|---|---|---|---|---|---|---|---|']
    for cfg in tau_configs:
        for baseline, label in ((gaussian, '高斯D1'), (kf75, 'Kalman τ=0.75')):
            full = next(p for p in paired if p['baseline'] == baseline and p['candidate'] == cfg['id'] and p['scope'] == 'all_cases_descriptive')
            strict = next(p for p in paired if p['baseline'] == baseline and p['candidate'] == cfg['id'] and p['scope'] == 'initial_raw_matched')
            interval = 'NA' if full['ci_low_pp'] is None else f"[{full['ci_low_pp']:+.1f},{full['ci_high_pp']:+.1f}]"
            strict_delta = 'NA' if strict['delta_pp'] is None else f"{strict['delta_pp']:+.1f}"
            lines.append(f"|{cfg['start_time']:.2f}|{label}|{full['recovered']}|{full['lost']}|{full['delta_pp']:+.1f}|{interval}|{strict['n']}|{strict_delta}|")
    lines += ['', f"本轮同一case的五个新条件按轮换顺序交替执行，正常路径在同一服务内；{len(recoveries)}个case因归档的GPU资源OOM恢复而跨服务续跑，保留已有效回合，未依据任务成败重跑。所有新组首轮输入、初态、噪声身份、raw及解码动作均通过原容差。历史参照与本轮不在同一次服务内，不能把以下耗时解释为受控部署加速比。", '',
        '这是每设置40例的探索性参数扫描，不是独立确认集。若根据本表选择冠军，其成功率会有选择偏差；参数此前也曾在相关任务评估，不应称为未见任务泛化结果。该结论限定固定V12＋残差刷新设置，不能混入此前有验证与Full接管的38/40。', '',
        '|条件|热态策略ms/动作|热态AE ms/请求|平均执行步数|闭环墙钟s|episode/h|动作/s|',
        '|---|---|---|---|---|---|---|']
    for name in [gaussian, *[cfg['id'] for cfg in tau_configs], d2]:
        row = total(name)
        label = descriptions[name] if name in descriptions else f"Kalman τ={metadata[name]['tau']:.2f}"
        lines.append(f"|{label}|{row['hot_ms_per_action']:.3f}|{row['mean_ae_ms']:.3f}|{row['mean_actions']:.1f}|{row['wall_s']:.2f}|{row['episodes_per_hour']:.2f}|{row['actions_per_second']:.2f}|")
    historical_checks = [c for c in first_checks if c['kind'] == 'historical_bootstrap']
    historical_gates = [p for p in preparation if p['historical_gate_passed'] is not None]
    lines += ['', '热态从每回合第4次请求开始，包含实际视觉/VLM、周期完整VLM刷新、残差、AE、解码及ack滤波更新；历史v4.2的D2原日志未记录策略内ack耗时，表中D2沿用原计时边界，不含该项。CPU预测和坐标编码在当前实现串行计时，可提前或与视觉计算重叠；这不是算法不可消除的GPU成本。20Hz是仿真控制频率，不以墙钟sleep；自由运行轨迹不同，墙钟吞吐仅为描述。', '',
        '服务加载及GPU机制检查：'+'; '.join(f"{p['worker']} {p.get('load_and_preflight_s',0.):.2f}s（{'参与评测' if p['used_for_evaluation'] else '未参与评测'}）" for p in preparation)+'。', '',
        f"历史机制门禁通过{sum(p['historical_gate_passed'] for p in historical_gates)}/{len(historical_gates)}个服务；这是固定门禁输入的记录，不替代实际case的首轮历史数值匹配。", '',
        f"审查全部200条新闭环、{checked}次新请求，实际NFE均为1，Full AE/Full动作/验证均为0；Kalman观察计数严格等于已执行步。历史首轮比较通过{sum(c['passed'] for c in historical_checks)}/{len(historical_checks)}项，逐项误差和剔除记录见initial_checks.csv；相同历史参照间及自比较也列入该计数。源文件和历史文件哈希未变，所有自有服务与GPU锁已释放。", '',
        '[逐回合](../../../results/idea1_v4_5_noise_sweep/episodes.csv) · [汇总](../../../results/idea1_v4_5_noise_sweep/summary.csv) · [配对](../../../results/idea1_v4_5_noise_sweep/paired.csv) · [初态数值核验](../../../results/idea1_v4_5_noise_sweep/initial_checks.csv)']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'references.json', references)
    atomic(OUT/'preparation.json', preparation)
    atomic(OUT/'initial_checks.json', first_checks)
    atomic(OUT/'resource_recoveries.json', recoveries)
    atomic(OUT/'audit.json', dict(passed=True, new_episodes=200,
        reused_episodes=sum(len(records[r['id']]) for r in manifest['references']), route_checks=checked,
        first_checks=len(first_checks), initial_new_passed=True, resource_recovery_cases=len(recoveries),
        historical_raw_matched=sum(c['passed'] for c in historical_checks),
        historical_raw_checks=len(historical_checks), atol=ATOL, rtol=RTOL,
        full_actions=0, full_ae_calls=0, verifier_calls=0,
        sources_match=True, references_unchanged=True, all_workers_released=True, time=time.time()))
    atomic(OUT/'complete.json', dict(passed=True, new_episodes=200, report=str(REPORT), time=time.time()))
    atomic(OUT/'status.json', dict(status='complete', completed=200, target=200,
        summary=summary, paired=paired, time=time.time()))


if __name__ == '__main__':
    finish()
