"""Audit the 16-episode gripper-noise intervention without changing outcomes."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from preparations.idea1_v2_1.runtime_client import input_digest
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v4_5_noise_sweep.finish import (
    ATOL, RTOL, audit_request as audit_base_request, comparison, episode_record,
    first_check, read, reference_directory, request_record, summarize,
)
from .common import ROOT, OUT, REPORT, SUITES, MASK, directory, key, sha


def audit_request(row, index, step, config, refresh_steps):
    """Keep motion, clock, NFE, residual, and causal-ack checks unchanged."""
    audit_base_request(row, index, step, config, refresh_steps)
    route = row['route']
    assert config['gripper_initial'] == route['gripper_initial'] == 'standard_gaussian'
    assert config['gripper_noise_std'] == route['actual_gripper_noise_std'] == 1.
    assert route['gripper_prior_weight'] == 0.
    assert route['configured_gripper_initial'] == config['gripper_initial']
    assert route['motion_noise_std'] == route['actual_noise_std']
    assert route['gripper_denoising_span'] == route['actual_start_time']
    assert route['heterogeneous_noise'] == (index > 0)
    epsilon = np.random.default_rng(row['seed']).standard_normal((50, 32), dtype=np.float32)
    assert hashlib.sha256(epsilon.tobytes()).hexdigest() == row['model_noise_sha256']
    return epsilon


def audit_initial_prefix(saved, index, config, epsilon):
    """Saved prefix records are checked against the actual deterministic mixing rule."""
    names = [f'{field}_{index}' for field in ('kalman_encoded_prefix', 'kalman_initial_prefix')]
    if index == 0:
        assert not any(name in saved for name in names), 'Bootstrap must stay ordinary Gaussian D1'
        return None
    encoded, actual = (np.asarray(saved[name]) for name in names)
    assert encoded.shape == actual.shape == (5, 32)
    assert np.isfinite(encoded).all() and np.isfinite(actual).all()
    assert np.array_equal(encoded[:, 7:], np.zeros((5, 25), np.float32))
    tau, rho = config['start_time'], config['noise_scale']
    expected = ((1-tau)*encoded + rho*tau*epsilon[:5]).astype(np.float32)
    expected[:, 6] = epsilon[:5, 6]
    # This is CPU state construction before BF16 AE, not a relaxed output check.
    assert np.array_equal(actual, expected), 'Unexpected change outside the gripper intervention'
    assert np.array_equal(actual[:, 6], epsilon[:5, 6]), 'Gripper must use independent sigma=1 noise'
    return dict(request=index, motion_exact=True, padding_exact=True, gripper_exact=True,
                prefix_shape=[5, 32], actual_gripper_noise_std=1., gripper_prior_weight=0.)


def gripper_record(condition, case, rows, saved, source, historical):
    """Use only actions actually executed, including a possibly shortened final prefix."""
    executed = []
    for index, row in enumerate(rows):
        actions = saved[f'actions_{index}']
        assert actions.shape == (50, 7) and np.isfinite(actions).all()
        executed.append(actions[:row['executed_length']])
    executed = np.concatenate(executed)
    telemetry_path = Path(source)/'telemetry.jsonl'
    telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    assert telemetry[0]['step'] == 0 and telemetry[0]['action'] is None
    actual = np.asarray([record['action'] for record in telemetry[1:]], np.float32)
    assert actual.shape == executed.shape and np.array_equal(actual, executed)
    assert [record['step'] for record in telemetry] == list(range(len(executed)+1))
    grip = actual[:, 6]
    switches = int(np.count_nonzero(np.diff(np.sign(grip))))
    return dict(condition=condition, **case, historical=historical, actions=len(actual),
                switches=switches, zero_switches=switches == 0,
                all_open_commands=bool(np.all(grip < 0)), all_close_commands=bool(np.all(grip > 0)),
                zero_commands=int(np.count_nonzero(grip == 0)), min_command=float(grip.min()),
                max_command=float(grip.max()), telemetry_sha256=sha(telemetry_path),
                arrays_sha256=sha(Path(source)/'arrays.npz'), executed_matches_telemetry=True)


def finish():
    manifest = read(OUT/'manifest.json')
    assert digest({k:v for k,v in manifest.items() if k != 'manifest_hash'}) == manifest['manifest_hash']
    assert all(sha(ROOT/p) == expected for p, expected in manifest['source_files'].items()), 'Frozen source changed'
    assert all(sha(p) == expected for p, expected in manifest['reference_files'].items()), 'Historical artifact changed'
    cases = manifest['cases']
    assert len(cases) == len({key(case) for case in cases}) == 8
    assert set(c['task'] for c in cases) == {0, 2} and all(c['init'] == 19 for c in cases)
    assert all(sum(case['suite'] == suite for case in cases) == 2 for suite in SUITES)
    assert len(manifest['conditions']) == 2
    assert {cfg['start_time'] for cfg in manifest['conditions']} == {.05, .10}
    assert manifest['target_episodes'] == 16
    records, metadata, raw_first, flat, requests = {}, {}, {}, [], []
    gripper, prefix_checks, noises = [], [], {}
    checked = 0
    used_service_pids = set()
    for cfg in manifest['conditions']:
        name = cfg['id']
        assert cfg['D'] == 1 and cfg['initial'] == 'kalman' and cfg['noise_scale'] == 1.
        metadata[name] = dict(historical=False, kind='kalman_gripper_gaussian_d1', tau=cfg['start_time'])
        records[name] = {}
        for case in cases:
            source = directory(name, case)
            complete, result, identity = [read(source/file) for file in ('complete.json', 'result.json', 'identity.json')]
            assert complete['valid'] and result['valid']
            assert complete['manifest_hash'] == result['manifest_hash'] == identity['manifest_hash'] == manifest['manifest_hash']
            assert result['config'] == identity['config'] == cfg
            assert result['termination'] in ('success', 'timeout')
            assert (result['termination'] == 'success') == result['success']
            assert identity['env_seed'] == 7 and identity['settling'] == 10
            rows = [json.loads(line) for line in (source/'infer.jsonl').read_text().splitlines()]
            refresh_steps, steps = [], 0
            with np.load(source/'arrays.npz') as saved:
                for index, row in enumerate(rows):
                    epsilon = audit_request(row, index, steps, cfg, refresh_steps)
                    check = audit_initial_prefix(saved, index, cfg, epsilon)
                    if check is not None:
                        prefix_checks.append(dict(condition=name, case=case, **check))
                    steps += row['executed_length']
                    noise_key = (key(case), index)
                    noise = (row['seed'], row['model_noise_sha256'])
                    assert noises.setdefault(noise_key, noise) == noise, 'Paired base Gaussian changed'
                    requests.append(request_record(name, case, row, False))
                    checked += 1
                gripper.append(gripper_record(name, case, rows, saved, source, False))
                raw_first[name, key(case)] = dict(raw=saved['raw_0'].copy(), actions=saved['actions_0'].copy())
            assert steps == result['actions'] and result['counts'] == rows[-1]['counts']
            first_input = source/'input_0000.npz'
            assert sha(first_input) == rows[0]['input_file_sha256']
            with np.load(first_input) as saved_input:
                obs = {field:saved_input[field] for field in saved_input.files if field not in ('instruction', 'seed')}
                assert input_digest(obs, str(saved_input['instruction'])) == rows[0]['input_hash']
                assert int(saved_input['seed']) == rows[0]['seed']
            record = episode_record(name, case, result, rows, identity, source, False,
                                    metadata[name]['kind'], cfg['start_time'])
            records[name][key(case)] = record
            flat.append(record)
            used_service_pids.add(record['server_pid'])
    assert len(flat) == manifest['target_episodes']
    references = []
    for reference in manifest['references']:
        name, cfg = reference['id'], reference['config']
        assert name not in records
        tau = cfg.get('start_time') if cfg['initial'] == 'kalman' else None
        metadata[name] = dict(historical=True, kind=reference['kind'], tau=tau)
        records[name] = {}
        for case in cases:
            source = reference_directory(reference, case)
            complete, result, identity = [read(source/file) for file in ('complete.json', 'result.json', 'identity.json')]
            assert complete['valid'] and result['valid']
            assert complete['manifest_hash'] == result['manifest_hash'] == identity['manifest_hash'] == reference['manifest_hash']
            assert result['config'] == identity['config'] == cfg
            rows = [json.loads(line) for line in (source/'infer.jsonl').read_text().splitlines()]
            record = episode_record(name, case, result, rows, identity, source, True, reference['kind'], tau)
            records[name][key(case)] = record
            flat.append(record)
            requests.extend(request_record(name, case, row, True) for row in rows)
            with np.load(source/'arrays.npz') as saved:
                raw_first[name, key(case)] = dict(raw=saved['raw_0'].copy(), actions=saved['actions_0'].copy())
                gripper.append(gripper_record(name, case, rows, saved, source, True))
        references.append(dict(reference, n=len(cases), historical=True, timing_comparison=False,
                               success=sum(record['success'] for record in records[name].values())))
    assert sum(len(records[ref['id']]) for ref in manifest['references']) == 24
    first_checks, recoveries = [], []
    for case in cases:
        case_key = key(case)
        pids = sorted({records[cfg['id']][case_key]['server_pid'] for cfg in manifest['conditions']})
        done = read(OUT/'case_done'/f'{case_key}.json')
        assert done['passed'] and done['case'] == case
        assert done['atol'] == ATOL and done['rtol'] == RTOL and done['server_pids'] == pids
        assert {(check['condition'], check['array']) for check in done['checks']} == {
            (cfg['id'], field) for cfg in manifest['conditions'] for field in ('raw_0', 'actions_0')}
        assert all(check['passed'] for check in done['checks'])
        if len(pids) != 1:
            recovery = read(OUT/'case_recovery'/f'{case_key}.json')
            assert recovery == done['technical_recovery']
            assert recovery['reason'] == 'resource_oom' and recovery['attempt'] == 1
            assert recovery['case'] == case and recovery['valid_results_preserved']
            assert len(pids) == 2 and recovery['previous_server_pid'] in pids
            assert (OUT/recovery['archive']).is_dir()
            recoveries.append(recovery)
    anchor = manifest['conditions'][0]['id']
    for cfg in manifest['conditions'][1:]:
        for case in cases:
            check = first_check(anchor, cfg['id'], records[anchor], records[cfg['id']], raw_first, case, 'new_same_service')
            assert check['passed'], 'New bootstrap outputs mismatch'
            first_checks.append(check)
    gaussian = next(ref['id'] for ref in manifest['references'] if ref['config']['initial'] == 'gaussian')
    paired = []
    for cfg in manifest['conditions']:
        candidate = cfg['id']
        held = next(ref['id'] for ref in manifest['references']
                    if ref['config']['initial'] == 'kalman' and ref['config']['start_time'] == cfg['start_time'])
        for baseline in (held, gaussian):
            checks = [first_check(baseline, candidate, records[baseline], records[candidate], raw_first,
                                  case, 'historical_bootstrap') for case in cases]
            first_checks.extend(checks)
            eligible = [key(check['case']) for check in checks if check['passed']]
            for scope, keys in (('all_cases_descriptive', [key(case) for case in cases]),
                                ('initial_raw_matched', eligible)):
                paired.append(comparison(baseline, candidate, records[baseline], records[candidate], keys, scope))
    workers = sorted((OUT/'workers').glob('gpu*')) + sorted((OUT/'attempts').glob('resource_conflict_*/gpu*'))
    assert workers
    preparation, seen_service_pids = [], set()
    for worker in workers:
        exits = read(worker/'exits.json')
        assert exits['released'], f'Worker resource not released: {worker}'
        protocol = read(worker/'protocol.json') if (worker/'protocol.json').exists() else {}
        preflight = read(worker/'preflight.json') if (worker/'preflight.json').exists() else {}
        used = protocol.get('server_pid') in used_service_pids
        if used:
            assert preflight['passed'] and read(worker/'cross_service.json')['passed']
            seen_service_pids.add(protocol['server_pid'])
        preparation.append(dict(worker=str(worker.relative_to(OUT)), used_for_evaluation=used, released=True,
                                historical_gate_passed=preflight.get('historical_gate_passed'), **protocol))
    assert seen_service_pids == used_service_pids
    assert all(sha(ROOT/p) == expected for p, expected in manifest['source_files'].items())
    assert all(sha(p) == expected for p, expected in manifest['reference_files'].items())
    summary = summarize(records, metadata)
    gripper_summary = []
    for name in records:
        group = [record for record in gripper if record['condition'] == name]
        gripper_summary.append(dict(condition=name, n=len(group),
            switches=sum(record['switches'] for record in group),
            zero_switch_episodes=sum(record['zero_switches'] for record in group),
            all_open_episodes=sum(record['all_open_commands'] for record in group),
            all_close_episodes=sum(record['all_close_commands'] for record in group),
            zero_commands=sum(record['zero_commands'] for record in group),
            min_command=min(record['min_command'] for record in group),
            max_command=max(record['max_command'] for record in group)))
    for filename, values in [('episodes.csv', flat), ('summary.csv', summary), ('requests.csv', requests),
                             ('paired.csv', paired), ('gripper_episodes.csv', gripper),
                             ('gripper_summary.csv', gripper_summary),
                             ('initial_checks.csv', [dict(check, case=key(check['case'])) for check in first_checks])]:
        write_csv(OUT/filename, values)
    group = {(row['condition'], row['suite']):row for row in summary}
    descriptions = {gaussian:'高斯＋一步 AE（历史对照）'}
    for ref in manifest['references']:
        if ref['config']['initial'] == 'kalman':
            descriptions[ref['id']] = f"Kalman τ={ref['config']['start_time']:.2f}，保持夹爪（历史）"
    for cfg in manifest['conditions']:
        descriptions[cfg['id']] = f"Kalman τ={cfg['start_time']:.2f}，夹爪标准噪声（新测）"
    order = [gaussian]
    for cfg in manifest['conditions']:
        order.extend([next(ref['id'] for ref in manifest['references']
                           if ref['config']['initial'] == 'kalman' and ref['config']['start_time'] == cfg['start_time']), cfg['id']])
    lines = ['# 低噪声 Kalman 一步：夹爪独立标准噪声的快速闭环对照', '',
        '本轮只回答低τ成功率能否摆脱0：τ=0.05、0.10各新测8回合，共16回合。四个LIBERO套件各task0/2、官方init19、seed7、settling10；只读复用相同8个case的原保持夹爪两组及高斯D1一组，共24回合。任务来自此前已观察过的参数探索，不是独立确认集；没有扩到40例或按成败调参重跑。', '',
        '沿用固定V12跳层[2,3,5,14,16,17]、每三轮完整VLM刷新真残差及自身残差外推。Full只刷新残差，不调用Full AE、不执行Full动作；关闭验证、夹爪保护和回退。H50/R5、仿真20Hz，encoder/AE均无CUDA Graph，每次实际动作请求只调用一次AE。首轮没有历史，两新组均保持原高斯D1启动。', '',
        '模型归一化动作空间中，六维运动0:6仍为(1−τ)×KF+τ×ε，填充7:仍为τ×ε；只把夹爪维6替换为同seed的独立标准高斯ε6，标准差1，旧夹爪权重0。每个噪声元素独立采样，同case及重规划序号使用相同基础噪声。AE全局时间、一步积分区间仍为τ→0；混合维度噪声与全局时间的训练分布匹配未经保证。本实验检验该干预是否有效，不预设夹爪保持是全部失败原因。', '',
        '|条件|Spatial|Object|Goal|Long|总成功率|', '|---|---|---|---|---|---|']
    for name in order:
        total = group[name, 'all']
        values = [descriptions[name], *[f"{group[name,suite]['success']}/2" for suite in SUITES],
                  f"{total['success']}/8（{100*total['success_rate']:.1f}%）"]
        lines.append('|'+'|'.join(values)+'|')
    lines += ['', '是否摆脱0仅按上表各8例的任务成功数判断：', '']
    for cfg in manifest['conditions']:
        successes = group[cfg['id'], 'all']['success']
        outcome = '本样本中已出现任务成功，摆脱0' if successes else '本样本仍为0，未观察到摆脱0'
        lines.append(f"- τ={cfg['start_time']:.2f}：{successes}/8，{outcome}。")
    lines += ['', '|候选|参照|参照成功|候选成功|恢复|新增失败|严格首轮匹配n|', '|---|---|---|---|---|---|---|']
    for cfg in manifest['conditions']:
        for full in [row for row in paired if row['candidate'] == cfg['id'] and row['scope'] == 'all_cases_descriptive']:
            strict = next(row for row in paired if row['candidate'] == cfg['id'] and row['baseline'] == full['baseline']
                          and row['scope'] == 'initial_raw_matched')
            lines.append(f"|τ={cfg['start_time']:.2f}|{descriptions[full['baseline']]}|{full['baseline_success']}/8|{full['candidate_success']}/8|{full['recovered']}|{full['lost']}|{strict['n']}|")
    lines += ['', '恢复/新增失败按同任务、同官方初态计算。全部8例均保留有效成功结果；严格配对另要求首轮共同高斯启动的输入、初态、噪声及raw/解码动作通过原atol=rtol=1e-5。跨服务超容差只公开剔除和实际n，不改变容差或依据任务成败重跑。每组仅8例，不能据此推断40例准确率或稳定泛化；夹爪切换增加也不等于任务成功。', '',
        '|条件|夹爪符号切换总次数|零切换回合|全程开爪命令回合|命令最小/最大|', '|---|---|---|---|---|']
    for name in order:
        row = next(value for value in gripper_summary if value['condition'] == name)
        lines.append(f"|{descriptions[name]}|{row['switches']}|{row['zero_switch_episodes']}/8|{row['all_open_episodes']}/8|{row['min_command']:.4f} / {row['max_command']:.4f}|")
    lines += ['', '夹爪统计只取真正传入env.step的执行前缀，包含首轮共同高斯启动；每条与arrays.npz精确交叉核对。负命令为开爪、正命令为闭爪，相邻已执行命令的np.sign不同计一次切换。这描述命令，不是独立测量手指实际位置。历史telemetry未列入原冻结文件，已用冻结执行数组精确交叉核对，并在gripper_episodes.csv保存其本次SHA-256。', '',
        '|条件|热态策略ms/动作|热态AE ms/请求|平均执行动作数|闭环墙钟s|', '|---|---|---|---|---|']
    for name in order:
        row = group[name, 'all']
        lines.append(f"|{descriptions[name]}|{row['hot_ms_per_action']:.3f}|{row['mean_ae_ms']:.3f}|{row['mean_actions']:.1f}|{row['wall_s']:.2f}|")
    historical_checks = [check for check in first_checks if check['kind'] == 'historical_bootstrap']
    lines += ['', '热态从每回合第4次请求开始，策略边界包含视觉/VLM、周期完整VLM刷新、残差、AE、解码及ack滤波更新。CPU预测和坐标编码在当前实现串行计时，可以提前或与视觉计算重叠，不是不可消除的GPU算法成本。仿真20Hz、不以墙钟sleep；轨迹长度和历史服务不同，计时仅描述，不称为受控部署加速比。', '',
        f"审查16条新闭环及{checked}次新请求，实际NFE均为1，Full AE/Full动作/验证均为0；{len(prefix_checks)}次有Kalman历史的保存起点前缀逐元素符合干预，六维运动及填充未变、夹爪为标准噪声且旧值权重0。观察计数等于ack的真实执行步。新组首轮数值全部通过原容差，历史首轮比较通过{sum(check['passed'] for check in historical_checks)}/{len(historical_checks)}项；逐项误差见initial_checks.csv。", '',
        f"冻结源文件与历史文件哈希未变，{len(recoveries)}个case有已归档的资源OOM续跑；所有自有服务及GPU锁已释放。", '',
        '[逐回合](../../../results/idea1_v4_6_gripper_noise/episodes.csv) · [汇总](../../../results/idea1_v4_6_gripper_noise/summary.csv) · [配对](../../../results/idea1_v4_6_gripper_noise/paired.csv) · [夹爪](../../../results/idea1_v4_6_gripper_noise/gripper_summary.csv) · [初态数值核验](../../../results/idea1_v4_6_gripper_noise/initial_checks.csv)']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'references.json', references)
    atomic(OUT/'preparation.json', preparation)
    atomic(OUT/'initial_checks.json', first_checks)
    atomic(OUT/'initial_prefix_checks.json', prefix_checks)
    atomic(OUT/'gripper_diagnostics.json', dict(episodes=gripper, summary=gripper_summary,
        intervention='raw normalized gripper dimension 6 = iid standard epsilon; old gripper coefficient zero',
        telemetry_crosschecked_with_frozen_arrays=True))
    atomic(OUT/'resource_recoveries.json', recoveries)
    atomic(OUT/'audit.json', dict(passed=True, new_episodes=16, reused_episodes=24, route_checks=checked,
        kalman_initial_prefix_checks=len(prefix_checks), gripper_noise_std=1., gripper_prior_weight=0.,
        first_checks=len(first_checks), initial_new_passed=True, resource_recovery_cases=len(recoveries),
        historical_raw_matched=sum(check['passed'] for check in historical_checks),
        historical_raw_checks=len(historical_checks), atol=ATOL, rtol=RTOL,
        full_actions=0, full_ae_calls=0, verifier_calls=0,
        sources_match=True, references_unchanged=True, all_workers_released=True, time=time.time()))
    atomic(OUT/'complete.json', dict(passed=True, new_episodes=16, report=str(REPORT), time=time.time()))
    atomic(OUT/'status.json', dict(status='complete', completed=16, target=16,
                                  summary=summary, paired=paired, time=time.time()))


if __name__ == '__main__':
    finish()
