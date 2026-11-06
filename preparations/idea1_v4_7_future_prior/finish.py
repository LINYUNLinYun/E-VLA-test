"""Audit the fixed 8-case comparison of future-gripper and shifted-chunk priors."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from preparations.idea1_v2_1.runtime_client import input_digest
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_4.analyze import write_csv
from preparations.idea1_v4_5_noise_sweep.finish import (
    ATOL, RTOL, comparison, episode_record, first_check, read,
    reference_directory, request_record, summarize,
)
from preparations.idea1_v4_6_gripper_noise.finish import gripper_record
from .common import ROOT, OUT, REPORT, SUITES, MASK, directory, key, sha


def audit_request(row, index, step, cfg, refresh_steps, previous):
    route, timing = row['route'], row['timing']
    assert row['round'] == row['replan'] == index and row['step'] == step
    assert 0 < row['executed_length'] <= row['execute_length'] == 5
    refresh = index % cfg['refresh_interval'] == 0
    assert route['residual_refresh'] == refresh
    if refresh:
        refresh_steps.append(step)
    assert route['history_steps'] == refresh_steps[-2:]
    assert route['has_second_history'] == (len(refresh_steps) >= 2)
    ratio = 0. if len(refresh_steps) < 2 else (step-refresh_steps[-1])/(refresh_steps[-1]-refresh_steps[-2])
    assert np.isclose(route['extrapolation_ratio'], ratio)
    assert route['D'] == 1 and route['skip'] == list(MASK)
    assert route['path'] == route['action_source'] == 'v12_d1'
    assert route['compensation'] == 'extrapolate'
    assert not route['full_ae_executed'] and not route['full_action_executed'] and not route['verifier']
    assert not route['native_encoder_graph'] and not route['native_ae_graph'] and not route['verify_graph']
    assert not timing['native_ae_graph']
    counts = row['counts']
    assert counts['full'] == counts['verify'] == 0
    assert counts['draft'] == index+1 and counts['residual_refresh'] == len(refresh_steps)
    use_prior = index > 0
    has_kf = cfg['initial'] == 'kalman'
    tau = cfg['start_time'] if use_prior else 1.
    assert cfg['start_time'] == .1 and cfg['noise_scale'] == 1.
    assert route['initial_state'] == ('gaussian' if not use_prior else 'kalman6' if has_kf else 'shifted_chunk')
    assert route['prior_source'] == (cfg['prior_source'] if use_prior else 'gaussian')
    assert route['prior_source_round'] == (index-1 if use_prior else None)
    assert route['shift_executed_length'] == (previous['executed_length'] if use_prior else 0)
    assert route['shift_source_step'] == (previous['step'] if use_prior else None)
    assert route['future_chunk_available'] == use_prior
    assert route['gripper_prior_source'] == ('previous_prediction_shifted' if use_prior else 'gaussian')
    assert route['prior_weight'] == (1-tau if use_prior else 0.)
    assert route['kalman_bootstrap'] == (has_kf and not use_prior)
    assert route['kalman_dimensions'] == (6 if has_kf else 0)
    observations = step if has_kf else 0
    assert route['kalman_observations'] == route['kalman_rotation_observations'] == observations
    assert route['kalman_source_last_step'] == (step-1 if use_prior and has_kf else None)
    assert route['kalman_prior_weight'] == (1-tau if use_prior and has_kf else 0.)
    assert counts['kalman'] == (index if has_kf else 0)
    assert counts['shifted_prior'] == index
    assert route['actual_start_time'] == route['warmstart_time'] == tau
    assert route['warmstart_noise_scale'] == 1. and route['actual_noise_std'] == tau
    assert route['fast_time_embedding'] == timing['time_embedding'] == 'openpi_linspace'
    nodes = timing['nodes']
    assert nodes['actual_NFE'] == nodes['NFE'] == 1
    assert nodes['nodes'] == [tau, 0.] and nodes['timesteps'] == [tau] and nodes['dt'] == [-tau]
    epsilon = np.random.default_rng(row['seed']).standard_normal((50,32), dtype=np.float32)
    assert hashlib.sha256(epsilon.tobytes()).hexdigest() == row['model_noise_sha256']
    return epsilon


def audit_prior(saved, index, cfg, previous, epsilon):
    """Check actual-execution shift, last-value tail, current encoding, and common noise."""
    fields = ('prior_commands', 'prior_command_prefix', 'prior_encoded_prefix',
              'prior_initial_prefix', 'decoder_offset', 'decoder_inverse')
    if index == 0:
        assert not any(f'{field}_{index}' in saved for field in fields)
        return None
    old = saved[f'actions_{index-1}']
    shift = previous['executed_length']
    assert old.shape == (50,7) and 0 < shift <= 5
    shifted = np.concatenate((old[shift:], np.repeat(old[-1:], shift, axis=0)))
    commands = saved[f'prior_commands_{index}']
    prefix = saved[f'prior_command_prefix_{index}']
    encoded = saved[f'prior_encoded_prefix_{index}']
    initial = saved[f'prior_initial_prefix_{index}']
    offset, inverse = saved[f'decoder_offset_{index}'], saved[f'decoder_inverse_{index}']
    assert commands.shape == (50,7) and prefix.shape == (5,7)
    assert encoded.shape == initial.shape == (5,32)
    assert offset.shape == (7,) and inverse.shape == (7,7)
    assert all(np.isfinite(value).all() for value in (commands, prefix, encoded, initial, offset, inverse))
    assert np.array_equal(prefix, commands[:5])
    assert np.array_equal(commands[:,6], shifted[:,6]), 'Future gripper must come from shifted previous prediction'
    if cfg['prior_source'] == 'shifted_chunk':
        assert np.array_equal(commands, shifted), 'Whole-chunk prior must preserve all seven predicted commands'
    else:
        assert cfg['prior_source'] == 'kalman6_future_grip'
        assert np.max(np.abs(commands[:,:6])) <= 1., 'Existing KF command clipping changed'
    encoded_reference = (prefix-offset)@inverse
    assert np.allclose(encoded[:,:7], encoded_reference, atol=ATOL, rtol=RTOL), 'Current decoder re-encoding mismatch'
    assert np.array_equal(encoded[:,7:], np.zeros((5,25), np.float32))
    tau, rho = cfg['start_time'], cfg['noise_scale']
    expected = ((1-tau)*encoded + rho*tau*epsilon[:5]).astype(np.float32)
    assert np.array_equal(initial, expected), 'All dimensions must use the same low-noise mixture'
    return dict(request=index, shift_executed_length=shift, source_round=index-1,
                full_prediction_shape=[50,7], shifted_future_gripper_exact=True,
                whole_chunk_exact=cfg['prior_source'] == 'shifted_chunk',
                tail_rule='repeat last prediction', reencoded_with_current_decoder=True,
                encoded_max_abs=float(np.max(np.abs(encoded[:,:7]-encoded_reference))),
                uniform_low_noise_exact=True)


def gripper_persistence(condition, case, rows, saved, source, historical):
    record = gripper_record(condition, case, rows, saved, source, historical)
    commands = np.concatenate([saved[f'actions_{i}'][:row['executed_length'],6] for i,row in enumerate(rows)])
    signs = np.sign(commands)
    starts = np.r_[0, np.flatnonzero(signs[1:] != signs[:-1])+1]
    lengths = np.diff(np.r_[starts, len(signs)])
    positive = lengths[signs[starts] > 0]
    record.update(closed_action_count=int(np.sum(commands > 0)),
        positive_run_count=len(positive), positive_run_steps_sum=int(positive.sum()),
        longest_positive_run=int(positive.max()) if len(positive) else 0,
        sustained_close_5=bool(len(positive) and positive.max() >= 5),
        sustained_close_10=bool(len(positive) and positive.max() >= 10),
        mean_sign_run_steps=float(lengths.mean()), max_sign_run_steps=int(lengths.max()),
        switches_per_100_actions=100*record['switches']/len(commands))
    return record


def worker_records(used_service_pids):
    workers = sorted((OUT/'workers').glob('gpu*')) + sorted((OUT/'attempts').glob('resource_conflict_*/gpu*'))
    assert workers
    preparation, seen = [], set()
    for worker in workers:
        exits = read(worker/'exits.json')
        assert exits['released'], f'Worker not released: {worker}'
        protocol = read(worker/'protocol.json') if (worker/'protocol.json').exists() else {}
        preflight = read(worker/'preflight.json') if (worker/'preflight.json').exists() else {}
        used = protocol.get('server_pid') in used_service_pids
        if used:
            assert preflight['passed'] and read(worker/'cross_service.json')['passed']
            seen.add(protocol['server_pid'])
        preparation.append(dict(worker=str(worker.relative_to(OUT)), used_for_evaluation=used, released=True,
                                historical_gate_passed=preflight.get('historical_gate_passed'), **protocol))
    assert seen == used_service_pids
    return preparation


def finish():
    manifest = read(OUT/'manifest.json')
    assert digest({name:value for name,value in manifest.items() if name != 'manifest_hash'}) == manifest['manifest_hash']
    for files in (manifest['source_files'], manifest['reference_files']):
        assert all(sha(ROOT/path if files is manifest['source_files'] else path) == expected for path,expected in files.items())
    cases = manifest['cases']
    assert len(cases) == len({key(case) for case in cases}) == 8
    assert all(case['task'] in (0,2) and case['init'] == 19 for case in cases)
    assert all(sum(case['suite'] == suite for case in cases) == 2 for suite in SUITES)
    assert len(manifest['conditions']) == 2 and manifest['target_episodes'] == 16
    assert {cfg['prior_source'] for cfg in manifest['conditions']} == {'kalman6_future_grip','shifted_chunk'}
    records, metadata, raw_first, flat, requests, gripper = {}, {}, {}, [], [], []
    prior_checks, noises, used_service_pids = [], {}, set()
    checked = 0
    for cfg in manifest['conditions']:
        name = cfg['id']
        assert cfg['D'] == 1 and cfg['start_time'] == .1 and cfg['noise_scale'] == 1.
        metadata[name] = dict(historical=False, kind=cfg['prior_source'], tau=.1)
        records[name] = {}
        for case in cases:
            source = directory(name,case)
            complete,result,identity = [read(source/file) for file in ('complete.json','result.json','identity.json')]
            assert complete['valid'] and result['valid']
            assert complete['manifest_hash'] == result['manifest_hash'] == identity['manifest_hash'] == manifest['manifest_hash']
            assert result['config'] == identity['config'] == cfg
            assert result['termination'] in ('success','timeout')
            assert (result['termination'] == 'success') == result['success']
            assert identity['env_seed'] == 7 and identity['settling'] == 10
            rows = [json.loads(line) for line in (source/'infer.jsonl').read_text().splitlines()]
            steps,refresh_steps = 0,[]
            with np.load(source/'arrays.npz') as saved:
                for i,row in enumerate(rows):
                    previous = rows[i-1] if i else None
                    epsilon = audit_request(row,i,steps,cfg,refresh_steps,previous)
                    check = audit_prior(saved,i,cfg,previous,epsilon)
                    if check is not None:
                        prior_checks.append(dict(condition=name,case=case,**check))
                    steps += row['executed_length']
                    noise = (row['seed'],row['model_noise_sha256'])
                    assert noises.setdefault((key(case),i),noise) == noise
                    requests.append(request_record(name,case,row,False))
                    checked += 1
                gripper.append(gripper_persistence(name,case,rows,saved,source,False))
                raw_first[name,key(case)] = dict(raw=saved['raw_0'].copy(),actions=saved['actions_0'].copy())
            assert steps == result['actions'] and result['counts'] == rows[-1]['counts']
            first_input = source/'input_0000.npz'
            assert sha(first_input) == rows[0]['input_file_sha256']
            with np.load(first_input) as saved_input:
                obs = {field:saved_input[field] for field in saved_input.files if field not in ('instruction','seed')}
                assert input_digest(obs,str(saved_input['instruction'])) == rows[0]['input_hash']
                assert int(saved_input['seed']) == rows[0]['seed']
            record = episode_record(name,case,result,rows,identity,source,False,cfg['prior_source'],.1)
            hot = rows[3:]
            record['hot_gpu_component_ms'] = sum(sum(row['timing'].get(field,0.) for field in
                ('vision_ms','vlm_ms','ae_ms','refresh_vision_ms','refresh_vlm_ms')) for row in hot)
            records[name][key(case)] = record
            flat.append(record)
            used_service_pids.add(record['server_pid'])
    assert len(flat) == 16
    references = []
    for reference in manifest['references']:
        name,cfg = reference['id'],reference['config']
        assert name not in records
        tau = cfg.get('start_time') if cfg['initial'] != 'gaussian' else None
        metadata[name] = dict(historical=True,kind=reference['kind'],tau=tau)
        records[name] = {}
        for case in cases:
            source = reference_directory(reference,case)
            complete,result,identity = [read(source/file) for file in ('complete.json','result.json','identity.json')]
            assert complete['valid'] and result['valid']
            assert complete['manifest_hash'] == result['manifest_hash'] == identity['manifest_hash'] == reference['manifest_hash']
            assert result['config'] == identity['config'] == cfg
            rows = [json.loads(line) for line in (source/'infer.jsonl').read_text().splitlines()]
            record = episode_record(name,case,result,rows,identity,source,True,reference['kind'],tau)
            record['hot_gpu_component_ms'] = sum(sum(row['timing'].get(field,0.) for field in
                ('vision_ms','vlm_ms','ae_ms','refresh_vision_ms','refresh_vlm_ms')) for row in rows[3:])
            records[name][key(case)] = record
            flat.append(record)
            requests.extend(request_record(name,case,row,True) for row in rows)
            with np.load(source/'arrays.npz') as saved:
                raw_first[name,key(case)] = dict(raw=saved['raw_0'].copy(),actions=saved['actions_0'].copy())
                gripper.append(gripper_persistence(name,case,rows,saved,source,True))
        references.append(dict(reference,n=8,historical=True,timing_comparison=False,
                               success=sum(row['success'] for row in records[name].values())))
    assert sum(len(records[reference['id']]) for reference in manifest['references']) == 24
    first_checks,recoveries = [],[]
    anchor = manifest['conditions'][0]['id']
    for case in cases:
        k = key(case)
        pids = sorted({records[cfg['id']][k]['server_pid'] for cfg in manifest['conditions']})
        done = read(OUT/'case_done'/f'{k}.json')
        assert done['passed'] and done['case'] == case and done['server_pids'] == pids
        assert done['atol'] == done['rtol'] == ATOL
        assert {(check['condition'],check['array']) for check in done['checks']} == {
            (cfg['id'],field) for cfg in manifest['conditions'] for field in ('raw_0','actions_0')}
        assert all(check['passed'] for check in done['checks'])
        if len(pids) != 1:
            recovery = read(OUT/'case_recovery'/f'{k}.json')
            assert recovery == done['technical_recovery'] and recovery['reason'] == 'resource_oom'
            assert recovery['attempt'] == 1 and recovery['valid_results_preserved'] and recovery['case'] == case
            assert len(pids) == 2 and recovery['previous_server_pid'] in pids
            assert (OUT/recovery['archive']).is_dir()
            recoveries.append(recovery)
        for cfg in manifest['conditions'][1:]:
            check = first_check(anchor,cfg['id'],records[anchor],records[cfg['id']],raw_first,case,'new_same_service')
            assert check['passed'], 'New bootstrap numerical mismatch'
            first_checks.append(check)
    paired = []
    candidates = [cfg['id'] for cfg in manifest['conditions']]
    comparisons = [(reference['id'],candidate) for reference in manifest['references'] for candidate in candidates]
    comparisons.append((candidates[0],candidates[1]))
    for baseline,candidate in comparisons:
        historical = metadata[baseline]['historical']
        checks = [first_check(baseline,candidate,records[baseline],records[candidate],raw_first,case,
                              'historical_bootstrap' if historical else 'new_pair') for case in cases]
        first_checks.extend(checks)
        eligible = [key(check['case']) for check in checks if check['passed']]
        for scope,keys in (('all_cases_descriptive',[key(case) for case in cases]),('initial_raw_matched',eligible)):
            record = comparison(baseline,candidate,records[baseline],records[candidate],keys,scope)
            record['historical_comparison'] = historical
            paired.append(record)
    preparation = worker_records(used_service_pids)
    assert all(sha(ROOT/path) == expected for path,expected in manifest['source_files'].items())
    assert all(sha(path) == expected for path,expected in manifest['reference_files'].items())
    summary = summarize(records,metadata)
    for row in summary:
        subset = [record for record in records[row['condition']].values() if row['suite'] == 'all' or record['suite'] == row['suite']]
        count = sum(record['hot_requests'] for record in subset)
        row['mean_hot_gpu_component_ms'] = sum(record['hot_gpu_component_ms'] for record in subset)/count if count else None
    gripper_summary = []
    for name in records:
        subset = [record for record in gripper if record['condition'] == name]
        actions = sum(record['actions'] for record in subset)
        switches = sum(record['switches'] for record in subset)
        positive_runs = sum(record['positive_run_count'] for record in subset)
        gripper_summary.append(dict(condition=name,n=8,actions=actions,switches=switches,
            switches_per_100_actions=100*switches/actions,
            zero_switch_episodes=sum(record['zero_switches'] for record in subset),
            all_open_episodes=sum(record['all_open_commands'] for record in subset),
            sustained_close_5_episodes=sum(record['sustained_close_5'] for record in subset),
            sustained_close_10_episodes=sum(record['sustained_close_10'] for record in subset),
            longest_positive_run=max(record['longest_positive_run'] for record in subset),
            mean_positive_run_steps=sum(record['positive_run_steps_sum'] for record in subset)/positive_runs if positive_runs else 0.,
            min_command=min(record['min_command'] for record in subset),
            max_command=max(record['max_command'] for record in subset)))
    for filename,values in [('episodes.csv',flat),('summary.csv',summary),('requests.csv',requests),('paired.csv',paired),
                            ('gripper_episodes.csv',gripper),('gripper_summary.csv',gripper_summary),
                            ('initial_checks.csv',[dict(check,case=key(check['case'])) for check in first_checks])]:
        write_csv(OUT/filename,values)
    group = {(row['condition'],row['suite']):row for row in summary}
    labels = {}
    for cfg in manifest['conditions']:
        labels[cfg['id']] = 'A：六维KF＋上轮未来夹爪' if cfg['prior_source'] == 'kalman6_future_grip' else 'B：上轮整块动作平移'
    for ref in manifest['references']:
        if ref['config']['initial'] == 'gaussian':
            labels[ref['id']] = '高斯D1（历史）'
        elif ref['config'].get('gripper_initial') == 'standard_gaussian':
            labels[ref['id']] = '六维KF＋夹爪标准噪声 τ=.10（历史）'
        else:
            labels[ref['id']] = '六维KF＋保持夹爪 τ=.10（历史）'
    order = [reference['id'] for reference in manifest['references']]+candidates
    lines = ['# 一步AE：六维Kalman＋未来夹爪与整块预测平移的最小闭环对照','',
        '固定τ=0.10、ρ=1，两条件各8例，共16条新闭环。四个LIBERO套件各task0/2、官方init19、seed7、settling10。只读复用相同8个case的保持夹爪、夹爪标准噪声及高斯D1三组，共24条历史结果；本轮不扩到40例或按成败调参。','',
        'A使用既定六维Kalman预测运动，并接上上一轮完整预测中尚未执行的未来夹爪。B使用上一轮完整七维动作预测。两者都按上一轮真实ack执行数量平移，尾部重复上一轮最后预测；动作先验经当前DecoderCoordinates重新编码到当前归一化坐标，填充25维置零，再统一混合xτ=0.9×encoded_prior+0.1×ε。所有维度共用一次AE、全局时间0.1→0；夹爪不再单独使用标准差1的噪声。B没有额外裁剪，A保留既定Kalman六维裁剪。','',
        '上一轮未执行预测仅用作先验，只有真正执行并ack的动作更新Kalman；reset清空跨回合历史。首轮没有历史，两组均用相同高斯D1启动。固定V12跳层[2,3,5,14,16,17]、残差外推及每三轮完整VLM刷新不变，Full只刷新残差，不调用Full AE或执行Full动作；无验证、保护或回退，H50/R5、仿真20Hz，encoder/AE均不使用CUDA Graph。','',
        '|条件|Spatial|Object|Goal|Long|总成功率|','|---|---|---|---|---|---|']
    for name in order:
        row = group[name,'all']
        lines.append('|'+'|'.join([labels[name],*[f"{group[name,suite]['success']}/2" for suite in SUITES],
            f"{row['success']}/8（{100*row['success_rate']:.1f}%）"] )+'|')
    lines += ['','|候选|参照|参照成功|候选成功|恢复|新增失败|严格匹配n|','|---|---|---|---|---|---|---|']
    for row in [record for record in paired if record['scope'] == 'all_cases_descriptive']:
        strict = next(record for record in paired if record['scope'] == 'initial_raw_matched' and
                      record['baseline'] == row['baseline'] and record['candidate'] == row['candidate'])
        lines.append(f"|{labels[row['candidate']]}|{labels[row['baseline']]}|{row['baseline_success']}/8|{row['candidate_success']}/8|{row['recovered']}|{row['lost']}|{strict['n']}|")
    lines += ['','全部有效回合均保留。严格配对要求首轮共同高斯启动的输入、初态、噪声及raw/解码动作通过原atol=rtol=1e-5；跨服务超容差只报告实际n，不改容差或依据任务成败重跑。本轮每组只有8例，任务此前已用于探索，不能称为独立泛化确认，也不能推断40例成功率。','',
        '|条件|夹爪切换数|每100动作切换|全程开爪回合|持续闭爪≥5步回合|持续闭爪≥10步回合|最长连续闭爪步数|平均连续闭爪步数|','|---|---|---|---|---|---|---|---|']
    for name in order:
        row = next(record for record in gripper_summary if record['condition'] == name)
        lines.append(f"|{labels[name]}|{row['switches']}|{row['switches_per_100_actions']:.2f}|{row['all_open_episodes']}/8|{row['sustained_close_5_episodes']}/8|{row['sustained_close_10_episodes']}/8|{row['longest_positive_run']}|{row['mean_positive_run_steps']:.2f}|")
    lines += ['','持续性统计来自真正执行前缀，包含共同高斯启动，并与telemetry逐条精确核对。负命令为开爪、正命令为闭爪，相邻np.sign变化计一次切换；连续5/10步对应仿真0.25/0.50秒。该表描述执行命令，不是手指实际位置；更多切换或更长闭爪都不能单独证明任务成功。历史telemetry未列入原冻结文件，已用冻结动作数组交叉核验，SHA-256保存在gripper_episodes.csv。','',
        '|条件|热态策略ms/动作|热态AE ms/请求|热态GPU组件记录总和ms/请求|平均执行动作数|闭环墙钟s|','|---|---|---|---|---|---|']
    for name in order:
        row = group[name,'all']
        lines.append(f"|{labels[name]}|{row['hot_ms_per_action']:.3f}|{row['mean_ae_ms']:.3f}|{row['mean_hot_gpu_component_ms']:.3f}|{row['mean_actions']:.1f}|{row['wall_s']:.2f}|")
    historical_checks = [check for check in first_checks if check['kind'] == 'historical_bootstrap']
    lines += ['','热态从每回合第4次请求开始。策略边界包含视觉/VLM、周期完整VLM刷新、残差、先验编码、AE、解码与ack更新；GPU组件记录总和为vision_ms＋vlm_ms＋ae_ms及当次refresh_vision_ms＋refresh_vlm_ms，不等于独立测得的端到端部署延迟。CPU部分可提前或重叠，不作为不可消除GPU成本。仿真20Hz、不以墙钟sleep；不同轨迹与历史服务的时间仅描述，不称为受控加速比。','',
        f"审查16条新闭环及{checked}次新请求，实际NFE均为1、Full AE/Full动作/验证均为0；{len(prior_checks)}次有历史的起点核对真实执行位移、尾部补齐、未来夹爪/整块预测、当前坐标重编码及统一噪声。A的Kalman观察数等于真实ack步数，B没有Kalman观察或更新。新组首轮通过原数值容差；历史首轮比较通过{sum(check['passed'] for check in historical_checks)}/{len(historical_checks)}项，逐项误差见initial_checks.csv。",'',
        f"冻结源文件及历史文件哈希未变，{len(recoveries)}个case有已归档资源OOM续跑，所有自有服务与GPU锁已释放。",'',
        '[逐回合](../../../results/idea1_v4_7_future_prior/episodes.csv) · [汇总](../../../results/idea1_v4_7_future_prior/summary.csv) · [配对](../../../results/idea1_v4_7_future_prior/paired.csv) · [夹爪持续性](../../../results/idea1_v4_7_future_prior/gripper_summary.csv) · [首轮核验](../../../results/idea1_v4_7_future_prior/initial_checks.csv)']
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    atomic(OUT/'references.json',references)
    atomic(OUT/'preparation.json',preparation)
    atomic(OUT/'initial_checks.json',first_checks)
    atomic(OUT/'prior_checks.json',prior_checks)
    atomic(OUT/'gripper_diagnostics.json',dict(episodes=gripper,summary=gripper_summary,
        semantics='Negative=open and positive=close commands; persistence counts actually executed contiguous positive-sign steps',
        telemetry_crosschecked_with_frozen_arrays=True))
    atomic(OUT/'resource_recoveries.json',recoveries)
    atomic(OUT/'audit.json',dict(passed=True,new_episodes=16,reused_episodes=24,route_checks=checked,
        prior_checks=len(prior_checks),first_checks=len(first_checks),initial_new_passed=True,
        historical_raw_matched=sum(check['passed'] for check in historical_checks),historical_raw_checks=len(historical_checks),
        atol=ATOL,rtol=RTOL,resource_recovery_cases=len(recoveries),full_actions=0,full_ae_calls=0,verifier_calls=0,
        sources_match=True,references_unchanged=True,all_workers_released=True,time=time.time()))
    atomic(OUT/'complete.json',dict(passed=True,new_episodes=16,report=str(REPORT),time=time.time()))
    atomic(OUT/'status.json',dict(status='complete',completed=16,target=16,summary=summary,paired=paired,time=time.time()))


if __name__ == '__main__':
    finish()
