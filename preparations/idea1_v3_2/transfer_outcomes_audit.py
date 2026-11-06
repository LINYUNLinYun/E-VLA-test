"""Read-only postprocessing of E4 outcomes; never modifies frozen run sources.

This separate supplement adds case-level native terminations and measured end
states. It performs no model or environment calls and has its own source hash.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from .transfer_protocol import ROOT, MODES, ENVIRONMENTS, load_manifest, complete_result, sha, write_json


def lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def csv_save(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def audit(output):
    output = Path(output).resolve()
    manifest = load_manifest(output/'manifest.json', verify_sources=True)
    destination = output/('outcomes_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    destination.mkdir(exist_ok=False)
    rows, receipts, violations, seed_groups, initial_groups = [], {}, [], defaultdict(set), defaultdict(set)
    case6_trajectories = []
    attempt_history, attempt_hashes, quality_by_logs = [], {}, Counter()
    complete_count, route_requests = 0, 0
    for c in manifest['cases']:
        for mode in MODES:
            candidates = []
            for p in (output/'episodes'/c['environment']/f'case{c["case"]:02d}'/mode).glob('attempt*/result.json'):
                r = json.loads(p.read_text())
                if r.get('manifest_hash') == manifest['manifest_hash']:
                    candidates.append((p, r))
                    attempt_hashes[str(p)] = sha(p)
                    attempt_history.append(dict(environment=c['environment'], case=c['case'], mode=mode,
                                                attempt=p.parent.name, status=r['status'], success=r.get('success'),
                                                steps=r.get('steps'), native_termination=r.get('termination'),
                                                error_last_line=r.get('error', '').strip().splitlines()[-1] if r.get('error') else None,
                                                result_path=str(p.relative_to(output)), result_sha256=sha(p)))
            complete = [(p, r) for p, r in candidates if complete_result(r, manifest, c, mode)]
            if len(complete) > 1:
                raise ValueError('Duplicate complete episode key')
            chosen = complete or sorted(candidates)
            if not chosen:
                rows.append(dict(environment=c['environment'], case=c['case'], mode=mode,
                                 status='unsupported' if c['support_status'] == 'unsupported' else 'pending', success=None))
                continue
            path, r = chosen[-1]
            receipts[str(path)] = sha(path)
            row = dict(environment=c['environment'], case=c['case'], mode=mode, status=r['status'],
                       success=r.get('success'), native_termination=r.get('termination'), steps=r.get('steps'),
                       cap=c['max_steps'], native_hz=c['native_hz'], task=c['task'], instruction=c['instruction'],
                       wall_s=r.get('wall_s'), environment_setup_s=r.get('environment_setup_s'),
                       gpu_uuid=r.get('gpu_uuid'), result_path=str(path.relative_to(output)),
                       result_sha256=sha(path), initial_raw_hash=r.get('initial_raw_hash'))
            if r['status'] != 'complete':
                row['technical_error'] = r.get('error', r.get('failure_code')); rows.append(row); continue
            complete_count += 1
            initial_groups[(c['environment'], c['case'])].add(r.get('initial_raw_hash'))
            trace_path = Path(r['telemetry_path'])
            trace = lines(trace_path); receipts[str(trace_path)] = sha(trace_path)
            if len(trace) != r['steps']+1 or [t['step'] for t in trace] != list(range(r['steps']+1)):
                violations.append(dict(kind='telemetry_coverage', environment=c['environment'], case=c['case'], mode=mode))
            position_key = 'agent_xy_pixels' if c['environment'] == 'pusht' else 'eef_xyz_world'
            positions = np.asarray([t['actual'][position_key] for t in trace])
            if not np.isfinite(positions).all() or positions.shape[1] != (2 if c['environment'] == 'pusht' else 3):
                raise ValueError('Actual telemetry has wrong dimension or nonfinite values')
            row['actual_quantity'] = 'measured XY pixels' if c['environment'] == 'pusht' else 'measured EEF XYZ metres'
            if c['environment'] == 'calvin' and c['case'] == 6:
                case6_trajectories.append(dict(mode=mode, success=r['success'], steps=r['steps'], positions=positions))
            for index, coordinate in enumerate('xy' if c['environment'] == 'pusht' else 'xyz'):
                row['actual_initial_'+coordinate] = float(positions[0, index])
                row['actual_final_'+coordinate] = float(positions[-1, index])
                row['actual_min_'+coordinate] = float(positions[:, index].min())
                row['actual_max_'+coordinate] = float(positions[:, index].max())
            row['actual_path_length_native_units'] = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
            clipped = sum(bool(t.get('executed_command', {}).get('clipped_components')) for t in trace[1:])
            row['steps_with_clipped_action'] = clipped
            commands = [t['executed_command'] for t in trace[1:]]
            raw_actions = np.asarray([a['raw'] for a in commands])
            row['raw_source_action_min_per_channel'] = json.dumps(raw_actions.min(axis=0).tolist())
            row['raw_source_action_max_per_channel'] = json.dumps(raw_actions.max(axis=0).tolist())
            row['source_action_range_scope'] = 'Executed native-step source commands; unused forecast tail excluded'
            row['source_clipped_component_fraction'] = sum(len(a['clipped_components']) for a in commands)/(7*len(commands))
            row['source_clipped_step_fraction'] = clipped/len(commands)
            clipping_mask = np.abs(raw_actions) > 1
            row['source_clipped_count_per_channel'] = json.dumps(clipping_mask.sum(axis=0).tolist())
            row['source_translation_clipped_component_fraction'] = float(clipping_mask[:, :3].mean())
            row['source_rotation_clipped_component_fraction'] = float(clipping_mask[:, 3:6].mean())
            row['source_gripper_clipped_step_fraction'] = float(clipping_mask[:, 6].mean())
            row['native_space_clipped_steps'] = sum(bool(a.get('native_space_clipped')) for a in commands)
            row['native_space_clipped_fraction'] = row['native_space_clipped_steps']/len(commands) if c['environment'] == 'simpler' else None
            if c['environment'] == 'simpler':
                row['native_action_space_low'] = json.dumps(commands[0]['native_space_low'])
                row['native_action_space_high'] = json.dumps(commands[0]['native_space_high'])
            inference_path = Path(r['inference_path'])
            requests = lines(inference_path); receipts[str(inference_path)] = sha(inference_path)
            timing = []
            for q in requests:
                response = q['response']; route_requests += 1
                quality_by_logs[mode] += 1
                seed_groups[(c['environment'], c['case'], q['request'])].add((q['seed'], response['noise_hash']))
                expected = dict(vision_encoder_calls=1, vlm_prefix_calls=0 if mode.startswith('draft_') else 1,
                                ae_layer_calls=0 if mode.startswith('draft_') else 180,
                                learned_head_calls=1 if mode.startswith('draft_') else 0,
                                verify_calls=0, constructor_full_record_calls=0,
                                ae_euler_updates=0 if mode.startswith('draft_') else 10)
                for name, value in expected.items():
                    if response['observed_calls'].get(name) != value:
                        violations.append(dict(kind='route_counter', environment=c['environment'], case=c['case'], mode=mode,
                                               request=q['request'], counter=name, expected=value,
                                               actual=response['observed_calls'].get(name)))
                if q['request'] > 0 and not response.get('startup_events'):
                    timing.append(response['timing']['request_wall_ms'])
            row['quality_requests'] = len(requests)
            row['same_shard_completed_request_wall_p50_ms'] = float(np.median(timing)) if timing else None
            rows.append(row)
    for k, values in seed_groups.items():
        if len(values) != 1:
            violations.append(dict(kind='cross_condition_seed_or_noise_mismatch', key=list(k)))
    initial_agreement = [dict(environment=k[0], case=k[1], distinct_initial_raw_hashes=len(v), equal_across_conditions=len(v) == 1)
                         for k, v in sorted(initial_groups.items())]
    for r in initial_agreement:
        if not r['equal_across_conditions']:
            violations.append(dict(kind='cross_condition_initial_input_mismatch', **r))
    source_snapshots = {}
    for filename, expected in manifest['source_files'].items():
        copied = output/'source_snapshot'/filename
        actual = sha(copied)
        if actual != expected:
            raise ValueError('Frozen source snapshot differs')
        source_snapshots[filename] = actual
    csv_save(destination/'episode_outcomes.csv', rows)
    csv_save(destination/'attempt_history.csv', attempt_history)
    csv_save(destination/'initial_input_agreement.csv', initial_agreement)
    if len(case6_trajectories) == 6:
        plot_case6(destination, case6_trajectories)
    controls = []
    for env in ENVIRONMENTS:
        full = [r for r in rows if r['environment'] == env and r['mode'] == 'full_d10' and r['status'] == 'complete']
        controls.append(dict(environment=env, full_successes=sum(r['success'] for r in full), full_valid=len(full),
                             conclusion='Current deployment inconclusive for capacity/generalization attribution' if not any(r['success'] for r in full)
                             else 'Only observed frozen deployment differences; no causal capacity attribution'))
    service_receipts, aggregate_counters = {}, defaultdict(Counter)
    for p in sorted(output.glob('shard*_service_*.json')):
        service = json.loads(p.read_text())
        if service.get('manifest_hash') != manifest['manifest_hash']:
            continue
        service_receipts[str(p)] = dict(sha256=sha(p), status=service['status'],
                                        legacy_gate_passed=service.get('legacy_gate_passed'),
                                        category_counters=service.get('category_counters', {}))
        for category, modes in service.get('category_counters', {}).items():
            for mode, counters in modes.items():
                aggregate_counters[(category, mode)].update({k: v for k, v in counters.items() if isinstance(v, (int, float))})
    aggregate_rows = [dict(category=k[0], mode=k[1], **v) for k, v in sorted(aggregate_counters.items())]
    csv_save(destination/'all_service_counters.csv', aggregate_rows)
    for mode in MODES:
        count = aggregate_counters[('quality', mode)]['logical_requests']
        if count != quality_by_logs[mode]:
            violations.append(dict(kind='service_vs_selected_quality_log_count', mode=mode,
                                   service_count=count, selected_log_count=quality_by_logs[mode]))
    receipt = dict(status='passed' if not violations and complete_count == 108 else 'partial_or_failed_checks',
                   manifest_hash=manifest['manifest_hash'], completed_native_episodes=complete_count, planned=108,
                   native_results_and_logs_sha256=receipts, quality_requests_audited=route_requests,
                   attempt_results_sha256=attempt_hashes, total_attempts=len(attempt_history),
                   historical_technical_attempts=sum(r['status'] == 'technical_failure' for r in attempt_history),
                   all_service_receipts=service_receipts,
                   violations=violations, full_controls=controls, frozen_source_snapshot_sha256=source_snapshots,
                   postprocessing_source=dict(path=str(Path(__file__).relative_to(ROOT)), sha256=sha(__file__)),
                   no_model_calls=True, no_environment_calls=True, no_frozen_source_changes=True)
    write_json(destination/'audit.json', receipt)
    text = ['# E4 逐案例终止与真实轨迹审计', '',
            f'有效原生episode：{complete_count}/108。此补充只读取既有记录，无模型调用、环境调用或冻结源码修改。', '',
            '“零样本”指未在目标环境训练或微调模型权重、未按结果挑选head；跨环境观测与控制接口是在运行前明确设计并冻结的，不表示没有接口桥接。实际成功样例按原生判据如实保留，不能预设全部draft失败，也不能由少量失败证明模型只能用于LIBERO。', '',
            '[逐案例：原生终止、实际步数、真实初末及范围XYZ/XY、源动作范围、两级裁剪和请求耗时](episode_outcomes.csv)', '',
            f'[全部尝试历史](attempt_history.csv)：{len(attempt_history)}次尝试，包含{sum(r["status"] == "technical_failure" for r in attempt_history)}次技术失败；历史失败保留且不并入策略成功率分母。首轮PushT在第0步因本地源码搜索路径缺失退出，随后只补齐父进程PYTHONPATH并使用同一不可变manifest续跑。前78条有效机械臂结果全部跳过，原attempt000未覆盖。', '',
            '[合并两个持久服务的分类路由计数](all_service_counters.csv)：重复原输入门控属于preflight，质量调用单独累计，并与选定有效episode的推理日志逐条件核对。', '',
            '[各条件相同初始原始观测哈希核对](initial_input_agreement.csv)', '',
            'XYZ来自原生环境的实测末端；PushT仅有原生XY像素，未补造Z。路径长度是测得位置差累加，不是模型命令累加。源动作裁剪只统计已执行命令，分别列出平移、旋转、夹爪和逐通道计数；夹爪接近±1时的小幅越界属于常见裁剪，七维汇总比例不能据此解释为不稳定或失败原因。另列Simpler原生physical action space的再次裁剪。耗时只限同一GPU分片、当前Triton实现，排除各episode首请求与记录的启动事件，不能与其他实验的JAX延迟混比。', '',
            '| 环境 | Full成功/有效 | 解释边界 |', '|---|---:|---|']
    for c in controls:
        text.append(f'| {c["environment"]} | {c["full_successes"]}/{c["full_valid"]} | {c["conclusion"]} |')
    text += ['', f'审计检查异常数：{len(violations)}。完整细节、所有来源hash和原始冻结源码快照校验见[audit.json](audit.json)。', '']
    if len(case6_trajectories) == 6:
        text += ['## 事后选取的成功对照样例', '',
                 '下图选择CALVIN case6（push_pink_block_right），因为运行后观察到V14和draft_object成功而Full失败；这是明确的事后选择，不是无偏代表样例。全部六条件、完整原生实测轨迹均保留，上排为全轨迹，下排统一展示前50个原生控制步以便辨认早期差异。没有重跑模型、环境或视频；预注册的case0图与18个视频仍是主报告固定索引。', '',
                 '![事后选择的CALVIN case6真实XYZ投影](posthoc_calvin_case6_xyz.png)', '']
    (destination/'report_zh.md').write_text('\n'.join(text))
    receipt['outputs_sha256'] = {p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != 'audit.json'}
    write_json(destination/'audit.json', receipt)
    print(json.dumps(dict(status=receipt['status'], complete=complete_count, violations=len(violations), output=str(destination)), indent=2))


def plot_case6(destination, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .transfer_report import STYLES
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, (i, j, label) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
        for r in trajectories:
            for row in (0, 1):
                p = r['positions'] if row == 0 else r['positions'][:51]
                style = dict(STYLES[r['mode']], linewidth=1.5)
                if 'marker' in style:
                    style.update(markevery=max(1, len(p)//16), markersize=3)
                axes[row, col].plot(p[:, i], p[:, j], label=r['mode']+(' success' if r['success'] else ' failure'), **style)
        for row in (0, 1):
            axes[row, col].set_title(label+(' complete trajectory' if row == 0 else ' first50 native steps'))
            axes[row, col].set_xlabel('Actual '+label[0]+' (m)'); axes[row, col].set_ylabel('Actual '+label[1]+' (m)')
            axes[row, col].grid(alpha=.2)
    axes[0, -1].legend(fontsize=7)
    fig.suptitle('Post-hoc selected CALVIN case6: all six measured EEF trajectories (not an unbiased representative)')
    fig.tight_layout(); fig.savefig(destination/'posthoc_calvin_case6_xyz.png', dpi=170); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', required=True)
    audit(parser.parse_args().output)
