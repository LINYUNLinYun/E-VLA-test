"""Summarize pre-existing Nsight attribution outputs; CPU-only, stdlib only."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sqlite3
import statistics

OUT = Path(__file__).resolve().parent
CASES = {
    'd10_full_serial_profile': ('Full serial', 0, 0, 0, 0),
    'd10_full_overlap_profile': ('Full layer overlap', 0, 0, 0, 0),
    'd10_fixed16_profile': ('Fixed AE16/Prefix112', 16, 16, 112, 112),
    'd10_dynamic16_profile': ('AE16 then primary / Prefix112', 16, 0, 112, 112),
    'd10_combined_profile': ('Full combined crossframe', 0, 0, 0, 0),
    'd10_encoder64_vlmfull_profile_resumed': ('AE64/Vision64/VLM primary', 64, 64, 0, 64),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def union_ms(rows):
    merged = []
    for start, end in sorted((r['start'], r['end']) for r in rows):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(e - s for s, e in merged) / 1e6


def phase(r):
    if r['stage'] in ('VisionEncoder', 'VisionProjector'):
        return 'Vision'
    if r['stage'] == 'VLM':
        return 'VLM'
    if r['branch'] == 'AE' and r['iteration'] == 1:
        return 'AE1'
    if r['branch'] == 'AE' and (r['iteration'] or 0) >= 2:
        return 'AE_tail2to10'
    return None


def resource_audit(report, kernels, expected):
    db = sqlite3.connect('file:' + report['source'] + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    contexts = [dict(r) for r in db.execute('SELECT * FROM TARGET_INFO_CUDA_CONTEXT_INFO')]
    prepare_ranges = [dict(r) for r in db.execute(
        "SELECT coalesce(n.text,s.value) AS label,count(*) AS ranges FROM NVTX_EVENTS n "
        "LEFT JOIN StringIds s ON n.textId=s.id "
        "WHERE lower(coalesce(n.text,s.value,'')) LIKE '%prepar%' GROUP BY 1")]
    runtime_api = [dict(r) for r in db.execute(
        "SELECT s.value AS name,count(*) AS calls,sum(r.end-r.start)/1e6 AS api_elapsed_ms "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "WHERE s.value LIKE '%Synchronize%' OR s.value LIKE '%Memcpy%' GROUP BY 1")]
    copies = [dict(r) for r in db.execute(
        "SELECT k.copyKind,e.label AS direction,count(*) AS copies,sum(k.bytes) AS bytes,"
        "sum(k.end-k.start)/1e6 AS summed_gpu_elapsed_ms FROM CUPTI_ACTIVITY_KIND_MEMCPY k "
        "LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON k.copyKind=e.id GROUP BY 1,2")]
    db.close()
    context_index = {(r['processId'], r['deviceId'], r['contextId']): r for r in contexts}
    counts, failures = Counter(), []
    by_frame = defaultdict(list)
    for r in kernels:
        if not r['frame_id'] or not r['branch']:
            continue
        by_frame[r['frame_id']].append(r)
        p = phase(r)
        if p is None:
            continue
        green = r['green_context']
        context = context_index.get(((r['process'] >> 24) & 0xffffff, r['device'], green))
        sms = 0 if green == 0 else context.get('numMultiprocessors') if context else None
        counts[p, green, sms] += 1
        if sms != expected[p]:
            failures.append(dict(kernel=r['kernel_rowid'], phase=p, expected=expected[p], actual=sms))
    boundaries = []
    for fid, rows in by_frame.items():
        ae1 = [r for r in rows if phase(r) == 'AE1']
        tail = [r for r in rows if phase(r) == 'AE_tail2to10']
        vlm = [r for r in rows if phase(r) == 'VLM']
        if ae1 and tail and vlm:
            gap = min(r['start'] for r in tail) - max(r['end'] for r in ae1 + vlm)
            boundaries.append(dict(frame=fid, tail_after_ae1_and_vlm=gap >= 0,
                                   model_kernel_gap_us=gap / 1e3,
                                   iterations=sorted({r['iteration'] for r in ae1 + tail})))
    return dict(resource_passed=not failures, expected=expected, contexts=contexts,
                host_input_preparation=dict(prepare_nvtx_ranges=prepare_ranges,
                    whole_trace_sync_and_copy_api=runtime_api, whole_trace_gpu_copies=copies,
                    scope='Whole profile only; without a dedicated prepare range, these cannot all be assigned to Host input preparation. prepare_cpu_ms includes deepcopy/transforms/H2D/preprocess, not pure CPU arithmetic.'),
                counts=[dict(phase=p, green_context=g, sms=s, kernels=n)
                        for (p, g, s), n in counts.items()], failures=failures,
                tail_boundaries=boundaries,
                definition='sms=0 means unrestricted primary-context eligibility, not zero SM or observed occupancy. Tail order is reported; full ungated modes need not pass it.')


def signature_summary(name, comparison):
    groups = defaultdict(list)
    for r in comparison['matches']:
        p = phase(r)
        if p is None:
            continue
        kernel_gemm = bool(re.search(r'gemm|gemv|matmul', r['name'], re.I))
        if kernel_gemm:
            groups[p, 'GEMM_named_kernels'].append(r)
            if r['operation'] == 'FFN':
                groups[p, 'FFN_GEMM_named_kernels'].append(r)
        if r['operation'] == 'Attention_non_projection':
            groups[p, 'Attention_non_projection'].append(r)
        groups[p, 'All_matched_model_kernels'].append(r)
    output = []
    for (p, category), rows in sorted(groups.items()):
        baseline = sum(r['baseline_mean_us'] * min(r['baseline_calls'], r['candidate_calls']) for r in rows)
        candidate = sum(r['candidate_mean_us'] * min(r['baseline_calls'], r['candidate_calls']) for r in rows)
        output.append(dict(case=name, phase=p, category=category, signature_types=len(rows),
                           common_calls=sum(min(r['baseline_calls'], r['candidate_calls']) for r in rows),
                           count_mismatched_types=sum(r['baseline_calls'] != r['candidate_calls'] for r in rows),
                           weighted_baseline_ms=baseline / 1000, weighted_candidate_ms=candidate / 1000,
                           candidate_over_baseline=candidate / baseline if baseline else None))
    return output


def main():
    cases, stages, overlaps, slowdowns, resource_audits = [], [], [], [], {}
    iteration_overlaps, crossframe_boundaries = [], []
    for name, (label, early, tail, vlm, vision) in CASES.items():
        folder = OUT / name
        if not (folder / 'report.json').exists():
            continue
        report = json.loads((folder / 'report.json').read_text())
        if len(report['groups']) != 1:
            raise ValueError('Expected one measured profile group: ' + name)
        group_name, group = next(iter(report['groups'].items()))
        n = len(report['requests'])
        with (folder / 'kernels.csv').open() as f:
            kernels = list(csv.DictReader(f))
        for r in kernels:
            for key in ('start', 'end', 'process', 'device', 'green_context', 'kernel_rowid', 'iteration'):
                r[key] = int(r[key]) if r[key] not in ('', 'None') else None
        measured = [r for r in kernels if r['group'] == group_name and r['frame_id'] and r['branch']]
        by_stage = defaultdict(list)
        by_frame = defaultdict(list)
        for r in measured:
            by_frame[r['frame_id']].append(r)
            p = phase(r)
            if p:
                by_stage[r['frame_id'], p].append(r)
        for p in ('Vision', 'VLM', 'AE1', 'AE_tail2to10'):
            selected = [rows for (fid, phase_name), rows in by_stage.items() if phase_name == p]
            busy = [union_ms(rows) for rows in selected]
            envelope = [(max(r['end'] for r in rows) - min(r['start'] for r in rows))/1e6 for rows in selected]
            stages.append(dict(case=name, phase=p, frames=len(selected),
                               kernel_busy_mean_ms=statistics.mean(busy),
                               kernel_busy_p50_ms=statistics.median(busy),
                               kernel_envelope_mean_ms=statistics.mean(envelope),
                               kernel_envelope_p50_ms=statistics.median(envelope)))
        for pair in ('AE_x_VLM', 'AE_x_Vision', 'AE_x_Prefill'):
            m = group['pairs'][pair]
            overlaps.append(dict(case=name, pair=pair, frames=n,
                overlap_total_ms=m['overlap_ms'], overlap_per_frame_ms=m['overlap_ms']/n,
                same_frame_total_ms=m['same_frame_overlap_ms'], cross_frame_total_ms=m['cross_frame_overlap_ms'],
                cross_frame_per_adjacent_opportunity_ms=m['cross_frame_overlap_ms']/max(1,n-1),
                frame_pairs_with_overlap=len(m['frame_pairs']), kernel_union_ms=m['kernel_union_ms'],
                overlap_over_union=m['overlap_over_union']))
        for iteration, pairs in group['iteration_pairs'].items():
            for pair, m in pairs.items():
                iteration_overlaps.append(dict(case=name, iteration=int(iteration), pair=pair,
                    overlap_total_ms=m['overlap_ms'], overlap_per_frame_ms=m['overlap_ms']/n,
                    same_frame_total_ms=m['same_frame_overlap_ms'], cross_frame_total_ms=m['cross_frame_overlap_ms']))
        expected = dict(AE1=early, AE_tail2to10=tail, VLM=vlm, Vision=vision)
        resource_audits[name] = resource_audit(report, measured, expected)
        comparison_path = folder / 'signature_comparison.json'
        comparison = json.loads(comparison_path.read_text()) if comparison_path.exists() else None
        if comparison:
            slowdowns.extend(signature_summary(name, comparison))
        next_frame_orders = []
        ordered = sorted(by_frame.values(), key=lambda rows: min(r['start'] for r in rows))
        # Logical frame index, rather than first timestamp: next Vision starts early.
        ordered.sort(key=lambda rows: (int(rows[0]['epoch']), int(rows[0]['frame'])))
        for previous, current in zip(ordered, ordered[1:]):
            ae = [r for r in previous if r['branch'] == 'AE']
            vlm_rows = [r for r in current if r['stage'] == 'VLM']
            if ae and vlm_rows:
                delta = min(r['start'] for r in vlm_rows) - max(r['end'] for r in ae)
                next_frame_orders.append(dict(previous_frame=previous[0]['frame'], next_frame=current[0]['frame'],
                                              next_vlm_after_previous_ae=delta>=0, gap_us=delta/1000))
                ae1 = [r for r in ae if r['iteration'] == 1]
                vision_rows = [r for r in current if phase(r) == 'Vision']
                if ae1 and vision_rows:
                    vstart = min(r['start'] for r in vision_rows)
                    crossframe_boundaries.append(dict(case=name,
                        previous_frame=previous[0]['frame'], next_frame=current[0]['frame'],
                        next_vision_start_after_ae1_ms=(vstart-max(r['end'] for r in ae1))/1e6,
                        next_vision_start_before_ae_final_ms=(max(r['end'] for r in ae)-vstart)/1e6,
                        next_vision_envelope_ms=(max(r['end'] for r in vision_rows)-vstart)/1e6,
                        next_vlm_after_previous_ae=delta>=0, next_vlm_gap_us=delta/1000))
        resource_audits[name]['next_frame_vlm_order'] = next_frame_orders
        save(folder / 'resource_audit_extended.json', resource_audits[name])
        case = dict(name=name, label=label, source=report['source'], group=group_name, frames=n,
                    attribution=report['attribution'], iterations=report['iterations_seen'],
                    resource_passed=resource_audits[name]['resource_passed'],
                    matched_signature_types=len(comparison['matches']) if comparison else None,
                    baseline_unmatched_types=comparison['baseline_unmatched_types'] if comparison else None,
                    candidate_unmatched_types=comparison['candidate_unmatched_types'] if comparison else None,
                    next_frame_vlm_order=next_frame_orders)
        cases.append(case)
    save(OUT/'trace_summary.json', dict(cases=cases, stages=stages, overlaps=overlaps, slowdowns=slowdowns,
         iteration_overlaps=iteration_overlaps, crossframe_boundaries=crossframe_boundaries,
         definitions=['All elapsed/overlap figures are profiler diagnostics, not steady-state end-to-end results.',
                      'Busy=union of actual CUPTI kernel intervals; envelope=first to last kernel. Envelopes include gaps and waits.',
                      'Signature ratios match name, launch geometry, stage/iteration/layer/operation, weighted by common call counts.',
                      'Attention_non_projection includes RoPE/reshape/repeat/mask kernels; not pure attention arithmetic.',
                      'Ordinary overlap uses whole AE graph; fixed/dynamic16 use split AE graphs. Compare matched kernels, not profile wall time.']))
    csv_write(OUT/'stage_summary.csv', stages)
    csv_write(OUT/'overlap_summary.csv', overlaps)
    csv_write(OUT/'iteration_overlap.csv', iteration_overlaps)
    csv_write(OUT/'crossframe_boundaries.csv', crossframe_boundaries)
    csv_write(OUT/'matched_kernel_slowdowns.csv', slowdowns)
    save(OUT/'resource_audits.json', resource_audits)
    def stage(name,p,k='kernel_busy_mean_ms'):
        return next(r[k] for r in stages if r['case']==name and r['phase']==p)
    def ov(name,p,k='overlap_per_frame_ms'):
        return next(r[k] for r in overlaps if r['case']==name and r['pair']==p)
    def ratio(name,p,cat='GEMM_named_kernels'):
        hit=next((r for r in slowdowns if r['case']==name and r['phase']==p and r['category']==cat),None)
        return f"{hit['candidate_over_baseline']:.3f}×" if hit else '—'
    lines=['# D10 Nsight 离线分析', '',
           '仅分析已生成 trace；未运行模型或 GPU 工作。各条为 8 个正式请求的短 profile。',
           '这些数字只解释实际 Kernel 重叠和并发减速，不能替代非 profile 的 100 对端到端结果。', '',
           '| 配置 | 帧数 | AE×VLM /帧 ms | AE×下一帧 Vision /机会 ms | VLM busy /帧 ms | AE1 busy /帧 ms | AE2–10 busy /帧 ms |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        n=c['name']
        lines.append(f"| {c['label']} | {c['frames']} | {ov(n,'AE_x_VLM'):.3f} | {ov(n,'AE_x_Vision','cross_frame_per_adjacent_opportunity_ms'):.3f} | {stage(n,'VLM'):.3f} | {stage(n,'AE1'):.3f} | {stage(n,'AE_tail2to10'):.3f} |")
    lines += ['', '下一帧 Vision 的分母为 N−1 个相邻帧机会；同帧/跨帧交集不能重复相加。busy 是真实 Kernel 区间并集，包含 Kernel 内部停顿，不是 SM 利用率。', '',
              '## 简短结论', '',
              '- 普通逐层方案已有真实 AE×VLM 重叠；AE1 的 GEMM/Attention 区域明显延长，后九轮基本恢复串行速度。',
              '- 固定 AE16 会持续拖慢后九轮；动态切回 primary 后，该尾段的匹配 Kernel 时间恢复到串行附近。动态释放资源本身有效，但 Prefix112 的 VLM 仍慢于不限额 VLM。',
              '- 跨帧 combined 保留 AE 后续轮次与下一帧 Vision 的真实交集，同时 AE 尾段也有减速。它是吞吐窗口，不代表单帧响应下降同样多。',
              '- Encoder64 的 VLM 已实证运行在 primary，VLM Kernel 时间恢复串行附近；其短 trace 中 6/7 次相邻帧产生 Encoder 交集。下一 Encoder 在上一 AE1 结束后 16.2–24.5 ms 才开始，最后一次已错过 AE 尾段。此边界说明窗口利用受启动时机影响，不能仅凭它把原因归给 CPU 计算；详见 crossframe_boundaries.csv。',
              '- 普通 ungated 路线允许 AE2 与最后 VLM 尾部有小段重叠；固定/动态 split 的 full-Prefix gate 则显式排除此段。逐轮交集见 iteration_overlap.csv。',
              '- 当前结论只限这些 8 请求的短 profile；正式选择以同配置非 profile 测量为准。', '',
              '## 匹配 Kernel 的减速', '',
              '相对 Full serial，匹配名称、launch geometry、阶段/层/轮次后，按共同调用数加权。大于 1 表示相同 Kernel elapsed 变长。', '',
              '| 配置 | VLM GEMM | VLM FFN GEMM | AE1 GEMM | AE1 Attention 区域 | AE2–10 GEMM | AE2–10 Attention 区域 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for c in cases[1:]:
        n=c['name']
        lines.append(f"| {c['label']} | {ratio(n,'VLM')} | {ratio(n,'VLM','FFN_GEMM_named_kernels')} | {ratio(n,'AE1')} | {ratio(n,'AE1','Attention_non_projection')} | {ratio(n,'AE_tail2to10')} | {ratio(n,'AE_tail2to10','Attention_non_projection')} |")
    lines += ['', 'Attention 区域包含 RoPE、reshape、mask 等，不能全部归为 Attention 算术。未匹配 signature 类型与调用数差异见 JSON/CSV，不对其填充或推断。', '',
              '## 资源与生命周期证据', '',
              '- 从每个模型 Kernel 的 greenContextId 关联 TARGET_INFO_CUDA_CONTEXT_INFO，检查实际配额；primary=不限额资格，不等于实际占满所有 SM。',
              '- AE16 固定组应全十轮 16 SM；动态组应 AE1=16 SM、AE2–10=primary；两组 VLM=112 SM。',
              '- Encoder64 组单独验证 Vision=64 SM、AE=64 SM、VLM=primary，避免只检查 replay stream。',
              '- 逐帧保存尾段相对 AE1/VLM 的边界；普通 ungated overlap 不要求额外 full-Prefix 栅栏。跨帧另查下一 VLM 未越过上一 AE 最终完成。', '',
              '| 配置 | 所有 Graph 阶段映射 | 所有请求映射 | SM 归属通过 | 匹配 signature | 未匹配 baseline/candidate |',
              '|---|---|---|---|---:|---|']
    for c in cases:
        a=c['attribution']
        lines.append(f"| {c['label']} | {a['all_graph_stages_mapped']} | {a['all_graph_requests_mapped']} | {c['resource_passed']} | {c['matched_signature_types']} | {c['baseline_unmatched_types']}/{c['candidate_unmatched_types']} |")
    lines += ['', '## Host 输入准备口径', '',
              '`prepare_cpu_ms` 是 OpenPIBackend.prepare 的 Host 墙钟，包括 deepcopy、input_transform、NumPy→Torch H2D 和 observation preprocess，不是纯 CPU 算术或 Kernel 提交耗时。',
              '资源审计 JSON 同时保留整条 trace 的 memcpy / synchronize API 次数与累计时间、GPU 拷贝字节数，以及是否存在独立 prepare NVTX。没有 prepare 范围时不把全部拷贝/同步归给该阶段，也不以其 API 时间解释完整准备墙钟。']
    lines += ['', '## 文件', '',
              '- `stage_summary.csv`：阶段 Kernel busy 与 envelope；`overlap_summary.csv`：实际交集；`iteration_overlap.csv`：逐轮交集。',
              '- `per_iteration_times.md` / `per_iteration_times.csv`：逐帧十轮 velocity 范围的 busy/envelope；明确不含 Euler。`preparation_or_euler_times.csv` 单列范围外 Kernel。由同目录 per_iteration_times.py 生成。',
              '- `crossframe_boundaries.csv`：下一 Vision 的首 Kernel 相对当前 AE1/最终完成的边界，以及下一 VLM 的依赖检查。',
              '- `matched_kernel_slowdowns.csv`：按操作汇总；各子目录 `signature_comparison.csv` 保留全部匹配项。',
              '- `resource_audits.json`：每阶段 Green Context/SM 配额及逐帧边界。',
              '- 各配置子目录包含全部 `kernels.csv`、阶段与请求表、代表 `timeline.png`；`offline_receipt.json` 记录导出/复用来源。',
              '- 重现：`policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/offline_trace_matrix.py`，再运行同目录 `summarize_traces.py`。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(cases=[c['name'] for c in cases], resource_passed=all(c['resource_passed'] for c in cases))))


if __name__ == '__main__':
    main()
