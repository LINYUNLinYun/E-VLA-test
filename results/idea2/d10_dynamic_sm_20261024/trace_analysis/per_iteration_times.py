"""CPU-only per-frame velocity-NVTX kernel timing from existing trace CSVs."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from summarize_traces import CASES, OUT, csv_write, save, union_ms


def measured(rows):
    return dict(kernel_count=len(rows), kernel_busy_ms=union_ms(rows),
                kernel_envelope_ms=(max(r['end'] for r in rows)-min(r['start'] for r in rows))/1e6,
                sum_kernel_elapsed_ms=sum(r['end']-r['start'] for r in rows)/1e6,
                first_kernel_ns=min(r['start'] for r in rows), last_kernel_ns=max(r['end'] for r in rows),
                green_contexts=','.join(str(x) for x in sorted({r['green_context'] for r in rows})))


def main():
    iterations, outside, checks, receipts = [], [], [], []
    for case in CASES:
        source=OUT/case/'kernels.csv'
        if not source.exists():
            raise FileNotFoundError(source)
        by_frame=defaultdict(list)
        with source.open() as handle:
            for row in csv.DictReader(handle):
                if row['branch']!='AE' or not row['frame_id'] or not row['group']:
                    continue
                for key in ('start','end','iteration','green_context'):
                    row[key]=int(row[key]) if row[key] not in ('','None') else None
                by_frame[row['frame_id']].append(row)
        for frame,rows in by_frame.items():
            first=rows[0]
            identity=dict(case=case,group=first['group'],frame_id=frame,
                          epoch=first['epoch'],frame=first['frame'],slot=first['slot'],generation=first['generation'])
            observed={r['iteration'] for r in rows if r['iteration'] is not None}
            if observed!=set(range(1,11)):
                raise ValueError(f'{case} {frame}: missing/unexpected iterations {observed}')
            frame_iterations=[]
            for step in range(1,11):
                selected=[r for r in rows if r['iteration']==step]
                item=identity|dict(iteration=step,scope='velocity_NVTX_excludes_Euler_and_time_update')|measured(selected)
                iterations.append(item)
                frame_iterations.append(item)
            residual=[r for r in rows if r['iteration'] is None]
            if any(r['operation']!='Preparation_or_Euler_update' for r in residual):
                raise ValueError(f'{case} {frame}: unlabeled AE kernel has an unexpected operation')
            residual_metrics=measured(residual) if residual else dict(kernel_count=0,kernel_busy_ms=0)
            outside.append(identity|dict(scope='AE_capture_outside_velocity_ranges_not_pure_Euler')|residual_metrics)
            partition_busy=sum(r['kernel_busy_ms'] for r in frame_iterations)+residual_metrics['kernel_busy_ms']
            whole=measured(rows)
            delta=partition_busy-whole['kernel_busy_ms']
            combined=[r for step in range(1,11) for r in rows if r['iteration']==step]+residual
            union_delta=union_ms(combined)-whole['kernel_busy_ms']
            checks.append(identity|dict(whole_AE_capture_kernel_count=whole['kernel_count'],
                classified_kernel_count=sum(r['kernel_count'] for r in frame_iterations)+residual_metrics['kernel_count'],
                whole_AE_capture_busy_ms=whole['kernel_busy_ms'],sum_partition_busy_ms=partition_busy,
                partition_overlap_or_roundoff_ms=delta,
                complete_kernel_partition=whole['kernel_count']==sum(r['kernel_count'] for r in frame_iterations)+residual_metrics['kernel_count'],
                partition_union_matches_whole=abs(union_delta)<1e-8,
                partitions_do_not_overlap=abs(delta)<1e-8))
        receipts.append(dict(case=case,source=str(source),frames=len(by_frame),
                             input_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    summary=[]
    for case in CASES:
        for step in range(1,11):
            rows=[r for r in iterations if r['case']==case and r['iteration']==step]
            summary.append(dict(case=case,iteration=step,frames=len(rows),
                kernel_count_per_frame=statistics.mean(r['kernel_count'] for r in rows),
                kernel_busy_mean_ms=statistics.mean(r['kernel_busy_ms'] for r in rows),
                kernel_busy_p50_ms=statistics.median(r['kernel_busy_ms'] for r in rows),
                kernel_envelope_mean_ms=statistics.mean(r['kernel_envelope_ms'] for r in rows),
                kernel_envelope_p50_ms=statistics.median(r['kernel_envelope_ms'] for r in rows)))
    csv_write(OUT/'per_iteration_times.csv',iterations)
    csv_write(OUT/'per_iteration_summary.csv',summary)
    csv_write(OUT/'preparation_or_euler_times.csv',outside)
    csv_write(OUT/'iteration_partition_checks.csv',checks)
    save(OUT/'per_iteration_receipt.json',dict(inputs=receipts,
        all_partition_checks_passed=all(r['complete_kernel_partition'] and r['partition_union_matches_whole'] for r in checks),
        maximum_summed_partition_busy_excess_us=max(r['partition_overlap_or_roundoff_ms'] for r in checks)*1000,
        scope='AE/iteration_N wraps original_velocity() only. Euler x=x+dt*velocity, time update, initial masks/position/time setup and final clone lie outside. Kernel busy is interval union; envelope includes gaps and KV-ready waits. Preparation_or_Euler_update is an aggregate, not pure Euler or cold capture cost.'))
    lines=['# D10 逐轮 velocity Kernel 统计','',
        '仅从六条已有 kernels.csv 做 CPU 统计；每条 8 帧，每帧 10 轮。预热/捕获不在这里计时。','',
        '`AE/iteration_N` 的 NVTX 仅包住 `original_velocity(...)`（layer_graphs.py:311）。因此下表是该轮 velocity 网络的 Kernel busy / envelope，不是含 Euler 的完整 Flow Matching 迭代。`x = x + dt * velocity`、`time += dt` 与最终 clone 在范围外（graph_sampler.py:127）。','',
        '- busy：该范围真实 CUPTI Kernel 区间的并集，包含 Kernel 内部停顿，不能当作纯算术时间或 SM 利用率。',
        '- envelope：首 Kernel 开始至末 Kernel 结束，包含 Kernel 间隙和逐层 KV-ready 等待；不含首 Kernel 启动前的等待。',
        '- Preparation_or_Euler_update：AE Graph 中全部 velocity 范围以外的 Kernel，混合 mask/position/time 准备、各轮 Euler/time 更新、最终 clone；当前标签不能将其精确拆成逐轮纯 Euler。这个集合的 envelope 跨过十轮 velocity，不应作为“准备耗时”或再加到 velocity envelope。',
        '- 这里的范围外 GPU 准备与 benchmark 的 `prepare_cpu_ms`（整个 Host 输入准备路径）是不同口径，不能用前者解释后者。',
        '- 固定 schedule/调制准备在冷构图时完成，这里的热态范围外 Kernel 不代表重复生成所有调制。Graph 捕获与预热均已结束，AE1 不等于首次编译/权重加载。','',
        '| 配置 | velocity1 busy ms | velocity1 envelope ms | velocity2–10 平均每轮 busy ms | velocity2–10 平均每轮 envelope ms | 范围外 busy/帧 ms |',
        '|---|---:|---:|---:|---:|---:|']
    for case,(label,*_) in CASES.items():
        first=next(r for r in summary if r['case']==case and r['iteration']==1)
        tail=[r for r in summary if r['case']==case and r['iteration']>=2]
        residual=[r for r in outside if r['case']==case]
        lines.append(f"| {label} | {first['kernel_busy_mean_ms']:.3f} | {first['kernel_envelope_mean_ms']:.3f} | {statistics.mean(r['kernel_busy_mean_ms'] for r in tail):.3f} | {statistics.mean(r['kernel_envelope_mean_ms'] for r in tail):.3f} | {statistics.mean(r['kernel_busy_ms'] for r in residual):.3f} |")
    lines+=['','简要解读：串行热态的第一轮与后九轮基本同量级；逐层重叠时第一轮的长 envelope 主要反映等待，busy 本身也变长。动态 SM 让后九轮恢复串行附近，而固定 AE16 的每一轮都持续变慢。combined 的后半段 velocity 成本逐渐升高，与下一帧 Encoder 重叠窗口相符；这仍是短 trace 的诊断，不能替代正式端到端收益。']
    lines+=['','完整十轮逐轮均值（busy / envelope，ms）：','',
            '| 配置 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for case,(label,*_) in CASES.items():
        cells=[f"{r['kernel_busy_mean_ms']:.3f} / {r['kernel_envelope_mean_ms']:.3f}" for r in summary if r['case']==case]
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','输出：`per_iteration_times.csv`（480 条帧×轮记录）、`per_iteration_summary.csv`（60 条配置×轮汇总）、`preparation_or_euler_times.csv`（48 条帧记录）、`iteration_partition_checks.csv`（覆盖与区间并集核对）。每帧 velocity1–10 加范围外 Kernel 覆盖整个已归因 AE capture；没有按相邻时间位置猜测 Euler 所属轮次。',
            f"48 帧分类完整、合并后的区间并集均与完整 AE capture 一致。分别求 busy 再求和与全局并集的最大差为 {max(r['partition_overlap_or_roundoff_ms'] for r in checks)*1000:.3f} μs/帧；极小边界交集显式保留，不把分类互斥假设成时间区间绝不相交。",'',
            '重现：`policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/per_iteration_times.py`。']
    (OUT/'per_iteration_times.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(iteration_rows=len(iterations),outside_rows=len(outside),
        partition_checks_passed=all(r['complete_kernel_partition'] and r['partition_union_matches_whole'] for r in checks))))


if __name__=='__main__':
    main()
