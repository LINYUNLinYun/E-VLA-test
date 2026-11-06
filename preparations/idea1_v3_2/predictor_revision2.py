"""Explicit increment/cumulative direction diagnostics; frozen predictions reused.

The historical v2.1 L2 is per-command while its along/lateral are cumulative.
This revision names both vector spaces and audits their separate identities.
"""
import csv
import json
from pathlib import Path
import numpy as np
try:
    from .predictor_experiment import OUT, ROOT, WINDOWS, macro, row_mean, save, table, sha
except ImportError:
    from predictor_experiment import OUT, ROOT, WINDOWS, macro, row_mean, save, table, sha

DEST = OUT/'revision2'
METHODS = ['WLP', 'ConstantTurn', 'VAR', 'KalmanFrozen', 'OldKalman', 'HoldLast', 'V14']


def coherent_metrics(prediction, reference):
    p, f = np.clip(prediction, -1, 1)*50, np.clip(reference, -1, 1)*50
    error = p-f
    norms = np.linalg.norm(f, axis=2)
    tangent = np.divide(f, norms[:, :, None], out=np.zeros_like(f), where=norms[:, :, None] > 0)
    valid = norms >= 1
    valid[:, 1:] &= ~((np.sum(f[:, 1:]*f[:, :-1], axis=2) <= 0) & (norms[:, 1:] >= 1) & (norms[:, :-1] >= 1))
    result = {}
    for prefix, vector in [('increment', error), ('cumulative', np.cumsum(error, axis=1))]:
        norm = np.linalg.norm(vector, axis=2)
        along = np.sum(vector*tangent, axis=2)
        lateral = np.linalg.norm(vector-along[:, :, None]*tangent, axis=2)
        along[~valid] = np.nan; lateral[~valid] = np.nan
        result[prefix+'_l2_mm'] = norm
        result[prefix+'_along_signed_mm'] = along
        result[prefix+'_along_abs_mm'] = np.abs(along)
        result[prefix+'_lateral_mm'] = lateral
    return result


def run():
    DEST.mkdir(exist_ok=False)
    records = json.loads((OUT/'records.json').read_text())
    arrays = np.load(OUT/'predictions.npz', allow_pickle=False)
    winners = json.loads((OUT/'frozen_winners.json').read_text())['winners']
    summary, offsets, stages, identity = [], [], [], []
    for method in METHODS:
        d = coherent_metrics(arrays[method], arrays['Full'])
        for prefix in ('increment', 'cumulative'):
            residual = d[prefix+'_l2_mm']**2-d[prefix+'_along_signed_mm']**2-d[prefix+'_lateral_mm']**2
            maximum = float(np.nanmax(np.abs(residual)))
            if maximum > 1e-6:
                raise ValueError('Direction Pythagorean identity failed')
            identity.append(dict(method=method, vector_space=prefix, max_squared_identity_residual_mm2=maximum))
        for field, a in d.items():
            for window, lo, hi in WINDOWS:
                sample = a[:, lo:hi].copy()
                if window == '1_5':
                    for i, m in enumerate(records):
                        sample[i, m['valid_executed_prefix']:] = np.nan
                value = row_mean(sample)
                summary.append(dict(method=method, window=window, metric=field, macro_mean=macro(value, records),
                                    sample_p95=float(np.nanpercentile(sample, 95)),
                                    valid_samples=int(np.isfinite(sample).sum()), replans=int(np.isfinite(value).sum())))
                for phase in sorted({m['phase'] for m in records}):
                    mask = np.array([m['phase'] == phase for m in records])
                    selected = [m for m, keep in zip(records, mask) if keep]
                    stages.append(dict(method=method, window=window, metric=field, phase=phase,
                                       macro_mean=macro(value[mask], selected),
                                       replans=int(np.isfinite(value[mask]).sum()),
                                       valid_samples=int(np.isfinite(sample[mask]).sum())))
            for h in range(50):
                offsets.append(dict(method=method, offset=h+1, metric=field,
                                    macro_mean=macro(a[:, h], records), valid_samples=int(np.isfinite(a[:, h]).sum())))
    table(DEST/'summary.csv', summary); table(DEST/'offsets.csv', offsets); table(DEST/'stages.csv', stages)
    save(DEST/'direction_identity_audit.json', dict(status='passed', checks=identity,
         identity='For each separately named vector space: norm² = signed_along² + lateral²; only tangent-valid samples',
         valid_mask='Full command norm>=1mm; adjacent >90degree reversals excluded',
         warning='Averages are not expected to satisfy the squared identity; audit applies per sample.'))
    receipt = dict(status='complete', purpose='Clarify direction metric vector spaces; no new search/fitting/evaluation predictions',
                   parent_predictions_sha256=sha(OUT/'predictions.npz'), parent_records_sha256=sha(OUT/'records.json'),
                   parent_frozen_winners_sha256=sha(OUT/'frozen_winners.json'),
                   parent_manifest_sha256=sha(OUT/'manifest.json'), parent_bootstrap_sha256=sha(OUT/'paired_bootstrap.csv'),
                   definitions=dict(increment='One normalized XYZ command difference ×50mm: not actual EEF motion.',
                                    cumulative='Sum of all command differences from offset1 to current offset ×50mm: not actual EEF path.',
                                    direction='Project chosen vector on current Full command direction, same tangent/mask as v2.1.'),
                   previous_fields=dict(l2_mm='increment_l2_mm', along_mm='cumulative_along_signed_mm', lateral_mm='cumulative_lateral_mm'),
                   code_sha256={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('*predictor*.py')})
    save(DEST/'receipt.json', receipt)
    make_figures(summary, offsets)
    make_report(summary, winners)
    save(DEST/'results.json', dict(summary=summary, definitions=receipt['definitions'], status='complete'))
    print(json.dumps(dict(status='complete', revision=str(DEST), identity_max=max(r['max_squared_identity_residual_mm2'] for r in identity))))


def make_figures(summary, offsets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for row, prefix in enumerate(('increment', 'cumulative')):
        for col, suffix in enumerate(('l2_mm', 'along_signed_mm', 'lateral_mm')):
            ax = axs[row, col]; metric = prefix+'_'+suffix
            for method in METHODS:
                rs = [r for r in offsets if r['method'] == method and r['metric'] == metric]
                ax.plot([r['offset'] for r in rs], [r['macro_mean'] for r in rs], label=method)
            ax.set_title(metric); ax.set_xlabel('Prediction offset'); ax.set_ylabel('Control-target proxy mm')
            ax.axvline(5, color='grey', ls=':'); ax.grid(alpha=.2)
    axs[-1, -1].legend(fontsize=7); fig.tight_layout(); fig.savefig(DEST/'direction_curves.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (label, lo, hi) in zip(axs, WINDOWS):
        values = [next(r['macro_mean'] for r in summary if r['method'] == m and r['window'] == label and r['metric'] == 'increment_l2_mm') for m in METHODS]
        ax.bar(METHODS, values); ax.set_title('Offsets '+label); ax.tick_params(axis='x', rotation=55)
        ax.set_ylabel('Increment control-target proxy L2 mm'); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(DEST/'window_comparison.png', dpi=160); plt.close(fig)


def make_report(summary, winners):
    lookup = {(r['method'], r['window'], r['metric']): r for r in summary}
    pairs = list(csv.DictReader((OUT/'paired_bootstrap.csv').open()))
    text = ['# v3.2 E3 最终分析（方向口径修订2）', '',
            '本轮没有发现可替代冻结 v2.1 Kalman 的更好历史命令预测器。51个候选全部未通过预先固定的校准尾部约束；以下三类最佳候选均是探索性结果，不具部署准入资格。', '',
            '指标全部是平移控制目标代理误差（normalized XYZ command ×50mm），不是实际末端 EEF 误差。increment 是单步命令差；cumulative 是从offset1累加到当前offset的命令差。两类向量分别计算自己的L2、along、lateral，不能混作同一向量分解。', '',
            '沿用同一数据：task0 的20 episode /634重规划点校准；task1–9 的180 episode /6345点评估。每个预测仅可访问当前时间之前的已执行命令；Full/V14为同观测离线标签。固定51候选，按五个official init分折；每折首5步增量L2 p95与累计lateral p95均不得超过旧Kalman的105%。参数在评估前冻结，无评估调参。本次是复用已分析历史数据，非独立盲测。', '',
            '| 方法 | 首1–5步 increment L2 mm | 6–12步 increment L2 mm | 13–50步 increment L2 mm | 校准准入 |',
            '|---|---:|---:|---:|---|']
    for method in METHODS:
        values = [lookup[method, w, 'increment_l2_mm']['macro_mean'] for w, _, _ in WINDOWS]
        flag = ('通过' if winners[method]['eligible'] else '未通过') if method in winners else '冻结参考'
        text.append(f'| {method} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | {flag} |')
    text += ['', '宏平均按replan→episode→task→suite逐级等权；首5步按真实剩余执行前缀截断。offset6–50只是当前Full chunk的诊断，不是未来真实执行标签。', '',
             '| 方法 vs 冻结Kalman | 首5步增量 L2 Δmm | 配对95%区间 mm |', '|---|---:|---|']
    for method in ('WLP', 'ConstantTurn', 'VAR'):
        r = next(r for r in pairs if r['method'] == method and r['baseline'] == 'KalmanFrozen' and r['window'] == '1_5')
        text.append(f'| {method} | {float(r["delta_mm"]):+.3f} | [{float(r["ci95_low_mm"]):+.3f}, {float(r["ci95_high_mm"]):+.3f}] |')
    text += ['', '区间按suite分层重采样task簇，保留全部init/replan，2000次、seed3203；负数表示更好。未校正候选搜索与多重比较。', '',
             '## 单步命令向量方向分解（首5步）', '',
             '| 方法 | increment L2 | increment abs(along) | increment lateral | 方向有效样本 |', '|---|---:|---:|---:|---:|']
    for method in METHODS:
        vals = [lookup[method, '1_5', f'increment_{field}_mm'] for field in ('l2', 'along_abs', 'lateral')]
        text.append(f'| {method} | {vals[0]["macro_mean"]:.3f} | {vals[1]["macro_mean"]:.3f} | {vals[2]["macro_mean"]:.3f} | {vals[2]["valid_samples"]} |')
    text += ['', '## 累计控制目标代理向量分解（首5步各offset的均值）', '',
             '| 方法 | cumulative L2 | cumulative abs(along) | cumulative lateral |', '|---|---:|---:|---:|']
    for method in METHODS:
        vals = [lookup[method, '1_5', f'cumulative_{field}_mm']['macro_mean'] for field in ('l2', 'along_abs', 'lateral')]
        text.append(f'| {method} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |')
    text += ['', '方向使用当前Full命令方向；Full幅值<1mm或相邻反向切换时方向量无定义并剔除，因此L2与方向分量覆盖数也不同。每个有效样本在各自向量空间均满足L2²=along²+lateral²，direction_identity_audit.json已审计；表中均值不要求满足平方关系。signed along另见CSV。', '',
             'WLP采用加权局部多项式并锚定最后命令；ConstantTurn对近期命令方向做3D恒转向外推，保持最后命令幅值；VAR使用近期短阶线性自回归、ridge持久性先验和稳定性约束。这些模型拟合命令序列，未将EEF位置误当同类观测。局部拟合无未来标签；未使用未来协方差或阶段标签。', '',
             '冻结Kalman保持q_u=.0001、q_w=100、g=.5、beta=1、q_full；旧Kalman参数及原创新完整保留。每类预测器创新是当前时刻已执行最后命令相对其先前一步预测的残差，不是未来Full标签残差。', '',
             '阶段沿用来源启发式规则，仅作事后分组，未独立人工认证。时间偏移也是纯事后oracle：固定Full第4–9步，对比预测offset±3，并不用于推断或校正。完整阶段方向表见stages.csv，逐offset方向表见offsets.csv；父目录time_shift_summary.csv含分阶段时间诊断。', '',
             '数据证据：父目录source_receipt.json记录404个当前消费原始日志哈希及v2.1历史完整验证收据；图像/noise NPZ未进入预测器，历史验证继承且未重新读取。父目录supplement_audit.json另独立重放四套件task1/init0完整episode的旧/冻结Kalman及HoldLast，逐点逐50步核对。', '',
             '本修订仅重新分解既有冻结预测，未重选参数。receipt.json链接原predictions、records、frozen_winners及bootstrap的SHA256。父目录旧字段l2_mm对应increment_l2_mm，along_mm/lateral_mm对应cumulative方向量；最终报告以本修订显式名称为准。', '',
             '![各窗口增量L2](window_comparison.png)', '', '![分开的单步及累计方向分解](direction_curves.png)', '']
    (DEST/'report_zh.md').write_text('\n'.join(text))


if __name__ == '__main__':
    run()
