"""Bounded CPU-only E3 experiment. Run with the existing policy .venv Python."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT/'policies/realtime-vla-flash'
sys.path.insert(0, str(POLICY))
from scripts.idea1_v2.core import Kalman, sha, SUITES, WEIGHT
from scripts.idea1_v2_1.filter import digest, CommandKalman, Parameters
from scripts.idea1_v2_1.diagnostics import decompose, phase_shift_oracle, PHASES
try:
    from .predictor import Config, candidates, predict_history
except ImportError:
    from predictor import Config, candidates, predict_history

OLD = POLICY/'results/idea1_v2_1_multimodel_kalman/run_20261016T045100Z/e4_revision3/e4_kalman'
SOURCE = POLICY/'results/experiment11_layer_skip_prefix_20261010'
OUT = ROOT/'results/idea1_v3_2/e3'
WINDOWS = [('1_5', 0, 5), ('6_12', 5, 12), ('13_50', 12, 50)]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def load_sources():
    """Revalidate consumed original commands/logs against v2.1 hash receipt.

    Image/noise NPZ files are not inputs to these predictors and are not reopened;
    their original source identities remain linked through reproduction.json.
    """
    receipt = json.loads((OLD/'reproduction.json').read_text())
    expected = receipt['sources']['files']
    consumed, groups, identities = {}, defaultdict(list), set()

    def checked(path):
        actual = sha(path)
        if expected[str(path)] != actual:
            raise ValueError('Original source hash changed: '+str(path))
        consumed[str(path)] = actual
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    for suite in SUITES:
        path = SOURCE/'shadow_supplement_v14/round2_20261011/V14_A03'/suite/'predictions.jsonl'
        for r in checked(path):
            key = (suite, int(r['task_id']), int(r['episode_idx']))
            identity = (*key, int(r['replan_id']))
            if identity in identities or r['checkpoint_sha256'] != WEIGHT or r['skip_layers'] != [3, 4, 9, 11]:
                raise ValueError('Source identity/protocol mismatch')
            identities.add(identity); groups[key].append(r)
    for key, records in sorted(groups.items()):
        records.sort(key=lambda r: r['environment_action_step'])
        infer = checked(Path(records[0]['full_infer_path']))
        telemetry = checked(Path(records[0]['full_infer_path']).with_name('telemetry.jsonl'))
        actions = np.asarray([r['action'] for r in telemetry[1:]], dtype=float)
        if [r['step'] for r in telemetry] != list(range(len(telemetry))):
            raise ValueError('Telemetry gap')
        if not np.allclose(np.diff([r['sim_time_s'] for r in telemetry]), .05, atol=1e-8, rtol=0):
            raise ValueError('Control clock changed')
        logged = {int(r['infer_id']): r for r in infer}
        if len(logged) != len(records):
            raise ValueError('Full source coverage mismatch')
        previous = -1
        for r in records:
            t = int(r['environment_action_step']); n = min(5, len(actions)-t)
            if t <= previous or n < 1:
                raise ValueError('Invalid prefix/time')
            previous = t
            full, v14 = np.asarray(r['full_actions']), np.asarray(r['variant_actions'])
            if full.shape != (50, 7) or v14.shape != (50, 7) or not np.isfinite([full, v14]).all():
                raise ValueError('Invalid source chunks')
            event = logged[int(r['replan_id'])]
            if t != event['frame_idx'] or event['experiment11']['noise_sha256'] != r['noise_sha256']:
                raise ValueError('Source event identity changed')
            if not np.allclose(event['chunk_actions'], full, atol=1e-6, rtol=0):
                raise ValueError('Full chunk differs from original log')
            if not np.allclose(actions[t:t+n], full[:n], atol=1e-5, rtol=1e-4):
                raise ValueError('Executed prefix mismatch')
            r.update(_actions=actions, _prefix=n, _key=key)
    counts = {split: sum(len(v) for k, v in groups.items() if (k[1] == 0) == (split == 'calibration'))
              for split in ('calibration', 'evaluation')}
    if len(groups) != 200 or counts != dict(calibration=634, evaluation=6345):
        raise ValueError('Frozen split mismatch')
    return groups, dict(consumed_files=consumed, source_root=str(SOURCE), counts=counts,
                       episodes=200, consumed_file_count=len(consumed),
                       inherited_full_validation_sha256=sha(OLD/'reproduction.json'),
                       inherited_source_manifest_sha256=receipt['source_manifest_sha256'],
                       validation_scope='All consumed prediction/infer/telemetry logs rehashed and checked. Original image/noise NPZ validation inherited, not repeated.',
                       old_parameters=receipt['old_parameters'])


def metadata(records):
    return [dict(suite=r['_key'][0], task=r['_key'][1], init=r['_key'][2],
                 replan=int(r['replan_id']), step=int(r['environment_action_step']),
                 valid_executed_prefix=r['_prefix'], phase=PHASES.get(r.get('semantic_phase'), 'unknown'),
                 input_sha256=r['full_input_sha256'], noise_sha256=r['noise_sha256']) for r in records]


def macro(values, meta):
    # replan -> episode -> task -> suite, with missing cells excluded.
    groups = defaultdict(list)
    for value, m in zip(values, meta):
        if np.isfinite(value):
            groups[(m['suite'], m['task'], m['init'])].append(float(value))
    tasks = defaultdict(list)
    for (suite, task, init), v in groups.items():
        tasks[(suite, task)].append(float(np.mean(v)))
    suites = defaultdict(list)
    for (suite, task), v in tasks.items():
        suites[suite].append(float(np.mean(v)))
    return float(np.mean([np.mean(v) for v in suites.values()])) if suites else None


def metrics(prediction, full):
    p, f = np.clip(prediction, -1, 1)*50, np.clip(full, -1, 1)*50
    error = p-f
    norm = np.linalg.norm(f, axis=2)
    tangent = np.divide(f, norm[:, :, None], out=np.zeros_like(f), where=norm[:, :, None] > 0)
    valid = norm >= 1
    valid[:, 1:] &= ~((np.sum(f[:, 1:]*f[:, :-1], axis=2) <= 0) & (norm[:, 1:] >= 1) & (norm[:, :-1] >= 1))
    delta = np.cumsum(error, axis=1)
    along = np.sum(delta*tangent, axis=2)
    lateral = np.linalg.norm(delta-along[:, :, None]*tangent, axis=2)
    along[~valid] = np.nan; lateral[~valid] = np.nan
    return dict(l2_mm=np.linalg.norm(error, axis=2), along_mm=along, lateral_mm=lateral)


def row_mean(a):
    count = np.isfinite(a).sum(axis=1)
    return np.divide(np.nansum(a, axis=1), count, out=np.full(len(a), np.nan), where=count > 0)


def prefix_metrics(pred, full, records):
    d = metrics(pred, full)
    for i, r in enumerate(records):
        for a in d.values():
            a[i, r['_prefix']:] = np.nan
    return d


def score(pred, full, records):
    d, m = prefix_metrics(pred, full, records), metadata(records)
    return dict(mean_mm=macro(row_mean(d['l2_mm']), m),
                l2_p95_mm=float(np.nanpercentile(d['l2_mm'], 95)),
                lateral_p95_mm=float(np.nanpercentile(d['lateral_mm'], 95)))


def old_predict(groups, parameters, horizon=5):
    result = []
    for key, records in sorted(groups.items()):
        model, last = Kalman(parameters['q0'], parameters['r0'], parameters['dt']), 0
        for r in records:
            t = int(r['environment_action_step'])
            for action in r['_actions'][last:t]:
                model.observe(action)
            last = t
            result.append(model.forecast(horizon)[0])
    return np.asarray(result)


def tune(calibration, old_parameters):
    records = [r for _, rs in sorted(calibration.items()) for r in rs]
    full = np.asarray([r['full_actions'] for r in records])[:, :5, :3]
    old = old_predict(calibration, old_parameters)
    folds = [np.asarray([r['_key'][2] == init for r in records]) for init in range(5)]
    reference = [score(old[mask], full[mask], [r for r, keep in zip(records, mask) if keep]) for mask in folds]
    results = []
    for number, c in enumerate(candidates(), 1):
        predicted = np.asarray([predict_history(r['_actions'], int(r['environment_action_step']), c, 5) for r in records])
        fs = []
        for init, mask in enumerate(folds):
            s = score(predicted[mask], full[mask], [r for r, keep in zip(records, mask) if keep])
            s.update(init=init, tail_pass=all(s[field] <= 1.05*reference[init][field]+1e-12
                                             for field in ('l2_p95_mm', 'lateral_p95_mm')))
            fs.append(s)
        result = dict(config=asdict(c), folds=fs, tail_pass=all(s['tail_pass'] for s in fs),
                      mean_mm=float(np.mean([s['mean_mm'] for s in fs])),
                      l2_p95_mm=float(np.mean([s['l2_p95_mm'] for s in fs])))
        results.append(result)
        print(f'Calibration {number}/{len(candidates())}: {c.family} {result["mean_mm"]:.4f} mm tail={result["tail_pass"]}', flush=True)
    winners = {}
    for family in ('WLP', 'ConstantTurn', 'VAR'):
        pool = [r for r in results if r['config']['family'] == family]
        pool.sort(key=lambda r: (not r['tail_pass'], r['mean_mm'], r['l2_p95_mm'], digest(r['config'])))
        winners[family] = pool[0]
        winners[family]['eligible'] = pool[0]['tail_pass']
    save(OUT/'calibration.json', dict(candidates=results, old_reference_folds=reference))
    save(OUT/'frozen_winners.json', dict(winners=winners, sha256=digest(winners),
         status='Frozen before evaluation predictions or metrics',
         rejected_family_rule='If no candidate passes all folds, best failed candidate shown as exploratory and ineligible'))
    return winners


def bootstrap(values, baseline, meta, iterations=2000):
    task_deltas = []
    for suite in SUITES:
        v = []
        for task in range(1, 10):
            mask = np.asarray([m['suite'] == suite and m['task'] == task for m in meta])
            selected = [m for m, keep in zip(meta, mask) if keep]
            v.append(macro(values[mask]-baseline[mask], selected))
        task_deltas.append(v)
    a = np.asarray(task_deltas)
    rng = np.random.default_rng(3203)
    indices = rng.integers(0, 9, (iterations, 4, 9))
    draws = np.take_along_axis(np.broadcast_to(a, (iterations, 4, 9)), indices, axis=2).mean(axis=(1, 2))
    return [float(x) for x in np.percentile(draws, [2.5, 97.5])]


def analyze(arrays, records, innovations):
    meta = metadata(records); summary, offsets, stages, pairs, oracle_rows = [], [], [], [], []
    all_metrics, record_rows = {}, []
    for method, pred in arrays.items():
        if method == 'Full':
            continue
        d = metrics(pred, arrays['Full']); all_metrics[method] = d
        for label, lo, hi in WINDOWS:
            for field, a in d.items():
                window = a[:, lo:hi].copy()
                if label == '1_5':
                    for i, r in enumerate(records):
                        window[i, r['_prefix']:] = np.nan
                value = row_mean(window)
                summary.append(dict(method=method, window=label, metric=field, macro_mean=macro(value, meta),
                                    point_mean=float(np.nanmean(window)), sample_p95=float(np.nanpercentile(window, 95)),
                                    valid_points=int(np.isfinite(window).sum()), replans=int(np.isfinite(value).sum())))
                for phase in sorted({m['phase'] for m in meta}):
                    mask = np.asarray([m['phase'] == phase for m in meta]); mm = [m for m, keep in zip(meta, mask) if keep]
                    stages.append(dict(method=method, window=label, metric=field, phase=phase,
                                       macro_mean=macro(value[mask], mm), replans=int(np.isfinite(value[mask]).sum()),
                                       episodes=len({(m['suite'], m['task'], m['init']) for m in mm})))
                if field == 'l2_mm':
                    for m, v in zip(meta, value):
                        record_rows.append(dict(method=method, window=label, l2_mm=float(v), **m))
            for baseline in ('KalmanFrozen', 'HoldLast', 'V14'):
                if method == baseline:
                    continue
                baseline_d = metrics(arrays[baseline], arrays['Full'])['l2_mm'][:, lo:hi].copy()
                a = d['l2_mm'][:, lo:hi].copy()
                if label == '1_5':
                    for i, r in enumerate(records):
                        a[i, r['_prefix']:] = np.nan; baseline_d[i, r['_prefix']:] = np.nan
                va, vb = row_mean(a), row_mean(baseline_d)
                ci = bootstrap(va, vb, meta)
                pairs.append(dict(method=method, baseline=baseline, window=label, delta_mm=macro(va-vb, meta),
                                  ci95_low_mm=ci[0], ci95_high_mm=ci[1], bootstrap=2000, seed=3203))
        for field, a in d.items():
            for h in range(50):
                offsets.append(dict(method=method, offset=h+1, metric=field, macro_mean=macro(a[:, h], meta),
                                    valid_points=int(np.isfinite(a[:, h]).sum())))
        for pred_chunk, full, m in zip(pred, arrays['Full'], meta):
            oracle = phase_shift_oracle(pred_chunk, full)
            oracle_rows.append(dict(method=method, suite=m['suite'], task=m['task'], init=m['init'], replan=m['replan'],
                                    phase=m['phase'], best_shift=oracle['best_shift'],
                                    zero_error_mm=oracle['zero_error_mm'], residual_mm=oracle['residual_mm'],
                                    improvement_mm=oracle['improvement_mm'], unidentifiable=oracle['unidentifiable']))
    table(OUT/'summary.csv', summary); table(OUT/'offsets.csv', offsets); table(OUT/'stages.csv', stages)
    table(OUT/'paired_bootstrap.csv', pairs); table(OUT/'record_windows.csv', record_rows); table(OUT/'time_shift_oracle.csv', oracle_rows)
    np.savez_compressed(OUT/'predictions.npz', **arrays, **{k+'_innovation': v for k, v in innovations.items()})
    save(OUT/'records.json', meta)
    save(OUT/'results.json', dict(summary=summary, paired_bootstrap=pairs,
                                metrics='Clipped normalized XYZ command differences ×50 mm control-target proxy, not actual EEF error',
                                along_lateral='Cumulative command proxy error projected on Full command tangent; Full magnitude >=1mm, reversals masked',
                                time_shift='Posthoc oracle only: Full offsets4..9 versus prediction shifts -3..+3, not used for fitting or inference'))
    figures(summary, offsets, stages)
    return summary, pairs, oracle_rows


def figures(summary, offsets, stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods = ['WLP', 'ConstantTurn', 'VAR', 'KalmanFrozen', 'OldKalman', 'HoldLast', 'V14']
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, (label, lo, hi) in zip(axs, WINDOWS):
        values = [next(r['macro_mean'] for r in summary if r['method'] == m and r['window'] == label and r['metric'] == 'l2_mm') for m in methods]
        ax.bar(methods, values); ax.set_title('Offsets '+label.replace('_', '-')); ax.tick_params(axis='x', rotation=55)
        ax.set_ylabel('Control-target proxy L2 (mm)'); ax.grid(axis='y', alpha=.2)
    fig.tight_layout(); fig.savefig(OUT/'window_comparison.png', dpi=160); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for ax, field in zip(axs, ('l2_mm', 'along_mm', 'lateral_mm')):
        for method in methods:
            rs = [r for r in offsets if r['method'] == method and r['metric'] == field]
            ax.plot([r['offset'] for r in rs], [r['macro_mean'] for r in rs], label=method)
        ax.set_xlabel('Prediction offset (20Hz)'); ax.set_ylabel(field+' (proxy)'); ax.axvline(5, color='grey', ls=':'); ax.grid(alpha=.2)
    axs[-1].legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT/'offset_curves.png', dpi=160); plt.close(fig)
    phases = sorted({r['phase'] for r in stages})
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(phases)); width=.11
    for j, method in enumerate(methods):
        values = [next(r['macro_mean'] for r in stages if r['method'] == method and r['phase'] == p and r['window'] == '1_5' and r['metric'] == 'l2_mm') for p in phases]
        ax.bar(x+(j-3)*width, values, width, label=method)
    ax.set_xticks(x, phases, rotation=15); ax.set_ylabel('First5 L2 control-target proxy (mm)'); ax.legend(fontsize=7, ncol=4)
    fig.tight_layout(); fig.savefig(OUT/'stage_first5.png', dpi=160); plt.close(fig)


def report(summary, pairs, winners, source_receipt, elapsed, oracle_rows):
    methods = ['WLP', 'ConstantTurn', 'VAR', 'KalmanFrozen', 'OldKalman', 'HoldLast', 'V14']
    lookup = {(r['method'], r['window']): r['macro_mean'] for r in summary if r['metric'] == 'l2_mm'}
    text = ['# v3.2 E3：历史命令外推 CPU 实验', '',
            '这些误差是平移控制目标代理（normalized command × 50 mm），不是机械臂实际末端 EEF 误差，也不代表闭环任务成功率。', '',
            '沿用 v2.1 同一批 Full 轨迹、同观测 V14：task0 的 20 episode / 634 重规划点用于校准；task1–9 的 180 episode / 6345 点用于评估。所有预测器只读取当前时间之前已执行的 XYZ 命令；Full/V14 未来 chunk 仅作评分标签。首次无历史返回零，短历史回退 HoldLast。', '',
            f'共 {len(candidates())} 个预先冻结候选，按五个 official init 分折、优先首5步宏平均，要求每折首5步 L2 p95 和累计代理 lateral p95 均不超过旧 Kalman 的105%。每类保留最佳候选；未通过者明确标为探索候选。没有用评估集选参。总耗时 {elapsed:.1f}s；环境步数和模型调用均为0。', '',
            '| 方法 | 1–5步 mm（主指标） | 6–12步 mm | 13–50步 mm | 校准尾部准入 |',
            '|---|---:|---:|---:|---|']
    for method in methods:
        eligible = ('通过' if winners[method]['eligible'] else '未通过，仅探索') if method in winners else '冻结参考'
        text.append(f'| {method} | {lookup[method,"1_5"]:.3f} | {lookup[method,"6_12"]:.3f} | {lookup[method,"13_50"]:.3f} | {eligible} |')
    text += ['', '宏平均按重规划→episode→task→suite逐层等权；首5步按真实剩余可执行前缀截断。6–50步是当前Full chunk的诊断，不能当作未来真实执行轨迹。', '',
             '| 新方法 vs 冻结v2.1 Kalman | 首5步 Δmm | 任务簇配对bootstrap 95%区间 |', '|---|---:|---|']
    for method in ('WLP', 'ConstantTurn', 'VAR'):
        r = next(r for r in pairs if r['method'] == method and r['baseline'] == 'KalmanFrozen' and r['window'] == '1_5')
        text.append(f'| {method} | {r["delta_mm"]:+.3f} | [{r["ci95_low_mm"]:+.3f}, {r["ci95_high_mm"]:+.3f}] |')
    text += ['', '区间按suite分层，重采样task并整簇保留init与replan，2000次，seed=3203；负数表示新方法误差更低。复用过往已分析数据，非盲测；本轮候选族与最优候选比较没有多重检验校正。', '',
             '## 模型含义与边界', '',
             '- WLP：近期命令序列加权多项式，最后观测锚定，阻尼外推。',
             '- ConstantTurn：用近期命令方向的3D Kabsch旋转估计转向，保持最后命令幅值并阻尼角速度。这里旋转的是命令向量，不是将EEF历史位置作圆弧拟合；接触动力学与真实机器人运动未建模。',
             '- VAR：近期命令向量短阶线性自回归，ridge向持久性先验收缩，并限制不稳定局部拟合。模型无全局未来标签训练。',
             '- 保留冻结 Kalman 的 q_u=.0001、q_w=100、g=.5、beta=1、q_full 及原旧 Kalman 参数，均未重新调参。保存新预测器已观测的一步创新与冻结 Kalman 原始创新。',
             '- along/lateral 是累计控制目标代理误差沿Full命令方向分解；低幅值与反向切换方向无定义时剔除。阶段标签沿用来源规则，仅作事后分组，未经独立人工认证。',
             '- 时间偏移是事后oracle：固定Full第4–9步，预测位移−3..+3；不进入任何校正或预测。', '',
             '## 数据与产物', '',
             f'本轮重新校验所有实际消费的 {source_receipt["consumed_file_count"]} 个预测/推理/telemetry日志SHA256，并检查原始Full、可执行前缀、时钟、ID、noise身份。图像与noise NPZ不进入预测器，继承v2.1完整验证收据，未重新读取。source_receipt.json中保留全部当前消费文件哈希及历史完整验证收据SHA256。', '',
             'summary.csv / offsets.csv / stages.csv：窗口、逐offset、阶段的L2与along/lateral宏均值；paired_bootstrap.csv：与Kalman/HoldLast/V14配对区间；record_windows.csv：重规划明细；time_shift_oracle.csv：时间偏移诊断；predictions.npz、records.json：全部50步预测、元数据与创新；frozen_winners.json：评估前冻结参数。', '',
             '![窗口比较](window_comparison.png)', '', '![逐offset误差与方向分解](offset_curves.png)', '', '![首5步阶段误差](stage_first5.png)', '']
    (OUT/'report_zh.md').write_text('\n'.join(text))


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = dict(candidate_count=len(candidates()), candidates=[asdict(c) for c in candidates()],
                    calibration='task0, all four suites, init0..4', evaluation='task1..9, all suites, init0..4',
                    selection='5-init-fold mean first5 macro; each fold L2/lateral p95 <= oldKalman*1.05; lexical tie break',
                    guard='Missing eligible family winner stays ineligible exploratory, never selected on eval',
                    max_wall_time_s=5400, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    code_hashes={str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob('predictor*.py')})
    save(OUT/'manifest.json', manifest)
    print('Validating consumed original source logs against v2.1 receipt', flush=True)
    groups, source_receipt = load_sources()
    save(OUT/'source_receipt.json', source_receipt)
    calibration = {k: v for k, v in groups.items() if k[1] == 0}
    winners = tune(calibration, source_receipt['old_parameters'])
    print('Winners frozen; loading original frozen baseline arrays', flush=True)
    status = json.loads((OLD/'evaluation/status.json').read_text())
    for name, key in [('predictions.npz', 'prediction_sha256'), ('records.json', 'records_sha256')]:
        if sha(OLD/'evaluation'/name) != status[key]:
            raise ValueError('Frozen baseline array hash changed')
    oldmeta = json.loads((OLD/'evaluation/records.json').read_text())
    records = [r for k, rs in sorted(groups.items()) if k[1] != 0 for r in rs]
    meta = metadata(records)
    with np.load(OLD/'evaluation/predictions.npz', allow_pickle=False) as archive:
        arrays = {name: archive[key].copy() for name, key in [('Full', 'full'), ('V14', 'v14'), ('HoldLast', 'hold_last'), ('KalmanFrozen', 'clipped_mean'), ('OldKalman', 'old_kalman')]}
        innovations = dict(KalmanFrozen=archive['innovation'].copy())
    for i, (r, m, oldm) in enumerate(zip(records, meta, oldmeta)):
        if any(m[k] != oldm[k] for k in ('suite', 'task', 'init', 'replan', 'step', 'valid_executed_prefix', 'input_sha256', 'noise_sha256')):
            raise ValueError('Frozen evaluation record identity differs')
        if not np.array_equal(arrays['Full'][i], np.asarray(r['full_actions'])[:, :3]) or not np.array_equal(arrays['V14'][i], np.asarray(r['variant_actions'])[:, :3]):
            raise ValueError('Stored baseline does not match same source commands')
    save(OUT/'baseline_receipt.json', dict(files={str(p): sha(p) for p in [OLD/'winner.json', OLD/'evaluation/predictions.npz', OLD/'evaluation/records.json', OLD/'evaluation/status.json']},
                                         frozen_winner=json.loads((OLD/'winner.json').read_text()),
                                         exact_full_v14_array_identity=True))
    for method, winner in winners.items():
        c = Config(**winner['config'])
        arrays[method] = np.asarray([predict_history(r['_actions'], int(r['environment_action_step']), c) for r in records])
        innovation = []
        for r in records:
            t = int(r['environment_action_step'])
            innovation.append(np.clip(r['_actions'][t-1, :3], -1, 1)-predict_history(r['_actions'], t-1, c, 1)[0] if t else np.full(3, np.nan))
        innovations[method] = np.asarray(innovation)
        print(f'Evaluation {method}: {len(records)} x50 predictions complete', flush=True)
    summary, pairs, oracles = analyze(arrays, records, innovations)
    elapsed = time.monotonic()-started
    report(summary, pairs, winners, source_receipt, elapsed, oracles)
    save(OUT/'completion.json', dict(status='complete', elapsed_s=elapsed, model_calls=0, environment_actions=0,
                                   evaluation_points=len(records), horizon=50, calibration_points=634,
                                   outputs={p.name: sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(f'COMPLETE {elapsed:.1f}s {OUT}', flush=True)


if __name__ == '__main__':
    main()
