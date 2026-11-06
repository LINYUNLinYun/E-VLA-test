"""Render a bounded CPU-only waiting report from a fixed draft/analyze snapshot.

--capture saves auxiliary metadata once. Normal execution only reads evidence.json.
No runtime imports, resampling, broad hashing, source edits, queue or GPU operations.
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path('/root/nfs/embodied/benchmark')
OUT = Path(__file__).resolve().parent
R = ROOT / 'results/ease_insights_20261103'
REPORTS = ROOT / 'reports/ease_insights_20261103'
SNAP = OUT / 'draft_analysis_snapshot'
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
LABEL = dict(zip(SUITES, ['Spatial', 'Object', 'Goal', 'Long']))
CONDITIONS = ['full_d10', 'fixed_v12_jacobian', 'fixed_v12_action_mse', 'fixed_v12_cosine']
COST_INTERPRETATION = ('实测成本并不随接受率单调变化，匹配来源也并非每个目标里最便宜：固定 Object 目标，匹配来源为23.12ms/action，Goal来源为21.02ms/action，双方均10/10成功，但接受率分别为61.37%与0%。这些发现支持 draft 来源依赖与 Full 回退掩盖质量损失；尚不能把实测成本差全部因果归于泛化下降，也不能声称原本存在的加速收益被跨suite切换消除，因为匹配FLASH条件本身也慢于同目标Full。冷启动、任务轨迹和同卡争用等仍限制解释，现有数据未识别具体主因。')
METHOD_BOUNDARY = ('当前训练/serving代码共享 base assets/physical-intelligence/libero 的归一化合同，未发现source专属变换或实际mismatch；但发布head缺少原训练checkpoint、norm hash及config快照，原cache也缺失，因此本轮只能说明固定共同推理合同下的发布head迁移表现，不能称已完全排除历史预处理混淆。teacher_zero_noise仅指训练缓存，官方online默认同样使用标准正态噪声；本轮显式NumPy case/replan配对高斯并非新增zero-vs-random错配，但RNG及CPU→GPU路径不同，不能称官方默认数值或时延的精确复现。policy_wall_ms只包围native original_infer，CPU种子生成和输入hash在计时外；完整client RPC包含wrapper与传输，故分别标为native-policy成本与完整RPC成本，指标值不变。')


def read(p):
    return json.loads(Path(p).read_text())


def link(name, p):
    return f'[{name}]({p})'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in rows])


def capture():
    if (OUT / 'evidence.json').exists():
        raise SystemExit('Keep this historical snapshot; capture a new report in a new directory.')
    episodes = read(SNAP / 'episodes.json')
    audit = read(SNAP / 'audit.json')
    vlm = read(R / 'vlm/analysis/analysis.json')
    selection = read(R / 'vlm/analysis/selection.json')
    conf = read(R / 'vlm_confirmation/manifest.json')
    state = read(R / 'pipeline/state.json')
    supplementary = read(R / 'vlm_confirmation/supplementary_goal_contrast.json')
    complete = {(x['mode'], x['source_suite'], x['target_suite'], x['task_id'], x['init_id']) for x in episodes}
    expected = {(mode, source, target, task, 20) for mode in ['flash', 'draft']
                for source in SUITES for target in SUITES for task in range(10)}
    expected |= {('full', 'libero_spatial', target, task, 20) for target in SUITES for task in range(10)}
    assert len(complete) == len(episodes) and complete <= expected
    processes = []
    for pid, name in [(246695, 'scheduler'), (255641, 'pipeline'), (257177, 'observer')]:
        p = Path(f'/proc/{pid}/status')
        processes.append(dict(pid=pid, name=name, present=p.exists(),
                              status_lines=[line for line in p.read_text().splitlines() if line.startswith(('Name:', 'State:', 'PPid:'))] if p.exists() else []))
    long = [r for r in episodes if r['mode'] == 'draft' and r['source_suite'] == 'libero_10']
    receipts = sorted(str(p) for p in (R / 'vlm_confirmation').rglob('*receipt*.json'))
    e = dict(captured_utc=datetime.now(timezone.utc).isoformat(),
             draft_snapshot_utc=datetime.fromtimestamp((SNAP / 'audit.json').stat().st_mtime, timezone.utc).isoformat(),
             draft_expected=len(expected), draft_complete=len(episodes),
             draft_missing=[dict(zip(['mode', 'source_suite', 'target_suite', 'task_id', 'init_id'], x)) for x in sorted(expected-complete)],
             draft_summary=read(SNAP / 'summary.json'), draft_speed_vs_full=read(SNAP / 'speed_vs_full.json'),
             draft_audit=audit, long_pure_completed=len(long), long_pure_requests=sum(r['requests'] for r in long),
             long_pure_canonical_run_result_exists=(R / 'draft_discovery/draft_libero_10/result.json').exists(),
             vlm_discovery_complete=vlm['valid_episodes'],
             vlm_summaries=[x for x in vlm['condition_summaries'] if x['level'] in ['all', 'suite']],
             vlm_fixed_input_cost={k: vlm['fixed_input_cost'][k] for k in ['valid_services', 'mean_cycle_ms', 'full_over_policy_cycle', 'scope']},
             vlm_frozen_selection=selection['selection']['selected'],
             vlm_confirmation_expected=conf['total_confirmation_episodes'],
             vlm_confirmation_receipt_count=len(receipts), vlm_confirmation_receipt_paths=receipts,
             vlm_supplementary_goal=supplementary,
             pipeline_phases={k: v.get('phase') for k, v in state['branches'].items()},
             processes=processes,
             sources=dict(draft_roots=[str(R / 'draft_smoke/flash_spatial'), str(R / 'draft_smoke/draft_spatial'), str(R / 'draft_discovery')],
                          draft_analyzer=str(ROOT / 'preparations/ease_insights_20261103/draft/analyze.py'),
                          vlm_analysis=str(R / 'vlm/analysis/analysis.json'),
                          original_selection=str(R / 'vlm/analysis/selection.json'),
                          goal_supplement=str(R / 'vlm_confirmation/supplementary_goal_contrast.json'),
                          perturbation=str(REPORTS / 'perturbation_context/README_zh.md'),
                          target_identity=str(REPORTS / 'target_identity/audit_20261103T132858Z.json'),
                          materiality=str(REPORTS / 'target_identity/materiality.md'),
                          source_normalization=str(REPORTS / 'source_normalization_audit/README_20261104.md'),
                          vision_negative=str(REPORTS / 'vision_pilot/scientific_verdict.md')),
             reporting_bounds=['All independent confirmation outcomes remain pending or incomplete; do not reselect from interim outcomes.',
                               'Complete target-fixed comparisons only; Long matched pure 2/6 is incomplete, not ranked against 10-case conditions.',
                               'First-input/noise metadata matches do not imply all first target actions passed tolerance.',
                               'Policy and RPC cost numerators include zero-action rejection requests; not steady-state exclusive-GPU speed.',
                               COST_INTERPRETATION,
                               METHOD_BOUNDARY])
    (OUT / 'evidence.json').write_text(json.dumps(e, ensure_ascii=False, indent=2) + '\n')


def render():
    e = read(OUT / 'evidence.json')
    summary = e['draft_summary']
    idx = {(x['mode'], x['source_suite'], x['target_suite']): x for x in summary}
    row = lambda mode, source, target: idx.get((mode, source, target))
    full = lambda target: row('full', 'libero_spatial', target)
    score = lambda x: '缺失' if x is None else f"{x['successes']}/{x['n']}" + (' **未完成**' if x['n'] != 10 else '')
    pct = lambda x: f'{100*x:.2f}%'
    stamp = datetime.fromisoformat(e['draft_snapshot_utc']).astimezone(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S 北京时间')
    assert sum(x['n'] for x in summary) == e['draft_complete']
    assert e['draft_complete'] + len(e['draft_missing']) == e['draft_expected']
    flash = [row('flash', s, t) for t in SUITES for s in SUITES]
    assert len(flash) == 16 and all(x and x['n'] == 10 for x in flash)
    missing = '、'.join(f"{LABEL[x['target_suite']]} task{x['task_id']:02d}/init{x['init_id']}" for x in e['draft_missing'])
    vs = {(x['suite'], x['condition']): x for x in e['vlm_summaries'] if x['level'] == 'suite'}
    va = {x['condition']: x for x in e['vlm_summaries'] if x['level'] == 'all'}
    speed = [x for x in e['draft_speed_vs_full'] if x['mode'] == 'flash' and x['n_paired'] == 10]
    parts = [
        '# EASE 等待 GPU 期间的中期证据更新',
        f'**仍未完成，不作独立验证结论。** Draft 固定截面为 **{e["draft_complete"]}/{e["draft_expected"]}**（{stamp}）；VLM 发现 **{e["vlm_discovery_complete"]}/160**，确认结果 receipt **{e["vlm_confirmation_receipt_count"]}/{e["vlm_confirmation_expected"]}**（辅助状态截取 {e["captured_utc"]}）。本次只做 CPU 汇总；原 21:21 快照和科学数据均保留。',
        f'缺失仅为 **Long 来源纯 Draft → {missing}**。该 run 的 {e["long_pure_completed"]} 条有效 case、{e["long_pure_requests"]} 次请求已保留；canonical run result 当前不存在属于恢复状态，不能把保留结果当作失效或把缺失当策略失败。pipeline 仍处于发现/确认等待阶段；scheduler246695、pipeline255641、observer257177 在辅助截取时存活。目标继续 active，按既定安排等可用 GPU 接续，不标为 blocked。',
        '**方法边界。** ' + METHOD_BOUNDARY + ' 见' + link('来源归一化与噪声审计', REPORTS / 'source_normalization_audit/README_20261104.md') + '。',
        '**纯 Draft：同目标比较支持发现集来源依赖，Long 匹配格仍不完整。** 每行固定目标 suite/task/init；四个来源只改变 draft checkpoint，纯 Draft 无 verify，接受率与 Full fallback 为 N/A。',
        table(['目标', 'Full', 'Spatial 纯 Draft', 'Object 纯 Draft', 'Goal 纯 Draft', 'Long 纯 Draft'],
              [[LABEL[t], score(full(t))] + [score(row('draft', s, t)) for s in SUITES] for t in SUITES]),
        'Spatial/Object/Goal 匹配来源为 9/10、10/10、8/10，而同一目标的其余三来源均为 0/10。Long 匹配来源只有 task00–05 的 2/6，**不与完整 10-case 格排名、不冻结候选，也不把缺失4例补成失败**。发现结束后仍须按既定规则冻结，再做独立 init21–24 确认。',
        '**FLASH：16格全部完整。** 接受率 = 接受前缀动作数 /（12×verify请求数）；“拒绝Full” = 拒绝触发的 Full 请求 / 全部请求；“全部Full”还包括初始化和周期刷新。native-policy与完整client RPC成本分别累计各自调用耗时并除以实际执行动作数，均包含执行零动作的拒绝请求。',
        table(['目标', 'Draft 来源', '成功/n', '动作接受率', '拒绝Full', '全部Full', 'native-policy ms/action', '完整client RPC ms/action'],
              [[LABEL[t], LABEL[s] + ('（匹配）' if s == t else ''), score(row('flash', s, t)),
                pct(row('flash', s, t)['acceptance_rate']), pct(row('flash', s, t)['rejection_full_fraction']),
                pct(row('flash', s, t)['full_round_fraction']), f"{row('flash', s, t)['policy_ms_per_executed_action']:.2f}",
                f"{row('flash', s, t)['client_rpc_ms_per_executed_action']:.2f}"] for t in SUITES for s in SUITES]),
        '固定 Object 目标的 Spatial/Goal 来源接受率都是 0%，FLASH 仍各10/10；两者各108次 verify 全部接受零动作并触发 Full。匹配来源接受率61.37%、成功10/10。这支持“FLASH 最终成功可以掩盖 draft 自身质量损失”，不代表异来源必然降低最终成功，也不构成禁用回退的单独因果消融。',
        '**实际成本仍有限制。** 以下 Full 与各格使用同分支、同目标 case；完整速度比和双方成功数见 ' + link('speed_vs_full.json', SNAP / 'speed_vs_full.json') + '。',
        table(['目标', 'Full 成功/n', 'Full native-policy ms/action', 'Full 完整client RPC ms/action', 'Full episode墙钟均值(s)'],
              [[LABEL[t], score(full(t)), f"{full(t)['policy_ms_per_executed_action']:.2f}",
                f"{full(t)['client_rpc_ms_per_executed_action']:.2f}", f"{full(t)['mean_episode_wall_seconds']:.2f}"] for t in SUITES]),
        f'16个完整FLASH格中，**{sum(x["policy_per_executed_action_speedup_vs_full"] < 1 for x in speed)}/16** 的 native-policy ms/action 高于 Full，**{sum(x["paired_episode_wall_speedup"] < 1 for x in speed)}/16** 的累计 episode墙钟更长；不能宣称 FLASH 已获得实际加速。成本保留全部请求，不以首请求删除或低延迟筛选制造热态；服务准备与潜在 lazy compile/capture、外部GPU争用仍限制解释。请求均值和GPU-stage时间不等于每执行动作成本或端到端完成时间。',
        COST_INTERPRETATION,
        '**VLM：原 primary 方向不变，Goal 为确认前登记的补充。**',
        table(['目标', 'Full', 'Jacobian', 'MSE', 'Cosine'],
              [[LABEL[t]] + [f"{vs[(t,c)]['successes']}/10" for c in CONDITIONS] for t in SUITES] +
              [['合计'] + [f"{va[c]['successes']}/40" for c in CONDITIONS]]),
        '原 primary 保持 Spatial: Jacobian>MSE +10pp、Long: MSE>Jacobian +10pp；这两项仍只是弱发现信号。Goal: Jacobian>MSE +30pp 在发现后、任何独立确认观测前登记为补充，利用原600例矩阵内的Goal案例；必须同时报告三项及原primary失败情形，按补充协议报告三项Holm结果，不以Goal替换Spatial、不依据确认中途结果改方向。见 ' + link('冻结选择', R / 'vlm/analysis/selection.json') + '、' + link('Goal补充协议', R / 'vlm_confirmation/supplementary_goal_contrast.json') + '。',
        'VLM三个候选均保留12/18层、周期Full-D10→Fast-D2→Fast-D2、H50/R5及同残差外推。5服务同输入完整policy周期成本为Full326.03ms、Jacobian202.87ms、MSE201.76ms、Cosine201.58ms，Full/候选约1.607/1.616/1.617×；范围不含RPC和环境，也不证明质量非劣或闭环同比加速。Draft/FLASH为H50/R12，不能混用这两个分支的Full分母。',
        '**补充上下文及反证保留。**',
        link('历史扰动探索', REPORTS / 'perturbation_context/README_zh.md') + '在已验收175例中为V14 89/175、匹配纯draft54/175，配对差+20.0pp（Full138/175）。这是已看过的描述性历史结果，不是本轮确认；不能证明无扰动→有扰动的因果降幅，也不能外推所有trained draft或完整FLASH。V14是D10/H50/R5固定路径，与本轮V12周期路径和纯draft并非同预算比较。本报告只引用既有统计，没有重采样。',
        link('首次Full身份审计', REPORTS / 'target_identity/README.md') + '覆盖160对：输入/noise/路由及base元数据一致，159对raw/decoded动作前缀精确一致；Spatial来源→Spatial task1/init20一对超过1e-5容差，raw最大差0.0078125、decoded最大差0.00390387135，各28/84元素超限。200条输出独立解码重建全部通过（最大误差5.55e-17），支持共享归一化，但不能声称所有首次Full动作数值一致。' + link('单例边界与敏感性', REPORTS / 'target_identity/materiality.md') + '保留主分析该例，原因未知；0步夹爪符号变化和双方最终成功都不能证明差异无因果影响。',
        link('Vision负结果', REPORTS / 'vision_pilot/scientific_verdict.md') + '保持Full39/40、A36/40、B38/40；Goal的预登记A>B没有复现，两个不一致case均B-only，未支持这组可靠互补性。未把该结果改写为VLM或在线selector的确认。',
        '**文件与复现。** ' + link('evidence.json', OUT / 'evidence.json') + '保存缺失case、16格指标、固定选择、补充协议和状态；' + link('分析快照', SNAP / 'audit.json') + '合并真实两处smoke与discovery，条件无重复、首输入/noise无不一致，但动作数值一致性以单例审计边界为准。' + link('生成脚本', OUT / 'build_report.py') + '只读保存快照重建；重新采集须使用新目录，不能覆盖本截面或旧interim/。',
        '```bash\npython3 /root/nfs/embodied/benchmark/reports/ease_insights_20261103/interim_20261104_waiting/build_report.py\n```',
    ]
    path = OUT / 'interim_evidence_report_zh.md'
    path.write_text('\n\n'.join(parts) + '\n')
    print(json.dumps(dict(report=str(path), draft=e['draft_complete'], missing=len(e['draft_missing']),
                         confirmation=e['vlm_confirmation_receipt_count']), ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', action='store_true')
    args = p.parse_args()
    if args.capture:
        capture()
    render()
