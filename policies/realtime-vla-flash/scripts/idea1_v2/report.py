"""Chinese result narrative generated only from available measured tables."""
import csv
import json

def read(root, name):
    path=root/'analysis'/name
    if not path.exists():return []
    with path.open() as stream:return list(csv.DictReader(stream))

def run(root):
    success=read(root,'success.csv');offsets=read(root,'kalman_offsets.csv')
    lines=['# idea1 v2 实验汇总','', '## 1. E2：周期交替的成功率与成本','']
    if success:
        lines+=['下表使用真实闭环结果；不从预测误差推算成功率。','',
                '| 条件 | suite | 成功/样本 | 对照 | 对照成功 | 差值 pp |',
                '| --- | --- | --- | --- | --- | --- |']
        for r in success:lines.append('| {condition} | {suite} | {successes}/{denominator} | {baseline} | {baseline_successes} | {difference_pp} |'.format(**r))
        lines+=['','任务聚类、suite分层配对bootstrap：[区间表](analysis/success_bootstrap.csv)。',
                '真实请求成本、Full调用比例：[耗时表](analysis/latency.csv)。不能用理论1/3比例代替实际调用比例。']
    else:lines+=['E2尚无有效闭环结果，不能给出成功率或加速结论。']
    lines+=['','## 2. E4：历史外推与模型预测的差异','']
    if offsets:
        lines+=['比较对象是同起点的平移控制目标增量，单位毫米；它不是末端实际走出的距离。',
                'Kalman只更新已执行历史；参数只在校准task0上选择，评估task1–9独立汇总。',
                '', '| 比较 | offset 1 (mm) | offset 5 (mm) | offset 12 (mm) | offset 50 (mm) |',
                '| --- | --- | --- | --- | --- |']
        for pair in ('Kalman-Full','HoldLast-Full','V14-Full','Kalman-V14'):
            values={int(r['offset']):float(r['equal_mean_mm']) for r in offsets if r['split']=='evaluation' and r['pair']==pair}
            lines.append('| %s | %s |'%(pair,' | '.join('%.4f'%values[h] if h in values else '缺失' for h in (1,5,12,50))))
        lines+=['','![未来1–50步](figures/kalman_offsets_50.png)','',
                '![未来1–12步](figures/kalman_offsets_12.png)','',
                '[前缀统计](analysis/kalman_prefixes.csv) · [阶段及冷启动](analysis/kalman_strata.csv) · [分歧箱](analysis/divergence_bins.csv)。',
                'offset6–50比较的是本次Full预测块，不是后续重规划后的真实执行路线；误差小不能证明安全或动作可替换。']
    else:lines+=['E4尚未产出完整统计，不给出预测结论。']
    lines+=['','## 3. E3：阶段路由','']
    deferred=root/'E3_deferred.json'
    phase=[r for r in success if r['condition'].startswith('PHASE_')]
    if phase:lines+=['阶段路由的真实成功率见第1节，Full占比见耗时表；全程Full的任务不能解释为变体验证成功。']
    elif deferred.exists():lines+=['依据预先登记的E2/E4门槛暂缓，原因：`%s`。'%json.dumps(json.loads(deferred.read_text()),ensure_ascii=False)]
    else:lines+=['阶段路由尚未完成；E2须200条有效且≥80%成功，并完成E4后才启动。']
    lines+=['','## 4. 诊断、边界与后置项目','',
            '[真实XYZ案例](media/cases.json) · [完整视频与图片入口](media/viewer.md) · [HTML画廊](media/gallery.html)。',
            '[阶段计数](analysis/phase_counts.csv) · [失败候选事件](analysis/failure_events.csv) · [事件±25步片段](media/event_clips.json)。',
            '最远阶段不等于失败起点；同step XYZ持续偏离和抓取丢失是描述性候选事件，正常释放也可能产生抓取丢失。',
            'Full路线上的Shadow不等于V14真实闭环。成功率改善不足以证明Full消除了此前累计误差；接管附近变化也不证明恢复因果。',
            '90%是探索目标。本轮未训练、未搜索新mask、未改变Action Expert，也未验证在线Kalman门控、额外比例、V15、恢复分支或Xiaomi。',
            '完整性与未完成项以[验收](completion_audit.json)和[缺项](missing_or_deferred.json)为准；MPE实际插件播放需要编辑器人工验收。','']
    # This is a derived report; regenerating it does not overwrite raw experiment data.
    (root/'report.md').write_text('\n'.join(lines))
    return {'report':'report.md'}
