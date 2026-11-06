import csv,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3_4_selection_async.common import OUT,ROOT,sha
from reports.idea1_v3_4_selection_async_analysis import analyze,profiling


def rows(path):return list(csv.DictReader(Path(path).open()))


def table(data,columns):
    return '\n'.join(['| '+' | '.join(title for key,title in columns)+' |',
        '| '+' | '.join('---' for _ in columns)+' |']+
        ['| '+' | '.join(str(row.get(key,'')) for key,title in columns)+' |' for row in data])


def render():
    a=json.loads((OUT/'analysis.json').read_text());assert a['completed']==a['target'] and a['prefix_passed']
    assert (OUT/'followup_complete.json').exists()
    compression=json.loads((OUT/'compression/complete.json').read_text())['summary']
    profile=json.loads((OUT/'profiling_analysis.json').read_text()) if (OUT/'profiling_analysis.json').exists() else profiling()
    paced=json.loads((OUT/'paced_analysis.json').read_text())
    diagnostic=json.loads((OUT/'diagnosis/complete.json').read_text())
    timing=json.loads((OUT/'timing_diagnosis.json').read_text())
    projections=json.loads((OUT/'projection_blindspots.json').read_text())
    fusion=json.loads((OUT/'fusion_analysis.json').read_text())['summary']
    waits=json.loads((OUT/'paced_wait_breakdown.json').read_text())['summary']
    by={r['condition']:r for r in a['summary']};pairs={r['condition']:r for r in a['paired']}
    costs=[]
    fusion_table=[dict(method=r['method'],mean=f"{r['mean_ms']:.2f}",p95=f"{r['p95_ms']:.2f}",
        max=f"{r['max_ms']:.2f}",raw=f"{r['raw_passes']}/8",gradient=f"{r['gradient_passes']}/8",
        layers=f"{r['layer_score_passes']}/8",mask=f"{8-r['v12_mask_changes']}/8",
        capture=f"{r['capture_mean_s']:.2f}",allocated=f"{r['peak_allocated_mib']:.0f}",
        reserved=f"{r['peak_reserved_mib']:.0f}") for r in fusion]
    preparation=rows(OUT/'compression/preparation.csv');prep_table=[]
    for name,value in compression.items():
        costs.append(dict(method=name,n=value['n'],mean=f"{value['mean_ms']:.2f}",p95=f"{value['p95_ms']:.2f}",
            passes=f"{value['numerical_passes']}/8",mask_changes=value['mask_changes']))
        pp=[r for r in preparation if r['method']==name]
        prep_table.append(dict(method=name,capture=f"{np.mean([float(r['capture_s']) for r in pp]):.2f}",
            allocated=f"{max(float(r['allocated_mib']) for r in pp):.0f}",
            reserved=f"{max(float(r['reserved_mib']) for r in pp):.0f}"))
    overlap=[]
    for row in rows(OUT/'overlap/summary.csv'):
        if row['priority']!='high' or row['execution_ms'] not in ('0','100','250'):continue
        overlap.append(dict(q=row['q'],execution=row['execution_ms'],
            ready=f"{float(row['action_ready_ms']):.2f}",critical=f"{float(row['critical_compute_ms']):.2f}",
            wait=f"{float(row['selector_wait_ms']):.2f}",hidden=f"{row['fully_hidden']}/{row['n']}"))
    quality=[]
    throughput=[]
    for row in a['summary']:
        pair=pairs.get(row['condition'],{})
        quality.append(dict(condition=row['condition'],success=f"{row['success']}/{row['n']}",
            changes=f"+{pair.get('recovered',0)}/−{pair.get('new_failures',0)}",
            warm=f"{row['warm_ms_per_action']:.2f}",tail=f"{row['warm_tail_ms_per_action']:.2f}",
            speed=f"{pair.get('tail_policy_speedup_geomean',1.):.3f}",wall=f"{pair.get('warm_wall_speedup_geomean',1.):.3f}",
            full=f"{row['full_fraction']*100:.1f}%",wait=f"{row['mean_selector_wait_ms']:.2f}",
            capture=f"{row['capture_s']:.1f}"))
        throughput.append(dict(condition=row['condition'],n=row['n'],success=row['success'],
            wall=f"{row['wall_s']:.1f}",prep=f"{row['capture_s']:.1f}",
            episodes=f"{row['episodes_per_hour']:.1f}",actions=f"{row['actions_per_second']:.2f}"))
    scores=[]
    for period in (3,5):
        for q in (1,2):
            old=by[f'v12_p{period}_q{q}_layer'];new=by[f'v12_p{period}_q{q}_combo']
            scores.append(dict(period=period,q=q,layer=f"{old['success']}/{old['n']}",
                combo=f"{new['success']}/{new['n']}",layer_ms=f"{old['mean_selector_score_ms']:.4f}",
                combo_ms=f"{new['mean_selector_score_ms']:.4f}"))
    refresh=[];requests=rows(OUT/'requests.csv')
    for condition in ('v12_p3_q4_serial_async','v12_p3_q4_stride2','v12_p3_q4_stride3'):
        row=by[condition];rr=[r for r in requests if r['condition']==condition]
        trigger=[float(r['warm_policy_ms']) for r in rr if r['refresh']=='True']
        fast=[float(r['warm_policy_ms']) for r in rr if r['path']=='fast_verified']
        refresh.append(dict(condition=condition,success=f"{row['success']}/{row['n']}",
            fulls=row['fulls'],selectors=row['selectors'],vjp=4*row['selectors'],
            amortized=f"{row['warm_ms_per_action']:.2f}",trigger=f"{np.mean(trigger):.2f}",
            trigger95=f"{np.percentile(trigger,95):.2f}",fast95=f"{np.percentile(fast,95):.2f}" if fast else ''))
    paced_table=[]
    for row in paced['paired']:
        summary=next(s for s in paced['summary'] if s['condition']==row['condition'])
        paced_table.append(dict(condition=row['condition'],success=f"{summary['success']}/{summary['n']}",
            n=row['n'],
            compute=f"{row['tail_compute_speedup_geomean']:.3f}",pooled=f"{row['tail_compute_speedup_pooled']:.3f}",
            wall=f"{row['warm_wall_per_action_speedup_geomean']:.3f}",episode=f"{row['warm_episode_wall_speedup_geomean']:.3f}",
            hidden=f"{row['selector_ready']}/{row['selectors']}",wait=f"{row['mean_selector_wait_ms']:.3f}"))
    diagnostics=[]
    wait_table=[dict(condition=r['condition'],ready=f"{r['before_fast_ready']}/{r['before_fast_jobs']}",
        mean=f"{r['before_fast_wait_mean_ms']:.4f}",p95=f"{r['before_fast_wait_p95_ms']:.4f}",
        end=f"{r['terminal_gpu_wait_total_ms']:.2f}",span=f"{r['selector_concurrent_span_mean_ms']:.2f}",
        ae=f"{r['native_full_ae_mean_ms']:.2f}",execution=f"{r['actual_execution_ms_per_action']:.2f}") for r in waits]
    projection_table=[dict(condition=r['condition'],n=r['n'],own=f"{r['own_signed_rms_mean']:.6f}",
        heldout=f"{r['heldout_signed_rms_mean']:.6f}",full=f"{r['full_four_signed_rms_mean']:.6f}") for r in projections['summary']]
    timing_table=[]
    for row in timing['summary']:
        timing_table.append(dict(condition=row['condition'],full=f"{row['full_fraction']*100:.1f}%",
            fallback=f"{row['fallback_rate_per_draft']*100:.1f}%",draft=f"{row['draft_mean_ms']:.2f}",verify=f"{row['verify_mean_ms']:.2f}",
            speed=f"{row['observed_pooled_speedup']:.3f}",oracle=f"{row['zero_exposed_selector_speedup']:.3f}",
            wait=f"{row['exposed_selector_wait_ms_per_action']:.2f}",waste=f"{row['wasted_rejected_proposal_ms_per_action']:.2f}"))
    for row in diagnostic['summary']:
        if row['scope']!='fixed_early_bank':continue
        diagnostics.append(dict(condition=row['condition'],n=row['n'],
            current=f"{row['mean_true_error']:.4f}",d2=f"{row['mean_d2_only_error']:.4f}",
            uncomp=f"{row['mean_uncompensated_error']:.4f}",
            bad=f"{row['accepted_bad_actions']}/{row['accepted_actions']}",
            gripper=row['accepted_gripper_disagreements'],replay=row['replay_mismatches']))
    base=by['full_d10'];optimized=by['v12_p3_q4_optimized_async'];opt_pair=pairs[optimized['condition']]
    opt_paced=next(r for r in paced['paired'] if r['condition']==optimized['condition'])
    opt_wait=next(r for r in waits if r['condition']==optimized['condition'])
    base_wait=next(r for r in waits if r['condition']=='full_d10')
    best_paced=max(paced['paired'],key=lambda r:r['tail_compute_speedup_geomean'])
    joint=[r for r in paced['paired'] if r['tail_compute_speedup_geomean']>=1.5 and by[r['condition']]['success']>=base['success']]
    joint_text='本轮观察到了成功数不低于Full且尾段策略达到1.5倍的候选；样本仍不足以证明统计非劣。' if joint else '本轮未同时达到“成功数不低于Full、稳定策略临界路径1.5倍”的指标。'
    text=f'''# 选层反传、执行重叠与 V12+D2 闭环复测

四方向单次选择器已做到195.70ms均值、198.68ms P95（8输入、40次测量）；整网梯度与层分数仍未过原容差，按近似数值分支独立评测。40例中Full为{base['success']}/40，融合四方向为{optimized['success']}/40，配对恢复{opt_pair['recovered']}条、新增失败{opt_pair['new_failures']}条。成功数相同不能解释为逐案例无损。

统一20Hz执行时，融合四方向在{opt_wait['before_fast_ready']}/{opt_wait['before_fast_jobs']}次后续Fast之前均已完成，Fast前等待均值{opt_wait['before_fast_wait_mean_ms']:.4f}ms；并发图跨度均值{opt_wait['selector_concurrent_span_mean_ms']:.2f}ms。含终端排空时每次选层等待均值{opt_paced['mean_selector_wait_ms']:.3f}ms。尾段策略加速为{opt_paced['tail_compute_speedup_geomean']:.3f}倍，含动作执行的墙钟/动作比为{opt_paced['warm_wall_per_action_speedup_geomean']:.3f}倍。所测候选最快尾段策略比为{best_paced['tail_compute_speedup_geomean']:.3f}倍（{best_paced['condition']}）。{joint_text}具体瓶颈分解和离线动作差诊断见后文。

## 范围与结论口径

π0 原导出 BF16 权重、FP32 RMS、Full-D10、Fast-D2、H50/R5、17 个门控、间隔跳层6层得到V12。四方向 seed340610 及前缀嵌套 q1/q2 不变。验证 tau=.10、t=.10/.05、原夹爪保护、每次 Full 刷新真实残差及双 Full 外推不变。原生推理和验证无图；选择器使用局部 CUDA Graph。所有候选是分名实验，方向数、组合评分、刷新间隔分开比较。

Full基线调用 `prepare_full(..., trace=False)`，实际不跳层、执行18层VLM和D10；其配置中的budget/skip_count字段由V12模板继承，仅为未使用的元数据，不能把它理解为V12基线。

本轮完成 {a['completed']} 条探索闭环：原矩阵{a['main_target']}条，加融合反传独立条件40条。每个主条件40例（四套件×10 task×init19），同步控制8例。它不是官方50 init全量基准，也不构成统计非劣证明。首 Full 输出和首 Fast 输入共 {a['prefix_checks']} 项核验全部通过；同步/异步四方向串行在8条完整轨迹中一致，详见 `closed_loop_sync_async_equality.json`。全部逐回合机制审计通过，所有有效超时均按失败保留。

## 四方向选层已压进200ms，但数值等价未成立

最快的四方向实现：固定8个真实输入×5次热态测量，均值195.70ms、P95 198.68ms、最大198.85ms。仍共享一次Full-D10可微前向，方向、原层评分和每Full刷新不变。前向最大差0，V12/V14共16个选层结果全部相同；四组梯度与层分数均未通过原容差，因此另列40例闭环候选，不能视作原实现的数值等价替换。

选择器单次口径包括重新投影/复制当前输入及显式噪声、一次可微 Full-D10、q 个VJP、梯度回CPU及评分。它不含此前原生Full的vision/VLM/AE，也不含验证或RPC。下方初始五种方法各测8个固定真实输入×10次；后续融合探测各测8输入×5次，另列构图成本。

{table(costs,[('method','方法'),('n','热态次数'),('mean','均值ms'),('p95','P95 ms'),('passes','与对应串行q方向数值核验'),('mask_changes','选层变化输入数（V12/V14合计判定）')])}

{table(prep_table,[('method','方法'),('capture','构图含3次eager预热均值s'),('allocated','构图后allocated最大MiB'),('reserved','构图后reserved最大MiB')])}

该表显存字段是构图后存量而非运行峰值。前轮独立四方向一批图探测记录的真正峰值为 allocated9779MiB、reserved16120MiB，模型加载21.07s；原始峰值见 `results/idea1_v3_4_selection_cost/batch200_graph/preparation.csv`。不能把warm时延当成首次部署成本。

q1的数值通过是相对原四方向中第一方向的梯度，并不表示q1与q4选层相同。q2和q4批量相对串行未通过既定梯度容差，必须作为浮点变化候选评测；掩码拆分版本连可微前向也未通过原容差，不能称等价优化。

历史455ms是四次串行的局部图热态，本轮459ms复现同量级；此前无图四方向一批约910ms，与图结果不同。无图串行约2019ms。不能拿455ms当无图四次串行基线。

原带掩码注意力反向kernel累计约 {profile['q4_batched']['top_kernels'][0]['ms']:.2f}ms；全部kernel工作累计 {profile['q4_batched']['kernel_work_ms']:.2f}ms。掩码按state/action数学语义拆为无掩码调用后实际切到Flash，但BF16舍入改变。上述kernel总量只计Chrome trace `cat=kernel`，不重复加算operator或record-function事件。

算子profile还显示，原四方向批量接口内部仍执行716次带掩码注意力反向（179×4）和64次前缀Flash反向（16×4）。注意力反向存在逐方向fallback，批量接口并未把每种算子都合成一次物理batch；这也是图版本收益没有达到四倍的原因之一。

显式展开方向维度后，注意力反向调用降为179次带掩码加16次Flash，四方向均在各次调用的batch维度中计算。保持原注意力掩码的版本先降到220.64ms。随后只融合RMS、旋转位置编码及FFN的反向kernel；前向继续用原表达式，保留FP32 RMS及必要BF16舍入边界。结果如下：

{table(fusion_table,[('method','四方向物理batch＋反向融合'),('mean','均值ms'),('p95','P95 ms'),('max','最大ms'),('raw','前向通过'),('gradient','梯度通过'),('layers','层分数通过'),('mask','V12选层一致'),('capture','构图含eager预热s'),('allocated','峰值allocated MiB'),('reserved','峰值reserved MiB')])}

正式闭环还增加3次图回放预热，其耗时计入各回合的`selector_capture_ms`和累计准备成本。

所有数值判断沿用raw atol/rtol=1e-5、梯度/层分数atol=1e-4及rtol=1e-3，未放宽。RMS＋RoPE均值虽低于200ms，但P95仍超过；加入FFN反向融合后40次全部低于200ms。新实现依赖当前PyTorch私有vmap接口和Triton反向kernel，部署前仍需覆盖新增shape；异步与渲染竞争时的GPU跨度应另外看闭环数据。

## 与AE和动作执行的重叠

原生Triton VLM没有可直接复用的反向激活，当前梯度还依赖可微AE的输出。因此这里在原生VLM完成后，启动独立可微Full-D10图，与高优先级原生AE及随后动作执行重叠。每个待完成任务绑定其Full轮次和step；下一Fast在取新mask前只等该任务的结束事件，不用旧mask提前放行。首次构图有全设备同步，常规launch没有；collect只同步相关结束事件。

固定输入的384项动作/梯度/选层对照全部通过。trace中不同流的真实计算kernel同时执行约 {profile['overlap']['kernel_overlap_ms']:.2f}ms。下表执行窗口是固定输入探测时的统一CPU等待，不是闭环成功率证据；动作就绪与剩余等待用真实GPU事件和墙钟测量。

此隔离表的q4采用原SDPA批量图（约375ms）。融合四方向的隐藏效果在后面的统一20Hz闭环中单独核对。

{table(overlap,[('q','方向'),('execution','执行窗口ms'),('ready','AE动作就绪ms'),('critical','扣除执行窗口的临界计算ms'),('wait','剩余选层等待ms'),('hidden','返回前已完成')])}

![固定输入下的250ms执行窗口与选择器重叠](/root/nfs/embodied/benchmark/results/idea1_v3_4_selection_async/figures/overlap_250ms.png)

## 自由闭环：质量与计算开销

策略成本按实际执行动作均摊，扣除单列构图/预热；包含下一Fast的延迟等待及回合末尚未结束的GPU任务排空。尾段列统一去掉每回合前三个请求，去除其余首次调用初始化影响。表中策略倍速采用尾段：与相同case的Full-D10按每动作成本求比后取几何均值；wall列是扣构图后的完整回合墙钟比。不同策略执行步数会变，不能把这两种比值混为一谈。实际sim步没有强制50ms等待。

{table(quality,[('condition','条件'),('success','成功'),('changes','相对Full恢复/新增失败'),('warm','策略ms/动作'),('tail','尾段ms/动作'),('speed','策略倍速'),('wall','回合wall倍速'),('full','实际Full比例'),('wait','每次反推剩余等候ms'),('capture','构图预热累计s')])}

每条件累计episode墙钟及其归一吞吐如下，包含构图、环境acquire/reset、RPC、实际step、输入保存与结果落盘；不含独立模型加载和episode返回后的env.close。它不是多卡同时运行的总墙钟。逐回合与分阶段原值在`episodes.csv`和各`result.json`中。

{table(throughput,[('condition','条件'),('n','例数'),('success','成功数'),('wall','累计episode墙钟s'),('prep','其中构图预热s'),('episodes','回合/小时'),('actions','动作/秒')])}

## 组合评分与减少方向

层分数相加为 `sum_selected sqrt(mean_q(g[q,l]^2))`；组合评分为 `sqrt(mean_q(sum_selected g[q,l])^2)`。穷举924个合法六层组合，以字典序打破并列；只复用带符号梯度，不增加反传。它仍是一阶近似，既不保证六层同时跳过的非线性误差，也不保证外推补偿或D2相对D10的采样误差。

四方向组合评分的前轮64例独立对照已完成：V12/P3从8/8到7/8，V12/P5均7/8；V14/P3从7/8到8/8，V14/P5均8/8。详见[前轮报告](/root/nfs/embodied/benchmark/docs/experiments/idea1/idea1_v3_4_selection_cost_report.md)。本轮进一步比较q1/q2的原评分与组合评分。

{table(scores,[('period','Full/Fast周期'),('q','方向'),('layer','原评分成功'),('combo','组合评分成功'),('layer_ms','原评分均值ms'),('combo_ms','组合评分均值ms')])}

共同启动Full的梯度还可用来检查少方向的盲区：对q1/q2所选mask，用没有参与该次选层的其余固定方向评分。下表全是线性代理分数，不是当前动作的真实非线性误差。少方向组合分数接近零时，仍须核对其余方向及真实闭环。

{table(projection_table,[('condition','条件'),('n','共同Full数'),('own','所用方向组合RMS均值'),('heldout','其余方向组合RMS均值'),('full','全部四方向组合RMS均值')])}

## 降低刷新频率

下面只改变每隔2/3次实际Full重新选层，使用原四方向串行图与原层评分。首次Full必选；Full/Fast周期均为3，外推残差在每次Full刷新。逐请求审计核对了实际Full索引、完整17层残差、两次最新Full历史及选层任务起点。触发Full延迟是动作返回时间，尾部等待另计在后续Fast。

{table(refresh,[('condition','条件'),('success','成功'),('fulls','实际Full'),('selectors','选层次数'),('vjp','VJP数'),('amortized','策略ms/动作'),('trigger','触发Full均值ms'),('trigger95','触发Full P95 ms'),('fast95','Fast P95 ms')])}

## 统一20Hz执行的闭环确认

给Full基线和所有候选都按每动作至少50ms执行；每步真实env.step/render耗时包含在50ms中。候选从40例发现集按成功数优先、尾段成本次优选q1/q2各一个，另测原四方向一批与融合四方向一批。这是时延确认，不是独立质量留出集。

原8例中，Libero10/task2/init19的双方向回合在58次请求中出现1次原始观测hash变化；全部原始输出、动作、梯度以及整条telemetry仍逐位相同。该回合保留，但整个case的五个条件均不进入严格速度配对。随后按冻结40例顺序选第一个未测case（Spatial/task1/init19）补充一次，未重跑原回合，未放宽容差。共计9case×5条件=45个有效回合；严格速度比较采用其中{len(paced['strict_cases'])}个共同case，所有input/动作/mask/route及执行长度核验通过。原始记录及补充规则见`paced_input_anomaly.json`和`paced_recovery_manifest.json`。

{table(paced_table,[('condition','条件'),('success','全部测量成功'),('n','严格配对case数'),('compute','策略几何倍速'),('pooled','策略合并倍速'),('wall','含执行wall/动作倍速'),('episode','回合wall倍速'),('hidden','严格样本选层提前完成/总数'),('wait','剩余等候均值ms')])}

计算路径达到1.5倍，并不表示包含共同动作执行窗口的回合墙钟也达到1.5倍；失败或轨迹变长还会影响回合吞吐。成功数相同也不代表逐案例没有退化，应同时看恢复/新增失败和样本量。

后续Fast的等待与回合结束时GPU任务排空分开如下；图跨度包含其他GPU工作带来的竞争，不能当作纯kernel工作量。原生Full AE在候选中和后台选择器同时运行，其时延也应与Full参考一起检查，隐藏等待不保证AE完全不受资源争用影响。执行耗时包含统一50ms步节奏，超时窗口另存`paced_wait_breakdown.csv`。

{table(wait_table,[('condition','条件'),('ready','后续Fast前选层已完成/次数'),('mean','Fast前等待均值ms'),('p95','Fast前等待P95 ms'),('end','终端GPU等待累计ms'),('span','并发选择器图跨度均值ms'),('ae','原生Full AE均值ms'),('execution','实际执行ms/动作')])}

融合路径的原生Full AE均值为{opt_wait['native_full_ae_mean_ms']:.2f}ms，Full参考为{base_wait['native_full_ae_mean_ms']:.2f}ms；动作执行分别为{opt_wait['actual_execution_ms_per_action']:.2f}与{base_wait['actual_execution_ms_per_action']:.2f}ms/动作。这里没有靠放慢动作执行来制造反传隐藏。

## 未达指标时的诊断

`requests.csv`逐请求保留Full/Fast、vision/VLM/AE/验证、刷新与等待。实际Full比例由周期、夹爪保护、验证拒绝接管共同决定；拒绝会同时支付draft、verify和Full。剩余等待归零只能消除选择器暴露在临界路径上的成本，不能消除这些共同成本，也没有减少GPU总反向工作。

{table(timing_table,[('condition','条件'),('full','实际Full比例'),('fallback','draft后零接受比例'),('draft','draft路径均值ms'),('verify','验证路径均值ms'),('speed','尾段策略合并倍速'),('oracle','去全部暴露选层开销的乐观会计倍速'),('wait','选层等待ms/动作'),('waste','被拒proposal额外ms/动作')])}

“去全部暴露选层开销”从实测策略时间减掉收集、启动和末尾排空，不改实际Full/draft组成，也不把重叠的GPU跨度相加。它是定位瓶颈的乐观会计对照，不是已实现的新策略或硬件下界；若它仍低于1.5，说明只隐藏反传不足以达到该指标。

离线诊断只在正式回合结束后，用候选实时观测和同一显式噪声重算当前Full-D10；参考动作只用于诊断。另算不跳层的D2，分离少步采样的影响；重建历史真实残差核对原补偿draft，再对照无补偿V12。误差是首5动作前6维差的RMS，坏接受指已接受动作相对当前Full-D10的该误差超过原tau=.10。它是动作差诊断，不能单独作为每条失败的因果解释。

{table(diagnostics,[('condition','条件'),('n','固定早期请求数'),('current','补偿draft对当前D10平均误差'),('d2','不跳层D2对D10平均误差'),('uncomp','无补偿V12 D2对D10平均误差'),('bad','接受后超阈值动作/接受数'),('gripper','接受夹爪符号差异数'),('replay','补偿重放不一致数')])}

## 结果与复现

数据根目录：`results/idea1_v3_4_selection_async/`。`closed_loop_manifest.json`冻结原矩阵，`optimized_manifest.json`单列融合条件；`optimized_source_snapshot.json`指向该条件启动时的源码快照。`compression/`、`overlap/`、`sdpa_batch_probe/`、`rms_batch_probe/`、`elementwise_batch_probe/`保留完整有界探测。`closed_loop/`与`optimized_closed_loop/`保留逐episode结果/推理日志/原始动作/必要观测/EEF；`episode_audits/`、`optimized_audits/`和`prefix.csv`保留机制及共同前缀核验。`paced/`、`diagnosis/`保存后续时延确认及离线诊断，`followup_manifest.json`记录选择规则及代码身份。冷构图和预热产生的额外反向计入准备成本，steady VJP计数仅指实际选层使用的方向数。

后续服务额外算子检查曾继承推理后端的禁梯度上下文而启动失败；局部启用梯度后重启。该次发生在paced回合开始前，608条原闭环均保留，旧启动记录在`attempts/followup_startup1/`，修正记录在`followup_startup_recovery.json`。独立GPU局部算子公式检查和四卡启动核验另存`gpu_operator_preflight.json`及各worker的`gpu_operator_checks.json`；它们不替代整网的严格梯度核验。
'''
    path=ROOT/'docs/experiments/idea1/idea1_v3_4_selection_async_report.md';path.write_text(text)
    from preparations.idea1_v3.io import atomic
    atomic(OUT/'report_generation.json',dict(report=str(path),report_sha256=sha(path),
        generator_sha256=sha(__file__),analysis_sha256=sha(OUT/'analysis.json'),time=time.time()))
    return path


if __name__=='__main__':print(render())
