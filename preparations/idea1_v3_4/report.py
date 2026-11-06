"""Render Chinese final reports only after complete evidence audit passes."""
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .manifest import ROOT,OUT,SUITES


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def nfmt(value,places=3):
    return '不可估计' if value is None else f'{float(value):.{places}f}'


def main():
    audit=json.loads((OUT/'final_audit.json').read_text())
    if not audit['passed']: raise RuntimeError('Final report requires complete passing audit')
    data=json.loads((OUT/'analysis/analysis_state.json').read_text()); quality=data['quality']; paired=data['comparisons']; mech=data['mechanism']; e3=data['E3']
    def q(name,suite='all'):
        return next(r for r in quality if r['condition']==name and r['scope']=='all' and r['suite']==suite)
    names=['full_d10','full_d2','v14_d2','periodic_v14','periodic_old16','periodic_old13','periodic_old12']
    quality_table=table(['路径','Spatial','Object','Goal','Long','总成功','策略ms/动作','平均动作数','回合/小时/worker'],[
        [name,*[f"{q(name,s)['success']}/{q(name,s)['n']}" for s in SUITES],f"{q(name)['success']}/{q(name)['n']}",nfmt(q(name)['policy_ms_per_action'],2),nfmt(q(name)['actions_mean'],1),nfmt(q(name)['worker_episode_per_hour'],1)] for name in names])
    comparison=next(r for r in paired if r['baseline']=='periodic_v14' and r['candidate']=='periodic_old16' and r['scope']=='all' and r['suite']=='all')
    spatial=next(r for r in paired if r['baseline']=='periodic_v14' and r['candidate']=='periodic_old16' and r['scope']=='all' and r['suite']=='libero_spatial')
    ci_low=comparison['delta_ci_low_pp']; ci_high=comparison['delta_ci_high_pp']
    if abs(ci_low)<1e-10: ci_low=0.
    if abs(ci_high)<1e-10: ci_high=0.
    ci_note='区间触及零，不能把这次净下降当作总体下降的确定证明。' if ci_low<=0<=ci_high else '区间为探索性task聚类估计，未作多重比较校正。'
    action_cost_drop=(1-1/comparison['paired_policy_cost_ratio'])*100
    campaign=json.loads((OUT/'status.json').read_text())['time']-json.loads((OUT/'controller_process.json').read_text())['started']
    loading=[]
    for gpu in (0,2,3):
        loaded=json.loads((OUT/'workers'/f'gpu{gpu}'/'loaded_protocol.json').read_text())
        loading.append([gpu,nfmt(loaded['protocol']['service_load_s'],2),nfmt(loaded['load_and_gate_s'],2)])
    loading_table=table(['GPU','模型服务构建秒','进程加载＋真实输入门秒'],loading)
    timing=table(['路径','请求wall均值ms','P50 ms','vision ms','VLM ms','AE ms'],[[r['path'],*[nfmt(r[k],3) for k in ('wall_ms_mean','wall_ms_p50','vision_ms','vlm_ms','ae_ms')]] for r in mech['timing']])
    timed={r['path']:r for r in mech['timing']}
    request_speed=timed['v14']['wall_ms_mean']/timed['old16']['wall_ms_mean']
    vlm_speed=timed['v14']['vlm_ms']/timed['old16']['vlm_ms']
    def m(candidate,d,suite='all'):
        return next(r for r in mech['summary'] if r['candidate']==candidate and r['D']==str(d) and r['horizon']=='5' and r['component']=='translation' and r['suite']==suite)
    mechanism_names=['v14',*[f'old_suffix_L{i}' for i in range(12,19)],'all_old','oracle_suffix_L12','oracle_suffix_L13','oracle_suffix_L16']
    mechanism_table=table(['KV条件','D2 全套件mm代理','D10 全套件mm代理','D2 Spatial','D10 Spatial'],[
        [name,nfmt(m(name,2)['controller_delta_proxy_mm']),nfmt(m(name,10)['controller_delta_proxy_mm']),nfmt(m(name,2,'libero_spatial')['controller_delta_proxy_mm']),nfmt(m(name,10,'libero_spatial')['controller_delta_proxy_mm'])] for name in mechanism_names])
    suffix_candidates=[f'old_suffix_L{i}' for i in range(12,19)]
    best_suffix=min(suffix_candidates,key=lambda name:m(name,2)['controller_delta_proxy_mm'])
    l16paired=next(r for r in mech['paired'] if r['candidate']=='old_suffix_L16' and r['D']=='2')
    layer_summary=list(csv.DictReader((OUT/'analysis/mechanism_layer_summary.csv').open()))
    def layer_metric(candidate,kv,layer):
        return float(next(r['relative_l2'] for r in layer_summary if r['candidate']==candidate and r['kv']==kv and r['region']=='all' and r['suite']=='all' and int(r['layer'])==layer))
    layer_table=table(['层','V14 K相对L2','旧Full K相对L2','V14 V相对L2','旧Full V相对L2'],[
        [layer,*[nfmt(layer_metric(candidate,kv,layer)) for kv,candidate in [('K','v14'),('K','old'),('V','v14'),('V','old')]]] for layer in range(12,19)])
    replication=('本次160对的全任务平均没有复现临时探针的L16–18误差改善。' if l16paired['delta_mm']>0 else '本次160对观察到L16–18平均误差改善，但须结合区间和闭环结果判断。')
    historical_table=table(['当前数据子集（事后诊断）','D','n','V14 mm','L16旧KV mm','差值mm'],[
        [r['scope'],r['D'],r['n'],nfmt(r['v14_mm']),nfmt(r['old16_mm']),nfmt(r['delta_mm'])] for r in mech['historical_scope']])
    effects=list(csv.DictReader((OUT/'analysis/mechanism_layer_effects.csv').open()))
    effect_table=table(['替换','D2动作响应RMS','误差交叉项均值 E[e·Δ]','E[Δ²]','MSE变化','组合交互RMS'],[
        [kind+str(layer),*[nfmt(np.mean([float(r[field]) for r in effects if r['D']=='2' and r['kind']==kind and int(r['layer'])==layer]),6) if field!='interaction_rms' or kind=='suffix' else '—' for field in ('response_rms','error_alignment','squared_response','delta_mse','interaction_rms')]]
        for kind,layer in [('single',i) for i in range(12,19)]+[('suffix',i) for i in (12,13,16)]])
    e1=f'''# idea1 v3.4 E1：旧 Full KV 的闭环质量、真实速度与逐层机制

本轮固定920条E1闭环和160对真实相邻观测的D2/D10消融已完成。主比较周期V14与L16后缀复用相对成功率差为 **{comparison['delta_pp']:+.2f}个百分点**，按suite分层task聚类bootstrap95%探索区间 **[{ci_low:+.2f},{ci_high:+.2f}]**；恢复{comparison['recovered']}例、损失{comparison['lost']}例。Spatial为{spatial['baseline_success']}→{spatial['candidate_success']}/{spatial['n']}。{ci_note}这些是探索结果，不是总体非劣保证。

{replication}D2前5步平移代理为{m('v14',2)['controller_delta_proxy_mm']:.3f}→{m('old_suffix_L16',2)['controller_delta_proxy_mm']:.3f}mm，细化网格中最低均值是{best_suffix}。原临时网格只测L10/L13/L16，不足以得出“只有L16–18有效”的结论。单层干预与当前Full oracle进一步说明，层数值偏差、动作敏感度、修正方向与旧观测时效性必须分别判断。

## 闭环比较

{quality_table}

四主条件各200条，其他诊断条件各40条。init14初筛与init14–18最终分母分开保存在CSV中，不把40与200直接做配对比较。周期路径每3次重规划为Full-D10、draft-D2、draft-D2；没有verify或夹爪保护。V14与hybrid的区别为尾部KV来源与对应真实计算截断。旧Full由各自自由闭环历史计算并独立保存，实际缓存年龄为5/10控制步。Full刷新、输入输出变换与被执行请求全部计入策略时间。

在相同200个case上，**基线周期V14的累计策略wall/动作 ÷ L16复用的累计策略wall/动作**为 **{comparison['paired_policy_cost_ratio']:.3f}×**，即候选每执行动作的策略耗时减少约 **{action_cost_drop:.2f}%**。自由闭环观测和回合长度会分叉，因此这个比值与同输入推理加速分别解释。旧KV回合平均动作数更长，实际回合吞吐为{q('periodic_v14')['worker_episode_per_hour']:.1f}→{q('periodic_old16')['worker_episode_per_hour']:.1f}回合/小时/worker，没有获得完整评测吞吐提升。该吞吐包含环境、RPC和必要I/O；Full初筛额外保存模型观测，不能把该I/O差全部算成算法加速。逐case、逐task、配对恢复/损失和终止动作数见analysis目录CSV。

## 真正停止后层计算后的同输入速度

{timing}

同卡、同真实输入、同noise、相同计时边界，预热后交替AB/BA三次；三个服务沿固定task分片，在每个服务内比较各路径。L16相对V14请求wall均值加速 **{request_speed:.3f}×**，VLM模块 **{vlm_speed:.3f}×**。Vision保持完整，AE保持18层D2。单个hybrid请求表不含建立旧Full的成本；上面的周期闭环包含刷新成本。Full-D10和Full-D2单独列出，不能用不同D的AE成本声称KV截断收益。加载/真实输入门耗时保存在各worker loaded_protocol.json。

{loading_table}

主实验队列从启动到三个worker退出共{campaign/60:.2f}分钟，包含960条闭环、160对机制重放、加载/门检查与各阶段切换。噪声补充在空闲卡并行完成。这是本次执行工期，不是某条策略的部署吞吐；逐条件动作/秒、请求/秒、reset/环境/RPC/物理/I/O分解保存在quality_summary.csv。

## 为什么不是所有后层都适合换旧KV

{mechanism_table}

新160对数据中，L16尾部替换的D2前5步平移代理为{m('v14',2)['controller_delta_proxy_mm']:.3f}→{m('old_suffix_L16',2)['controller_delta_proxy_mm']:.3f}mm，配对差{l16paired['delta_mm']:+.3f}mm，task聚类探索区间[{l16paired['low']:+.3f},{l16paired['high']:+.3f}]mm；改善比例{l16paired['improved_ratio']:.1%}。本次后缀网格的最低均值为{best_suffix}（{m(best_suffix,2)['controller_delta_proxy_mm']:.3f}mm），因此“只有L16–18可能改善”不是结构定律。该最小值是探索网格内描述，不把事后最佳后缀当作已确认的闭环候选。

Spatial在本轮shadow中的L16代理由{m('v14',2,'libero_spatial')['controller_delta_proxy_mm']:.3f}降至{m('old_suffix_L16',2,'libero_spatial')['controller_delta_proxy_mm']:.3f}mm，闭环成功数却由{spatial['baseline_success']}降至{spatial['candidate_success']}/{spatial['n']}。短期动作接近程度不足以预测自由闭环成功率。shadow固定age5控制步，闭环使用age5/10并累积自身轨迹分叉；本轮未单独操控age，不能把退化唯一归因于缓存年龄。

表中mm为前5步clip后平移控制命令差×50mm，仅是控制目标增量差代理，不是真实EEF位置差。160对来自四套件40条Full轨迹，每条按索引预选4点，每对上一重规划相隔5个真实控制步；D2/D10使用同一批观测。机制轨迹与候选自由闭环成绩独立。oracle使用当前Full对应尾层，只作条件干预，不能作为可部署加速路径。

临时探针覆盖各suite的task0/task5、init0/1；本轮覆盖全部task、init14。两批都使用π0 Full轨迹，但当前状态/选点不同，属于扩展覆盖，不能把总体均值变化唯一归因于某个实现因素。为检查任务覆盖的作用，按**既有探针的task集合**对当前数据做以下事后诊断；它不改变主样本、参数或候选选择。

{historical_table}

在同一本轮backend/init规则下，原task集合与其他task的平均方向不同，支持任务覆盖/状态分布会改变平均收益。旧集合在本轮只有32对，不把它与原64对当成同一批重测；全160对仍是主要口径。该分组也没有独立操控任务语义与运动状态，不能称唯一因果解释。

{layer_table}

上述KV偏差与动作误差来自不同量纲。即便L12–15的旧KV更接近当前Full，换入这些层后仍需检查AE动作响应。相同位置的当前Full oracle将L12起的前缀误差代理降至{m('oracle_suffix_L12',2)['controller_delta_proxy_mm']:.3f}mm，而旧Full L12起为{m('old_suffix_L12',2)['controller_delta_proxy_mm']:.3f}mm；这表明用当前正确特征修复跳层偏差与用陈旧特征替换是不同干预，不能按KV相似度直接等同。

{effect_table}

这里e是原V14相对当前Full的normalized前5步前6维误差，Δ是替换造成的动作变化。MSE变化严格分解为2E[e·Δ]+E[Δ²]。旧KV在数值上更接近Full，并不保证AE响应方向能抵消原误差：动作敏感度、修正方向和旧观测缺失的信息都要同时考虑。单层与整段响应之差量化跨层组合的非线性；不能把整段效果当作逐层误差简单相加。当前Full尾层oracle与旧Full尾层的差异用于区分同位置跳层特征误差和时效性，不据此声称识别了唯一内部因果机制。

单独替换L15、L16、L17的D2平移代理分别为{m('old_single_L15',2)['controller_delta_proxy_mm']:.3f}、{m('old_single_L16',2)['controller_delta_proxy_mm']:.3f}、{m('old_single_L17',2)['controller_delta_proxy_mm']:.3f}mm。L15单层也可能改善，L16单层反而变差；整段L15起为{m('old_suffix_L15',2)['controller_delta_proxy_mm']:.3f}mm，进一步说明不能仅按“位于跳层之后”给各层统一判定可靠性。

逐层K/V、token区域、范数与余弦统计在mechanism_layers.csv；逐对单层/整段响应在mechanism_layer_effects.csv。Spatial及其他suite分别统计，不以总体均值掩盖相反方向。相机RGB变化和EEF位移仅为观测变化代理，保存在observation_changes.csv，不是对象语义或失败原因真值。

![闭环质量与策略成本](../../../results/idea1_v3_4/analysis/closed_loop.png)

![后缀替换与套件差异](../../../results/idea1_v3_4/analysis/suffix_mechanism.png)

![逐层KV偏差、动作干预与当前Full oracle](../../../results/idea1_v3_4/analysis/layer_mechanism.png)

## 可复现边界

原π0 checkpoint、BF16 Triton、V14原始1-based mask[3,4,9,11]、H50/R5、20Hz、官方初态、10步settling和固定request seed。每服务真实四套件输入的截断KV与“完整计算再替换”逐位一致，原始/decoded动作差0；跨服务参考24个数组差0。三项针对性CPU测试与q=1 warm-start原生D2等价检查通过。详见[冻结方案](idea1_v3_4_experiment_plan.md)、[manifest](../../../results/idea1_v3_4/manifest.json)、[最终验收](../../../results/idea1_v3_4/final_audit.json)及各worker gate/cross_service/qk收据。未修改原idea1.md，不使用历史临时探针充当新闭环数据。
'''
    e3table=table(['原请求分组','候选','请求数','平移mm代理','旋转命令L2','夹爪符号差','重新非零接受率','平移改进比例','平移Pearson','AE ms'],[
        [r['group'],r['candidate'],r['n'],nfmt(r['translation_proxy_mm']),nfmt(r['rotation_l2'],4),nfmt(r['gripper_sign'],4),nfmt(r['verify_nonzero_rate'],4),nfmt(r['translation_improved_ratio'],4),nfmt(r['translation_pearson'],4),nfmt(r['ae_ms'],3)] for r in e3['summary']])
    cause_table=table(['原验证触发因素','候选','n','平移mm代理','相对D2变化mm','重新非零接受率'],[
        [r['cause'],r['candidate'],r['n'],nfmt(r['translation_proxy_mm']),nfmt(r['translation_delta_mm']),nfmt(r['verify_nonzero_rate'],4)]
        for r in e3['cause_summary'] if r['candidate'] in ('v14_d2','v14_d10','full_d10','draft_full_q03','remaining_full_q03')])
    zero={r['candidate']:r for r in e3['summary'] if r['group']=='zero'}
    noise=json.loads((OUT/'E3_noise_control/summary.json').read_text())
    def noise_row(name,reference):
        return next(r for r in noise if r['group']=='zero' and r['candidate']==name and r['reference']==reference)
    noise_names=['full_d2_e0','full_d2_e1','draft_q03_e0','draft_q03_e1','remaining_q03_e0','remaining_q03_e1']
    noise_table=table(['候选','n','相对Full-D10(ε0) mm','相对Full-D10(ε1) mm','旋转L2(ε0参考)','旋转L2(ε1参考)','ε0参考夹爪差','AE ms'],[
        [name,noise_row(name,'e0')['n'],nfmt(noise_row(name,'e0')['translation_mm']),nfmt(noise_row(name,'e1')['translation_mm']),
         nfmt(noise_row(name,'e0')['rotation_l2'],4),nfmt(noise_row(name,'e1')['rotation_l2'],4),
         nfmt(noise_row(name,'e0')['gripper_sign'],4),nfmt(noise_row(name,'e0')['ae_ms'],3)] for name in noise_names])
    e3report=f'''# idea1 v3.4 E3：真实被拒draft的信息、修复与剩余动作初始化

40条strict父系统闭环完成，父系统成功{q('E3_parent')['success']}/40。共有{e3['parent_requests']}个请求，其中{e3['verified_requests']}个实际验证请求、{e3['zero_requests']}个零接受；固定规则抽取{e3['selected_requests']}个shadow请求，其中{e3['actual_selected_zero']}个原零接受。选择每episode最早4个零接受、最早2个非零接受，不根据动作误差挑样。这批结果回答动作空间和修复机制问题，修复动作未执行，不能报告修复策略闭环成功率。

在原零接受样本上，draft平移与Full的跨样本Pearson为{nfmt(zero['v14_d2']['translation_pearson'],4)}，保留了相关信息，但平移代理误差仍有{zero['v14_d2']['translation_proxy_mm']:.3f}mm。同一V14条件增加到D10后为{zero['v14_d10']['translation_proxy_mm']:.3f}mm；换为当前Full条件、仍用D2则为{zero['full_d2']['translation_proxy_mm']:.3f}mm。步数和条件不能混为一个问题，本批条件干预的差异更明显。

当前Full条件下，拒绝draft重加噪q0.3修复两步为{zero['draft_full_q03']['translation_proxy_mm']:.3f}mm，上一轮剩余动作初始化为{zero['remaining_full_q03']['translation_proxy_mm']:.3f}mm；它们的AE成本约{zero['draft_full_q03']['ae_ms']:.2f}/{zero['remaining_full_q03']['ae_ms']:.2f}ms，而直接fresh Full-D2为{zero['full_d2']['ae_ms']:.2f}ms。三者都需要当前Full条件。在主设置ε1修复→ε0参考的比较中，warm-start的平移代理误差更大，但该比较还含噪声身份差异，补充控制见下文。单凭“复用了信息”不能推出节省了完整fallback成本。

上述主比较的fresh Full-D2与参考共享生成噪声ε0，修复使用独立ε1，因此还包含噪声身份差异。补充试验固定全部207个样本，保持实际draft/剩余动作、当前Full条件与两步日程，只切换重加噪的ε0/ε1，并使用两个Full-D10参考；未根据结果增删样本。

{noise_table}

在同ε0参考下，matched-noise draft修复为{noise_row('draft_q03_e0','e0')['translation_mm']:.3f}mm，ε1修复为{noise_row('draft_q03_e1','e0')['translation_mm']:.3f}mm，fresh Full-D2(ε0)为{noise_row('full_d2_e0','e0')['translation_mm']:.3f}mm。双参考表揭示接近程度对噪声配对的依赖，初始化结论须结合这些控制而不是只看主表。它仍是动作空间比较，不能据此断言自由闭环任务质量。

固定ε0参考时，修复的旋转命令L2为{noise_row('draft_q03_e0','e0')['rotation_l2']:.4f}，低于fresh D2的{noise_row('full_d2_e0','e0')['rotation_l2']:.4f}；夹爪符号差则为{noise_row('draft_q03_e0','e0')['gripper_sign']:.1%}，高于fresh D2的{noise_row('full_d2_e0','e0')['gripper_sign']:.1%}。配对ε1参考也保留这种平移/夹爪与旋转之间的差异。被拒draft存在可利用的信息，当前两步修复没有在全部动作维度一致优于fresh D2，不能概括成“没有任何价值”。

仅改变Full-D10本身的噪声，两个Full参考间也有{noise_row('full_d10_e1','e0')['translation_mm']:.3f}mm的平移代理差。因此这些距离含随机采样轨迹差异；模型间接近程度与真实任务质量需要分别解释。

补充数据在原GPU2 worker实际退出且空闲后运行，没有干扰主队列。207个原始输入的Full-D10参考及原ε1修复路径逐样本复现检查、所有候选动作指标重算通过，详见[补充方案](idea1_v3_4_E3_noise_control_plan.md)、[补充汇总](../../../results/idea1_v3_4/E3_noise_control/summary.csv)和[补充验收](../../../results/idea1_v3_4/E3_noise_control/audit.json)。

## 动作相关性与修复

{e3table}

Full-D10为同当前观测、state和生成noise的参考。V14-D2与V14-D10使用同一当前V14 KV，Full-D2与Full-D10使用同一当前Full KV，分别隔离步数和条件。draft_full在当前Full KV下、draft_v14在原当前V14 KV下重加噪并修复两步：z=(1−q)draft+q noise，从q均匀Euler到0。q固定0.1/0.3；refine noise独立域哈希，不与生成或验证noise共用。remaining先将上一轮实际提出的raw动作前移**实际执行长度**，尾部保持最后动作，再q0.3修复。缺历史不虚构剩余动作。

相关系数按平移/旋转/夹爪独立计算，含任务和请求之间变化；高相关不代表逐动作误差足够小。前5步误差、全50步指标、原始normalized输出与候选decoded输出均保存。重新非零接受率沿用原严格验证、旧Full KV和夹爪规则，不能充当任务成功率。零接受同时包含空间/夹爪触发；请求级详细分组见E3_routes.csv，不把路由触发因素写成最终任务失败原因。

{cause_table}

上表按原验证的空间/夹爪触发因素分组。当前Full-D10本身若在旧缓存验证器下也被拒绝，说明重新接受率同时受到旧条件与夹爪规则影响，不能将其完全归因于draft质量。单纯将D2换D10、换当前Full条件、进行修复是不同干预，不合并成一个收益。

原零接受组中，连当前Full-D10参照的重新非零接受率也只有{zero['full_d10']['verify_nonzero_rate']:.1%}；remaining_full虽达到{zero['remaining_full_q03']['verify_nonzero_rate']:.1%}，其平移代理误差与夹爪符号差仍为{zero['remaining_full_q03']['translation_proxy_mm']:.3f}mm和{zero['remaining_full_q03']['gripper_sign']:.1%}。这直接说明重新通过旧验证规则不能代替动作质量排序。原保护规则本来就会为夹爪事件强制Full，并非只根据动作是否接近当前Full来判断。

## 成本解释

E3_requests.csv保存候选AE成本，E3_costs.csv保存当前Full条件构建、Full-D10 AE与整个shadow成本，原shadow JSON保存已付draft/verify和其他Full路由时间。使用当前Full KV的修复必须重新支付当前vision/VLM成本；不能免费借用参考Full，不能把两步AE快解释成完整fallback同样快。V14条件修复可复用同请求已构建的V14 KV，但仍新增AE；这些是分项测量，不声称已部署的端到端加速。E3父系统原策略wall与shadow额外wall分别记录，shadow不污染执行动作或模型内的Full缓存。

首若干拒绝请求是固定但有选择性的诊断样本，不代表所有拒绝的无偏分布；每个episode内请求相关。离线改进需要另行命名并冻结修复策略闭环实验才能判断任务质量，本轮不拼接父系统成绩。E2原文为空，因此未编造E2实验。

数据与复现入口：[冻结方案](idea1_v3_4_experiment_plan.md)、[父系统逐case](../../../results/idea1_v3_4/E3_parent/episodes)、[逐请求分析](../../../results/idea1_v3_4/analysis/E3_requests.csv)、[完整验收](../../../results/idea1_v3_4/final_audit.json)。
'''
    docs=ROOT/'docs/experiments/idea1'
    (docs/'idea1_v3_4_E1_report.md').write_text(e1)
    (docs/'idea1_v3_4_E3_report.md').write_text(e3report)
    index=f'''# idea1 v3.4 实验验证报告

E1的920条闭环、E3父系统40条闭环、160对相邻真实观测D2/D10机制消融已完成并通过逐案例、逐执行动作及原始数组指标重算验收。E2内容为空，未执行。

主对照：周期V14成功{q('periodic_v14')['success']}/200，L16旧Full KV路径{q('periodic_old16')['success']}/200；Spatial分别{spatial['baseline_success']}/{spatial['n']}与{spatial['candidate_success']}/{spatial['n']}。L16路径相对同输入V14的请求wall均值加速{request_speed:.3f}×；计入周期Full刷新后，每执行动作的策略耗时仅减少{action_cost_drop:.2f}%。闭环成功率差为{comparison['delta_pp']:+.2f}个百分点，探索区间[{ci_low:+.2f},{ci_high:+.2f}]，上界触零。统计口径与限制见分报告。

- [E1：闭环成功率、真实加速与逐层机制](idea1_v3_4_E1_report.md)
- [E3：真实被拒draft的相关性、修复与剩余动作](idea1_v3_4_E3_report.md)
- [逐案例最终验收](../../../results/idea1_v3_4/final_audit.json)
- [完整机器可读分析](../../../results/idea1_v3_4/analysis/analysis_state.json)

E3修复仅作真实父系统状态上的shadow机制验证，未执行修复动作，其相关性或重新接受率不代表修复策略闭环成功率。
'''
    (docs/'idea1_v3_4_report.md').write_text(index)
    atomic(OUT/'report_receipt.json',dict(time=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reports={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in docs.glob('idea1_v3_4*report.md')}))
    print(docs/'idea1_v3_4_report.md')

if __name__=='__main__': main()
