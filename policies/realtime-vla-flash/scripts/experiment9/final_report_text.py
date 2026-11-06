"""Evidence-linked Chinese report for the completed Experiment 9 round."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results/experiment9_standard_vlm_pruning_20261004'
BASE='results/experiment9_standard_vlm_pruning_20261004/analysis/final/'
SUITES=('libero_object','libero_spatial','libero_goal','libero_10')


def csv_rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def build_report(suites,tasks,failures,reach,deviations,cases):
    conditions=list(dict.fromkeys(r['condition'] for r in suites))
    lookup={(r['condition'],r['suite']):r for r in suites}
    probe=json.loads((RESULTS/'action_expert_layer_probe/probe/summary.json').read_text())
    lines=['# Experiment 9：LIBERO 标准四套件完整实验分析报告','',
           '数据版本：2026-10-05；本轮主评测 **28,000 条**，补充 Action Expert 单层闭环初筛 **800 条**，合计 **28,800 条**。层编号均从 1 开始。', '',
           '## 结论摘要','',
           '1. 标准协议下 Full18 为 **1,888/2,000（94.4%）**，不是历史 Control 的 21%。Object / Spatial / Goal / LIBERO-10 分别为 98.0% / 95.2% / 96.6% / 87.8%。',
           '2. VLM 单层 L2–L5 可以作为任务限定的加速候选，但不是全套件无损层；**L3 的总体损失最小**。同时跳过 L2–L5 显著失效，不能把单层候选直接组合。',
           '3. V12 在 Object 上是 **0/500**；V13-Custom 四套件为 **71.0% / 50.6% / 24.4% / 29.0%**。这两种多层裁剪都不适合作为当前直接部署方案。',
           '4. 优先加速方向是减少 Action Expert 降噪步数：D5/D3 成功率接近完整模型。须结合下文配对区间和时延，不能把点估计更高称为显著改进。',
           '5. Action Expert L2/L8 在真实标准任务离线探针中扰动很小，但闭环 Goal 只有 81%/84%，LIBERO-10 均 80%。**余弦高、离线 NMSE 低，都不足以证明闭环可安全删除。**','',
           '## 1. 实验协议、覆盖范围与数据质量','',
           '| 数据 | 任务 / 状态 | 用途与限制 |','|---|---|---|',
           '| 本轮主评测 | 14 个条件 × 4 套件 × 10 任务 × 50 官方初始状态（0–49） | 28,000 条；VLM 裁剪与 D7/D5/D3 全量对比 |',
           '| Action Expert 单层初筛 | L2 / L8 × 4 套件 × 10 任务 × 10 官方初始状态（0–9） | 800 条；仅与 Full18 相同 0–9 子集配对，不能称 500 条全量 |',
           '| 本轮 Action Expert 离线结构探针 | 40 任务 baseline init0、110 真实阶段观测 × 3 噪声种子 | 有目的阶段采样，不是均匀随机；阶段/噪声不作为独立任务 |',
           '| 既有 VLM 整层探针 | 3 个诊断任务、12 个真实观测 | 用于候选提出；不是本轮四套件全面 VLM 表征采样 |',
           '| 既有 Attention / FFN 探针 | LIBERO-90 8 任务 + LIBERO-10 4 任务，64 观测 × 3 种子 | 诊断来源，不能写成本轮标准四套件子模块全量结果 |','',
           '本轮统一使用训练后权重、Triton、官方初始状态、环境 seed=7、warmup=10、replan=12、强制每轮完整推理。VLM 组固定完整 Action Expert / D10；降噪组固定完整 VLM / 完整专家层；SkipAE-L2/L8 固定完整 VLM / D10，仅跳过专家的一层。比较组共享动作噪声种子与 episode/replan 噪声生成规则。','',
           'V16 跳过 L9/L10；V14 跳过 L8–L11；V12 跳过 L7–L12；V10-Mid 保留 L1–L4、L9/L10、L15–L18。V13-Custom 保留 L1/L2/L5/L6/L8/L9/L10/L11/L13/L14/L16/L17/L18，跳过 L3/L4/L7/L12/L15。','',
           '主评测按 task/init 键核查完整矩阵，重试取最新完整 attempt，未完成或重复 attempt 不计入分母。V13 曾遗漏 0–9 的 400 条，现已补齐。首次配对推理 noise hash 无不一致。此次进一步逐行读取全部 28,800 条 telemetry：验证 step=0…N、N+1 状态、有限 XYZ、配对初始 XYZ 一致。视频不是全部人工观看：原核查抽检 56 个视频头、8 个全帧解码；报告配对视频另行解码验证结果见可复核文件。','',
           '统计：Wilson 95% 只描述单格成功率；相对 baseline 的主要区间先计算每任务配对差，再以任务为单位 bootstrap 10,000 次（每套件 10 任务）。McNemar 是未多重校正的 episode 配对描述性检验，任务内相关与多重比较使其不适合单独作为最终结论。这里没有预注册非劣界限，不能声称“已证明无损”。','',
           '## 2. 成功率与 baseline 对比','',
           '| 变体 | Object | Spatial | Goal | LIBERO-10 | 总体 |','|---|---:|---:|---:|---:|---:|']
    for c in conditions:
        data=[lookup[c,s] for s in SUITES]
        total=sum(r['successes'] for r in data);n=sum(r['episodes'] for r in data)
        lines.append('| '+c+' | '+' | '.join(f'{r["success_rate"]:.1%} ({r["successes"]}/{r["episodes"]})' for r in data)+f' | {total/n:.2%} ({total}/{n}) |')
    lines += ['',f'![成功率热图]({BASE}success_heatmap.png)','',
              '### 2.1 同任务、同初始状态的配对损失','',
              '| 变体 | 套件 | 对照成功 / 配对数 | 差值 pp | 任务聚类 95%（pp） | 新失败 / 恢复 |','|---|---|---:|---:|---:|---:|']
    for r in suites:
        if r['condition']=='Full18':continue
        lines.append(f'| {r["condition"]} | {r["suite"]} | {r["matched_baseline_successes"]}/{r["episodes"]} | {r["delta_pp"]:+.1f} | [{r["task_cluster_delta95_low_pp"]:+.1f}, {r["task_cluster_delta95_high_pp"]:+.1f}] | {r["new_failures"]}/{r["recoveries"]} |')
    lines += ['', 'L3 全量总体为 1,870/2,000（93.5%），相对 baseline −0.9 pp；L2/L4/L5 总体分别 92.7% / 92.6% / 92.8%。即使 Object/Spatial 都较高，Goal 的单层下降仍为 2.6–4.2 pp，L2/L4/L5 的 LIBERO-10 下降约 4.2–5.2 pp。L3 是当前完整 block 候选里更值得继续验证的一个，但不能据此宣布跨任务无损。','',
              '### 2.2 哪些变体达到 90%？','',
              '这里的 ≥90% 指观测点估计，不能把它与达到 baseline 混为一谈。','',
              '| 变体 | 点估计 ≥90% 的套件 | Wilson 下界也 ≥90% 的套件 |','|---|---|---|']
    for c in conditions:
        point=[s.replace('libero_','') for s in SUITES if lookup[c,s]['success_rate']>=.9]
        lower=[s.replace('libero_','') for s in SUITES if lookup[c,s]['wilson95_low']>=.9]
        lines.append(f'| {c} | {", ".join(point) or "无"} | {", ".join(lower) or "无"} |')
    lines += ['', 'LIBERO-10 baseline 本身是 87.8%，所以“90% 门槛”不能作为唯一筛选规则；应比较同协议 baseline 的配对变化。套件平均会掩盖任务间差异，逐任务表已附：优先检查 Goal 中下降最大的任务，而不是只挑高成功率套件。','',
              '### 2.3 退化最明显的任务（每变体前三个，主评测）','',
              '| 变体 | 套件 / task | 变体 / baseline | 差值 pp | 任务 |','|---|---|---:|---:|---|']
    for c in conditions[1:]:
        candidates=sorted([r for r in tasks if r['condition']==c],key=lambda r:r['delta_pp'])[:3]
        for r in candidates:lines.append(f'| {c} | {r["suite"]} / {r["task_id"]} | {r["success_rate"]:.0%} / {r["matched_baseline_rate"]:.0%} | {r["delta_pp"]:+.0f} | {r["task_description"]} |')
    lines += ['', '## 3. VLM 层间相似度：来源、测法与裁剪证据','',
              '**不是随机初始化模型或随机图像。** 既有 VLM 探针使用训练好的权重，来自诊断 rollout 的真实双相机图像、状态、语言指令，按语义阶段有目的选取。已生成整层分析纳入的是 12 观测 / 3 任务；采集 manifest 后续新增到 13 条，不应改写旧产物的样本量。本轮新增的是这些候选的标准闭环评测，未重采四套件 VLM 全层探针，需明确这个证据范围。','',
              '记录 H0 输入和 H1…H18 输出，对同一 token 算 `dot(H_i,H_j)/(||H_i||·||H_j||)`，先按 agent-view / wrist-view / language 模态的 token 平均，再跨观测汇总。agent 与 wrist 各 256 token，语言长度变化；避免视觉 token 数量掩盖语言变化。同时记录 `||H_l−H_(l−1)||_F/||H_(l−1)||_F`、中心化 linear CKA，以及单层跳过的动作敏感性。','',
              '![VLM 全层对余弦热图](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_similarity_heatmap.png)','',
              '| VLM 层 | Agent 相邻余弦 | Wrist 相邻余弦 | Language 相邻余弦 | Agent 相对残差 |','|---|---:|---:|---:|---:|']
    hidden=csv_rows(ROOT/'results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_hidden_metrics.csv')
    indexed={(int(r['layer']),r['modality']):r for r in hidden}
    for layer in range(1,19):
        vals=[]
        for modality in ('agent_view','wrist_view','language'):
            r=indexed.get((layer,modality));vals.append(f'{float(r["adjacent_cosine"]):.6f}' if r else '—')
        lines.append(f'| L{layer} | '+ ' | '.join(vals)+f' | {float(indexed[layer,"agent_view"]["relative_residual"]):.6f} |')
    lines += ['', 'L2–L5 的 agent-view 相邻余弦约 0.9984–0.9991、残差约 0.05–0.07；既有固定观测单层跳过 NMSE 分别约 0.0083、0.0091、0.0113、0.0159。这些证据支持“单层候选”，本轮闭环进一步支持优先 L3、按套件评估其他三层。**联合删除 L2–L5 的标准成功率大幅下降，已直接反驳候选层的线性可组合性。**','',
              'L1/L17 变化大，优先保留。L12 以后多层残差增大；V12/V14 结果也不支持当前按中间层连续删除。L18 的余弦=1、残差=0 是当前 prefix 实现只生成 KV、未继续更新 hidden 的行为，并非 KV 无用，不能当成安全删除依据。','',
              '## 4. Layer 内 Attention 与 FFN','',
              '已有扩展探针是 12 个诊断任务（90 套件 8 个、10 套件 4 个）、64 个真实阶段观测、3 噪声种子，192 个观测×种子。记录 Attention 前、残差相加后、FFN 残差相加后的 hidden；分别测两个子模块的方向/幅度更新。Attention 干预保留 Q/K/V，跳过输出投影更新；FFN 干预仅跳过 FFN 残差更新，不能与整层删除混同。L18 KV-only 不作为普通子模块候选。','',
              '![Attention/FFN 相邻余弦](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_cosine_by_layer.png)','',
              '![子模块动作敏感性](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_action_sensitivity.png)','',
              '| 子模块 | 前 12 动作 NMSE | 任务聚类 95% | 相对残差（候选汇总口径） |','|---|---:|---:|---:|']
    ranking=csv_rows(ROOT/'results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_candidate_ranking.csv')
    for module,layer in [('attn',16),('attn',15),('ffn',2),('ffn',16)]:
        r=next(r for r in ranking if r['module']==module and int(r['layer'])==layer)
        lines.append(f'| {module.upper()} L{layer} | {float(r["prefix12_action_nmse"]):.6f} | [{float(r["prefix12_action_nmse_ci95_low"]):.6f}, {float(r["prefix12_action_nmse_ci95_high"]):.6f}] | {float(r["relative_residual"]):.6f} |')
    lines += ['', '**Attention L16 是后续子模块验证候选，不是已通过标准套件的结论。** FFN L2 隐藏更新较小，但前 12 动作 NMSE 达 0.514，表明下游对某些小更新很敏感。Attention/FFN 不应共用一个裁剪结论。当前这一轮没有 Attention-L16 / FFN-L2 标准四套件新闭环评测，这是下一轮可选补充，不是把本轮已有整层变体漏记。','',
              '## 5. 分阶段失败：一开始偏了，还是精细操作失败？','',
              '阶段标签来自真实仿真器 EEF/物体位姿、接触、抓取和 goal predicates，不是相机截图猜测。contact 条件包含真实接触 **或** EEF 距相关物体 ≤6 cm；approach 使用 ≤15 cm 或相对初始缩短 ≥4 cm；抓取后物体位移 ≥3 cm 标记 transport；目标距离 ≤10 cm 或 goal progress 提升标记 alignment。因此下面的“接触前”严格说是 **contact-or-near 阶段之前**，不能解读为经过人工验证的物理接触事件。最高到达阶段是定位线索，不是因果故障标签。','',
              '为避免把 baseline 原有失败算给裁剪，主图只统计 **baseline 成功、变体失败**。另附 `failure_stage_counts.csv` 中的 all_failures，保留完整失败分母。','',
              f'![裁剪新增失败阶段]({BASE}new_failure_stages.png)','',
              '| 变体（汇总四套件） | 新失败数 | 接触/靠近前 | 已接触/靠近但未抓取 | 抓取/操作或更后 |','|---|---:|---:|---:|---:|']
    for c in conditions[1:]:
        grouped=defaultdict(int)
        for r in failures:
            if r['condition']==c and r['cohort']=='new_failures':grouped[r['bucket']]+=r['count']
        total=sum(grouped.values())
        def value(k):return f'{grouped[k]} ({grouped[k]/total:.1%})' if total else '0'
        lines.append(f'| {c} | {total} | {value("pre_contact")} | {value("contact_before_grasp")} | {value("grasp_or_later")} |')
    lines += ['', '**阶段证据支持但限定你的精细操作假设：** L3 的 86 个新增失败中，77 个（89.5%）曾进入抓取/操作或更后阶段；专家 L2 的 35 个新增失败中有 32 个（91.4%）进入该阶段。轻度单层干预的失败较集中在后段。不过这不等于它们前段轨迹完全正确。', '', '不能从“末端轨迹的最高阶段较晚”推出前段没有偏差；有些 episode 早早偏离，但仍碰到物体后才最终失败。为拆开这两点，下面追加首次持续轨迹偏离的时点分析。','',
              '### 5.1 首次明显轨迹偏离','',
              '以同环境 step 的 `||EEF_variant−EEF_baseline||₂` 为准，连续 **5 个状态超过 2 cm** 视为一次持续偏离起点；同时保存 1/2/5 cm 阈值以便敏感性复核。只比较两个轨迹共同存在的时间，不把 baseline 提前成功后的末端保持当作实测轨迹。分类用偏离时 baseline 的阶段，避免变体本身阶段错乱造成解释偏差。','',
              '| 变体 | 新失败配对数 | 共同时间内出现持续 2 cm 偏差 | 首偏差在 baseline 接触/靠近前 | 首偏差在接触/靠近或之后 |','|---|---:|---:|---:|---:|']
    for c in conditions[1:]:
        ds=[d for d in deviations if d['condition']==c and d['outcome']=='new_failure']
        on=[d for d in ds if d['onset_2cm_step'] is not None]
        early=sum(d['onset_2cm_baseline_phase'] in ('initial','approach') for d in on)
        lines.append(f'| {c} | {len(ds)} | {len(on)} | {early} | {len(on)-early} |')
    lines += ['', '**多层裁剪不是只在精细操作才出问题：** Skip-L2-5 的 1,186 个新增失败中，1,180 个（99.5%）在 baseline 的接触/靠近阶段之前就出现持续 2 cm 偏离；V13-Custom 是 1,035/1,036（99.9%）；V12 是 1,885/1,885。相比之下，L3 为 32/86（37.2%）。因此多层变体即使最终碰到物体，轨迹也往往已在早期改变，而轻度单层干预更值得做后段精细控制分析。此阈值定义的是与 baseline 的差异，不是任意差异都必然是错误。', '', '这两张表应联合解释：首次偏离时点回答“何时开始与 baseline 不同”，最高阶段回答“最终推进到哪里”；后者无法直接证明失败由夹爪精度造成。NMSE 的 translation 通道是控制命令误差，不能把它写成实测 EEF 米数。','',
              '## 6. 真实 XYZ 轨迹、偏差与配对视频','',
              '**绘图使用 Matplotlib**。数据是逐 step 保存的 MuJoCo 世界坐标 EEF XYZ（米），不是 action 命令积分，也不是相机帧。蓝色半透明实线+圆点为 Full18，橙色半透明虚线+三角点为变体；包含 3D、XY/XZ/YZ 投影及偏差曲线，重叠时仍可分辨。','',
              '每个变体×套件选一个案例：优先新增失败，按该 cohort 的平均 XYZ 偏差取中位案例，避免只展示最极端样本；无新增失败时选双成功，否则其他配对。全部配对的数值统计仍覆盖全量，不由展示案例决定。共 60 张图，每个变体另有一段 LIBERO-10 同步配对视频（15 段），其余案例也链接原始双方视频。','',
              f'[打开四套件 XYZ 图与视频画廊]({BASE}xyz_gallery.html)','',
              '| 变体 | 全配对数 | 平均偏差的 episode 中位数（m） | episode p95 偏差的中位数（m） |','|---|---:|---:|---:|']
    for c in conditions[1:]:
        ds=[d for d in deviations if d['condition']==c]
        lines.append(f'| {c} | {len(ds)} | {np.median([d["mean_deviation_m"] for d in ds]):.4f} | {np.median([d["p95_deviation_m"] for d in ds]):.4f} |')
    lines += ['', '这里的差异是 **时间同步轨迹差异**：同一几何路径如果执行速度不同，也会有非零偏差；没有做动态时间规整，不能单凭该指标称路径几何错误。不同轨迹长度只用重叠窗口，较晚失败的独有尾段不进入配对距离，但会在图/视频中显示。','',
              '### 6.1 每个变体的 LIBERO-10 示例','',
              '| 变体 | task / init | 配对结果 | mean / p95 / max（m） | 图 | 同步视频 |','|---|---|---|---:|---|---|']
    for c in cases:
        if c['suite']=='libero_10':lines.append(f'| {c["condition"]} | {c["task_id"]}/{c["episode_idx"]} | {c["outcome"]} | {c["mean_deviation_m"]:.3f}/{c["p95_deviation_m"]:.3f}/{c["max_deviation_m"]:.3f} | [XYZ]({c["plot"]}) | [MP4]({c["video"]}) |')
    for condition in ('Skip-L3','Skip-L2-5','V13-Custom','AE-D3','SkipAE-L2'):
        c=next(c for c in cases if c['condition']==condition and c['suite']=='libero_10')
        lines += ['',f'#### {condition}：task {c["task_id"]} / init {c["episode_idx"]}','',f'![{condition} 配对 XYZ]({c["plot"]})']
    lines += ['', '## 7. Action Expert 层探针与闭环反证','',
              '从本轮 Full18 每任务官方 init0 轨迹重放 40 条，保存 initial/contact/goal_alignment 的真实观测，共 110 条（未出现阶段显式记录，不伪造）。重放每一步 EEF 与原轨迹一致，success 标签匹配。动作噪声 12345/23456/34567 固定；baseline 与逐层删除干预用相同输入、相同噪声。整个去噪过程中都删除同一专家层，测动作 token（排除 state token）的 H0…H18，记录 flow steps 0/2/5/9，即 t=1/.8/.5/.1。','',
              '隐藏余弦是同 token 的方向相似度；动作敏感性是反归一化后前 7 个控制通道，重点看实际 replan 执行的前 12 动作。NMSE=`mean((a_skip−a_base)^2)/max(mean(a_base^2),epsilon)`。在每任务内平均观测/噪声，再对 40 任务 bootstrap，不能把 330 个观测×噪声当成 330 个独立任务。', '',
              f'![Action expert hidden 与动作敏感性]({BASE}action_expert_probe.png)','',
              '| 专家层 | 前 12 动作 NMSE（任务均值） | 95% 区间 | 夹爪符号翻转 | 初始 / 接触 / 对齐 NMSE | 筛选通过 |','|---|---:|---:|---:|---|---|']
    for r in sorted(probe['ranking'],key=lambda r:r['layer']):
        ph=r['phase_task_mean_nmse']
        lines.append(f'| L{r["layer"]} | {r["task_mean_nmse"]:.6f} | [{r["bootstrap95_low"]:.6f}, {r["bootstrap95_high"]:.6f}] | {r["task_mean_gripper_flip"]:.3%} | '+ ' / '.join(f'{ph[p]:.6f}' for p in ('initial','contact','goal_alignment'))+f' | {"是" if r["eligible"] else "否"} |')
    lines += ['', '预先门槛：任务均值 NMSE≤.01、95%上界≤.02、夹爪翻转≤1%、最坏阶段均值≤.05、每 suite×阶段至少 3 个任务；最多 2 候选。L2/L8 通过，随后标准初筛 Goal 的明显下降说明这套门槛只适合筛选，不是闭环安全判据。仅 init0、有目的阶段采样和未覆盖所有重规划输入都可能限制外推；误差积累是合理假设，但本轮没有隔离其因果贡献。','',
              '探针内部普通/带记录 baseline parity、干预结束后 baseline parity 均通过，权重 hash 未变。初次跨新实例对已保存动作的严格比对有约 0.0096 最大命令差；同实例 graph/eager 差为 0，完整 baseline 跨实例也有差。对两个真实 prompt 长度的编译预热后，6 个阶段×候选校验逐值一致。证据支持编译上下文影响复现；这不是已完整隔离的编译器根因。诊断保留于 `pattern_parity_failure.json` 和 `registered_pattern_parity.json`，未通过宽松阈值掩盖差异。','',
              '## 8. 时延与更优加速方向','',
              '每个 episode 排除 infer_id=0，再取其剩余 policy 调用的中位数；表内再按该条件×suite 的 episode 中位数取中位数。这样避免首轮编译和较长失败轨迹大量调用支配统计。policy 是外部调用耗时，total 是内部记录，不与端到端混同。','',
              '`action_denoise_ms` 原始字段为空；强制完整推理时代码把 staged decoder_ms 保存为 `full_fallback_ms`，此报告明确以该来源恢复 decoder 列（详见 `timing_sources.json`）。它不是重新跑出的新计时，不应无注明改名。多 GPU/不同时间段并行测量仍受硬件/负载影响，适合本轮观察而非专用无干扰加速基准。','',
              '| 变体 | 四套件 policy 中位数（ms，等权平均） | prefix 中位数（ms） | decoder 中位数（ms） | 内部 total 中位数（ms） |','|---|---:|---:|---:|---:|']
    for c in conditions:
        group=[lookup[c,s] for s in SUITES]
        values=[np.mean([r[f'warm_episode_median_{name}'] for r in group]) for name in ('policy_time_ms','vlm_prefill_ms','decoder_ms','total_ms')]
        lines.append(f'| {c} | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    base_latency=np.mean([lookup['Full18',s]['warm_episode_median_policy_time_ms'] for s in SUITES])
    latencies={c:np.mean([lookup[c,s]['warm_episode_median_policy_time_ms'] for s in SUITES]) for c in ('AE-D5','AE-D3','Skip-L3')}
    lines += ['', '本轮四套件等权时延观察：'+f'Full18 {base_latency:.2f} ms；D5 {latencies["AE-D5"]:.2f} ms（下降 {1-latencies["AE-D5"]/base_latency:.1%}）；D3 {latencies["AE-D3"]:.2f} ms（下降 {1-latencies["AE-D3"]/base_latency:.1%}）；单删 VLM L3 {latencies["Skip-L3"]:.2f} ms（下降 {1-latencies["Skip-L3"]/base_latency:.1%}）。'+ 'D3 的 decoder 大幅缩短但端到端收益仍受 prefix/encoder 等固定成本限制，不能把降噪步数 10→3 宣称为整模型 3.3 倍加速。', '', '部署排序建议：**先 D5，验证收益后考虑 D3；其次 VLM 单删 L3 并按套件设回退；不直接部署联合 L2–L5 / V12 / V13-Custom；暂不把专家 L2/L8 删除作为跨套件通用优化。** 即使精简模型内部延迟下降，成功率低会增加重试成本；不能只按单次延迟排名。子模块 Attention L16 值得下一轮标准闭环验证，不能与已完成的专家 L2/L8 结果互相替代。','',
              '## 9. 历史 Control 21%：现在能说什么？','',
              '旧 Control 与本轮不是同协议：旧批次为 LIBERO-90+LIBERO-10 的 100 任务混合、冻结 HDF5 初态、D3 等设置；本轮为四标准套件官方初态/D10/当前完整实现。旧同一批 100 条改 D10 只提升到约 24%，所以“只是降噪步数少”不足以解释原来 21%。','',
              '**已确认：当前标准基线并不低，94.4%；仍未确认：旧 21% 的单一根因。** 当前高基线排除了将 21% 当作标准模型性能的解释，但不能证明是哪一项旧协议差异造成下降。若继续追旧根因，应固定旧任务/初态/噪声，每次只换一个因素；这轮没有新增该专项析因实验，报告不把相关性写成根因。','',
              '## 10. 产物与仍需补充的工作','',
              '- 主评测与条件式专家闭环初筛均完成，数据总量 28,800；不存在本轮主矩阵缺批。',
              '- 全量失败阶段、任务成功率、配对 XYZ、60 张轨迹图和 15 段同步视频已生成。其余 45 个展示案例链接双方原始视频，并非为每个案例生成合成视频。',
              '- 尚未完成的研究扩展：四标准套件的 VLM 全层/Attention/FFN 新探针；Attention-L16 标准闭环；专家 L2/L8 的 50 状态扩展（当前负向初筛不建议直接扩展）；旧 Control 的单因素根因审计。上述范围不能混成“本轮全量评测已漏跑”。',
              '- 统计未做跨全部条件的多重校正、未做预注册非劣检验；当前报告用于候选判断与退化定位。','',
              f'[套件与配对区间 CSV]({BASE}suite_success.csv) · [逐任务 CSV]({BASE}task_success.csv) · [失败阶段 CSV]({BASE}failure_stage_counts.csv) · [阶段到达 CSV]({BASE}stage_reach.csv)','',
              f'[全部配对 XYZ 与首偏差 CSV]({BASE}paired_xyz_all.csv) · [案例索引]({BASE}xyz_case_index.csv) · [轨迹/视频画廊]({BASE}xyz_gallery.html) · [图、视频及链接校验]({BASE}artifact_validation.json)','',
              '[主矩阵核查](experiment9_completion_audit.md) · [AE 探针原始汇总](results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/probe/summary.json) · [AE 闭环原始汇总](results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/closed_loop/screening_summary.json)','',
              '复现：`.venv/bin/python scripts/experiment9/build_final_analysis.py`。报告的绘图/视频复用既有真实 XYZ 实现；生成的分析文件不作为模型训练输入。','']
    return '\n'.join(lines)
