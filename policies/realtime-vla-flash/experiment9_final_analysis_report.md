# Experiment 9：LIBERO 标准四套件完整实验分析报告

数据版本：2026-10-05；本轮主评测 **28,000 条**，补充 Action Expert 单层闭环初筛 **800 条**，合计 **28,800 条**。层编号均从 1 开始。

## 结论摘要

1. 标准协议下 Full18 为 **1,888/2,000（94.4%）**，不是历史 Control 的 21%。Object / Spatial / Goal / LIBERO-10 分别为 98.0% / 95.2% / 96.6% / 87.8%。
2. VLM 单层 L2–L5 可以作为任务限定的加速候选，但不是全套件无损层；**L3 的总体损失最小**。同时跳过 L2–L5 显著失效，不能把单层候选直接组合。
3. V12 在 Object 上是 **0/500**；V13-Custom 四套件为 **71.0% / 50.6% / 24.4% / 29.0%**。这两种多层裁剪都不适合作为当前直接部署方案。
4. 优先加速方向是减少 Action Expert 降噪步数：D5/D3 成功率接近完整模型。须结合下文配对区间和时延，不能把点估计更高称为显著改进。
5. Action Expert L2/L8 在真实标准任务离线探针中扰动很小，但闭环 Goal 只有 81%/84%，LIBERO-10 均 80%。**余弦高、离线 NMSE 低，都不足以证明闭环可安全删除。**

## 1. 实验协议、覆盖范围与数据质量

| 数据 | 任务 / 状态 | 用途与限制 |
|---|---|---|
| 本轮主评测 | 14 个条件 × 4 套件 × 10 任务 × 50 官方初始状态（0–49） | 28,000 条；VLM 裁剪与 D7/D5/D3 全量对比 |
| Action Expert 单层初筛 | L2 / L8 × 4 套件 × 10 任务 × 10 官方初始状态（0–9） | 800 条；仅与 Full18 相同 0–9 子集配对，不能称 500 条全量 |
| 本轮 Action Expert 离线结构探针 | 40 任务 baseline init0、110 真实阶段观测 × 3 噪声种子 | 有目的阶段采样，不是均匀随机；阶段/噪声不作为独立任务 |
| 既有 VLM 整层探针 | 3 个诊断任务、12 个真实观测 | 用于候选提出；不是本轮四套件全面 VLM 表征采样 |
| 既有 Attention / FFN 探针 | LIBERO-90 8 任务 + LIBERO-10 4 任务，64 观测 × 3 种子 | 诊断来源，不能写成本轮标准四套件子模块全量结果 |

本轮统一使用训练后权重、Triton、官方初始状态、环境 seed=7、warmup=10、replan=12、强制每轮完整推理。VLM 组固定完整 Action Expert / D10；降噪组固定完整 VLM / 完整专家层；SkipAE-L2/L8 固定完整 VLM / D10，仅跳过专家的一层。比较组共享动作噪声种子与 episode/replan 噪声生成规则。

V16 跳过 L9/L10；V14 跳过 L8–L11；V12 跳过 L7–L12；V10-Mid 保留 L1–L4、L9/L10、L15–L18。V13-Custom 保留 L1/L2/L5/L6/L8/L9/L10/L11/L13/L14/L16/L17/L18，跳过 L3/L4/L7/L12/L15。

主评测按 task/init 键核查完整矩阵，重试取最新完整 attempt，未完成或重复 attempt 不计入分母。V13 曾遗漏 0–9 的 400 条，现已补齐。首次配对推理 noise hash 无不一致。此次进一步逐行读取全部 28,800 条 telemetry：验证 step=0…N、N+1 状态、有限 XYZ、配对初始 XYZ 一致。视频不是全部人工观看：原核查抽检 56 个视频头、8 个全帧解码；报告配对视频另行解码验证结果见可复核文件。

统计：Wilson 95% 只描述单格成功率；相对 baseline 的主要区间先计算每任务配对差，再以任务为单位 bootstrap 10,000 次（每套件 10 任务）。McNemar 是未多重校正的 episode 配对描述性检验，任务内相关与多重比较使其不适合单独作为最终结论。这里没有预注册非劣界限，不能声称“已证明无损”。

## 2. 成功率与 baseline 对比

| 变体 | Object | Spatial | Goal | LIBERO-10 | 总体 |
|---|---:|---:|---:|---:|---:|
| Full18 | 98.0% (490/500) | 95.2% (476/500) | 96.6% (483/500) | 87.8% (439/500) | 94.40% (1888/2000) |
| Skip-L2 | 97.4% (487/500) | 95.8% (479/500) | 94.0% (470/500) | 83.6% (418/500) | 92.70% (1854/2000) |
| Skip-L3 | 97.2% (486/500) | 96.4% (482/500) | 93.0% (465/500) | 87.4% (437/500) | 93.50% (1870/2000) |
| Skip-L4 | 97.2% (486/500) | 97.4% (487/500) | 92.4% (462/500) | 83.4% (417/500) | 92.60% (1852/2000) |
| Skip-L5 | 97.8% (489/500) | 97.2% (486/500) | 93.6% (468/500) | 82.6% (413/500) | 92.80% (1856/2000) |
| Skip-L2-5 | 61.6% (308/500) | 43.8% (219/500) | 13.2% (66/500) | 25.8% (129/500) | 36.10% (722/2000) |
| V13-Custom | 71.0% (355/500) | 50.6% (253/500) | 24.4% (122/500) | 29.0% (145/500) | 43.75% (875/2000) |
| V16 | 98.4% (492/500) | 95.4% (477/500) | 87.4% (437/500) | 78.2% (391/500) | 89.85% (1797/2000) |
| V14 | 81.8% (409/500) | 65.4% (327/500) | 32.6% (163/500) | 18.4% (92/500) | 49.55% (991/2000) |
| V12 | 0.0% (0/500) | 0.0% (0/500) | 0.6% (3/500) | 0.0% (0/500) | 0.15% (3/2000) |
| V10-Mid | 0.0% (0/500) | 0.0% (0/500) | 0.0% (0/500) | 0.0% (0/500) | 0.00% (0/2000) |
| AE-D7 | 97.8% (489/500) | 95.2% (476/500) | 95.6% (478/500) | 87.0% (435/500) | 93.90% (1878/2000) |
| AE-D5 | 99.0% (495/500) | 97.0% (485/500) | 96.6% (483/500) | 86.8% (434/500) | 94.85% (1897/2000) |
| AE-D3 | 98.2% (491/500) | 97.6% (488/500) | 97.0% (485/500) | 86.8% (434/500) | 94.90% (1898/2000) |
| SkipAE-L2 | 99.0% (99/100) | 96.0% (96/100) | 81.0% (81/100) | 80.0% (80/100) | 89.00% (356/400) |
| SkipAE-L8 | 97.0% (97/100) | 93.0% (93/100) | 84.0% (84/100) | 80.0% (80/100) | 88.50% (354/400) |

![成功率热图](results/experiment9_standard_vlm_pruning_20261004/analysis/final/success_heatmap.png)

### 2.1 同任务、同初始状态的配对损失

| 变体 | 套件 | 对照成功 / 配对数 | 差值 pp | 任务聚类 95%（pp） | 新失败 / 恢复 |
|---|---|---:|---:|---:|---:|
| Skip-L2 | libero_object | 490/500 | -0.6 | [-2.0, +0.6] | 11/8 |
| Skip-L2 | libero_spatial | 476/500 | +0.6 | [-2.6, +3.6] | 17/20 |
| Skip-L2 | libero_goal | 483/500 | -2.6 | [-5.0, -0.2] | 24/11 |
| Skip-L2 | libero_10 | 439/500 | -4.2 | [-8.4, -0.0] | 51/30 |
| Skip-L3 | libero_object | 490/500 | -0.8 | [-1.8, +0.2] | 11/7 |
| Skip-L3 | libero_spatial | 476/500 | +1.2 | [-1.8, +4.0] | 13/19 |
| Skip-L3 | libero_goal | 483/500 | -3.6 | [-7.2, -0.8] | 27/9 |
| Skip-L3 | libero_10 | 439/500 | -0.4 | [-2.6, +1.8] | 35/33 |
| Skip-L4 | libero_object | 490/500 | -0.8 | [-2.8, +1.0] | 12/8 |
| Skip-L4 | libero_spatial | 476/500 | +2.2 | [+0.4, +4.0] | 10/21 |
| Skip-L4 | libero_goal | 483/500 | -4.2 | [-8.8, -0.6] | 33/12 |
| Skip-L4 | libero_10 | 439/500 | -4.4 | [-9.4, -0.0] | 53/31 |
| Skip-L5 | libero_object | 490/500 | -0.2 | [-2.2, +2.0] | 8/7 |
| Skip-L5 | libero_spatial | 476/500 | +2.0 | [-0.0, +4.0] | 11/21 |
| Skip-L5 | libero_goal | 483/500 | -3.0 | [-5.8, -0.4] | 26/11 |
| Skip-L5 | libero_10 | 439/500 | -5.2 | [-7.8, -2.4] | 57/31 |
| Skip-L2-5 | libero_object | 490/500 | -36.4 | [-54.0, -21.0] | 186/4 |
| Skip-L2-5 | libero_spatial | 476/500 | -51.4 | [-71.8, -30.2] | 267/10 |
| Skip-L2-5 | libero_goal | 483/500 | -83.4 | [-92.4, -70.6] | 418/1 |
| Skip-L2-5 | libero_10 | 439/500 | -62.0 | [-73.2, -50.8] | 315/5 |
| V13-Custom | libero_object | 490/500 | -27.0 | [-37.4, -16.2] | 144/9 |
| V13-Custom | libero_spatial | 476/500 | -44.6 | [-64.4, -25.0] | 232/9 |
| V13-Custom | libero_goal | 483/500 | -72.2 | [-91.0, -49.8] | 363/2 |
| V13-Custom | libero_10 | 439/500 | -58.8 | [-72.6, -45.4] | 297/3 |
| V16 | libero_object | 490/500 | +0.4 | [-1.0, +2.0] | 7/9 |
| V16 | libero_spatial | 476/500 | +0.2 | [-2.4, +3.0] | 18/19 |
| V16 | libero_goal | 483/500 | -9.2 | [-17.8, -2.6] | 59/13 |
| V16 | libero_10 | 439/500 | -9.6 | [-15.4, -4.4] | 75/27 |
| V14 | libero_object | 490/500 | -16.2 | [-25.8, -8.0] | 88/7 |
| V14 | libero_spatial | 476/500 | -29.8 | [-43.8, -16.8] | 163/14 |
| V14 | libero_goal | 483/500 | -64.0 | [-81.2, -46.4] | 321/1 |
| V14 | libero_10 | 439/500 | -69.4 | [-79.0, -60.0] | 348/1 |
| V12 | libero_object | 490/500 | -98.0 | [-99.2, -96.4] | 490/0 |
| V12 | libero_spatial | 476/500 | -95.2 | [-97.4, -93.0] | 476/0 |
| V12 | libero_goal | 483/500 | -96.0 | [-97.8, -94.0] | 480/0 |
| V12 | libero_10 | 439/500 | -87.8 | [-96.0, -77.0] | 439/0 |
| V10-Mid | libero_object | 490/500 | -98.0 | [-99.2, -96.4] | 490/0 |
| V10-Mid | libero_spatial | 476/500 | -95.2 | [-97.4, -93.0] | 476/0 |
| V10-Mid | libero_goal | 483/500 | -96.6 | [-97.8, -95.2] | 483/0 |
| V10-Mid | libero_10 | 439/500 | -87.8 | [-96.2, -77.0] | 439/0 |
| AE-D7 | libero_object | 490/500 | -0.2 | [-1.6, +0.8] | 7/6 |
| AE-D7 | libero_spatial | 476/500 | +0.0 | [-1.6, +1.8] | 17/17 |
| AE-D7 | libero_goal | 483/500 | -1.0 | [-3.0, +1.0] | 16/11 |
| AE-D7 | libero_10 | 439/500 | -0.8 | [-3.6, +2.2] | 39/35 |
| AE-D5 | libero_object | 490/500 | +1.0 | [-0.4, +2.8] | 3/8 |
| AE-D5 | libero_spatial | 476/500 | +1.8 | [+0.0, +3.6] | 11/20 |
| AE-D5 | libero_goal | 483/500 | +0.0 | [-1.0, +1.0] | 11/11 |
| AE-D5 | libero_10 | 439/500 | -1.0 | [-2.6, +0.8] | 41/36 |
| AE-D3 | libero_object | 490/500 | +0.2 | [-1.4, +1.8] | 8/9 |
| AE-D3 | libero_spatial | 476/500 | +2.4 | [+0.6, +4.2] | 10/22 |
| AE-D3 | libero_goal | 483/500 | +0.4 | [-1.6, +2.4] | 13/15 |
| AE-D3 | libero_10 | 439/500 | -1.0 | [-2.6, +1.0] | 39/34 |
| SkipAE-L2 | libero_object | 97/100 | +2.0 | [+0.0, +5.0] | 0/2 |
| SkipAE-L2 | libero_spatial | 94/100 | +2.0 | [-2.0, +7.0] | 3/5 |
| SkipAE-L2 | libero_goal | 98/100 | -17.0 | [-38.0, -3.0] | 19/2 |
| SkipAE-L2 | libero_10 | 89/100 | -9.0 | [-23.0, +2.0] | 13/4 |
| SkipAE-L8 | libero_object | 97/100 | +0.0 | [-3.0, +3.0] | 2/2 |
| SkipAE-L8 | libero_spatial | 94/100 | -1.0 | [-5.0, +3.0] | 5/4 |
| SkipAE-L8 | libero_goal | 98/100 | -14.0 | [-36.0, +2.0] | 16/2 |
| SkipAE-L8 | libero_10 | 89/100 | -9.0 | [-22.0, +3.0] | 15/6 |

L3 全量总体为 1,870/2,000（93.5%），相对 baseline −0.9 pp；L2/L4/L5 总体分别 92.7% / 92.6% / 92.8%。即使 Object/Spatial 都较高，Goal 的单层下降仍为 2.6–4.2 pp，L2/L4/L5 的 LIBERO-10 下降约 4.2–5.2 pp。L3 是当前完整 block 候选里更值得继续验证的一个，但不能据此宣布跨任务无损。

### 2.2 哪些变体达到 90%？

这里的 ≥90% 指观测点估计，不能把它与达到 baseline 混为一谈。

| 变体 | 点估计 ≥90% 的套件 | Wilson 下界也 ≥90% 的套件 |
|---|---|---|
| Full18 | object, spatial, goal | object, spatial, goal |
| Skip-L2 | object, spatial, goal | object, spatial, goal |
| Skip-L3 | object, spatial, goal | object, spatial, goal |
| Skip-L4 | object, spatial, goal | object, spatial |
| Skip-L5 | object, spatial, goal | object, spatial, goal |
| Skip-L2-5 | 无 | 无 |
| V13-Custom | 无 | 无 |
| V16 | object, spatial | object, spatial |
| V14 | 无 | 无 |
| V12 | 无 | 无 |
| V10-Mid | 无 | 无 |
| AE-D7 | object, spatial, goal | object, spatial, goal |
| AE-D5 | object, spatial, goal | object, spatial, goal |
| AE-D3 | object, spatial, goal | object, spatial, goal |
| SkipAE-L2 | object, spatial | object, spatial |
| SkipAE-L8 | object, spatial | object |

LIBERO-10 baseline 本身是 87.8%，所以“90% 门槛”不能作为唯一筛选规则；应比较同协议 baseline 的配对变化。套件平均会掩盖任务间差异，逐任务表已附：优先检查 Goal 中下降最大的任务，而不是只挑高成功率套件。

### 2.3 退化最明显的任务（每变体前三个，主评测）

| 变体 | 套件 / task | 变体 / baseline | 差值 pp | 任务 |
|---|---|---:|---:|---|
| Skip-L2 | libero_10 / 8 | 32% / 48% | -16 | put both moka pots on the stove |
| Skip-L2 | libero_10 / 2 | 84% / 98% | -14 | turn on the stove and put the moka pot on it |
| Skip-L2 | libero_spatial / 6 | 90% / 100% | -10 | pick up the black bowl next to the cookie box and place it on the plate |
| Skip-L3 | libero_goal / 9 | 78% / 96% | -18 | put the wine bottle on the rack |
| Skip-L3 | libero_spatial / 6 | 92% / 100% | -8 | pick up the black bowl next to the cookie box and place it on the plate |
| Skip-L3 | libero_goal / 4 | 90% / 98% | -8 | put the bowl on top of the cabinet |
| Skip-L4 | libero_goal / 9 | 74% / 96% | -22 | put the wine bottle on the rack |
| Skip-L4 | libero_10 / 5 | 74% / 96% | -22 | pick up the book and place it in the back compartment of the caddy |
| Skip-L4 | libero_10 / 2 | 86% / 98% | -12 | turn on the stove and put the moka pot on it |
| Skip-L5 | libero_goal / 9 | 84% / 96% | -12 | put the wine bottle on the rack |
| Skip-L5 | libero_10 / 2 | 86% / 98% | -12 | turn on the stove and put the moka pot on it |
| Skip-L5 | libero_10 / 8 | 38% / 48% | -10 | put both moka pots on the stove |
| Skip-L2-5 | libero_goal / 2 | 2% / 100% | -98 | put the wine bottle on top of the cabinet |
| Skip-L2-5 | libero_object / 9 | 2% / 98% | -96 | pick up the orange juice and place it in the basket |
| Skip-L2-5 | libero_goal / 9 | 0% / 96% | -96 | put the wine bottle on the rack |
| V13-Custom | libero_goal / 0 | 0% / 96% | -96 | open the middle drawer of the cabinet |
| V13-Custom | libero_goal / 9 | 0% / 96% | -96 | put the wine bottle on the rack |
| V13-Custom | libero_10 / 3 | 2% / 98% | -96 | put the black bowl in the bottom drawer of the cabinet and close it |
| V16 | libero_goal / 3 | 52% / 92% | -40 | open the top drawer and put the bowl inside |
| V16 | libero_10 / 8 | 18% / 48% | -30 | put both moka pots on the stove |
| V16 | libero_goal / 0 | 78% / 96% | -18 | open the middle drawer of the cabinet |
| V14 | libero_goal / 0 | 0% / 96% | -96 | open the middle drawer of the cabinet |
| V14 | libero_goal / 9 | 0% / 96% | -96 | put the wine bottle on the rack |
| V14 | libero_10 / 2 | 2% / 98% | -96 | turn on the stove and put the moka pot on it |
| V12 | libero_object / 1 | 0% / 100% | -100 | pick up the cream cheese and place it in the basket |
| V12 | libero_object / 2 | 0% / 100% | -100 | pick up the salad dressing and place it in the basket |
| V12 | libero_object / 6 | 0% / 100% | -100 | pick up the butter and place it in the basket |
| V10-Mid | libero_object / 1 | 0% / 100% | -100 | pick up the cream cheese and place it in the basket |
| V10-Mid | libero_object / 2 | 0% / 100% | -100 | pick up the salad dressing and place it in the basket |
| V10-Mid | libero_object / 6 | 0% / 100% | -100 | pick up the butter and place it in the basket |
| AE-D7 | libero_10 / 2 | 90% / 98% | -8 | turn on the stove and put the moka pot on it |
| AE-D7 | libero_10 / 6 | 74% / 80% | -6 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| AE-D7 | libero_object / 4 | 92% / 98% | -6 | pick up the ketchup and place it in the basket |
| AE-D5 | libero_10 / 2 | 92% / 98% | -6 | turn on the stove and put the moka pot on it |
| AE-D5 | libero_10 / 1 | 96% / 100% | -4 | put both the cream cheese box and the butter in the basket |
| AE-D5 | libero_object / 2 | 98% / 100% | -2 | pick up the salad dressing and place it in the basket |
| AE-D3 | libero_goal / 4 | 92% / 98% | -6 | put the bowl on top of the cabinet |
| AE-D3 | libero_object / 3 | 94% / 98% | -4 | pick up the bbq sauce and place it in the basket |
| AE-D3 | libero_object / 9 | 94% / 98% | -4 | pick up the orange juice and place it in the basket |
| SkipAE-L2 | libero_goal / 9 | 0% / 100% | -100 | put the wine bottle on the rack |
| SkipAE-L2 | libero_10 / 8 | 0% / 60% | -60 | put both moka pots on the stove |
| SkipAE-L2 | libero_goal / 5 | 70% / 100% | -30 | push the plate to the front of the stove |
| SkipAE-L8 | libero_goal / 9 | 0% / 100% | -100 | put the wine bottle on the rack |
| SkipAE-L8 | libero_10 / 8 | 10% / 60% | -50 | put both moka pots on the stove |
| SkipAE-L8 | libero_goal / 5 | 60% / 100% | -40 | push the plate to the front of the stove |

## 3. VLM 层间相似度：来源、测法与裁剪证据

**不是随机初始化模型或随机图像。** 既有 VLM 探针使用训练好的权重，来自诊断 rollout 的真实双相机图像、状态、语言指令，按语义阶段有目的选取。已生成整层分析纳入的是 12 观测 / 3 任务；采集 manifest 后续新增到 13 条，不应改写旧产物的样本量。本轮新增的是这些候选的标准闭环评测，未重采四套件 VLM 全层探针，需明确这个证据范围。

记录 H0 输入和 H1…H18 输出，对同一 token 算 `dot(H_i,H_j)/(||H_i||·||H_j||)`，先按 agent-view / wrist-view / language 模态的 token 平均，再跨观测汇总。agent 与 wrist 各 256 token，语言长度变化；避免视觉 token 数量掩盖语言变化。同时记录 `||H_l−H_(l−1)||_F/||H_(l−1)||_F`、中心化 linear CKA，以及单层跳过的动作敏感性。

![VLM 全层对余弦热图](results/pi0_vlm_pruning_diagnosis/layer_analysis/layer_similarity_heatmap.png)

| VLM 层 | Agent 相邻余弦 | Wrist 相邻余弦 | Language 相邻余弦 | Agent 相对残差 |
|---|---:|---:|---:|---:|
| L1 | 0.239344 | 0.236194 | 0.267550 | 2.216180 |
| L2 | 0.998430 | 0.998244 | 0.992822 | 0.067591 |
| L3 | 0.999135 | 0.999039 | 0.995635 | 0.049779 |
| L4 | 0.998656 | 0.998501 | 0.994921 | 0.063518 |
| L5 | 0.998773 | 0.998609 | 0.995742 | 0.059416 |
| L6 | 0.998505 | 0.998364 | 0.993882 | 0.081672 |
| L7 | 0.997672 | 0.996964 | 0.992604 | 0.080673 |
| L8 | 0.998114 | 0.997630 | 0.985144 | 0.071811 |
| L9 | 0.997231 | 0.996077 | 0.991778 | 0.091415 |
| L10 | 0.996705 | 0.995980 | 0.990431 | 0.097384 |
| L11 | 0.994738 | 0.994618 | 0.986423 | 0.122870 |
| L12 | 0.989708 | 0.990885 | 0.978691 | 0.228489 |
| L13 | 0.978241 | 0.965192 | 0.977002 | 0.293577 |
| L14 | 0.950309 | 0.949915 | 0.959052 | 0.430526 |
| L15 | 0.967298 | 0.961340 | 0.968035 | 0.871537 |
| L16 | 0.951933 | 0.954720 | 0.931991 | 0.230195 |
| L17 | 0.579082 | 0.641470 | 0.727980 | 2.018403 |
| L18 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |

L2–L5 的 agent-view 相邻余弦约 0.9984–0.9991、残差约 0.05–0.07；既有固定观测单层跳过 NMSE 分别约 0.0083、0.0091、0.0113、0.0159。这些证据支持“单层候选”，本轮闭环进一步支持优先 L3、按套件评估其他三层。**联合删除 L2–L5 的标准成功率大幅下降，已直接反驳候选层的线性可组合性。**

L1/L17 变化大，优先保留。L12 以后多层残差增大；V12/V14 结果也不支持当前按中间层连续删除。L18 的余弦=1、残差=0 是当前 prefix 实现只生成 KV、未继续更新 hidden 的行为，并非 KV 无用，不能当成安全删除依据。

## 4. Layer 内 Attention 与 FFN

已有扩展探针是 12 个诊断任务（90 套件 8 个、10 套件 4 个）、64 个真实阶段观测、3 噪声种子，192 个观测×种子。记录 Attention 前、残差相加后、FFN 残差相加后的 hidden；分别测两个子模块的方向/幅度更新。Attention 干预保留 Q/K/V，跳过输出投影更新；FFN 干预仅跳过 FFN 残差更新，不能与整层删除混同。L18 KV-only 不作为普通子模块候选。

![Attention/FFN 相邻余弦](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_cosine_by_layer.png)

![子模块动作敏感性](results/pi0_attention_ffn_diagnosis/analysis_expanded_12task_three_seeds/submodule_action_sensitivity.png)

| 子模块 | 前 12 动作 NMSE | 任务聚类 95% | 相对残差（候选汇总口径） |
|---|---:|---:|---:|
| ATTN L16 | 0.024081 | [0.001602, 0.068379] | 0.101056 |
| ATTN L15 | 0.090413 | [0.003949, 0.198733] | 0.199766 |
| FFN L2 | 0.513863 | [0.318175, 0.716424] | 0.086963 |
| FFN L16 | 0.248535 | [0.098264, 0.406947] | 0.344920 |

**Attention L16 是后续子模块验证候选，不是已通过标准套件的结论。** FFN L2 隐藏更新较小，但前 12 动作 NMSE 达 0.514，表明下游对某些小更新很敏感。Attention/FFN 不应共用一个裁剪结论。当前这一轮没有 Attention-L16 / FFN-L2 标准四套件新闭环评测，这是下一轮可选补充，不是把本轮已有整层变体漏记。

## 5. 分阶段失败：一开始偏了，还是精细操作失败？

阶段标签来自真实仿真器 EEF/物体位姿、接触、抓取和 goal predicates，不是相机截图猜测。contact 条件包含真实接触 **或** EEF 距相关物体 ≤6 cm；approach 使用 ≤15 cm 或相对初始缩短 ≥4 cm；抓取后物体位移 ≥3 cm 标记 transport；目标距离 ≤10 cm 或 goal progress 提升标记 alignment。因此下面的“接触前”严格说是 **contact-or-near 阶段之前**，不能解读为经过人工验证的物理接触事件。最高到达阶段是定位线索，不是因果故障标签。

为避免把 baseline 原有失败算给裁剪，主图只统计 **baseline 成功、变体失败**。另附 `failure_stage_counts.csv` 中的 all_failures，保留完整失败分母。

![裁剪新增失败阶段](results/experiment9_standard_vlm_pruning_20261004/analysis/final/new_failure_stages.png)

| 变体（汇总四套件） | 新失败数 | 接触/靠近前 | 已接触/靠近但未抓取 | 抓取/操作或更后 |
|---|---:|---:|---:|---:|
| Skip-L2 | 103 | 6 (5.8%) | 3 (2.9%) | 94 (91.3%) |
| Skip-L3 | 86 | 6 (7.0%) | 3 (3.5%) | 77 (89.5%) |
| Skip-L4 | 108 | 6 (5.6%) | 2 (1.9%) | 100 (92.6%) |
| Skip-L5 | 102 | 4 (3.9%) | 4 (3.9%) | 94 (92.2%) |
| Skip-L2-5 | 1186 | 458 (38.6%) | 36 (3.0%) | 692 (58.3%) |
| V13-Custom | 1036 | 398 (38.4%) | 16 (1.5%) | 622 (60.0%) |
| V16 | 159 | 15 (9.4%) | 9 (5.7%) | 135 (84.9%) |
| V14 | 920 | 151 (16.4%) | 18 (2.0%) | 751 (81.6%) |
| V12 | 1885 | 1019 (54.1%) | 46 (2.4%) | 820 (43.5%) |
| V10-Mid | 1888 | 1232 (65.3%) | 16 (0.8%) | 640 (33.9%) |
| AE-D7 | 79 | 2 (2.5%) | 1 (1.3%) | 76 (96.2%) |
| AE-D5 | 66 | 3 (4.5%) | 1 (1.5%) | 62 (93.9%) |
| AE-D3 | 70 | 3 (4.3%) | 1 (1.4%) | 66 (94.3%) |
| SkipAE-L2 | 35 | 1 (2.9%) | 2 (5.7%) | 32 (91.4%) |
| SkipAE-L8 | 38 | 1 (2.6%) | 3 (7.9%) | 34 (89.5%) |

**阶段证据支持但限定你的精细操作假设：** L3 的 86 个新增失败中，77 个（89.5%）曾进入抓取/操作或更后阶段；专家 L2 的 35 个新增失败中有 32 个（91.4%）进入该阶段。轻度单层干预的失败较集中在后段。不过这不等于它们前段轨迹完全正确。

不能从“末端轨迹的最高阶段较晚”推出前段没有偏差；有些 episode 早早偏离，但仍碰到物体后才最终失败。为拆开这两点，下面追加首次持续轨迹偏离的时点分析。

### 5.1 首次明显轨迹偏离

以同环境 step 的 `||EEF_variant−EEF_baseline||₂` 为准，连续 **5 个状态超过 2 cm** 视为一次持续偏离起点；同时保存 1/2/5 cm 阈值以便敏感性复核。只比较两个轨迹共同存在的时间，不把 baseline 提前成功后的末端保持当作实测轨迹。分类用偏离时 baseline 的阶段，避免变体本身阶段错乱造成解释偏差。

| 变体 | 新失败配对数 | 共同时间内出现持续 2 cm 偏差 | 首偏差在 baseline 接触/靠近前 | 首偏差在接触/靠近或之后 |
|---|---:|---:|---:|---:|
| Skip-L2 | 103 | 102 | 30 | 72 |
| Skip-L3 | 86 | 86 | 32 | 54 |
| Skip-L4 | 108 | 107 | 58 | 49 |
| Skip-L5 | 102 | 100 | 43 | 57 |
| Skip-L2-5 | 1186 | 1186 | 1180 | 6 |
| V13-Custom | 1036 | 1036 | 1035 | 1 |
| V16 | 159 | 159 | 141 | 18 |
| V14 | 920 | 920 | 905 | 15 |
| V12 | 1885 | 1885 | 1885 | 0 |
| V10-Mid | 1888 | 1888 | 1888 | 0 |
| AE-D7 | 79 | 76 | 11 | 65 |
| AE-D5 | 66 | 65 | 12 | 53 |
| AE-D3 | 70 | 70 | 19 | 51 |
| SkipAE-L2 | 35 | 33 | 16 | 17 |
| SkipAE-L8 | 38 | 38 | 19 | 19 |

**多层裁剪不是只在精细操作才出问题：** Skip-L2-5 的 1,186 个新增失败中，1,180 个（99.5%）在 baseline 的接触/靠近阶段之前就出现持续 2 cm 偏离；V13-Custom 是 1,035/1,036（99.9%）；V12 是 1,885/1,885。相比之下，L3 为 32/86（37.2%）。因此多层变体即使最终碰到物体，轨迹也往往已在早期改变，而轻度单层干预更值得做后段精细控制分析。此阈值定义的是与 baseline 的差异，不是任意差异都必然是错误。

这两张表应联合解释：首次偏离时点回答“何时开始与 baseline 不同”，最高阶段回答“最终推进到哪里”；后者无法直接证明失败由夹爪精度造成。NMSE 的 translation 通道是控制命令误差，不能把它写成实测 EEF 米数。

## 6. 真实 XYZ 轨迹、偏差与配对视频

**绘图使用 Matplotlib**。数据是逐 step 保存的 MuJoCo 世界坐标 EEF XYZ（米），不是 action 命令积分，也不是相机帧。蓝色半透明实线+圆点为 Full18，橙色半透明虚线+三角点为变体；包含 3D、XY/XZ/YZ 投影及偏差曲线，重叠时仍可分辨。

每个变体×套件选一个案例：优先新增失败，按该 cohort 的平均 XYZ 偏差取中位案例，避免只展示最极端样本；无新增失败时选双成功，否则其他配对。全部配对的数值统计仍覆盖全量，不由展示案例决定。共 60 张图，每个变体另有一段 LIBERO-10 同步配对视频（15 段），其余案例也链接原始双方视频。

[打开四套件 XYZ 图与视频画廊](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_gallery.html)

| 变体 | 全配对数 | 平均偏差的 episode 中位数（m） | episode p95 偏差的中位数（m） |
|---|---:|---:|---:|
| Skip-L2 | 2000 | 0.0213 | 0.0505 |
| Skip-L3 | 2000 | 0.0204 | 0.0468 |
| Skip-L4 | 2000 | 0.0223 | 0.0494 |
| Skip-L5 | 2000 | 0.0202 | 0.0468 |
| Skip-L2-5 | 2000 | 0.1476 | 0.2896 |
| V13-Custom | 2000 | 0.1323 | 0.2664 |
| V16 | 2000 | 0.0314 | 0.0669 |
| V14 | 2000 | 0.0870 | 0.1870 |
| V12 | 2000 | 0.1777 | 0.3231 |
| V10-Mid | 2000 | 0.1990 | 0.3283 |
| AE-D7 | 2000 | 0.0103 | 0.0276 |
| AE-D5 | 2000 | 0.0138 | 0.0354 |
| AE-D3 | 2000 | 0.0186 | 0.0441 |
| SkipAE-L2 | 400 | 0.0135 | 0.0369 |
| SkipAE-L8 | 400 | 0.0203 | 0.0510 |

这里的差异是 **时间同步轨迹差异**：同一几何路径如果执行速度不同，也会有非零偏差；没有做动态时间规整，不能单凭该指标称路径几何错误。不同轨迹长度只用重叠窗口，较晚失败的独有尾段不进入配对距离，但会在图/视频中显示。

### 6.1 每个变体的 LIBERO-10 示例

| 变体 | task / init | 配对结果 | mean / p95 / max（m） | 图 | 同步视频 |
|---|---|---|---:|---|---|
| Skip-L2 | 9/28 | new_failure | 0.076/0.280/0.307 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L2__libero_10__task09_ep28.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/Skip_L2__libero_10__task09_ep28.mp4) |
| Skip-L3 | 4/40 | new_failure | 0.060/0.188/0.190 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L3__libero_10__task04_ep40.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/Skip_L3__libero_10__task04_ep40.mp4) |
| Skip-L4 | 2/39 | new_failure | 0.048/0.133/0.145 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L4__libero_10__task02_ep39.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/Skip_L4__libero_10__task02_ep39.mp4) |
| Skip-L5 | 3/18 | new_failure | 0.062/0.168/0.189 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L5__libero_10__task03_ep18.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/Skip_L5__libero_10__task03_ep18.mp4) |
| Skip-L2-5 | 0/24 | new_failure | 0.141/0.241/0.251 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L2_5__libero_10__task00_ep24.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/Skip_L2_5__libero_10__task00_ep24.mp4) |
| V13-Custom | 5/28 | new_failure | 0.150/0.269/0.289 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/V13_Custom__libero_10__task05_ep28.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/V13_Custom__libero_10__task05_ep28.mp4) |
| V16 | 9/16 | new_failure | 0.102/0.298/0.325 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/V16__libero_10__task09_ep16.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/V16__libero_10__task09_ep16.mp4) |
| V14 | 4/26 | new_failure | 0.168/0.489/0.521 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/V14__libero_10__task04_ep26.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/V14__libero_10__task04_ep26.mp4) |
| V12 | 5/7 | new_failure | 0.186/0.312/0.318 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/V12__libero_10__task05_ep07.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/V12__libero_10__task05_ep07.mp4) |
| V10-Mid | 4/15 | new_failure | 0.197/0.304/0.322 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/V10_Mid__libero_10__task04_ep15.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/V10_Mid__libero_10__task04_ep15.mp4) |
| AE-D7 | 6/33 | new_failure | 0.060/0.179/0.234 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/AE_D7__libero_10__task06_ep33.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/AE_D7__libero_10__task06_ep33.mp4) |
| AE-D5 | 8/1 | new_failure | 0.065/0.338/0.353 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/AE_D5__libero_10__task08_ep01.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/AE_D5__libero_10__task08_ep01.mp4) |
| AE-D3 | 6/35 | new_failure | 0.058/0.167/0.223 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/AE_D3__libero_10__task06_ep35.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/AE_D3__libero_10__task06_ep35.mp4) |
| SkipAE-L2 | 8/1 | new_failure | 0.083/0.408/0.451 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/SkipAE_L2__libero_10__task08_ep01.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/SkipAE_L2__libero_10__task08_ep01.mp4) |
| SkipAE-L8 | 9/1 | new_failure | 0.097/0.396/0.424 | [XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/SkipAE_L8__libero_10__task09_ep01.png) | [MP4](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_videos/SkipAE_L8__libero_10__task09_ep01.mp4) |

#### Skip-L3：task 4 / init 40

![Skip-L3 配对 XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L3__libero_10__task04_ep40.png)

#### Skip-L2-5：task 0 / init 24

![Skip-L2-5 配对 XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/Skip_L2_5__libero_10__task00_ep24.png)

#### V13-Custom：task 5 / init 28

![V13-Custom 配对 XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/V13_Custom__libero_10__task05_ep28.png)

#### AE-D3：task 6 / init 35

![AE-D3 配对 XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/AE_D3__libero_10__task06_ep35.png)

#### SkipAE-L2：task 8 / init 1

![SkipAE-L2 配对 XYZ](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_plots/SkipAE_L2__libero_10__task08_ep01.png)

## 7. Action Expert 层探针与闭环反证

从本轮 Full18 每任务官方 init0 轨迹重放 40 条，保存 initial/contact/goal_alignment 的真实观测，共 110 条（未出现阶段显式记录，不伪造）。重放每一步 EEF 与原轨迹一致，success 标签匹配。动作噪声 12345/23456/34567 固定；baseline 与逐层删除干预用相同输入、相同噪声。整个去噪过程中都删除同一专家层，测动作 token（排除 state token）的 H0…H18，记录 flow steps 0/2/5/9，即 t=1/.8/.5/.1。

隐藏余弦是同 token 的方向相似度；动作敏感性是反归一化后前 7 个控制通道，重点看实际 replan 执行的前 12 动作。NMSE=`mean((a_skip−a_base)^2)/max(mean(a_base^2),epsilon)`。在每任务内平均观测/噪声，再对 40 任务 bootstrap，不能把 330 个观测×噪声当成 330 个独立任务。

![Action expert hidden 与动作敏感性](results/experiment9_standard_vlm_pruning_20261004/analysis/final/action_expert_probe.png)

| 专家层 | 前 12 动作 NMSE（任务均值） | 95% 区间 | 夹爪符号翻转 | 初始 / 接触 / 对齐 NMSE | 筛选通过 |
|---|---:|---:|---:|---|---|
| L1 | 1.430106 | [1.085573, 1.847004] | 9.838% | 0.795598 / 1.736980 / 1.884870 | 否 |
| L2 | 0.002590 | [0.000335, 0.005415] | 0.069% | 0.000191 / 0.005260 / 0.003267 | 是 |
| L3 | 0.128820 | [0.093028, 0.164478] | 3.600% | 0.002049 / 0.330408 / 0.104153 | 否 |
| L4 | 0.017944 | [0.009916, 0.027132] | 0.486% | 0.002104 / 0.038639 / 0.019965 | 否 |
| L5 | 0.089774 | [0.054140, 0.136534] | 0.845% | 0.010225 / 0.149466 / 0.124635 | 否 |
| L6 | 0.010877 | [0.005094, 0.018118] | 0.231% | 0.002347 / 0.023912 / 0.009741 | 否 |
| L7 | 0.010926 | [0.005653, 0.017306] | 0.255% | 0.002503 / 0.017444 / 0.016361 | 否 |
| L8 | 0.004087 | [0.001038, 0.008511] | 0.093% | 0.000826 / 0.009195 / 0.003554 | 是 |
| L9 | 0.014553 | [0.006716, 0.023305] | 0.417% | 0.001099 / 0.035730 / 0.012507 | 否 |
| L10 | 0.029910 | [0.016433, 0.044627] | 0.856% | 0.002328 / 0.054012 / 0.043697 | 否 |
| L11 | 0.028452 | [0.017324, 0.040735] | 0.856% | 0.001837 / 0.059383 / 0.034046 | 否 |
| L12 | 0.305638 | [0.208613, 0.426211] | 2.465% | 0.049217 / 0.393255 / 0.522428 | 否 |
| L13 | 0.056330 | [0.039354, 0.075078] | 1.551% | 0.003342 / 0.161412 / 0.026355 | 否 |
| L14 | 0.035407 | [0.020704, 0.054097] | 0.972% | 0.003117 / 0.086100 / 0.030913 | 否 |
| L15 | 0.085494 | [0.060760, 0.114477] | 2.292% | 0.017616 / 0.177276 / 0.085438 | 否 |
| L16 | 0.103756 | [0.073274, 0.141257] | 2.801% | 0.019054 / 0.270959 / 0.055516 | 否 |
| L17 | 0.140952 | [0.109798, 0.174703] | 2.569% | 0.054356 / 0.214501 / 0.195739 | 否 |
| L18 | 0.138363 | [0.115476, 0.164771] | 2.280% | 0.083697 / 0.221792 / 0.132897 | 否 |

预先门槛：任务均值 NMSE≤.01、95%上界≤.02、夹爪翻转≤1%、最坏阶段均值≤.05、每 suite×阶段至少 3 个任务；最多 2 候选。L2/L8 通过，随后标准初筛 Goal 的明显下降说明这套门槛只适合筛选，不是闭环安全判据。仅 init0、有目的阶段采样和未覆盖所有重规划输入都可能限制外推；误差积累是合理假设，但本轮没有隔离其因果贡献。

探针内部普通/带记录 baseline parity、干预结束后 baseline parity 均通过，权重 hash 未变。初次跨新实例对已保存动作的严格比对有约 0.0096 最大命令差；同实例 graph/eager 差为 0，完整 baseline 跨实例也有差。对两个真实 prompt 长度的编译预热后，6 个阶段×候选校验逐值一致。证据支持编译上下文影响复现；这不是已完整隔离的编译器根因。诊断保留于 `pattern_parity_failure.json` 和 `registered_pattern_parity.json`，未通过宽松阈值掩盖差异。

## 8. 时延与更优加速方向

每个 episode 排除 infer_id=0，再取其剩余 policy 调用的中位数；表内再按该条件×suite 的 episode 中位数取中位数。这样避免首轮编译和较长失败轨迹大量调用支配统计。policy 是外部调用耗时，total 是内部记录，不与端到端混同。

`action_denoise_ms` 原始字段为空；强制完整推理时代码把 staged decoder_ms 保存为 `full_fallback_ms`，此报告明确以该来源恢复 decoder 列（详见 `timing_sources.json`）。它不是重新跑出的新计时，不应无注明改名。多 GPU/不同时间段并行测量仍受硬件/负载影响，适合本轮观察而非专用无干扰加速基准。

| 变体 | 四套件 policy 中位数（ms，等权平均） | prefix 中位数（ms） | decoder 中位数（ms） | 内部 total 中位数（ms） |
|---|---:|---:|---:|---:|
| Full18 | 43.55 | 20.68 | 11.66 | 37.09 |
| Skip-L2 | 42.42 | 19.43 | 11.64 | 35.87 |
| Skip-L3 | 42.61 | 19.51 | 11.65 | 35.94 |
| Skip-L4 | 42.42 | 19.44 | 11.63 | 35.84 |
| Skip-L5 | 42.34 | 19.46 | 11.63 | 35.88 |
| Skip-L2-5 | 38.81 | 15.83 | 11.64 | 32.34 |
| V13-Custom | 38.08 | 14.59 | 11.63 | 30.99 |
| V16 | 41.25 | 18.31 | 11.67 | 34.72 |
| V14 | 38.75 | 15.84 | 11.63 | 32.33 |
| V12 | 36.66 | 13.41 | 11.62 | 29.92 |
| V10-Mid | 34.35 | 11.07 | 11.69 | 27.66 |
| AE-D7 | 40.23 | 20.69 | 8.16 | 33.54 |
| AE-D5 | 37.73 | 20.63 | 5.82 | 31.15 |
| AE-D3 | 35.72 | 20.59 | 3.51 | 28.71 |
| SkipAE-L2 | 42.61 | 20.55 | 11.00 | 36.24 |
| SkipAE-L8 | 42.60 | 20.52 | 10.99 | 36.23 |

本轮四套件等权时延观察：Full18 43.55 ms；D5 37.73 ms（下降 13.4%）；D3 35.72 ms（下降 18.0%）；单删 VLM L3 42.61 ms（下降 2.2%）。D3 的 decoder 大幅缩短但端到端收益仍受 prefix/encoder 等固定成本限制，不能把降噪步数 10→3 宣称为整模型 3.3 倍加速。

部署排序建议：**先 D5，验证收益后考虑 D3；其次 VLM 单删 L3 并按套件设回退；不直接部署联合 L2–L5 / V12 / V13-Custom；暂不把专家 L2/L8 删除作为跨套件通用优化。** 即使精简模型内部延迟下降，成功率低会增加重试成本；不能只按单次延迟排名。子模块 Attention L16 值得下一轮标准闭环验证，不能与已完成的专家 L2/L8 结果互相替代。

## 9. 历史 Control 21%：现在能说什么？

旧 Control 与本轮不是同协议：旧批次为 LIBERO-90+LIBERO-10 的 100 任务混合、冻结 HDF5 初态、D3 等设置；本轮为四标准套件官方初态/D10/当前完整实现。旧同一批 100 条改 D10 只提升到约 24%，所以“只是降噪步数少”不足以解释原来 21%。

**已确认：当前标准基线并不低，94.4%；仍未确认：旧 21% 的单一根因。** 当前高基线排除了将 21% 当作标准模型性能的解释，但不能证明是哪一项旧协议差异造成下降。若继续追旧根因，应固定旧任务/初态/噪声，每次只换一个因素；这轮没有新增该专项析因实验，报告不把相关性写成根因。

## 10. 产物与仍需补充的工作

- 主评测与条件式专家闭环初筛均完成，数据总量 28,800；不存在本轮主矩阵缺批。
- 全量失败阶段、任务成功率、配对 XYZ、60 张轨迹图和 15 段同步视频已生成。其余 45 个展示案例链接双方原始视频，并非为每个案例生成合成视频。
- 尚未完成的研究扩展：四标准套件的 VLM 全层/Attention/FFN 新探针；Attention-L16 标准闭环；专家 L2/L8 的 50 状态扩展（当前负向初筛不建议直接扩展）；旧 Control 的单因素根因审计。上述范围不能混成“本轮全量评测已漏跑”。
- 统计未做跨全部条件的多重校正、未做预注册非劣检验；当前报告用于候选判断与退化定位。

[套件与配对区间 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/final/suite_success.csv) · [逐任务 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/final/task_success.csv) · [失败阶段 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/final/failure_stage_counts.csv) · [阶段到达 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/final/stage_reach.csv)

[全部配对 XYZ 与首偏差 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/final/paired_xyz_all.csv) · [案例索引](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_case_index.csv) · [轨迹/视频画廊](results/experiment9_standard_vlm_pruning_20261004/analysis/final/xyz_gallery.html) · [图、视频及链接校验](results/experiment9_standard_vlm_pruning_20261004/analysis/final/artifact_validation.json)

[主矩阵核查](experiment9_completion_audit.md) · [AE 探针原始汇总](results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/probe/summary.json) · [AE 闭环原始汇总](results/experiment9_standard_vlm_pruning_20261004/action_expert_layer_probe/closed_loop/screening_summary.json)

复现：`.venv/bin/python scripts/experiment9/build_final_analysis.py`。报告的绘图/视频复用既有真实 XYZ 实现；生成的分析文件不作为模型训练输入。
