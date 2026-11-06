# E2决策机小规模调参结果

固定init8四suite×10task，六臂各40，总240。仅两个残差臂采集新指标；主成本统一predict+ack服务端wall，公用预热单列。该40例仅探索筛选，不证明部署优势。

| 条件 | 成功 | policy ms/action | predict-only | ack | warm s |
|---|---:|---:|---:|---:|---:|
| parent | 34/40 | 18.241 | 18.064 | 0.1768 | 32.67 |
| alternate | 38/40 | 16.337 | 16.159 | 0.1789 | 35.01 |
| old_c5 | 37/40 | 16.565 | 16.385 | 0.1795 | 33.81 |
| relaxed_abc | 35/40 | 16.049 | 15.869 | 0.1803 | 33.79 |
| error_abs | 37/40 | 16.444 | 16.239 | 0.2047 | 31.76 |
| error_relative | 36/40 | 16.861 | 16.659 | 0.2025 | 30.98 |

old_c5为a30°/b.10/c<5°；relaxed_abc仅放宽为a45°/b.15/c<10°。error_abs用最近5个已执行命令的一步先验预测RMS残差最大值≤.15替代a/b；error_relative用该最大值除以max(.02, 最近5步实际XYZ命令RMS)，阈值≤.50替代a/b；两个残差臂均保留c<5°。误差目标为实际送入环境的XYZ控制命令（按原filter规则clip至[-1,1]），不是世界EEF位移。所有主门控保留共同历史/夹爪/旋转约束、禁连续Kalman、verify τ=.10、P2、冻结Kalman参数及数学实现；alternate保留原简单交替定义。

每case六臂同物理卡、同resident服务，各臂使用新环境并检查初始RGB/state/seed和startup Full输出相等。Williams顺序轮换；GPU2/3各20case、每suite各5case，位置计数各3或4次。公用prompt/model准备在每臂闭环前执行并单列，不把首次条件的冷准备计入其策略成本。

本轮三个新候选均未显示优于简单交替的明确证据。alternate为38/40，old_c5为37/40；放宽阈值35/40，绝对残差37/40，相对残差36/40。相对old_c5，绝对残差是2恢复/2新失败，且成本在GPU2下降2.4%、GPU3上升1.0%，没有稳定降本。放宽阈值两卡略快但质量下降2例。当前保留原对照，不推荐据这40例替换为新候选。

下列汇总wall与成本比仅描述这批两卡样本；速度判断以逐卡同case分层为准。

## 各套件成功数

| 条件 | Spatial | Object | Goal | Long |
|---|---:|---:|---:|---:|
| parent | 9/10 | 10/10 | 8/10 | 7/10 |
| alternate | 10/10 | 10/10 | 10/10 | 8/10 |
| old_c5 | 9/10 | 10/10 | 10/10 | 8/10 |
| relaxed_abc | 8/10 | 10/10 | 8/10 | 9/10 |
| error_abs | 10/10 | 10/10 | 8/10 | 9/10 |
| error_relative | 10/10 | 10/10 | 8/10 | 8/10 |

## 配对对照

| 候选 / 参考 | 恢复 | 新失败 | 描述性汇总成本比 | 分层bootstrap95%成功率差 |
|---|---:|---:|---:|---|
| error_abs / old_c5 | 2 | 2 | 0.9927 | [-10.00,+10.00]pp |
| error_abs / alternate | 1 | 2 | 1.0065 | [-10.00,+5.00]pp |
| error_abs / parent | 3 | 0 | 0.9015 | [+0.00,+15.00]pp |
| parent / old_c5 | 0 | 3 | 1.1012 | [-15.00,+0.00]pp |
| parent / alternate | 0 | 4 | 1.1165 | [-20.00,-2.50]pp |
| alternate / old_c5 | 1 | 0 | 0.9863 | [+0.00,+7.50]pp |
| alternate / parent | 4 | 0 | 0.8956 | [+2.50,+20.00]pp |
| error_relative / old_c5 | 2 | 3 | 1.0179 | [-12.50,+7.50]pp |
| error_relative / alternate | 1 | 3 | 1.0321 | [-15.00,+2.50]pp |
| error_relative / parent | 3 | 1 | 0.9244 | [-5.00,+15.00]pp |
| relaxed_abc / old_c5 | 1 | 3 | 0.9689 | [-15.00,+2.50]pp |
| relaxed_abc / alternate | 1 | 4 | 0.9823 | [-17.50,+2.50]pp |
| relaxed_abc / parent | 2 | 1 | 0.8798 | [-5.00,+10.00]pp |
| old_c5 / alternate | 0 | 1 | 1.0139 | [-7.50,+0.00]pp |
| old_c5 / parent | 3 | 0 | 0.9081 | [+0.00,+15.00]pp |

## 同卡成本分层

| 候选 / 参考 | GPU | 同case n | candidate ms/action | reference ms/action | 比率 |
|---|---:|---:|---:|---:|---:|
| alternate / old_c5 | 2 | 20 | 16.638 | 16.920 | 0.9833 |
| alternate / old_c5 | 3 | 20 | 16.022 | 16.214 | 0.9882 |
| alternate / parent | 2 | 20 | 16.638 | 18.113 | 0.9185 |
| alternate / parent | 3 | 20 | 16.022 | 18.372 | 0.8721 |
| old_c5 / alternate | 2 | 20 | 16.920 | 16.638 | 1.0170 |
| old_c5 / alternate | 3 | 20 | 16.214 | 16.022 | 1.0120 |
| old_c5 / parent | 2 | 20 | 16.920 | 18.113 | 0.9341 |
| old_c5 / parent | 3 | 20 | 16.214 | 18.372 | 0.8826 |
| relaxed_abc / old_c5 | 2 | 20 | 16.452 | 16.920 | 0.9723 |
| relaxed_abc / old_c5 | 3 | 20 | 15.595 | 16.214 | 0.9618 |
| relaxed_abc / alternate | 2 | 20 | 16.452 | 16.638 | 0.9888 |
| relaxed_abc / alternate | 3 | 20 | 15.595 | 16.022 | 0.9733 |
| relaxed_abc / parent | 2 | 20 | 16.452 | 18.113 | 0.9083 |
| relaxed_abc / parent | 3 | 20 | 15.595 | 18.372 | 0.8488 |
| error_abs / old_c5 | 2 | 20 | 16.508 | 16.920 | 0.9757 |
| error_abs / old_c5 | 3 | 20 | 16.378 | 16.214 | 1.0101 |
| error_abs / alternate | 2 | 20 | 16.508 | 16.638 | 0.9922 |
| error_abs / alternate | 3 | 20 | 16.378 | 16.022 | 1.0222 |
| error_abs / parent | 2 | 20 | 16.508 | 18.113 | 0.9114 |
| error_abs / parent | 3 | 20 | 16.378 | 18.372 | 0.8915 |
| error_relative / old_c5 | 2 | 20 | 16.525 | 16.920 | 0.9766 |
| error_relative / old_c5 | 3 | 20 | 17.186 | 16.214 | 1.0599 |
| error_relative / alternate | 2 | 20 | 16.525 | 16.638 | 0.9932 |
| error_relative / alternate | 3 | 20 | 17.186 | 16.022 | 1.0726 |
| error_relative / parent | 2 | 20 | 16.525 | 18.113 | 0.9123 |
| error_relative / parent | 3 | 20 | 17.186 | 18.372 | 0.9355 |

## 实际路由与门控

| 条件 | 请求 | KF提案 | KF执行请求 | KF实际动作占比 | Full |
|---|---:|---:|---:|---:|---:|
| parent | 1493 | 0 | 0 | 0.00% | 627 |
| alternate | 1428 | 649 | 446 | 31.14% | 666 |
| old_c5 | 1448 | 401 | 310 | 21.13% | 616 |
| relaxed_abc | 1480 | 528 | 412 | 27.64% | 632 |
| error_abs | 1395 | 475 | 343 | 24.51% | 609 |
| error_relative | 1432 | 436 | 304 | 20.94% | 638 |

requests_and_causal_features.csv逐请求重放所有已执行历史，包含a/b/c与残差数值；非残差臂的新增残差列是离线诊断，不伪装成其运行时工作。execution_audit.json核对全240的门控、连续约束、真实动作前缀、旋转/夹爪保持、初态/Full签名和predict+ack成本。

旧parent轨迹2012个候选时点，原c5允许1138，放宽阈值1561，绝对残差1338，相对残差1263；均未纳入新路由会改变历史的反馈，不能当实际Kalman使用率。NIS仅诊断，不将冻结任意命令协方差当作已校准χ²分布。参数在新40观测前冻结，old_coverage_receipt.json记录重放来源。

原v3.3≥10%降本仅作参考，即成本比≤0.90；应同时看成功计数与两物理卡的分层比值。未校正候选筛选/多重比较，不能把小样本相等成功数说成总体非劣。新成本含ack，因此不直接与v3.3旧predict-only表比速度。

父对照沿用现有共享Policy：即便不选择Kalman，它的ack仍按旧实现调用Kalman.observe。本轮计入此真实计算，不冒称最简/优化过的no-Kalman实现，也不做未经实测的扣时。调参收益主要对照old_c5与alternate；predict-only与ack分别列示。

## 同卡成本与复现

逐卡、逐case配对和置信区间见paired_results.json；原始种子/噪声、初态/Full签名、动作、实际EEF XYZ/姿态和时延见每episode目录。共同warm包含固定真实初态的模型准备，动作未进入环境；与实际闭环policy wall分开保存。

旧init6/7仅用于参数/覆盖诊断，新40固定后未改阈值；任务/初态可能曾用于更早研究，不称外部未见测试集。所有有效超时失败保留，技术attempt另列。

## 固定真实XYZ

![首8个预定profile案例的实际EEF XYZ](fixed_profile_cases_xyz.png)

只使用预先选定的task0/5四suite，不按结果换案例；不累加预测命令、不补零或时间拉齐。

## 完成与验收

两卡从worker资源核验至正常退出共23.38分钟；240例有效闭环，技术重试0。全量8676请求审计通过，40组初态/启动Full输出逐位一致，17项冻结执行源码/旧依赖hash保持一致。两worker成功退出，末次C/G均空，本轮GPU2/3租约inactive。共同预热共198.02 worker秒单列；模型加载各34.36/33.78秒。未追加GPU实验。详见final_receipt.json和evidence_index.json。
