# Experiment 9: LIBERO 标准套件 VLM 裁剪实验报告

> 本文为历史初筛报告。完整 28,800 条实验分析、全量阶段统计及新轨迹图/视频请看 [最新完整分析报告](experiment9_final_analysis_report.md)。本文旧分母和图表不代表最终结果。

数据阶段：**初筛 / 当前可用数据**。 已完成 Full18 baseline 378/400。

VLM 裁剪组固定 Action Expert Full18、D10；Action Expert 降噪组固定 VLM Full18。四套件各 10 任务。实际层索引、noise hash、denoise steps 和服务端配置保存在 inference 元数据中。

## 套件成功率

| 变体 | 套件 | 成功 / episodes | Wilson 95% | 相对 Full18 (pp) | 任务聚类配对差 95% | policy p50 / p95 ms | denoise p50 / p95 ms | ≥90% | 新失败 / 恢复 | McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full18 | object | 97/100 (97.0%) | [91.5%, 99.0%] | — | — | 43.0 / 50.5 | — | 是 | — | — |
| Full18 | spatial | 94/100 (94.0%) | [87.5%, 97.2%] | — | — | 43.5 / 51.6 | — | 是 | — | — |
| Full18 | goal | 98/100 (98.0%) | [93.0%, 99.4%] | — | — | 43.2 / 51.6 | — | 是 | — | — |
| Full18 | 10 | 89/100 (89.0%) | [81.4%, 93.7%] | — | — | 43.4 / 49.6 | — | 否 | — | — |
| Skip-L2 | object | 98/100 (98.0%) | [93.0%, 99.4%] | +1.0 | [-2.0, +4.0] | 42.1 / 49.2 | — | 是 | 1 / 2 | 1 |
| Skip-L2 | spatial | 96/100 (96.0%) | [90.2%, 98.4%] | +2.0 | [-4.0, +7.0] | 42.4 / 50.4 | — | 是 | 3 / 5 | 0.7266 |
| Skip-L2 | goal | 96/100 (96.0%) | [90.2%, 98.4%] | -2.0 | [-6.0, +2.0] | 42.7 / 52.0 | — | 是 | 4 / 2 | 0.6875 |
| Skip-L2 | 10 | 83/100 (83.0%) | [74.5%, 89.1%] | -6.0 | [-14.0, +2.0] | 42.4 / 48.4 | — | 否 | 11 / 5 | 0.2101 |
| Skip-L3 | object | 97/100 (97.0%) | [91.5%, 99.0%] | +0.0 | [+0.0, +0.0] | 42.5 / 49.9 | — | 是 | 2 / 2 | 1 |
| Skip-L3 | spatial | 97/100 (97.0%) | [91.5%, 99.0%] | +3.0 | [+0.0, +6.0] | 43.0 / 51.6 | — | 是 | 2 / 5 | 0.4531 |
| Skip-L3 | goal | 95/100 (95.0%) | [88.8%, 97.8%] | -3.0 | [-7.0, +1.0] | 42.1 / 50.2 | — | 是 | 5 / 2 | 0.4531 |
| Skip-L3 | 10 | 90/100 (90.0%) | [82.6%, 94.5%] | +1.0 | [-3.0, +5.0] | 42.2 / 49.5 | — | 是 | 6 / 7 | 1 |
| Skip-L4 | object | 95/100 (95.0%) | [88.8%, 97.8%] | -2.0 | [-7.0, +2.0] | 42.0 / 50.0 | — | 是 | 4 / 2 | 0.6875 |
| Skip-L4 | spatial | 98/100 (98.0%) | [93.0%, 99.4%] | +4.0 | [+0.0, +8.0] | 42.5 / 51.2 | — | 是 | 1 / 5 | 0.2188 |
| Skip-L4 | goal | 93/100 (93.0%) | [86.3%, 96.6%] | -5.0 | [-11.0, +0.0] | 42.1 / 49.4 | — | 是 | 7 / 2 | 0.1797 |
| Skip-L4 | 10 | 78/100 (78.0%) | [68.9%, 85.0%] | -11.0 | [-22.0, +0.0] | 42.8 / 50.0 | — | 否 | 15 / 4 | 0.01921 |
| Skip-L5 | object | 100/100 (100.0%) | [96.3%, 100.0%] | +3.0 | [+0.0, +6.0] | 42.3 / 48.7 | — | 是 | 0 / 3 | 0.25 |
| Skip-L5 | spatial | 96/100 (96.0%) | [90.2%, 98.4%] | +2.0 | [-3.0, +8.0] | 42.5 / 51.1 | — | 是 | 3 / 5 | 0.7266 |
| Skip-L5 | goal | 93/100 (93.0%) | [86.3%, 96.6%] | -5.0 | [-10.0, +0.0] | 42.4 / 49.8 | — | 是 | 7 / 2 | 0.1797 |
| Skip-L5 | 10 | 83/100 (83.0%) | [74.5%, 89.1%] | -6.0 | [-15.0, +2.0] | 42.5 / 49.4 | — | 否 | 12 / 6 | 0.2379 |
| Skip-L2-5 | object | 57/100 (57.0%) | [47.2%, 66.3%] | -40.0 | [-57.0, -23.0] | 38.6 / 45.4 | — | 否 | 41 / 1 | 1.955e-11 |
| Skip-L2-5 | spatial | 45/100 (45.0%) | [35.6%, 54.8%] | -49.0 | [-71.0, -27.0] | 38.9 / 46.9 | — | 否 | 52 / 3 | 1.542e-12 |
| Skip-L2-5 | goal | 15/100 (15.0%) | [9.3%, 23.3%] | -83.0 | [-95.0, -65.0] | 38.6 / 44.5 | — | 否 | 84 / 1 | 4.446e-24 |
| Skip-L2-5 | 10 | 25/100 (25.0%) | [17.5%, 34.3%] | -64.0 | [-73.0, -55.0] | 38.7 / 44.0 | — | 否 | 65 / 1 | 1.816e-18 |
| V16 | object | 99/100 (99.0%) | [94.6%, 99.8%] | +2.0 | [+0.0, +5.0] | 40.9 / 47.4 | — | 是 | 0 / 2 | 0.5 |
| V16 | spatial | 98/100 (98.0%) | [93.0%, 99.4%] | +4.0 | [-2.0, +10.0] | 41.4 / 48.0 | — | 是 | 2 / 6 | 0.2891 |
| V16 | goal | 83/100 (83.0%) | [74.5%, 89.1%] | -15.0 | [-30.0, -2.0] | 41.6 / 48.7 | — | 否 | 17 / 2 | 0.0007286 |
| V16 | 10 | 77/100 (77.0%) | [67.8%, 84.2%] | -12.0 | [-23.0, -1.0] | 41.3 / 47.0 | — | 否 | 17 / 5 | 0.0169 |
| V14 | object | 78/100 (78.0%) | [68.9%, 85.0%] | -19.0 | [-32.0, -8.0] | 38.6 / 45.7 | — | 否 | 21 / 2 | 6.604e-05 |
| V14 | spatial | 72/100 (72.0%) | [62.5%, 79.9%] | -22.0 | [-40.0, -6.0] | 38.8 / 45.4 | — | 否 | 25 / 3 | 2.744e-05 |
| V14 | goal | 38/100 (38.0%) | [29.1%, 47.8%] | -60.0 | [-84.0, -35.0] | 38.7 / 44.9 | — | 否 | 61 / 1 | 2.732e-17 |
| V14 | 10 | 16/100 (16.0%) | [10.1%, 24.4%] | -73.0 | [-85.0, -61.0] | 38.6 / 43.5 | — | 否 | 73 / 0 | 2.118e-22 |
| V12 | object | 0/100 (0.0%) | [0.0%, 3.7%] | -97.0 | [-100.0, -94.0] | 36.3 / 41.9 | — | 否 | 97 / 0 | 1.262e-29 |
| V12 | spatial | 0/100 (0.0%) | [0.0%, 3.7%] | -94.0 | [-98.0, -90.0] | 36.5 / 42.5 | — | 否 | 94 / 0 | 1.01e-28 |
| V12 | goal | 0/100 (0.0%) | [0.0%, 3.7%] | -98.0 | [-100.0, -95.0] | 36.4 / 41.4 | — | 否 | 98 / 0 | 6.311e-30 |
| V12 | 10 | 0/100 (0.0%) | [0.0%, 3.7%] | -89.0 | [-96.0, -81.0] | 36.4 / 42.3 | — | 否 | 89 / 0 | 3.231e-27 |
| V10-Mid | object | 0/100 (0.0%) | [0.0%, 3.7%] | -97.0 | [-100.0, -94.0] | 73.1 / 81.1 | — | 否 | 97 / 0 | 1.262e-29 |
| V10-Mid | spatial | 0/100 (0.0%) | [0.0%, 3.7%] | -94.0 | [-98.0, -90.0] | 75.8 / 83.8 | — | 否 | 94 / 0 | 1.01e-28 |
| V10-Mid | goal | 0/100 (0.0%) | [0.0%, 3.7%] | -98.0 | [-100.0, -95.0] | 34.4 / 77.2 | — | 否 | 98 / 0 | 6.311e-30 |
| V10-Mid | 10 | 0/100 (0.0%) | [0.0%, 3.7%] | -89.0 | [-96.0, -81.0] | 34.1 / 40.9 | — | 否 | 89 / 0 | 3.231e-27 |
| AE-D7 | object | 97/100 (97.0%) | [91.5%, 99.0%] | +0.0 | [+0.0, +0.0] | 40.3 / 46.3 | — | 是 | 1 / 1 | 1 |
| AE-D7 | spatial | 96/100 (96.0%) | [90.2%, 98.4%] | +2.0 | [-2.0, +7.0] | 40.1 / 50.0 | — | 是 | 2 / 4 | 0.6875 |
| AE-D5 | object | 98/100 (98.0%) | [93.0%, 99.4%] | +1.0 | [-2.0, +4.0] | 37.6 / 43.8 | — | 是 | 1 / 2 | 1 |
| AE-D5 | spatial | 98/100 (98.0%) | [93.0%, 99.4%] | +4.0 | [+1.0, +7.0] | 37.7 / 45.8 | — | 是 | 0 / 4 | 0.125 |
| AE-D5 | goal | 98/100 (98.0%) | [93.0%, 99.4%] | +0.0 | [-4.0, +4.0] | 38.4 / 51.8 | — | 是 | 2 / 2 | 1 |
| AE-D3 | object | 98/100 (98.0%) | [93.0%, 99.4%] | +1.0 | [-2.0, +4.0] | 35.5 / 44.1 | — | 是 | 1 / 2 | 1 |

≥90% 是观测点估计阈值，不等价于达到 baseline 或通过非劣检验。配对差区间以任务为聚类单位；初筛只有每任务 10 个状态，区间仍较宽。

![套件成功率热图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/suite_success_heatmap.png)

## 失败阶段与转换

阶段来自仿真器末端、物体位姿、接触/抓取状态及 goal predicate。reach 表示到达该阶段或更后阶段；conditional transition 表示已到达前一阶段的 episode 中继续到达本阶段的比例。

| 变体 | 套件 | 阶段 | 到达数 / episode | 到达率 | 前一阶段条件转换率 |
|---|---|---|---:|---:|---:|
| Full18 | object | initial | 100/100 | 100.0% | — |
| Full18 | object | approach | 100/100 | 100.0% | 100.0% |
| Full18 | object | contact | 100/100 | 100.0% | 100.0% |
| Full18 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Full18 | object | transport | 99/100 | 99.0% | 99.0% |
| Full18 | object | goal_alignment | 97/100 | 97.0% | 98.0% |
| Full18 | object | release_or_final_action | 97/100 | 97.0% | 100.0% |
| Full18 | object | success | 97/100 | 97.0% | 100.0% |
| Full18 | spatial | initial | 100/100 | 100.0% | — |
| Full18 | spatial | approach | 100/100 | 100.0% | 100.0% |
| Full18 | spatial | contact | 100/100 | 100.0% | 100.0% |
| Full18 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Full18 | spatial | transport | 99/100 | 99.0% | 99.0% |
| Full18 | spatial | goal_alignment | 97/100 | 97.0% | 98.0% |
| Full18 | spatial | release_or_final_action | 97/100 | 97.0% | 100.0% |
| Full18 | spatial | success | 94/100 | 94.0% | 96.9% |
| Full18 | goal | initial | 100/100 | 100.0% | — |
| Full18 | goal | approach | 100/100 | 100.0% | 100.0% |
| Full18 | goal | contact | 100/100 | 100.0% | 100.0% |
| Full18 | goal | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Full18 | goal | transport | 100/100 | 100.0% | 100.0% |
| Full18 | goal | goal_alignment | 99/100 | 99.0% | 99.0% |
| Full18 | goal | release_or_final_action | 99/100 | 99.0% | 100.0% |
| Full18 | goal | success | 98/100 | 98.0% | 99.0% |
| Full18 | 10 | initial | 100/100 | 100.0% | — |
| Full18 | 10 | approach | 100/100 | 100.0% | 100.0% |
| Full18 | 10 | contact | 100/100 | 100.0% | 100.0% |
| Full18 | 10 | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Full18 | 10 | transport | 99/100 | 99.0% | 99.0% |
| Full18 | 10 | goal_alignment | 99/100 | 99.0% | 100.0% |
| Full18 | 10 | release_or_final_action | 94/100 | 94.0% | 94.9% |
| Full18 | 10 | success | 89/100 | 89.0% | 94.7% |
| Skip-L2 | object | initial | 100/100 | 100.0% | — |
| Skip-L2 | object | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2 | object | contact | 100/100 | 100.0% | 100.0% |
| Skip-L2 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L2 | object | transport | 98/100 | 98.0% | 98.0% |
| Skip-L2 | object | goal_alignment | 98/100 | 98.0% | 100.0% |
| Skip-L2 | object | release_or_final_action | 98/100 | 98.0% | 100.0% |
| Skip-L2 | object | success | 98/100 | 98.0% | 100.0% |
| Skip-L2 | spatial | initial | 100/100 | 100.0% | — |
| Skip-L2 | spatial | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2 | spatial | contact | 100/100 | 100.0% | 100.0% |
| Skip-L2 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L2 | spatial | transport | 100/100 | 100.0% | 100.0% |
| Skip-L2 | spatial | goal_alignment | 100/100 | 100.0% | 100.0% |
| Skip-L2 | spatial | release_or_final_action | 100/100 | 100.0% | 100.0% |
| Skip-L2 | spatial | success | 96/100 | 96.0% | 96.0% |
| Skip-L2 | goal | initial | 100/100 | 100.0% | — |
| Skip-L2 | goal | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2 | goal | contact | 99/100 | 99.0% | 99.0% |
| Skip-L2 | goal | grasp_or_manipulate | 98/100 | 98.0% | 99.0% |
| Skip-L2 | goal | transport | 97/100 | 97.0% | 99.0% |
| Skip-L2 | goal | goal_alignment | 96/100 | 96.0% | 99.0% |
| Skip-L2 | goal | release_or_final_action | 96/100 | 96.0% | 100.0% |
| Skip-L2 | goal | success | 96/100 | 96.0% | 100.0% |
| Skip-L2 | 10 | initial | 100/100 | 100.0% | — |
| Skip-L2 | 10 | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2 | 10 | contact | 99/100 | 99.0% | 99.0% |
| Skip-L2 | 10 | grasp_or_manipulate | 99/100 | 99.0% | 100.0% |
| Skip-L2 | 10 | transport | 98/100 | 98.0% | 99.0% |
| Skip-L2 | 10 | goal_alignment | 98/100 | 98.0% | 100.0% |
| Skip-L2 | 10 | release_or_final_action | 89/100 | 89.0% | 90.8% |
| Skip-L2 | 10 | success | 83/100 | 83.0% | 93.3% |
| Skip-L3 | object | initial | 100/100 | 100.0% | — |
| Skip-L3 | object | approach | 100/100 | 100.0% | 100.0% |
| Skip-L3 | object | contact | 100/100 | 100.0% | 100.0% |
| Skip-L3 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L3 | object | transport | 97/100 | 97.0% | 97.0% |
| Skip-L3 | object | goal_alignment | 97/100 | 97.0% | 100.0% |
| Skip-L3 | object | release_or_final_action | 97/100 | 97.0% | 100.0% |
| Skip-L3 | object | success | 97/100 | 97.0% | 100.0% |
| Skip-L3 | spatial | initial | 100/100 | 100.0% | — |
| Skip-L3 | spatial | approach | 100/100 | 100.0% | 100.0% |
| Skip-L3 | spatial | contact | 100/100 | 100.0% | 100.0% |
| Skip-L3 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L3 | spatial | transport | 100/100 | 100.0% | 100.0% |
| Skip-L3 | spatial | goal_alignment | 100/100 | 100.0% | 100.0% |
| Skip-L3 | spatial | release_or_final_action | 100/100 | 100.0% | 100.0% |
| Skip-L3 | spatial | success | 97/100 | 97.0% | 97.0% |
| Skip-L3 | goal | initial | 100/100 | 100.0% | — |
| Skip-L3 | goal | approach | 100/100 | 100.0% | 100.0% |
| Skip-L3 | goal | contact | 99/100 | 99.0% | 99.0% |
| Skip-L3 | goal | grasp_or_manipulate | 99/100 | 99.0% | 100.0% |
| Skip-L3 | goal | transport | 98/100 | 98.0% | 99.0% |
| Skip-L3 | goal | goal_alignment | 98/100 | 98.0% | 100.0% |
| Skip-L3 | goal | release_or_final_action | 97/100 | 97.0% | 99.0% |
| Skip-L3 | goal | success | 95/100 | 95.0% | 97.9% |
| Skip-L3 | 10 | initial | 100/100 | 100.0% | — |
| Skip-L3 | 10 | approach | 100/100 | 100.0% | 100.0% |
| Skip-L3 | 10 | contact | 100/100 | 100.0% | 100.0% |
| Skip-L3 | 10 | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L3 | 10 | transport | 100/100 | 100.0% | 100.0% |
| Skip-L3 | 10 | goal_alignment | 100/100 | 100.0% | 100.0% |
| Skip-L3 | 10 | release_or_final_action | 98/100 | 98.0% | 98.0% |
| Skip-L3 | 10 | success | 90/100 | 90.0% | 91.8% |
| Skip-L4 | object | initial | 100/100 | 100.0% | — |
| Skip-L4 | object | approach | 100/100 | 100.0% | 100.0% |
| Skip-L4 | object | contact | 100/100 | 100.0% | 100.0% |
| Skip-L4 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L4 | object | transport | 98/100 | 98.0% | 98.0% |
| Skip-L4 | object | goal_alignment | 97/100 | 97.0% | 99.0% |
| Skip-L4 | object | release_or_final_action | 97/100 | 97.0% | 100.0% |
| Skip-L4 | object | success | 95/100 | 95.0% | 97.9% |
| Skip-L4 | spatial | initial | 100/100 | 100.0% | — |
| Skip-L4 | spatial | approach | 100/100 | 100.0% | 100.0% |
| Skip-L4 | spatial | contact | 100/100 | 100.0% | 100.0% |
| Skip-L4 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L4 | spatial | transport | 100/100 | 100.0% | 100.0% |
| Skip-L4 | spatial | goal_alignment | 100/100 | 100.0% | 100.0% |
| Skip-L4 | spatial | release_or_final_action | 100/100 | 100.0% | 100.0% |
| Skip-L4 | spatial | success | 98/100 | 98.0% | 98.0% |
| Skip-L4 | goal | initial | 100/100 | 100.0% | — |
| Skip-L4 | goal | approach | 100/100 | 100.0% | 100.0% |
| Skip-L4 | goal | contact | 97/100 | 97.0% | 97.0% |
| Skip-L4 | goal | grasp_or_manipulate | 97/100 | 97.0% | 100.0% |
| Skip-L4 | goal | transport | 96/100 | 96.0% | 99.0% |
| Skip-L4 | goal | goal_alignment | 95/100 | 95.0% | 99.0% |
| Skip-L4 | goal | release_or_final_action | 93/100 | 93.0% | 97.9% |
| Skip-L4 | goal | success | 93/100 | 93.0% | 100.0% |
| Skip-L4 | 10 | initial | 100/100 | 100.0% | — |
| Skip-L4 | 10 | approach | 100/100 | 100.0% | 100.0% |
| Skip-L4 | 10 | contact | 100/100 | 100.0% | 100.0% |
| Skip-L4 | 10 | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L4 | 10 | transport | 100/100 | 100.0% | 100.0% |
| Skip-L4 | 10 | goal_alignment | 97/100 | 97.0% | 97.0% |
| Skip-L4 | 10 | release_or_final_action | 90/100 | 90.0% | 92.8% |
| Skip-L4 | 10 | success | 78/100 | 78.0% | 86.7% |
| Skip-L5 | object | initial | 100/100 | 100.0% | — |
| Skip-L5 | object | approach | 100/100 | 100.0% | 100.0% |
| Skip-L5 | object | contact | 100/100 | 100.0% | 100.0% |
| Skip-L5 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L5 | object | transport | 100/100 | 100.0% | 100.0% |
| Skip-L5 | object | goal_alignment | 100/100 | 100.0% | 100.0% |
| Skip-L5 | object | release_or_final_action | 100/100 | 100.0% | 100.0% |
| Skip-L5 | object | success | 100/100 | 100.0% | 100.0% |
| Skip-L5 | spatial | initial | 100/100 | 100.0% | — |
| Skip-L5 | spatial | approach | 100/100 | 100.0% | 100.0% |
| Skip-L5 | spatial | contact | 100/100 | 100.0% | 100.0% |
| Skip-L5 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L5 | spatial | transport | 100/100 | 100.0% | 100.0% |
| Skip-L5 | spatial | goal_alignment | 99/100 | 99.0% | 99.0% |
| Skip-L5 | spatial | release_or_final_action | 99/100 | 99.0% | 100.0% |
| Skip-L5 | spatial | success | 96/100 | 96.0% | 97.0% |
| Skip-L5 | goal | initial | 100/100 | 100.0% | — |
| Skip-L5 | goal | approach | 100/100 | 100.0% | 100.0% |
| Skip-L5 | goal | contact | 99/100 | 99.0% | 99.0% |
| Skip-L5 | goal | grasp_or_manipulate | 98/100 | 98.0% | 99.0% |
| Skip-L5 | goal | transport | 96/100 | 96.0% | 98.0% |
| Skip-L5 | goal | goal_alignment | 95/100 | 95.0% | 99.0% |
| Skip-L5 | goal | release_or_final_action | 94/100 | 94.0% | 98.9% |
| Skip-L5 | goal | success | 93/100 | 93.0% | 98.9% |
| Skip-L5 | 10 | initial | 100/100 | 100.0% | — |
| Skip-L5 | 10 | approach | 100/100 | 100.0% | 100.0% |
| Skip-L5 | 10 | contact | 100/100 | 100.0% | 100.0% |
| Skip-L5 | 10 | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| Skip-L5 | 10 | transport | 99/100 | 99.0% | 99.0% |
| Skip-L5 | 10 | goal_alignment | 98/100 | 98.0% | 99.0% |
| Skip-L5 | 10 | release_or_final_action | 91/100 | 91.0% | 92.9% |
| Skip-L5 | 10 | success | 83/100 | 83.0% | 91.2% |
| Skip-L2-5 | object | initial | 100/100 | 100.0% | — |
| Skip-L2-5 | object | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2-5 | object | contact | 79/100 | 79.0% | 79.0% |
| Skip-L2-5 | object | grasp_or_manipulate | 79/100 | 79.0% | 100.0% |
| Skip-L2-5 | object | transport | 74/100 | 74.0% | 93.7% |
| Skip-L2-5 | object | goal_alignment | 59/100 | 59.0% | 79.7% |
| Skip-L2-5 | object | release_or_final_action | 59/100 | 59.0% | 100.0% |
| Skip-L2-5 | object | success | 57/100 | 57.0% | 96.6% |
| Skip-L2-5 | spatial | initial | 100/100 | 100.0% | — |
| Skip-L2-5 | spatial | approach | 90/100 | 90.0% | 90.0% |
| Skip-L2-5 | spatial | contact | 72/100 | 72.0% | 80.0% |
| Skip-L2-5 | spatial | grasp_or_manipulate | 63/100 | 63.0% | 87.5% |
| Skip-L2-5 | spatial | transport | 55/100 | 55.0% | 87.3% |
| Skip-L2-5 | spatial | goal_alignment | 52/100 | 52.0% | 94.5% |
| Skip-L2-5 | spatial | release_or_final_action | 50/100 | 50.0% | 96.2% |
| Skip-L2-5 | spatial | success | 45/100 | 45.0% | 90.0% |
| Skip-L2-5 | goal | initial | 100/100 | 100.0% | — |
| Skip-L2-5 | goal | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2-5 | goal | contact | 66/100 | 66.0% | 66.0% |
| Skip-L2-5 | goal | grasp_or_manipulate | 63/100 | 63.0% | 95.5% |
| Skip-L2-5 | goal | transport | 43/100 | 43.0% | 68.3% |
| Skip-L2-5 | goal | goal_alignment | 16/100 | 16.0% | 37.2% |
| Skip-L2-5 | goal | release_or_final_action | 16/100 | 16.0% | 100.0% |
| Skip-L2-5 | goal | success | 15/100 | 15.0% | 93.8% |
| Skip-L2-5 | 10 | initial | 100/100 | 100.0% | — |
| Skip-L2-5 | 10 | approach | 100/100 | 100.0% | 100.0% |
| Skip-L2-5 | 10 | contact | 92/100 | 92.0% | 92.0% |
| Skip-L2-5 | 10 | grasp_or_manipulate | 90/100 | 90.0% | 97.8% |
| Skip-L2-5 | 10 | transport | 78/100 | 78.0% | 86.7% |
| Skip-L2-5 | 10 | goal_alignment | 75/100 | 75.0% | 96.2% |
| Skip-L2-5 | 10 | release_or_final_action | 67/100 | 67.0% | 89.3% |
| Skip-L2-5 | 10 | success | 25/100 | 25.0% | 37.3% |
| V16 | object | initial | 100/100 | 100.0% | — |
| V16 | object | approach | 100/100 | 100.0% | 100.0% |
| V16 | object | contact | 100/100 | 100.0% | 100.0% |
| V16 | object | grasp_or_manipulate | 99/100 | 99.0% | 99.0% |
| V16 | object | transport | 99/100 | 99.0% | 100.0% |
| V16 | object | goal_alignment | 99/100 | 99.0% | 100.0% |
| V16 | object | release_or_final_action | 99/100 | 99.0% | 100.0% |
| V16 | object | success | 99/100 | 99.0% | 100.0% |
| V16 | spatial | initial | 100/100 | 100.0% | — |
| V16 | spatial | approach | 100/100 | 100.0% | 100.0% |
| V16 | spatial | contact | 99/100 | 99.0% | 99.0% |
| V16 | spatial | grasp_or_manipulate | 99/100 | 99.0% | 100.0% |
| V16 | spatial | transport | 99/100 | 99.0% | 100.0% |
| V16 | spatial | goal_alignment | 99/100 | 99.0% | 100.0% |
| V16 | spatial | release_or_final_action | 99/100 | 99.0% | 100.0% |
| V16 | spatial | success | 98/100 | 98.0% | 99.0% |
| V16 | goal | initial | 100/100 | 100.0% | — |
| V16 | goal | approach | 100/100 | 100.0% | 100.0% |
| V16 | goal | contact | 98/100 | 98.0% | 98.0% |
| V16 | goal | grasp_or_manipulate | 96/100 | 96.0% | 98.0% |
| V16 | goal | transport | 89/100 | 89.0% | 92.7% |
| V16 | goal | goal_alignment | 86/100 | 86.0% | 96.6% |
| V16 | goal | release_or_final_action | 86/100 | 86.0% | 100.0% |
| V16 | goal | success | 83/100 | 83.0% | 96.5% |
| V16 | 10 | initial | 100/100 | 100.0% | — |
| V16 | 10 | approach | 100/100 | 100.0% | 100.0% |
| V16 | 10 | contact | 100/100 | 100.0% | 100.0% |
| V16 | 10 | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| V16 | 10 | transport | 99/100 | 99.0% | 99.0% |
| V16 | 10 | goal_alignment | 99/100 | 99.0% | 100.0% |
| V16 | 10 | release_or_final_action | 96/100 | 96.0% | 97.0% |
| V16 | 10 | success | 77/100 | 77.0% | 80.2% |
| V14 | object | initial | 100/100 | 100.0% | — |
| V14 | object | approach | 100/100 | 100.0% | 100.0% |
| V14 | object | contact | 100/100 | 100.0% | 100.0% |
| V14 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| V14 | object | transport | 87/100 | 87.0% | 87.0% |
| V14 | object | goal_alignment | 78/100 | 78.0% | 89.7% |
| V14 | object | release_or_final_action | 78/100 | 78.0% | 100.0% |
| V14 | object | success | 78/100 | 78.0% | 100.0% |
| V14 | spatial | initial | 100/100 | 100.0% | — |
| V14 | spatial | approach | 100/100 | 100.0% | 100.0% |
| V14 | spatial | contact | 99/100 | 99.0% | 99.0% |
| V14 | spatial | grasp_or_manipulate | 99/100 | 99.0% | 100.0% |
| V14 | spatial | transport | 87/100 | 87.0% | 87.9% |
| V14 | spatial | goal_alignment | 83/100 | 83.0% | 95.4% |
| V14 | spatial | release_or_final_action | 83/100 | 83.0% | 100.0% |
| V14 | spatial | success | 72/100 | 72.0% | 86.7% |
| V14 | goal | initial | 100/100 | 100.0% | — |
| V14 | goal | approach | 100/100 | 100.0% | 100.0% |
| V14 | goal | contact | 73/100 | 73.0% | 73.0% |
| V14 | goal | grasp_or_manipulate | 72/100 | 72.0% | 98.6% |
| V14 | goal | transport | 51/100 | 51.0% | 70.8% |
| V14 | goal | goal_alignment | 39/100 | 39.0% | 76.5% |
| V14 | goal | release_or_final_action | 39/100 | 39.0% | 100.0% |
| V14 | goal | success | 38/100 | 38.0% | 97.4% |
| V14 | 10 | initial | 100/100 | 100.0% | — |
| V14 | 10 | approach | 100/100 | 100.0% | 100.0% |
| V14 | 10 | contact | 99/100 | 99.0% | 99.0% |
| V14 | 10 | grasp_or_manipulate | 98/100 | 98.0% | 99.0% |
| V14 | 10 | transport | 79/100 | 79.0% | 80.6% |
| V14 | 10 | goal_alignment | 74/100 | 74.0% | 93.7% |
| V14 | 10 | release_or_final_action | 61/100 | 61.0% | 82.4% |
| V14 | 10 | success | 16/100 | 16.0% | 26.2% |
| V12 | object | initial | 100/100 | 100.0% | — |
| V12 | object | approach | 100/100 | 100.0% | 100.0% |
| V12 | object | contact | 26/100 | 26.0% | 26.0% |
| V12 | object | grasp_or_manipulate | 23/100 | 23.0% | 88.5% |
| V12 | object | transport | 2/100 | 2.0% | 8.7% |
| V12 | object | goal_alignment | 0/100 | 0.0% | 0.0% |
| V12 | object | release_or_final_action | 0/100 | 0.0% | — |
| V12 | object | success | 0/100 | 0.0% | — |
| V12 | spatial | initial | 100/100 | 100.0% | — |
| V12 | spatial | approach | 88/100 | 88.0% | 88.0% |
| V12 | spatial | contact | 41/100 | 41.0% | 46.6% |
| V12 | spatial | grasp_or_manipulate | 39/100 | 39.0% | 95.1% |
| V12 | spatial | transport | 14/100 | 14.0% | 35.9% |
| V12 | spatial | goal_alignment | 12/100 | 12.0% | 85.7% |
| V12 | spatial | release_or_final_action | 9/100 | 9.0% | 75.0% |
| V12 | spatial | success | 0/100 | 0.0% | 0.0% |
| V12 | goal | initial | 100/100 | 100.0% | — |
| V12 | goal | approach | 96/100 | 96.0% | 96.0% |
| V12 | goal | contact | 63/100 | 63.0% | 65.6% |
| V12 | goal | grasp_or_manipulate | 63/100 | 63.0% | 100.0% |
| V12 | goal | transport | 22/100 | 22.0% | 34.9% |
| V12 | goal | goal_alignment | 1/100 | 1.0% | 4.5% |
| V12 | goal | release_or_final_action | 1/100 | 1.0% | 100.0% |
| V12 | goal | success | 0/100 | 0.0% | 0.0% |
| V12 | 10 | initial | 100/100 | 100.0% | — |
| V12 | 10 | approach | 100/100 | 100.0% | 100.0% |
| V12 | 10 | contact | 69/100 | 69.0% | 69.0% |
| V12 | 10 | grasp_or_manipulate | 68/100 | 68.0% | 98.6% |
| V12 | 10 | transport | 18/100 | 18.0% | 26.5% |
| V12 | 10 | goal_alignment | 7/100 | 7.0% | 38.9% |
| V12 | 10 | release_or_final_action | 3/100 | 3.0% | 42.9% |
| V12 | 10 | success | 0/100 | 0.0% | 0.0% |
| V10-Mid | object | initial | 100/100 | 100.0% | — |
| V10-Mid | object | approach | 100/100 | 100.0% | 100.0% |
| V10-Mid | object | contact | 7/100 | 7.0% | 7.0% |
| V10-Mid | object | grasp_or_manipulate | 7/100 | 7.0% | 100.0% |
| V10-Mid | object | transport | 0/100 | 0.0% | 0.0% |
| V10-Mid | object | goal_alignment | 0/100 | 0.0% | — |
| V10-Mid | object | release_or_final_action | 0/100 | 0.0% | — |
| V10-Mid | object | success | 0/100 | 0.0% | — |
| V10-Mid | spatial | initial | 100/100 | 100.0% | — |
| V10-Mid | spatial | approach | 75/100 | 75.0% | 75.0% |
| V10-Mid | spatial | contact | 27/100 | 27.0% | 36.0% |
| V10-Mid | spatial | grasp_or_manipulate | 27/100 | 27.0% | 100.0% |
| V10-Mid | spatial | transport | 12/100 | 12.0% | 44.4% |
| V10-Mid | spatial | goal_alignment | 7/100 | 7.0% | 58.3% |
| V10-Mid | spatial | release_or_final_action | 7/100 | 7.0% | 100.0% |
| V10-Mid | spatial | success | 0/100 | 0.0% | 0.0% |
| V10-Mid | goal | initial | 100/100 | 100.0% | — |
| V10-Mid | goal | approach | 88/100 | 88.0% | 88.0% |
| V10-Mid | goal | contact | 43/100 | 43.0% | 48.9% |
| V10-Mid | goal | grasp_or_manipulate | 43/100 | 43.0% | 100.0% |
| V10-Mid | goal | transport | 4/100 | 4.0% | 9.3% |
| V10-Mid | goal | goal_alignment | 2/100 | 2.0% | 50.0% |
| V10-Mid | goal | release_or_final_action | 2/100 | 2.0% | 100.0% |
| V10-Mid | goal | success | 0/100 | 0.0% | 0.0% |
| V10-Mid | 10 | initial | 100/100 | 100.0% | — |
| V10-Mid | 10 | approach | 100/100 | 100.0% | 100.0% |
| V10-Mid | 10 | contact | 74/100 | 74.0% | 74.0% |
| V10-Mid | 10 | grasp_or_manipulate | 72/100 | 72.0% | 97.3% |
| V10-Mid | 10 | transport | 9/100 | 9.0% | 12.5% |
| V10-Mid | 10 | goal_alignment | 1/100 | 1.0% | 11.1% |
| V10-Mid | 10 | release_or_final_action | 0/100 | 0.0% | 0.0% |
| V10-Mid | 10 | success | 0/100 | 0.0% | — |
| AE-D7 | object | initial | 100/100 | 100.0% | — |
| AE-D7 | object | approach | 100/100 | 100.0% | 100.0% |
| AE-D7 | object | contact | 100/100 | 100.0% | 100.0% |
| AE-D7 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| AE-D7 | object | transport | 100/100 | 100.0% | 100.0% |
| AE-D7 | object | goal_alignment | 99/100 | 99.0% | 99.0% |
| AE-D7 | object | release_or_final_action | 99/100 | 99.0% | 100.0% |
| AE-D7 | object | success | 97/100 | 97.0% | 98.0% |
| AE-D7 | spatial | initial | 100/100 | 100.0% | — |
| AE-D7 | spatial | approach | 100/100 | 100.0% | 100.0% |
| AE-D7 | spatial | contact | 100/100 | 100.0% | 100.0% |
| AE-D7 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| AE-D7 | spatial | transport | 100/100 | 100.0% | 100.0% |
| AE-D7 | spatial | goal_alignment | 100/100 | 100.0% | 100.0% |
| AE-D7 | spatial | release_or_final_action | 100/100 | 100.0% | 100.0% |
| AE-D7 | spatial | success | 96/100 | 96.0% | 96.0% |
| AE-D5 | object | initial | 100/100 | 100.0% | — |
| AE-D5 | object | approach | 100/100 | 100.0% | 100.0% |
| AE-D5 | object | contact | 100/100 | 100.0% | 100.0% |
| AE-D5 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| AE-D5 | object | transport | 99/100 | 99.0% | 99.0% |
| AE-D5 | object | goal_alignment | 98/100 | 98.0% | 99.0% |
| AE-D5 | object | release_or_final_action | 98/100 | 98.0% | 100.0% |
| AE-D5 | object | success | 98/100 | 98.0% | 100.0% |
| AE-D5 | spatial | initial | 100/100 | 100.0% | — |
| AE-D5 | spatial | approach | 100/100 | 100.0% | 100.0% |
| AE-D5 | spatial | contact | 100/100 | 100.0% | 100.0% |
| AE-D5 | spatial | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| AE-D5 | spatial | transport | 100/100 | 100.0% | 100.0% |
| AE-D5 | spatial | goal_alignment | 99/100 | 99.0% | 99.0% |
| AE-D5 | spatial | release_or_final_action | 99/100 | 99.0% | 100.0% |
| AE-D5 | spatial | success | 98/100 | 98.0% | 99.0% |
| AE-D5 | goal | initial | 100/100 | 100.0% | — |
| AE-D5 | goal | approach | 100/100 | 100.0% | 100.0% |
| AE-D5 | goal | contact | 100/100 | 100.0% | 100.0% |
| AE-D5 | goal | grasp_or_manipulate | 99/100 | 99.0% | 99.0% |
| AE-D5 | goal | transport | 99/100 | 99.0% | 100.0% |
| AE-D5 | goal | goal_alignment | 98/100 | 98.0% | 99.0% |
| AE-D5 | goal | release_or_final_action | 98/100 | 98.0% | 100.0% |
| AE-D5 | goal | success | 98/100 | 98.0% | 100.0% |
| AE-D3 | object | initial | 100/100 | 100.0% | — |
| AE-D3 | object | approach | 100/100 | 100.0% | 100.0% |
| AE-D3 | object | contact | 100/100 | 100.0% | 100.0% |
| AE-D3 | object | grasp_or_manipulate | 100/100 | 100.0% | 100.0% |
| AE-D3 | object | transport | 100/100 | 100.0% | 100.0% |
| AE-D3 | object | goal_alignment | 98/100 | 98.0% | 98.0% |
| AE-D3 | object | release_or_final_action | 98/100 | 98.0% | 100.0% |
| AE-D3 | object | success | 98/100 | 98.0% | 100.0% |

![套件阶段到达率](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/stage_reach_by_suite.png)

## 配对 XYZ 图与视频

蓝色实线和圆点为 Full18，橙色虚线和三角点为变体；真实 MuJoCo 世界坐标，单位米。阶段标注、xyz 同步偏差和 N+1 帧视频见画廊。

[打开 XYZ 轨迹与配对视频画廊](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/xyz_gallery.html)

| 变体 | 套件 | 案例 | 配对结果 | XYZ mean / p95 / max (m) | 图 | 视频 |
|---|---|---|---|---:|---|---|
| Skip-L2 | object | task 9 / state 0 | both_failure | 0.066 / 0.227 / 0.245 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_object__both_failure__task09_ep00.mp4) |
| Skip-L2 | object | task 1 / state 1 | both_success | 0.134 / 0.407 / 0.412 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_object__both_success__task01_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_object__both_success__task01_ep01.mp4) |
| Skip-L2 | object | task 3 / state 0 | new_failure | 0.106 / 0.352 / 0.359 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_object__new_failure__task03_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_object__new_failure__task03_ep00.mp4) |
| Skip-L2 | object | task 0 / state 3 | recovery | 0.146 / 0.445 / 0.456 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_object__recovery__task00_ep03.mp4) |
| Skip-L2 | spatial | task 9 / state 0 | both_failure | 0.105 / 0.369 / 0.379 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_spatial__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_spatial__both_failure__task09_ep00.mp4) |
| Skip-L2 | spatial | task 7 / state 0 | both_success | 0.205 / 0.510 / 0.512 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_spatial__both_success__task07_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_spatial__both_success__task07_ep00.mp4) |
| Skip-L2 | spatial | task 1 / state 7 | new_failure | 0.100 / 0.330 / 0.335 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_spatial__new_failure__task01_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_spatial__new_failure__task01_ep07.mp4) |
| Skip-L2 | spatial | task 3 / state 5 | recovery | 0.040 / 0.151 / 0.159 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_spatial__recovery__task03_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_spatial__recovery__task03_ep05.mp4) |
| Skip-L2 | goal | task 4 / state 2 | both_success | 0.075 / 0.333 / 0.355 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_goal__both_success__task04_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_goal__both_success__task04_ep02.mp4) |
| Skip-L2 | goal | task 4 / state 0 | new_failure | 0.102 / 0.225 / 0.228 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_goal__new_failure__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_goal__new_failure__task04_ep00.mp4) |
| Skip-L2 | goal | task 4 / state 3 | recovery | 0.069 / 0.300 / 0.309 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_goal__recovery__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_goal__recovery__task04_ep03.mp4) |
| Skip-L2 | 10 | task 8 / state 5 | both_failure | 0.365 / 1.586 / 1.697 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_10__both_failure__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_10__both_failure__task08_ep05.mp4) |
| Skip-L2 | 10 | task 8 / state 7 | both_success | 0.072 / 0.398 / 0.428 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_10__both_success__task08_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_10__both_success__task08_ep07.mp4) |
| Skip-L2 | 10 | task 8 / state 1 | new_failure | 0.362 / 0.943 / 1.090 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_10__new_failure__task08_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_10__new_failure__task08_ep01.mp4) |
| Skip-L2 | 10 | task 9 / state 0 | recovery | 0.175 / 0.391 / 0.403 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2__libero_10__recovery__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2__libero_10__recovery__task09_ep00.mp4) |
| Skip-L3 | object | task 9 / state 0 | both_failure | 0.069 / 0.296 / 0.426 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_object__both_failure__task09_ep00.mp4) |
| Skip-L3 | object | task 4 / state 0 | both_success | 0.103 / 0.452 / 0.459 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_object__both_success__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_object__both_success__task04_ep00.mp4) |
| Skip-L3 | object | task 0 / state 8 | new_failure | 0.209 / 0.526 / 0.528 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_object__new_failure__task00_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_object__new_failure__task00_ep08.mp4) |
| Skip-L3 | object | task 0 / state 3 | recovery | 0.150 / 0.458 / 0.475 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_object__recovery__task00_ep03.mp4) |
| Skip-L3 | spatial | task 0 / state 7 | both_failure | 0.126 / 0.354 / 0.357 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_spatial__both_failure__task00_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_spatial__both_failure__task00_ep07.mp4) |
| Skip-L3 | spatial | task 4 / state 1 | both_success | 0.121 / 0.344 / 0.349 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_spatial__both_success__task04_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_spatial__both_success__task04_ep01.mp4) |
| Skip-L3 | spatial | task 3 / state 2 | new_failure | 0.011 / 0.041 / 0.053 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_spatial__new_failure__task03_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_spatial__new_failure__task03_ep02.mp4) |
| Skip-L3 | spatial | task 9 / state 0 | recovery | 0.123 / 0.440 / 0.444 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_spatial__recovery__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_spatial__recovery__task09_ep00.mp4) |
| Skip-L3 | goal | task 4 / state 2 | both_success | 0.097 / 0.346 / 0.354 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_goal__both_success__task04_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_goal__both_success__task04_ep02.mp4) |
| Skip-L3 | goal | task 5 / state 9 | new_failure | 0.167 / 0.397 / 0.403 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_goal__new_failure__task05_ep09.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_goal__new_failure__task05_ep09.mp4) |
| Skip-L3 | goal | task 4 / state 3 | recovery | 0.067 / 0.305 / 0.313 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_goal__recovery__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_goal__recovery__task04_ep03.mp4) |
| Skip-L3 | 10 | task 8 / state 8 | both_failure | 0.219 / 0.796 / 1.042 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_10__both_failure__task08_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_10__both_failure__task08_ep08.mp4) |
| Skip-L3 | 10 | task 7 / state 8 | both_success | 0.183 / 0.420 / 0.429 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_10__both_success__task07_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_10__both_success__task07_ep08.mp4) |
| Skip-L3 | 10 | task 8 / state 7 | new_failure | 0.062 / 0.361 / 0.382 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_10__new_failure__task08_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_10__new_failure__task08_ep07.mp4) |
| Skip-L3 | 10 | task 8 / state 5 | recovery | 0.174 / 0.619 / 0.945 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L3__libero_10__recovery__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L3__libero_10__recovery__task08_ep05.mp4) |
| Skip-L4 | object | task 9 / state 0 | both_failure | 0.075 / 0.251 / 0.403 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_object__both_failure__task09_ep00.mp4) |
| Skip-L4 | object | task 1 / state 1 | both_success | 0.134 / 0.407 / 0.417 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_object__both_success__task01_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_object__both_success__task01_ep01.mp4) |
| Skip-L4 | object | task 5 / state 6 | new_failure | 0.137 / 0.414 / 0.420 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_object__new_failure__task05_ep06.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_object__new_failure__task05_ep06.mp4) |
| Skip-L4 | object | task 0 / state 3 | recovery | 0.172 / 0.514 / 0.539 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_object__recovery__task00_ep03.mp4) |
| Skip-L4 | spatial | task 0 / state 7 | both_failure | 0.105 / 0.276 / 0.314 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_spatial__both_failure__task00_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_spatial__both_failure__task00_ep07.mp4) |
| Skip-L4 | spatial | task 7 / state 0 | both_success | 0.168 / 0.421 / 0.429 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_spatial__both_success__task07_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_spatial__both_success__task07_ep00.mp4) |
| Skip-L4 | spatial | task 5 / state 3 | new_failure | 0.023 / 0.043 / 0.047 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_spatial__new_failure__task05_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_spatial__new_failure__task05_ep03.mp4) |
| Skip-L4 | spatial | task 9 / state 0 | recovery | 0.173 / 0.500 / 0.507 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_spatial__recovery__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_spatial__recovery__task09_ep00.mp4) |
| Skip-L4 | goal | task 4 / state 2 | both_success | 0.071 / 0.316 / 0.324 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_goal__both_success__task04_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_goal__both_success__task04_ep02.mp4) |
| Skip-L4 | goal | task 4 / state 0 | new_failure | 0.091 / 0.307 / 0.329 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_goal__new_failure__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_goal__new_failure__task04_ep00.mp4) |
| Skip-L4 | goal | task 4 / state 3 | recovery | 0.102 / 0.326 / 0.331 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_goal__recovery__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_goal__recovery__task04_ep03.mp4) |
| Skip-L4 | 10 | task 8 / state 5 | both_failure | 0.358 / 1.584 / 1.647 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_10__both_failure__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_10__both_failure__task08_ep05.mp4) |
| Skip-L4 | 10 | task 7 / state 8 | both_success | 0.183 / 0.408 / 0.419 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_10__both_success__task07_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_10__both_success__task07_ep08.mp4) |
| Skip-L4 | 10 | task 8 / state 1 | new_failure | 0.199 / 0.828 / 1.058 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_10__new_failure__task08_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_10__new_failure__task08_ep01.mp4) |
| Skip-L4 | 10 | task 4 / state 5 | recovery | 0.079 / 0.276 / 0.315 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L4__libero_10__recovery__task04_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L4__libero_10__recovery__task04_ep05.mp4) |
| Skip-L5 | object | task 1 / state 1 | both_success | 0.133 / 0.406 / 0.415 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_object__both_success__task01_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_object__both_success__task01_ep01.mp4) |
| Skip-L5 | object | task 9 / state 0 | recovery | 0.066 / 0.321 / 0.333 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_object__recovery__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_object__recovery__task09_ep00.mp4) |
| Skip-L5 | spatial | task 9 / state 0 | both_failure | 0.140 / 0.466 / 0.473 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_spatial__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_spatial__both_failure__task09_ep00.mp4) |
| Skip-L5 | spatial | task 4 / state 1 | both_success | 0.131 / 0.374 / 0.382 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_spatial__both_success__task04_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_spatial__both_success__task04_ep01.mp4) |
| Skip-L5 | spatial | task 7 / state 2 | new_failure | 0.084 / 0.353 / 0.359 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_spatial__new_failure__task07_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_spatial__new_failure__task07_ep02.mp4) |
| Skip-L5 | spatial | task 3 / state 5 | recovery | 0.039 / 0.145 / 0.152 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_spatial__recovery__task03_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_spatial__recovery__task03_ep05.mp4) |
| Skip-L5 | goal | task 4 / state 2 | both_success | 0.078 / 0.329 / 0.339 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_goal__both_success__task04_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_goal__both_success__task04_ep02.mp4) |
| Skip-L5 | goal | task 3 / state 9 | new_failure | 0.179 / 0.289 / 0.294 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_goal__new_failure__task03_ep09.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_goal__new_failure__task03_ep09.mp4) |
| Skip-L5 | goal | task 4 / state 3 | recovery | 0.078 / 0.303 / 0.312 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_goal__recovery__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_goal__recovery__task04_ep03.mp4) |
| Skip-L5 | 10 | task 8 / state 5 | both_failure | 0.340 / 1.549 / 1.614 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_10__both_failure__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_10__both_failure__task08_ep05.mp4) |
| Skip-L5 | 10 | task 7 / state 8 | both_success | 0.182 / 0.414 / 0.420 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_10__both_success__task07_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_10__both_success__task07_ep08.mp4) |
| Skip-L5 | 10 | task 8 / state 1 | new_failure | 0.243 / 0.652 / 0.929 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_10__new_failure__task08_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_10__new_failure__task08_ep01.mp4) |
| Skip-L5 | 10 | task 9 / state 0 | recovery | 0.183 / 0.422 / 0.449 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L5__libero_10__recovery__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L5__libero_10__recovery__task09_ep00.mp4) |
| Skip-L2-5 | object | task 9 / state 0 | both_failure | 0.269 / 0.408 / 0.418 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_object__both_failure__task09_ep00.mp4) |
| Skip-L2-5 | object | task 4 / state 4 | both_success | 0.231 / 0.560 / 0.567 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_object__both_success__task04_ep04.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_object__both_success__task04_ep04.mp4) |
| Skip-L2-5 | object | task 7 / state 8 | new_failure | 0.214 / 0.573 / 0.579 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_object__new_failure__task07_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_object__new_failure__task07_ep08.mp4) |
| Skip-L2-5 | object | task 0 / state 3 | recovery | 0.235 / 0.530 / 0.544 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_object__recovery__task00_ep03.mp4) |
| Skip-L2-5 | spatial | task 7 / state 8 | both_failure | 0.146 / 0.358 / 0.392 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_spatial__both_failure__task07_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_spatial__both_failure__task07_ep08.mp4) |
| Skip-L2-5 | spatial | task 4 / state 5 | both_success | 0.161 / 0.366 / 0.377 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_spatial__both_success__task04_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_spatial__both_success__task04_ep05.mp4) |
| Skip-L2-5 | spatial | task 9 / state 5 | new_failure | 0.174 / 0.461 / 0.486 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_spatial__new_failure__task09_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_spatial__new_failure__task09_ep05.mp4) |
| Skip-L2-5 | spatial | task 5 / state 1 | recovery | 0.169 / 0.276 / 0.281 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_spatial__recovery__task05_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_spatial__recovery__task05_ep01.mp4) |
| Skip-L2-5 | goal | task 4 / state 3 | both_failure | 0.235 / 0.348 / 0.352 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_goal__both_failure__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_goal__both_failure__task04_ep03.mp4) |
| Skip-L2-5 | goal | task 3 / state 0 | both_success | 0.144 / 0.277 / 0.288 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_goal__both_success__task03_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_goal__both_success__task03_ep00.mp4) |
| Skip-L2-5 | goal | task 2 / state 0 | new_failure | 0.202 / 0.461 / 0.464 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_goal__new_failure__task02_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_goal__new_failure__task02_ep00.mp4) |
| Skip-L2-5 | goal | task 8 / state 5 | recovery | 0.116 / 0.208 / 0.220 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_goal__recovery__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_goal__recovery__task08_ep05.mp4) |
| Skip-L2-5 | 10 | task 8 / state 5 | both_failure | 0.421 / 1.620 / 1.684 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_10__both_failure__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_10__both_failure__task08_ep05.mp4) |
| Skip-L2-5 | 10 | task 1 / state 9 | both_success | 0.213 / 0.493 / 0.500 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_10__both_success__task01_ep09.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_10__both_success__task01_ep09.mp4) |
| Skip-L2-5 | 10 | task 9 / state 9 | new_failure | 0.202 / 0.659 / 0.741 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_10__new_failure__task09_ep09.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_10__new_failure__task09_ep09.mp4) |
| Skip-L2-5 | 10 | task 5 / state 0 | recovery | 0.106 / 0.241 / 0.256 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/Skip_L2_5__libero_10__recovery__task05_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/Skip_L2_5__libero_10__recovery__task05_ep00.mp4) |
| V16 | object | task 9 / state 0 | both_failure | 0.069 / 0.283 / 0.395 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_object__both_failure__task09_ep00.mp4) |
| V16 | object | task 4 / state 0 | both_success | 0.159 / 0.410 / 0.416 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_object__both_success__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_object__both_success__task04_ep00.mp4) |
| V16 | object | task 5 / state 7 | recovery | 0.022 / 0.077 / 0.090 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_object__recovery__task05_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_object__recovery__task05_ep07.mp4) |
| V16 | spatial | task 7 / state 0 | both_success | 0.167 / 0.417 / 0.425 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_spatial__both_success__task07_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_spatial__both_success__task07_ep00.mp4) |
| V16 | spatial | task 4 / state 3 | new_failure | 0.146 / 0.429 / 0.459 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_spatial__new_failure__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_spatial__new_failure__task04_ep03.mp4) |
| V16 | spatial | task 9 / state 0 | recovery | 0.173 / 0.530 / 0.532 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_spatial__recovery__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_spatial__recovery__task09_ep00.mp4) |
| V16 | goal | task 4 / state 2 | both_success | 0.072 / 0.305 / 0.313 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_goal__both_success__task04_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_goal__both_success__task04_ep02.mp4) |
| V16 | goal | task 3 / state 4 | new_failure | 0.141 / 0.365 / 0.378 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_goal__new_failure__task03_ep04.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_goal__new_failure__task03_ep04.mp4) |
| V16 | goal | task 4 / state 3 | recovery | 0.076 / 0.271 / 0.277 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_goal__recovery__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_goal__recovery__task04_ep03.mp4) |
| V16 | 10 | task 8 / state 5 | both_failure | 0.325 / 1.271 / 1.412 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_10__both_failure__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_10__both_failure__task08_ep05.mp4) |
| V16 | 10 | task 1 / state 8 | both_success | 0.203 / 0.495 / 0.501 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_10__both_success__task01_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_10__both_success__task01_ep08.mp4) |
| V16 | 10 | task 8 / state 1 | new_failure | 0.242 / 0.605 / 0.744 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_10__new_failure__task08_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_10__new_failure__task08_ep01.mp4) |
| V16 | 10 | task 4 / state 5 | recovery | 0.109 / 0.306 / 0.333 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V16__libero_10__recovery__task04_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V16__libero_10__recovery__task04_ep05.mp4) |
| V14 | object | task 9 / state 0 | both_failure | 0.059 / 0.315 / 0.395 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_object__both_failure__task09_ep00.mp4) |
| V14 | object | task 4 / state 4 | both_success | 0.240 / 0.578 / 0.586 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_object__both_success__task04_ep04.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_object__both_success__task04_ep04.mp4) |
| V14 | object | task 4 / state 5 | new_failure | 0.239 / 0.633 / 0.638 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_object__new_failure__task04_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_object__new_failure__task04_ep05.mp4) |
| V14 | object | task 0 / state 3 | recovery | 0.155 / 0.460 / 0.481 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_object__recovery__task00_ep03.mp4) |
| V14 | spatial | task 9 / state 0 | both_failure | 0.227 / 0.528 / 0.535 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_spatial__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_spatial__both_failure__task09_ep00.mp4) |
| V14 | spatial | task 1 / state 7 | both_success | 0.114 / 0.328 / 0.334 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_spatial__both_success__task01_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_spatial__both_success__task01_ep07.mp4) |
| V14 | spatial | task 9 / state 5 | new_failure | 0.170 / 0.529 / 0.556 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_spatial__new_failure__task09_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_spatial__new_failure__task09_ep05.mp4) |
| V14 | spatial | task 7 / state 8 | recovery | 0.125 / 0.236 / 0.239 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_spatial__recovery__task07_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_spatial__recovery__task07_ep08.mp4) |
| V14 | goal | task 4 / state 3 | both_failure | 0.136 / 0.229 / 0.264 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_goal__both_failure__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_goal__both_failure__task04_ep03.mp4) |
| V14 | goal | task 2 / state 4 | both_success | 0.123 / 0.306 / 0.313 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_goal__both_success__task02_ep04.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_goal__both_success__task02_ep04.mp4) |
| V14 | goal | task 7 / state 4 | new_failure | 0.246 / 0.389 / 0.392 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_goal__new_failure__task07_ep04.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_goal__new_failure__task07_ep04.mp4) |
| V14 | goal | task 8 / state 5 | recovery | 0.074 / 0.145 / 0.151 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_goal__recovery__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_goal__recovery__task08_ep05.mp4) |
| V14 | 10 | task 8 / state 8 | both_failure | 0.533 / 1.361 / 1.808 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_10__both_failure__task08_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_10__both_failure__task08_ep08.mp4) |
| V14 | 10 | task 4 / state 9 | both_success | 0.174 / 0.522 / 0.551 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_10__both_success__task04_ep09.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_10__both_success__task04_ep09.mp4) |
| V14 | 10 | task 9 / state 8 | new_failure | 0.412 / 1.072 / 1.351 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V14__libero_10__new_failure__task09_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V14__libero_10__new_failure__task09_ep08.mp4) |
| V12 | object | task 9 / state 0 | both_failure | 0.308 / 0.388 / 0.391 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_object__both_failure__task09_ep00.mp4) |
| V12 | object | task 7 / state 5 | new_failure | 0.235 / 0.482 / 0.488 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_object__new_failure__task07_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_object__new_failure__task07_ep05.mp4) |
| V12 | spatial | task 9 / state 0 | both_failure | 0.272 / 0.410 / 0.418 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_spatial__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_spatial__both_failure__task09_ep00.mp4) |
| V12 | spatial | task 9 / state 1 | new_failure | 0.255 / 0.475 / 0.486 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_spatial__new_failure__task09_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_spatial__new_failure__task09_ep01.mp4) |
| V12 | goal | task 4 / state 3 | both_failure | 0.216 / 0.349 / 0.350 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_goal__both_failure__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_goal__both_failure__task04_ep03.mp4) |
| V12 | goal | task 9 / state 2 | new_failure | 0.223 / 0.524 / 0.526 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_goal__new_failure__task09_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_goal__new_failure__task09_ep02.mp4) |
| V12 | 10 | task 9 / state 2 | both_failure | 0.589 / 1.626 / 1.661 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_10__both_failure__task09_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_10__both_failure__task09_ep02.mp4) |
| V12 | 10 | task 9 / state 5 | new_failure | 0.267 / 0.591 / 0.621 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V12__libero_10__new_failure__task09_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V12__libero_10__new_failure__task09_ep05.mp4) |
| V10-Mid | object | task 9 / state 0 | both_failure | 0.207 / 0.304 / 0.391 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_object__both_failure__task09_ep00.mp4) |
| V10-Mid | object | task 4 / state 3 | new_failure | 0.262 / 0.475 / 0.483 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_object__new_failure__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_object__new_failure__task04_ep03.mp4) |
| V10-Mid | spatial | task 9 / state 0 | both_failure | 0.299 / 0.428 / 0.439 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_spatial__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_spatial__both_failure__task09_ep00.mp4) |
| V10-Mid | spatial | task 9 / state 8 | new_failure | 0.293 / 0.447 / 0.452 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_spatial__new_failure__task09_ep08.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_spatial__new_failure__task09_ep08.mp4) |
| V10-Mid | goal | task 4 / state 3 | both_failure | 0.256 / 0.374 / 0.402 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_goal__both_failure__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_goal__both_failure__task04_ep03.mp4) |
| V10-Mid | goal | task 9 / state 2 | new_failure | 0.227 / 0.533 / 0.537 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_goal__new_failure__task09_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_goal__new_failure__task09_ep02.mp4) |
| V10-Mid | 10 | task 8 / state 5 | both_failure | 0.511 / 1.383 / 1.404 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_10__both_failure__task08_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_10__both_failure__task08_ep05.mp4) |
| V10-Mid | 10 | task 8 / state 2 | new_failure | 0.285 / 0.533 / 0.544 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/V10_Mid__libero_10__new_failure__task08_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/V10_Mid__libero_10__new_failure__task08_ep02.mp4) |
| AE-D7 | object | task 9 / state 0 | both_failure | 0.058 / 0.231 / 0.254 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_object__both_failure__task09_ep00.mp4) |
| AE-D7 | object | task 4 / state 0 | both_success | 0.126 / 0.412 / 0.417 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_object__both_success__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_object__both_success__task04_ep00.mp4) |
| AE-D7 | object | task 0 / state 5 | new_failure | 0.182 / 0.520 / 0.543 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_object__new_failure__task00_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_object__new_failure__task00_ep05.mp4) |
| AE-D7 | object | task 0 / state 3 | recovery | 0.125 / 0.333 / 0.337 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_object__recovery__task00_ep03.mp4) |
| AE-D7 | spatial | task 0 / state 7 | both_failure | 0.034 / 0.131 / 0.180 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_spatial__both_failure__task00_ep07.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_spatial__both_failure__task00_ep07.mp4) |
| AE-D7 | spatial | task 9 / state 1 | both_success | 0.043 / 0.186 / 0.204 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_spatial__both_success__task09_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_spatial__both_success__task09_ep01.mp4) |
| AE-D7 | spatial | task 7 / state 2 | new_failure | 0.053 / 0.146 / 0.149 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_spatial__new_failure__task07_ep02.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_spatial__new_failure__task07_ep02.mp4) |
| AE-D7 | spatial | task 3 / state 5 | recovery | 0.028 / 0.145 / 0.146 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D7__libero_spatial__recovery__task03_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D7__libero_spatial__recovery__task03_ep05.mp4) |
| AE-D5 | object | task 9 / state 0 | both_failure | 0.068 / 0.270 / 0.297 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_object__both_failure__task09_ep00.mp4) |
| AE-D5 | object | task 1 / state 1 | both_success | 0.126 / 0.396 / 0.405 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_object__both_success__task01_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_object__both_success__task01_ep01.mp4) |
| AE-D5 | object | task 4 / state 0 | new_failure | 0.056 / 0.169 / 0.173 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_object__new_failure__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_object__new_failure__task04_ep00.mp4) |
| AE-D5 | object | task 0 / state 3 | recovery | 0.124 / 0.329 / 0.333 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_object__recovery__task00_ep03.mp4) |
| AE-D5 | spatial | task 9 / state 0 | both_failure | 0.152 / 0.477 / 0.493 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_spatial__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_spatial__both_failure__task09_ep00.mp4) |
| AE-D5 | spatial | task 9 / state 1 | both_success | 0.040 / 0.174 / 0.201 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_spatial__both_success__task09_ep01.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_spatial__both_success__task09_ep01.mp4) |
| AE-D5 | spatial | task 3 / state 5 | recovery | 0.050 / 0.127 / 0.156 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_spatial__recovery__task03_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_spatial__recovery__task03_ep05.mp4) |
| AE-D5 | goal | task 4 / state 0 | both_success | 0.091 / 0.320 / 0.325 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_goal__both_success__task04_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_goal__both_success__task04_ep00.mp4) |
| AE-D5 | goal | task 3 / state 0 | new_failure | 0.063 / 0.364 / 0.375 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_goal__new_failure__task03_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_goal__new_failure__task03_ep00.mp4) |
| AE-D5 | goal | task 4 / state 3 | recovery | 0.057 / 0.267 / 0.273 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D5__libero_goal__recovery__task04_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D5__libero_goal__recovery__task04_ep03.mp4) |
| AE-D3 | object | task 9 / state 0 | both_failure | 0.080 / 0.221 / 0.332 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D3__libero_object__both_failure__task09_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D3__libero_object__both_failure__task09_ep00.mp4) |
| AE-D3 | object | task 0 / state 5 | both_success | 0.153 / 0.496 / 0.502 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D3__libero_object__both_success__task00_ep05.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D3__libero_object__both_success__task00_ep05.mp4) |
| AE-D3 | object | task 3 / state 0 | new_failure | 0.121 / 0.381 / 0.397 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D3__libero_object__new_failure__task03_ep00.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D3__libero_object__new_failure__task03_ep00.mp4) |
| AE-D3 | object | task 0 / state 3 | recovery | 0.191 / 0.533 / 0.552 | [图](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_plots/AE_D3__libero_object__recovery__task00_ep03.png) | [视频](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/analysis/pilot/xyz_videos/AE_D3__libero_object__recovery__task00_ep03.mp4) |

## Action Expert 降噪步数补充实验

本节保持 VLM Full18、Action Expert Full18，对比 D7、D5、D3 与同状态/噪声的 D10 Full18 baseline。初筛成功率 ≥90% 或距 D10 baseline 不超过 2 个百分点的变体×套件扩展到每任务 50 个状态。推理时间列为 episode 级 policy latency；各次 denoise timing 与实际步数记录在 infer.jsonl。

已有 Action Expert early-exit 层探针（200 个真实观测）见 [Experiment 1 分析](results/pi0_intermediate_probe/experiment1_report.md)。

## 可复核数据

- [套件汇总 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/suite_success.csv)
- [逐任务汇总 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/task_success.csv)
- [阶段到达与配对 cohort CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/failure_stage_analysis.csv)
- [配对失败与恢复 episode CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/paired_failure_cohorts.csv)
- [XYZ 案例索引 CSV](results/experiment9_standard_vlm_pruning_20261004/analysis/pilot/xyz_case_index.csv)
- [运行器事件日志](results/experiment9_standard_vlm_pruning_20261004/runner_events.jsonl)

标准套件 Full18 的历史基线（2026-10-03）为 Spatial 95.4%、Object 99.2%、Goal 96.8%、LIBERO-10 86.8%。本轮配对对照以本轮重跑 Full18 为准，历史数据只作背景。
