# v3 终稿衍生小表与计算口径

2026-10-20。仅从已生成CSV作有限CPU聚合，无新实验、视频解码或XYZ重绘。新数据来源统一为 [`final_joint_1976_20261020`](../../../policies/realtime-vla-flash/results/idea1_v3/final_joint_1976_20261020/)。校准与科学样本同名condition，所有新指标均显式过滤stage=E1或E3，未混入calibration。

## 历史共同120：描述参照，不是正式因果配对

旧来源为[v2.2 trajectory_index.csv](../../../policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z/quick_analysis/quick_final_light_repeat_v3_20261019T0945Z/trajectory_index.csv)，filter=`condition in {pi0_V14_A03_u50,pi0_full_u50}, init in {6,7,8}`。新来源为联合`trajectory_index.csv`的对应D2科学condition，同suite/task/init交集各120；四格依次是双成功／旧成功新失败／旧失败新成功／双失败。旧mixed revision、limits、后端／计时可比性限制保留，无正式CI或速度比。

| 新条件 | 旧参照 | 旧成功 | 新成功 | 双成功 | 旧成功新失败 | 旧失败新成功 | 双失败 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 D2参考 .15/P2 | 纯V14-D2 | 85/120 | 107/120 | 82 | 3 | 25 | 10 |
| E1 D2选中 .10/P2 | 纯V14-D2 | 85/120 | 110/120 | 83 | 2 | 27 | 8 |
| E3 .10/P2 | 纯V14-D2 | 85/120 | 110/120 | 81 | 4 | 29 | 6 |
| E1 D2参考 .15/P2 | Full-D2 | 109/120 | 107/120 | 98 | 11 | 9 | 2 |
| E1 D2选中 .10/P2 | Full-D2 | 109/120 | 110/120 | 102 | 7 | 8 | 3 |
| E3 .10/P2 | Full-D2 | 109/120 | 110/120 | 102 | 7 | 8 | 3 |

## Future 1–5：episode等权距离

使用联合[`future_error_by_phase_step_branch_per_episode.csv`](../../../policies/realtime-vla-flash/results/idea1_v3/final_joint_1976_20261020/future_error_by_phase_step_branch_per_episode.csv)。固定condition/future/branch，先按该episode各phase的`observations`加权合并`distance_mean`，再对episode均值等权；不是对phase均值直接平均。距离为前六维`L2/sqrt(6)`模型归一化空间，非毫米、真实EEF或fresh Full动作误差；历史prefix不入样本。每格200独立episode；每future、每branch的观测数E1=5146、E3=5350。

| 当前未来步 | E1父 t=.10 | E1父 t=.05 | E3 t=.10 | E3 t=.05 |
| --- | ---: | ---: | ---: | ---: |
| 1 | .06666 | .03498 | .08716 | .04876 |
| 2 | .05351 | .02775 | .05841 | .03100 |
| 3 | .04917 | .02593 | .05066 | .02793 |
| 4 | .04736 | .02506 | .04755 | .02647 |
| 5 | .04649 | .02470 | .04616 | .02589 |

E1父=`E1_D2_tau0.1_P2`，E3=`E3_D2_tau0.1_P2_previous_anchor`。t大小影响加噪／端点重建缩放，不能将t=.05较小说成独立准确性更高。两方法越往后误差并未增大；E3最明显差异在未来第1步更大，未显示历史前缀普遍降低代理误差。

## 关键阶段：误差与路由分母分开

误差仍从上述per_episode表计算，选t=.10，在同episode、同phase内按观测数合并未来1–5步，再跨有误差观测的episode等权。V为verify请求数，对应5V个未来向量；E为有误差观测的独立episode数。路由来自联合[`phase_routing.csv`](../../../policies/realtime-vla-flash/results/idea1_v3/final_joint_1976_20261020/phase_routing.csv)，零接受率=zero_accept/V（包含夹爪触发），Full率=Full请求/全部请求，均为request-weighted而非episode等权。一个请求可能同时verify与Full，不能将两者率相加当分区。

| 阶段 | 条件 | t=.10距离 | V / E | 零接受 / V | Full / 全请求 |
| --- | --- | ---: | ---: | ---: | ---: |
| 接近 | E1父 | .05145 | 1191 / 199 | 170/1191（14.3%） | 681/1702（40.0%） |
| 接近 | E3 | .05262 | 1230 / 200 | 421/1230（34.2%） | 806/1615（49.9%） |
| 接触 | E1父 | .05213 | 262 / 126 | 101/262（38.5%） | 182/343（53.1%） |
| 接触 | E3 | .05181 | 282 / 129 | 143/282（50.7%） | 208/347（59.9%） |
| 抓取/操作 | E1父 | .04839 | 465 / 181 | 40/465（8.6%） | 209/634（33.0%） |
| 抓取/操作 | E3 | .05317 | 440 / 186 | 92/440（20.9%） | 230/578（39.8%） |
| 搬运 | E1父 | .05846 | 1336 / 167 | 314/1336（23.5%） | 779/1801（43.3%） |
| 搬运 | E3 | .05796 | 1354 / 174 | 507/1354（37.4%） | 904/1751（51.6%） |
| 对齐 | E1父 | .05581 | 1333 / 153 | 354/1333（26.6%） | 771/1750（44.1%） |
| 对齐 | E3 | .06320 | 1527 / 158 | 726/1527（47.5%） | 1083/1884（57.5%） |

这是关键phase子集，不将各行episode数相加。两条件路由phase=unknown均0条，不代表contact/grasp负值可靠；负值availability_unknown分母仍在原CSV。标签依据当前重规划状态，而E3 verifier state可更旧。E3在所列阶段均更常零接受及Full接管，误差不一致下降；此为代理一致性／路由描述，不是失败致因判断。
