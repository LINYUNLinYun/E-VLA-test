# 联合层组合的离线代理图

后续闭环检验已完成且未复现任务偏好：两路径成功60/80、62/80，Object/Spatial均19/20，Long均6/20；[完整阴性结果](../ae_complement_beam_v2/README_zh.md)保留。下方图是筛选阶段的离线误差快照，图题中的“成功率尚未验证”描述其生成时状态，不能作为当前互补性结论。

[PNG](AE16_joint_proxy.png)、[PDF](AE16_joint_proxy.pdf)、[SVG](AE16_joint_proxy.svg)、[CSV](proxy_metrics.csv)及[来源哈希](provenance.json)来自完整110条历史观测的冻结分析与独立复算。图已检查，未读取正在运行的闭环成功率。

A跳过[2,13]，B跳过[7,14]，均为AE16/D2。左图为四suite的task等权NMSE；中、右图为两个冻结方向逐task的配对误差差值，正值对应预期侧重。每task先平均观测内三组噪声，再平均已有快照；不是把噪声或快照当作独立task。

这是筛选后代理现象，不能证明动作成功率、独立偏好复现或加速。初始快照子集的最佳组合不同，不能据此替换主选择；旧Full轨迹状态也不代表新路径闭环分布。完整定义及限制见[筛选报告](../ae_complement_beam_v1/README_zh.md)，真正的质量检验见[闭环实验](../ae_complement_beam_v2/README_zh.md)。
