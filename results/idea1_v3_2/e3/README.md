# E3 最终产物入口

最终中文报告：[revision2/report_zh.md](revision2/report_zh.md)。

三类方法的最佳校准候选均未通过预设尾部约束。评估集首5步增量控制目标代理L2：WLP 9.276 mm、ConstantTurn 9.732 mm、VAR 9.831 mm；冻结Kalman 8.478 mm、旧Kalman 9.030 mm、HoldLast 10.576 mm、同观测V14 7.446 mm。这些不是实际EEF误差。

方向分析以revision2为准：明确区分increment与cumulative，各自包含L2、signed/absolute along和lateral。逐样本正交分解审计最大平方残差为1.025e−8 mm²。原始报告及CSV保留历史v2.1命名：`l2_mm`是单步增量，`along_mm/lateral_mm`是累计代理；不可混当同一向量的分量。

数据/审计：source_receipt.json、baseline_receipt.json、supplement_audit.json、revision2/receipt.json。source_snapshot保留与初始执行manifest哈希一致的源码。predictions.npz保存全部6345点×50步冻结预测和已观测创新；frozen_winners.json在评估前保存51候选搜索的每类选择。

补充表：paired_bootstrap.csv、suites.csv、innovation_summary.csv、time_shift_summary.csv；显式方向的窗口/逐offset/逐阶段表位于revision2/。

复现命令（仓库根目录，使用现有环境；主实验目标目录已存在时会拒绝覆盖）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python -m unittest preparations.idea1_v3_2.test_predictor preparations.idea1_v3_2.test_predictor_metrics -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_2.predictor_experiment
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_2.predictor_audit
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 policies/realtime-vla-flash/.venv/bin/python -m preparations.idea1_v3_2.predictor_revision2
```

11项单元测试已通过；主实验耗时98.5秒，无模型调用、无仿真动作、无GPU。
