# 固定前两个 Spatial task 的预览

使用 task0/1 各 init0–2，共 6 条真实轨迹、112 个观测。init0/1 训练，init2 留出。完整实验仍在提取 2326 个观测，本预览不替代全四套件结果。

两个预览 task 分别在 GPU1／GPU2 的持续服务上采集，task 与服务绑定，任务分离仅供流程检查；正式报告另提供同服务内 5-task 对照。每个 task 的 phase 对照保持同一服务。

- [二维／三维与前四维配对交互图](pca_explorer.html)
- [H0 与最后一层的 PCA 静态图](pca_preview.png)
- [task 与同 task 内 phase 的留出读出数值](probe_preview.csv)
- [样本量与输入基线核对](summary.json)

phase 是 trace 推断标签。本配置关闭离散机器人状态输入，H0 语言 token 的同 task 内特征应固定。PCA 使用 float64；仅将训练数据中数值上未确定的零空间坐标置零，保留有效低方差方向；原始特征 probe 不受影响。
此预览不用于挑选最终分析参数、样本或最终结论。
