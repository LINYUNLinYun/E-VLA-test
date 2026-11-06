# π0.5 LIBERO 表征探索

方案位于根目录 `idea1_可视化探索性实验.md`；执行记录在
`docs/experiments/idea1/idea1_visualization_execution_status.md`。

## 模块

| 模块 | 用途 |
| --- | --- |
| `common.py` | 固定协议、稳定任务噪声、物理 GPU／EGL 绑定 |
| `server.py` | 持续原生 JAX 官方 checkpoint 服务、原始／解码动作及显式噪声 |
| `rollout.py` | 完整 LIBERO 轨迹、逐推理输入和逐控制步 trace |
| `selection.py` | 带证据的 phase 规则、12＋8 选点及去重／补齐 |
| `features.py` | 原始 Gemma block／scan 插桩、Vision／VLM／AE 全层数据与数值核验 |
| `pooling.py` | 复用 float32 转换及逐层范数的 CPU pooling／残差统计 |
| `extract.py` | 对所选请求重放提取，逐请求保存身份和完成标记 |
| `controller.py` | 双服务采集、提取和分析阶段调度 |
| `analyze.py` | 留出轨迹 probe、PCA、聚类稳定性、残差与独立交互 HTML |
| `render_pca.py` | 从已保存的 PCA 坐标重绘静态图、补全阶段图例，不重拟合分析 |
| `suite_preferences.py` | 基于既有 probe 表比较 suite 的模块／深度偏好，聚类及 task bootstrap；不重跑模型 |
| `joint_suite_clustering.py` | 四个 suite 的真实 pooled hidden state 联合 PCA／k-means，按 suite 与簇并排着色，可切换层、k、种子和预处理 |
| `task_pair_separability.py` | 具体 task 对的真实散点、k=2 聚类和留出轨迹线性读出；含同服务与同 phase 对照 |
| `goal_h9_phases.py` | Goal 各 task 内 H9 语言特征的阶段散点、训练集 PCA／k-means 和留出阶段读出；只用 CPU 复用已有特征 |
| `preview.py` | 固定前两个 Spatial task 的真实数据预览，不替代最终报告 |
| `audit.py` | 全部所选输入／噪声哈希与原始控制步、执行前缀对齐检查 |
| `audit_features.py` | 可复用的逐 NPY 结构、mask 布局、来源及最终动作核验；不重读大 hidden 张量 |
| `storage.py` | 无损压缩 profiling 辅助入口；本轮未启用压缩存储 |

## 本轮输出

根目录：`results/idea1_visualization/run_20261026`。本轮 120 条轨迹、2326 个所选观测及完整分析已完成，
报告为 `analysis/report_zh.md`，交互图为 `analysis/pca_explorer.html`，最终核验为 `final_audit.json`。
当次服务使用端口 48161／48162 和 GPU1／GPU2，结束后均已退出；端口与设备不是长期资源约定。

每个所选请求对应一个目录，包含 `activations.npz`、`pooled.npz`、
`residual_stats.npz`、`metadata.json`、`sample.json`、`parity.json`、`complete.json`。
完整激活按原生精度保存；BF16 存为 uint16 位模式，读取时依据 metadata 的 dtype
使用 `array.view(ml_dtypes.bfloat16)` 恢复，再按需转为 float32。分析直接读取小型
`pooled.npz`，不重复加载全部激活。

VLM 完整激活的 token 轴仅保留有效 prefix token；同时保存原始完整布局的
`prefix_mask`、prompt token/mask 与 image masks，可据此还原原序列索引。
无效相机的完整视觉激活不落盘。元数据中的 `extract_s` 包含前向及设备到主机取回，
`pool_s` 为主机统计，`pack_s` 为有效 token 打包，`save_s` 为 NPZ 保存，`total_s` 为请求总耗时。

## 检查

四套件联合特征散点与 k-means（复用 pooled 特征，不运行策略）：

```bash
python3 -m preparations.idea1_visualization.joint_suite_clustering \
  --root results/idea1_visualization/run_20261026 \
  --output results/idea1_visualization/run_20261026/analysis/joint_suite_scatter_20261028
python3 -m unittest preparations.idea1_visualization.test_joint_suite_clustering
```

交互入口为输出目录中的 `joint_suite_explorer.html`。同一层的四个 suite 联合拟合 PCA，
k-means 在至多 64 维 PCA 中拟合；suite 标签仅用于着色和事后评价。
`--render-only` 可从保存的坐标／簇重绘，不重新加载特征或拟合模型。

suite 偏好补充分析复用既有表格，无需模型服务：

```bash
python3 -m preparations.idea1_visualization.suite_preferences \
  --analysis results/idea1_visualization/run_20261026/analysis \
  --output results/idea1_visualization/run_20261026/analysis/suite_preferences_20261028
python3 -m unittest preparations.idea1_visualization.test_suite_preferences
```

输出模块／层偏好 CSV、四张对照图、suite 与层聚类、task bootstrap 区间和中文 HTML 报告。
偏好描述固定 probe 的相对可读出性，不测量动作因果重要性。补充 PDF 为该 HTML 的静态导出。

```bash
python3 -m unittest preparations.idea1_visualization.test_selection
python3 -m unittest preparations.idea1_visualization.test_analysis preparations.idea1_visualization.test_pooling
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/root/nfs/embodied/benchmark:/root/nfs/embodied/benchmark/policies/openpi/src \
  policies/openpi/.venv/bin/python -m unittest preparations.idea1_visualization.test_features
```

rollout 使用 `policies/libero-eval-v21/.venv/bin/python`，模型服务使用
`policies/openpi/.venv/bin/python`，分析使用系统 Python 的 NumPy／SciPy／sklearn／matplotlib。
独立交互 HTML 使用本包 `_deps/` 中局部安装的 Plotly 7.1.0（含 narwhals／packaging），
没有修改已有策略环境。

## Goal H9 阶段可视化补充（2026-11-01）

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 python3 -m preparations.idea1_visualization.goal_h9_phases \
  --root results/idea1_visualization/run_20261026 \
  --output results/idea1_visualization/run_20261026/analysis/goal_h9_phases_20261101
```

输出目录含 `goal_H09_phase_explorer.html`、阶段／聚类总览、全部配置 `metrics.csv` 和保存的投影。
`goal_H09_phase_report_zh.pdf` 为 `report_zh.html` 的中文静态导出。各 task 分开拟合，init0/1 训练、
init2 留出，unknown 不参加拟合与评分；原始维度逐 task probe 分数与既有报告完全复现。
阶段可读出与自然分簇是不同的诊断；本补充不构成环境感知跳层的干预证据。

阶段合并对照使用 `goal_h9_merged_phases.py`，输出为
`analysis/goal_h9_merged_phases_20261101`（相对于本轮结果根目录）。
固定三阶段／两阶段规则，PCA 坐标、样本、未知点排除和轨迹划分与原 H9 分析完全一致；
复用已有聚类，只补缺少的 k=3。`merged_phase_explorer.html` 可切换合并规则，
`goal_H09_merged_phase_report_zh.pdf` 保留完整十任务结果。task0 合并后仅一类，不计入分离评价。

`goal_h9_joint_phases.py` 将十个 Goal task 的实际 H9 点放进一套共享训练 PCA，
联合比较两／三阶段、k-means 和 task 颜色，输出在 `analysis/goal_h9_joint_phases_20261101`。
共521个阶段已知点（347训练、174留出），另28个unknown排除；没有拼接各task独立PCA坐标，
没有按task减均值。`joint_phase_explorer.html` 可筛选task和切换2D/3D；
`goal_H09_joint_phase_report_zh.pdf` 保存联合分布及任务身份对照。

`per_task_phase_views.py` 从阶段合并分析的已保存独立投影重绘每个task的两／三阶段并排图，
不重新拟合。输出在 `analysis/goal_h9_per_task_views_20261101`，含task0–4／5–9总览，
以及每页一个task、共十页的 `goal_H09_per_task_two_three_stages.pdf`。
