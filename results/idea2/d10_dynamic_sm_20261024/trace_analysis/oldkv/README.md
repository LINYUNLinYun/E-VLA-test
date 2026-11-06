# Old-KV D10 离线分析入口

只读取已完成的 nsys trace；缺少 SQLite 时执行 `nsys export`。不导入模型或启动 GPU 工作。运行时源码不变。

三条采集完成后，从项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python \
  results/idea2/d10_dynamic_sm_20261024/trace_analysis/oldkv/analyze_oldkv.py
```

默认文件在 `nsys/idea2_experiment/`：

- `d10_oldkv_graph_serial_20261024_v1.nsys-rep`
- `d10_oldkv_graph_encoder_overlap_20261024_v1.nsys-rep`
- `d10_oldkv_graph_ae_first_20261024_v1.nsys-rep`

可传入其他 basename，第一条作为串行对照；`--no-export` 只允许现有 SQLite，`--skip-missing` 用于先分析已采集部分。默认输出在本目录 `analysis/`。

归因通过 Kernel 的 `(globalPid, correlationId)` 对应 runtime `cudaGraphLaunch`，再取同一 Host 线程最内层请求与 `ae` / `prefill/vision` / `prefill/vlm` NVTX。GPU 区间完整保留，绝不按 Host 请求结束截断。单独记录请求返回后继续执行的 Prefill Kernel 数量和完成时间差。未分类的 Prefill 保留为 `Prefill_other`，缺失或歧义的关联显式计数。图层、AE 迭代、旧 KV 观测身份均不从模块标签猜测。

此处 `Vision` 对应整个 `prefill/vision` Graph，包括 Encoder、projector、语言 embedding 与 mask/position 准备，不能解释成纯 Encoder 算术时间。

主要输出：

- `analysis/module_overlap_summary.csv`、`analysis/README.md`：AE×Vision / AE×VLM 实际 Kernel 交集、模块 busy/envelope。
- 各子目录 `kernels.csv` / `graph_launch_map.csv` / `requests.csv` / `request_modules.csv`：完整 Kernel、提交、请求以及后台尾部证据。
- `report.json`：完整性、Green Context 归属、整条 trace 的 CUDA API 统计。
- `signature_comparison.csv` / `slowdown_summary.csv`：同 phase + 模块 + Kernel 名称 + launch geometry 匹配的耗时比、未匹配项和调用数差异。没有 Attention 内层标签，故不宣称纯 Attention 阶段减速。

这些是短 profile 的诊断值。CPU→CPU 延迟、吞吐、数值对照、双槽/Lease/epoch 完整性与旧 KV 观测年龄应以对应 benchmark 的 `result.json` 联合核验。

CPU 回归使用已有 `cewp_d10_vision_wait_modulation.2.sqlite`：5 个测量请求、64,475 个 Graph Kernel 均完成模块归因；无缺失/歧义 correlation。正确保留请求返回后仍运行的 4,127 个 Prefill Kernel（尾部约 29.2–29.8 ms），AE×VLM 为零。bootstrap 的 12,895 个 Graph Kernel 保留在原始表并显式标为 unscoped。回归产物在 `fixture_validation/`，不属于本轮 D10 三条新实验结果。

## 可选的同接口异步串行控制

这个独立入口不加入默认三条，控制 trace 尚未生成时不会阻塞上面的分析：

```bash
CUDA_VISIBLE_DEVICES='' policies/openpi/.venv/bin/python \
  results/idea2/d10_dynamic_sm_20261024/trace_analysis/oldkv/analyze_async_control.py
```

默认读取 `d10_async_serial_profile_20261024_v1`（`graph_async_serial`）与 `d10_oldkv_graph_encoder_overlap_20261024_v1`（`graph_encoder_overlap`）。输出到 `async_control_analysis/`；使用 `--skip-missing` 可在缺失时记录并正常退出。也支持 `--control` / `--overlap` 自定义 trace basename，`--control-mode` / `--overlap-mode` 指定同一或不同 trace 中的模式，`--phase` 默认 `continuous`。

检查 AE×Vision、AE×VLM、AE×整个 Prefill 的真实 Kernel 交集是否严格为零；逐请求核对 Vision 首 Kernel 在 AE 最后 Kernel 之后、VLM 在 Vision 之后、API 返回前 AE 已结束而 Prefill 返回后仍在执行。完整 GPU 区间不会按 Host 返回剪裁。`control_checks.json`、`control_requests.csv` 保留所有检查与尾部时差；匹配 signature 报告的比值是 Encoder-overlap / async-serial（诊断值，不能当作端到端加速比）。模式混合的 trace 会先明确筛选，防止混合两个调度组的 Kernel 统计。
