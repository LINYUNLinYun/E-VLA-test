# D10 Nsight 离线分析

仅分析已生成 trace；未运行模型或 GPU 工作。各条为 8 个正式请求的短 profile。
这些数字只解释实际 Kernel 重叠和并发减速，不能替代非 profile 的 100 对端到端结果。

| 配置 | 帧数 | AE×VLM /帧 ms | AE×下一帧 Vision /机会 ms | VLM busy /帧 ms | AE1 busy /帧 ms | AE2–10 busy /帧 ms |
|---|---:|---:|---:|---:|---:|---:|
| Full serial | 8 | 0.000 | 0.000 | 31.849 | 2.664 | 23.670 |
| Full layer overlap | 8 | 5.896 | 0.000 | 33.399 | 5.733 | 23.583 |
| Fixed AE16/Prefix112 | 8 | 6.157 | 0.000 | 38.544 | 6.150 | 37.908 |
| AE16 then primary / Prefix112 | 8 | 6.186 | 0.000 | 38.565 | 6.181 | 23.574 |
| Full combined crossframe | 8 | 5.863 | 6.625 | 33.924 | 5.719 | 25.563 |
| AE64/Vision64/VLM primary | 8 | 0.000 | 4.934 | 31.566 | 2.683 | 24.619 |

下一帧 Vision 的分母为 N−1 个相邻帧机会；同帧/跨帧交集不能重复相加。busy 是真实 Kernel 区间并集，包含 Kernel 内部停顿，不是 SM 利用率。

## 简短结论

- 普通逐层方案已有真实 AE×VLM 重叠；AE1 的 GEMM/Attention 区域明显延长，后九轮基本恢复串行速度。
- 固定 AE16 会持续拖慢后九轮；动态切回 primary 后，该尾段的匹配 Kernel 时间恢复到串行附近。动态释放资源本身有效，但 Prefix112 的 VLM 仍慢于不限额 VLM。
- 跨帧 combined 保留 AE 后续轮次与下一帧 Vision 的真实交集，同时 AE 尾段也有减速。它是吞吐窗口，不代表单帧响应下降同样多。
- Encoder64 的 VLM 已实证运行在 primary，VLM Kernel 时间恢复串行附近；其短 trace 中 6/7 次相邻帧产生 Encoder 交集。下一 Encoder 在上一 AE1 结束后 16.2–24.5 ms 才开始，最后一次已错过 AE 尾段。此边界说明窗口利用受启动时机影响，不能仅凭它把原因归给 CPU 计算；详见 crossframe_boundaries.csv。
- 普通 ungated 路线允许 AE2 与最后 VLM 尾部有小段重叠；固定/动态 split 的 full-Prefix gate 则显式排除此段。逐轮交集见 iteration_overlap.csv。
- 当前结论只限这些 8 请求的短 profile；正式选择以同配置非 profile 测量为准。

## 匹配 Kernel 的减速

相对 Full serial，匹配名称、launch geometry、阶段/层/轮次后，按共同调用数加权。大于 1 表示相同 Kernel elapsed 变长。

| 配置 | VLM GEMM | VLM FFN GEMM | AE1 GEMM | AE1 Attention 区域 | AE2–10 GEMM | AE2–10 Attention 区域 |
|---|---:|---:|---:|---:|---:|---:|
| Full layer overlap | 1.034× | 0.994× | 2.497× | 2.463× | 1.000× | 0.994× |
| Fixed AE16/Prefix112 | 1.236× | 1.158× | 2.729× | 3.007× | 2.040× | 1.875× |
| AE16 then primary / Prefix112 | 1.235× | 1.158× | 2.723× | 3.017× | 1.000× | 0.992× |
| Full combined crossframe | 1.053× | 1.011× | 2.466× | 2.466× | 1.078× | 1.103× |
| AE64/Vision64/VLM primary | 0.989× | 0.989× | 1.016× | 1.031× | 1.046× | 1.064× |

Attention 区域包含 RoPE、reshape、mask 等，不能全部归为 Attention 算术。未匹配 signature 类型与调用数差异见 JSON/CSV，不对其填充或推断。

## 资源与生命周期证据

- 从每个模型 Kernel 的 greenContextId 关联 TARGET_INFO_CUDA_CONTEXT_INFO，检查实际配额；primary=不限额资格，不等于实际占满所有 SM。
- AE16 固定组应全十轮 16 SM；动态组应 AE1=16 SM、AE2–10=primary；两组 VLM=112 SM。
- Encoder64 组单独验证 Vision=64 SM、AE=64 SM、VLM=primary，避免只检查 replay stream。
- 逐帧保存尾段相对 AE1/VLM 的边界；普通 ungated overlap 不要求额外 full-Prefix 栅栏。跨帧另查下一 VLM 未越过上一 AE 最终完成。

| 配置 | 所有 Graph 阶段映射 | 所有请求映射 | SM 归属通过 | 匹配 signature | 未匹配 baseline/candidate |
|---|---|---|---|---:|---|
| Full serial | True | True | True | None | None/None |
| Full layer overlap | True | True | True | 7738 | 0/0 |
| Fixed AE16/Prefix112 | True | True | True | 7738 | 0/0 |
| AE16 then primary / Prefix112 | True | True | True | 7738 | 0/0 |
| Full combined crossframe | True | True | True | 7738 | 0/0 |
| AE64/Vision64/VLM primary | True | True | True | 7738 | 0/0 |

## Host 输入准备口径

`prepare_cpu_ms` 是 OpenPIBackend.prepare 的 Host 墙钟，包括 deepcopy、input_transform、NumPy→Torch H2D 和 observation preprocess，不是纯 CPU 算术或 Kernel 提交耗时。
资源审计 JSON 同时保留整条 trace 的 memcpy / synchronize API 次数与累计时间、GPU 拷贝字节数，以及是否存在独立 prepare NVTX。没有 prepare 范围时不把全部拷贝/同步归给该阶段，也不以其 API 时间解释完整准备墙钟。

## 文件

- `stage_summary.csv`：阶段 Kernel busy 与 envelope；`overlap_summary.csv`：实际交集；`iteration_overlap.csv`：逐轮交集。
- `per_iteration_times.md` / `per_iteration_times.csv`：逐帧十轮 velocity 范围的 busy/envelope；明确不含 Euler。`preparation_or_euler_times.csv` 单列范围外 Kernel。由同目录 per_iteration_times.py 生成。
- `crossframe_boundaries.csv`：下一 Vision 的首 Kernel 相对当前 AE1/最终完成的边界，以及下一 VLM 的依赖检查。
- `matched_kernel_slowdowns.csv`：按操作汇总；各子目录 `signature_comparison.csv` 保留全部匹配项。
- `resource_audits.json`：每阶段 Green Context/SM 配额及逐帧边界。
- 各配置子目录包含全部 `kernels.csv`、阶段与请求表、代表 `timeline.png`；`offline_receipt.json` 记录导出/复用来源。
- 重现：`policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/offline_trace_matrix.py`，再运行同目录 `summarize_traces.py`。
