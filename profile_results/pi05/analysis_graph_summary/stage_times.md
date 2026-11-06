# π0.5 三模块 GPU 执行时间

分析日期：2026-10-24。来源：[原始 SQLite](../pi05-server.sqlite)、[采集 manifest](../manifest.json)。读取已有 trace，不运行模型、不重新采集。

统计 capture 中全部 10 次请求，取每请求算术平均，单位 ms。配置为 π0.5 LIBERO、B1、H10、10 次 flow 更新，component_breakdown=true、stage_boundary_sync=true。采用 SQLite 中实际 NVTX 起止时间，区别于 manifest 中 Python 计时器的边界，不能混合相减。

| 阶段 | 独立 kernel | Graph 执行区间 | kernel＋Graph 区间并集 | 单独 memcpy | 全 GPU 活动并集 | 阶段墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision encoder | 5.280 | 3.500 | **8.780** | 2.210 | 10.990 | 13.854 |
| VLM backbone | 0.993 | 30.387 | **31.380** | 2.715 | 34.095 | 34.886 |
| AE：10 个 action_expert 区间累加 | 1.244 | 14.168 | **15.413** | 3.192 | 18.605 | 34.387 |
| 完整 flow_matching 区间（含上述 AE） | 1.244 | 14.168 | **15.413** | 3.192 | 18.605 | 42.243 |

AE 单次更新的平均 kernel＋Graph 区间为 1.541 ms，含单独复制的 GPU 活动为 1.861 ms。最后一行包含第三行，不能再次累加。

## 边界与解释

- kernel 列只统计 `CUPTI_ACTIVITY_KIND_KERNEL`；Graph 列统计 `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`。Graph 内部节点没有逐项采集，Graph 区间可能含内部间隙或复制，因此 **kernel＋Graph 不能冒称全部纯 kernel 时长之和**。若需要纯 kernel 明细，应单独用 node 粒度重新采集，并考虑采集开销。
- 归属到 GPU0/context1，按阶段时间窗口统计；所有计入事件均完整落在对应窗口，无边界截断。kernel 和 Graph 均在 stream13；并集计算防止重复计数。memcpy 包括该 context 在窗口内的复制事件。
- AE 每请求包含 10 个 action_expert 区间。完整 flow_matching 还包含区间间的调度等耗时，二者 GPU 活动相同。两种墙钟差约 7.856 ms/请求。
- 完整 flow_matching 中没有被记录的 kernel、Graph、memcpy 覆盖的时间均值约 23.638 ms；VLM 约 0.791 ms，encoder 约 2.864 ms。差额不直接等于可消除 CPU 开销，也不能用活动覆盖比例当 SM 算力利用率。
- 这是拆分 JIT、同步组件输出的诊断路径。跨组件融合与调用方式不同于原生单体 JIT；不能直接用这些差额宣称原生路径优化上限。
- 10 请求均保留，没有挑除较慢记录。之前讨论的 46.499 ms 是第 2 次请求，不是总体均值。

## 第 2 次请求：与界面选区对照

| 阶段 | 起止时间（秒） | kernel＋Graph | memcpy | 阶段墙钟 |
| --- | --- | ---: | ---: | ---: |
| encoder | 0.291658748–0.309359574 | 8.873 | 2.214 | 17.701 |
| VLM | 0.309840807–0.344774561 | 31.358 | 2.715 | 34.934 |
| 完整 flow_matching | 0.345332670–0.391831726 | 15.396 | 3.192 | 46.499 |

数据：[逐请求 CSV](stage_times_per_request.csv)、[均值与中位数 CSV](stage_times_summary.csv)、[JSON](stage_times_summary.json)、[第 2 次请求 Graph ID 汇总](flow_matching_request02_graphs.csv)。
