# D10 逐轮 velocity Kernel 统计

仅从六条已有 kernels.csv 做 CPU 统计；每条 8 帧，每帧 10 轮。预热/捕获不在这里计时。

`AE/iteration_N` 的 NVTX 仅包住 `original_velocity(...)`（layer_graphs.py:311）。因此下表是该轮 velocity 网络的 Kernel busy / envelope，不是含 Euler 的完整 Flow Matching 迭代。`x = x + dt * velocity`、`time += dt` 与最终 clone 在范围外（graph_sampler.py:127）。

- busy：该范围真实 CUPTI Kernel 区间的并集，包含 Kernel 内部停顿，不能当作纯算术时间或 SM 利用率。
- envelope：首 Kernel 开始至末 Kernel 结束，包含 Kernel 间隙和逐层 KV-ready 等待；不含首 Kernel 启动前的等待。
- Preparation_or_Euler_update：AE Graph 中全部 velocity 范围以外的 Kernel，混合 mask/position/time 准备、各轮 Euler/time 更新、最终 clone；当前标签不能将其精确拆成逐轮纯 Euler。这个集合的 envelope 跨过十轮 velocity，不应作为“准备耗时”或再加到 velocity envelope。
- 这里的范围外 GPU 准备与 benchmark 的 `prepare_cpu_ms`（整个 Host 输入准备路径）是不同口径，不能用前者解释后者。
- 固定 schedule/调制准备在冷构图时完成，这里的热态范围外 Kernel 不代表重复生成所有调制。Graph 捕获与预热均已结束，AE1 不等于首次编译/权重加载。

| 配置 | velocity1 busy ms | velocity1 envelope ms | velocity2–10 平均每轮 busy ms | velocity2–10 平均每轮 envelope ms | 范围外 busy/帧 ms |
|---|---:|---:|---:|---:|---:|
| Full serial | 2.664 | 2.729 | 2.630 | 2.681 | 0.054 |
| Full layer overlap | 5.733 | 32.299 | 2.620 | 2.774 | 0.073 |
| Fixed AE16/Prefix112 | 6.150 | 37.188 | 4.212 | 4.248 | 0.066 |
| AE16 then primary / Prefix112 | 6.181 | 37.247 | 2.619 | 2.670 | 0.066 |
| Full combined crossframe | 5.719 | 32.043 | 2.840 | 3.097 | 0.068 |
| AE64/Vision64/VLM primary | 2.683 | 2.827 | 2.735 | 2.774 | 0.054 |

简要解读：串行热态的第一轮与后九轮基本同量级；逐层重叠时第一轮的长 envelope 主要反映等待，busy 本身也变长。动态 SM 让后九轮恢复串行附近，而固定 AE16 的每一轮都持续变慢。combined 的后半段 velocity 成本逐渐升高，与下一帧 Encoder 重叠窗口相符；这仍是短 trace 的诊断，不能替代正式端到端收益。

完整十轮逐轮均值（busy / envelope，ms）：

| 配置 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full serial | 2.664 / 2.729 | 2.722 / 2.775 | 2.625 / 2.676 | 2.626 / 2.677 | 2.624 / 2.676 | 2.626 / 2.678 | 2.624 / 2.675 | 2.611 / 2.662 | 2.606 / 2.657 | 2.606 / 2.656 |
| Full layer overlap | 5.733 / 32.299 | 2.726 / 3.703 | 2.606 / 2.657 | 2.608 / 2.659 | 2.606 / 2.657 | 2.608 / 2.659 | 2.607 / 2.658 | 2.608 / 2.658 | 2.607 / 2.658 | 2.607 / 2.658 |
| Fixed AE16/Prefix112 | 6.150 / 37.188 | 4.323 / 4.362 | 4.198 / 4.234 | 4.196 / 4.231 | 4.202 / 4.238 | 4.193 / 4.229 | 4.202 / 4.238 | 4.194 / 4.230 | 4.204 / 4.240 | 4.195 / 4.231 |
| AE16 then primary / Prefix112 | 6.181 / 37.247 | 2.726 / 2.779 | 2.605 / 2.655 | 2.606 / 2.657 | 2.607 / 2.658 | 2.605 / 2.656 | 2.607 / 2.657 | 2.606 / 2.657 | 2.606 / 2.657 | 2.606 / 2.656 |
| Full combined crossframe | 5.719 / 32.043 | 2.762 / 3.761 | 2.646 / 2.698 | 2.648 / 2.700 | 2.647 / 2.699 | 2.768 / 2.880 | 2.882 / 3.069 | 3.028 / 3.289 | 3.109 / 3.410 | 3.073 / 3.371 |
| AE64/Vision64/VLM primary | 2.683 / 2.827 | 2.741 / 2.781 | 2.648 / 2.686 | 2.647 / 2.686 | 2.647 / 2.685 | 2.647 / 2.686 | 2.668 / 2.707 | 2.792 / 2.830 | 2.866 / 2.905 | 2.963 / 3.001 |

输出：`per_iteration_times.csv`（480 条帧×轮记录）、`per_iteration_summary.csv`（60 条配置×轮汇总）、`preparation_or_euler_times.csv`（48 条帧记录）、`iteration_partition_checks.csv`（覆盖与区间并集核对）。每帧 velocity1–10 加范围外 Kernel 覆盖整个已归因 AE capture；没有按相邻时间位置猜测 Euler 所属轮次。
48 帧分类完整、合并后的区间并集均与完整 AE capture 一致。分别求 busy 再求和与全局并集的最大差为 0.448 μs/帧；极小边界交集显式保留，不把分类互斥假设成时间区间绝不相交。

重现：`policies/openpi/.venv/bin/python results/idea2/d10_dynamic_sm_20261024/trace_analysis/per_iteration_times.py`。
