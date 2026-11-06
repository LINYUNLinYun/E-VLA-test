# O_async_serial：保留API早返回，禁止AE与Prefill重叠

目的：旧O0的API会等新Prefill完成；O1会在AE输出完成后返回。因此O0→O1的整请求加速同时包含API返回规则和后续Host输入准备可被隐藏的收益。这个控制组让两臂都提前返回，只改变GPU等待位置，以分离允许Encoder重叠的净效果。

| 模式 | 固定设置 | 当前请求GPU依赖 | API返回 |
|---|---|---|---|
| `graph_async_serial` | `async_reuse`、AE-first、priority0、`vision_only` | AE → Prefix staging → Vision → VLM | AE输出完成后；当前Prefill可仍在后台运行 |
| `graph_vision_overlap`（O1） | 同上 | AE与Prefix staging/Vision可重叠；VLM等AE | 相同 |

两臂都保持 `await_previous_prefix=true`，所以下一AE必须消费上一请求完整新KV；每请求一次Prefill、age1、完整旧KV固定十轮。bootstrap没有已提交AE，不等待未记录的Event。控制组复用O1同一个`ae_done`，将其等待移到`build_prefix`入口，并清除对应pending标志，避免原实现再次插入等待。两臂都是一次等待，未改Kernel、模型、FP32时间递推或Pool。

`async_serial_control.py`为104行results层wrapper，复用原`cewp_bench`、`SelectiveGraphs`捕获、两槽、Oracle、100对AB/BA和连续计时。原`configure`先排空/reset，wrapper随后才切gate；Graph owner必须相同。每实测行保留`prefix_dependency_gate`、early-return语义、实际`selective_kind=vision_only`和engine ordinal；拒绝新engine或稳态重捕获。原Prefill完成时间、source_age、response/all_work/drain字段不变。

默认不修改任何运行时文件。wrapper在结果`source/async_serial_control_wrapper.py`中保存自身快照和SHA256，结果command指向真实wrapper模块；运行时检测wrapper源码漂移会排除结果。原base源码快照同时保留。

## 已准备矩阵

`async_serial_control_matrix.json`可直接由现有`run_matrix.py`读取，无需修改runner。四项按顺序：

1. 20对＋两臂各20连续，reset10，作为门禁（80行）。
2. 100对＋两臂各100连续，reset50（400行）。
3. 独立进程反转AB/BA首臂及连续组顺序复测（400行）。
4. 仅控制组8条连续请求的nsys短窗口，性能计时与profile分开。

全部D10、noise-period13、20 warmup、`--hoist-modulation --selective vision_only --await-previous-prefix`，其他设备和线程环境沿用现有runner。

```bash
# 仅在当前GPU队列边界，由调度Host执行；本次交付未运行该命令。
policies/openpi/.venv/bin/python \
  results/idea2/d10_dynamic_sm_20261024/run_matrix.py \
  results/idea2/d10_dynamic_sm_20261024/async_serial_control_matrix.json
```

控制trace目标：`nsys/idea2_experiment/d10_async_serial_profile_20261024_v1.nsys-rep`。应核对AE与Vision/VLM实际Kernel交集为0，以及API返回时后台Prefix是否仍未完成。下一请求输入准备中的GPU预处理/拷贝、输出D2H与后台工作仍按原实现执行；“串行”指AE/完整Prefill主计算之间的顺序，不声称整个设备每时刻只有一个Kernel。

## 验证及解释

9项CPU检查通过，见`async_serial_control_cpu.log`：相同单个Event等待位置、bootstrap无等待、真实Pool producer/reader排空后切gate、活lease拒绝reset、Graph owner/两槽/stream不变、单engine门禁、源码快照；module `--help`入口也已验证。未启动GPU。

结果解释以 `graph_async_serial → graph_vision_overlap` 的实测请求/完整连续轨迹差异为准。收益包括允许Encoder重叠后的资源竞争、Prefill完成时点和下一请求等待变化；不是孤立Kernel微基准。旧O0仍是“同步完成整个请求”的有效基线，但不能把它对O1的全部收益叫作纯GPU并行收益，也不能把多个speedup直接相加。
