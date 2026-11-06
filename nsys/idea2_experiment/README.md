# idea2 pipeline：Nsight Systems诊断

## 2026-10-21 调度修复补充

后续调度实验已完成：[完整结果与复现命令](../../docs/experiments/idea2/idea2_scheduling_fix_20261021.md)。新增两种提交顺序、CPU／Event 诊断及可选双 Host Worker；旧 profile 与本页下文的原始结果保留。

- [最终固定 KV 六臂 profile](scheduling_final_20261021.1.nsys-rep)：原始 kernel 重叠率 0.034%，单线程 ae_first 0%，双 Worker prefill_first 5.279%。
- [最终真实 Policy 回放 profile](scheduling_final_20261021.2.nsys-rep)：排除初始化请求后，原始重叠率 0.051%，双 Worker prefill_first 5.883%。
- 无 profiler 的真实回放响应 P50：324.15 → 306.84 ms，1.056×；P95：330.22 → 359.93 ms。双 Worker 因尾延迟回退保留为显式开关：`--host-workers --submission-order prefill_first`。

以下为修复前的历史诊断，不能与新增实验混为同一次采集。

本目录用于保存用户要求的真实`.nsys-rep`与相应输入／标签／数值检查凭据。对象是idea2原始完整π0 PyTorch eager pipeline：vision27层、VLM18层、action expert18层，D10／H50／每轮执行至多5个动作。此次只做有界profiling，不重跑LIBERO成功率，不修改原模型或调度逻辑。

## 看什么

- `encode`：prefix输入embedding，包括vision编码与projector。相机分支及vision的每个block分别标记。LIBERO是两个真实相机加一个原生零填充masked分支，不是三个物理相机。
- `vlm`：视觉／语言prefix融合及KV生成，每个原始层分别标记。
- `action_expert`：读取已选只读KV的完整10步动作生成；每个denoise step与该步所有expert层均有标签。
- 固定输入的原E1串行／双stream对照，与原`PipelinePolicy`三种模式的真实观测序列回放分开标记。后者不是仿真闭环，不能从此次profile推断新的任务成功率。

## 判定并行的正确方式

用Nsight Systems GUI打开`.nsys-rep`，展开CUDA GPU的context及不同stream，放大同一请求或同一E1配对区间，查看**实际GPU kernel横条是否在时间上重叠**。CPU上的NVTX范围仅标记调用／提交区间；NVTX GPU投影标记相关kernel的首尾包络，内部可能有空隙。两者均不能仅凭外框相交就称为同时计算。

建议对照：串行参考应基本无两模块kernel重叠；双stream／`async_reuse`是否真的同时运行、还是先把后台整条prefix提交完才开始AE。观察CUDA API中的同步、H2D／D2H与两模块之间的kernel空隙。

NVTX逐层标签和Nsight tracing会增加开销，本次时间不替代最终实验报告中的无profiler稳态延迟。CPU perf sampling在当前环境不可用，因此使用`--sample=none --cpuctxsw=none`；仍采集CUDA、NVTX和OS runtime事件，不更改系统权限。

## 已生成文件

2026-10-21采集，使用原先空闲的物理GPU2，RTX4090，UUID `GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25`。Nsight Systems CLI版本2026.2.1；建议用同版或更新版GUI打开。

| 文件 | 用途 |
| --- | --- |
| [idea2_pipeline_eager.nsys-rep](idea2_pipeline_eager.nsys-rep) | **主交付，13.79MiB**：完整逐层标签，直接用Nsight Systems GUI打开 |
| [idea2_pipeline_coarse.nsys-rep](idea2_pipeline_coarse.nsys-rep) | 13.67MiB：仅模块／去噪步／请求标签，检查逐层标记扰动，不替代主文件 |
| [主文件SQLite](idea2_pipeline_eager.sqlite)／[轻量SQLite](idea2_pipeline_coarse.sqlite) | 官方`nsys export`从对应rep导出，供SQL分析，不是GUI主文件 |
| [详细trace重叠分析](eager_timeline_analysis.json)／[轻量分析](coarse_timeline_analysis.json) | 原始kernel区间、CUDA launch关联、逐请求重叠统计和标签计数核验 |
| [NVTX汇总](detailed_stats_nvtx_sum.csv)／[CUDA API汇总](detailed_stats_cuda_api_sum.csv) | 官方`nsys stats`生成 |
| [详细运行凭据](detailed_metadata_20261021/result.json)／[身份](detailed_metadata_20261021/identity.json) | 完整逐层调用计数、raw/decoded数值验证、KV只读、实际source IDs |
| [轻量运行凭据](coarse_metadata_20261021/result.json)／[身份](coarse_metadata_20261021/identity.json) | 同样的原始路径及数值检查，关闭逐层标签 |
| [交付核验](delivery_receipt.json) | 文件SHA、深度、数值与来源验证记录 |

在远程VSCode中，可以右键`.nsys-rep`下载到本机，再拖入Nsight Systems。Markdown Preview Enhanced不能直接展示二进制profile；请不要把SQLite或这里的JSON误当作主profile。

## 时间线结构与定位

两份文件都先执行原E1配对：串行3次、双stream3次；随后执行原`PipelinePolicy`三模式，各5条真实记录的请求，interval=1。加载、每模式20次预热、数值验证放在capture区外。预热会循环复用真实bank，20表示请求数，不表示20个独立观测；该bank实际仅17条。

在NVTX时间线上搜索下列名字：

```text
section/E1_fixed_KV_causal_reference
  E1/serial/iteration_01 ... 03
  E1/two_stream/iteration_01 ... 03
section/runtime_PipelinePolicy_real_observation_replay
  runtime/fresh_serial/request_00 ... 04
  runtime/serial_reuse/request_00 ... 04
  runtime/async_reuse/request_00 ... 04
```

主文件中，各请求／配对范围内部的实际模块标签：

```text
prefill
  encode
    encode/camera_BRANCH01_base_0_rgb
    encode/camera_BRANCH02_left_wrist_0_rgb
    encode/camera_BRANCH03_right_wrist_0_rgb  # 原生零填充分支
      vision/L01 ... vision/L27
    encode/projector
  vlm
    vlm/L01 ... vlm/L18
action_expert
  action_expert/denoise_step01 ... denoise_step10
    action_expert/denoise_step01/layer01 ... layer18
```

上述是标签层级示意，不表示所有range在时间上同时发生。每次camera调用都分别有其vision层和projector；在第三分支上看到kernel并不说明存在第三个真实相机。另有`input_preparation`、`noise_preparation`、`output_copy_and_decode`、`runtime/reset`和`runtime/drain_owned_events`，原位置与原同步保留。

展开GPU的CUDA context和stream：这两份采集中，runtime前台主要为stream7，后台prefix为stream29；E1每次双stream配对新建的后台stream为117／121／125。物理GPU2因`CUDA_VISIBLE_DEVICES=2`在进程／trace中可能显示逻辑device0，不是跑到了物理GPU0。

建议首先看`runtime/async_reuse/request_01`，再与`runtime/serial_reuse/request_01`对比；request00需要初始化新KV，本来就必须先完成prefix，不能用它代表稳态并行。

## 本次实际看到了多少并行

下表不是NVTX范围相交率：用CUDA launch的`correlationId`和进程ID关联真实kernel，再按同线程与时间包含关系归属NVTX；对同一GPU不同stream的实际kernel时间区间求交。分母是所选区间的**kernel活跃时间并集**，不是episode时长、NVTX包络，也不是GPU利用率。排除memcpy／memset。

| 捕获内容 | 详细标签：重叠时间／kernel活跃并集 | 轻量标签：重叠时间／kernel活跃并集 |
| --- | ---: | ---: |
| E1串行3次 | 0／211.936ms＝0% | 0／211.919ms＝0% |
| E1双stream3次 | 0.027392／212.047ms＝**0.0129%** | 0.051040／211.967ms＝**0.0241%** |
| runtime fresh_serial，5请求 | 0／353.407ms＝0% | 0／353.379ms＝0% |
| runtime serial_reuse，5请求 | 0／353.421ms＝0% | 0／353.406ms＝0% |
| runtime async_reuse，5请求 | 0.151201／353.448ms＝**0.0428%** | 0.149538／353.434ms＝**0.0423%** |

两批的不同stream kernel重叠都对应encode/VLM与AE的重叠，不是其他无关kernel造成。**确实用了多个stream，但并没有形成持续、有效的两模块计算流水线。** 关闭逐层标签后runtime结果仍相近，不支持把几乎无重叠全部归于逐层标记。

轻量trace中，非初始化异步请求的AE第一个kernel开始前，2295个encode/VLM kernel中已有2282–2284个结束，约**99.4–99.5%的kernel数量**；这是数量比例，不能换算成FLOPs或耗时比例。结合原代码，后台prefix由同一个Python主线程先完整派发，才开始AE派发；大部分prefix工作已完成，剩下少量尾部kernel才与AE碰在一起。时间线上应重点看主线程提交顺序、CUDA API之间的间隙，而不是仅看到两个stream就认为pipeline已成功。

这仍是受profiling扰动的一次有界诊断，不是对所有提交顺序、CUDA Graph实现或其他硬件的不可行证明。此次**没有运行Graph pipeline**；原idea2的Graph实验是另一个固定输入微基准，不能混进这份实际eager policy回放。实际部署优化需另立实验，不能从这份trace直接声称能达到某个加速比。

## 数值与标签验收

- 两份capture均正常exit0，官方SQLite导出成功；各有271053个真实kernel，全部可关联CUDA launch。
- 主文件实际NVTX标签计数与runner逐项完全一致：21次prefix、21次完整D10 AE；每个vision层63次（21×3分支）、每个VLM层21次、每一步的每个AE层21次。不是只挂了没执行的hook。
- 与原idea2相同checkpoint和bank SHA。原生／拆分、装饰前／后、E1串行／并发、runtime按自身实际KV来源重建的raw和decoded误差均为0；KV字节不变，全部动态方法已恢复。7项CPU测试通过。
- runtime的Full当前KV来源为0／1／2／3／4；serial_reuse和async_reuse本批均为0／0／1／2／3。来源如实记录，不在profiling改变cache发布时机后强行假定一致。
- 原E1固定prefix在runtime段仍驻留，用于停止capture后做完整KV字节检查；runtime自己的原双槽所有权不变，但不能据本次峰值显存宣称生产仅驻留两份完整cache。原始NN／采样／调度源码未改。

Nsight诊断限制也保留：当前GPU不支持本工具的hardware trace，实际采集为CUPTI软件跟踪；CPU调度采样未启用，OS runtime推断的线程活跃状态不能当精确的CPU运行／抢占记录。两份文件均有全局“Not all NVTX/CUDA/OS runtime events might have been collected”警告。本次不是对进程全生命周期的完整采集；已独立证明所需应用标签在capture内逐项计数完整，并且所有已记录kernel与launch一一对应，但不据此保证所有其他事件类别或capture外事件都完整。重叠率是**本次已记录kernel**的诊断统计，不冒充无扰动生产测量；这也限制了对CPU瓶颈的进一步归因。

## 复现命令

从项目根目录执行，先确保选择的GPU可用，使用**新的输出名和metadata目录**，避免覆盖本次文件：

```bash
CUDA_VISIBLE_DEVICES=2 JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu \
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi \
PYTHONPATH=.:policies/openpi/src \
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=false --output=nsys/idea2_experiment/REPRODUCE_NEW \
  policies/openpi/.venv/bin/python -m preparations.idea2.nsys_profile \
  --bank policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/episodes/E1/supp_E1_D2_tau0.1_P2_Gconfirmed_disagreement_rejectD2/A/libero_spatial/task00/init07/real_predict_inputs_first25.pkl \
  --output nsys/idea2_experiment/REPRODUCE_METADATA_NEW --launch
```

轻量对照使用另一组新输出名，追加`--coarse-only`；其他模型、输入、seed1701、D10与请求预算相同。不带`--launch`只打印dry-run，不加载模型或创建产物。

新增诊断代码与CPU测试由6.1 sol medium实现：[nsys_profile.py](../../preparations/idea2/nsys_profile.py)、[test_nsys_profile.py](../../preparations/idea2/test_nsys_profile.py)。
