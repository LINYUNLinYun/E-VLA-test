# π0.5 SAE 语义—层/模块实验

依据根目录 `pi05_sae_semantic_skipping_plan.md` 执行。先复用 Action Atlas 的公开成果，新增代码只负责数据筛选、原生 OpenPI 接入和计划中的 FFN 干预。

## 已接入的公开资产

- `bag100/action-atlas-pi05`：L5/L16 的 per-token SAE、**原样使用的官方 `loader.py`**、36 个 pathway/layer 的概念索引、完整消融/轨迹索引。
- `bag100/action-atlas-pi05-activations`：原始层激活、原作者已有层统计和采集 summary。
- `bag100/action-atlas-viz`：网站的 feature metadata 和逐层、逐 suite 特征说明。
- `CWRU-AISM/action-atlas`：公开实验代码和网站索引，保存在缓存中用于核对协议。

Hugging Face 资产均固定 revision，文件 hash 记在 manifest。概念索引的 ID 是候选池，不是逐特征排名；逐特征效应量/频率不在该索引中时，从原始结果或小批激活补算。网站文字说明只用于寻找线索。

## 当前实施入口

| 脚本 | 作用 |
|---|---|
| `prepare_assets.py` | 精确下载两层 SAE、官方 loader 和概念/消融索引 |
| `inspect_public_assets.py` | 清点公开文件及体量，避免全量下载 |
| `download_references.py` | 复用已有层统计和网站特征说明 |
| `analyze_public.py` | 生成跨层概念表、L5/L16 候选池、同层概念重叠及图表 |
| `check_public_sae.py` | 直接使用官方 loader，在一条公开轨迹的两层激活上做 CPU 检查 |
| `prepare_observations.py` | 复用完整本地 π0.5 baseline 轨迹，冻结 12/4/8 任务划分和观测清单 |
| `native_probe.py` | 复用现有 OpenPI loader/预处理，检查只读 hook 与 L5 后分段前向 |
| `check_local_sae.py` | 使用官方 loader 检查一条本地原生观测的层激活 |
| `check_cpu_threads.py` | 在已保存的原生张量上核对 CPU 线程设置变更 |
| `public_tail_check.py` | 检查公开 L5 激活是否能直接由本地尾段重放；失败不阻止复用公开候选和统计 |
| `collect_local.py` | 逐观测保存原生 L5 缓存及两层 SAE 读出，可在观测边界停止和续跑 |
| `select_sae_concepts.py` | 发现集筛选特征、校准集检查语义触发，不读取动作或闭环 SR |
| `map_concept_ffn.py` | 同一 L5 起点的六个 FFN/组合干预；输出未截断 D 与辅助 E 的表和热图 |
| `dictionary_readouts.py` | 从已有层 token 恢复全字典汇总，逐项检查与已有候选列一致；不新增 VLM 前向 |
| `eval_frozen_readout.py` | 固定较早的 feature ID、归一化及阈值，在完整观测上复核而不重选 |
| `followup_diagnostics.py` | 在相同完整 64 观测上比较有限的公开候选/全字典/指令读出变体 |
| `build_semantic_rules.py` | 从实测单模块矩阵提议并实际验证最多三个组合，输出条件校准表和候选规则 |
| `make_quality_candidate.py` | 保留小阈值结果，单独冻结可检验的相对影响先验，不使用测试 SR |
| `runtime_router.py` | 复用原生完整动作生成路径，仅在 L5 读出后切换真实 FFN 开关 |
| `check_runtime.py` | 同设备固定噪声检查只读 no-op 与指定 mask 路径；不要求 CPU/GPU 张量逐位相同 |
| `quality_eval.py` / `quality_service.py` | 复用现有 LIBERO runner、初始化、噪声与本地 RPC，冻结及执行 24 条小批闭环 |

本地模型使用 **`policies/openpi/.venv/bin/python`**，该环境已有 OpenPI 的 Transformers 补丁。公开数据分析可用 `policies/realtime-vla-flash/.venv/bin/python`。不修改共享 Python 环境。

## 首轮产物

`results/pi05_sae_semantic/run_20261105/`：

- `public_inventory/`：三个固定发布版本的文件清单。
- `public_analysis/public_analysis.md`：公开数据复用结果；CSV、PNG/PDF 与原始表可互查。
- `public_sae_check/public_sae_check.json`：公开原始激活的 shape、采样位置、SAE 重构与稀疏度实测。
- `local_observations/observation_manifest.json`：12 个发现任务、4 个校准任务、8 个留出任务及真实轨迹来源。
- `local_observations/pilot_observations.jsonl`：64 个本地小批观测；完整可用观测清单含 691 个实际请求，短轨迹不重复填满。
- `native_preflight_cpu/`：本地只读捕获、尾段等价性和后续 SAE 接口检查。

原始公开资产缓存在 `shared/caches/pi05_sae_semantic/`，无需再次下载或训练。`native_probe.py` 的分段 full 校验通过后，才使用该尾段运行六个候选 FFN 的跳过干预。

公开数据重分析、单条本地接入检查、跨任务语义区分和 FFN 条件冗余分别报告；前两项完成不代表后两项已经完成。GPU 时延和闭环质量另按计划测量。

## 当前小批配置

原生只读捕获及分段尾段均已与原始前向逐位一致。32 线程在同一真实观测上也与已保存的 8 线程张量一致；64 个观测现已全部完成。前 12 个记录按启动命令记为 8 线程，其余记录保存 `threads=32`，设置变更见 `runtime_history.jsonl`。

先检查三个有正负任务覆盖的候选标签：`object/bowl`、`object/stove`、`spatial/on`。`turn on` 不标为空间 `on`。沿用公开候选 ID 池，在发现集按任务/episode 等权计算 R、负样本 95 分位阈值和触发正样本参考幅度；每概念每层最多四个特征。低覆盖率本身不构成剔除理由。

本轮的经验筛选参考为触发 precision≥0.8 且覆盖至少两个正 episode；概念分数在校准集只检查 0.8、0.6、0.9 三个阈值，记录 precision、误触发率、覆盖率和支持量。这些是探索配置，不是部署正确率或动作安全保证。校准不成立时保留候选统计，限定结论范围，不扩大为整个 SAE 路线的否定。

采集完 64 个观测后自动做语义筛选；有支持的概念才启动初批模块干预。初批从发现集按冻结的 L5 概念轮询选择最多 12 个观测，并优先覆盖不同 task/episode；不根据 L16 证据或真实概念标签删除样本。后续按信号决定是否扩到更多观测及组合 mask，原计划每集合最多 128 个观测的边界不变。

初批单模块矩阵不会自动启动闭环或宣称加速。需先解释 D/E、检查实际组合及独立校准集支持，再按方案执行策略与闭环阶段。

L16 的全局分类表现只作为诊断。实际干预由已校准的 L5 触发定义条件，L16 方向需在发现集有关联；可用于规则的条目检查该条件群内的晚期基线证据，而不强求后置探针成为第二个全局语义分类器。`late_semantic_calibrated` 列记录前者，`routing_supported` 记录实际双 FFN 校准条目的条件支撑；两者不是同一个判断。

第一个相同最优 pair 会先实际检验。必要时在原有最多三个 mask 范围内，用不含其各分量的候选检验组合非线性；不直接把单模块相加当作组合结果。`threshold_sweep` 同时报告 mask 数和固定观测上的可触发数，避免有一个 mask 条目就误称实际有跳过覆盖。首尾帧重放覆盖不能代替自由闭环覆盖。

扩样应另建输出目录，读取完整 `observations.jsonl` 并使用 `--reuse-collection` 复用兼容的完成捕获；有界采样按 task/init 交叠并优先覆盖轨迹首尾，不只截取前几个长轨迹。首批样本计入扩样总数。

## 当前完整策略接入

已实际验证三个 pair：`[12,15]`、`[6,15]`、`[12,14]`。默认 `delta=0.1` 没有可跳过规则，原结果保存在 `mask_study_frozen36_diverse/skip_rules.json`。另存的 `quality_candidate_relative_prior.json` 选择 `[6,15]`，阈值为 1.1613027445，准备检验相对影响先验，尚未证明语义保真或动作质量。

CPU 完整动作路径已核对：只读 SAE 时 raw/decoded 动作与原路径逐位一致；在线触发 `[6,15]` 的动作与直接指定同一 FFN mask 的动作逐位一致。完整策略继续使用原始 H10、R5、D10、预处理、Action Expert 与后处理。原始 FULL 对照移除全部路由 hook。

闭环复用 `preparations.idea1_v3_1.rollout.run_episode`，不重写 LIBERO 协议。第一批固定八个留出任务、各 init0、三个方法，共 24 条；固定方法与语义方法使用同一 pair。固定周期只在校准回放上配预算，实际覆盖和时延从闭环逐调用记录。GPU 动作路径检查通过后再启动对应设备的批次；当前 CPU 检查不能替代 GPU 时延结论。
