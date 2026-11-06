# Realtime-VLA FLASH 复现

## 结论速览（更新于 2026-09-26）

按论文写明值（δ=0.15 / PF=2，四个 suite 同一套变量）完成两轮 4 suite × 10 task × 50 trials。最新一轮为每张 RTX 4090 独占 1 个 suite（1 服务端 + 1 客户端），共 **2000 集**：

| suite | 单卡独占 | 原共卡实验 | 论文 FLASH+Triton-π0 | 独占 Δ论文 |
| --- | --- | --- | --- | --- |
| LIBERO-Spatial | 98.0%（490/500） | 97.6%（488/500） | 96.8 | +1.2 |
| LIBERO-Object | 98.4%（492/500） | 99.6%（498/500） | 99.2 | -0.8 |
| LIBERO-Goal | 93.8%（469/500） | 94.8%（474/500） | 94.4 | -0.6 |
| LIBERO-10 | 85.2%（426/500） | 85.6%（428/500） | 84.6 | +0.6 |
| **平均** | **93.9%（1877/2000）** | **94.4%（1888/2000）** | 93.8 | **+0.1** |

- **成功率复现完成**：单卡独占轮平均为 **93.9%**，与论文 93.8% 基本一致。默认参数（δ=0.3、PF 关闭）下 Goal 仅 88.4%；参数对齐后的两轮分别为 94.8% 和 93.8%，说明原 6 个点的缺口来自参数错配，而非成功判据或环境。
- **延迟复测完成**：单卡独占轮的 `total_ms` 均值为 **19.0–20.1 ms**，flash 轮中位数为 **7.8–8.0 ms**，接近论文的 7.8 ms。与原共卡轮的 19.2–20.4 ms 相比只降低 0.1–0.4 ms，说明当时的共卡争抢影响很小。
- **轨迹长度已记录**：2000/2000 个 episode 的 `trajectory_length` 均与 trace 中实际执行动作数一致。长度与成功强负相关，但这是因为成功会提前结束、失败会跑满 horizon，不能解释为“长轨迹导致失败”。
- **原始 π0 全路径对照完成**：同协议 2000 集中，π0 为 **1867/2000（93.35%）**，FLASH 为 **1877/2000（93.85%）**，净变化 **+0.50 pp**；配对 95% CI 为 **[-0.90, +1.90] pp**，McNemar `p=0.529`。两边都成功的 1770 集中，FLASH 轨迹平均只变化 **-0.93 步**（95% CI `[-2.26,+0.39]`）。成功率与轨迹长度均无总体退化证据，但同口径 round latency 从 37.0–37.5 ms 降到 19.0–20.1 ms。
- **基线对照完成**：Triton-π0 全路径基线中，Goal task 5 为 98%，证明默认配置下 FLASH 的 46% 是参数错配造成的；task 3 的 FLASH 82% 与基线 88% 仅差 0.9σ，只能说明“没有证据表明差异由 FLASH 引入”，不能证明二者等效。
- **本轮按要求未开展其他事项**：失败 rollout 回放、逐 suite 核实论文 PF 设置、δ/PF 分离消融仍保留为后续项。


## 论文的实验设置

- 环境：单张 NVIDIA RTX 4090D；标准 LIBERO 的 Spatial、Object、Goal、10 四个 suite。
- 规模：每 suite 10 个 task，每 task 50 trials，共 2000 集。
- 策略：action chunk size `H=50`，每执行 12 个动作重新规划；每个 suite 单独训练一份 draft 权重。
- 对比方法：Torch-π0、Triton-π0、FLASH-π0、FLASH+Triton-π0；指标为 suite 成功率、平均推理延迟 `Lat.` 和每动作延迟 `/Act`。
- 路径成本：FLASH-π0 的 flash-path round 为 17.9 ms，FLASH+Triton-π0 为 7.8 ms，full-path round 为 58.0 ms。
- 跨 suite 统计：66.8% 的 replan 轮走 flash path，接受前缀平均覆盖 replan 窗口的 69.7%。


## 原论文复现结果

Table 1: Simulation results on four LIBERO suites

| Methods | Spatial | Object | Goal | 10 | Avg SR | Lat. (ms) | /Act (ms) | ΔSR | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Torch-π0 | 96.8 | 98.8 | 95.8 | 85.2 | 94.1 | 58.0 | 5.0 | – | 1.00× |
| Triton-π0 | 96.4 | 98.8 | 95.0 | 86.6 | 94.2 | 39.7 | 3.5 | +0.1 | 1.46× |
| FLASH-π0 | 96.4 | 99.2 | 94.6 | 83.4 | 93.4 | 34.9 | 3.0 | -0.7 | 1.66× |
| **FLASH+Triton-π0** | 96.8 | 99.2 | **94.4** | 84.6 | 93.8 | **19.1** | 1.9 | -0.3 | **3.04×** |

（成功率为 suite 级 %，每 suite = 10 task × 50 trials）


## 现复现结果

> 本节是**仓库默认参数**（δ=0.3、PF 关闭，与论文写明值不同）下的 LIBERO-Goal，作为对照保留。
> 参数对齐后的四 suite 结果见下方「[参数对齐后的四 suite 复现（δ=0.15 / PF=2）](#参数对齐后的四-suite-复现δ015--pf2)」一节。

### 复现协议（LIBERO-Goal）

| 项 | 取值 |
| --- | --- |
| 基座 | openpi 官方 `pi0_libero`（GCS），未做任何训练 |
| draft | HF `Dexmal/RealtimeVLA-Flash` → `draft_libero_goal.pt`，官方 `convert_for_triton.py` 转 Triton |
| 服务端 | `spec_serve_policy.py --config pi0_libero --backend triton --task-suite-name libero_goal --tokenizer-source local`，GPU 0 |
| 客户端 | `spec_client_libero.py`，渲染 GPU 1，`MUJOCO_GL=egl` |
| 协议 | `--num-trials-per-task 50`；`num_steps_wait=10`；`resize_size=224`；`replan_steps=12`（服务端 `max_exec_steps=12`）；seed=7 |
| 初始状态 | `init_files/libero_goal/<task>.pruned_init` 的 50 条，episode_idx 0–49 逐一对应 |
| 成功判据 | LIBERO 原生 `_check_success()`（BDDL goal 谓词求值），非自定义 |
| 验证参数 | 服务端命令行未传 `--tau-radius` / `--periodic-full-every-n-draft-rounds`，**全取仓库默认**（δ=0.3、PF 关闭）；与论文写明值不同，详见下文「参数对齐核查」 |
| 硬件 | 单张 RTX 4090（论文为 4090D） |
| 输出 | `data/test/goal_t00` … `goal_t09`（每集 `rollout_*.mp4` / `trace.jsonl` / `infer.jsonl`） |

### 分 task 结果（10 task × 50 trials = 500 集）

| suite | task | 任务描述 | 成功/总数 | 成功率 | 备注 |
| --- | --- | --- | --- | --- | --- |
| LIBERO-Goal | 0 | open the middle drawer of the cabinet | 46/50 | 92.0% | |
| LIBERO-Goal | 1 | put the bowl on the stove | 50/50 | 100.0% | |
| LIBERO-Goal | 2 | put the wine bottle on top of the cabinet | 48/50 | 96.0% | |
| LIBERO-Goal | 3 | open the top drawer and put the bowl inside | 41/50 | 82.0% | |
| LIBERO-Goal | 4 | put the bowl on top of the cabinet | 46/50 | 92.0% | |
| LIBERO-Goal | 5 | push the plate to the front of the stove | 23/50 | 46.0% | |
| LIBERO-Goal | 6 | put the cream cheese in the bowl | 49/50 | 98.0% | |
| LIBERO-Goal | 7 | turn on the stove | 50/50 | 100.0% | |
| LIBERO-Goal | 8 | put the bowl on the plate | 50/50 | 100.0% | |
| LIBERO-Goal | 9 | put the wine bottle on the rack | 39/50 | 78.0% | |
| | | **LIBERO-Goal 平均** | **442/500** | **88.4%** | |

对照论文 Goal **94.4%**，本次 **88.4%**，差 **-6.0** 个点。

### 延迟（排除每个 task 的首集，490 样本；剔首集依据见下方口径说明）

| 口径 | 本次复现（δ=0.3 / PF=0） | 论文 FLASH+Triton |
| --- | --- | --- |
| 服务端 `sample_actions_ms` 按集 mean / median | 15.3 / 16.0 ms | — |
| 服务端 `serve_time_ms` 按集 mean | 18.1 ms | 19.1 ms（task-level avg） |
| 服务端每动作 mean | 1.61 ms | 1.9 ms |
| 客户端往返按轮 mean / median | 18.6 / 12.7 ms | — |

口径与首集剔除说明：

- `episode_log.json: infer_latency_mean_ms` 实际记录服务端 `sample_actions_ms`，并非客户端往返；真正的往返值是 `infer.jsonl: client_roundtrip_ms`。论文对照应采用服务端口径。`total_ms` 为 encoder、VLM prefill、draft、verify、fallback 之和，最接近论文的 round 延迟边界。
- 每个 task 的 `episode_idx=0` 首轮可能出现 **5–15 s** 一次性成本，因此延迟统计统一剔除每个 task 的首集。同一 task 其余集的典型首轮仅约 44 ms。
- 成因是 Triton runtime 按 prompt token 长度缓存；长度变化时会清空旧 runtime 并重做 CUDA graph capture。该“池”实际是单槽：**付费次数 = prompt 长度变化次数 + 1**。20 个 task 中 17 个付费（Goal 9 个、LIBERO-10 8 个），与长度序列 20/20 吻合；免费首轮为 45–52 ms，付费首轮为 5087–12139 ms。
- 这笔成本取决于 task 顺序和 prompt 长度，不是新 prompt 的 language embedding 计算，也与被测模型无关。`language_embeds/*.pt` 均在 episode 开始前生成。
- `did_prefill` 只是 full/draft 路径指示位，不能指示缓存或上述一次性成本。每集第一轮为 full，是因为客户端逐集发送 `__reset_policy_state__` 并清空 KV 缓存。
- 不要与进程级冷启动混淆：`readme.md:135` 的前两次调用分别约 41.6 s（加载 6.5 GB 权重和 VLM prefill）与 13.3 s（首次编译 Triton kernel）；这是另一层开销。

### 参数对齐核查（关键）

服务端实际命令行（`ps` 读到的运行中进程，未传 `--periodic-full-every-n-draft-rounds`、未传 `--tau-radius`，全取仓库默认）：

| 项 | 论文 | 仓库默认 | 本次实际 |
| --- | --- | --- | --- |
| 接受阈值 δ | **0.15**（Table 3 caption、Table 7–8：*We fix the verification threshold to δ = 0.15*） | `tau_radius = 0.3` | **0.3** |
| 周期全路径刷新 PF | **PF = 2**（每 2 个 flash 轮强制 1 次全路径） | `periodic_full_every_n_draft_rounds = 0`（关闭） | **关闭** |
| 验证时间步数 K | K = 2 | `t_list = (0.10, 0.05)` → K = 2 | K = 2 ✓ |
| 硬件 | RTX 4090D | — | RTX 4090 |

- **PF 有硬证据**：论文 Table 3 明写 "+FB & PF=2 … for the LIBERO-10 result in Table 1"，且 Table 3 该行为 84.6 SR / **24.1 ms**，与 Table 1/2 中 LIBERO-10 的 Lat. **24.1 ms** 完全一致 → Table 1 用的就是 PF=2。本次跑的是 **PF 关闭**，即论文消融表里 LIBERO-10 仅 58.4% 的那一档。
- **δ 的位置**：论文 Algorithm 1 的判据 `dj ≤ δ` 即 [spec_pi0_pytorch.py:153](../policies/realtime-vla-flash/src/openpi/models_pytorch/spec_pi0_pytorch.py#L153) 的 `dist <= tau_radius`。

判定：**是变量没对齐，不是成功判据设错。** 判据为 LIBERO 原生 `_check_success()`，已全量核对 `episode_log.json` 的 `success` == 末条 trace 的 `done_after_step` == trace 中出现过 True，0 处不一致。

三处差异方向一致，都把配置推向论文 Pareto 曲线的"更激进 / 更快 / 更不准"一端：

- PF 关闭 → 全路径刷新少 → flash 路由占比高（79.7% vs 论文 62.4%）→ 长程误差无人纠正 → SR 下降；同时省下昂贵的全路径轮 → 延迟下降。
- 4090D 比 4090 少约 11% CUDA core（14592 vs 16384），同频慢约 11%：20.5 × 0.891 ≈ 18.3 ms，与实测 18.1 ms 吻合 → 延迟差主要是硬件，不是算法。
- δ = 0.3 比论文的 0.15 松一倍 → 接受前缀更长（79.4% vs 66.9%）。论文 Table 8 自扫过该曲线：δ 由 0.30 收紧到 0.15，Goal 上 SR 86.6 → 90.8、Acc 99.7 → 77.5，方向一致。

**后续实验结论**：δ=0.15 与 PF=2 同时启用后，Goal 从 88.4% 提升至 94.8%，总体缺口已经消失。但现有实验只有“默认参数”和“全部对齐”两个端点，尚不能分别量化 δ 与 PF 的贡献。论文仅明确坐实 LIBERO-10 使用 PF=2，其他 suite 是否相同仍待核实。

### 初步分析

1. **缺口高度集中**：58 个失败中 task 5 占 27 个（47%），task 9 占 11 个，task 3 占 9 个 —— 三个 task 合计占 81% 的失败。其余七个 task 均在 92–100%。

2. **失败集反而更依赖 flash path**：

   | | 集数 | draft 路由占比 | accepted 长度 | env_steps | infer 次数 |
   | --- | --- | --- | --- | --- | --- |
   | 成功 | 442 | 0.762 | 9.68 | 117 | 10.9 |
   | 失败 | 58 | 0.861 | 8.88 | 310（跑满上限） | 33.2 |

3. **flash 路径占比与接受长度都显著高于论文（Goal 对 Goal）**：

   | Goal 单 suite | 论文 FLASH+Triton（Table 2 Goal 列） | 本次复现 |
   | --- | --- | --- |
   | flash 路径占比 FR | 62.4% | **79.7%**（5378 / 6749 轮） |
   | 接受前缀 Acc | 66.9%（≈8.0 / 12） | **79.4%**（9.52 / 12） |
   | 延迟 Lat. | 20.5 ms | 18.1 ms |
   | 每动作 /Act | 2.1 ms | 1.61 ms |

   接受长度分布高度两极：76.4% 的 draft 轮整段 12 个动作全收，15.8% 整段被拒（前缀 0，直接回落全路径），中间 1–11 只占 7.8%。说明接受判定不是平滑衰减，而是"要么全过、要么全推"。逐轮距离 `radius_dist_mean`：p50 = 0.066、p90 = 0.124、p99 = 0.213，仅 0.4% 的轮次均值超过当前的 δ = 0.3。

   差距的成因见上一节「参数对齐核查」：PF 关闭 + δ 取 0.3，两处都与论文写明值不同，方向与"更快 + 更激进 + 成功率低"完全吻合。

4. **task 5 在默认参数下出现「draft 自信地做错」**：

   | task | 成功集 draft / accepted | 失败集 draft / accepted |
   | --- | --- | --- |
   | 5（46%） | 0.938 / **11.98** | 0.947 / 9.32 |
   | 9（78%） | 0.818 / 10.36 | 0.745 / 7.87 |
   | 3（82%） | 0.822 / 10.22 | 0.874 / 10.33 |

   task 5 的成功集里 draft 接受长度达 11.98/12，但仍有近一半 episode 失败，说明 δ=0.3 时验证器未拦住错误 draft。后续基线与参数对齐实验已验证这一解释：全路径基线为 98%，δ=0.15 / PF=2 为 96%，默认参数仅 46%。具体失败行为仍待逐条回放。

5. **归因更新**：默认配置的主要缺口来自参数错配；但 δ 与 PF 各自贡献仍未分离。task 3 的 FLASH 82% 与全路径基线 88% 仅差 0.9σ，当前样本量不足以判断是否存在真实差异。

### 待办 / 下一步

- [x] 参数对齐后重跑四个 suite：δ=0.15、PF=2，各 500 集；Goal 由 88.4% 提升至 94.8%。
- [x] Triton-π0 全路径基线：使用 `--force-full-each-round`，Goal task 3/5/9 各 50 集。Triton 模式不能省略 `--draft-triton-path`，否则 `spec_serve_policy.py:882` 会报错。
- [x] 补齐 Spatial、Object、Goal、10：`paper_libero_{spatial,object,goal,10}`，共 2000 集。
- [ ] 逐条回放 112 个失败 rollout，填写「失败的rollout观察」；优先检查 LIBERO-10 task 8 的 22 条和 task 9 的 17 条。
- [ ] 如需拆分 δ 与 PF 的贡献，补跑两组中间配置：δ=0.15 / PF=0 与 δ=0.3 / PF=2。

> 已知字段限制：Triton 路径没有写入 `gripper_override_rate`；客户端记录的 0.0 是默认值，不能用于判断 gripper override 是否生效。

> 重新生成以上表格：
> `python3 scripts/summarize_flash_runs.py 'policies/realtime-vla-flash/data/test/goal_t0*'`


## 参数对齐后的四 suite 复现（δ=0.15 / PF=2）

上面那节是**仓库默认参数**（δ=0.3、PF 关闭）下的 LIBERO-Goal。本节按论文写明值重跑，并把四个 suite 补齐 —— 四者共用同一套变量，因此彼此可比、也与论文可比。

### 复现协议（四个 suite 完全同一套变量）

| 项 | 取值 |
| --- | --- |
| 基座 | openpi 官方 `pi0_libero`（GCS），未做任何训练 |
| draft | 官方 `Dexmal/RealtimeVLA-Flash`，**每个 suite 一份**，官方 `convert_for_triton.py` 转 Triton：`converted/draft`（goal）/ `draft_spatial` / `draft_object` / `draft_10` |
| 服务端 | 每 suite 独立进程（draft 权重不同，无法共用），`--tau-radius 0.15` `--periodic-full-every-n-draft-rounds 2`，K=2（`t_list` 取仓库默认 `(0.10,0.05)`），`max_exec_steps=12` |
| 客户端 | `--replan-steps 12 --num-trials-per-task 50 --task 0-9`，`num_steps_wait=10`，`resize_size=224`，seed=7 |
| 参数依据 | δ=0.15 与 K=2 见论文正文：*We use K = 2, δ = 0.15 as the base verifier setting for the full runtime*；PF=2 由 Table 1 的 LIBERO-10 延迟 24.1 ms 与 Table 3 的 `+FB & PF=2` 行（84.6 SR / 24.1 ms）逐一坐实 |
| 成功判据 | LIBERO 原生 `_check_success()`（BDDL goal 谓词求值），非自定义 |
| 硬件与并行 | RTX 4090 ×2；每张卡压 2 服务端 + 2 客户端（spatial+object 在 GPU 0，goal+10 在 GPU 1）。论文是单张 4090D |
| 输出 | `data/test/paper_libero_{spatial,object,goal,10}`，每集 `rollout_*.mp4` / `trace.jsonl` / `infer.jsonl` |

### 四 suite 成功率 vs 论文 Table 1

| suite | 本次 | 论文 FLASH+Triton-π0 | Δ | 失败集 |
| --- | --- | --- | --- | --- |
| LIBERO-Spatial | 97.6%（488/500） | 96.8 | **+0.8** | 12 |
| LIBERO-Object | 99.6%（498/500） | 99.2 | **+0.4** | 2 |
| LIBERO-Goal | 94.8%（474/500） | 94.4 | **+0.4** | 26 |
| LIBERO-10 | 85.6%（428/500） | 84.6 | **+1.0** | 72 |
| **平均** | **94.4%（1888/2000）** | 93.8 | **+0.6** | 112 |

四个 suite 全部不低于论文，平均高 0.6 个点。同一份复现在默认参数下 Goal 只有 88.4% —— **差的那 6 个点是参数没对齐造成的，不是成功判据、也不是环境**。

### 分 task 结果（4 suite × 10 task × 50 trials = 2000 集）

| suite | task | 任务描述 | 成功/总数 | 成功率 | 备注 |
| --- | --- | --- | --- | --- | --- |
| LIBERO-Spatial | 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 50/50 | 100.0% | |
| LIBERO-Spatial | 1 | pick up the black bowl next to the ramekin and place it on the plate | 49/50 | 98.0% | |
| LIBERO-Spatial | 2 | pick up the black bowl from table center and place it on the plate | 50/50 | 100.0% | |
| LIBERO-Spatial | 3 | pick up the black bowl on the cookie box and place it on the plate | 50/50 | 100.0% | |
| LIBERO-Spatial | 4 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 49/50 | 98.0% | |
| LIBERO-Spatial | 5 | pick up the black bowl on the ramekin and place it on the plate | 48/50 | 96.0% | |
| LIBERO-Spatial | 6 | pick up the black bowl next to the cookie box and place it on the plate | 50/50 | 100.0% | |
| LIBERO-Spatial | 7 | pick up the black bowl on the stove and place it on the plate | 46/50 | 92.0% | |
| LIBERO-Spatial | 8 | pick up the black bowl next to the plate and place it on the plate | 48/50 | 96.0% | |
| LIBERO-Spatial | 9 | pick up the black bowl on the wooden cabinet and place it on the plate | 48/50 | 96.0% | |
| LIBERO-Object | 0 | pick up the alphabet soup and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 1 | pick up the cream cheese and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 2 | pick up the salad dressing and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 3 | pick up the bbq sauce and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 4 | pick up the ketchup and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 5 | pick up the tomato sauce and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 6 | pick up the butter and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 7 | pick up the milk and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 8 | pick up the chocolate pudding and place it in the basket | 50/50 | 100.0% | |
| LIBERO-Object | 9 | pick up the orange juice and place it in the basket | 48/50 | 96.0% | |
| LIBERO-Goal | 0 | open the middle drawer of the cabinet | 49/50 | 98.0% | |
| LIBERO-Goal | 1 | put the bowl on the stove | 49/50 | 98.0% | |
| LIBERO-Goal | 2 | put the wine bottle on top of the cabinet | 50/50 | 100.0% | |
| LIBERO-Goal | 3 | open the top drawer and put the bowl inside | 41/50 | 82.0% | |
| LIBERO-Goal | 4 | put the bowl on top of the cabinet | 45/50 | 90.0% | |
| LIBERO-Goal | 5 | push the plate to the front of the stove | 48/50 | 96.0% | |
| LIBERO-Goal | 6 | put the cream cheese in the bowl | 46/50 | 92.0% | |
| LIBERO-Goal | 7 | turn on the stove | 50/50 | 100.0% | |
| LIBERO-Goal | 8 | put the bowl on the plate | 49/50 | 98.0% | |
| LIBERO-Goal | 9 | put the wine bottle on the rack | 47/50 | 94.0% | |
| LIBERO-10 | 0 | put both the alphabet soup and the tomato sauce in the basket | 42/50 | 84.0% | |
| LIBERO-10 | 1 | put both the cream cheese box and the butter in the basket | 48/50 | 96.0% | |
| LIBERO-10 | 2 | turn on the stove and put the moka pot on it | 43/50 | 86.0% | |
| LIBERO-10 | 3 | put the black bowl in the bottom drawer of the cabinet and close it | 45/50 | 90.0% | |
| LIBERO-10 | 4 | put the white mug on the left plate and put the yellow and white mug on the right plate | 43/50 | 86.0% | |
| LIBERO-10 | 5 | pick up the book and place it in the back compartment of the caddy | 48/50 | 96.0% | |
| LIBERO-10 | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate | 48/50 | 96.0% | |
| LIBERO-10 | 7 | put both the alphabet soup and the cream cheese box in the basket | 50/50 | 100.0% | |
| LIBERO-10 | 8 | put both moka pots on the stove | 28/50 | 56.0% | |
| LIBERO-10 | 9 | put the yellow and white mug in the microwave and close it | 33/50 | 66.0% | |

失败最集中的两个格子是 **LIBERO-10 task 8（28/50）** 和 **LIBERO-10 task 9（33/50）**，占全部 112 个失败的 39/112。

### FR / Acc / 延迟 vs 论文 Table 2

| suite | FR 本次 | FR 论文(+Tri) | Acc 本次 | Acc 论文(+Tri) | Lat. 本次 | Lat. 论文(+Tri) | /Act 本次 | /Act 论文(+Tri) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial | 58.5% | 68.9% | 59.2% | 60.2% | 20.4 ms | 19.2 | 2.18 | 2.1 |
| Object | 58.6% | 86.4% | 69.7% | 89.3% | 20.1 ms | 12.6 | 1.98 | 1.2 |
| Goal | 60.8% | 62.4% | 63.1% | 66.9% | 19.8 ms | 20.5 | 2.25 | 2.1 |
| 10 | 61.2% | 49.6% | 52.3% | 62.5% | 19.2 ms | 24.1 | 2.33 | 2.3 |

口径：FR = `route_type == "draft"` 的轮数 / 全部 replan 轮数；Acc = draft 轮的接受前缀长度 / 12 再对轮加权（`accepted_metric_rounds = draft_rounds` 是客户端自己的定义，见 `spec_client_libero.py:320`）；表里 Lat. 取服务端 `total_ms`（`encoder + vlm_prefill + draft + verify + full_fallback` 之和，最接近论文 flash-path round 的边界）按轮均值。

同一批数据的其他延迟口径（四 suite，均剔除每 task 首集 —— 首集第一轮有一笔 5–15 s 的一次性成本，见上方口径说明）：

| 口径 | 定义位置 | 四个 suite |
| --- | --- | --- |
| 服务端 `total_ms` 按轮 mean / median | `infer.jsonl` | 19.2–20.4 / **7.9 ms** |
| 服务端 `sample_actions_ms` 按集 mean | `episode_log.json: infer_latency_mean_ms`（**注意它叫 infer_latency，实际是服务端 `sample_actions_ms`**，`spec_client_libero.py:191`） | 20.2–21.6 ms |
| 服务端 `serve_time_ms` 按集 mean | `episode_log.json: serve_time_mean_ms` | 25.2–26.7 ms |
| 客户端往返（含 WS 收发与客户端自身开销）按轮 mean / median | `infer.jsonl: client_roundtrip_ms` | 26.4–28.0 / 16.7–17.3 ms |
| 同上，按路径拆分 mean（draft / full） | `infer.jsonl: client_roundtrip_ms` | goal **16.1 / 45.4 ms**；10 **15.1 / 44.2 ms** |

只有 `total_ms` 与论文的 Lat. 边界一致；客户端往返那一项天然比论文高，不可直接比。生成脚本 `scripts/flash_fr_acc.py`。

按路径拆分那一行值得单看：**draft 轮在客户端视角约 15 ms，full 轮约 44 ms，相差近三倍** —— 这才是 FLASH 加速在端到端视角的实际形态，也和 FR≈61% 的权重自洽。

三个观察：

1. **延迟中位数 7.9 ms（四个 suite 完全一致）≈ 论文 flash-path round 的 7.8 ms（FLASH+Triton-π0）**。均值之所以到 ~20 ms，是因为 PF=2 下每三轮就有一次 37–38 ms 的全路径轮。这与论文 Table 3 里 PF=2 行的 24.1 ms 是同一量级。
2. **Goal 的三项几乎逐一对上**（FR 60.8 / 62.4、Acc 63.1 / 66.9、Lat 19.8 / 20.5），说明这套复现在 Goal 上复刻的是论文的 +Tri 档。
3. **FR 的形状与论文不同**：论文四个 suite 的 FR 从 49.6% 到 86.4% 大幅变化，我们则稳定在 58.5–61.2%。PF=2 每两轮投一轮全路径，理论上限是 66.7%，我们贴着这个上限下方；而论文 Object 的 86.4% **超过了这个上限**，说明论文各 suite 未必都用 PF=2（论文只把 LIBERO-10 的 PF=2 用延迟数字钉死了，其余 suite 的消融在 Appendix C.1）。我们为了控制变量对四个 suite 用了同一套设置，这是一处已知的、方向明确的偏差。

### 基线对照：Triton-π0（无投机执行）

为回答"某些 task 差是 π0 本身的天花板，还是 FLASH 的接受链引入的"，在空出的 GPU 0 上跑了论文的 Triton-π0 基线：服务端加 `--force-full-each-round`（`spec_pi0_pytorch.py:206`：force_full 时每轮都走全路径，不会有任何 draft 动作被接受），只跑 Goal 的三个关键 task 各 50 集。全路径轮的 `draft_ms=0.00`，即 draft 权重确实没参与。

| LIBERO-Goal | π0 全路径基线（50 集/task） | FLASH δ=0.3 / PF=0 | FLASH δ=0.15 / PF=2 |
| --- | --- | --- | --- |
| task 3 open the top drawer and put the bowl inside | 44/50 = **88.0%** | 82.0% | 82.0% |
| task 5 push the plate to the front of the stove | 49/50 = **98.0%** | 46.0% | 96.0% |
| task 9 put the wine bottle on the rack | 43/50 = **86.0%** | 78.0% | 94.0% |
| 合计 | 136/150 = 90.7% | 103/150 = 68.7% | 136/150 = 90.7% |

三条结论：

1. **task 5 的 46% 是错配配置的假象**。π0 全路径能拿到 98%，δ=0.15 下 FLASH 也拿到 96%，只有 δ=0.3 时崩到 46%。之前那轮把 task 5 归因成"验证器没拦住自信的错误 draft"，方向对，但根因是 δ=0.3 让验证器形同虚设，而不是 FLASH 机制本身。
2. **task 3 的 82% 更像是 π0 / 环境的天花板，但这点数据分不出来**：纯 π0 也只有 88%，差 6 个点在 n=50（SE≈7%）下约 0.9σ。证据强度只到"**没有证据表明这 6 个点是 FLASH 引入的**"，**不能反过来读成"FLASH 在这格和基线一样好"** —— 那是把"未观察到差异"当成了"已证明无差异"。要把 6 个点判成真差异，需把集数加到 n≈300 才能从 1σ 拉到 2σ。
3. **task 9 上 FLASH（94%）高于纯 π0（86%）的方向**有个机制解释：`spec_client_libero.py:210` 的 `exec_len = min(replan_steps, accepted)` —— 接受前缀越短，本次执行的步数越少、下一次 replan 越早，等于**重规划更频繁、闭环修正更及时**。但 8 个点在 n=50 下约 1.6σ，**尚不能排除噪声，不作为结论**。

### 本节的口径与已知瑕疵

- **本节延迟是“共卡争抢”口径**：每张 4090 上压着 2 个服务端 + 2 个客户端，因此不能直接作为单卡隔离结果。后续单卡独占全量复测已完成，见下一节；其 `total_ms` 均值只降低 0.1–0.4 ms，证明争抢影响很小。
- **`paper_libero_goal` / `paper_libero_10` 的 `manifest.json` 里 `created_at` 是错的**（指向 18:28:08Z）：起因是第一个会话的 driver 跑完 spatial+object 后自动进入了第二轮，在 8000/8001 上起了 goal/10 服务端，客户端启动的下一秒就被人工终止。已核对 `episode_log.json`：goal n=500、10 n=500，`(task_id, episode_idx)` 唯一对均无重复、每 task 恰好 50、`run_id` 单一、`seed=7` —— **一条 episode 都没有混入**，只有这一个 cosmetic 字段被覆盖。产物本身未做任何手工修改。根治：`run_flash_parallel.sh` 增加 `ROUNDS_OVERRIDE`，单跑某一轮不再需要抢时间差。
- **`episodes/` 目录里有上一轮夭折留下的陈旧 episode 目录**（与新一轮同名者已被覆盖，但若某集由 success 变 failure 会留下两个目录）。统计一律以 `episode_log.json` 为准，不数目录。
- 硬件是 RTX 4090（16384 CUDA core），论文是 4090D（14592）；单看延迟论文应比我们略慢约 11%，本节延迟反而略高，差异来自共卡争抢而非硬件。

### 待办 / 下一步（本节）

- [x] 对齐参数后重跑（δ=0.15、PF=2，见上文）
- [x] 基线对照（Triton-π0，无投机）：task 3 / 5 / 9 各 50 集
- [x] 补齐其余三个 suite（spatial / object / 10）
- [x] **单卡独占延迟复测**：已于 2026-09-25 完成，而且按新需求跑了全量 2000 集并记录轨迹长度，见下一节。
- [ ] **失败回放**：逐条检查 112 个失败 rollout，优先处理 LIBERO-10 task 8 的 22 条和 task 9 的 17 条，并填写下一节。
- [ ] **核实逐 suite 的 PF 设置**：论文 Object 的 FR 86.4% 超过 PF=2 的 66.7% 理论上限，需查 Appendix C.1，确认论文是否仅部分 suite 使用 PF=2。
- [ ] **分离 δ 与 PF 的贡献（可选消融）**：补跑 δ=0.15 / PF=0 与 δ=0.3 / PF=2 两个中间配置。

> 重新生成本节表格：
> `python3 scripts/summarize_flash_runs.py 'policies/realtime-vla-flash/data/test/paper_libero_*'`
> `python3 scripts/flash_fr_acc.py 'policies/realtime-vla-flash/data/test/paper_libero_*'`


## 单卡独占全量复测（2026-09-25）

本轮只完成用户指定的两项：单卡独占复测，以及逐 episode 记录 trajectory length；未开展失败回放、PF 核查或其他消融。

### 实验协议

- 规模与参数不变：4 suite × 10 task × 50 trials = **2000 集**，δ=0.15、PF=2、K=2、`replan_steps=12`、seed=7。
- 四张 RTX 4090 同时运行，但每张卡只承载一个 suite 的 **1 个服务端 + 1 个 EGL 客户端**：Spatial→GPU0、Object→GPU1、Goal→GPU2、LIBERO-10→GPU3。卡之间无任务共享，因此每个 suite 都是单卡独占口径。
- 输出目录：`data/test/exclusive_full_libero_{spatial,object,goal,10}`；日志：`logs/flash_exclusive_full/`；总耗时 86 分钟。
- `trajectory_length` 定义为 episode 中实际执行的策略动作数，不含 `num_steps_wait=10` 的 warm-up dummy actions；它应等于 `executed_action_count`，也应等于 `trace.jsonl` 中 `executed_action != null` 的记录数。定义同时写入各 run 的 `manifest.json`。

### 成功率

| suite | 单卡独占 | 原共卡实验 | 论文 | 独占 Δ论文 |
| --- | --- | --- | --- | --- |
| LIBERO-Spatial | 98.0%（490/500） | 97.6% | 96.8% | +1.2 |
| LIBERO-Object | 98.4%（492/500） | 99.6% | 99.2% | -0.8 |
| LIBERO-Goal | 93.8%（469/500） | 94.8% | 94.4% | -0.6 |
| LIBERO-10 | 85.2%（426/500） | 85.6% | 84.6% | +0.6 |
| **平均** | **93.9%（1877/2000）** | **94.4%** | **93.8%** | **+0.1** |

单卡独占轮仍复现了论文的总体成功率；逐 suite 有 ±1 个点左右波动，不应把某一轮的逐格高低解读为硬件隔离造成的确定性变化。

分 task 成功数（每格分母均为 50）：

| suite | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial | 50 | 49 | 50 | 50 | 49 | 49 | 50 | 47 | 48 | 48 |
| Object | 49 | 50 | 48 | 50 | 48 | 49 | 50 | 50 | 50 | 48 |
| Goal | 48 | 50 | 50 | 44 | 45 | 43 | 49 | 50 | 49 | 41 |
| 10 | 40 | 47 | 40 | 49 | 41 | 49 | 46 | 49 | 25 | 40 |

### 单卡独占延迟

以下均排除每个 task 的首集；`Lat.` 是服务端 `total_ms` 按轮统计，与论文 round latency 的边界最接近。

| suite | 独占 Lat. mean / median | 原共卡 mean / median | mean 变化 | 独占 /Act | 论文 Lat. / /Act |
| --- | --- | --- | --- | --- | --- |
| Spatial | 20.1 / 8.0 ms | 20.4 / 7.9 ms | -0.3 ms | 2.15 ms | 19.2 / 2.1 ms |
| Object | 20.0 / 8.0 ms | 20.1 / 7.9 ms | -0.1 ms | 2.00 ms | 12.6 / 1.2 ms |
| Goal | 19.4 / 7.9 ms | 19.8 / 7.9 ms | -0.4 ms | 2.30 ms | 20.5 / 2.1 ms |
| 10 | 19.0 / 7.8 ms | 19.2 / 7.9 ms | -0.2 ms | 2.33 ms | 24.1 / 2.3 ms |

独占后均值只下降 **0.1–0.4 ms**，flash-path 中位数仍为 **7.8–8.0 ms**，因此原共卡实验的争抢影响很小。Object 与论文的延迟差异仍不能归因于共卡；更可能与论文逐 suite 的 PF 设置不一致有关（论文 Object FR=86.4%，不可能来自 PF=2 的 66.7% 上限）。

同一轮的 FR / Acc：Spatial 58.5% / 60.1%，Object 58.6% / 68.5%，Goal 60.7% / 61.4%，LIBERO-10 61.3% / 52.2%。四者的 FR 仍贴近 PF=2 下的共同上限。

### 轨迹长度与任务成败

| suite | 成功轨迹 mean / median（range） | 失败轨迹 mean / median（range） | r(length, success) |
| --- | --- | --- | --- |
| LIBERO-Spatial | 105.4 / 104.0（66–212） | 220.0 / 220.0（220–220） | -0.627 |
| LIBERO-Object | 133.6 / 130.0（106–265） | 280.0 / 280.0（280–280） | -0.736 |
| LIBERO-Goal | 111.4 / 92.0（64–286） | 300.0 / 300.0（300–300） | -0.752 |
| LIBERO-10 | 258.1 / 247.0（139–491） | 520.0 / 520.0（520–520） | -0.864 |
| **全部** | **148.9 / 126.0（64–491）** | **424.6 / 520.0（220–520）** | **-0.664** |

表中的 `r` 是 trajectory length 与布尔 success 的点二列相关系数。这里的强负相关是**终止规则直接造成的结果**：1877 条成功轨迹都会在 `_check_success()` 成立时提前结束，123 条失败轨迹则全部跑满各 suite 的 `max_steps`。本轮数据中，`trajectory_length < max_steps` 与成功、`trajectory_length == max_steps` 与失败是 2000/2000 完全对应的，因此该字段事后能完美区分成败，但属于标签泄漏，不能据此声称“轨迹越长越容易失败”。若要研究有预测意义的关系，应比较固定早期窗口内的轨迹/动作特征，而不是最终 rollout 长度。

### 完整性核验

- 四个 suite 均为 500 episode；`(task_id, episode_idx)` 各自 500 个唯一值，每 task 恰好 50 条。
- 2000/2000 条 `trajectory_length == executed_action_count == trace 中非空动作数`。
- 2000/2000 条 `episode_log.success` 与 trace 中是否出现 `done_after_step=true` 一致。
- 四个客户端均以 0 退出；完成后 GPU 回到 1 MiB / 0%。客户端销毁 EGL context 时各打印一次缺少 `libGLU.so.0` 的 `Exception ignored`，发生在全部 500 集写盘之后，不影响退出状态和上述核验。

> 运行：`bash scripts/run_flash_exclusive.sh`
>
> 重新汇总：`python3 scripts/summarize_flash_runs.py 'policies/realtime-vla-flash/data/test/exclusive_full_*'`
>
> FR / Acc / round latency：`python3 scripts/flash_fr_acc.py 'policies/realtime-vla-flash/data/test/exclusive_full_*'`


## 原始 π0 全路径基线与 FLASH 对照（2026-09-26）

### 对照协议

- 原始模型采用论文的 **Triton-π0** 对照：服务端加 `--force-full-each-round`，每轮都走 π0 full path；draft artifact 仅因 runtime 初始化接口要求而加载，实际所有轮次 `is_full_pipeline_round=1` 且 `draft_ms=0`。
- 其余条件与最新 FLASH 单卡独占轮完全相同：4 suite × 10 task × 50 trials = **2000 集**，seed=7，初始状态逐集对应，`num_steps_wait=10`，`replan_steps=max_exec_steps=12`，每个 suite 独占一张 RTX 4090。
- π0 输出为 `data/test/pi0_baseline_full_libero_{spatial,object,goal,10}`；FLASH 对照为 `data/test/exclusive_full_libero_{spatial,object,goal,10}`。π0 全量耗时 90 分钟。
- 对比键为 `(suite, task_id, episode_idx)`；2000 对一一对应。注意两次运行使用相同初始状态，但 diffusion noise 未固定成逐轮相同，因此是“初始状态配对”，不是逐动作确定性复演。

### 成功率：总体无显著变化

| suite | 原始 π0 | FLASH | ΔFLASH | π0-only / FLASH-only |
| --- | --- | --- | --- | --- |
| LIBERO-Spatial | 96.6%（483/500） | 98.0%（490/500） | +1.4 pp | 9 / 16 |
| LIBERO-Object | 97.6%（488/500） | 98.4%（492/500） | +0.8 pp | 8 / 12 |
| LIBERO-Goal | 94.0%（470/500） | 93.8%（469/500） | -0.2 pp | 27 / 26 |
| LIBERO-10 | 85.2%（426/500） | 85.2%（426/500） | 0.0 pp | 53 / 53 |
| **全部** | **93.35%（1867/2000）** | **93.85%（1877/2000）** | **+0.50 pp** | **97 / 107** |

总体成功差的配对正态近似 95% CI 为 **[-0.90,+1.90] pp**，McNemar exact `p=0.529`。因此这批 2000 集支持的结论是：**FLASH 在把推理路径大幅缩短的同时，保持了原始 π0 的总体成功率**；不能把观测到的 +0.5 pp 宣称成真实提升。

同一初始状态并不保证两种运行得到同一结果：1770 集二者都成功，26 集都失败，97 集只有 π0 成功，107 集只有 FLASH 成功。这里既包含 draft 接受/更早重规划带来的策略差异，也包含两次独立 diffusion noise；不能把某一集的翻转全部归因于 FLASH。

### 分 task 成功数（每格分母 50）

原始 π0：

| suite | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial | 50 | 49 | 50 | 49 | 47 | 50 | 48 | 46 | 47 | 47 |
| Object | 48 | 50 | 49 | 50 | 50 | 47 | 48 | 49 | 50 | 47 |
| Goal | 48 | 47 | 50 | 45 | 47 | 43 | 46 | 50 | 50 | 44 |
| 10 | 47 | 50 | 44 | 45 | 40 | 47 | 38 | 49 | 28 | 38 |

`FLASH - π0` 的成功集数变化（每多/少 1 集等于 ±2 pp）：

| suite | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Spatial | 0 | 0 | 0 | +1 | +2 | -1 | +2 | +1 | +1 | +1 |
| Object | +1 | 0 | -1 | 0 | -2 | +2 | +2 | +1 | 0 | +1 |
| Goal | 0 | +3 | 0 | -1 | -2 | 0 | +3 | 0 | -1 | -3 |
| 10 | -7 | -3 | -4 | +4 | +1 | +2 | +8 | 0 | -3 | +2 |

最大正负变化分别是 LIBERO-10 task 6 的 +16 pp 和 task 0 的 -14 pp，但 40 个 task 同时比较、每格仅 50 集且两次 noise 独立；这些逐格摆动只作为定位线索，不作为“FLASH 改善/破坏某任务”的确定结论。

### 轨迹长度：总体几乎不变

`成功轨迹` 列各自使用该方法成功的样本，因此集合不同；`双成功配对` 只使用同一初始状态下两边都成功的 episode，是判断机制是否改变完成步数的更干净口径。

| suite | π0 成功轨迹 mean / median | FLASH 成功轨迹 mean / median | 双成功 N | 双成功配对 π0→FLASH | 配对 Δ步数 |
| --- | --- | --- | --- | --- | --- |
| Spatial | 106.3 / 105 | 105.4 / 104 | 474 | 106.0 → 104.8 | -1.2 |
| Object | 136.2 / 134 | 133.6 / 130 | 480 | 136.2 → 133.5 | -2.6 |
| Goal | 111.9 / 93 | 111.4 / 92 | 443 | 110.4 → 110.3 | -0.1 |
| LIBERO-10 | 259.0 / 249 | 258.1 / 247 | 373 | 252.8 → 253.3 | +0.6 |
| **全部** | **150.4 / 127** | **148.9 / 126** | **1770** | **146.2 → 145.3** | **-0.93** |

总体配对差的 95% CI 是 **[-2.26,+0.39] 步**，中位数为 -1 步：没有证据表明 FLASH 系统性拉长或缩短成功轨迹。Object 的双成功轨迹平均短 2.65 步（该 suite 的 95% CI `[-4.18,-1.12]`），量级约为其平均轨迹的 2%；考虑同时查看四个 suite，这只算小幅局部差异。

失败轨迹在两种方法下都由环境上限决定：Spatial 220、Object 280、Goal 300、LIBERO-10 520 个策略动作。因此“轨迹长度与成功强负相关”仍是提前终止规则造成的标签泄漏，不是 FLASH 的因果效应。

### FLASH 模块真正改变了什么

| suite | π0 轮数 / Lat. | FLASH 轮数 / Lat. | FLASH FR / Acc | round 加速 | π0→FLASH /Act |
| --- | --- | --- | --- | --- | --- |
| Spatial | 4711 / 37.5 ms | 6032 / 20.1 ms | 58.5% / 60.1% | 1.87× | 3.12 → 2.15 ms |
| Object | 5927 / 37.1 ms | 7059 / 20.0 ms | 58.6% / 68.5% | 1.86× | 3.09 → 2.00 ms |
| Goal | 5261 / 37.3 ms | 6812 / 19.4 ms | 60.7% / 61.4% | 1.92× | 3.11 → 2.30 ms |
| LIBERO-10 | 12413 / 37.0 ms | 17482 / 19.0 ms | 61.3% / 52.2% | 1.95× | 3.08 → 2.33 ms |

- 原始 π0 的 FR=0%，所有轮次均为约 37 ms 的 full path；FLASH 约 59–61% 的轮次走 7.8–7.9 ms draft path，使平均 round latency 降低约 **46–49%**。
- FLASH 接受前缀常小于 12，因此会更早重规划；推理轮数相对 π0 增加约 **19–41%**。即便如此，每动作延迟仍降低约 **24–35%**。
- 因此本次观察到的主要影响是**计算路径和重规划频率改变**，而不是最终成功率或任务完成轨迹长度改变。

### 完整性与复现入口

- 4 suite 均为 500 集；2000 个 `(suite, task_id, episode_idx)` 与 FLASH 一一匹配。
- 2000/2000 条 `trajectory_length == executed_action_count`；所有 infer 记录均为 full path 且 `draft_ms=0`；客户端均正常退出，运行异常为 0。
- 运行：`bash scripts/run_pi0_baseline_exclusive.sh`
- 独立汇总：`python3 scripts/summarize_flash_runs.py 'policies/realtime-vla-flash/data/test/pi0_baseline_full_*'`
- 配对比较：`python3 scripts/compare_pi0_flash.py --baseline 'policies/realtime-vla-flash/data/test/pi0_baseline_full_*' --flash 'policies/realtime-vla-flash/data/test/exclusive_full_*' --json-out policies/realtime-vla-flash/data/test/pi0_vs_flash_comparison.json`
- 机器可读结果：`data/test/pi0_vs_flash_comparison.json`


## 失败的rollout观察

（本轮按要求未做。原共卡轮有 112 条失败；最新单卡独占轮有 123 条失败，后续回放时需明确使用哪一轮数据。）


## ALOHA 复现（已按要求暂停，2026-09-26）

### 范围说明

- Realtime-VLA FLASH 论文的实机实验不是 ALOHA，而是单臂 UR5 + 两台 Intel RealSense D435i 的传送带分拣；官方公开仓库和 `Dexmal/RealtimeVLA-Flash` 权重目前只提供 LIBERO 的四份 draft，没有 ALOHA draft、ALOHA FLASH 客户端或论文实机数据。
- 因而可直接复现的 ALOHA 起点是 OpenPI 官方 **ALOHA-Sim TransferCube π0 baseline**。在此基础上生成 cache、训练 14 维 ALOHA draft 并接入 FLASH，属于新增扩展实验，不能写成论文原表复现。
- 当前只做 TransferCube；OpenPI 没有发布 ALOHA-Sim Insertion 的专用 checkpoint，先不把不匹配的 checkpoint 用到 Insertion 上。

### 实验协议

| 项 | 取值 |
| --- | --- |
| 环境 | `gym_aloha/AlohaTransferCube-v0` |
| 基座 | OpenPI 官方 `pi0_aloha_sim` / `gs://openpi-assets/checkpoints/pi0_aloha_sim` |
| 数据 | `lerobot/aloha_sim_transfer_cube_human` 的 v2.1 tag；50 episodes、20,000 frames、50 Hz |
| 任务文本 | `Transfer cube` |
| action | 14 维双臂 joint + gripper；模型 chunk=50，每轮执行 10 步后重规划 |
| 规模 | seed 0–49，共 50 trials；每集上限 300 actions |
| 成功判据 | gym-aloha 原生 `reward == 4`，此时 `info["is_success"] = True` 并立即终止 |
| trajectory length | 实际送入环境的 action 数；成功时提前终止，失败时通常跑满 300 |
| 硬件 | 单张 RTX 4090 独占；启动前若显存已占用超过 1 GiB，脚本拒绝运行 |
| 输出 | `data/aloha_sim/pi0_transfer_cube_seed0_50ep/{manifest,episode_log,summary}.json` 和逐集视频 |

运行入口：

```bash
GPU=0 TRIALS=50 bash scripts/run_aloha_baseline.sh
```

`summary.json` 会分别给出成功/失败轨迹长度的 mean、median，以及 trajectory length 与 success 的点二列相关系数；`episode_log.json` 保留每集的 seed、成败、最大 reward、trajectory length、推理次数和逐轮延迟。

### 当前状态 / 待办

- [x] ALOHA-Sim 环境 reset / render / step 冒烟验证。
- [x] 下载并验证兼容当前代码的 LeRobot v2.1 数据（主分支 v3.0 缺少本仓库 LeRobot 0.1.0 所需的 `tasks.jsonl`，不能直接使用）。
- [x] 完成定量 evaluator 和单卡独占启动脚本；脚本与 Python 语法检查通过。
- [x] 官方 `pi0_aloha_sim` checkpoint 已完整下载并核对参数元数据与 norm stats（约 12 GB）。
- [ ] GPU 重新空闲后，先跑 1 集端到端冒烟。
- [ ] 单卡独占跑 50 集 π0 baseline，填入成功率及 trajectory length—成败统计。
- [ ] 用 ALOHA 数据生成 encoder/action cache，训练 14 维 draft，并完成 ALOHA FLASH 50 集对照。
