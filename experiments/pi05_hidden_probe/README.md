# π0.5 隐藏表征探测

实现根目录的 `idea1_pi05_hidden_representation_inspect_plan.md` 主路线：正常策略输入 → 冻结 Source 表征 → 冻结 Target 注入 → 属性回答及反事实对照。

当前支持本项目 `policies/openpi` 的 **标准 JAX `pi05_libero`**。复用原输入变换、checkpoint、Gemma block、final RMSNorm 和共享词表投影，不修改策略源码。其他 JAX 权重只有符合相同结构和输入契约时才能通过 `target_model` 使用；原生 Hugging Face PaliGemma 可作为文本任务阳性参考，暂不支持将其作为跨模型 patching Target。学习映射、ROI 和开放式事实人工评分属于后续扩展。

## 环境与验证

在项目根目录执行；已有 Python 环境不需要另行安装模型依赖：

```bash
cd /root/nfs/embodied/benchmark
PROBE_PY="$PWD/policies/openpi/.venv/bin/python"
export OPENPI_DATA_HOME="$PWD/shared/checkpoints/openpi"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PROBE_PY" -m pytest -q experiments/pi05_hidden_probe/tests
```

单元测试使用小随机模型和明确标记的测试数据，不下载权重，不访问 GPU，不产生语义实验结果。真实实验请在已有空闲设备上设置 `CUDA_VISIBLE_DEVICES` 后运行下列模型命令。`JAX_PLATFORMS=cpu` 可用于 CPU 验证。

## 1. 准备有标签的数据

输入为人工或仿真元数据标注的 JSONL，每行一个观测＋指令。以下仅为字段格式示例，路径需要替换为真实文件：

```json
{
  "sample_id": "scene01_red",
  "episode_id": "episode01",
  "scene_id": "scene01",
  "counterfactual_group_id": "color_pair01",
  "original_task_instruction": "pick up the cup",
  "observation_path": "observations/scene01_red.npz",
  "image_convention": "policy_input",
  "task_type": "visual_attribute",
  "question": "answer en What color is the cup?\n",
  "prompt_id": "cup_color_v1",
  "label": "red",
  "labels": ["red", "blue"],
  "label_source": "human_annotation",
  "object_text": "cup",
  "is_synthetic": false
}
```

实际 JSONL 中每个对象须占一行。代码不会自动编造标签。`original_task_instruction` 是 Source 的唯一任务文本；`question` 是完整的 Target 提示。禁止把被测颜色等答案写入原指令。`object_text` 必须在清洗后的原任务文本中唯一出现，用于 `text:object` 位置选择。

NPZ 通过 `allow_pickle=False` 读取，需要包含：

- `observation/image`、`observation/wrist_image`：uint8 HWC RGB。
- `observation/state`：真实原始 8D LIBERO state。

图像应当已处于**正式 policy 接口收到的方向**。原始 LIBERO render 通常需要上下、左右同时翻转一次；本读取器不会再次翻转。已有 `preparations/idea1_v2_1/runtime_models.py:native_observation` 可将原始 LIBERO 相机和机器人观测转成相同约定。不要直接把历史原始 render NPZ 当成 policy-input NPZ。

也可使用 `camera_paths: {"agent": "agent.png", "wrist": "wrist.png"}` 加 `state_path: "state.npy"` 或 `state: [...]`。路径相对于 JSONL 所在目录。仅支持两路有效 LIBERO 相机；第三路填充相机由官方适配器生成并 mask。传入不支持的相机 mask 会报错。

默认配置使用三类 `task_type`：`visual_attribute`（颜色）、`spatial_relation`、`instruction_conditioned`。左右任务必须提供 `coordinate_frame`（例如 `preprocessed_agent_view`），所有左右标签按这个定义标注。2×2 样本共享同一个 `counterfactual_group_id`；episode、scene、counterfactual 任意关联的样本都会并入同一独立组。`scene_id` 应指具体场景实例，不能把所有相同 benchmark 任务统称为一个 scene 后又期望按帧划分。

```bash
"$PROBE_PY" -m experiments.pi05_hidden_probe.build_probe_manifest \
  --input /absolute/path/labeled_samples.jsonl \
  --output experiments/pi05_hidden_probe/data/manifest.jsonl \
  --calibration-fraction 0.3 --seed 42
```

同时生成 `split_manifest.json`。已有明确的 calibration/test 划分会保留并检查。只有一个独立组时会报错，需要增加独立数据。分区内还应包含同提示、同标签集合、同坐标系、同布局但不同标签的 donor。`--check-only` 只检查元数据，不声称验证图像或标签内容。

## 2. 文本与工程冒烟检查

配置里的相对路径一律相对于 YAML 所在目录。先使用 `configs/smoke.yaml`，它只配置颜色任务的视觉块／文本 token 两条路径。

```bash
"$PROBE_PY" -m experiments.pi05_hidden_probe.audit_model \
  --config experiments/pi05_hidden_probe/configs/smoke.yaml --limit-groups 2

"$PROBE_PY" -m experiments.pi05_hidden_probe.test_text_decode \
  --config experiments/pi05_hidden_probe/configs/smoke.yaml --limit-groups 10

"$PROBE_PY" -m experiments.pi05_hidden_probe.test_patch_integrity \
  --config experiments/pi05_hidden_probe/configs/smoke.yaml --limit-groups 2
```

`--limit-groups` 保留完整独立组，因此不会截断成对／2×2 样本。文本冒烟结果同时保存真实图像、灰色控制图、黑色控制图输出，以及自由生成、EOS、候选似然。检查普通对象问答是否随图像改变；含 `X` 的提示只用于诊断读出接口，不能要求未注入时知道 X 指谁。

可给文本冒烟命令加 `--hf-reference /path/to/paligemma-3b-mix-224 --reference-device cpu`。它用官方 Transformers 路径读取 agent-view，并写入 `external_task_positive` 记录；这是任务参考，不是自定义 decoder 数值等价证明。使用远程 Hugging Face ID 时需要网络、相应访问权限及磁盘空间。

工程检查比较原 OpenPI prefix 与显式 block 路径、恒等自替换、Target 自身 layer-0 图像替换，包括 logits。结果写入 `patch_integrity.json`；主扫描也会执行该检查，失败即停止。容差在配置中预设，报告包含原路径重复运行误差；不要为了通过检查而任意放宽。

## 3. 校准并冻结 Target 配置

`calibration.yaml` 包含三类任务、两种读出路径。可先复制配置并缩小任务或候选层，再运行小规模验证。正式校准的候选层为 Source `{0,3,6,9,12,15,18}`，Target `{3,6,9,12}`。

```bash
"$PROBE_PY" -m experiments.pi05_hidden_probe.run_patch_probe \
  --config experiments/pi05_hidden_probe/configs/calibration.yaml --dry-run

"$PROBE_PY" -m experiments.pi05_hidden_probe.run_patch_probe \
  --config experiments/pi05_hidden_probe/configs/calibration.yaml

"$PROBE_PY" -m experiments.pi05_hidden_probe.run_patch_probe \
  --config experiments/pi05_hidden_probe/configs/calibration.yaml \
  --calibration-results results/pi05_hidden_probe/calibration/probe_results.jsonl \
  --freeze-config results/pi05_hidden_probe/frozen_probe_config.yaml
```

冻结时仅使用 calibration，检查运行清单及全部样本×层×对照组合已完成。每个 probe 按不同 Source 层和控制图的平均 BAcc 选择一个 q；并列时固定选择较浅 q。`selection_prediction: prediction` 默认使用生成结果；若事先决定用候选分数选 q，改为 `score_prediction`，结果仍按封闭候选任务报告。

冻结文件保存校准结果哈希、选择依据和完整科学配置摘要。评估阶段修改提示、目标层、答案解析规则等会导致摘要校验失败；需要回到校准重新生成。可修改 `manifest`、`output_dir`、`cache_dir` 的文件位置，但新 manifest 仍须满足相同数据与分组要求。`configs/evaluation.yaml` 是说明性占位文件，不是已选好层的实验配置。

## 4. 独立测试与分析

```bash
"$PROBE_PY" -m experiments.pi05_hidden_probe.run_patch_probe \
  --config results/pi05_hidden_probe/frozen_probe_config.yaml

"$PROBE_PY" -m experiments.pi05_hidden_probe.analyze_results \
  --input results/pi05_hidden_probe/evaluation/probe_results.jsonl \
  --output-dir results/pi05_hidden_probe/evaluation/analysis \
  --bootstrap-samples 1000 --seed 42
```

测试阶段扫描 Source `0…L`，每条路径固定校准选出的 Target 层。中断后给相同扫描命令加 `--resume`。续跑检查 checkpoint 文件签名、代码、配置、数据分区和实际观测内容；改变数据或配置后应使用新输出目录。每个进程使用独立输出目录；不支持多个进程同时写同一个 JSONL。

可提前运行 `extract_source --config ...` 填充缓存；扫描也会按需提取。缓存保存选定层与位置的 float32 NPZ（从 bfloat16 转存无精度损失），键包含 checkpoint／实现签名、原图／state／指令内容哈希、位置规则和层列表。正式动作接口不受影响。

## 对照、指标与实现约定

- `real`：当前样本 Source 表示；`shuffle`：不同标签匹配 donor；`random`：逐 token 范数匹配的随机方向。
- `nopatch`：固定图像＋问题；`self`：写回 Target 同层同位置表示。
- `direct`：Target 直接看真实图像；任务条件化路径的直接参考可显式提供原指令，并在结果中标记。patch 条件始终只看到固定图像、零 state 和探测问题。
- `layer0`：Target 自己对真实图片得到的全部有效视觉输入 token 注入控制图；这是工程阳性检查。
- Source 层 0 已含视觉编码器处理。最后一层 residual 与 final RMSNorm 输出分开。Target q 必须小于 L，使注入能经过后续 token 间计算。
- 回答固定使用贪心生成、完整重算、每步重新 patch；前缀双向、回答因果。候选似然只输入已生成的前序答案 token。
- `text:last` 包括可能的换行分隔 token；`text:last_task` 排除分隔符；`text:object` 是对象名最后一个子词。占位符必须实际为单 token，按 tokenizer offset 定位，遇歧义报错。
- 自由生成解析仅接受完整短标签、大小写和边缘标点；允许事先在 `aliases` 下按 task_type 配置同义词。长段描述中出现答案词不算回答正确。

分析输出 `layer_metrics.csv`、`summary.md`、逐层曲线、源层×目标层热图、随机生成案例。包含 BAcc、非法答案率、成对双正确率、2×2 四正确率、donor 跟随率，以及与无注入／shuffle 的配对差值和组级 bootstrap 区间。无完整标签覆盖或独立组不足时报告未定义，不制造置信区间。生成与候选分数各自统计；没有绘图库时会明确跳过画图。

可报告的结论限于指定冻结读出器下的可解码性。真实文本能力、跨层注入的有效性和语义准确率需要运行上述真实样本实验确定；单元测试通过不代表这些科学假设已经成立。
