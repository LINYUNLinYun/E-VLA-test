# SigLIP So400m/14 训练无关特征上采样：实验报告

日期：2026-10-17  ·  硬件：NVIDIA RTX 4090 24 GB（batch=1）  ·  代码：`docs/experiments/feature_upsampling/code/`

## 0. 结论（TL;DR）

1. **严格意义的“无损”训练无关上采样不存在。** 任何只由 64 个低分辨率 token 线性组合得到的 256-token 输出，其秩 ≤ 64；在 288 张 LIBERO 图上实测，real-112 的最佳逐图线性重建 rel_L2 下界为 **0.4145**（pooled-stem 为 0.4577）。这不是调参问题，是秩约束。
2. **最佳严格训练无关方案：112 输入 + 学习式 pos-emb 用 `nearest` 下采样 + bilinear 上采样。**
   cos_mean **0.7954**，cos_p10 0.5725，rel_L2 **0.6680**；vision 耗时 **4.51 ms vs 7.52 ms（1.67×，fp32，batch=1）**。
   关键细节：pos-emb 用 bicubic 只有 0.7635；**nearest 明显更好**（保留了学习到的位置编码范数分布）。
3. **新提出的 EPR 混合方法**（高分辨率前缀 + 64-token 池化尾部 + 高频残差回注）能把 cos 提到 **0.81–0.89**，但在 batch=1 的真实 GPU 上 **没有加速**（1.0–1.5×），因为 64-token transformer block 的单块延迟只比 256-token 便宜 ~1.65×，而前缀块是全额成本。**不推荐**作为加速方案。
4. **判定：暂不值得**为端到端加速采用这些方法。若目标只是“特征尽量接近”且下游能容忍 cos≈0.80 / rel_L2≈0.67，则 112+nearest+bilinear 是一个可运行的训练无关起点，但必须先做 action-level 验证（本次未做）。

---

## 1. 实验设置

| 项 | 值 |
|---|---|
| checkpoint | `shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero/params`（只读，未修改） |
| openpi commit | `215abfb217dbac7d5f1273282331b9b1866c0479`（工作区仅一处注释改动，不影响计算） |
| 模型 | π0 的 SigLIP So400m/14，27 blocks，width 1152，patch 14，224→16×16=256 tokens，head 1152→2048 |
| 硬件 | RTX 4090 24 GB；GPU 1–3 隔离使用；JAX 0.5.3 / flax 0.10.2 / torch 2.7.1 / CUDA 12.6 |
| dtype | 特征指标 fp32；延迟分别测 fp32 与 bf16（`dtype_mm`） |
| 数据 | LIBERO `agentview_rgb`，4 suites × 6 tasks × 2 demos × 6 frames = **288 张**（seed=0，确定性采样） |
| 预处理 | 官方 `resize_with_pad(224)` → `uint8`；112 视图由 224 视图 `resize_with_pad(112)` 得到；输入 `[-1,1]` |
| 位置编码 | 从 checkpoint 读取 `pos_embedding (1,256,1152)`，用 `jax.image.resize` 插值到 8×8，**绝不随机初始化** |
| 训练 | 无。没有新 adapter、没有蒸馏、没有更新任何预训练权重 |

复现命令见第 9 节。

---

## 2. 为什么“无损”不可能

设低分辨率特征 `L ∈ R^{64×1152}`，任何逐 token 线性上采样 `P = A·L`（A 与图像无关）满足 `rank(P) ≤ 64`，而教师 `T ∈ R^{256×1152}` 通常满秩。Eckart–Young 给出 rel_L2 的硬下界；逐图最优线性投影（允许 A 依赖图像）实测：

| 来源 | 逐图最优线性 rel_L2（288 图均值） |
|---|---|
| real112（真实 112 输入） | **0.4145** |
| pooled_stem（224 stem 2×2 池化） | **0.4577** |

即：即使允许每张图解一个最优线性映射，也只能到 rel_L2≈0.41/0.46，远非无损。非线性、逐图自适应的训练无关方法可以略微突破该界（见 EPR），但不可能归零。

---

## 3. 路线 A–D 全量结果（288 图，pre-projector 空间）

`oracle_pool` = 教师 256 token 做 2×2 平均后再上采样（**诊断上界，不是可部署方法**）；
`real112` = 真实 112 输入完整编码；`pooled_stem` = 224 stem 池化后跑 27 blocks（即 EPR 去掉前缀）。

| source | method | cos_mean | cos_median | cos_p10 | rel_L2 | norm_ratio | spatial_delta |
|---|---|---|---|---|---|---|---|
| oracle_pool | bicubic | **0.9180** | 0.9502 | 0.8172 | **0.4067** | 0.902 | 0.411 |
| oracle_pool | nearest | 0.9121 | 0.9476 | 0.7892 | 0.4194 | 0.912 | 0.402 |
| oracle_pool | bilinear | 0.9117 | 0.9451 | 0.7997 | 0.4255 | 0.873 | 0.302 |
| oracle_pool | guided_filter | 0.8341 | 0.8873 | 0.6035 | 0.5721 | 0.831 | 0.186 |
| oracle_pool | jbu5 | 0.8175 | 0.8664 | 0.5856 | 0.5959 | 0.823 | 0.200 |
| oracle_pool | jbu3 | 0.8019 | 0.8557 | 0.5497 | 0.6177 | 0.846 | 0.269 |
| oracle_pool | patch_guided | 0.7889 | 0.8523 | 0.5094 | 0.6366 | 0.875 | 0.381 |
| real112 | bilinear | 0.7635 | 0.7832 | 0.5482 | 0.7166 | 1.019 | 0.482 |
| real112 | lanczos | 0.7565 | 0.8096 | 0.4862 | 0.6979 | 0.918 | 0.453 |
| real112 | bicubic | 0.7465 | 0.7686 | 0.5062 | 0.7655 | 1.072 | 0.673 |
| real112 | guided_filter | 0.7433 | 0.7793 | 0.5123 | 0.7145 | 0.934 | 0.261 |
| real112 | jbu5 | 0.7341 | 0.7494 | 0.5041 | 0.7330 | 0.953 | 0.278 |
| real112 | nearest | 0.7314 | 0.7525 | 0.4639 | 0.8009 | 1.097 | 0.671 |
| real112 | patch_guided | 0.6782 | 0.6971 | 0.4022 | 0.8395 | 1.042 | 0.558 |
| pooled_stem | bilinear | 0.7336 | 0.7536 | 0.4820 | 0.7595 | 1.011 | 0.450 |

（`real112` 行使用 bicubic 插值的 pos-emb；修正后的 pos-emb 方法见第 4 节。）

要点：

- **Oracle vs Real 的差距很大**：把教师的 256 个 token 池化成 64 个再上采样还能到 cos 0.918；但真实 112 输入重新编码只有 0.76。说明 64-token 瓶颈本身损失可控，**信息损失主要来自“在 112 像素上重新看图”**。
- **JBU / guided filter / patch-guided 全部不如 bilinear/bicubic**。在冻结特征上，引导滤波类方法在默认参数下稳定退化；这与文献中“JBU≈bicubic、guided filter 在冻结探针下不如 bilinear”的报告一致。
- 在最佳 112 路线（nearest pos-emb）上做完整上采样器对比（288 图全量）：bilinear **0.7954 / rel_L2 0.6680** 仍是最好的，其后依次为 bicubic 0.7838、lanczos 0.7710、guided_filter 0.7703、nearest 0.7693、jbu5 0.7524、jbu3 0.7268、patch_guided 0.7031。推荐管线无需再改上采样器。
- `spatial_delta`（预测/教师的高频能量比）显示所有上采样器都严重低通：oracle bicubic 只有 0.41，bilinear 0.30。cos 不惩罚平滑，这是“看起来还行、细粒度操作可能受损”的风险来源。

---

## 4. 学习式位置编码的插值方式（重要消融）

112 输入需要把 `(1,16,16,1152)` 的 pos-emb 变成 `(1,8,8,1152)`。288 图全量结果：

| pos 插值 | cos_mean | cos_p10 | rel_L2 | pos8 范数均值 (pos16=21.46) |
|---|---|---|---|---|
| **nearest** | **0.7954** | **0.5725** | **0.6680** | 19.49 |
| bicubic | 0.7635 | 0.5482 | 0.7166 | 17.45 |
| lanczos3 | 0.7610 | 0.5517 | 0.7190 | 18.27 |
| linear | 0.7467 | 0.5204 | 0.7490 | 16.01 |

结论：**nearest 最好**，因为它保留了学习到的位置编码的高频结构与范数；bicubic/linear 平滑后范数收缩 19–25%，造成尺度分布漂移。32 图子集上额外测试了 `avg2`（0.7659/0.7026）与 `nearest_scaled`（0.7866/0.6799）——把 nearest 的范数强行缩放到 21.46 没有收益，说明起作用的是**保留高频模式**而不是单纯范数。

> 这条结论直接修正了文档中的默认假设（bicubic）。后续 EPR 不需要插值 pos-emb（hidden state 已含位置信息），但任何“112 直跑”的基线都应使用 nearest。

---

## 5. EPR：Early-exact Prefix + pooled tail + Residual reinjection（本次提出的方法）

流程（全部使用现有权重，无训练）：

1. 224 输入做 patch-embed（stem，精确）；
2. 前 `p` 个 transformer block 在 256 token 上精确执行，得到 `H_p`；
3. `H_p` 自适应平均池化到 `g×g`（g=8 → 64 token；g=12 → 144 token）；
4. 剩余 `27-p` 个 block 在 `g²` token 上执行，得到 `S`；
5. half-pixel bilinear 把 `S` 上采样回 16×16；
6. 回注高频细节：`P = up(S) + α·(H_p − up(pool(H_p)))`。

（g=16、α=0 时该链路与 224 教师**逐位相同**：`max|diff|=0.0`，作为实现正确性 sanity check。）

288 图结果（α 取 0/0.05/0.1 中最好者；g=8 用 avg 池化）：

| p (精确块数) | g | cos_mean | cos_p10 | rel_L2 | 实测 vision_total (fp32) | vision speedup |
|---|---|---|---|---|---|---|
| 2 | 8 | 0.7557 | 0.5250 | 0.7238 | 4.83 ms | 1.56× |
| 5 | 8 | 0.8129 | 0.6222 | 0.6252 | 5.16 ms | 1.46× |
| 9 | 8 | 0.8563 | 0.6989 | 0.5437 | 5.76 ms | 1.31× |
| 13 | 8 | 0.8725 | 0.7382 | 0.5184 | ~6.4 ms（按块数估计，未单测） | ~1.17× |
| 17 | 8 | 0.8849 | 0.7564 | 0.5028 | ~7.1 ms（未单测） | ~1.06× |
| 2 | 12 | 0.7870 | 0.5653 | 0.6850 | — | — |
| 5 | 12 | 0.8406 | 0.6611 | 0.5848 | 6.72 ms | 1.12× |
| 9 | 12 | 0.8765 | 0.7275 | 0.5113 | — | — |
| 13 | 12 | 0.8863 | 0.7567 | 0.4967 | — | — |
| 17 | 12 | **0.8936** | **0.7754** | **0.4914** | — | — |

(full data: `results/epr_metrics.csv`; 表中 p=13/17 的延迟按每块实测成本外推，报告不将其用于结论。)

**判定：EPR 被“112 + nearest + bilinear”支配。**
- p=2 g=8：4.83 ms、cos 0.7557 —— 比 112（4.51 ms、0.7954）**又慢又差**。
- p=5 g=8：5.16 ms、0.8129 —— 质量 +0.017，但慢 14%。
- 要高 cos（0.87–0.89）必须 p≥9，此时已经和 224 一样慢甚至更慢。

原因见第 6 节：batch=1 时 block 延迟由权重读取/kernel launch 主导，token 数从 256 降到 64 只节省约 1.65× 的单块时间，而前缀块是全额开销。

---

## 6. 真实 GPU 延迟（batch=1，CUDA 同步 wall-clock，warmup=30，iters=100）

完整数据：`results/latency.csv`。`vision_total_ms` 含 resize + GPU 转换 + encoder + upsampler + head，是**一体化链路独立测量**，不是分项相加。`end2end_ms` 未接 `sample_actions`，按文档要求留空（未填 0）。

| method | dtype | vision_total | median | p90 | speedup vs 224 |
|---|---|---|---|---|---|
| siglip224（教师） | fp32 | 7.52 ms | 7.52 | 8.09 | 1.00× |
| **siglip112 + bilinear** | fp32 | **4.51 ms** | 4.51 | 4.78 | **1.67×** |
| epr p=2 g=8 | fp32 | 4.83 ms | 4.83 | 5.03 | 1.56× |
| epr p=5 g=8 | fp32 | 5.16 ms | 5.16 | 5.58 | 1.46× |
| epr p=9 g=8 | fp32 | 5.76 ms | 5.76 | 7.65 | 1.31× |
| epr p=5 g=12 | fp32 | 6.72 ms | 6.72 | 6.91 | 1.12× |
| siglip224（教师） | bf16 | 4.91 ms | 4.91 | 5.24 | 1.00× |
| **siglip112 + bilinear** | bf16 | **3.57 ms** | 3.57 | 3.88 | **1.37×** |
| epr p=2 g=8 | bf16 | 3.72 ms | 3.72 | 3.94 | 1.32× |
| epr p=5 g=8 | bf16 | 3.80 ms | 3.80 | 3.90 | 1.29× |

分项（fp32, p=2 g=8）：stem 0.10 / prefix(含stem) 0.76 / tail(25 blocks@64) **4.12** / upsample 0.08 / head 0.11 ms。

关键洞察：64-token block ≈ 0.165 ms，256-token block ≈ 0.27 ms，单块只快 **1.65×**（理论上 attention 应快 16×、总 FLOPs 快 ~4×）。**batch=1 下瓶颈是每块固定的权重读取与 kernel launch，不是 token 数。** 因此：
- 只降 token 不给大加速；
- 减少 block 数才能加速，但训练无关条件下不能跳 block；
- 端到端（VLM prefix + action expert 仍是 256 token）的收益会进一步被稀释，预计远低于 10%（本次未测，不作断言）。

bf16 下 112 的加速比反而降到 1.37×（权重读取减半后，固定开销占比更高），说明精度选择也要实测。

---

## 7. 可视化

`plots/cases_real112_bilinear.png`：按 real112-bilinear cos 排序取 **worst / median / best** 三张（非 cherry-pick），每行展示 输入 RGB / 教师特征 PCA / 112-bilinear 上采样 PCA / L2 误差图。

- 全局布局与物体位置保持良好；误差集中在机械臂边缘、盒子/盘子轮廓等高频区域。
- 最差案例（cos 0.704）误差几乎铺满全图；最好案例（0.824）仍有明显的边缘模糊。
- PCA 是每行共享基，颜色可直接比较。

---

## 8. 失败路线与负面结果

| 路线 | 结果 | 证据 |
|---|---|---|
| A: nearest/bilinear/bicubic/lanczos | 可用但非无损；bicubic 在 oracle pool 上最好（0.918），在 real112 上 bilinear/lanczos 最好（~0.76） | §3 |
| B: JBU（3×3/5×5/spatial-only） | **全部差于 bilinear**；oracle 0.80–0.82 vs 0.912；real112 0.707–0.734 vs 0.764 | §3 |
| C: guided filter | oracle 0.834 / real112 0.743，均差于 bilinear；且高频能量比只有 0.19–0.26 | §3 |
| D: patch-embedding-guided | 最差（oracle 0.789 / real112 0.678），引导权重没有带来空间选择性收益 | §3 |
| E: stem 高频残差直接回注 | 无增益。raw stem+pos 与最终特征的 cosine = **−0.0195**（几乎正交）；real112 bilinear 0.7635 → +0.05α 0.7643 → +0.25α 0.7610 → +1.0α 0.7193（残差按 RMS 对齐后同样无增益） | 288 图全量（输入取自 npz，无模型前向，结论有效） |
| F: 文献方法（FeatUp/AnyUp/NAF/Upsample Anything） | 均需训练或逐图优化，不满足“严格训练无关 + 实时”；未部署 | 调研结论 |
| EPR suffix（上采样后再跑精确 block） | 上采样特征偏离真实 hidden state 太远，精确 block 会放大误差；该实验早期版本有预处理 bug，**不作为有效结论**，仅记录不再采用 | — |

**关于一个已修正的实验 bug**：早期诊断脚本把 uint8 图像先转 float32 再送 `resize_with_pad`，后者按 float 分支把像素 clip 到 `[-1,1]`，导致模型实际看到的是常数图像（所有值 -0.992），早期“cos 0.89”的 EPR 结论无效。本报告所有数字均来自修正后的 288 图全量评估（`eval_features.py` / `eval_epr.py` / `eval_student112.py`，均直接读 `rgb224_u8` 并正确归一化）。

---

## 9. 结论与建议

**判定：暂不值得（for end-to-end speedup）。**

依据：

1. 无损不可能（秩下界 + 实测线性下界 0.41）。
2. 最好的训练无关 112 方案有 1.67× vision-only 加速，但 cos 0.795 / rel_L2 0.668，高频能量只有教师的 48%；对精细操作的风险未知。
3. batch=1 下 token 缩减的延迟收益远低于 FLOPs 预期（单块 1.65×，不是 4×），EPR 的任何高保真配置都不比 224 快。
4. 即使 vision 快 1.67×，VLM prefix 与 action expert 的 256-token 计算不变，端到端提升大概率 <10%，需要实测 `sample_actions` 才能定论。

**如果继续，建议的下一步（按优先级）：**

1. **Action-level 验证**（唯一能改变结论的实验）：固定 LIBERO 初始状态，对比 224 vs 112+nearest+bilinear 的 `action_mae` / `action_rmse` 与成功率；若成功率掉 <2%，则 1.67× 的 vision 加速可能值得。
2. 测 `sample_actions` 的端到端耗时，拆分 vision / VLM prefix / AE denoise，确认是否真的 ≥10%。
3. 若端到端瓶颈确实在 vision，考虑 bf16 + batch>1（batch 增大后 token 数重新成为主导，EPR 的理论优势才会显现）。
4. 若必须“近似无损”，只能放开约束：训练轻量 adapter / 蒸馏（文档明确禁止，本工作未做）。

---

## 10. 复现

```bash
PY=/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python
cd /root/nfs/embodied/benchmark/docs/experiments/feature_upsampling/code

# 1) 提取 288 张图的教师/112/pooled-stem 特征（~2 min）
CUDA_VISIBLE_DEVICES=2 $PY extract_features.py --tag libero --max-images 288

# 2) 路线 A–D 全量指标（~7 min）
CUDA_VISIBLE_DEVICES=2 $PY eval_features.py --tag libero --out feature_metrics

# 3) pos-emb 插值消融 + 最佳 112 基线（~2 min each）
CUDA_VISIBLE_DEVICES=3 $PY eval_student112.py
CUDA_VISIBLE_DEVICES=3 $PY eval_best112.py

# 4) EPR 全量扫描（~10 min）
CUDA_VISIBLE_DEVICES=1 $PY eval_epr.py --tag libero --batch 16 --out epr \
    --prefixes 2 5 9 13 17 --grids 8 12

# 5) 真实 GPU 延迟（~15 min）
CUDA_VISIBLE_DEVICES=2 $PY bench_latency.py

# 6) 可视化 + 单元测试
$PY make_plots.py
$PY -m pytest -q tests/
```

产物（路径相对于 `docs/experiments/feature_upsampling/`）：

- `results/feature_metrics.csv` / `_summary.csv` —— 每图×方法×source，列含 `image_id,camera,method,source,pos_emb_method,cos_mean,cos_p10,rel_l2,norm_ratio,spatial_delta,valid`
- `results/epr_metrics.csv` —— EPR 每图指标
- `results/latency.csv` —— 延迟（`end2end_ms` 留空）
- `plots/cases_real112_bilinear.png`
- `data/features_libero.npz` + `data/meta_libero.json`（形状/dtype/记录/耗时；可由上面第 1 步重新生成）
- 单元测试：`code/tests/test_upsamplers.py`（7 passed）、`code/tests/test_epr_ops.py`（6 passed：常量保持、对齐、与 torch 数值一致、池化权重归一）
