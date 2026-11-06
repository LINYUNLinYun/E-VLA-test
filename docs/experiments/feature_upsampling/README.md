# SigLIP 特征上采样实验（training-free feature upsampling）

任务来源：`/root/nfs/embodied/benchmark/vla_training_free_feature_upsampling_experiment.md`
完整报告：[`reports/summary.md`](reports/summary.md)

## 目标

π0/π0.5 的 SigLIP So400m/14 在 224×224 下产生 256 个 token。尝试在 112×112（64 token）下运行编码器，
再用**纯训练无关**的方法把特征无损上采样回 256 token，从而降低 vision 编码耗时。

## 结论

- **严格无损不可能**：64 个 token 的任意线性组合秩 ≤ 64，288 张 LIBERO 图上最佳逐图线性重建
  rel_L2 下界为 0.4145（real-112）/ 0.4577（pooled-stem）。
- **可用的最佳方案**：112 输入 + 学习式 pos-emb 以 `nearest` 下采样 + `bilinear` 上采样。
  cos_mean 0.7954 / rel_L2 0.6680；vision 4.51 ms vs 7.52 ms（**1.67×**，RTX 4090，fp32，batch=1）。
- 自研 EPR（高分辨率前缀 + 64-token 尾部 + 高频残差回注）质量更高（cos 0.81–0.89），但 batch=1
  时没有实际加速，被 112 路线支配。
- **判定：暂不值得**为端到端加速采用；下一步应先做 action-level 成功率验证。

## 目录

```
docs/experiments/feature_upsampling/
├── README.md                 # 本文件
├── reports/summary.md        # 完整实验报告（方法、全部负结果、延迟、判定）
├── code/                     # 全部实验代码
│   ├── common.py             # 模型加载、数据采样、预处理、指标
│   ├── extract_features.py   # 提取教师/112/pooled-stem 特征 → data/
│   ├── upsamplers.py         # bilinear/bicubic/lanczos/JBU/guided/patch-guided
│   ├── eval_features.py      # 路线 A–D 全量评估 → results/feature_metrics.csv
│   ├── ablate_posemb.py      # pos-emb 插值方式消融
│   ├── eval_student112.py    # 112 输入基线
│   ├── eval_best112.py       # 最佳 112 路线上的上采样器对比
│   ├── epr.py                # EPR 方法实现
│   ├── eval_epr.py           # EPR 全量扫描 → results/epr_metrics.csv
│   ├── bench_latency.py      # 真实 GPU 延迟 → results/latency.csv
│   ├── make_plots.py         # 可视化 → plots/
│   ├── diagnose*.py          # 诊断脚本（秩下界、残差回注、block 相关性等）
│   ├── configs/default.yaml
│   └── tests/                # 单元测试（13 passed）
├── data/                     # 特征缓存（features_libero.npz，可由代码重新生成）
├── results/                  # 指标 CSV
├── plots/                    # 可视化
└── logs/                     # 最终各次运行的日志
```

## 复现

```bash
PY=/root/nfs/embodied/benchmark/policies/openpi/.venv/bin/python
cd /root/nfs/embodied/benchmark/docs/experiments/feature_upsampling/code

CUDA_VISIBLE_DEVICES=2 $PY extract_features.py --tag libero --max-images 288
CUDA_VISIBLE_DEVICES=2 $PY eval_features.py --tag libero --out feature_metrics
CUDA_VISIBLE_DEVICES=1 $PY eval_epr.py --tag libero --batch 16 --out epr \
    --prefixes 2 5 9 13 17 --grids 8 12
CUDA_VISIBLE_DEVICES=2 $PY bench_latency.py
$PY -m pytest -q tests/
```

依赖：openpi venv（JAX 0.5.3 / flax 0.10.2 / torch 2.7.1 / CUDA 12.6），RTX 4090。
checkpoint 只读：`shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero/params`。

## 数据/产物说明

- `data/`、`results/`、`plots/` 的路径由 `code/common.py` 中的 `EXP_ROOT` 决定。
- 所有位置编码均从 checkpoint 的 `pos_embedding (1,256,1152)` 插值得到，未随机初始化。
- 早期一次诊断存在预处理 bug（float 图像送入 `resize_with_pad` 被 clip 成常数图），
  相关早期结论已作废；报告中的数字全部来自修正后的 288 图全量评估。
