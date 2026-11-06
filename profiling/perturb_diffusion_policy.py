#!/usr/bin/env python3
"""Diffusion Policy 的 batch / 去噪步数扰动实验。

目的：在拿不到 GPU 硬件计数器（本机 `ERR_NVGPUCTRPERM`）的前提下，用**扰动标定**
判断单次 forward 是 compute-bound 还是 memory-bound。依据：

  * compute-bound  → 每样本延迟随 batch 基本不变（吞吐饱和）
  * 权重带宽受限    → batch 增大时每样本延迟按 ~1/B 下降（权重只读一次，被摊薄）
  * 激活带宽受限    → 每样本延迟随 batch 基本不变（激活随 B 线性增长）

再配合 FLOPs 实测值算出的 achieved TFLOPS，与 RTX 4090 的 FP32 峰值比较，构成 roofline
判据。stePs 方向的线性拟合给出每步成本与固定成本（obs encoder + 前后处理）。

用法：
    policies/diffusion_policy/.venv/bin/python profiling/perturb_diffusion_policy.py
    ... --batches 1,2,4,8,16 --steps 10,25,50,100 --reps 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_REPO = ROOT / "policies" / "diffusion_policy"
DEFAULT_CHECKPOINT = (
    ROOT / "shared" / "checkpoints" / "diffusion-policy"
    / "pusht-image-cnn-seed42-epoch0500-score0.884.ckpt"
)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(POLICY_REPO))

from profiling.common import (  # noqa: E402
    StageAccounting,
    git_state,
    measure_call,
    nvtx_range,
    python_packages,
    sha256_file,
    system_manifest,
    timing_summary,
    write_json,
)

# RTX 4090 datasheet 峰值，用于 roofline 判据。
# FP32 (CUDA core): 16384 cores x 2 (FMA) x 2.52 GHz ≈ 82.6 TFLOPS
# 显存带宽: GDDR6X 21 Gbps x 384 bit / 8 = 1008 GB/s
PEAK_FP32_TFLOPS = 82.6
PEAK_BANDWIDTH_GBPS = 1008.0


class MacCounter:
    """用 forward hook 统计 Conv/Linear 的乘加次数（MACs）。

    只在独立的一遍里跑（不计时），避免 hook 开销污染延迟测量。
    """

    def __init__(self) -> None:
        self.macs = 0
        self._handles: list[Any] = []

    def attach(self, module: Any) -> None:
        import torch

        for child in module.modules():
            if isinstance(child, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                self._handles.append(child.register_forward_hook(self._conv))
            elif isinstance(child, torch.nn.Linear):
                self._handles.append(child.register_forward_hook(self._linear))

    def _conv(self, module: Any, inputs: Any, output: Any) -> None:
        volume = 1
        for k in module.kernel_size:
            volume *= k
        self.macs += output.numel() * (module.in_channels // module.groups) * volume

    def _linear(self, module: Any, inputs: Any, output: Any) -> None:
        self.macs += output.numel() * module.in_features

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def flops(self) -> int:
        return 2 * self.macs


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """最小二乘拟合 y = a + b*x，返回 (a, b, r2)。"""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batches", default="1,2,4,8,16")
    parser.add_argument("--steps", default="10,25,50,100")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--burn-in", type=float, default=5.0, help="计时前的预热秒数，用于升频")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="在计时区间外套 NVTX（profile_window / policy_infer），供 nsys 采集 kernel 级数据。"
        "该模式下必须只给一个 (batch, steps) 组合。",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="打开 torch.backends.cudnn.benchmark，让 cuDNN 对每个形状实测选算法而不走启发式。",
    )
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "profile_results" / "diffusion-policy" / "perturbation")
    args = parser.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    steps_grid = [int(x) for x in args.steps.split(",")]

    import numpy as np
    import torch
    import dill
    import hydra

    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    payload = torch.load(args.checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(args.output_dir.resolve()))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if type(policy).__name__ != "DiffusionUnetHybridImagePolicy":
        raise TypeError(f"Unexpected policy type: {type(policy).__name__}")

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()
    sync = lambda: torch.cuda.synchronize(device)  # noqa: E731

    # 真实 benchmark observation，batch 维留待扩展；simulator 不进测量。
    env = PushTImageEnv(render_size=96)
    env.seed(args.seed)
    first_observation = env.reset()
    base_observation = {
        key: np.stack([value, value], axis=0)[None, ...].astype(np.float32)
        for key, value in first_observation.items()
    }

    def make_observation(batch: int) -> dict[str, Any]:
        numpy_obs = {
            key: np.repeat(value, batch, axis=0) for key, value in base_observation.items()
        }
        return dict_apply(numpy_obs, lambda v: torch.from_numpy(v).to(device=device))

    def infer(observation: dict[str, Any]) -> dict[str, Any]:
        with torch.no_grad():
            result = policy.predict_action(observation)
        return dict_apply(result, lambda v: v.detach().cpu().numpy())

    parameter_count = sum(p.numel() for p in policy.parameters())
    parameter_bytes = parameter_count * 4  # FP32

    # burn-in：网格从低负载配置开始，GPU 若还在低频率会让前几个点偏高。
    # 先用最大 batch 连续推理若干秒，把 SM 频率升到稳态再开始计时。
    import time

    burn_observation = make_observation(max(batches))
    policy.num_inference_steps = 10
    burn_deadline = time.perf_counter() + args.burn_in
    while time.perf_counter() < burn_deadline:
        infer(burn_observation)
    sync()

    accounting = StageAccounting() if args.nvtx else None
    if accounting is not None and (len(batches) != 1 or len(steps_grid) != 1):
        raise SystemExit(
            "--nvtx 模式下 --batches 和 --steps 都只能给一个值："
            "nsys capture-range 一次只采集一个 profile_window。"
        )

    records: list[dict[str, Any]] = []
    flop_records: list[dict[str, Any]] = []
    for batch in batches:
        observation = make_observation(batch)
        for steps in steps_grid:
            policy.num_inference_steps = steps

            for _ in range(args.warmup):
                infer(observation)
            sync()

            def measure_once() -> float:
                if accounting is None:
                    _, elapsed = measure_call(lambda: infer(observation), sync)
                    return elapsed
                with nvtx_range("policy_infer", accounting=accounting):
                    _, elapsed = measure_call(lambda: infer(observation), sync)
                return elapsed

            torch.cuda.reset_peak_memory_stats(device)
            if accounting is None:
                samples = [measure_once() for _ in range(args.reps)]
            else:
                # nsys --capture-range 一次只认一个 profile_window，因此 --nvtx 模式下
                # 必须限制成单个 (batch, steps) 组合，warmup 和 FLOP 统计留在窗口外。
                with nvtx_range("profile_window"):
                    samples = [measure_once() for _ in range(args.reps)]
            peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2

            summary = timing_summary(samples)
            summary.update(
                {
                    "batch": batch,
                    "steps": steps,
                    "per_sample_ms": summary["mean_ms"] / batch,
                    "throughput_samples_per_s": batch / (summary["mean_ms"] / 1000),
                    "peak_memory_mib": peak_mib,
                }
            )
            records.append(summary)
            print(
                f"B={batch:<3} N={steps:<4} mean={summary['mean_ms']:>9.3f} ms  "
                f"per-sample={summary['per_sample_ms']:>8.3f} ms  "
                f"thrpt={summary['throughput_samples_per_s']:>6.2f} samp/s  "
                f"peak={peak_mib:>7.1f} MiB",
                flush=True,
            )

        # FLOPs 单独一遍（不计时）。只依赖 batch 的 obs encoder + 固定步数的 UNet，
        # 用 steps=1 分离出「单步 UNet + encoder」的量，再线性外推。
        policy.num_inference_steps = 1
        counter = MacCounter()
        counter.attach(policy)
        with torch.no_grad():
            infer(observation)
        counter.detach()
        flop_records.append(
            {"batch": batch, "flops_per_step_plus_encoder": counter.flops, "macs": counter.macs}
        )

    policy.num_inference_steps = steps_grid[-1]

    # 每个 batch 上对 steps 做线性拟合：T(N) = 固定成本 + 每步成本 * N
    fits: list[dict[str, Any]] = []
    for batch in batches:
        rows = [r for r in records if r["batch"] == batch]
        intercept, slope, r2 = linear_fit(
            [float(r["steps"]) for r in rows], [r["mean_ms"] for r in rows]
        )
        flops_row = next(f for f in flop_records if f["batch"] == batch)
        flops_per_step = flops_row["flops_per_step_plus_encoder"]
        # --nvtx 模式一次只跑一个 steps，退化为单点，斜率无定义。
        achieved = flops_per_step / (slope / 1000) / 1e12 if slope > 0 else None
        fits.append(
            {
                "batch": batch,
                "fixed_cost_ms": intercept,
                "per_step_ms": slope,
                "r2": r2,
                "flops_per_step": flops_per_step,
                "achieved_tflops_per_step": achieved,
                "fixed_cost_fraction": (
                    intercept / (intercept + slope * steps_grid[-1])
                    if slope > 0
                    else None
                ),
            }
        )

    manifest = {
        "purpose": "Diffusion Policy batch / denoising-step 扰动实验",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "repo_commit": git_state(POLICY_REPO),
        },
        "grid": {
            "batches": batches,
            "steps": steps_grid,
            "reps": args.reps,
            "warmup": args.warmup,
            "burn_in_s": args.burn_in,
            "cudnn_benchmark": args.cudnn_benchmark,
        },
        "model": {
            "class": type(policy).__name__,
            "parameters": parameter_count,
            "parameter_bytes_fp32": parameter_bytes,
            "horizon": policy.horizon,
            "n_obs_steps": policy.n_obs_steps,
            "n_action_steps": policy.n_action_steps,
            "action_dim": policy.action_dim,
            "obs_feature_dim": policy.obs_feature_dim,
        },
        "roofline_reference": {
            "peak_fp32_tflops": PEAK_FP32_TFLOPS,
            "peak_bandwidth_gbps": PEAK_BANDWIDTH_GBPS,
            "note": "RTX 4090 datasheet 峰值；本机无法用 ncu 采集实测计数器",
        },
        "records": records,
        "flops": flop_records,
        "fits": fits,
        "environment": system_manifest(),
        "packages": python_packages(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "perturbation.json", manifest)
    print(f"\nwrote: {args.output_dir / 'perturbation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
