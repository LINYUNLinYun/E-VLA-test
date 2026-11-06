#!/usr/bin/env python3
"""Profile the official image Push-T CNN/UNet Diffusion Policy checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
POLICY_REPO = ROOT / "policies" / "diffusion_policy"
DEFAULT_CHECKPOINT = (
    ROOT
    / "shared"
    / "checkpoints"
    / "diffusion-policy"
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
    wrap_method,
    write_json,
)


CHECKPOINT_URL = (
    "https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/"
    "diffusion_policy_cnn/train_0/checkpoints/epoch%3D0500-test_mean_score%3D0.884.ckpt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cuda:0")
    # 逐去噪 step 的标记是默认行为：denoising_denoising 整块会掩盖每一步的开销，
    # 而步数由 checkpoint 的 num_inference_steps 决定（该 checkpoint 为 100 步）。
    parser.add_argument(
        "--mark-denoising-steps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="为每个去噪 step 单独打 NVTX 标记（默认开启）。",
    )
    # 阶段边界同步：PyTorch 异步下发，不同步时 NVTX 区间度量的是下发时刻而非
    # GPU 执行时刻，子区间的总和无法与父区间对上。
    parser.add_argument(
        "--sync-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在每个阶段区间前后同步 CUDA，使区间反映真实 GPU 执行时间（默认开启）。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "profile_results" / "diffusion-policy"
    )
    return parser.parse_args()


def install_ranges(
    policy: Any,
    mark_denoising_steps: bool,
    accounting: StageAccounting,
    synchronize: Callable[[], None],
) -> None:
    from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer

    stage = {"accounting": accounting, "synchronize": synchronize}
    wrap_method(policy, "predict_action", "model_forward", **stage)
    wrap_method(policy.normalizer, "normalize", "observation_preprocess", **stage)
    wrap_method(policy.obs_encoder, "forward", "vision_encoder", **stage)
    wrap_method(policy, "conditional_sample", "diffusion_denoising", **stage)
    # LinearNormalizer.__getitem__ constructs a fresh wrapper on every access, so
    # decorating one returned instance would never observe predict_action's call.
    wrap_method(SingleFieldLinearNormalizer, "unnormalize", "action_postprocess", **stage)
    if mark_denoising_steps:
        # conditional_sample 每个 step 调用一次 self.model(...)，因此这一层标记
        # 会随去噪步数重复出现（该 checkpoint 为 100 次/推理）。
        wrap_method(policy.model, "forward", "denoising_step", **stage)
        wrap_method(policy.noise_scheduler, "step", "scheduler_step", **stage)


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be >= 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\nDownload the official seed-42 checkpoint from:\n{CHECKPOINT_URL}"
        )

    import dill
    import hydra
    import numpy as np
    import torch

    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
    from diffusion_policy.workspace.base_workspace import BaseWorkspace

    payload = torch.load(args.checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(args.output_dir.resolve()))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if type(policy).__name__ != "DiffusionUnetHybridImagePolicy":
        raise TypeError(f"Unexpected policy type: {type(policy).__name__}")

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    accounting = StageAccounting()
    sync_device = lambda: torch.cuda.synchronize(device)
    install_ranges(
        policy,
        args.mark_denoising_steps,
        accounting,
        sync_device if args.sync_stages else None,
    )

    # Reset the real benchmark environment outside the capture window. MultiStepWrapper
    # repeats the first observation at reset, so two identical frames match its behavior.
    env = PushTImageEnv(render_size=96)
    env.seed(args.seed)
    first_observation = env.reset()
    numpy_observation = {
        key: np.stack([value, value], axis=0)[None, ...].astype(np.float32)
        for key, value in first_observation.items()
    }

    def infer() -> dict[str, Any]:
        tensor_observation = dict_apply(
            numpy_observation,
            lambda value: torch.from_numpy(value).to(device=device),
        )
        with torch.no_grad():
            result = policy.predict_action(tensor_observation)
        return dict_apply(result, lambda value: value.detach().cpu().numpy())

    for _ in range(args.warmup):
        infer()
        sync_device()

    # 只统计 capture window 内的调用，丢弃 warmup 的样本。
    accounting.reset()
    samples_ms: list[float] = []
    final_result: dict[str, Any] | None = None
    with nvtx_range("profile_window"):
        for _ in range(args.iterations):
            with nvtx_range("policy_infer", accounting=accounting):
                final_result, elapsed_ms = measure_call(infer, sync_device)
            samples_ms.append(elapsed_ms)

    assert final_result is not None
    action = final_result["action"][0, 0]
    _, reward, done, _ = env.step(action)
    env.close()

    checkpoint_sha256 = sha256_file(args.checkpoint)
    ranges = [
        "profile_window",
        "policy_infer",
        "model_forward",
        "observation_preprocess",
        "vision_encoder",
        "diffusion_denoising",
        "action_postprocess",
    ]
    if args.mark_denoising_steps:
        ranges.extend(["denoising_step", "scheduler_step"])

    step_samples = accounting.samples("denoising_step")
    expected_steps = int(policy.num_inference_steps) * args.iterations
    if args.mark_denoising_steps and len(step_samples) != expected_steps:
        raise AssertionError(
            f"expected {expected_steps} denoising_step samples "
            f"({policy.num_inference_steps} steps x {args.iterations} iterations), "
            f"got {len(step_samples)}"
        )
    denoising_children = ["denoising_step", "scheduler_step"] if args.mark_denoising_steps else []
    reconciliation = {
        "policy_infer": accounting.reconcile("policy_infer", ["model_forward"]),
        "model_forward": accounting.reconcile(
            "model_forward",
            ["observation_preprocess", "vision_encoder", "diffusion_denoising", "action_postprocess"],
        ),
    }
    if denoising_children:
        reconciliation["diffusion_denoising"] = accounting.reconcile(
            "diffusion_denoising", denoising_children
        )

    manifest = {
        "policy": "diffusion-policy",
        "framework": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "repository": git_state(POLICY_REPO),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "source": CHECKPOINT_URL,
            "sha256": checkpoint_sha256,
        },
        "benchmark": "Push-T image",
        "model_class": type(policy).__name__,
        "workspace_class": workspace_cls.__name__,
        "input": {
            "batch_size": 1,
            "image_shape": list(numpy_observation["image"].shape),
            "agent_pos_shape": list(numpy_observation["agent_pos"].shape),
            "n_obs_steps": int(policy.n_obs_steps),
            "horizon": int(policy.horizon),
            "n_action_steps": int(policy.n_action_steps),
            "num_inference_steps": int(policy.num_inference_steps),
            "seed": args.seed,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": timing_summary(samples_ms),
        "denoising": {
            "num_inference_steps": int(policy.num_inference_steps),
            "marked_per_step": args.mark_denoising_steps,
            "step_count_captured": len(step_samples),
            "step_timing": timing_summary(step_samples),
        },
        "stage_accounting_ms": accounting.summary(),
        "nvtx_reconciliation": reconciliation,
        "stage_boundary_sync": args.sync_stages,
        "action_shape": list(final_result["action"].shape),
        "benchmark_step": {"reward": float(reward), "done": bool(done)},
        "environment": system_manifest(),
        "python_packages": python_packages(),
        "command": [sys.executable, *sys.argv],
        "relevant_environment_variables": {
            key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "PYTHONPATH")
        },
        "nvtx_ranges": ranges,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "last_action.json", final_result["action"].tolist())
    print(json.dumps({"timing": manifest["timing"], "denoising": manifest["denoising"]}, indent=2))
    print(json.dumps({"nvtx_reconciliation": reconciliation}, indent=2))


if __name__ == "__main__":
    main()
