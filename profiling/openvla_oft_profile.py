#!/usr/bin/env python3
"""Profile official OpenVLA-OFT L1 inference on a real LIBERO observation."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
POLICY_REPO = ROOT / "policies" / "openvla-oft"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(POLICY_REPO))

from profiling.common import (  # noqa: 
    StageAccounting,
    git_state,
    measure_call,
    nvtx_range,
    python_packages,E402
    system_manifest,
    timing_summary,
    wrap_method,
    write_json,
)


DEFAULT_CHECKPOINT = "moojink/openvla-7b-oft-finetuned-libero-spatial"
DEFAULT_OBSERVATION = (
    POLICY_REPO / "experiments" / "robot" / "libero" / "sample_libero_spatial_observation.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--observation",
        type=Path,
        default=None,
        help="Optional pickled observation. By default task 0 is reset in LIBERO-Spatial.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--initial-state", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    # 阶段边界同步：不加同步时 NVTX 区间度量的是 Python 下发时刻。实测 vision_encoder
    # 区间 59.6 ms 内只有 12.3 ms 是 GPU kernel（GPU 空转 80%），而 vlm_backbone 区间
    # 47.9 ms 里塞进了 47.9 ms kernel —— 两个组件的耗时被互相串味了。
    parser.add_argument(
        "--sync-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在每个阶段区间前后同步 CUDA，使区间反映真实 GPU 执行时间（默认开启）。",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profile_results" / "openvla-oft")
    return parser.parse_args()


class NvtxCallable:
    """在不改动官方源码的前提下，给一个可调用对象套上 NVTX 区间。

    `get_vla_action` 直接调用传进来的 HF processor；包一层转发对象即可把图像/文本
    预处理从 `policy_infer` 的未标记头部里摘出来。
    """

    def __init__(self, target: Any, range_name: str, accounting: StageAccounting):
        self._target = target
        self._range_name = range_name
        self._accounting = accounting

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with nvtx_range(self._range_name, accounting=self._accounting):
            return self._target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def install_ranges(
    vla: Any,
    processor: Any,
    action_head: Any,
    proprio_projector: Any,
    accounting: StageAccounting,
    synchronize: Callable[[], None] | None,
) -> Any:
    """挂上全部 NVTX 区间，返回替换了 `prepare_images_for_vla` 的原函数。"""
    import experiments.robot.openvla_utils as openvla_utils

    stage = {"accounting": accounting, "synchronize": synchronize}
    wrap_method(vla, "predict_action", "model_forward", **stage)
    wrap_method(vla.vision_backbone, "forward", "vision_encoder", **stage)
    wrap_method(vla.projector, "forward", "multimodal_projector", **stage)
    wrap_method(vla.language_model, "forward", "vlm_backbone", **stage)
    wrap_method(action_head, "predict_action", "action_head", **stage)
    wrap_method(proprio_projector, "forward", "proprio_projector", **stage)
    wrap_method(vla, "_unnormalize_actions", "action_postprocess", **stage)
    # `get_vla_action` 按模块全局名调用 prepare_images_for_vla，替换模块属性即可生效。
    original_prepare_images = openvla_utils.prepare_images_for_vla
    openvla_utils.prepare_images_for_vla = NvtxCallable(
        original_prepare_images, "input_preprocess", accounting
    )
    return original_prepare_images


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("--warmup must be >= 0 and --iterations must be >= 1")

    import torch
    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env
    from experiments.robot.libero.run_libero_eval import GenerateConfig, prepare_observation, process_action
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_processor,
        get_proprio_projector,
        get_vla,
        get_vla_action,
    )
    from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

    cfg = GenerateConfig(
        pretrained_checkpoint=args.checkpoint,
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        unnorm_key="libero_spatial_no_noops",
    )

    env = None
    task_description = None
    if args.observation is not None:
        with args.observation.open("rb") as stream:
            observation = pickle.load(stream)
        task_description = observation["task_description"]
    else:
        from libero.libero import benchmark

        task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
        task = task_suite.get_task(args.task_id)
        initial_states = task_suite.get_task_init_states(args.task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        env.seed(args.seed)
        env.reset()
        simulator_observation = env.set_init_state(initial_states[args.initial_state])
        for _ in range(cfg.num_steps_wait):
            simulator_observation, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))
        observation, _ = prepare_observation(simulator_observation, 224)
        observation["task_description"] = task_description

    vla = get_vla(cfg)
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

    accounting = StageAccounting()
    sync_device = torch.cuda.synchronize
    # processor 也要包一层，才能标出 get_vla_action 里 predict_action 之前的 CPU 前处理。
    processor = NvtxCallable(get_processor(cfg), "input_preprocess", accounting)
    install_ranges(
        vla,
        processor,
        action_head,
        proprio_projector,
        accounting,
        sync_device if args.sync_stages else None,
    )

    def infer() -> list[Any]:
        # get_vla_action mutates observation["state"], so each call receives its own shallow copy.
        current_observation = dict(observation)
        return get_vla_action(
            cfg,
            vla,
            processor,
            current_observation,
            current_observation["task_description"],
            action_head,
            proprio_projector,
        )

    for _ in range(args.warmup):
        infer()
        sync_device()

    # 只统计 capture window 内的调用，丢弃 warmup 的样本。
    accounting.reset()
    samples_ms: list[float] = []
    final_actions: list[Any] | None = None
    with nvtx_range("profile_window"):
        for _ in range(args.iterations):
            with nvtx_range("policy_infer", accounting=accounting):
                final_actions, elapsed_ms = measure_call(infer, sync_device)
            samples_ms.append(elapsed_ms)

    reconciliation = {
        "policy_infer": accounting.reconcile("policy_infer", ["input_preprocess", "model_forward"]),
        "model_forward": accounting.reconcile(
            "model_forward",
            [
                "vision_encoder",
                "multimodal_projector",
                "proprio_projector",
                "vlm_backbone",
                "action_head",
                "action_postprocess",
            ],
        ),
    }

    benchmark_step = None
    if env is not None:
        processed_action = process_action(final_actions[0], cfg.model_family)
        _, reward, done, _ = env.step(processed_action.tolist())
        benchmark_step = {"reward": float(reward), "done": bool(done)}
        env.close()

    resolved_output = args.output_dir.resolve()
    manifest = {
        "policy": "openvla-oft",
        "framework": {"torch": torch.__version__, "cuda": torch.version.cuda},
        "repository": git_state(POLICY_REPO),
        "checkpoint": {
            "id": args.checkpoint,
            "revision": getattr(vla.config, "_commit_hash", None),
        },
        "benchmark": "LIBERO-Spatial",
        "observation": {
            "source": str(args.observation.resolve()) if args.observation is not None else "live_libero_reset",
            "task_id": args.task_id if args.observation is None else None,
            "initial_state": args.initial_state if args.observation is None else None,
            "seed": args.seed if args.observation is None else None,
        },
        "input": {
            "batch_size": 1,
            "num_images": 2,
            "use_proprio": True,
            "center_crop": True,
            "dtype": "bfloat16",
            "task_description": task_description,
        },
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": timing_summary(samples_ms),
        "stage_accounting_ms": accounting.summary(),
        "nvtx_reconciliation": reconciliation,
        "stage_boundary_sync": args.sync_stages,
        "action_shape": [len(final_actions), len(final_actions[0])],
        "benchmark_step": benchmark_step,
        "environment": system_manifest(),
        "python_packages": python_packages(),
        "command": [sys.executable, *sys.argv],
        "relevant_environment_variables": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "LIBERO_CONFIG_PATH", "PYTHONPATH")
        },
        "nvtx_ranges": [
            "profile_window",
            "policy_infer",
            "input_preprocess",
            "model_forward",
            "vision_encoder",
            "multimodal_projector",
            "vlm_backbone",
            "proprio_projector",
            "action_head",
            "action_postprocess",
        ],
    }
    write_json(resolved_output / "manifest.json", manifest)
    write_json(
        resolved_output / "last_actions.json",
        json.loads(json.dumps(final_actions, default=lambda value: value.tolist())),
    )
    print(json.dumps(manifest["timing"], indent=2))
    print(json.dumps({"nvtx_reconciliation": reconciliation}, indent=2))


if __name__ == "__main__":
    main()
