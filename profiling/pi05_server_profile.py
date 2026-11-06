#!/usr/bin/env python3
"""Run the official pi0.5 LIBERO JAX policy server with bounded profiling."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
POLICY_REPO = ROOT / "policies" / "openpi"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(POLICY_REPO / "src"))

from profiling.common import (  # noqa: E402
    StageAccounting,
    git_state,
    nvtx_range,
    python_packages,
    system_manifest,
    timing_summary,
    write_json,
)


DEFAULT_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default="pi05_libero")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    # 请求生命周期：reference_warmup -> reference -> warmup -> iterations。
    # 前两段跑官方单体 jit，用于给出可比对的端到端基线；后两段跑组件拆解路径。
    parser.add_argument("--reference-warmup", type=int, default=2)
    parser.add_argument("--reference-iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--exit-after-profile", action="store_true")
    # 组件拆解：把官方单体 jit 重新编排成逐组件单独 jit，使 visual encoder /
    # vlm backbone / action expert 在 nsys 上各自可见。
    parser.add_argument(
        "--component-breakdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="拆解 pi0.5 内部组件并逐组件打 NVTX 标记（默认开启）。",
    )
    # 阶段边界同步：JAX 同样是异步下发，不同步时组件区间度量的是下发时刻。
    parser.add_argument(
        "--sync-stages",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="在每个组件区间前后等待设备完成，使区间反映真实执行时间（默认开启）。",
    )
    parser.add_argument(
        "--check-equivalence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="校验拆解路径与官方单体路径的数值一致性（默认开启）。",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "profile_results" / "pi05")
    return parser.parse_args()


def total_requests(args: argparse.Namespace) -> int:
    """客户端需要发送的请求总数，供 launcher 计算。"""
    return args.reference_warmup + args.reference_iterations + args.warmup + args.iterations


class ComponentBreakdown:
    """把官方 `Pi0.sample_actions` 重新编排成逐组件单独 jit 的版本。

    官方 `Policy` 用 `nnx_utils.module_jit(model.sample_actions)` 把整次推理编成**一个**
    XLA 模块。Python 层的 NVTX 标记在 jit 内部只会在 trace 期执行一次，不会在运行时
    触发，因此 nsys 上只能看到 `model_forward` 一整块，看不到模型内部组件。

    这里把同一份官方计算拆成若干各自 jit 的组件，每个组件外面套 NVTX，得到：

        model_forward
        ├── model_preprocess   (_model.preprocess_observation)
        ├── vision_encoder     (PaliGemma.img，SigLIP So400m/14)
        ├── vlm_backbone       (文本嵌入 + 前缀组装 + PaliGemma.llm 前缀前向 -> KV cache)
        └── flow_matching      (流匹配去噪循环)
            └── denoising_step × num_steps
                └── action_expert  (embed_suffix + gemma_300m PaliGemma.llm 后缀前向
                                    + action_out_proj + Euler 更新)

    代价：失去跨组件的 XLA 融合、每次组件调用多一层 dispatch。`monolithic_reference`
    记录官方单体路径的延迟，用于量化这份开销；`decomposition_equivalence` 校验拆解
    路径与官方路径数值一致。

    各组件内部的计算体逐行对照 `policies/openpi/src/openpi/models/pi0.py` 的
    `embed_prefix` / `embed_suffix` / `sample_actions`，只重组调用边界，不改数学。
    """

    def __init__(
        self,
        model: Any,
        default_num_steps: int,
        accounting: StageAccounting,
        block_output: bool,
    ):
        self._model = model
        self._default_num_steps = default_num_steps
        self._accounting = accounting
        self._block_output = block_output
        self._jits: dict[str, Callable[..., Any]] = {}
        self._image_keys: tuple[str, ...] = ()
        self._batch_size = 1
        self._built = False

    # -- 构建 ---------------------------------------------------------------

    def build(self, observation: Any) -> None:
        import einops
        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        model = self._model
        block_output = self._block_output
        self._image_keys = tuple(observation.images)
        self._batch_size = int(observation.state.shape[0])
        horizon = int(model.action_horizon)
        image_keys = self._image_keys
        batch_size = self._batch_size

        # 官方 Policy 在 __init__ 里已经对同一个 model 做过 nnx.split；这里再拆一次
        # 拿到同一份参数（State 里是同一个数组对象，不复制设备内存），再把每个组件
        # 单独 jit 成一个 XLA 模块。
        graphdef, state = nnx.split(model)

        def component(call: Callable[..., Any]) -> Callable[..., Any]:
            def body(state: Any, *args: Any, **kwargs: Any) -> Any:
                return call(nnx.merge(graphdef, state), *args, **kwargs)

            jitted = jax.jit(body)
            # 必须在这里阻塞，否则拿不到真实的设备边界：`jax.effects_barrier()` 只建立
            # 执行顺序、不会阻塞 host（实测 0.13 ms，而 block_until_ready 3.51 ms），
            # 用它做阶段同步等于没同步，区间量的仍是 Python 下发时刻。block_until_ready
            # 等到本组件输出就绪，而同一条流上的前序计算必然已完成，因此区间两端都是
            # 真实设备时间。
            if block_output:
                return lambda *args, **kwargs: jax.block_until_ready(
                    jitted(state, *args, **kwargs)
                )
            return lambda *args, **kwargs: jitted(state, *args, **kwargs)

        def model_preprocess(m: Any, obs: Any) -> Any:
            return _model.preprocess_observation(None, obs, train=False)

        def vision_encoder(m: Any, obs: Any) -> Any:
            return tuple(m.PaliGemma.img(obs.images[key], train=False)[0] for key in image_keys)

        def vlm_backbone(m: Any, obs: Any, image_tokens: Any) -> Any:
            # 对应 Pi0.embed_prefix + sample_actions 里的 KV cache 填充
            input_mask, ar_mask, tokens = [], [], []
            for key, image_token in zip(image_keys, image_tokens):
                tokens.append(image_token)
                input_mask.append(
                    einops.repeat(obs.image_masks[key], "b -> b s", s=image_token.shape[1])
                )
                ar_mask += [False] * image_token.shape[1]
            if obs.tokenized_prompt is not None:
                tokenized = m.PaliGemma.llm(obs.tokenized_prompt, method="embed")
                tokens.append(tokenized)
                input_mask.append(obs.tokenized_prompt_mask)
                ar_mask += [False] * tokenized.shape[1]
            prefix_tokens = jnp.concatenate(tokens, axis=1)
            prefix_mask = jnp.concatenate(input_mask, axis=1)
            prefix_ar_mask = jnp.array(ar_mask)
            attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = m.PaliGemma.llm(
                [prefix_tokens, None], mask=attn_mask, positions=positions
            )
            return kv_cache, prefix_mask

        def action_expert(
            m: Any, obs: Any, x_t: Any, time_value: Any, kv_cache: Any, prefix_mask: Any, dt: Any
        ) -> Any:
            # 对应 Pi0.sample_actions.step
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = m.embed_suffix(
                obs, x_t, jnp.broadcast_to(time_value, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = m.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = m.action_out_proj(suffix_out[:, -horizon:])
            return x_t + dt * v_t

        self._jits = {
            "model_preprocess": component(model_preprocess),
            "vision_encoder": component(vision_encoder),
            "vlm_backbone": component(vlm_backbone),
            "action_expert": component(action_expert),
        }
        self._built = True

    # -- 执行 ---------------------------------------------------------------

    @contextlib.contextmanager
    def _stage(self, name: str) -> Iterator[None]:
        # 不做 synchronize：阶段的设备边界由 `component()` 里的 block_until_ready 保证，
        # 内层组件已经阻塞过，外层区间自然落在真实设备时间上。
        with nvtx_range(name, accounting=self._accounting):
            yield

    def sample_actions(
        self, rng: Any, observation: Any, *, num_steps: int | None = None, noise: Any = None
    ) -> Any:
        """与 `Pi0.sample_actions` 签名兼容的拆解版本。"""
        import jax
        import jax.numpy as jnp

        steps = int(self._default_num_steps if num_steps is None else num_steps)
        dt = -1.0 / steps

        with self._stage("model_preprocess"):
            observation = self._jits["model_preprocess"](observation)

        with self._stage("vision_encoder"):
            image_tokens = self._jits["vision_encoder"](observation)

        with self._stage("vlm_backbone"):
            kv_cache, prefix_mask = self._jits["vlm_backbone"](observation, image_tokens)

        if noise is None:
            noise = jax.random.normal(
                rng, (self._batch_size, int(self._model.action_horizon), int(self._model.action_dim))
            )
        x_t = jnp.asarray(noise)
        time_value: Any = 1.0
        # 官方用 jax.lax.while_loop(cond, step, (noise, 1.0))，从 time=1.0 起按
        # dt=-1/steps 递减，条件 time >= -dt/2，正好执行 steps 次。这里换成等价的
        # Python 循环，才能在每一步外面套上运行时可见的 NVTX 区间。
        with self._stage("flow_matching"):
            for _ in range(steps):
                with self._stage("denoising_step"):
                    with self._stage("action_expert"):
                        x_t = self._jits["action_expert"](
                            observation, x_t, time_value, kv_cache, prefix_mask, dt
                        )
                time_value = time_value + dt
        return x_t

    # -- 数值一致性校验 -----------------------------------------------------

    def check_equivalence(self, monolithic: Callable[..., Any], observation: Any) -> dict[str, Any]:
        """同一 rng 与同一 noise 下，比较拆解路径与官方单体路径的输出。"""
        import jax
        import jax.numpy as jnp

        rng = jax.random.key(0)
        noise = jax.random.normal(
            rng, (self._batch_size, int(self._model.action_horizon), int(self._model.action_dim))
        )
        expected = jax.block_until_ready(
            monolithic(rng, observation, num_steps=self._default_num_steps, noise=noise)
        )
        actual = jax.block_until_ready(
            self.sample_actions(rng, observation, num_steps=self._default_num_steps, noise=noise)
        )
        difference = jnp.abs(jnp.asarray(actual) - jnp.asarray(expected))
        return {
            "max_abs_diff": float(difference.max()),
            "mean_abs_diff": float(difference.mean()),
            "allclose_atol_1e-2": bool(jnp.allclose(actual, expected, atol=1e-2, rtol=1e-2)),
        }


class ProfilingPolicy:
    """Wrap a Policy without changing OpenPI's official model or server sources."""

    def __init__(self, policy: Any, args: argparse.Namespace):
        import jax

        self._policy = policy
        self._args = args
        self._request_count = 0
        self._samples_ms: list[float] = []
        self._monolithic_samples_ms: list[float] = []
        self._window = None
        self._jax = jax
        self._accounting = StageAccounting()
        self._equivalence: dict[str, Any] | None = None
        self._breakdown: ComponentBreakdown | None = None

        self._monolithic_sample_actions = policy._sample_actions
        sample_kwargs = getattr(policy, "_sample_kwargs", {}) or {}
        self._default_num_steps = int(sample_kwargs.get("num_steps", 10))

        original_input_transform = policy._input_transform
        original_output_transform = policy._output_transform

        def input_transform(value: Any) -> Any:
            with nvtx_range("observation_preprocess", accounting=self._accounting):
                return original_input_transform(value)

        def output_transform(value: Any) -> Any:
            with nvtx_range("action_postprocess", accounting=self._accounting):
                return original_output_transform(value)

        policy._input_transform = input_transform
        policy._output_transform = output_transform

        if args.component_breakdown:
            policy._sample_actions = self._sample_actions_breakdown
        else:
            policy._sample_actions = self._sample_actions_monolithic

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    # -- 采样路径 -----------------------------------------------------------

    def _ensure_breakdown(self, observation: Any) -> None:
        if self._breakdown is not None:
            return
        # 阶段边界的同步由 `component()` 里的 `jax.block_until_ready` 完成。
        # 不能用 `jax.effects_barrier()`：它不阻塞 host，拿它当同步是无效的。
        self._breakdown = ComponentBreakdown(
            self._policy._model,
            self._default_num_steps,
            self._accounting,
            block_output=self._args.sync_stages,
        )
        self._breakdown.build(observation)

    def _sample_actions_monolithic(self, rng: Any, observation: Any, **kwargs: Any) -> Any:
        with nvtx_range("model_forward", accounting=self._accounting):
            output = self._monolithic_sample_actions(rng, observation, **kwargs)
            # JAX dispatch is asynchronous. Keep the synchronization inside
            # model_forward so both NVTX and wall-clock boundaries are honest.
            return self._jax.block_until_ready(output)

    def _sample_actions_breakdown(self, rng: Any, observation: Any, **kwargs: Any) -> Any:
        self._ensure_breakdown(observation)
        if self._args.check_equivalence:
            # 校验跑在 capture window 之前，不会进入 nsys 报告。
            self._check_equivalence_once(observation)
        with nvtx_range("model_forward", accounting=self._accounting):
            output = self._breakdown.sample_actions(rng, observation, **kwargs)
            return self._jax.block_until_ready(output)

    # -- 请求生命周期 -------------------------------------------------------

    def _phase(self, index: int) -> str:
        reference_warmup = self._args.reference_warmup
        reference_end = reference_warmup + self._args.reference_iterations
        capture_start = reference_end + self._args.warmup
        if index < reference_warmup:
            return "reference_warmup"
        if index < reference_end:
            return "reference"
        if index < capture_start:
            return "breakdown_warmup"
        return "capture"

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        index = self._request_count
        self._request_count += 1
        phase = self._phase(index)

        if phase == "reference_warmup":
            result = self._policy.infer(observation)
            self._jax.block_until_ready(result)
        elif phase == "reference":
            started_ns = time.perf_counter_ns()
            result = self._policy.infer(observation)
            result = self._jax.block_until_ready(result)
            self._monolithic_samples_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)
        elif phase == "breakdown_warmup":
            result = self._policy.infer(observation)
            self._jax.block_until_ready(result)
        else:
            result = self._captured_infer(observation)

        return result

    def _captured_infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        if not self._samples_ms:
            self._accounting.reset()
            self._window = nvtx_range("profile_window")
            self._window.__enter__()

        started_ns = time.perf_counter_ns()
        with nvtx_range("policy_infer", accounting=self._accounting):
            result = self._policy.infer(observation)
            result = self._jax.block_until_ready(result)
        self._samples_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)

        if len(self._samples_ms) == self._args.iterations:
            assert self._window is not None
            self._window.__exit__(None, None, None)
            self._window = None
            self._write_manifest(result)
            if self._args.exit_after_profile:
                # Give the websocket handler time to send the final response.
                threading.Timer(2.0, lambda: os._exit(0)).start()
        return result

    def _check_equivalence_once(self, observation: Any) -> None:
        """用第一次收到的真实 observation 比较拆解路径与单体路径。"""
        if self._equivalence is not None or self._breakdown is None:
            return
        self._equivalence = self._breakdown.check_equivalence(
            self._monolithic_sample_actions, observation
        )
        print(json.dumps({"decomposition_equivalence": self._equivalence}, indent=2), flush=True)

    # -- manifest -----------------------------------------------------------

    def _write_manifest(self, result: dict[str, Any]) -> None:
        import jax

        children = (
            [
                "model_preprocess",
                "vision_encoder",
                "vlm_backbone",
                "flow_matching",
            ]
            if self._args.component_breakdown
            else []
        )
        reconciliation: dict[str, Any] = {
            "policy_infer": self._accounting.reconcile(
                "policy_infer", ["observation_preprocess", "model_forward", "action_postprocess"]
            ),
            "model_forward": self._accounting.reconcile("model_forward", children),
        }
        if self._args.component_breakdown:
            reconciliation["flow_matching"] = self._accounting.reconcile(
                "flow_matching", ["denoising_step"]
            )
            reconciliation["denoising_step"] = self._accounting.reconcile(
                "denoising_step", ["action_expert"]
            )

        ranges = ["profile_window", "policy_infer", "observation_preprocess", "model_forward"]
        if self._args.component_breakdown:
            ranges += [
                "model_preprocess",
                "vision_encoder",
                "vlm_backbone",
                "flow_matching",
                "denoising_step",
                "action_expert",
            ]
        ranges.append("action_postprocess")

        manifest = {
            "policy": "pi05",
            "framework": {"jax": jax.__version__, "backend": jax.default_backend()},
            "repository": git_state(POLICY_REPO),
            "checkpoint": self._args.checkpoint,
            "config": self._args.config,
            "benchmark": "LIBERO-Spatial",
            "input": {
                "batch_size": 1,
                "action_horizon": 10,
                "flow_steps": self._default_num_steps,
                "image_size": [224, 224],
                "num_images": 2,
            },
            "warmup": self._args.warmup,
            "reference_warmup": self._args.reference_warmup,
            "reference_iterations": self._args.reference_iterations,
            "iterations": self._args.iterations,
            "timing": timing_summary(self._samples_ms),
            "component_breakdown": self._args.component_breakdown,
            "stage_boundary_sync": self._args.sync_stages,
            "monolithic_reference": timing_summary(self._monolithic_samples_ms),
            "decomposition_equivalence": self._equivalence,
            "stage_accounting_ms": self._accounting.summary(),
            "nvtx_reconciliation": reconciliation,
            "action_shape": list(result["actions"].shape),
            "environment": system_manifest(),
            "python_packages": python_packages(),
            "command": [sys.executable, *sys.argv],
            "relevant_environment_variables": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "OPENPI_DATA_HOME",
                    "XLA_PYTHON_CLIENT_MEM_FRACTION",
                    "XLA_FLAGS",
                )
            },
            "nvtx_ranges": ranges,
            # XLA 默认开启 command buffer（CUDA graph），这部分计算只出现在
            # CUPTI_ACTIVITY_KIND_GRAPH_TRACE 里，cuda_gpu_kern_sum 读不到。
            "gpu_activity_tables": [
                "CUPTI_ACTIVITY_KIND_KERNEL",
                "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
            ],
        }
        write_json(self._args.output_dir / "manifest.json", manifest)
        print(json.dumps({"timing": manifest["timing"]}, indent=2), flush=True)
        print(json.dumps({"nvtx_reconciliation": reconciliation}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if min(
        args.reference_warmup, args.reference_iterations, args.warmup, args.iterations
    ) < 0 or args.iterations < 1:
        raise ValueError("request counts must be >= 0 and --iterations must be >= 1")
    if args.component_breakdown and args.reference_iterations < 1:
        raise ValueError("--component-breakdown needs --reference-iterations >= 1")

    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config as training_config

    policy = policy_config.create_trained_policy(
        training_config.get_config(args.config),
        args.checkpoint,
    )
    profiled_policy = ProfilingPolicy(policy, args)
    server = WebsocketPolicyServer(
        policy=profiled_policy,
        host=args.host,
        port=args.port,
        metadata=profiled_policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
