import ast
import dataclasses
import logging
import math
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any
from typing import Literal


def _prefer_local_repo_src(script_path: str | Path, sys_path: list[str] | None = None) -> Path | None:
    path_list = sys.path if sys_path is None else sys_path
    repo_root = Path(script_path).resolve().parents[2]
    src_dir = repo_root / "src"
    if not src_dir.is_dir():
        return None

    src_dir = src_dir.resolve()
    repo_root = repo_root.resolve()
    path_list[:] = [p for p in path_list if Path(p).resolve() not in {src_dir, repo_root}]
    path_list.insert(0, str(repo_root))
    path_list.insert(0, str(src_dir))
    return src_dir


_prefer_local_repo_src(__file__)

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
import tyro

from openpi.models_pytorch.spec_pi0_pytorch import SpecArgs
from openpi.models_pytorch.spec_pi0_pytorch import SpecPI0Pytorch
from openpi.models_pytorch.spec_pi0_pytorch import _compute_radius_prefix_acceptance
from openpi.models_pytorch.spec_pi0_pytorch import _detect_verify_gripper_switch_any_k
from openpi.models_pytorch.spec_pi0_pytorch import _populate_full_round_timing
from openpi.models_pytorch.spec_pi0_pytorch import _set_legacy_timing_compat_fields
from openpi.models_pytorch.spec_pi0_pytorch import _should_run_full_pipeline_round
from openpi.models_pytorch.spec_pi0_pytorch import _should_schedule_full_fallback
from openpi.models_pytorch.spec_pi0_pytorch import _stitch_radius_prefix_output
from openpi.models_pytorch.spec_pi0_pytorch import _truncate_accepted_prefix_on_gripper_switch
from openpi.shared.nvtx import nvtx_call
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as _transforms
from scripts.spec.triton import triton_pi0_runtime as _triton_runtime


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000

    config: str = "pi0_libero"
    checkpoint_dir: str = "/path/to/pi0_libero_pytorch/"
    jax_checkpoint_dir: str = "/path/to/openpi-assets/checkpoints/pi0_libero"
    cache_dir: str = "/path/to/spec_cache"
    base_triton_path: str | None = None
    pytorch_device: str | None = None

    task_suite_name: str | None = None
    tokenizer_source: str = "auto"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_tokenizer_id: str = "google/paligemma-3b-pt-224"

    num_views: int = 2
    max_exec_steps: int = 12
    draft_history_len: int = 6
    ablation_layer_pattern: Literal["full18", "first3_last3", "first5", "first6", "last6"] = "full18"
    ablation_flow_pattern: Literal["full10", "first3", "last3"] = "full10"
    ablation_noise_seed: int | None = None
    vlm_layer_pattern: Literal[
        "vlm_full18",
        "vlm_first3_last3",
        "vlm_first5_last5",
        "vlm_first6_last6",
        "vlm_first7_last7",
        "vlm_first8_last8",
        "vlm_first4_middle2_last4",
        "vlm_first6",
        "vlm_last6",
        "vlm_skip_l2",
        "vlm_skip_l3",
        "vlm_skip_l4",
        "vlm_skip_l5",
        "vlm_skip_l2_l5",
        "vlm_keep_l1_l2_l5_l6_l8_l9_l10_l11_l13_l14_l16_l17_l18",
    ] = "vlm_full18"
    action_layer_pattern: Literal["full18", "first5", "first8", "first12", "first15"] = "full18"
    denoise_steps: Literal[3, 5, 7, 10] = 10

    draft_checkpoint: str = "/path/to/draft_vlm_block_4x4090_goal.pt"
    draft_triton_path: str | None = None
    backend: Literal["compiled", "triton"] = "triton"
    draft_backend: Literal["compiled", "triton"] | None = None
    verify_backend: Literal["compiled", "triton"] | None = None

    t_list: tuple[float, ...] = (0.10, 0.05)
    tau_radius: float = 0.3
    dist_dims: int = 7
    gripper_switch_threshold: float = 0.0
    enable_gripper_verify: bool = True
    enable_gripper_post_verify: bool = True
    gripper_full_window: int = 1
    full_fallback: bool = True
    force_full_each_round: bool = False
    periodic_full_every_n_draft_rounds: int = 0


class TritonServerPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        *,
        input_transform,
        output_transform,
        metadata: dict[str, Any],
        runtime_pool: Any,
        action_horizon: int,
        action_dim: int,
        max_exec_steps: int,
        pytorch_device: str,
        prompt_cache_builder=None,
        draft_history_len: int = 6,
        t_list: tuple[float, ...] = (0.10, 0.05),
        tau_radius: float = 0.3,
        dist_dims: int = 7,
        gripper_switch_threshold: float = 0.0,
        enable_gripper_verify: bool = True,
        enable_gripper_post_verify: bool = True,
        gripper_full_window: int = 1,
        full_fallback: bool = True,
        force_full_each_round: bool = False,
        periodic_full_every_n_draft_rounds: int = 0,
        ablation_layer_pattern: str = "full18",
        ablation_flow_pattern: str = "full10",
        ablation_noise_seed: int | None = None,
        vlm_layer_pattern: str = "vlm_full18",
        action_layer_pattern: str = "full18",
        denoise_steps: int = 10,
    ) -> None:
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._metadata = dict(metadata or {})
        self._runtime_pool = runtime_pool
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._max_exec_steps = int(max_exec_steps)
        self._device = str(pytorch_device)
        self._prompt_cache_builder = prompt_cache_builder
        self._draft_history_len = int(max(1, int(draft_history_len)))
        self._t_list = tuple(float(t) for t in t_list)
        self._tau_radius = float(tau_radius)
        self._dist_dims = int(dist_dims)
        self._gripper_switch_threshold = float(gripper_switch_threshold)
        self._enable_gripper_verify = bool(enable_gripper_verify)
        self._enable_gripper_post_verify = bool(enable_gripper_post_verify)
        self._gripper_full_window = int(gripper_full_window)
        self._full_fallback = bool(full_fallback)
        self._force_full_each_round = bool(force_full_each_round)
        self._periodic_full_every_n_draft_rounds = int(periodic_full_every_n_draft_rounds)
        if str(action_layer_pattern) == "full18" and str(ablation_layer_pattern) != "full18":
            action_layer_pattern = str(ablation_layer_pattern)
        self._ablation_fields = _triton_runtime.ablation_pattern_metadata(
            action_layer_pattern,
            ablation_flow_pattern,
        )
        self._experiment_fields = _triton_runtime.experiment_pattern_metadata(
            vlm_layer_pattern=vlm_layer_pattern,
            action_layer_pattern=action_layer_pattern,
            denoise_steps=denoise_steps,
        )
        self._ablation_noise_seed = None if ablation_noise_seed is None else int(ablation_noise_seed)
        if self._ablation_noise_seed is not None and self._ablation_noise_seed < 0:
            raise ValueError("ablation_noise_seed must be non-negative")
        self._ablation_noise_fallback_counter = 0
        self._metadata["action_ablation"] = {
            **self._ablation_fields,
            **self._experiment_fields,
            "ablation_noise_seed": self._ablation_noise_seed,
        }
        self._last_gripper: torch.Tensor | None = None
        self._action_chunk_cache: torch.Tensor | None = None
        self._action_cache_ptr: int = 0
        self._draft_rounds_since_full: int = 0
        self._pending_full_fallback: bool = False
        self._gripper_full_rounds_left: int = 0
        self._full_cache_snapshot = None

    @staticmethod
    def _full_round_accepted_prefix_len(*, action_horizon: int, max_exec_steps: int) -> int:
        return int(min(int(action_horizon), max(1, int(max_exec_steps))))

    @staticmethod
    def _timing_value(timing: dict[str, Any], key: str) -> float:
        value = timing.get(key, 0.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0

    def _validate_runtime_experiment_fields(self, timing: dict[str, Any]) -> None:
        """Reject a runtime fallback whose layer/schedule metadata disagrees."""
        if not isinstance(timing, dict):
            return
        sequence_keys = {
            "vlm_layer_indices_1based",
            "vlm_full_block_indices_1based",
            "vlm_kv_shell_indices_1based",
            "prefix_kv_indices_1based",
            "action_layer_indices_1based",
            "denoise_times",
        }
        scalar_keys = {
            "physical_vlm_layers",
            "physical_action_expert_layers",
            "num_vlm_layers",
            "num_action_layers",
            "vlm_full_block_visits",
            "vlm_kv_shell_visits",
            "kv_projection_visits",
            "denoise_steps",
            "num_flow_steps",
            "action_layer_visits",
            "denoise_updates",
            "vlm_layer_pattern",
            "action_layer_pattern",
        }
        for key in sequence_keys | scalar_keys:
            if key not in timing:
                continue
            actual = tuple(timing[key]) if key in sequence_keys else timing[key]
            expected = tuple(self._experiment_fields[key]) if key in sequence_keys else self._experiment_fields[key]
            if actual != expected:
                raise RuntimeError(
                    f"Runtime experiment metadata mismatch for {key}: "
                    f"actual={actual!r}, expected={expected!r}"
                )

    @staticmethod
    def _route_type_from_timing(timing: dict[str, Any]) -> str:
        return "full" if TritonServerPolicy._timing_value(timing, "is_full_pipeline_round") >= 0.5 else "draft"

    @staticmethod
    def _to_writable(x):
        if isinstance(x, np.ndarray):
            return np.array(x, copy=True)
        return x

    @staticmethod
    def _to_tensor_or_none(x, *, device: str, dtype: torch.dtype) -> torch.Tensor | None:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        return torch.as_tensor(np.asarray(x), device=device, dtype=dtype)

    @staticmethod
    def _as_action_batch(actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2:
            return actions.unsqueeze(0)
        return actions

    def _init_runtime_state(self, batch_size: int) -> None:
        if int(self._action_dim) < 7:
            self._last_gripper = None
            return
        if (
            self._last_gripper is None
            or int(self._last_gripper.shape[0]) != int(batch_size)
            or self._last_gripper.device.type != torch.device(self._device).type
        ):
            self._last_gripper = torch.zeros(
                (int(batch_size),),
                device=self._device,
                dtype=torch.float32,
            )

    def _advance_last_gripper(self, executed_steps: int | None) -> None:
        if self._last_gripper is None or self._action_chunk_cache is None:
            return
        if (
            int(self._action_chunk_cache.shape[0]) != int(self._last_gripper.shape[0])
            or int(self._action_chunk_cache.shape[2]) < 7
        ):
            return
        horizon = int(self._action_chunk_cache.shape[1])
        step_adv = int(self._max_exec_steps) if executed_steps is None else int(executed_steps)
        step_adv = max(0, min(step_adv, horizon))
        self._action_cache_ptr = min(int(self._action_cache_ptr) + step_adv, horizon)
        if self._action_cache_ptr <= 0:
            return
        executed_anchor = self._action_chunk_cache[:, self._action_cache_ptr - 1, 6].to(dtype=torch.float32)
        self._last_gripper = executed_anchor.detach()

    def reset_runtime_state(self) -> None:
        self._last_gripper = None
        self._action_chunk_cache = None
        self._action_cache_ptr = 0
        self._draft_rounds_since_full = 0
        self._pending_full_fallback = False
        self._gripper_full_rounds_left = 0
        self._full_cache_snapshot = None

    def _accept_full_round_actions(self, actions: torch.Tensor, *, cache_snapshot) -> None:
        self._draft_rounds_since_full = 0
        if int(self._gripper_full_rounds_left) > 0:
            self._gripper_full_rounds_left = max(0, int(self._gripper_full_rounds_left) - 1)
            self._pending_full_fallback = bool(self._gripper_full_rounds_left > 0)
        else:
            self._pending_full_fallback = False
        self._full_cache_snapshot = cache_snapshot
        self._action_chunk_cache = actions.detach().clone()
        self._action_cache_ptr = 0

    def _schedule_gripper_full_fallback(self) -> None:
        self._pending_full_fallback = True
        self._gripper_full_rounds_left = max(1, int(self._gripper_full_window))

    @staticmethod
    def _supports_spec_session(session: Any) -> bool:
        return all(
            hasattr(session, name)
            for name in (
                "run_draft_with_timing",
                "run_verify_with_timing",
                "capture_full_cache_snapshot",
            )
        )

    def _format_outputs(self, *, transformed: dict[str, Any], actions: torch.Tensor) -> dict[str, Any]:
        actions_np = actions.detach().to(dtype=torch.float32).cpu().numpy()
        if actions_np.ndim == 3 and int(actions_np.shape[0]) == 1:
            actions_np = actions_np[0]
        outputs = {
            "state": np.asarray(transformed["state"]),
            "actions": actions_np,
        }
        return self._output_transform(outputs)

    def _log_response(self, outputs: dict[str, Any]) -> None:
        timing = outputs.get("policy_timing", {})
        actions_np = np.asarray(outputs.get("actions", []))
        logging.info("Response")
        logging.info("actions shape=%s", tuple(actions_np.shape))
        logging.info(
            "summary accepted=%d max_exec=%d total_ms=%.2f prefill=%d fallback=%d random_verify=%d",
            int(outputs.get("accepted_prefix_len", 0)),
            self._max_exec_steps,
            self._timing_value(timing, "sample_actions_ms"),
            int(round(self._timing_value(timing, "did_prefill"))),
            int(round(self._timing_value(timing, "used_full_fallback"))),
            int(round(self._timing_value(timing, "verify_mode_random"))),
        )
        logging.info(
            "timing enc=%.2f prefill=%.2f draft=%.2f verify=%.2f denoise=%.2f total=%.2f",
            self._timing_value(timing, "encoder_ms"),
            self._timing_value(timing, "vlm_prefill_ms"),
            self._timing_value(timing, "draft_ms"),
            self._timing_value(timing, "action_verify_ms"),
            self._timing_value(timing, "full_fallback_ms"),
            self._timing_value(timing, "total_ms"),
        )
        logging.info(
            "ablation layers=%s(%s) flow=%s(%s) dt=%.1f noise_seed=%s noise_mode=%s episode=%s infer=%s",
            timing.get("ablation_layer_pattern", self._ablation_fields["ablation_layer_pattern"]),
            timing.get("action_layer_indices_1based", self._ablation_fields["action_layer_indices_1based"]),
            timing.get("ablation_flow_pattern", self._ablation_fields["ablation_flow_pattern"]),
            timing.get("flow_step_indices", self._ablation_fields["flow_step_indices"]),
            float(timing.get("flow_dt", self._ablation_fields["flow_dt"])),
            timing.get("ablation_noise_seed", self._ablation_noise_seed),
            timing.get("noise_seed_mode", "unseeded"),
            timing.get("trace_episode_idx"),
            timing.get("trace_infer_id"),
        )
        logging.info(
            "experiment vlm=%s(%s) action=%s(%s) denoise_steps=%s times=%s dt=%s",
            timing.get("vlm_layer_pattern", self._experiment_fields["vlm_layer_pattern"]),
            timing.get("vlm_layer_indices_1based", self._experiment_fields["vlm_layer_indices_1based"]),
            timing.get("action_layer_pattern", self._experiment_fields["action_layer_pattern"]),
            timing.get("action_layer_indices_1based", self._experiment_fields["action_layer_indices_1based"]),
            timing.get("denoise_steps", self._experiment_fields["denoise_steps"]),
            timing.get("denoise_times", self._experiment_fields["denoise_times"]),
            timing.get("flow_dt", self._experiment_fields["flow_dt"]),
        )

    @staticmethod
    def _optional_scalar_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            value_np = np.asarray(value)
            if value_np.size != 1:
                return None
            return int(value_np.reshape(-1)[0])
        except (TypeError, ValueError, OverflowError):
            return None

    def _make_ablation_noise(
        self,
        *,
        episode_idx: int | None,
        infer_id: int | None,
    ) -> tuple[np.ndarray, str]:
        if self._ablation_noise_seed is None:
            raise RuntimeError("Seeded ablation noise requested without ablation_noise_seed")
        if episode_idx is not None and infer_id is not None:
            seed_parts = [self._ablation_noise_seed, int(episode_idx), int(infer_id)]
            mode = "episode_infer"
        else:
            seed_parts = [self._ablation_noise_seed, self._ablation_noise_fallback_counter]
            self._ablation_noise_fallback_counter += 1
            mode = "request_order_fallback"
        generator = np.random.default_rng(np.random.SeedSequence(seed_parts))
        noise = generator.standard_normal((self._action_horizon, self._action_dim)).astype(np.float32)
        return noise, mode

    def _attach_ablation_metadata(
        self,
        outputs: dict[str, Any],
        *,
        episode_idx: int | None,
        infer_id: int | None,
        noise_seed_mode: str,
    ) -> None:
        timing = outputs.get("policy_timing")
        if not isinstance(timing, dict):
            timing = {}
            outputs["policy_timing"] = timing
        timing.update(self._ablation_fields)
        timing.update(self._experiment_fields)
        timing["ablation_noise_seed"] = self._ablation_noise_seed
        timing["global_noise_seed"] = self._ablation_noise_seed
        timing["noise_seed_rule"] = "SeedSequence(global_noise_seed, episode_idx, replan_idx)"
        timing["noise_seed_mode"] = noise_seed_mode
        if noise_seed_mode == "episode_infer" and episode_idx is not None and infer_id is not None:
            # Reconstruct only the deterministic seeded sample for audit; never
            # advance the fallback RNG counter or change the actual model input.
            import hashlib
            audit_rng = np.random.default_rng(np.random.SeedSequence(
                [self._ablation_noise_seed, int(episode_idx), int(infer_id)]))
            audit_noise = audit_rng.standard_normal((self._action_horizon, self._action_dim)).astype(np.float32)
            timing["noise_sha256"] = hashlib.sha256(audit_noise.tobytes()).hexdigest()
        if episode_idx is not None:
            timing["trace_episode_idx"] = int(episode_idx)
        if infer_id is not None:
            timing["trace_infer_id"] = int(infer_id)

    def _run_runtime(
        self,
        *,
        prompt: str,
        images: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        if torch.cuda.is_available() and self._device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        stage_timing: dict[str, float] = {}

        def _invoke_runtime():
            if hasattr(self._runtime_pool, "start_session"):
                session = self._runtime_pool.start_session(prompt)
                prepared = session.prepare_observation(images=images, state=state)
                if hasattr(session, "run_full_with_timing"):
                    return session.run_full_with_timing(prepared=prepared, noise=noise)
                return session.run_full(prepared=prepared, noise=noise)
            return self._runtime_pool.forward(prompt=prompt, images=images, state=state, noise=noise)

        try:
            runtime_out = _invoke_runtime()
        except KeyError:
            if self._prompt_cache_builder is None:
                raise
            self._prompt_cache_builder(prompt)
            self._runtime_pool.reload_manifest()
            runtime_out = _invoke_runtime()
        if isinstance(runtime_out, tuple):
            actions, stage_timing = runtime_out
        else:
            actions = runtime_out
        self._validate_runtime_experiment_fields(stage_timing)
        if torch.cuda.is_available() and self._device.startswith("cuda"):
            torch.cuda.synchronize()
        return actions, (time.perf_counter() - t0) * 1000.0, stage_timing

    def _run_spec_session(
        self,
        *,
        session: Any,
        transformed: dict[str, Any],
        prompt: str,
        images: torch.Tensor,
        state: torch.Tensor,
        noise: torch.Tensor,
        trace_episode_idx: int | None = None,
        trace_infer_id: int | None = None,
        noise_seed_mode: str = "unseeded",
    ) -> dict[str, Any]:
        if torch.cuda.is_available() and self._device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        prepared = session.prepare_observation(images=images, state=state)
        run_full_pipeline_round = _should_run_full_pipeline_round(
            cache_ready=self._full_cache_snapshot is not None,
            full_fallback=bool(self._full_fallback),
            pending_full_fallback=bool(self._pending_full_fallback),
            force_full_each_round=bool(self._force_full_each_round),
            periodic_full_every_n_draft_rounds=int(self._periodic_full_every_n_draft_rounds),
            draft_rounds_since_full=int(self._draft_rounds_since_full),
        )

        if run_full_pipeline_round:
            full_out = nvtx_call(
                "realtime/round/full",
                session.run_full_with_timing,
                prepared=prepared,
                noise=noise,
            )
            if isinstance(full_out, tuple):
                actions, stage_timing = full_out
            else:
                actions, stage_timing = full_out, {}
            self._validate_runtime_experiment_fields(stage_timing)
            full_snapshot = nvtx_call("realtime/full/cache_snapshot", session.capture_full_cache_snapshot)
            action_batch = self._as_action_batch(actions)
            self._accept_full_round_actions(action_batch, cache_snapshot=full_snapshot)
            outputs = self._format_outputs(transformed=transformed, actions=action_batch)
            accepted_prefix_len = self._full_round_accepted_prefix_len(
                action_horizon=self._action_horizon,
                max_exec_steps=self._max_exec_steps,
            )
            outputs["accepted_prefix_len"] = accepted_prefix_len
            sample_actions_ms = (time.perf_counter() - t0) * 1000.0
            timing = {
                "sample_actions_ms": float(sample_actions_ms),
                "encoder_ms": float(stage_timing.get("encoder_ms", float("nan"))),
            }
            _populate_full_round_timing(
                timing,
                verify_mode="radius",
                action_horizon=self._action_horizon,
                max_exec_steps=self._max_exec_steps,
                full_prefill_ms=float(stage_timing.get("vlm_prefill_ms", 0.0)),
                full_action_ms=float(stage_timing.get("decoder_ms", sample_actions_ms)),
                gripper_verify_enabled=bool(self._enable_gripper_verify),
            )
            timing["sample_actions_ms"] = float(sample_actions_ms)
            timing["total_ms"] = float(stage_timing.get("total_ms", sample_actions_ms))
            timing["vision_ms"] = float(stage_timing.get("vision_ms", stage_timing.get("encoder_ms", float("nan"))))
            timing["prefix_total_ms"] = float(stage_timing.get("prefix_total_ms", stage_timing.get("vlm_prefill_ms", 0.0)))
            timing["vlm_full_block_ms"] = stage_timing.get("vlm_full_block_ms")
            timing["vlm_kv_shell_ms"] = stage_timing.get("vlm_kv_shell_ms")
            timing["action_denoise_ms"] = float(stage_timing.get("action_denoise_ms", stage_timing.get("decoder_ms", sample_actions_ms)))
            timing["final_output_ms"] = stage_timing.get("final_output_ms")
            timing["e2e_policy_ms"] = float(stage_timing.get("e2e_policy_ms", stage_timing.get("total_ms", sample_actions_ms)))
            timing["radius_dist_mean"] = float("nan")
            timing["route_type"] = "full"
            outputs["policy_timing"] = timing
            self._attach_ablation_metadata(
                outputs,
                episode_idx=trace_episode_idx,
                infer_id=trace_infer_id,
                noise_seed_mode=noise_seed_mode,
            )
            self._log_response(outputs)
            return outputs

        x0_draft, draft_timing = nvtx_call(
            "realtime/round/draft",
            session.run_draft_with_timing,
            prepared=prepared,
        )
        x0_hat, verify_timing = nvtx_call(
            "realtime/round/verify",
            session.run_verify_with_timing,
            cache_snapshot=self._full_cache_snapshot,
            prepared=prepared,
            noise=self._as_action_batch(noise),
            x0_draft=x0_draft,
            t_list=self._t_list,
        )
        eval_h = int(min(int(x0_draft.shape[1]), max(1, int(self._max_exec_steps))))
        accepted_prefix_len, dist = nvtx_call(
            "realtime/verify/acceptance_distance",
            _compute_radius_prefix_acceptance,
            x0_draft=x0_draft,
            x0_hat=x0_hat,
            tau_radius=float(self._tau_radius),
            dist_dims=int(self._dist_dims),
            eval_h=int(eval_h),
        )
        x0_tail = x0_hat.mean(dim=1)
        x0_out = _stitch_radius_prefix_output(
            x0_draft=x0_draft,
            x0_tail=x0_tail,
            accepted_prefix_len=accepted_prefix_len,
        )
        batch_size = int(x0_out.shape[0])
        gripper_switch_cut_mask = torch.zeros((batch_size,), device=x0_out.device, dtype=torch.bool)
        gripper_verify_stop_mask = torch.zeros((batch_size,), device=x0_out.device, dtype=torch.bool)
        last_gripper = None
        if self._last_gripper is not None and int(x0_out.shape[2]) >= 7:
            last_gripper = self._last_gripper.to(device=x0_out.device, dtype=torch.float32)
        if bool(self._enable_gripper_verify):
            gripper_verify_stop_mask = _detect_verify_gripper_switch_any_k(
                x0_hat=x0_hat,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(self._gripper_switch_threshold),
                eval_h=int(eval_h),
            )
            accepted_prefix_len = torch.where(
                gripper_verify_stop_mask,
                torch.zeros_like(accepted_prefix_len),
                accepted_prefix_len,
            )
            x0_out = torch.where(gripper_verify_stop_mask[:, None, None], x0_tail, x0_out)
        if bool(self._enable_gripper_post_verify):
            accepted_after_cut, gripper_switch_cut_mask = _truncate_accepted_prefix_on_gripper_switch(
                x0_out=x0_out,
                accepted_prefix_len=accepted_prefix_len,
                gripper_prev=last_gripper,
                gripper_switch_threshold=float(self._gripper_switch_threshold),
            )
            accepted_prefix_len = torch.where(gripper_verify_stop_mask, accepted_prefix_len, accepted_after_cut)
            gripper_switch_cut_mask = gripper_switch_cut_mask & (~gripper_verify_stop_mask)

        scheduled_full_fallback_gripper = gripper_switch_cut_mask | gripper_verify_stop_mask
        should_schedule_full_fallback = _should_schedule_full_fallback(
            full_fallback=bool(self._full_fallback),
            accepted_prefix_len=accepted_prefix_len,
            gripper_switch_cut_mask=scheduled_full_fallback_gripper,
        )
        if should_schedule_full_fallback:
            if bool(scheduled_full_fallback_gripper.any().item()):
                self._schedule_gripper_full_fallback()
            else:
                self._pending_full_fallback = True
                self._gripper_full_rounds_left = 0
        else:
            self._action_chunk_cache = x0_out.detach().clone()
            self._action_cache_ptr = 0

        self._draft_rounds_since_full = int(self._draft_rounds_since_full) + 1
        outputs = self._format_outputs(transformed=transformed, actions=x0_out)
        accepted_prefix_len_scalar = float(accepted_prefix_len.to(dtype=torch.float32).mean().item())
        outputs["accepted_prefix_len"] = int(round(accepted_prefix_len_scalar))
        sample_actions_ms = (time.perf_counter() - t0) * 1000.0
        encoder_ms = float(draft_timing.get("encoder_ms", float("nan")))
        draft_ms = float(draft_timing.get("draft_ms", float("nan")))
        verify_ms = float(verify_timing.get("action_verify_ms", float("nan")))
        timing = {
            "sample_actions_ms": float(sample_actions_ms),
            "encoder_ms": encoder_ms,
            "vlm_prefill_ms": 0.0,
            "draft_ms": draft_ms,
            "action_verify_ms": verify_ms,
            "full_fallback_ms": 0.0,
            "total_ms": float(encoder_ms + draft_ms + verify_ms),
            "accepted_prefix_len": float(accepted_prefix_len_scalar),
            "accepted_prefix_len_mean": float(accepted_prefix_len_scalar),
            "radius_dist": float(dist.mean().item()),
            "radius_dist_mean": float(dist.mean().item()),
            "did_prefill": 0.0,
            "is_full_pipeline_round": 0.0,
            "used_full_fallback": 0.0,
            "scheduled_full_fallback": 1.0 if should_schedule_full_fallback else 0.0,
            "verify_mode_random": 0.0,
            "gripper_verify_enabled": 1.0 if self._enable_gripper_verify else 0.0,
            "include_in_draft_accept_metrics": 1.0,
            "gripper_switch_cut_rate": float(gripper_switch_cut_mask.to(dtype=torch.float32).mean().item()),
            "scheduled_full_fallback_gripper": float(scheduled_full_fallback_gripper.any().to(dtype=torch.float32).item()),
            "gripper_verify_stop_rate": float(gripper_verify_stop_mask.to(dtype=torch.float32).mean().item()),
            "route_type": "draft",
        }
        _set_legacy_timing_compat_fields(timing)
        outputs["policy_timing"] = timing
        self._attach_ablation_metadata(
            outputs,
            episode_idx=trace_episode_idx,
            infer_id=trace_infer_id,
            noise_seed_mode=noise_seed_mode,
        )
        self._log_response(outputs)
        return outputs

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        if torch.cuda.is_available() and hasattr(torch, "compiler"):
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if callable(mark):
                mark()

        inputs = jax.tree.map(self._to_writable, obs)
        prompt = str(inputs.get("prompt", ""))
        reset_flag = inputs.pop("__reset_policy_state__", False)
        if isinstance(reset_flag, np.ndarray):
            reset_runtime_state = bool(reset_flag.reshape(-1)[0]) if reset_flag.size > 0 else False
        else:
            reset_runtime_state = bool(reset_flag)
        if reset_runtime_state:
            if hasattr(self._runtime_pool, "reset_runtime_state"):
                self._runtime_pool.reset_runtime_state()
            else:
                self.reset_runtime_state()

        executed_steps_raw = inputs.pop("__executed_steps__", None)
        if executed_steps_raw is None:
            executed_steps = None
        elif isinstance(executed_steps_raw, np.ndarray):
            executed_steps = int(executed_steps_raw.reshape(-1)[0]) if executed_steps_raw.size > 0 else None
        else:
            executed_steps = int(executed_steps_raw)
        inputs.pop("__trace_run_id__", None)
        inputs.pop("__trace_task_id__", None)
        trace_episode_idx = self._optional_scalar_int(inputs.pop("__trace_episode_idx__", None))
        trace_infer_id = self._optional_scalar_int(inputs.pop("__trace_infer_id__", None))

        noise_seed_mode = "provided" if noise is not None else "unseeded"
        if noise is None and self._ablation_noise_seed is not None:
            noise, noise_seed_mode = self._make_ablation_noise(
                episode_idx=trace_episode_idx,
                infer_id=trace_infer_id,
            )

        transformed = self._input_transform(inputs)
        images, state, noise_tensor = _triton_runtime.prepare_triton_inputs_from_transformed(
            transformed=transformed,
            device=self._device,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            noise=noise,
        )
        tokenized_prompt = self._to_tensor_or_none(
            transformed.get("tokenized_prompt"),
            device=self._device,
            dtype=torch.long,
        )
        tokenized_prompt_mask = self._to_tensor_or_none(
            transformed.get("tokenized_prompt_mask"),
            device=self._device,
            dtype=torch.bool,
        )
        batch_size = 1 if state.ndim == 1 else int(state.shape[0])
        self._init_runtime_state(batch_size)
        self._advance_last_gripper(executed_steps)

        if hasattr(self._runtime_pool, "sample_actions_with_timing"):
            if torch.cuda.is_available() and self._device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            try:
                actions, stage_timing = self._runtime_pool.sample_actions_with_timing(
                    prompt=prompt,
                    images=images,
                    state=state,
                    noise=noise_tensor,
                    tokenized_prompt=tokenized_prompt,
                    tokenized_prompt_mask=tokenized_prompt_mask,
                    executed_steps=executed_steps,
                )
            except KeyError:
                if self._prompt_cache_builder is None:
                    raise
                self._prompt_cache_builder(prompt)
                if hasattr(self._runtime_pool, "reload_manifest"):
                    self._runtime_pool.reload_manifest()
                actions, stage_timing = self._runtime_pool.sample_actions_with_timing(
                    prompt=prompt,
                    images=images,
                    state=state,
                    noise=noise_tensor,
                    tokenized_prompt=tokenized_prompt,
                    tokenized_prompt_mask=tokenized_prompt_mask,
                    executed_steps=executed_steps,
                )
            if torch.cuda.is_available() and self._device.startswith("cuda"):
                torch.cuda.synchronize()
            sample_actions_ms = (time.perf_counter() - t0) * 1000.0
            outputs = self._format_outputs(transformed=transformed, actions=self._as_action_batch(actions))
            outputs["accepted_prefix_len"] = int(round(float(stage_timing.get("accepted_prefix_len", 0.0))))
            policy_timing = {
                "sample_actions_ms": float(sample_actions_ms),
                **stage_timing,
            }
            if "route_type" not in policy_timing:
                policy_timing["route_type"] = self._route_type_from_timing(policy_timing)
            self._validate_runtime_experiment_fields(policy_timing)
            outputs["policy_timing"] = policy_timing
            self._attach_ablation_metadata(
                outputs,
                episode_idx=trace_episode_idx,
                infer_id=trace_infer_id,
                noise_seed_mode=noise_seed_mode,
            )
            self._log_response(outputs)
            return outputs

        if hasattr(self._runtime_pool, "start_session"):
            session = self._runtime_pool.start_session(prompt)
            if self._supports_spec_session(session):
                return self._run_spec_session(
                    session=session,
                    transformed=transformed,
                    prompt=prompt,
                    images=images,
                    state=state,
                    noise=noise_tensor,
                    trace_episode_idx=trace_episode_idx,
                    trace_infer_id=trace_infer_id,
                    noise_seed_mode=noise_seed_mode,
                )

        actions, sample_actions_ms, stage_timing = self._run_runtime(
            prompt=prompt,
            images=images,
            state=state,
            noise=noise_tensor,
        )

        outputs = {
            "state": np.asarray(transformed["state"]),
            "actions": actions.detach().to(dtype=torch.float32).cpu().numpy(),
        }
        outputs = self._output_transform(outputs)

        accepted_prefix_len = self._full_round_accepted_prefix_len(
            action_horizon=self._action_horizon,
            max_exec_steps=self._max_exec_steps,
        )
        outputs["accepted_prefix_len"] = accepted_prefix_len
        outputs["policy_timing"] = {
            "sample_actions_ms": float(sample_actions_ms),
            "encoder_ms": float(stage_timing.get("encoder_ms", float("nan"))),
            "vision_ms": float(stage_timing.get("vision_ms", stage_timing.get("encoder_ms", float("nan")))),
            "vlm_prefill_ms": float(stage_timing.get("vlm_prefill_ms", float("nan"))),
            "vlm_full_block_ms": stage_timing.get("vlm_full_block_ms"),
            "vlm_kv_shell_ms": stage_timing.get("vlm_kv_shell_ms"),
            "prefix_total_ms": float(stage_timing.get("prefix_total_ms", stage_timing.get("vlm_prefill_ms", float("nan")))),
            "draft_ms": 0.0,
            "action_verify_ms": 0.0,
            "action_denoise_ms": float(stage_timing.get("action_denoise_ms", stage_timing.get("decoder_ms", float("nan")))),
            "final_output_ms": stage_timing.get("final_output_ms"),
            "e2e_policy_ms": float(stage_timing.get("e2e_policy_ms", stage_timing.get("total_ms", sample_actions_ms))),
            "full_fallback_ms": float(stage_timing.get("decoder_ms", sample_actions_ms)),
            "total_ms": float(stage_timing.get("total_ms", sample_actions_ms)),
            "accepted_prefix_len": float(accepted_prefix_len),
            "accepted_prefix_len_mean": float(accepted_prefix_len),
            "radius_dist_mean": float("nan"),
            "gripper_override_rate": 0.0,
            "did_prefill": 1.0,
            "is_full_pipeline_round": 1.0,
            "used_full_fallback": 1.0,
            "scheduled_full_fallback": 0.0,
            "verify_mode_random": 0.0,
            "gripper_verify_enabled": 0.0,
            "include_in_draft_accept_metrics": 0.0,
            "route_type": "full",
        }
        self._attach_ablation_metadata(
            outputs,
            episode_idx=trace_episode_idx,
            infer_id=trace_infer_id,
            noise_seed_mode=noise_seed_mode,
        )
        self._log_response(outputs)
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _build_policy_components(*, config_name: str, checkpoint_dir: str | Path):
    train_config = _config.get_config(str(config_name))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(Path(checkpoint_dir) / "assets", data_config.asset_id)
    input_transform = _transforms.compose(
        [
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    return train_config, input_transform, output_transform


def _resolve_triton_base_artifact(base_triton_path: str | Path) -> tuple[Path, Path]:
    artifact_path = Path(base_triton_path).expanduser().resolve()
    if artifact_path.is_dir():
        return artifact_path / "base_weights.pkl", artifact_path
    return artifact_path, artifact_path.parent


def _resolve_triton_draft_artifact(draft_path: str | Path) -> tuple[Path, Path]:
    artifact_path = Path(draft_path).expanduser().resolve()
    if artifact_path.is_dir():
        return artifact_path / "draft_triton.pkl", artifact_path
    return artifact_path, artifact_path.parent


def _language_from_bddl(bddl_path: Path) -> str:
    for line in bddl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("(:language ") and line.endswith(")"):
            return line.removeprefix("(:language ").removesuffix(")").strip()
    raise ValueError(f"Could not find a (:language ...) entry in {bddl_path}")


def _libero_task_map(task_map_path: Path) -> dict[str, list[str]]:
    tree = ast.parse(task_map_path.read_text(encoding="utf-8"), filename=str(task_map_path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets
        ):
            task_map = ast.literal_eval(node.value)
            if not isinstance(task_map, dict):
                break
            return {str(suite): [str(task) for task in tasks] for suite, tasks in task_map.items()}
    raise ValueError(f"Could not find libero_task_map in {task_map_path}")


def _suite_prompts_from_bddl(task_suite_name: str) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    libero_root = repo_root / "third_party" / "libero" / "libero" / "libero"
    suite_dir = libero_root / "bddl_files" / task_suite_name
    if not suite_dir.is_dir():
        raise ModuleNotFoundError(
            f"Could not load LIBERO task prompts for {task_suite_name!r}. "
            "Install the LIBERO Python package, initialize the third_party/libero submodule, "
            "or omit --task-suite-name to build prompt caches lazily from client requests."
        )

    task_map_path = libero_root / "benchmark" / "libero_suite_task_map.py"
    if task_map_path.is_file():
        task_map = _libero_task_map(task_map_path)
        task_names = task_map.get(task_suite_name)
        if task_names is None:
            raise ValueError(f"Unknown LIBERO task suite {task_suite_name!r} in {task_map_path}")
        bddl_paths = [suite_dir / f"{task_name}.bddl" for task_name in task_names]
    else:
        bddl_paths = sorted(suite_dir.glob("*.bddl"))

    if not bddl_paths:
        raise ValueError(f"No BDDL files found for LIBERO task suite {task_suite_name!r} in {suite_dir}")
    missing_paths = [path for path in bddl_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"Missing {len(missing_paths)} BDDL files for LIBERO task suite {task_suite_name!r}; "
            f"first missing file: {missing_paths[0]}"
        )
    return [_language_from_bddl(path) for path in bddl_paths]


def _suite_prompts(task_suite_name: str) -> list[str]:
    try:
        from libero.libero import benchmark
    except ModuleNotFoundError as exc:
        if exc.name != "libero":
            raise
        prompts = _suite_prompts_from_bddl(task_suite_name)
        logging.info("Loaded %d LIBERO prompts from local BDDL files for %s", len(prompts), task_suite_name)
        return prompts

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    return [str(task_suite.get_task(task_id).language) for task_id in range(task_suite.n_tasks)]


class _CompiledSpecSessionPool:
    def reload_manifest(self) -> None:
        pass

    def start_session(self, prompt: str):
        return _triton_runtime.TritonRuntimeSession(prompt=str(prompt), runtime=object())


def _resolve_backend(args: Args) -> Literal["compiled", "triton"]:
    override_backends = [backend for backend in (args.draft_backend, args.verify_backend) if backend is not None]
    if not override_backends:
        return args.backend
    if len(set(override_backends)) != 1:
        raise ValueError("draft_backend and verify_backend must match when using the unified Spec backend.")
    override_backend = override_backends[0]
    if args.backend != "triton" and args.backend != override_backend:
        raise ValueError(f"Conflicting Spec backends: backend={args.backend!r}, override={override_backend!r}")
    return override_backend


def _move_compiled_spec_modules_to_device(spec_model: SpecPI0Pytorch, *, device: str) -> None:
    spec_model.to("cpu")
    spec_model.paligemma_with_expert.paligemma.to(device)
    spec_model.paligemma_with_expert.gemma_expert.to(device)
    spec_model.action_in_proj.to(device)
    spec_model.action_out_proj.to(device)
    if hasattr(spec_model, "_verify_tks") and getattr(spec_model, "_verify_tks") is not None:
        spec_model._verify_tks = spec_model._verify_tks.to(device)  # noqa: SLF001
    if getattr(spec_model, "_draft_head", None) is not None:
        spec_model._draft_head.to(device)  # noqa: SLF001
    if hasattr(spec_model, "state_proj"):
        spec_model.state_proj.to(device)
    if hasattr(spec_model, "action_time_mlp_in"):
        spec_model.action_time_mlp_in.to(device)
    if hasattr(spec_model, "action_time_mlp_out"):
        spec_model.action_time_mlp_out.to(device)
    if hasattr(spec_model, "time_mlp_in"):
        spec_model.time_mlp_in.to(device)
    if hasattr(spec_model, "time_mlp_out"):
        spec_model.time_mlp_out.to(device)


def _build_compiled_spec_runtime(
    *,
    train_config,
    checkpoint_dir: str | Path,
    draft_checkpoint: str | Path,
    device: str,
    max_exec_steps: int,
    t_list: tuple[float, ...],
    tau_radius: float,
    dist_dims: int,
    gripper_switch_threshold: float,
    enable_gripper_verify: bool,
    enable_gripper_post_verify: bool,
    gripper_full_window: int,
):
    base_policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device="cpu",
    )
    base_model = base_policy._model  # noqa: SLF001
    spec_model = SpecPI0Pytorch(
        base_model.config,
        spec_args=SpecArgs(
            max_exec_steps=int(max_exec_steps),
            t_list=tuple(float(x) for x in t_list),
            tau_radius=float(tau_radius),
            dist_dims=int(dist_dims),
            gripper_switch_threshold=float(gripper_switch_threshold),
            enable_gripper_verify=bool(enable_gripper_verify),
            enable_gripper_post_verify=bool(enable_gripper_post_verify),
            gripper_full_window=int(gripper_full_window),
        ),
    )
    spec_model.load_state_dict(base_model.state_dict(), strict=True)
    spec_model.init_spec_modules()
    spec_model.load_draft_head(str(draft_checkpoint))
    _move_compiled_spec_modules_to_device(spec_model, device=str(device))
    spec_model.eval()
    if hasattr(torch, "compile"):
        spec_model._encoder_stage = torch.compile(spec_model._encoder_stage_impl, mode="max-autotune")
        spec_model._vlm_prefill_stage = torch.compile(spec_model._vlm_prefill_stage_impl, mode="max-autotune")
        spec_model._action_stage = torch.compile(spec_model._action_stage_impl, mode="max-autotune")
        spec_model._full_action_stage = torch.compile(spec_model._full_action_stage_impl, mode="max-autotune")
        if getattr(spec_model, "_draft_head", None) is not None:
            spec_model._draft_predict_actions = torch.compile(  # noqa: SLF001
                spec_model._draft_head.forward,
                mode="max-autotune",
            )
    return _triton_runtime.CompiledSpecVerifyRuntime(spec_model=spec_model, device=str(device))


def main(args: Args) -> None:
    device = args.pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(device).startswith("cuda"):
        raise RuntimeError("The Triton Libero server requires a CUDA device.")

    backend = _resolve_backend(args)
    # New experiments use independent VLM/action/denoising axes. Preserve the
    # legacy ablation flags for existing callers, but make an explicit new
    # action pattern win unless only the legacy flag was changed.
    vlm_layer_pattern = str(args.vlm_layer_pattern)
    action_layer_pattern = str(args.action_layer_pattern)
    denoise_steps = int(args.denoise_steps)
    if action_layer_pattern == "full18" and args.ablation_layer_pattern != "full18":
        action_layer_pattern = str(args.ablation_layer_pattern)
    if denoise_steps == 10 and args.ablation_flow_pattern != "full10":
        # Legacy first3/last3 remain available for the previous experiment;
        # the new 5-step experiment always uses denoise_steps=5.
        legacy_flow_pattern = str(args.ablation_flow_pattern)
    else:
        legacy_flow_pattern = "full10"
    _triton_runtime.validate_ablation_patterns(
        action_layer_pattern,
        legacy_flow_pattern,
    )
    _triton_runtime.validate_experiment_patterns(
        vlm_layer_pattern=vlm_layer_pattern,
        action_layer_pattern=action_layer_pattern,
        denoise_steps=denoise_steps,
    )
    startup_fields = _triton_runtime.experiment_pattern_metadata(
        vlm_layer_pattern=vlm_layer_pattern,
        action_layer_pattern=action_layer_pattern,
        denoise_steps=denoise_steps,
    )
    logging.info(
        "Physical VLM layers = %d; Physical Action Expert layers = %d",
        startup_fields["physical_vlm_layers"],
        startup_fields["physical_action_expert_layers"],
    )
    logging.info(
        "Experiment layers: VLM full=%s KV-shell=%s prefix-KV=%s AE=%s; denoise times=%s dt=%s",
        startup_fields["vlm_full_block_indices_1based"],
        startup_fields["vlm_kv_shell_indices_1based"],
        startup_fields["prefix_kv_indices_1based"],
        startup_fields["action_layer_indices_1based"],
        startup_fields["denoise_times"],
        startup_fields["flow_dt"],
    )
    if backend == "compiled" and (
        action_layer_pattern != "full18" or legacy_flow_pattern != "full10" or denoise_steps != 10
    ):
        raise ValueError("Action Expert ablation patterns are supported only by the Triton backend.")
    triton_base_weights_path: Path | None = None
    triton_artifact_dir: Path | None = None
    if backend == "triton" and args.base_triton_path is not None:
        triton_base_weights_path, triton_artifact_dir = _resolve_triton_base_artifact(args.base_triton_path)

    train_config, input_transform, output_transform = _build_policy_components(
        config_name=args.config,
        checkpoint_dir=triton_artifact_dir if triton_artifact_dir is not None else args.checkpoint_dir,
    )

    compiled_spec_runtime = None
    prompt_cache_builder = None
    if backend == "compiled":
        compiled_spec_runtime = _build_compiled_spec_runtime(
            train_config=train_config,
            checkpoint_dir=args.checkpoint_dir,
            draft_checkpoint=args.draft_checkpoint,
            device=str(device),
            max_exec_steps=int(args.max_exec_steps),
            t_list=tuple(float(x) for x in args.t_list),
            tau_radius=float(args.tau_radius),
            dist_dims=int(args.dist_dims),
            gripper_switch_threshold=float(args.gripper_switch_threshold),
            enable_gripper_verify=bool(args.enable_gripper_verify),
            enable_gripper_post_verify=bool(args.enable_gripper_post_verify),
            gripper_full_window=int(args.gripper_full_window),
        )
        runtime_backend = _CompiledSpecSessionPool()
    else:
        if args.base_triton_path is not None:
            if args.draft_triton_path is None:
                raise ValueError("--draft-triton-path is required when --base-triton-path is set.")
            if triton_base_weights_path is None:
                raise RuntimeError("Internal error: triton base path was not resolved.")
            base_weights_path = triton_base_weights_path
            draft_triton_path, draft_artifact_dir = _resolve_triton_draft_artifact(args.draft_triton_path)
            prompts = _suite_prompts(args.task_suite_name) if args.task_suite_name else []
            cache_artifacts = _triton_runtime.build_prompt_cache_from_base(
                base_weights_path=base_weights_path,
                cache_dir=draft_artifact_dir,
                prompts=prompts,
                tokenizer_source=args.tokenizer_source,
                hf_endpoint=args.hf_endpoint,
                hf_tokenizer_id=args.hf_tokenizer_id,
            )
            manifest_path = cache_artifacts["manifest_path"]
        else:
            prompts = _suite_prompts(args.task_suite_name) if args.task_suite_name else []
            cache_artifacts = _triton_runtime.build_prompt_cache(
                jax_checkpoint_dir=args.jax_checkpoint_dir,
                cache_dir=args.cache_dir,
                prompts=prompts,
                tokenizer_source=args.tokenizer_source,
                hf_endpoint=args.hf_endpoint,
                hf_tokenizer_id=args.hf_tokenizer_id,
            )
            base_weights_path = cache_artifacts["base_weights_path"]
            manifest_path = cache_artifacts["manifest_path"]
            draft_triton_path = (
                Path(args.draft_triton_path).expanduser().resolve()
                if args.draft_triton_path is not None
                else Path(args.cache_dir).expanduser().resolve() / "draft_triton.pkl"
            )
            draft_triton_path = _triton_runtime.ensure_spec_draft_checkpoint(
                draft_checkpoint_path=args.draft_checkpoint,
                output_path=draft_triton_path,
            )
        runtime_backend = _triton_runtime.SpecTritonRuntimePool(
            base_weights_path=base_weights_path,
            manifest_path=manifest_path,
            draft_checkpoint_path=draft_triton_path,
            num_views=int(args.num_views),
            chunk_size=int(train_config.model.action_horizon),
            tokenizer_source=args.tokenizer_source,
            hf_endpoint=args.hf_endpoint,
            hf_tokenizer_id=args.hf_tokenizer_id,
            ablation_layer_pattern=action_layer_pattern,
            ablation_flow_pattern=legacy_flow_pattern,
            vlm_layer_pattern=vlm_layer_pattern,
            denoise_steps=denoise_steps,
        )

        def _build_missing_prompt(prompt: str) -> None:
            if args.base_triton_path is not None:
                _triton_runtime.build_prompt_cache_from_base(
                    base_weights_path=base_weights_path,
                    cache_dir=Path(manifest_path).parent,
                    prompts=[prompt],
                    tokenizer_source=args.tokenizer_source,
                    hf_endpoint=args.hf_endpoint,
                    hf_tokenizer_id=args.hf_tokenizer_id,
                )
                return
            _triton_runtime.build_prompt_cache(
                jax_checkpoint_dir=args.jax_checkpoint_dir,
                cache_dir=args.cache_dir,
                prompts=[prompt],
                tokenizer_source=args.tokenizer_source,
                hf_endpoint=args.hf_endpoint,
                hf_tokenizer_id=args.hf_tokenizer_id,
            )

        prompt_cache_builder = _build_missing_prompt

    runtime_pool = _triton_runtime.SpecTritonPolicyRuntime(
        runtime_pool=runtime_backend,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
        max_exec_steps=int(args.max_exec_steps),
        device=str(device),
        draft_history_len=int(args.draft_history_len),
        t_list=tuple(float(x) for x in args.t_list),
        tau_radius=float(args.tau_radius),
        dist_dims=int(args.dist_dims),
        gripper_switch_threshold=float(args.gripper_switch_threshold),
        enable_gripper_verify=bool(args.enable_gripper_verify),
        enable_gripper_post_verify=bool(args.enable_gripper_post_verify),
        gripper_full_window=int(args.gripper_full_window),
        full_fallback=bool(args.full_fallback),
        force_full_each_round=bool(args.force_full_each_round),
        periodic_full_every_n_draft_rounds=int(args.periodic_full_every_n_draft_rounds),
        compiled_encoder_runtime=compiled_spec_runtime if backend == "compiled" else None,
        compiled_draft_runtime=compiled_spec_runtime if backend == "compiled" else None,
        compiled_verify_runtime=compiled_spec_runtime if backend == "compiled" else None,
    )

    policy = TritonServerPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        metadata=train_config.policy_metadata,
        runtime_pool=runtime_pool,
        action_horizon=int(train_config.model.action_horizon),
        action_dim=int(train_config.model.action_dim),
        max_exec_steps=int(args.max_exec_steps),
        pytorch_device=str(device),
        prompt_cache_builder=prompt_cache_builder,
        draft_history_len=int(args.draft_history_len),
        t_list=tuple(float(x) for x in args.t_list),
        tau_radius=float(args.tau_radius),
        dist_dims=int(args.dist_dims),
        gripper_switch_threshold=float(args.gripper_switch_threshold),
        enable_gripper_verify=bool(args.enable_gripper_verify),
        enable_gripper_post_verify=bool(args.enable_gripper_post_verify),
        gripper_full_window=int(args.gripper_full_window),
        full_fallback=bool(args.full_fallback),
        force_full_each_round=bool(args.force_full_each_round),
        periodic_full_every_n_draft_rounds=int(args.periodic_full_every_n_draft_rounds),
        ablation_layer_pattern=action_layer_pattern,
        ablation_flow_pattern=legacy_flow_pattern,
        ablation_noise_seed=args.ablation_noise_seed,
        vlm_layer_pattern=vlm_layer_pattern,
        action_layer_pattern=action_layer_pattern,
        denoise_steps=denoise_steps,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating Triton server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
