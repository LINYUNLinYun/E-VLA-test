#!/usr/bin/env python3
"""Serve one eager PI0 Attention/FFN diagnostic configuration over WebSocket.

This is a fixed-configuration diagnostic server. Run separate server instances
for a baseline and each intervention; request-key-derived noise makes their
per-infer noise identical when the request identity is identical. The eager
probe path is for diagnosis and is not a production-latency measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT / "src", REPO_ROOT / "packages/openpi-client/src", REPO_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from openpi_client.base_policy import BasePolicy  # noqa: E402


ACTION_HORIZON = 50
ACTION_DIM = 32
OUTPUT_ACTION_DIM = 7
DENOISE_STEPS = 10
ACTION_SPACE = "pi0_libero_unnormalized_7d_policy_command"

SEMANTIC_TASK_KEY = "__semantic_task_key__"
INIT_STATE_HASH_KEY = "__init_state_sha256__"
NOISE_SEED_KEY = "__noise_seed__"
INFER_ID_KEY = "__infer_id__"
EXPLICIT_NOISE_KEY = "__noise__"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _provenance_digest(provenance: dict[str, Any]) -> str:
    payload = dict(provenance)
    payload.pop("provenance_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _verify_run_provenance(provenance: dict[str, Any]) -> None:
    expected = provenance.get("provenance_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Run provenance is missing a valid provenance_sha256")
    if _provenance_digest(provenance) != expected:
        raise ValueError("Run provenance checksum verification failed")


def _file_identity(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> dict[str, Any]:
    """Resolve and stream-hash one regular file, rejecting concurrent mutation."""
    resolved = Path(path).expanduser().resolve(strict=True)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    def signature(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    digest = hashlib.sha256()
    bytes_read = 0
    with resolved.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"Provenance input is not a regular file: {resolved}")
        while True:
            chunk = stream.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        after = os.fstat(stream.fileno())
    after_path = resolved.stat()
    if signature(before) != signature(after) or signature(after) != signature(after_path):
        raise RuntimeError(f"File changed while hashing provenance input: {resolved}")
    if bytes_read != before.st_size:
        raise RuntimeError(
            f"Read {bytes_read} bytes while hashing {resolved}, expected {before.st_size}"
        )
    return {"path": str(resolved), "sha256": digest.hexdigest(), "size_bytes": bytes_read}


def _build_run_provenance(
    *,
    checkpoint_path: str | Path,
    norm_stats_path: str | Path,
    prompt_manifest_path: str | Path,
    source_script_path: str | Path,
    runtime_config: dict[str, Any],
) -> dict[str, Any]:
    """Capture immutable startup identities once, before loading model weights."""
    checkpoint = _file_identity(checkpoint_path)
    norm_stats = _file_identity(norm_stats_path)
    prompt_manifest = _file_identity(prompt_manifest_path)
    source_script = _file_identity(source_script_path)
    # Round-trip through JSON to fail early if a future config adds non-JSON data.
    runtime_config_json = json.loads(_canonical_json_bytes(runtime_config))
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["sha256"],
        "checkpoint_size_bytes": checkpoint["size_bytes"],
        "norm_stats_path": norm_stats["path"],
        "norm_stats_sha256": norm_stats["sha256"],
        "norm_stats_size_bytes": norm_stats["size_bytes"],
        "prompt_manifest_path": prompt_manifest["path"],
        "prompt_manifest_sha256": prompt_manifest["sha256"],
        "prompt_manifest_size_bytes": prompt_manifest["size_bytes"],
        "source_script_path": source_script["path"],
        "source_script_sha256": source_script["sha256"],
        "source_script_size_bytes": source_script["size_bytes"],
        "runtime_config": runtime_config_json,
    }
    provenance["provenance_sha256"] = _provenance_digest(provenance)
    _verify_run_provenance(provenance)
    return provenance


def _resolve_run_metadata_target(path: str | Path | None) -> Path | None:
    """Resolve the destination without following a possibly-existing final symlink."""
    if path is None:
        return None
    supplied = Path(path).expanduser()
    if supplied.name in ("", ".", ".."):
        raise ValueError(f"run metadata path must name a file: {path}")
    parent = supplied.parent.resolve()
    target = parent / supplied.name
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to overwrite run metadata: {target}")
    return target


def _publish_run_metadata(path: str | Path, provenance: dict[str, Any]) -> Path:
    """Verify and atomically publish JSON, failing rather than replacing any file."""
    _verify_run_provenance(provenance)
    target = _resolve_run_metadata_target(path)
    assert target is not None
    target.parent.mkdir(parents=True, exist_ok=True)
    # Recheck just before publication; os.link below also provides an atomic
    # no-replace operation if another process races this startup.
    if os.path.lexists(target):
        raise FileExistsError(f"Refusing to overwrite run metadata: {target}")

    serialized = json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        with temp_path.open("r", encoding="utf-8") as stream:
            verified = json.load(stream)
        if verified != provenance:
            raise ValueError("Run metadata read-back verification failed")
        _verify_run_provenance(verified)

        # Hard-linking a same-directory temp file is atomic and fails with
        # FileExistsError instead of replacing an existing destination.
        os.link(temp_path, target)
        temp_path.unlink()
        temp_path = None
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _normalize_layers(layers: tuple[int, ...] | list[int], *, name: str) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in layers:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} entries must be integer 1-based layer indices")
        normalized.append(int(value))
    values = tuple(normalized)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate 1-based layer indices")
    invalid = [value for value in values if value < 1 or value > 17]
    if invalid:
        raise ValueError(f"{name} indices must be in 1..17; got {invalid}")
    return tuple(sorted(values))


def _scalar_text(value: Any, *, key: str) -> str:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{key} must be a scalar string")
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    result = str(value).strip()
    if not result:
        raise ValueError(f"{key} must not be empty")
    return result


def _scalar_nonnegative_int(value: Any, *, key: str) -> int:
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"{key} must be a scalar nonnegative integer")
        value = value.reshape(-1)[0]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{key} must be a scalar nonnegative integer")
    if isinstance(value, (int, np.integer)):
        result = int(value)
    else:
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError(f"{key} must be a scalar nonnegative integer")
        result = int(numeric)
    if result < 0:
        raise ValueError(f"{key} must be a scalar nonnegative integer")
    return result


def _request_noise_identity(obs: dict[str, Any]) -> dict[str, Any]:
    required = (SEMANTIC_TASK_KEY, INIT_STATE_HASH_KEY, NOISE_SEED_KEY, INFER_ID_KEY)
    missing = [key for key in required if key not in obs]
    if missing:
        raise ValueError(
            "Request-key noise requires all fields "
            f"{', '.join(required)}; missing {', '.join(missing)}"
        )
    semantic_task_key = _scalar_text(obs[SEMANTIC_TASK_KEY], key=SEMANTIC_TASK_KEY)
    init_state_sha256 = _scalar_text(obs[INIT_STATE_HASH_KEY], key=INIT_STATE_HASH_KEY).lower()
    if len(init_state_sha256) != 64:
        raise ValueError(f"{INIT_STATE_HASH_KEY} must be a 64-character SHA-256 hex digest")
    try:
        int(init_state_sha256, 16)
    except ValueError as exc:
        raise ValueError(f"{INIT_STATE_HASH_KEY} must be a 64-character SHA-256 hex digest") from exc
    return {
        "semantic_task_key": semantic_task_key,
        "init_state_sha256": init_state_sha256,
        "noise_seed": _scalar_nonnegative_int(obs[NOISE_SEED_KEY], key=NOISE_SEED_KEY),
        "infer_id": _scalar_nonnegative_int(obs[INFER_ID_KEY], key=INFER_ID_KEY),
    }


def _generate_noise(identity: dict[str, Any], *, action_horizon: int, action_dim: int) -> np.ndarray:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    seed_bytes = hashlib.sha256(payload).digest()[:16]
    seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
    return np.random.default_rng(seed).standard_normal(
        (int(action_horizon), int(action_dim)), dtype=np.float32
    )


def _extract_noise(
    obs: dict[str, Any], *, action_horizon: int, action_dim: int,
) -> tuple[np.ndarray, str, dict[str, Any] | None]:
    if EXPLICIT_NOISE_KEY in obs:
        noise = np.asarray(obs[EXPLICIT_NOISE_KEY], dtype=np.float32)
        source = "explicit_request_noise"
        identity = None
    else:
        identity = _request_noise_identity(obs)
        noise = _generate_noise(identity, action_horizon=action_horizon, action_dim=action_dim)
        source = "request_key_deterministic"
    if noise.ndim == 3 and noise.shape[0] == 1:
        noise = noise[0]
    expected = (int(action_horizon), int(action_dim))
    if noise.shape != expected:
        raise ValueError(f"noise shape must be {expected}, got {tuple(noise.shape)}")
    if not np.isfinite(noise).all():
        raise ValueError("noise must contain only finite values")
    return np.ascontiguousarray(noise, dtype=np.float32), source, identity


def _copy_observation_tree(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_observation_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_observation_tree(item) for item in value)
    if isinstance(value, list):
        return [_copy_observation_tree(item) for item in value]
    return value


def _as_numpy_float32(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _default_input_preparer(
    *, transformed: dict[str, Any], device: str, action_horizon: int,
    action_dim: int, noise: np.ndarray,
) -> tuple[Any, Any, Any]:
    from scripts.spec.triton import triton_pi0_runtime

    return triton_pi0_runtime.prepare_triton_inputs_from_transformed(
        transformed=transformed,
        device=device,
        action_horizon=action_horizon,
        action_dim=action_dim,
        noise=noise,
    )


class SubmoduleDiagnosticPolicy(BasePolicy):
    """BasePolicy that runs exactly one configured eager Attention/FFN probe."""

    def __init__(
        self,
        *,
        input_transform: Callable[[dict[str, Any]], dict[str, Any]],
        output_transform: Callable[[dict[str, Any]], dict[str, Any]],
        runtime_pool: Any,
        action_horizon: int = ACTION_HORIZON,
        action_dim: int = ACTION_DIM,
        device: str = "cuda",
        skip_attn_layers: tuple[int, ...] | list[int] = (),
        skip_ffn_layers: tuple[int, ...] | list[int] = (),
        input_preparer: Callable[..., tuple[Any, Any, Any]] | None = None,
        denoise_steps: int = DENOISE_STEPS,
        run_provenance: dict[str, Any] | None = None,
    ) -> None:
        if int(action_horizon) != ACTION_HORIZON or int(action_dim) != ACTION_DIM:
            raise ValueError(
                f"This diagnostic is configured for noise shape ({ACTION_HORIZON}, {ACTION_DIM}); "
                f"got ({action_horizon}, {action_dim})"
            )
        if int(denoise_steps) != DENOISE_STEPS:
            raise ValueError(f"The paired diagnostic requires denoise_steps={DENOISE_STEPS}")
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._runtime_pool = runtime_pool
        self._action_horizon = int(action_horizon)
        self._action_dim = int(action_dim)
        self._device = str(device)
        self._skip_attn_layers_1based = _normalize_layers(skip_attn_layers, name="skip_attn_layers")
        self._skip_ffn_layers_1based = _normalize_layers(skip_ffn_layers, name="skip_ffn_layers")
        self._skip_attn_layers = tuple(layer - 1 for layer in self._skip_attn_layers_1based)
        self._skip_ffn_layers = tuple(layer - 1 for layer in self._skip_ffn_layers_1based)
        self._input_preparer = input_preparer or _default_input_preparer
        self._denoise_steps = int(denoise_steps)
        self._run_provenance = (
            json.loads(_canonical_json_bytes(run_provenance)) if run_provenance is not None else None
        )
        self.metadata = {
            "policy_type": "pi0_attention_ffn_eager_diagnostic",
            "eager_diagnostic": True,
            "production_latency_claim": False,
            "vlm_layer_pattern": "vlm_full18",
            "action_layer_pattern": "full18",
            "denoise_steps": self._denoise_steps,
            "skip_attn_layers": list(self._skip_attn_layers_1based),
            "skip_ffn_layers": list(self._skip_ffn_layers_1based),
            "action_space": ACTION_SPACE,
        }
        if self._run_provenance is not None:
            _verify_run_provenance(self._run_provenance)
            self.metadata["run_provenance"] = self._run_provenance

    def reset(self) -> None:
        """No-op: the diagnostic policy intentionally keeps no episode state."""

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(obs, dict):
            raise TypeError(f"Expected an observation dict, got {type(obs)!r}")
        if EXPLICIT_NOISE_KEY in obs:
            # The request-key metadata is optional with explicit noise, but when
            # present it is still validated and echoed for reproducibility.
            present = [key for key in (SEMANTIC_TASK_KEY, INIT_STATE_HASH_KEY, NOISE_SEED_KEY, INFER_ID_KEY) if key in obs]
            if present and len(present) != 4:
                raise ValueError("When explicit noise is supplied, pass all four request identity keys or none")
            identity = _request_noise_identity(obs) if present else None
            noise, noise_source, _ = _extract_noise(
                obs, action_horizon=self._action_horizon, action_dim=self._action_dim
            )
        else:
            noise, noise_source, identity = _extract_noise(
                obs, action_horizon=self._action_horizon, action_dim=self._action_dim
            )

        required_input_keys = ("observation/image", "observation/wrist_image", "observation/state", "prompt")
        missing = [key for key in required_input_keys if key not in obs]
        if missing:
            raise ValueError(f"Missing policy observation fields: {', '.join(missing)}")
        inputs = {
            key: _copy_observation_tree(obs[key])
            for key in required_input_keys
        }
        prompt = str(inputs["prompt"])
        if not prompt:
            raise ValueError("prompt must not be empty")

        # These transforms and the output unnormalization mirror collect_submodules.py.
        transformed = self._input_transform(inputs)
        images, state, prepared_noise = self._input_preparer(
            transformed=transformed,
            device=self._device,
            action_horizon=self._action_horizon,
            action_dim=self._action_dim,
            noise=noise,
        )
        session = self._runtime_pool.start_session(prompt)
        prepared = session.prepare_observation(images=images, state=state)
        raw_actions = session.run_submodule_probe(
            prepared=prepared,
            noise=prepared_noise,
            skip_attn_layers=self._skip_attn_layers,
            skip_ffn_layers=self._skip_ffn_layers,
        )
        raw_actions_np = _as_numpy_float32(raw_actions)
        if raw_actions_np.ndim == 3 and raw_actions_np.shape[0] == 1:
            raw_actions_np = raw_actions_np[0]
        expected_raw_shape = (self._action_horizon, self._action_dim)
        if raw_actions_np.shape != expected_raw_shape:
            raise ValueError(
                f"Eager probe returned raw action shape {tuple(raw_actions_np.shape)}; "
                f"expected {expected_raw_shape}"
            )
        if not np.isfinite(raw_actions_np).all():
            raise ValueError("Eager probe returned non-finite raw actions")

        transformed_output = self._output_transform(
            {
                "state": np.asarray(transformed["state"]).copy(),
                "actions": raw_actions_np.copy(),
            }
        )
        actions = np.asarray(transformed_output["actions"], dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, OUTPUT_ACTION_DIM):
            raise ValueError(
                f"pi0_libero output_transform returned actions shape {tuple(actions.shape)}; "
                f"expected ({ACTION_HORIZON}, {OUTPUT_ACTION_DIM})"
            )
        if not np.isfinite(actions).all():
            raise ValueError("pi0_libero output_transform returned non-finite actions")

        noise_hash = hashlib.sha256(noise.tobytes(order="C")).hexdigest()
        diagnostic_metadata = {
            "noise_sha256": noise_hash,
            "noise_shape": list(noise.shape),
            "noise_source": noise_source,
            "noise_identity": identity,
            "skip_attn_layers": list(self._skip_attn_layers_1based),
            "skip_ffn_layers": list(self._skip_ffn_layers_1based),
            "runtime_skip_attn_layers_zero_based": list(self._skip_attn_layers),
            "runtime_skip_ffn_layers_zero_based": list(self._skip_ffn_layers),
            "denoise_steps": self._denoise_steps,
            "action_space": ACTION_SPACE,
            "position_semantics": "not_measured_eef_xyz",
            "eager_diagnostic": True,
            "latency_note": "Eager submodule tracing/intervention path; latency is diagnostic and not production-policy latency.",
            "vlm_layer_pattern": "vlm_full18",
            "action_layer_pattern": "full18",
        }
        if self._run_provenance is not None:
            # Identity data was captured at startup; no filesystem reads or
            # hashing occur on the inference path.
            diagnostic_metadata["run_provenance"] = self._run_provenance
            diagnostic_metadata["provenance_sha256"] = self._run_provenance["provenance_sha256"]
        return {"actions": actions, "diagnostic_metadata": diagnostic_metadata}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-triton-path", type=Path, required=True)
    parser.add_argument("--prompt-manifest", type=Path, required=True)
    parser.add_argument(
        "--run-metadata-path",
        type=Path,
        default=None,
        help="Optional new-file-only JSON destination for startup checkpoint/runtime provenance.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", default="pi0_libero")
    parser.add_argument("--skip-attn-layers", type=int, nargs="*", default=[], metavar="LAYER")
    parser.add_argument("--skip-ffn-layers", type=int, nargs="*", default=[], metavar="LAYER")
    parser.add_argument("--tokenizer-source", default="auto")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--hf-tokenizer-id", default="google/paligemma-3b-pt-224")
    return parser


def create_server(args: argparse.Namespace) -> Any:
    """Build policy and standard OpenPI WebSocket server without starting it."""
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from scripts.spec import spec_serve_policy
    from scripts.spec.triton import triton_pi0_runtime

    skip_attn = _normalize_layers(args.skip_attn_layers, name="skip_attn_layers")
    skip_ffn = _normalize_layers(args.skip_ffn_layers, name="skip_ffn_layers")
    run_metadata_target = _resolve_run_metadata_target(args.run_metadata_path)
    weights_path, artifact_dir = spec_serve_policy._resolve_triton_base_artifact(args.base_triton_path)
    train_config, input_transform, output_transform = spec_serve_policy._build_policy_components(
        config_name=args.config,
        checkpoint_dir=artifact_dir,
    )
    horizon = int(train_config.model.action_horizon)
    action_dim = int(train_config.model.action_dim)
    if horizon != ACTION_HORIZON or action_dim != ACTION_DIM:
        raise ValueError(
            f"Expected pi0_libero horizon/dim {ACTION_HORIZON}/{ACTION_DIM}, got {horizon}/{action_dim}"
        )

    # spec_serve_policy loads normalization stats from checkpoint_dir/assets/
    # <asset_id>/norm_stats.json. Derive asset_id using DataConfigFactory's
    # create_base_config rule without performing another asset/download lookup.
    data_config = train_config.data
    asset_id = getattr(data_config, "asset_id", None)
    if not isinstance(asset_id, str) or not asset_id:
        assets_config = getattr(data_config, "assets", None)
        asset_id = getattr(assets_config, "asset_id", None)
    if not isinstance(asset_id, str) or not asset_id:
        repo_id = getattr(data_config, "repo_id", None)
        asset_id = repo_id if isinstance(repo_id, str) and repo_id else None
    if asset_id is None:
        raise ValueError(f"Config {args.config!r} has no resolvable data asset_id for norm-stat provenance")
    norm_stats_path = artifact_dir / "assets" / asset_id / "norm_stats.json"
    prompt_manifest_path = Path(args.prompt_manifest).expanduser().resolve(strict=True)
    runtime_config = {
        "config_name": str(args.config),
        "server_host": str(args.host),
        "server_port": int(args.port),
        "device": str(args.device),
        "runtime_backend": "triton_runtime_pool_eager_submodule_probe",
        "eager_diagnostic": True,
        "vlm_layer_pattern": "vlm_full18",
        "action_layer_pattern": "full18",
        "denoise_steps": DENOISE_STEPS,
        "action_horizon": horizon,
        "raw_action_dim": action_dim,
        "output_action_dim": OUTPUT_ACTION_DIM,
        "num_views": 2,
        "tokenizer_source": str(args.tokenizer_source),
        "hf_endpoint": str(args.hf_endpoint),
        "hf_tokenizer_id": str(args.hf_tokenizer_id),
        "action_space": ACTION_SPACE,
        "skip_attn_layers_1based": list(skip_attn),
        "skip_ffn_layers_1based": list(skip_ffn),
    }
    run_provenance = _build_run_provenance(
        checkpoint_path=weights_path,
        norm_stats_path=norm_stats_path,
        prompt_manifest_path=prompt_manifest_path,
        source_script_path=Path(__file__),
        runtime_config=runtime_config,
    )
    if run_metadata_target is not None:
        _publish_run_metadata(run_metadata_target, run_provenance)

    # TritonRuntimePool below unpickles the checkpoint; provenance is already
    # captured and, when requested, atomically published before that point.
    runtime_pool = triton_pi0_runtime.TritonRuntimePool(
        base_weights_path=weights_path,
        manifest_path=prompt_manifest_path,
        num_views=2,
        chunk_size=horizon,
        tokenizer_source=args.tokenizer_source,
        hf_endpoint=args.hf_endpoint,
        hf_tokenizer_id=args.hf_tokenizer_id,
        vlm_layer_pattern="vlm_full18",
        ablation_layer_pattern="full18",
        denoise_steps=DENOISE_STEPS,
    )
    policy = SubmoduleDiagnosticPolicy(
        input_transform=input_transform,
        output_transform=output_transform,
        runtime_pool=runtime_pool,
        action_horizon=horizon,
        action_dim=action_dim,
        device=args.device,
        skip_attn_layers=skip_attn,
        skip_ffn_layers=skip_ffn,
        denoise_steps=DENOISE_STEPS,
        run_provenance=run_provenance,
    )
    return WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    # Validate indices and the fixed eager experiment setup before loading weights.
    _normalize_layers(args.skip_attn_layers, name="skip_attn_layers")
    _normalize_layers(args.skip_ffn_layers, name="skip_ffn_layers")
    server = create_server(args)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "eager_diagnostic": True,
                "skip_attn_layers": sorted(args.skip_attn_layers),
                "skip_ffn_layers": sorted(args.skip_ffn_layers),
                "latency_note": "Diagnostic eager path; not production latency.",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
