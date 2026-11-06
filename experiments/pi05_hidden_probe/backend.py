"""Faithful, local OpenPI JAX prefix inspection and Patchscopes decoding.

Layer 0 is the already-scaled image/text embedding; layer k is the residual
after k transformer blocks and before the final RMSNorm.  The implementation
uses upstream ``gemma.Block.apply`` and restored parameters, without modifying
OpenPI or initializing a language head.  All public arrays omit the batch axis.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np


@dataclasses.dataclass
class Prefix:
    embedded: Any  # [tokens, width], retained on the JAX device
    input_mask: np.ndarray  # [tokens]; includes original padded camera/text slots
    image_spans: dict[str, list[int]]
    text_token_ids: list[int]
    text_positions: list[int]
    prompt_text: str = ""
    text_offsets: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    task_text: str | None = None
    task_span: tuple[int, int] | None = None
    state: np.ndarray | None = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Patch:
    layer: int
    positions: list[int]
    values: np.ndarray


@dataclasses.dataclass
class ForwardResult:
    hidden_states: dict[int, np.ndarray]
    logits: np.ndarray
    normalized: np.ndarray


def _span_positions(prefix: Prefix, start: int, end: int) -> list[int]:
    if len(prefix.text_offsets) != len(prefix.text_positions):
        raise ValueError("Prefix has no verified tokenizer offset map")
    return [position for position, (a, b) in zip(prefix.text_positions, prefix.text_offsets, strict=True)
            if b > a and a < end and b > start]


def select_positions(prefix: Prefix, selector: str, object_text: str | None = None) -> list[int]:
    """Select only valid positions. Object selection returns its final subtoken.

    ``text:placeholder`` requires exactly one whole-word occurrence of X (or
    object_text) and exactly one tokenizer token, never a character offset guess.
    """
    if selector == "images" or selector.startswith("image:"):
        names = list(prefix.image_spans) if selector == "images" else [selector.split(":", 1)[1]]
        if any(name not in prefix.image_spans for name in names):
            raise ValueError(f"Unknown camera in selector {selector!r}")
        positions = [p for name in names for p in prefix.image_spans[name] if prefix.input_mask[p]]
    elif selector == "text:last":
        positions = prefix.text_positions[-1:]
    elif selector == "text:last_task":
        if prefix.task_span is None:
            raise ValueError("text:last_task requires a source task span")
        positions = _span_positions(prefix, *prefix.task_span)[-1:]
    elif selector in {"text:object", "text:placeholder"}:
        term = object_text if object_text is not None else ("X" if selector == "text:placeholder" else None)
        if not term or not term.strip():
            raise ValueError("text:object requires object_text")
        # Object names occur in the normal task, never in any appended state/template.
        search_start, search_end = prefix.task_span if selector == "text:object" and prefix.task_span else (0, len(prefix.prompt_text))
        region = prefix.prompt_text[search_start:search_end]
        hits = list(re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", region))
        if len(hits) != 1:
            raise ValueError(f"Expected one unambiguous occurrence of {term!r}; found {len(hits)}")
        start, end = hits[0].span()
        positions = _span_positions(prefix, start + search_start, end + search_start)
        if selector == "text:placeholder":
            if len(positions) != 1:
                raise ValueError(f"Placeholder {term!r} spans {len(positions)} tokens; choose a single-token placeholder")
            offset = prefix.text_offsets[prefix.text_positions.index(positions[0])]
            if prefix.prompt_text[slice(*offset)].strip() != term:
                raise ValueError("Placeholder shares a token with surrounding non-whitespace text")
        else:
            positions = positions[-1:]
    else:
        raise ValueError(f"Unknown position selector: {selector!r}")
    if not positions:
        raise ValueError(f"Selector {selector!r} selected no valid tokens")
    return list(positions)


def attention_layout(input_mask: Any, prefix_length: int):
    """Fixed bidirectional prefix followed by causal answer tokens.

    Padding stays at its original physical position; RoPE positions count only
    valid tokens, exactly as upstream pi0.make_attn_mask/sample_actions do.
    """
    import jax.numpy as jnp
    from openpi.models.pi0 import make_attn_mask

    valid = jnp.asarray(input_mask, dtype=bool)[None]
    ar = jnp.arange(valid.shape[1]) >= prefix_length
    return jnp.cumsum(valid, axis=-1) - 1, make_attn_mask(valid, ar)


def _inspect_linen(module, embedded, positions, mask, readout_position, patch_values, patch_positions,
                   capture_positions, *, capture_layers=(), patch_layer=-1):
    """Callable Linen method; ToNNX exposes the checkpoint's existing variables."""
    import jax
    import jax.numpy as jnp
    from openpi.models import gemma

    x = embedded.astype(module.embed_dtype)
    values = patch_values.astype(x.dtype)
    if patch_layer == 0:
        x = x.at[:, patch_positions, :].set(values[None])
    h0 = x[:, capture_positions, :]
    block = gemma.Block(configs=module.configs, dropout=module.dropout,
                        dropout_bdims=module.dropout_bdims, parent=None)
    depth = module.configs[0].depth
    expert_padding = [None] * (len(module.configs) - 1)

    def step(current, args):
        layer, params = args
        xs, _ = block.apply({"params": params}, [current, *expert_padding], None,
                            positions, mask[:, None], [None] * len(module.configs), True)
        current = xs[0]
        if patch_layer > 0:
            current = jax.lax.cond(layer + 1 == patch_layer,
                                   lambda a: a.at[:, patch_positions, :].set(values[None]),
                                   lambda a: a, current)
        return current, current[:, capture_positions, :]

    residual, intermediate = jax.lax.scan(
        step, x, (jnp.arange(depth), module.variables["params"]["layers"]))
    normalized = module.final_norms[0](residual, None)[0]
    logits = module.embedder.decode(normalized[:, readout_position, :])
    captures = {layer: h0 if layer == 0 else intermediate[layer - 1] for layer in capture_layers}
    return normalized, logits, captures


def _reference_linen(module, embedded, positions, mask, readout_position):
    out, _ = module([embedded, *([None] * (len(module.configs) - 1))], positions=positions, mask=mask)
    return out[0], module.embedder.decode(out[0][:, readout_position, :])


def _embedding_shape(module):
    return tuple(module.variables["params"]["embedder"]["input_embedding"].shape)


class Backend:
    """Single-example backend for the standard local JAX pi05_libero checkpoint."""

    def __init__(self, checkpoint: str, config_name: str = "pi05_libero", openpi_root: str | None = None,
                 tokenizer_path: str | None = None):
        root = pathlib.Path(openpi_root) if openpi_root else pathlib.Path(__file__).resolve().parents[2] / "policies/openpi"
        if (root / "src/openpi").is_dir():
            sys.path.insert(0, str(root / "src"))
            client = root / "packages/openpi-client/src"
            if client.is_dir():
                sys.path.insert(0, str(client))
        if config_name not in {"pi05_libero", "pi0_libero"}:
            raise ValueError("This backend supports the audited pi05_libero and pi0_libero configurations")
        checkpoint_path = pathlib.Path(checkpoint).expanduser().resolve()
        if not (checkpoint_path / "params").is_dir():
            raise ValueError("Expected a local JAX checkpoint directory containing params/")
        if (checkpoint_path / "model.safetensors").exists():
            raise ValueError("PyTorch checkpoints are unsupported; supply the original JAX checkpoint")
        from openpi import transforms
        from openpi.policies import policy_config
        from openpi.training import config

        cfg = config.get_config(config_name)
        if cfg.model.paligemma_variant != "gemma_2b" or cfg.model.action_expert_variant != "gemma_300m":
            raise ValueError("The local LIBERO config differs from the audited standard architecture")
        self._policy = policy_config.create_trained_policy(cfg, checkpoint_path)
        self._model = self._policy._model
        self._model.eval()
        self._llm = self._model.PaliGemma.llm
        self._transforms = tuple(self._policy._input_transform.transforms)
        tokenizer_transforms = [t for t in self._transforms if isinstance(t, transforms.TokenizePrompt)]
        if len(tokenizer_transforms) != 1:
            raise ValueError("Expected exactly one upstream TokenizePrompt transform")
        self._tokenize_transform = tokenizer_transforms[0]
        self._tokenizer = self._tokenize_transform.tokenizer._tokenizer
        if tokenizer_path is not None:
            import sentencepiece
            candidate = sentencepiece.SentencePieceProcessor(model_file=str(pathlib.Path(tokenizer_path).resolve()))
            if candidate.serialized_model_proto() != self._tokenizer.serialized_model_proto():
                raise ValueError("Explicit tokenizer differs from the source policy tokenizer")
        self._config = cfg
        self._checkpoint = checkpoint_path
        self._openpi_root = root.resolve()
        self._configure_runtime()
        from openpi.shared import nnx_utils
        self._embed_prefix = nnx_utils.module_jit(self._model.embed_prefix)

    def _configure_runtime(self):
        from openpi.shared import nnx_utils

        self.depth = int(self._llm.module.configs[0].depth)
        self.width = int(self._llm.module.configs[0].width)
        self.vocab_size, width = map(int, self._llm(method=_embedding_shape))
        if width != self.width or self._tokenizer.vocab_size() != self.vocab_size:
            raise ValueError("Tokenizer, tied embedding head, and model dimensions do not agree")
        # Freezes the already-loaded weights. No model or head initialization.
        self._run_llm = nnx_utils.module_jit(
            self._llm.__call__, static_argnames=("method", "capture_layers", "patch_layer"))
        self._embed_tokens = nnx_utils.module_jit(self._llm.__call__, static_argnames=("method",))

    def _token_map(self, text: str, *, bos: bool = True) -> tuple[list[int], list[tuple[int, int]]]:
        # SentencePiece's proto API does not implement add_bos; add it explicitly
        # and verify against the ordinary encoding used by the upstream tokenizer.
        proto = self._tokenizer.encode(text, out_type="immutable_proto")
        ids = [int(piece.id) for piece in proto.pieces]
        offsets = [(int(piece.begin), int(piece.end)) for piece in proto.pieces]
        if bos:
            ids.insert(0, int(self._tokenizer.bos_id()))
            offsets.insert(0, (0, 0))
        expected = list(self._tokenizer.encode(text, add_bos=bos))
        if ids != expected:
            raise ValueError("Tokenizer proto offsets do not match actual encoded token IDs")
        return ids, offsets

    @staticmethod
    def _raw_prompt(raw_obs: dict) -> str:
        prompt = raw_obs.get("prompt")
        if isinstance(prompt, np.ndarray) and prompt.shape == ():
            prompt = prompt.item()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Source observation must contain a nonempty string prompt")
        return prompt

    def prepare_source(self, raw_obs: dict) -> Prefix:
        prompt = self._raw_prompt(raw_obs)
        inputs = copy.deepcopy(raw_obs)
        tokenizer_state = None
        for transform in self._transforms:
            if transform is self._tokenize_transform:
                tokenizer_state = np.asarray(inputs["state"]).copy()
            inputs = transform(inputs)
        cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
        if self._tokenize_transform.discrete_state_input:
            discrete = np.digitize(tokenizer_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
            text = f"Task: {cleaned}, State: {' '.join(map(str, discrete))};\nAction: "
            ids, offsets = self._token_map(text)
            task_span = (6, 6 + len(cleaned))
        else:
            ids, offsets = self._token_map(cleaned)
            newline_ids, newline_offsets = self._token_map("\n", bos=False)
            ids += newline_ids
            offsets += [(a + len(cleaned), b + len(cleaned)) for a, b in newline_offsets]
            text, task_span = cleaned + "\n", (0, len(cleaned))
        valid_ids = np.asarray(inputs["tokenized_prompt"])[np.asarray(inputs["tokenized_prompt_mask"], dtype=bool)].tolist()
        if ids != valid_ids:
            raise ValueError("Source token map differs from policy tokens (possibly prompt truncation); shorten the task")
        return self._make_prefix(inputs, text, ids, offsets, task_text=cleaned, task_span=task_span, kind="source",
                                 tokenizer_state=tokenizer_state)

    def prepare_target(self, raw_obs: dict, prompt: str) -> Prefix:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Target prompt must be a nonempty string")
        inputs = copy.deepcopy(raw_obs)
        inputs["prompt"] = prompt
        for transform in self._transforms:
            if transform is not self._tokenize_transform:
                inputs = transform(inputs)
        inputs.pop("prompt", None)
        # No stripping, lowercasing, underscore replacement, source action
        # template, or automatically added newline in this independent entry.
        ids, offsets = self._token_map(prompt)
        if len(ids) > self._config.model.max_token_len:
            raise ValueError(f"Target has {len(ids)} tokens, exceeding configured max_token_len={self._config.model.max_token_len}")
        inputs["tokenized_prompt"] = np.asarray(ids, dtype=np.int32)
        inputs["tokenized_prompt_mask"] = np.ones(len(ids), dtype=bool)
        return self._make_prefix(inputs, prompt, ids, offsets, kind="target")

    def _make_prefix(self, inputs, text, ids, offsets, *, kind, task_text=None, task_span=None, tokenizer_state=None):
        import jax
        import jax.numpy as jnp
        from openpi.models import model

        batched = jax.tree.map(lambda value: jnp.asarray(value)[None], inputs)
        observation = model.preprocess_observation(None, model.Observation.from_dict(batched), train=False)
        embedded, mask, ar = getattr(self, "_embed_prefix", self._model.embed_prefix)(observation)
        if np.asarray(ar).any():
            raise ValueError("Expected an entirely bidirectional original policy prefix")
        text_length = int(observation.tokenized_prompt.shape[-1])
        total_image_tokens = int(embedded.shape[1]) - text_length
        camera_names = list(observation.images)
        if total_image_tokens % len(camera_names):
            raise ValueError("Image-token layout is not the supported equal-resolution camera layout")
        per_camera = total_image_tokens // len(camera_names)
        spans = {name: list(range(i * per_camera, (i + 1) * per_camera)) for i, name in enumerate(camera_names)}
        text_positions = (total_image_tokens + np.flatnonzero(np.asarray(observation.tokenized_prompt_mask[0]))).tolist()
        return Prefix(embedded=embedded[0], input_mask=np.asarray(mask[0], dtype=bool), image_spans=spans,
                      text_token_ids=ids, text_positions=text_positions, prompt_text=text, text_offsets=offsets,
                      task_text=task_text, task_span=task_span, state=np.asarray(observation.state[0]),
                      metadata={"kind": kind, "camera_masks": {k: bool(np.asarray(v[0])) for k, v in observation.image_masks.items()},
                                "image_shapes": {k: list(v.shape[1:]) for k, v in observation.images.items()},
                                "tokens_per_camera": per_camera,
                                "state_seen_by_tokenizer": None if tokenizer_state is None else tokenizer_state.tolist(),
                                "prefix_ar_mask": np.asarray(ar, dtype=bool).tolist()})

    select_positions = staticmethod(select_positions)

    def _validate_patch(self, prefix: Prefix, patch: Patch | None):
        if patch is None:
            return
        if not isinstance(patch.layer, (int, np.integer)) or not 0 <= patch.layer <= self.depth:
            raise ValueError(f"Patch layer must be an integer in [0, {self.depth}]")
        positions = np.asarray(patch.positions)
        if positions.ndim != 1 or not len(positions) or not np.issubdtype(positions.dtype, np.integer):
            raise ValueError("Patch positions must be a nonempty integer list")
        if len(set(positions.tolist())) != len(positions) or np.any(positions < 0) or np.any(positions >= len(prefix.input_mask)):
            raise ValueError("Patch positions must be unique and inside the fixed prefix")
        if not np.asarray(prefix.input_mask)[positions].all():
            raise ValueError("Cannot patch masked padding tokens")
        values = np.asarray(patch.values)
        if values.shape != (len(positions), self.width) or not np.isfinite(values.astype(np.float32)).all():
            raise ValueError(f"Patch values must be finite with shape {(len(positions), self.width)}")

    def _forward(self, prefix, *, capture_layers=(), capture_positions=None, patch=None, answer_ids=(), reference=False):
        import jax.numpy as jnp

        self._validate_patch(prefix, patch)
        if np.shape(prefix.embedded) != (len(prefix.input_mask), self.width):
            raise ValueError("Prefix embedding and input mask shapes disagree")
        if any(not isinstance(layer, (int, np.integer)) for layer in capture_layers):
            raise ValueError("Capture layers must be integers")
        layers = tuple(int(layer) for layer in capture_layers)
        if len(set(layers)) != len(layers) or any(layer < 0 or layer > self.depth for layer in layers):
            raise ValueError(f"Capture layers must be in [0, {self.depth}]")
        if capture_positions is None:
            # Preserve physical indexing by default, including masked camera
            # and text slots. Explicit selectors return only active positions.
            selected = np.arange(len(prefix.input_mask)) if layers else np.empty(0, dtype=np.int32)
        else:
            selected = np.asarray(capture_positions, dtype=np.int32)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= len(prefix.input_mask)):
            raise ValueError("Capture positions must be within the original prefix")
        embedded = jnp.asarray(prefix.embedded)[None]
        valid = np.asarray(prefix.input_mask, dtype=bool)
        if not valid.any():
            raise ValueError("Prefix contains no valid tokens")
        if len(answer_ids):
            answer_embeddings = self._embed_tokens(jnp.asarray(answer_ids, dtype=jnp.int32)[None], method="embed")
            embedded = jnp.concatenate((embedded, answer_embeddings), axis=1)
            valid = np.concatenate((valid, np.ones(len(answer_ids), dtype=bool)))
        positions, mask = attention_layout(valid, len(prefix.input_mask))
        readout = jnp.asarray(int(np.flatnonzero(valid)[-1]), dtype=jnp.int32)
        if reference:
            normalized, logits = self._run_llm(embedded, positions, mask, readout, method=_reference_linen)
            captures = {}
        else:
            values = np.empty((0, self.width), dtype=np.float32) if patch is None else patch.values
            locations = [] if patch is None else patch.positions
            normalized, logits, captures = self._run_llm(
                embedded, positions, mask, readout, jnp.asarray(values), jnp.asarray(locations, dtype=jnp.int32),
                jnp.asarray(selected, dtype=jnp.int32), capture_layers=layers, patch_layer=-1 if patch is None else int(patch.layer),
                method=_inspect_linen)
        return ForwardResult({k: np.asarray(v[0], dtype=np.float32) for k, v in captures.items()},
                             np.asarray(logits[0], dtype=np.float32), np.asarray(normalized[0], dtype=np.float32))

    def forward(self, prefix: Prefix, capture_layers=(), capture_positions=None, patch: Patch | None = None) -> ForwardResult:
        return self._forward(prefix, capture_layers=capture_layers, capture_positions=capture_positions, patch=patch)

    def reference_forward(self, prefix: Prefix) -> ForwardResult:
        return self._forward(prefix, reference=True)

    def generate(self, prefix: Prefix, patch: Patch | None = None, max_new_tokens: int = 16) -> dict:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        answer = []
        eos_id = int(self._tokenizer.eos_id())
        for _ in range(max_new_tokens):
            # Recompute from the unchanged prefix, applying the intervention on
            # every step. Prefix tokens can never attend to answer tokens.
            out = self._forward(prefix, patch=patch, answer_ids=answer)
            token = int(np.argmax(out.logits))
            answer.append(token)
            if token == eos_id:
                break
        return {"text": self._tokenizer.decode(answer), "token_ids": answer, "eos": bool(answer[-1] == eos_id)}

    def score_answers(self, prefix: Prefix, answers: list[str], patch: Patch | None = None,
                      length_normalize: bool = False) -> dict[str, float]:
        scores = {}
        for answer in answers:
            tokens = list(self._tokenizer.encode(answer, add_bos=False, add_eos=False))
            if not tokens:
                raise ValueError("Candidate answers must tokenize to at least one token")
            total = 0.0
            for index, token in enumerate(tokens):
                logits = self._forward(prefix, patch=patch, answer_ids=tokens[:index]).logits.astype(np.float64)
                largest = logits.max()
                log_z = largest + np.log(np.exp(logits - largest).sum())
                total += float(logits[token] - log_z)
            scores[answer] = total / len(tokens) if length_normalize else total
        return scores

    def audit(self, prefix: Prefix | None = None) -> dict:
        def git(*args):
            try:
                return subprocess.check_output(["git", "-C", str(self._openpi_root), *args], text=True,
                                               stderr=subprocess.DEVNULL, timeout=10).strip()
            except (OSError, subprocess.SubprocessError):
                return None

        tokenizer_hash = hashlib.sha256(self._tokenizer.serialized_model_proto()).hexdigest()
        module = self._llm.module
        sources = {}
        for name in ("openpi.models.gemma", "openpi.models.pi0", "openpi.models.tokenizer", "openpi.training.config"):
            filename = pathlib.Path(importlib.import_module(name).__file__)
            sources[str(filename)] = hashlib.sha256(filename.read_bytes()).hexdigest()
        cfg = self._config
        weights = self._checkpoint / "params"
        files = [{"path": str(p.relative_to(self._checkpoint)), "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for p in sorted(weights.rglob("*")) if p.is_file()]
        norm_paths = sorted((self._checkpoint / "assets").rglob("norm_stats.json"))
        norms = [str(p) for p in norm_paths]
        info = {"backend": "openpi-jax", "pi05": bool(cfg.model.pi05), "config": cfg.name, "checkpoint": str(self._checkpoint),
                "python_version": sys.version,
                "dependencies": {name: importlib.metadata.version(name) for name in
                                 ("jax", "jaxlib", "flax", "numpy", "sentencepiece", "orbax-checkpoint")},
                "weight_loading": "upstream BaseModel.load validates all required parameter names and shapes; extra checkpoint parameters may be ignored",
                "missing_required_weight_keys": [],
                "checkpoint_files": files, "openpi_root": str(self._openpi_root), "git_revision": git("rev-parse", "HEAD"),
                "git_dirty": bool(git("status", "--porcelain")), "code_sha256": sources,
                "depth": self.depth, "width": self.width, "vocab_size": self.vocab_size,
                "dtype": str(module.embed_dtype), "model_configs": [dataclasses.asdict(c) for c in module.configs],
                "head": "checkpoint embedder/input_embedding transpose (tied); no new parameters",
                "final_norm": "checkpoint final_norm; applied exactly once after layer depth",
                "layer_convention": "0=scaled embeddings; k=residual after k blocks, before final norm",
                "tokenizer_sha256": tokenizer_hash, "bos_id": int(self._tokenizer.bos_id()), "eos_id": int(self._tokenizer.eos_id()),
                "source_transforms": [type(t).__name__ for t in self._transforms], "norm_stats_files": norms,
                "norm_stats_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in norm_paths},
                "normalization_use_quantiles": [bool(t.use_quantiles) for t in self._transforms if type(t).__name__ == "Normalize"],
                "source_discrete_state_input": bool(self._tokenize_transform.discrete_state_input),
                "proprio_state_enters_prefix": bool(self._tokenize_transform.discrete_state_input),
                "target_text_contract": "BOS + exact supplied prompt; no action template, cleanup, or implicit newline",
                "generation": "greedy, full recomputation, patch reapplied each step, causal answers after fixed bidirectional prefix",
                "raw_input_keys": ["observation/image", "observation/wrist_image", "observation/state", "prompt"]}
        if prefix is not None:
            info["prefix"] = {"shape": list(prefix.embedded.shape), "input_mask": prefix.input_mask.tolist(),
                              "image_spans": prefix.image_spans, "text_token_ids": prefix.text_token_ids,
                              "text_positions": prefix.text_positions, "text_offsets": [list(x) for x in prefix.text_offsets],
                              "prompt_text": prefix.prompt_text, "task_span": prefix.task_span,
                              "state": None if prefix.state is None else prefix.state.tolist(), **prefix.metadata}
        return info
