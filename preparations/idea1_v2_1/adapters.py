"""Isolated E1 inference adapters; importing this module does not import torch.

Masks use original 1-based layer numbers. This is preparation code, not evidence
of checkpoint/action parity. Run native/empty/reference/fast parity before use.
Adapters are temporary forward overrides: do not use with compile, training,
concurrent inference, or other forward overrides.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
from types import MethodType


XIAOMI_SOURCE_SHA256 = "bcf26ea6e7da0714a5a61834337504a13f28d6b43ed9c4ff0fc93bb6a43c2a38"
# Official tag URL is provenance, never fetched by the production adapter.
DB_QWEN2_SOURCE_URL = "https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/qwen2/modeling_qwen2.py"
DB_QWEN2_SOURCE_SHA256 = "a59fa06524227361fb401baf4d177124a27aec146bc01ec931235a9abbab17cb"
DB_COGACT_SOURCE_SHA256 = "ae14afc89c3d300e2b27d3cabf818b73843f2e37a7f9d2f55f08c7fb3878f523"


def source_sha256(obj):
    """Hash the complete defining source file; no remote import/download."""
    path = inspect.getsourcefile(type(obj))
    if path is None:
        raise ValueError("source file unavailable; architecture cannot be audited")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mask(skip, depth, protected=()):
    values = tuple(skip)
    if any(type(v) is not int for v in values):
        raise ValueError("mask must contain integer 1-based layer numbers")
    if len(set(values)) != len(values) or any(v < 1 or v > depth for v in values):
        raise ValueError("duplicate or out-of-range mask")
    if set(values).intersection(protected):
        raise ValueError("mask includes protected fusion layers")
    return tuple(sorted(values))


def _unmodified(module):
    if module.training:
        raise ValueError("adapter requires eval mode")
    if "forward" in vars(module) or hasattr(module, "_orig_mod"):
        raise ValueError("already overridden/compiled forward is unsupported")


@contextmanager
def _override(overrides):
    installed = []
    try:
        for layer, fn in overrides:
            _unmodified(layer)
            layer.forward = MethodType(fn, layer)
            installed.append(layer)
        yield
    finally:
        for layer in reversed(installed):
            del layer.forward


def _xiaomi_kv_shell(layer, hidden_states, position_embeddings,
                     attention_mask=None, position_ids=None, past_key_values=None,
                     use_cache=False, cache_position=None, **kwargs):
    """Exact audited K/V path; no Q, attention, output projection, or FFN."""
    if layer.training:
        raise ValueError("KV shell is inference-only")
    if past_key_values is None:
        raise ValueError("Xiaomi action KV shell requires the original cache")
    # This is intentionally the same pre-norm/layout/K-norm/RoPE/cache path as
    # the pinned modeling_mibot.py. Q is unnecessary to compute rotary K.
    attn = layer.self_attn
    normalized = layer.input_layernorm(hidden_states)
    shape = (*normalized.shape[:-1], -1, attn.head_dim)
    key = attn.k_norm(attn.k_proj(normalized).view(shape)).transpose(1, 2)
    value = attn.v_proj(normalized).view(shape).transpose(1, 2)
    cos, sin = position_embeddings
    rotary = type(attn).forward.__globals__["apply_rotary_pos_emb"]
    _, key = rotary(key, key, cos, sin)
    past_key_values.update(key, value, attn.layer_idx,
                           {"sin": sin, "cos": cos, "cache_position": cache_position})
    return hidden_states


@contextmanager
def xiaomi_hidden_bypass(model, skip=(), *, reference=False):
    """Adapt the pinned MiBoT 36-layer / 16-DiT architecture, in-place temporarily.

    L1..3 remain intact. Every skipped layer retains a KV shell, including early
    layers, to avoid changing DynamicCache allocation/index/length behavior.
    Consequently fully_eliminated_layers is empty. reference=True executes the
    original block then discards its hidden update, retaining its original KV.
    The model's visual fusion loop, final norm, and action head are untouched.
    """
    _unmodified(model)
    if type(model).__name__ != "MiBoTForActionGeneration" or source_sha256(model) != XIAOMI_SOURCE_SHA256:
        raise ValueError("unsupported Xiaomi model source; audit before adapting")
    text = model.vlm.model.language_model
    if len(text.layers) != 36 or len(model.dit.layers) != 16:
        raise ValueError("expected original 36 VLM / 16 DiT layers")
    if len(model.vlm.model.visual.deepstack_visual_indexes) != 3:
        raise ValueError("expected exactly three DeepStack injections")
    _unmodified(text)
    for i, layer in enumerate(text.layers):
        _unmodified(layer)
        if (type(layer).__name__ != "Qwen3VLTextDecoderLayer"
                or source_sha256(layer) != XIAOMI_SOURCE_SHA256
                or type(layer.self_attn).__name__ != "Qwen3VLTextAttention"
                or source_sha256(layer.self_attn) != XIAOMI_SOURCE_SHA256
                or layer.self_attn.layer_idx != i):
            raise ValueError("unexpected decoder architecture or original KV index")
        _unmodified(layer.self_attn)
    selected = _mask(skip, 36, (1, 2, 3))
    overrides = []
    for number in selected:
        layer = text.layers[number - 1]
        original = layer.forward
        if reference:
            def bypass(self, hidden_states, *args, _original=original, **kwargs):
                bound = inspect.signature(_original).bind(hidden_states, *args, **kwargs)
                if self.training or bound.arguments.get("past_key_values") is None:
                    raise ValueError("Xiaomi reference requires eval mode and the original cache")
                _original(hidden_states, *args, **kwargs)
                return hidden_states
        else:
            bypass = _xiaomi_kv_shell
        overrides.append((layer, bypass))
    audit = {"skipped_hidden_update_layers": list(selected),
             "fully_eliminated_layers": [], "kv_shell_layers": [] if reference else list(selected),
             "reference_discard_layers": list(selected) if reference else [],
             "reference": reference, "source_sha256": XIAOMI_SOURCE_SHA256,
             "retained": ["all original cache indices", "DeepStack L1..3", "final norm", "action DiT"]}
    with _override(overrides):
        yield audit


def _db_qwen2_kv_shell(layer, hidden_states, attention_mask=None,
                        position_ids=None, past_key_values=None, use_cache=False,
                        cache_position=None, position_embeddings=None, **kwargs):
    """4.57.6 Qwen2 K/V path without Q, attention, O or hidden FFN updates."""
    if layer.training or past_key_values is None or not use_cache:
        raise ValueError("DB adapter requires eval mode and native use_cache=True cache")
    if kwargs.get("output_attentions") or "past_key_value" in kwargs:
        raise ValueError("attention outputs/deprecated cache aliases are unsupported")
    if position_embeddings is None:
        raise ValueError("expected original shared Qwen2 rotary position embeddings")
    attn = layer.self_attn
    normalized = layer.input_layernorm(hidden_states)
    shape = (*normalized.shape[:-1], -1, attn.head_dim)
    key = attn.k_proj(normalized).view(shape).transpose(1, 2)
    value = attn.v_proj(normalized).view(shape).transpose(1, 2)
    cos, sin = position_embeddings
    # Qwen2.forward is wrapped by deprecate_kwarg; its wrapper globals belong
    # to utils.deprecation rather than modeling_qwen2.
    rotary = inspect.unwrap(type(attn).forward).__globals__["apply_rotary_pos_emb"]
    _, key = rotary(key, key, cos, sin)
    past_key_values.update(key, value, attn.layer_idx,
                           {"sin": sin, "cos": cos, "cache_position": cache_position})
    return hidden_states


@contextmanager
def db_cogact_hidden_bypass(model, skip=(), *, reference=False):
    """Dedicated DB-CogACT + exact transformers 4.57.6 Qwen2Model adapter.

    The pinned DB inference_action consumes final normalized hidden[:, -1, :].
    Its native use_cache=True remains intact; all skipped layers retain original
    K/V computation and original cache indices. This also preserves append cache
    semantics, although the native action path starts a fresh prefill each call.
    No vision/action head/protocol setting changes. Real-checkpoint/action parity
    remains a prerequisite, separately from source and CPU arithmetic tests.
    """
    _unmodified(model)
    if (type(model).__name__ != "CogACTForCausalLM"
            or source_sha256(model) != DB_COGACT_SOURCE_SHA256):
        raise ValueError("unsupported DB-CogACT readout source")
    _unmodified(model.model)
    if (type(model.model).__name__ != "CogActModel"
            or source_sha256(model.model) != DB_COGACT_SOURCE_SHA256):
        raise ValueError("unsupported DB-CogACT multimodal container source")
    text = model.model.llm
    _unmodified(text)
    if (type(text).__name__ != "Qwen2Model"
            or source_sha256(text) != DB_QWEN2_SOURCE_SHA256
            or len(text.layers) != 28 or text.config.num_hidden_layers != 28):
        raise ValueError("expected pinned 4.57.6 Qwen2Model with 28 original layers")
    for i, layer in enumerate(text.layers):
        _unmodified(layer)
        attn = layer.self_attn
        _unmodified(attn)
        if (type(layer).__name__ != "Qwen2DecoderLayer"
                or source_sha256(layer) != DB_QWEN2_SOURCE_SHA256
                or type(attn).__name__ != "Qwen2Attention"
                or source_sha256(attn) != DB_QWEN2_SOURCE_SHA256
                or attn.layer_idx != i):
            raise ValueError("unexpected DB decoder architecture or original KV index")
    selected = _mask(skip, 28, (1, 2))
    overrides = []
    for number in selected:
        layer = text.layers[number - 1]
        original = layer.forward
        if reference:
            def bypass(self, hidden_states, *args, _original=original, **kwargs):
                bound = inspect.signature(_original).bind(hidden_states, *args, **kwargs)
                bound.apply_defaults()
                if (self.training or bound.arguments.get("past_key_values") is None
                        or not bound.arguments.get("use_cache")):
                    raise ValueError("DB reference requires eval mode and native use_cache=True cache")
                if bound.arguments.get("kwargs", {}).get("output_attentions"):
                    raise ValueError("attention outputs are unsupported")
                _original(hidden_states, *args, **kwargs)
                return hidden_states
        else:
            bypass = _db_qwen2_kv_shell
        overrides.append((layer, bypass))
    audit = {"skipped_hidden_update_layers": list(selected),
             "fully_eliminated_layers": [], "kv_shell_layers": [] if reference else list(selected),
             "reference_discard_layers": list(selected) if reference else [],
             "reference": reference, "source_sha256": DB_QWEN2_SOURCE_SHA256,
             "readout_source_sha256": DB_COGACT_SOURCE_SHA256,
             "multimodal_base_source_sha256": next(
                 (hashlib.sha256(Path(inspect.getsourcefile(base)).read_bytes()).hexdigest()
                  for base in type(model.model).__mro__ if base.__name__ == "DexboticVLMModel"), None),
             "upstream_source_url": DB_QWEN2_SOURCE_URL,
             "retained": ["native use_cache=True", "all original cache indices", "final norm",
                          "last-token cognition readout", "vision tower", "action diffusion"]}
    with _override(overrides):
        yield audit


@dataclass(frozen=True)
class HiddenLayerContract:
    """Explicit, caller-audited protocol for a cache-free decoder.

    Use db_cogact_hidden_bypass for the supported native cached DB action path.
    File hashes include unrelated source intentionally.
    return_kind is 'tensor' or 'tuple' (a single hidden-state element).
    """
    class_name: str
    source_sha256: str
    depth: int
    return_kind: str
    protected: tuple = (1, 2)


@contextmanager
def hidden_update_bypass(layers, skip, contract, *, reference=False):
    """Generic audited cache-free bypass; fail closed on unknown signatures.

    This helper does not modify a model's outer loop/readout or switch caching.
    The caller must verify those semantics. Active cache/use_cache/attention
    requests are rejected, because bypass would otherwise invent a cache/output
    protocol. Empty masks still validate architecture and make no override.
    """
    if len(layers) != contract.depth or contract.return_kind not in ("tensor", "tuple"):
        raise ValueError("invalid depth/return contract")
    selected = _mask(skip, len(layers), contract.protected)
    for layer in layers:
        _unmodified(layer)
        if type(layer).__name__ != contract.class_name or source_sha256(layer) != contract.source_sha256:
            raise ValueError("decoder does not match audited contract")
        if "hidden_states" not in inspect.signature(layer.forward).parameters:
            raise ValueError("decoder lacks explicit hidden_states argument")
    overrides = []
    for number in selected:
        layer = layers[number - 1]
        original, signature = layer.forward, inspect.signature(layer.forward)
        def bypass(self, *args, _original=original, _signature=signature, **kwargs):
            if self.training:
                raise ValueError("bypass is inference-only")
            bound = _signature.bind(*args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)
            for parameter in _signature.parameters.values():
                if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                    params.update(params.pop(parameter.name, {}))
            if any(params.get(k) is not None for k in ("past_key_value", "past_key_values")):
                raise ValueError("generic bypass does not define a cache protocol")
            if params.get("use_cache") or params.get("output_attentions"):
                raise ValueError("cache/attention outputs require a dedicated adapter")
            hidden = params["hidden_states"]
            if reference:
                _original(*args, **kwargs)
            return hidden if contract.return_kind == "tensor" else (hidden,)
        overrides.append((layer, bypass))
    with _override(overrides):
        yield {"skipped_hidden_update_layers": list(selected),
               "fully_eliminated_layers": [] if reference else list(selected), "kv_shell_layers": [],
               "reference_discard_layers": list(selected) if reference else [],
               "reference": reference, "source_sha256": contract.source_sha256}


@contextmanager
def layer_probes(layers, token_masks, records):
    """Append per-call CPU scalars by explicit [batch, sequence] bool masks.

    Padding must be excluded by the caller from every token mask. Layers use
    tensor or tuple-first hidden outputs; submodule probes require explicitly
    named self_attn/mlp. Hooks are removed even on failure; no tensors retained.
    Attn/FFN values measure branch output magnitude relative to branch input.
    This calibrates Full runs, not latency measurement.
    """
    import torch
    import torch.nn.functional as F
    handles = []
    def hook(number, kind):
        def record(module, args, kwargs, output):
            before = args[0] if args else kwargs.get("hidden_states")
            after = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(before) or not torch.is_tensor(after) or before.shape != after.shape:
                raise ValueError("probe requires matching hidden tensors")
            before, after = before.detach().float(), after.detach().float()
            for group, mask in token_masks.items():
                if mask.dtype != torch.bool or tuple(mask.shape) != tuple(before.shape[:-1]):
                    raise ValueError("token masks must be bool [batch, sequence], without padding")
                mask = mask.to(before.device)
                count = int(mask.sum().item())
                row = {"layer": number, "kind": kind, "tokens": group, "count": count}
                if count:
                    a, b = before[mask], after[mask]
                    magnitude = b - a if kind == "hidden" else b
                    row.update(cosine=float(F.cosine_similarity(a, b, dim=-1).mean().item()),
                               update_rms=float(magnitude.square().mean().sqrt().item()),
                               input_rms=float(a.square().mean().sqrt().item()))
                records.append(row)
        return record
    try:
        for number, layer in enumerate(layers, 1):
            handles.append(layer.register_forward_hook(hook(number, "hidden"), with_kwargs=True))
            for name, kind in (("self_attn", "attn"), ("mlp", "ffn")):
                module = getattr(layer, name, None)
                if module is not None:
                    handles.append(module.register_forward_hook(hook(number, kind), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
