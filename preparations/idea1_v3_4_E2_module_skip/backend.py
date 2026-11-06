"""Native A/F interventions with current-observation K/V and module residuals.

All public layer numbers are one-based L1..L17.  L18 keeps the historical
Q/K/V-only execution.  Capture observes the stored BF16 states, so residuals
are FP32 differences between the actual native states, not pre-rounding GEMM
outputs.  No skipped operator is evaluated for capture or compensation.
"""
from contextlib import contextmanager
import math
import operator
import threading

import torch
import triton
import triton.language as tl

from preparations.idea1_v3.policy import TritonBackend


LAYERS = tuple(range(1, 18))
KINDS = ("attn", "ffn")
# The released backend installs the eager encoder by replacing a module-level
# function.  Serialize this experiment's installs, including two clients whose
# requests reach the same B1 backend.  The server must also serialize sampling.
_ENCODER_LOCK = threading.RLock()


@triton.jit
def _difference(X, B, R, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    b = tl.load(B + i, i < N, other=0).to(tl.float32)
    tl.store(R + i, x - b, i < N)


@triton.jit
def _patch_hidden(X, R, S, ratio, N: tl.constexpr,
                  EXTRAP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0).to(tl.float32)
    r = tl.load(R + i, i < N, other=0).to(tl.float32)
    if EXTRAP:
        s = tl.load(S + i, i < N, other=0).to(tl.float32)
        r = r + ratio * (r - s)
    tl.store(X + i, x + r, i < N)


def _layers(values, name):
    try:
        values = tuple(values)
        result = tuple(operator.index(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requires integer layer numbers L1..L17") from exc
    if any(isinstance(v, bool) for v in values) or any(v not in LAYERS for v in result):
        raise ValueError(f"{name} requires layer numbers L1..L17; L18 is KV-only")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate layers")
    return frozenset(result)


def _capture_kinds(capture):
    if capture is True:
        return frozenset(KINDS)
    if capture is False or capture is None:
        return frozenset()
    kinds = (capture,) if isinstance(capture, str) else tuple(capture)
    if not set(kinds) <= set(KINDS):
        raise ValueError("capture must be bool or an iterable of 'attn'/'ffn'")
    return frozenset(kinds)


def module_identity(instruction, seq, shape):
    """Attach this identity to a residual bank to validate prompt and shape."""
    return dict(instruction=instruction, seq=int(seq), hidden_shape=list(shape))


def _history_payload(bank, identity):
    if bank is None:
        return None
    if not isinstance(bank, dict):
        raise ValueError("Residual bank must be a module dictionary")
    if "residuals" in bank:
        supplied = bank.get("identity", bank.get("module_identity"))
        if supplied is not None and supplied != identity:
            raise ValueError("Residual prompt/shape identity mismatch")
        bank = bank["residuals"]
    if not isinstance(bank, dict) or not set(bank) <= set(KINDS):
        raise ValueError("Residual bank must have 'attn'/'ffn' module dictionaries")
    return bank


def _validated_residual(bank, kind, layer, x):
    try:
        value = bank[kind][layer]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing {kind} residual at L{layer}") from exc
    if (not isinstance(value, torch.Tensor) or value.shape != x.shape
            or value.device != x.device or value.dtype != torch.float32
            or not value.is_contiguous()):
        raise ValueError(f"{kind} L{layer} residual must be contiguous FP32 on the current device/shape")
    return value


def _residual(x, before, reference=False):
    # CPU reference is useful for operator-route tests; native captures use the
    # same difference kernel as the previous whole-block backend.
    if reference or not x.is_cuda:
        return x.float() - before.float()
    result = torch.empty_like(x, dtype=torch.float32)
    _difference[(triton.cdiv(x.numel(), 1024),)](x, before, result, x.numel(), 1024)
    return result


def _patch(x, current, previous, ratio, reference=False):
    if reference or not x.is_cuda:
        estimate = current if previous is None else current + ratio * (current - previous)
        x.copy_((x.float() + estimate).to(x.dtype))
    else:
        _patch_hidden[(triton.cdiv(x.numel(), 1024),)](
            x, current, current if previous is None else previous, ratio,
            x.numel(), previous is not None, 1024, enable_fp_fusion=False)


class _Profiler:
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = []

    @contextmanager
    def stage(self, name, layer=None):
        if not self.enabled:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events.append((dict(stage=name, layer=layer), start, end))


def resolve_profile(meta, synchronize=False):
    """Resolve event rows after the encoder boundary, never between modules."""
    events = meta.pop("_profile_events", ())
    if synchronize and events:
        events[-1][2].synchronize()
    rows = [dict(row, ms=float(start.elapsed_time(end))) for row, start, end in events]
    if rows:
        meta["profile_ms"] = rows
        meta["capture_ms"] = sum(row["ms"] for row in rows if row["stage"].startswith("capture_"))
        meta["patch_ms"] = sum(row["ms"] for row in rows if row["stage"].startswith("patch_"))
    return meta


def encoder(k, w, b, seq, skip_attn=(), skip_ffn=(), capture=False,
            history=None, older=None, ratio=0., reference=False, traced=(),
            profile=False, identity=None):
    """Execute the native prefix and return ``(residuals, traces, metadata)``.

    ``history``/``older`` accept ``{'attn': {layer: tensor}, 'ffn': {...}}``
    or envelopes ``{'residuals': bank, 'identity': c['module_identity']}``.
    Capture only records modules actually executed; a skipped module is never
    evaluated.  A/F compensation is injected after its respective update slot,
    in Attention then FFN order.  ``profile`` adds CUDA events without per-stage
    synchronization; call :func:`resolve_profile` after the encoder boundary.
    """
    skip_attn = _layers(skip_attn, "skip_attn")
    skip_ffn = _layers(skip_ffn, "skip_ffn")
    traced = _layers(traced, "traced")
    captures = _capture_kinds(capture)
    ratio = float(ratio)
    if not math.isfinite(ratio):
        raise ValueError("Residual extrapolation ratio must be finite")
    if older is not None and history is None:
        raise ValueError("An older residual bank requires the latest residual bank")
    x = b["encoder_x"]
    if x.shape[0] != seq:
        raise ValueError("Encoder sequence length does not match the current hidden state")
    history = _history_payload(history, identity)
    older = _history_payload(older, identity)
    # Validate all requested patches before mutating the runtime.  Partial
    # histories are allowed for FFN-only runs, but every skipped module needs one.
    patches = {}
    for kind, skipped in (("attn", skip_attn), ("ffn", skip_ffn)):
        for layer in skipped:
            if history is not None:
                latest = _validated_residual(history, kind, layer, x)
                previous = None if older is None else _validated_residual(older, kind, layer, x)
                patches[kind, layer] = (latest, previous)
    if profile and not x.is_cuda:
        raise ValueError("Module profiling requires CUDA tensors")
    timer = _Profiler(profile)
    residuals = {kind: {} for kind in KINDS if kind in captures}
    traces = {}
    routes = []
    with timer.stage("projector"):
        k.layer_norm_matmul_n256_1152_2048_bias(
            b["vision_x"], w["vision_final_norm_w"], w["vision_final_norm_b"],
            w["encoder_multi_modal_projector_w"], w["encoder_multi_modal_projector_b"],
            x, b["vision_x_norm"])
    for i in range(18):
        layer = i + 1
        if layer in traced:
            traces[layer] = dict(input=x.clone())
        do_attn = i < 17 and layer not in skip_attn
        do_ffn = i < 17 and layer not in skip_ffn
        # Match the old Full's QKV route for L18; this final layer has no hidden
        # update.  Attention skip on L1..17 uses the actual KV-only kernel.
        projection = "qkv" if do_attn or i == 17 else "kv"
        with timer.stage(projection, layer):
            if projection == "qkv":
                k.rms_matmul_n_2048_2560_qkv_rope(
                    x, w["encoder_attn_qkv_w"][i], b["encoder_rope_weights"],
                    b["encoder_Q"], b["encoder_K"][i, :seq], b["encoder_V"][i, :seq],
                    b["encoder_x_norm"])
            else:
                k.rms_matmul_n_2048_2560_kv_rope(
                    x, w["encoder_attn_qkv_w"][i], b["encoder_rope_weights"],
                    b["encoder_K"][i, :seq], b["encoder_V"][i, :seq], b["encoder_x_norm"])
        routes.append(dict(layer=layer, projection=projection,
                           attn=do_attn, ffn=do_ffn,
                           patch_attn=("attn", layer) in patches,
                           patch_ffn=("ffn", layer) in patches))
        if i == 17:
            continue
        before = None
        if do_attn and "attn" in captures:
            with timer.stage("capture_attn_input", layer):
                before = x.clone()
        if do_attn:
            with timer.stage("attn", layer):
                attn = k.AttnSingleKey(b["encoder_Q"], b["encoder_K"][i, :seq],
                                     b["encoder_V"][i, :seq], 256 ** -.5)
                k.matmul_n_2048_2048_res(attn, w["encoder_attn_o_w"][i], x)
        if before is not None:
            with timer.stage("capture_attn_difference", layer):
                residuals["attn"][layer] = _residual(x, before, reference)
            del before
        if ("attn", layer) in patches:
            with timer.stage("patch_attn", layer):
                _patch(x, *patches["attn", layer], ratio, reference)
        if layer in traced:
            traces[layer]["after_attn"] = x.clone()
        before = None
        if do_ffn and "ffn" in captures:
            with timer.stage("capture_ffn_input", layer):
                before = x.clone()
        if do_ffn:
            with timer.stage("ffn", layer):
                k.rms_matmul_n_2048_16384_gate(
                    x, w["encoder_ffn_gate_w"][i], w["encoder_ffn_up_w"][i],
                    b["encoder_hidden"], b["encoder_x_norm"])
                k.matmul_n_16384_2048_res(b["encoder_hidden"], w["encoder_ffn_down_w"][i], x)
        if before is not None:
            with timer.stage("capture_ffn_difference", layer):
                residuals["ffn"][layer] = _residual(x, before, reference)
            del before
        if ("ffn", layer) in patches:
            with timer.stage("patch_ffn", layer):
                _patch(x, *patches["ffn", layer], ratio, reference)
        if layer in traced:
            traces[layer]["output"] = x.clone()
    meta = dict(skip_attn=sorted(skip_attn), skip_ffn=sorted(skip_ffn),
                captured={kind: sorted(values) for kind, values in residuals.items()},
                patched={kind: sorted(layer for knd, layer in patches if knd == kind) for kind in KINDS},
                ratio=ratio, has_older=older is not None, routes=routes,
                current_observation_kv=True, last_layer="historical QKV-only L18",
                capture_source="FP32 difference of native BF16 hidden states",
                _profile_events=timer.events)
    return residuals, traces, meta


class Backend(TritonBackend):
    """New experiment backend; the released shared encoder is never edited."""
    def prepare_modules(self, obs, instruction, skip_attn=(), skip_ffn=(),
                        capture=False, history=None, older=None, ratio=0.,
                        reference=False, traced=(), profile=False):
        from scripts.spec.triton import pi0_infer as k
        captured = {}
        traces = {}
        meta = {}
        identity = {}
        capture = _capture_kinds(capture)
        skip_attn = tuple(sorted(_layers(skip_attn, "skip_attn")))
        skip_ffn = tuple(sorted(_layers(skip_ffn, "skip_ffn")))
        traced = tuple(sorted(_layers(traced, "traced")))

        def wrapped(w, b, seq, **kwargs):
            identity.update(module_identity(instruction, seq, b["encoder_x"].shape))
            values, tr, data = encoder(
                k, w, b, seq, skip_attn, skip_ffn, capture, history, older,
                ratio, reference, traced, profile, identity)
            captured.update(values)
            traces.update(tr)
            meta.update(data)

        with _ENCODER_LOCK:
            original = k.transformer_encoder
            k.transformer_encoder = wrapped
            try:
                # The wrapper itself owns both independent masks; passing a
                # whole-layer skip to the legacy backend would hide this fact.
                c, timing = super().prepare(obs, instruction, ())
            finally:
                k.transformer_encoder = original
        resolve_profile(meta)
        c["module_identity"] = identity
        c["module_meta"] = meta
        if _capture_kinds(capture):
            c["residuals"] = captured
        if traced:
            c["traces"] = traces
        if profile:
            timing["residual_capture_ms"] = meta.get("capture_ms", 0.)
            timing["residual_patch_ms"] = meta.get("patch_ms", 0.)
        return c, timing
