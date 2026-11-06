"""Deterministic ToMeSD-style A/B compression for the pi0.5 SigLIP tower.

This is an experiment-specific module added for
``docs/experiments/vision_ab_experiment_plan``.  It is inert unless
``configure(mode, keep_tokens)`` is called before the model is created.

A compresses only the attention residual branch; B compresses the MLP branch
too.  Both use one input-dependent mapping per block, computed from the block
entry tokens.  Each branch is restored to the dense layout before the full
residual is added, so the sequence length never changes across blocks.

Grid/algorithm (experiment plan section 4.2): tokens form a 16x16 grid; each
2x2 cell contributes its top-left token as one of 64 destinations, the other
192 tokens are sources.  Cosine similarities use FP32, are global (every
source against all destinations), and ties resolve to the smaller original
index.  Sources are merged by descending best score; unmerged sources keep one
group each (ascending original index), followed by the 64 destination groups
(ascending original index).  Merge averages members in FP32, destination
included.  Unmerge gathers the compact value back to every original position.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

N_TOKENS = 256
GRID = 16
CELL = 2
N_DST = (GRID // CELL) * (GRID // CELL)
N_SRC = N_TOKENS - N_DST
NORM_EPS = 1e-6

_COORDS = np.arange(N_TOKENS).reshape(GRID, GRID)
_DST_MASK = np.zeros((GRID, GRID), dtype=bool)
_DST_MASK[::CELL, ::CELL] = True
DST_INDEX = jnp.asarray(_COORDS[_DST_MASK].astype(np.int32))
SRC_INDEX = jnp.asarray(_COORDS[~_DST_MASK].astype(np.int32))
IDENTITY_INDEX = jnp.arange(N_TOKENS, dtype=jnp.int32)
assert int(DST_INDEX.shape[0]) == N_DST and int(SRC_INDEX.shape[0]) == N_SRC

CONFIG = {"mode": "off", "keep_tokens": N_TOKENS}

# Trace-time shape log used by the correctness audit (PI05_VISION_AB_TRACE=1).
TRACE: list[tuple] = []


def _record_trace(kind: str, shape, keep_tokens: int) -> None:
    if os.environ.get("PI05_VISION_AB_TRACE") != "1":
        return
    entry = (kind, tuple(int(x) for x in shape), int(keep_tokens))
    if entry not in TRACE:
        TRACE.append(entry)


def configure(mode: str, keep_tokens: int) -> None:
    """Static config; must be set before model construction/JIT tracing."""
    if mode not in ("off", "A", "B"):
        raise ValueError(f"unknown compression mode {mode!r}")
    keep_tokens = int(keep_tokens)
    if mode == "off" and keep_tokens != N_TOKENS:
        raise ValueError("mode 'off' requires keep_tokens == 256")
    if not N_DST <= keep_tokens <= N_TOKENS:
        raise ValueError(f"keep_tokens must be in [{N_DST},{N_TOKENS}]")
    CONFIG.update(mode=mode, keep_tokens=keep_tokens)


def is_active() -> bool:
    return CONFIG["mode"] != "off"


def make_mapping(x: jax.Array, keep_tokens: int) -> jax.Array:
    """Return ``g[B,256]`` mapping original tokens to compact groups.

    ``keep_tokens == 256`` is the identity plumbing check: ``g_i = i`` and no
    matching is performed.
    """
    if x.ndim != 3 or x.shape[1] != N_TOKENS:
        raise ValueError(f"expected [B,{N_TOKENS},D] block entry, got {x.shape}")
    _record_trace("mapping_input", x.shape, keep_tokens)
    b = x.shape[0]
    if keep_tokens == N_TOKENS:
        return jnp.broadcast_to(IDENTITY_INDEX, (b, N_TOKENS))
    merge_count = N_TOKENS - keep_tokens

    xf = x.astype(jnp.float32)
    norm = jnp.maximum(jnp.linalg.norm(xf, axis=-1, keepdims=True), NORM_EPS)
    xn = xf / norm
    src_feat = jnp.take(xn, SRC_INDEX, axis=1)
    dst_feat = jnp.take(xn, DST_INDEX, axis=1)
    sim = jnp.einsum("bsd,btd->bst", src_feat, dst_feat)

    # Destinations are ascending, so first-max ties pick the smaller index.
    best_dst = jnp.argmax(sim, axis=-1)
    best_score = jnp.max(sim, axis=-1)

    # Stable descending score: ties keep ascending source index.
    order = jnp.argsort(-best_score, axis=-1, stable=True)
    order_position = jnp.argsort(order, axis=-1, stable=True)
    merged = order_position < merge_count
    kept_count = N_SRC - merge_count
    kept_rank = jnp.cumsum((~merged).astype(jnp.int32), axis=-1) - 1

    src_group = jnp.where(merged, kept_count + best_dst, kept_rank)
    groups = jnp.zeros((b, N_TOKENS), dtype=jnp.int32)
    groups = groups.at[:, SRC_INDEX].set(src_group)
    groups = groups.at[:, DST_INDEX].set(kept_count + jnp.arange(N_DST, dtype=jnp.int32))
    return groups


def merge(z: jax.Array, groups: jax.Array, keep_tokens: int) -> jax.Array:
    """Group-average ``z[B,N,D]`` into ``[B,K,D]`` with FP32 accumulation."""
    b, n, d = z.shape
    if not 0 < int(keep_tokens) <= n:
        raise ValueError(f"merge expects a valid K in [1,{n}]; got {z.shape}, K={keep_tokens}")
    if groups.shape != (b, n):
        raise ValueError(f"groups {groups.shape} do not match z {z.shape}")
    _record_trace("branch_merge_input", z.shape, keep_tokens)
    _record_trace("branch_compute", (b, int(keep_tokens), d), keep_tokens)
    flat_groups = (groups + jnp.arange(b, dtype=jnp.int32)[:, None] * keep_tokens).reshape(-1)
    zf = z.astype(jnp.float32).reshape(b * n, d)
    sums = jax.ops.segment_sum(zf, flat_groups, num_segments=b * keep_tokens).reshape(b, keep_tokens, d)
    counts = jax.ops.segment_sum(
        jnp.ones((b * n,), dtype=jnp.float32), flat_groups, num_segments=b * keep_tokens
    ).reshape(b, keep_tokens)
    return (sums / counts[:, :, None]).astype(z.dtype)


def unmerge(z_short: jax.Array, groups: jax.Array) -> jax.Array:
    """Gather compact ``[B,K,D]`` values back to original ``[B,N,D]`` order."""
    b, n = groups.shape
    d = z_short.shape[-1]
    _record_trace("unmerge_output", (b, n, d), int(z_short.shape[1]))
    index = jnp.broadcast_to(groups[:, :, None], (b, n, d))
    return jnp.take_along_axis(z_short, index, axis=1)
