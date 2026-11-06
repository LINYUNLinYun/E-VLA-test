#!/usr/bin/env python3
"""Real GPU latency: batch=1, CUDA-synchronized wall clock, warmup + median/p90.

Methods:
  siglip224        full SigLIP So400m/14 at 224 (teacher)
  siglip112        full SigLIP at 112 with bicubically interpolated learned pos-emb
  epr_p2/p5/p9     Early-exact prefix + pooled 8x8 tail + residual, 224 input
  epr_p5_g12       same with a 12x12 middle grid

Every number includes the resize from the LIBERO 128x128 camera image, the GPU
data conversion, the encoder work and the upsampler, unless a row is marked as a
component. `vision_total_ms` is measured on the integrated pipeline, never summed.
`end2end_ms` is left empty (not wired into pi0.sample_actions here).
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from epr import EPRRunner


def bench(fn, args, warmup, iters):
    for _ in range(warmup):
        import jax
        jax.block_until_ready(fn(*args))
    ts = []
    import jax
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--dtype", nargs="+", default=["float32", "bfloat16"])
    ap.add_argument("--out", default="latency")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from openpi.shared.image_tools import resize_with_pad

    # LIBERO agentview frames are 128x128 uint8; content does not affect resize cost.
    rng = np.random.default_rng(0)
    u8 = jnp.asarray(rng.integers(0, 256, (1, 128, 128, 3), dtype=np.uint8))
    dev = jax.devices()[0]
    print("device:", dev.device_kind, flush=True)

    rows = []

    def resize_pre(size):
        @jax.jit
        def f(x):
            y = resize_with_pad(x, size, size)  # uint8
            return (y.astype(jnp.float32) / 255.0 * 2.0 - 1.0)
        return f

    for dtype in args.dtype:
        model = common.SigLip.load(dtype=dtype)
        params = jax.tree.map(jnp.asarray, model.params)
        from openpi.models import siglip as _siglip

        mod = _siglip.Module(num_classes=common.NUM_CLASSES, variant="So400m/14", pool_type="none",
                             scan=True, dtype_mm=dtype)
        params224 = params
        params112 = dict(params)
        params112["pos_embedding"] = jnp.asarray(model.interp_pos((8, 8), "bicubic"))

        @jax.jit
        def full224(x):
            return mod.apply({"params": params224}, x, train=False)[1]["encoded"]

        @jax.jit
        def full112(x):
            return mod.apply({"params": params112}, x, train=False)[1]["encoded"]

        resize224, resize112 = resize_pre(224), resize_pre(112)
        x128 = u8
        x224 = resize224(x128)
        x112 = resize112(x128)

        @jax.jit
        def full224_total(x):
            y = resize_with_pad(x, 224, 224).astype(jnp.float32) / 255.0 * 2.0 - 1.0
            return mod.apply({"params": params224}, y, train=False)[1]["encoded"]

        @jax.jit
        def full112_total(x):
            y = resize_with_pad(x, 112, 112).astype(jnp.float32) / 255.0 * 2.0 - 1.0
            return mod.apply({"params": params112}, y, train=False)[1]["encoded"]

        m, p90 = bench(full224, (x224,), args.warmup, args.iters)
        mt, p90t = bench(full224_total, (x128,), args.warmup, args.iters)
        rows.append(dict(method="siglip224", device=dev.device_kind, framework="jax/flax",
                         dtype=dtype, batch_size=1, warmup=args.warmup, iters=args.iters,
                         resize_ms=bench(resize224, (x128,), args.warmup, args.iters)[0],
                         patch_guide_ms="", encoder_ms=m, upsampler_ms="",
                         vision_total_ms=mt, end2end_ms="", median_ms=mt, p90_ms=p90t,
                         peak_vram_mb=""))
        print(f"{dtype} siglip224 enc={m:.2f}ms total={mt:.2f}ms p90={p90t:.2f}", flush=True)

        m, p90 = bench(full112, (x112,), args.warmup, args.iters)
        mt, p90t = bench(full112_total, (x128,), args.warmup, args.iters)
        rows.append(dict(method="siglip112", device=dev.device_kind, framework="jax/flax",
                         dtype=dtype, batch_size=1, warmup=args.warmup, iters=args.iters,
                         resize_ms=bench(resize112, (x128,), args.warmup, args.iters)[0],
                         patch_guide_ms="", encoder_ms=m, upsampler_ms="",
                         vision_total_ms=mt, end2end_ms="", median_ms=mt, p90_ms=p90t,
                         peak_vram_mb=""))
        print(f"{dtype} siglip112 enc={m:.2f}ms total={mt:.2f}ms p90={p90t:.2f}", flush=True)

        for p, g in ((2, 8), (5, 8), (9, 8), (5, 12)):
            runner = EPRRunner(params=params, prefix_blocks=p, grid=g, alpha=0.05, dtype=dtype)
            # integrated: resize + stem ... head
            @jax.jit
            def epr_total(x, alpha):
                x224_ = (resize_with_pad(x, 224, 224).astype(jnp.float32) / 255.0 * 2.0 - 1.0)
                return runner._jitted(x224_, alpha)[1]

            m, p90 = bench(epr_total, (x128, 0.05), args.warmup, args.iters)
            stages = runner.stages()
            x224f = jnp.asarray(x224)
            stem_ms = bench(stages["stem"], (x224f,), args.warmup, args.iters)[0]
            prefix_ms = bench(stages["prefix"], (x224f,), args.warmup, args.iters)[0]
            # tail/upsample components fed with fixed shapes
            l = jnp.zeros((1, 64, 1152), dtype=jnp.float32 if dtype == "float32" else jnp.bfloat16)
            tail_ms = bench(stages["tail"], (l,), args.warmup, args.iters)[0]
            sg = jnp.zeros((1, g, g, 1152), jnp.float32)
            up_ms = bench(stages["upsample"], (sg,), args.warmup, args.iters)[0]
            tok = jnp.zeros((1, 256, 1152), jnp.float32)
            head_ms = bench(stages["head"], (tok,), args.warmup, args.iters)[0]
            rows.append(dict(method=f"epr_p{p}_g{g}", device=dev.device_kind, framework="jax/flax",
                             dtype=dtype, batch_size=1, warmup=args.warmup, iters=args.iters,
                             resize_ms=bench(resize224, (x128,), args.warmup, args.iters)[0],
                             patch_guide_ms="", encoder_ms=prefix_ms + tail_ms,
                             upsampler_ms=up_ms + head_ms,
                             vision_total_ms=m, end2end_ms="", median_ms=m, p90_ms=p90,
                             peak_vram_mb=""))
            print(f"{dtype} epr p={p} g={g} total={m:.2f}ms p90={p90:.2f} | stem={stem_ms:.2f} "
                  f"prefix={prefix_ms:.2f} tail={tail_ms:.2f} up={up_ms:.2f} head={head_ms:.2f}",
                  flush=True)

        # free compiled state before the next dtype
        del mod, full224, full112, params224, params112, runner
        jax.clear_caches()

    out = common.EXP_ROOT / "results" / f"{args.out}.csv"
    fields = ["method", "device", "framework", "dtype", "batch_size", "warmup", "iters", "resize_ms",
              "patch_guide_ms", "encoder_ms", "upsampler_ms", "vision_total_ms", "end2end_ms",
              "median_ms", "p90_ms", "peak_vram_mb"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
