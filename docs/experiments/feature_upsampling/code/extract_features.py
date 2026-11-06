#!/usr/bin/env python3
"""Extract teacher (224), real-lowres (112) and shared-stem (pooled 224 stem) features.

Output: data/features_<tag>.npz with stacked [N, ...] arrays + metadata.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=288)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--tag", default="libero")
    ap.add_argument("--suites", nargs="+", default=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ap.add_argument("--tasks-per-suite", type=int, default=6)
    ap.add_argument("--demos-per-task", type=int, default=2)
    ap.add_argument("--frames-per-demo", type=int, default=6)
    ap.add_argument("--camera", default="agentview_rgb")
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    t0 = time.time()
    print("loading checkpoint...", flush=True)
    model = common.SigLip.load(dtype=args.dtype)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    recs = common.list_libero_frames(
        args.suites, args.tasks_per_suite, args.demos_per_task, args.frames_per_demo, args.camera
    )[: args.max_images]
    n = len(recs)
    print("images:", n, flush=True)

    # preprocess
    rgb224 = np.empty((n, 224, 224, 3), np.float32)
    rgb112 = np.empty((n, 112, 112, 3), np.float32)
    u224 = np.empty((n, 224, 224, 3), np.uint8)
    for i, r in enumerate(recs):
        a, b, c, _ = common.preprocess(r["image"])
        rgb224[i], rgb112[i], u224[i] = a, b, c

    # jitted applies with params as device-resident arguments
    from openpi.models import siglip as _siglip

    mod = _siglip.Module(num_classes=common.NUM_CLASSES, variant="So400m/14", pool_type="none", scan=True,
                         dtype_mm=args.dtype)
    mj = model
    params_j = jax.tree.map(jnp.asarray, model.params)
    pos8 = jnp.asarray(model.interp_pos((8, 8), "bicubic"))
    params8 = dict(params_j)
    params8["pos_embedding"] = pos8

    enc = _siglip.Encoder(depth=common.DEPTH, mlp_dim=4304, num_heads=16, scan=True, dtype_mm=args.dtype)
    enc_params = params_j["Transformer"]
    head_k, head_b = params_j["head"]["kernel"], params_j["head"]["bias"]

    @jax.jit
    def apply_full(p, x):
        return mod.apply({"params": p}, x, train=False)

    @jax.jit
    def apply_pooled_stem(stem):
        b, h, w, c = stem.shape
        x = stem.reshape(b, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4)).reshape(b, (h // 2) * (w // 2), c)
        x = (x + pos8).astype(stem.dtype)
        y, _ = enc.apply({"params": enc_params}, x, deterministic=True)
        return y, y @ head_k + head_b

    out = {
        "teacher_encoded": [],
        "teacher_logits": [],
        "stem": [],
        "student_encoded": [],
        "student_logits": [],
        "pooled_stem_encoded": [],
        "pooled_stem_logits": [],
        "student_stem": [],
    }
    bs = args.batch
    for s in range(0, n, bs):
        e = min(n, s + bs)
        x224 = jnp.asarray(rgb224[s:e])
        x112 = jnp.asarray(rgb112[s:e])
        logit_t, o224 = apply_full(params_j, x224)
        logit_s, o112 = apply_full(params8, x112)
        stem = o224["stem"]
        y_ps, logit_ps = apply_pooled_stem(stem)
        out["teacher_encoded"].append(np.asarray(o224["encoded"]))
        out["teacher_logits"].append(np.asarray(logit_t))
        out["stem"].append(np.asarray(stem))
        out["student_encoded"].append(np.asarray(o112["encoded"]))
        out["student_logits"].append(np.asarray(logit_s))
        out["student_stem"].append(np.asarray(o112["stem"]))
        out["pooled_stem_encoded"].append(np.asarray(y_ps))
        out["pooled_stem_logits"].append(np.asarray(logit_ps))
        print(f"  {e}/{n} ({time.time() - t0:.1f}s)", flush=True)

    arrays = {k: np.concatenate(v, axis=0) for k, v in out.items()}
    arrays["rgb224_u8"] = u224
    arrays["rgb112_u8"] = (rgb112 * 127.5 + 127.5).round().clip(0, 255).astype(np.uint8)
    # small pieces needed by eval (avoid reloading the 12GB checkpoint)
    arrays["head_kernel"] = np.asarray(head_k)
    arrays["head_bias"] = np.asarray(head_b)
    arrays["pos16"] = np.asarray(model.params["pos_embedding"]).reshape(16, 16, common.WIDTH)
    arrays["pos8"] = model.interp_pos((8, 8), "bicubic").reshape(8, 8, common.WIDTH)
    path = common.EXP_ROOT / "data" / f"features_{args.tag}.npz"
    np.savez(path, **arrays)

    meta = dict(
        n=n,
        camera=args.camera,
        suites=args.suites,
        checkpoint=str(common.CKPT),
        dtype=args.dtype,
        teacher_shape=list(arrays["teacher_encoded"].shape),
        student_shape=list(arrays["student_encoded"].shape),
        records=[{k: r[k] for k in ("suite", "task", "demo", "frame", "camera")} for r in recs],
        extraction_seconds=time.time() - t0,
    )
    (common.EXP_ROOT / "data" / f"meta_{args.tag}.json").write_text(json.dumps(meta, indent=2))
    print("wrote", path, {k: v.shape for k, v in arrays.items()}, flush=True)


if __name__ == "__main__":
    main()
