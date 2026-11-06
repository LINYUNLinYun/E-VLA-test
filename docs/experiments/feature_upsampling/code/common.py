"""Shared utilities: SigLIP loading/encoding from the pi0 checkpoint, LIBERO frame sampling, metrics."""
from __future__ import annotations

import functools
import json
import pathlib
from dataclasses import dataclass

import numpy as np

REPO = pathlib.Path("/root/nfs/embodied/benchmark")
OPENPI_ROOT = REPO / "policies/openpi"
CKPT = REPO / "shared/checkpoints/openpi/openpi-assets/checkpoints/pi0_libero/params"
LIBERO_ROOT = REPO / "LIBERO/libero/datasets"
EXP_ROOT = REPO / "docs/experiments/feature_upsampling"

WIDTH = 1152
DEPTH = 27
NUM_CLASSES = 2048


# ----------------------------------------------------------------------------- model
@dataclass
class SigLip:
    """Thin wrapper around the pi0 SigLIP So400m/14 weights (JAX, frozen)."""

    params: dict
    dtype: str = "float32"

    @classmethod
    def load(cls, dtype: str = "float32", restore_dtype=np.float32):
        import sys

        if str(OPENPI_ROOT) not in sys.path:
            sys.path.insert(0, str(OPENPI_ROOT))
        from openpi.models import model as _model

        params = _model.restore_params(CKPT, restore_type=np.ndarray, dtype=restore_dtype)
        return cls(params=params["PaliGemma"]["img"], dtype=dtype)

    # -- pieces -------------------------------------------------------------
    def _module(self):
        from openpi.models import siglip as _siglip

        return _siglip.Module(
            num_classes=NUM_CLASSES,
            variant="So400m/14",
            pool_type="none",
            scan=True,
            dtype_mm=self.dtype,
        )

    def _encoder(self):
        from openpi.models import siglip as _siglip

        return _siglip.Encoder(depth=DEPTH, mlp_dim=4304, num_heads=16, scan=True, dtype_mm=self.dtype)

    def interp_pos(self, hw: tuple[int, int], method: str = "bicubic") -> np.ndarray:
        """Interpolate the learned 16x16 position embedding to hw. Returns [1,h*w,W]."""
        import jax
        import jax.numpy as jnp

        pos = jnp.asarray(self.params["pos_embedding"]).reshape(1, 16, 16, WIDTH)
        out = jax.image.resize(pos, (1, hw[0], hw[1], WIDTH), method=method)
        return np.asarray(out.reshape(1, hw[0] * hw[1], WIDTH))

    # -- forwards -----------------------------------------------------------
    def encode(self, images: np.ndarray, resolution: int, pos_method: str = "bicubic"):
        """Full SigLIP at `resolution` px. images [B,res,res,3] in [-1,1].

        Returns dict with stem (patch embed, pre-pos), encoded (pre-projector, [B,N,1152]),
        logits (post-projector, [B,N,2048]).
        """
        import jax.numpy as jnp

        mod = self._module()
        params = dict(self.params)
        if resolution == 224:
            assert images.shape[1:3] == (224, 224)
        else:
            assert images.shape[1:3] == (resolution, resolution)
            params["pos_embedding"] = self.interp_pos((resolution // 14, resolution // 14), pos_method)
        logits, out = mod.apply({"params": params}, jnp.asarray(images), train=False)
        return {
            "stem": np.asarray(out["stem"]),
            "encoded": np.asarray(out["encoded"]),
            "logits": np.asarray(logits),
            "block_outputs": out["encoder"],
        }

    def encode_pooled_stem(self, stem_hi: np.ndarray, pos_method: str = "bicubic"):
        """Run the 27 transformer blocks on 2x2-pooled high-res patch embeddings.

        stem_hi [B,16,16,1152] -> x0 = avgpool2x2(stem_hi) + pos_8 -> blocks -> LN -> head.
        Only the transformer sees 64 tokens; the patch-embed conv already ran at 224.
        """
        import jax.numpy as jnp

        enc = self._encoder()
        x = jnp.asarray(stem_hi)
        b, h, w, c = x.shape
        x = x.reshape(b, h // 2, 2, w // 2, 2, c).mean(axis=(2, 4)).reshape(b, (h // 2) * (w // 2), c)
        # interpolated learned position embedding for the 8x8 grid
        pos8 = jnp.asarray(self.interp_pos((h // 2, w // 2), pos_method))
        x = (x + pos8).astype(jnp.float32 if self.dtype == "float32" else jnp.bfloat16)
        encoded, _ = enc.apply({"params": self.params["Transformer"]}, x, deterministic=True)
        encoded = jnp.asarray(encoded)
        kernel = jnp.asarray(self.params["head"]["kernel"])
        bias = jnp.asarray(self.params["head"]["bias"])
        logits = encoded @ kernel + bias
        return {"encoded": np.asarray(encoded), "logits": np.asarray(logits)}

    def encode_teacher(self, images224: np.ndarray):
        return self.encode(images224, 224)

    def encode_student(self, images112: np.ndarray, pos_method: str = "bicubic"):
        return self.encode(images112, 112, pos_method)


# ----------------------------------------------------------------------------- data
def list_libero_frames(suites, tasks_per_suite, demos_per_task, frames_per_demo, camera="agentview_rgb", seed=0):
    """Deterministically sample frames across LIBERO suites/tasks/demos."""
    import h5py

    rng = np.random.default_rng(seed)
    picked = []
    for suite in suites:
        suite_dir = LIBERO_ROOT / suite
        files = sorted(suite_dir.glob("*_demo.hdf5"))
        if not files:
            continue
        # keep task order stable but sample which tasks
        idx = np.arange(len(files))
        if len(idx) > tasks_per_suite:
            idx = np.sort(rng.choice(idx, tasks_per_suite, replace=False))
        for fi in idx:
            fpath = files[int(fi)]
            with h5py.File(fpath, "r") as f:
                demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
                didx = np.arange(len(demos))
                if len(didx) > demos_per_task:
                    didx = np.sort(rng.choice(didx, demos_per_task, replace=False))
                for di in didx:
                    g = f["data"][demos[int(di)]]
                    arr = g["obs"][camera]
                    n = arr.shape[0]
                    fidx = np.linspace(0, n - 1, frames_per_demo).round().astype(int)
                    for fr in fidx:
                        picked.append(
                            dict(
                                suite=suite,
                                task=fpath.stem.replace("_demo", ""),
                                demo=demos[int(di)],
                                frame=int(fr),
                                camera=camera,
                                image=np.asarray(arr[int(fr)]),
                            )
                        )
    return picked


def preprocess(img_uint8: np.ndarray):
    """Official pi0 image path: uint8 -> resize_with_pad(224) -> [-1,1]. Also returns 112 view and rgb01."""
    import jax.numpy as jnp
    import sys

    if str(OPENPI_ROOT) not in sys.path:
        sys.path.insert(0, str(OPENPI_ROOT))
    from openpi.shared.image_tools import resize_with_pad

    x = jnp.asarray(img_uint8)
    x224 = np.asarray(resize_with_pad(x, 224, 224))  # uint8
    x112 = np.asarray(resize_with_pad(x224, 112, 112))  # uint8
    rgb224 = x224.astype(np.float32) / 255.0 * 2.0 - 1.0
    rgb112 = x112.astype(np.float32) / 255.0 * 2.0 - 1.0
    return rgb224, rgb112, x224, x112


# ----------------------------------------------------------------------------- metrics
def feature_metrics(pred: np.ndarray, teacher: np.ndarray) -> dict:
    """pred/teacher: [N,D] in raster order, same layout (N=256 typically).

    Returns cos_mean / cos_median / cos_p10, rel_l2, norm_ratio, spatial_delta_ratio.
    """
    pred = np.asarray(pred, dtype=np.float64)
    teacher = np.asarray(teacher, dtype=np.float64)
    assert pred.shape == teacher.shape
    pn = np.linalg.norm(pred, axis=-1)
    tn = np.linalg.norm(teacher, axis=-1)
    cos = np.sum(pred * teacher, axis=-1) / (pn * tn + 1e-12)
    rel_l2 = np.linalg.norm(pred - teacher) / (np.linalg.norm(teacher) + 1e-8)
    norm_ratio = pn.mean() / (tn.mean() + 1e-12)
    return dict(
        cos_mean=float(cos.mean()),
        cos_median=float(np.median(cos)),
        cos_p10=float(np.percentile(cos, 10)),
        rel_l2=float(rel_l2),
        norm_ratio=float(norm_ratio),
    )


def spatial_delta(feat_hw: np.ndarray) -> float:
    """Mean absolute neighbour difference of [H,W,D] features."""
    f = np.asarray(feat_hw, dtype=np.float64)
    dx = np.abs(np.diff(f, axis=1)).mean()
    dy = np.abs(np.diff(f, axis=0)).mean()
    return float((dx + dy) / 2)


def grid_to_tokens(f: np.ndarray) -> np.ndarray:
    """[B,C,H,W] -> [B,H*W,C] raster order (row-major, same as ViT)."""
    b, c, h, w = f.shape
    return np.transpose(f, (0, 2, 3, 1)).reshape(b, h * w, c)


def tokens_to_grid(t: np.ndarray, hw=(16, 16)) -> np.ndarray:
    if t.ndim == 2:
        return tokens_to_grid(t[None], hw)[0]
    b, n, c = t.shape
    assert n == hw[0] * hw[1]
    return np.transpose(t.reshape(b, hw[0], hw[1], c), (0, 3, 1, 2))
