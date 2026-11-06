"""Unit tests for the training-free EPR operators (no checkpoint, CPU-friendly)."""
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from epr import adaptive_avg_pool2d, pool_matrix, resize_bilinear  # noqa: E402


def test_bilinear_constant_field():
    x = np.full((2, 8, 8, 3), 0.37, np.float32)
    y = np.asarray(resize_bilinear(x, 16, 16))
    assert np.allclose(y, 0.37, atol=1e-6)


def test_bilinear_identity():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 7, 5, 4)).astype(np.float32)
    y = np.asarray(resize_bilinear(x, 7, 5))
    assert np.allclose(y, x, atol=1e-6)


def test_bilinear_alignment_matches_torch():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((2, 8, 8, 4)).astype(np.float32)
    got = np.asarray(resize_bilinear(x, 16, 16))
    ref = F.interpolate(torch.as_tensor(x).permute(0, 3, 1, 2), size=(16, 16), mode="bilinear",
                        align_corners=False).permute(0, 2, 3, 1).numpy()
    assert np.allclose(got, ref, atol=1e-6)


def test_pool_matrix_rows_sum_to_one():
    for n_in, n_out in ((16, 8), (16, 12), (16, 16), (7, 3)):
        m = np.asarray(pool_matrix(n_in, n_out))
        assert m.shape == (n_out, n_in)
        assert np.allclose(m.sum(axis=1), 1.0)


def test_adaptive_pool_matches_torch():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 16, 16, 4)).astype(np.float32)
    for g in (8, 12, 16):
        got = np.asarray(adaptive_avg_pool2d(x, g, g))
        ref = F.adaptive_avg_pool2d(torch.as_tensor(x).permute(0, 3, 1, 2), (g, g)).permute(0, 2, 3, 1).numpy()
        assert np.allclose(got, ref, atol=1e-6), (g, np.abs(got - ref).max())


def test_adaptive_pool_constant():
    x = np.full((1, 16, 16, 2), -3.0, np.float32)
    for g in (8, 12):
        assert np.allclose(np.asarray(adaptive_avg_pool2d(x, g, g)), -3.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
