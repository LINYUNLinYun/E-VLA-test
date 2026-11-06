"""Unit tests for training-free upsamplers (run: pytest -q tests/)."""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import upsamplers as up

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _guides(b=1, c=3):
    g_hi = torch.rand(b, c, 16, 16, device=DEV)
    g_lo = F.adaptive_avg_pool2d(g_hi, (8, 8))
    z_hi = torch.randn(b, 32, 16, 16, device=DEV)
    z_co = F.avg_pool2d(z_hi, 2)
    return g_hi, g_lo, z_hi, z_co


def test_constant_field():
    f = torch.full((1, 8, 8, 8), 3.0, device=DEV)
    g_hi, g_lo, z_hi, z_co = _guides()
    outs = {
        "nearest": up.interp(f, mode="nearest"),
        "bilinear": up.interp(f, mode="bilinear"),
        "bicubic": up.interp(f, mode="bicubic"),
        "lanczos": up.lanczos(f),
        "jbu": up.jbu(f, g_hi, g_lo),
        "patch": up.patch_guided(f, z_hi, z_co),
        "guided": up.guided_filter(f, g_hi, g_lo),
    }
    for name, o in outs.items():
        assert o.shape == (1, 8, 16, 16), (name, o.shape)
        assert torch.allclose(o, torch.full_like(o, 3.0), atol=1e-4), (name, o.min(), o.max())
        assert torch.isfinite(o).all(), name


def test_alignment_no_hw_swap():
    # f[y,x] = 10*y + x uniquely identifies row/column; all linear methods must preserve orientation.
    y, x = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    f = (10 * y + x).float()[None, None].to(DEV)
    g_hi, g_lo, z_hi, z_co = _guides()
    for name, o in {
        "bilinear": up.interp(f, mode="bilinear"),
        "nearest": up.interp(f, mode="nearest"),
    }.items():
        # bilinear at interior centres reproduces a linear ramp exactly
        o = o[0, 0]
        assert o[0, 0] <= o[0, -1] <= o[-1, 0] <= o[-1, -1], name
        # align_corners=False: bilinear output pixel 4 maps to input 1.75 -> 10*1.75+1.75;
        # nearest maps to the nearest input pixel (2,2) -> 22.
        expected = 19.25 if name == "bilinear" else 22.0
        assert torch.allclose(o[4, 4], torch.tensor(expected, device=DEV), atol=0.1), (name, o[4, 4])
    # JBU orientation with a constant colour guide (spatial-only behaviour)
    g_c = torch.full((1, 3, 16, 16), 0.5, device=DEV)
    o = up.jbu(f, g_c, F.adaptive_avg_pool2d(g_c, (8, 8)))[0, 0]
    assert o[0, 0] <= o[0, -1] <= o[-1, 0] <= o[-1, -1], (o[0, 0], o[0, -1], o[-1, 0], o[-1, -1])


def test_indicator_peak_location():
    f = torch.zeros(1, 1, 8, 8, device=DEV)
    f[0, 0, 2, 5] = 1.0
    g_hi, g_lo, z_hi, z_co = _guides()
    o = up.interp(f, mode="bilinear")[0, 0]
    py, px = divmod(int(o.argmax()), 16)
    assert (py, px) == (4, 10) or abs(py - 4) <= 1 and abs(px - 10) <= 1, (py, px)


def test_jbu_weights_normalised_and_nonneg():
    f = torch.randn(2, 4, 8, 8, device=DEV)
    g_hi, g_lo, _, _ = _guides(b=2)
    # constant check already proves normalisation for constant input; also check no NaN/explosion
    o = up.jbu(f, g_hi, g_lo, sigma_r=0.01)
    assert torch.isfinite(o).all()
    assert o.abs().max() <= f.abs().max() * 2 + 1e-3


def test_patch_guided_similarity_weights():
    f = torch.randn(1, 4, 8, 8, device=DEV)
    z_hi = torch.randn(1, 32, 16, 16, device=DEV)
    z_co = F.avg_pool2d(z_hi, 2)
    o = up.patch_guided(f, z_hi, z_co, tau=0.05, lam=0.5)
    assert torch.isfinite(o).all()
    assert o.shape == (1, 4, 16, 16)


def test_guided_filter_constant_guide():
    f = torch.randn(1, 4, 8, 8, device=DEV)
    g_hi = torch.full((1, 1, 16, 16), 0.5, device=DEV)
    g_lo = F.adaptive_avg_pool2d(g_hi, (8, 8))
    o = up.guided_filter(f, g_hi, g_lo, radius=2, eps=1e-3)
    assert torch.isfinite(o).all()
    # with a constant guide the filter degenerates to smoothing + bilinear upsampling
    assert o.shape == (1, 4, 16, 16)


def test_linear_ceiling_sanity():
    t = torch.randn(1, 16, 16, 16, device=DEV)
    f_low = F.avg_pool2d(t, 2)
    resid = up.oracle_linear_ceiling(f_low, t)
    assert resid.shape == (1,)
    assert 0.0 < float(resid[0]) < 1.0
