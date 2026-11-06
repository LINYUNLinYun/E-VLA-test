"""Training-free feature upsamplers (PyTorch).

All functions take f_low [B,C,h,w] and return [B,C,2h,2w] unless noted.
No learned parameters, no optimization unless explicitly named `per_image_*`.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- basic
def interp(f_low: torch.Tensor, size=None, mode: str = "bilinear", align_corners: bool = False):
    if size is None:
        size = (f_low.shape[-2] * 2, f_low.shape[-1] * 2)
    if mode == "nearest":
        return F.interpolate(f_low, size=size, mode="nearest")
    if mode == "bicubic":
        return F.interpolate(f_low, size=size, mode="bicubic", align_corners=align_corners)
    return F.interpolate(f_low, size=size, mode=mode, align_corners=align_corners)


def lanczos(f_low: torch.Tensor, size=None, a: int = 3):
    """Separable Lanczos resampling implemented with 1D convs (GPU, all channels)."""
    if size is None:
        size = (f_low.shape[-2] * 2, f_low.shape[-1] * 2)
    oh, ow = size
    ih, iw = f_low.shape[-2:]
    dev, dt = f_low.device, f_low.dtype

    def kernel_1d(orig, out):
        scale = out / orig
        # output pixel centre mapped to input coordinates (align_corners=False)
        pos = (torch.arange(out, device=dev, dtype=torch.float64) + 0.5) / scale - 0.5
        base = torch.floor(pos)
        taps = []
        for k in range(-a + 1, a + 1):
            d = pos - (base + k)
            x = d.abs()
            val = torch.where(
                x < 1, a * x**3 - (a + 1) * x**2 + 1,
                torch.where(x < a, a * (2 * x**3 - 9 * x**2 + 12 * x - 4) / (2 * x.clamp_min(1e-9)), torch.zeros_like(x)),
            )
            taps.append((base + k).long().clamp(0, orig - 1))
        return taps

    taps_h = kernel_1d(ih, oh)
    taps_w = kernel_1d(iw, ow)
    # weights must be recomputed (not index-only)
    def weights(orig, out):
        scale = out / orig
        pos = (torch.arange(out, device=dev, dtype=torch.float64) + 0.5) / scale - 0.5
        base = torch.floor(pos)
        ws = []
        for k in range(-a + 1, a + 1):
            x = (pos - (base + k)).abs()
            w = torch.where(
                x < 1, a * x**3 - (a + 1) * x**2 + 1,
                torch.where(x < a, a * (2 * x**3 - 9 * x**2 + 12 * x - 4) / (2 * x.clamp_min(1e-9)), torch.zeros_like(x)),
            )
            ws.append(w)
        ws = torch.stack(ws, 0)
        return ws / ws.sum(0, keepdim=True)

    wh = weights(ih, oh).to(dt)
    ww = weights(iw, ow).to(dt)
    f = f_low
    # height
    outs = []
    for wi, ti in zip(wh, taps_h):
        outs.append(f.index_select(-2, ti) * wi.view(1, 1, -1, 1))
    f = torch.stack(outs, 0).sum(0)
    outs = []
    for wi, ti in zip(ww, taps_w):
        outs.append(f.index_select(-1, ti) * wi.view(1, 1, 1, -1))
    f = torch.stack(outs, 0).sum(0)
    return f


# ----------------------------------------------------------------------------- guided helpers
def _neighbor_index(h_hi: int, w_hi: int, h_lo: int, w_lo: int, radius: int, device):
    """Indices into a (h_lo+2r)x(w_lo+2r) replicate-padded map for every high pixel/neighbour."""
    ys = (torch.arange(h_hi, device=device, dtype=torch.float64) + 0.5) * (h_lo / h_hi) - 0.5
    xs = (torch.arange(w_hi, device=device, dtype=torch.float64) + 0.5) * (w_lo / w_hi) - 0.5
    cy = torch.floor(ys + 0.5).long()  # nearest low-res centre
    cx = torch.floor(xs + 0.5).long()
    K = (2 * radius + 1) ** 2
    idx = torch.empty(h_hi, w_hi, K, dtype=torch.long, device=device)
    offs = torch.arange(-radius, radius + 1, device=device)
    k = 0
    for dy in offs:
        for dx in offs:
            qy = (cy[:, None] + dy).clamp(0, h_lo - 1)
            qx = (cx[:, None] + dx).clamp(0, w_lo - 1)
            idx[..., k] = (qy + radius) * (w_lo + 2 * radius) + (qx + radius)
            k += 1
    # spatial distance to the actual high-pixel coordinate, in low-res pixel units
    yy = ys[:, None, None]
    xx = xs[None, :, None]
    qyy = (cy[:, None].double() + offs.view(1, 1, -1)).unsqueeze(-2)  # placeholder, rebuilt below
    dist = torch.empty(h_hi, w_hi, K, device=device)
    k = 0
    for dy in offs:
        for dx in offs:
            dist[..., k] = (yy[..., 0] - (cy[:, None].double() + dy)) ** 2 + (
                xx[..., 0] - (cx[None, :].double() + dx)
            ) ** 2
            k += 1
    return idx, dist


def _pad_gather(f: torch.Tensor, idx: torch.Tensor, radius: int) -> torch.Tensor:
    """f [B,C,h,w] -> [B,C,H,W,K] gathered at low-res neighbour indices."""
    b, c = f.shape[:2]
    fpad = F.pad(f, (radius, radius, radius, radius), mode="replicate")
    flat = fpad.reshape(b, c, -1)
    out = flat[:, :, idx.reshape(-1)].reshape(b, c, *idx.shape)
    return out


def jbu(
    f_low: torch.Tensor,
    guide_hi: torch.Tensor,
    guide_lo: torch.Tensor,
    radius: int = 1,
    sigma_s: float = 1.0,
    sigma_r: float = 0.15,
    spatial_only: bool = False,
) -> torch.Tensor:
    """Joint bilateral upsampling. guide_* are [B,3,H,W] in [0,1] (or [B,1,...])."""
    b, c, h_lo, w_lo = f_low.shape
    h_hi, w_hi = guide_hi.shape[-2:]
    dev = f_low.device
    idx, dist = _neighbor_index(h_hi, w_hi, h_lo, w_lo, radius, dev)

    w = -dist / (2 * sigma_s**2)  # [H,W,K]
    if not spatial_only:
        c_hi = F.pad(guide_hi, (radius, radius, radius, radius), mode="replicate")
        c_lo = F.pad(guide_lo, (radius, radius, radius, radius), mode="replicate")
        # colour of high pixel (unpadded) and of each low neighbour
        c_hi_flat = c_hi[:, :, radius : radius + h_hi, radius : radius + w_hi].permute(0, 2, 3, 1)
        c_lo_flat = c_lo.reshape(b, c_hi.shape[1], -1)[:, :, idx.reshape(-1)].reshape(b, c_hi.shape[1], *idx.shape)
        dr = ((c_hi_flat[:, :, :, None, :] - c_lo_flat.permute(0, 2, 3, 4, 1)) ** 2).sum(-1)
        w = w[None] - dr / (2 * sigma_r**2)
    w = torch.softmax(w.reshape(b, h_hi, w_hi, -1), dim=-1) if w.dim() == 4 else torch.softmax(w, dim=-1)
    if w.dim() == 3:
        w = w[None].expand(b, -1, -1, -1)
    vals = _pad_gather(f_low, idx, radius)  # [B,C,H,W,K]
    return (vals * w[:, None]).sum(-1)


def patch_guided(
    f_low: torch.Tensor,
    z_hi: torch.Tensor,
    z_co: torch.Tensor,
    radius: int = 1,
    tau: float = 0.2,
    lam: float = 2.0,
    detach_norm: bool = True,
) -> torch.Tensor:
    """Patch-embedding-guided upsampling (doc route D). z_hi [B,Cz,16,16], z_co [B,Cz,8,8]."""
    b, c, h_lo, w_lo = f_low.shape
    h_hi, w_hi = z_hi.shape[-2:]
    dev = f_low.device
    idx, dist = _neighbor_index(h_hi, w_hi, h_lo, w_lo, radius, dev)
    if detach_norm:
        zh = F.normalize(z_hi, dim=1)
        zc = F.normalize(z_co, dim=1)
    else:
        zh = z_hi / z_hi.norm(dim=1, keepdim=True).clamp_min(1e-6)
        zc = z_co / z_co.norm(dim=1, keepdim=True).clamp_min(1e-6)
    zc_g = _pad_gather(zc, idx, radius)  # [B,Cz,H,W,K]
    cos = (zh.permute(0, 2, 3, 1)[:, :, :, None, :] * zc_g.permute(0, 2, 3, 4, 1)).sum(-1)
    score = cos / tau - lam * dist[None]
    w = torch.softmax(score, dim=-1)
    vals = _pad_gather(f_low, idx, radius)
    return (vals * w[:, None]).sum(-1)


def guided_filter(f_low: torch.Tensor, guide_hi: torch.Tensor, guide_lo: torch.Tensor, radius: int = 2, eps: float = 1e-3):
    """Fast guided filter with a single-channel (luma) guide. Upsample coefficients bilinearly."""
    b, c, h_lo, w_lo = f_low.shape
    h_hi, w_hi = guide_hi.shape[-2:]
    I = guide_lo if guide_lo.shape[1] == 1 else guide_lo.mean(1, keepdim=True)
    p = f_low
    k = 2 * radius + 1
    ker = torch.ones(1, 1, k, k, device=f_low.device, dtype=f_low.dtype) / (k * k)

    def box(x):
        return F.conv2d(F.pad(x, (radius, radius, radius, radius), mode="reflect"), ker, groups=1)

    # channel-wise box means: group conv over C channels for p
    ker_c = ker.repeat(c, 1, 1, 1)
    mean_I = box(I)
    mean_p = F.conv2d(F.pad(p, (radius, radius, radius, radius), mode="reflect"), ker_c, groups=c)
    mean_Ip = F.conv2d(F.pad(p * I, (radius, radius, radius, radius), mode="reflect"), ker_c, groups=c)
    mean_II = box(I * I)
    var_I = mean_II - mean_I * mean_I
    cov = mean_Ip - mean_p * mean_I
    a = cov / (var_I + eps)
    bb = mean_p - a * mean_I
    a_hi = F.interpolate(a, size=(h_hi, w_hi), mode="bilinear", align_corners=False)
    b_hi = F.interpolate(bb, size=(h_hi, w_hi), mode="bilinear", align_corners=False)
    I_hi = guide_hi if guide_hi.shape[1] == 1 else guide_hi.mean(1, keepdim=True)
    return a_hi * I_hi + b_hi


# ----------------------------------------------------------------------------- diagnostics
def oracle_linear_ceiling(f_low: torch.Tensor, teacher: torch.Tensor):
    """Fraction of teacher energy lying outside the per-image row-space of the low-res features."""
    # f_low [B,C,h,w] -> flat [B,C,N], teacher [B,C,H,W] -> [B,C,M]
    b, c, h, w = f_low.shape
    fl = f_low.reshape(b, c, -1)  # [B,C,64]
    t = teacher.reshape(b, c, -1)  # [B,C,256]
    # orthonormal basis of span(fl) (columns), rank <= min(C,64)
    q, _ = torch.linalg.qr(fl.float())  # [B,C,r]
    proj = q @ (q.transpose(1, 2) @ t.float())
    resid = t.float() - proj
    return (resid.norm(dim=(1, 2)) / t.float().norm(dim=(1, 2))).cpu()
