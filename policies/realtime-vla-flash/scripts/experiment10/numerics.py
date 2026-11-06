"""Numerical activity and paired action metrics. No model weights are modified."""
from __future__ import annotations

import numpy as np
from scripts.experiment10.common import HORIZONS


def activity(tensor):
    import torch
    x = tensor.float()
    if not torch.isfinite(x).all():
        raise ValueError('Nonfinite activation')
    if not x.numel():
        return {'valid': False, 'reason': 'empty_modality'}
    flat = x.reshape(-1)
    rms = torch.sqrt(torch.mean(flat.square()))
    quantiles = torch.quantile(flat.abs(), torch.tensor([.5, .9, .99], device=x.device))
    values = {'valid': True, 'original_dtype': str(tensor.dtype), 'elements': x.numel(),
              'mean': float(flat.mean()), 'std': float(flat.std(unbiased=False)), 'rms': float(rms),
              'abs_p50': float(quantiles[0]), 'abs_p90': float(quantiles[1]), 'abs_p99': float(quantiles[2]),
              'max_abs': float(flat.abs().max()), 'zero_fraction': float((flat == 0).float().mean()),
              'all_zero': bool(rms == 0), 'nan_count': 0, 'inf_count': 0}
    for alpha in (.001, .01, .1):
        values['near_zero_' + str(alpha)] = float((flat.abs() <= alpha * rms).float().mean())
    energy = x.square().sum(dim=-1).reshape(-1)
    n = max(1, int(np.ceil(.1 * len(energy))))
    total = energy.sum()
    values['top10pct_token_energy_fraction'] = float(energy.topk(n).values.sum() / total) if total > 0 else None
    return values


def representation(a, b):
    import torch
    x, y = a.float(), b.float()
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError('Representation comparison requires matching [tokens, hidden]')
    norms = torch.linalg.vector_norm(x, dim=1) * torch.linalg.vector_norm(y, dim=1)
    valid = norms > 0
    cosine = (x[valid] * y[valid]).sum(dim=1) / norms[valid]
    norm_x = torch.linalg.vector_norm(x)
    result = {'cosine': float(cosine.mean()) if len(cosine) else None,
              'valid_cosine_tokens': int(valid.sum()), 'token_count': len(x),
              'relative_residual': float(torch.linalg.vector_norm(y - x) / norm_x) if norm_x > 0 else None}
    xc, yc = x - x.mean(dim=0), y - y.mean(dim=0)
    gx, gy = xc @ xc.T, yc @ yc.T
    denominator = torch.linalg.vector_norm(gx) * torch.linalg.vector_norm(gy)
    result['centered_linear_cka'] = float((gx * gy).sum() / denominator) if denominator > 0 else None
    return result


def action_metrics(candidate_raw, baseline_raw, candidate_command, baseline_command):
    raw_a, raw_b = np.asarray(candidate_raw, dtype=np.float64), np.asarray(baseline_raw, dtype=np.float64)
    a, b = np.asarray(candidate_command, dtype=np.float64), np.asarray(baseline_command, dtype=np.float64)
    if any(x.ndim != 2 or x.shape[0] < 12 or x.shape[1] < 7 or not np.isfinite(x).all()
           for x in (raw_a, raw_b, a, b)):
        raise ValueError('Need finite complete action sequences, including first12')
    rows = []
    for horizon in HORIZONS:
        diff = raw_a[:horizon, :6] - raw_b[:horizon, :6]
        command = a[:horizon] - b[:horizon]
        row = {'horizon': horizon, 'normalized_continuous_rmse': float(np.sqrt(np.mean(diff**2))),
               'translation_command_rmse': float(np.sqrt(np.mean(command[:, :3]**2))),
               'rotation_command_rmse': float(np.sqrt(np.mean(command[:, 3:6]**2))),
               'gripper_command_rmse': float(np.sqrt(np.mean(command[:, 6]**2))),
               'gripper_flip_rate': float(np.mean((a[:horizon, 6] > 0) != (b[:horizon, 6] > 0)))}
        for name, columns in [('translation', slice(0, 3)), ('rotation', slice(3, 6))]:
            error = raw_a[:horizon, columns] - raw_b[:horizon, columns]
            energy = float(np.mean(raw_b[:horizon, columns]**2))
            row[name + '_baseline_energy'] = energy
            row[name + '_near_zero_baseline'] = energy <= 1e-8
            row[name + '_normalized_nmse'] = float(np.mean(error**2) / energy) if energy > 1e-8 else None
        rows.append(row)
    return rows
