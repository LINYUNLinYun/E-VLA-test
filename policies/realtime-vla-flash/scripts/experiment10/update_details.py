"""Focused residual token/channel diagnostics; retain locations, not only averages."""
import numpy as np


def update_details(tensor):
    import torch
    x = tensor.float()
    if x.ndim != 2 or x.shape[0] == 0 or not torch.isfinite(x).all():
        raise ValueError('Nonempty finite [token, channel] residual required')
    token_norm = torch.linalg.vector_norm(x, dim=1)
    channel_energy = x.square().sum(dim=0)
    p = torch.quantile(token_norm, torch.tensor([.5, .9, .99], device=x.device))
    total = channel_energy.sum()
    n = max(1, int(np.ceil(.1*len(channel_energy))))
    token_energy = token_norm.square()
    token_top_count = max(1, int(np.ceil(.1*len(token_norm))))
    statistics = {'token_count': x.shape[0], 'channel_count': x.shape[1],
                  'token_norm_p50': float(p[0]), 'token_norm_p90': float(p[1]), 'token_norm_p99': float(p[2]),
                  'token_norm_p90_over_p50': float(p[1]/p[0]) if p[0] > 0 else None,
                  'top10pct_token_energy_fraction': float(token_energy.topk(token_top_count).values.sum()/total) if total > 0 else None,
                  'top10pct_channel_energy_fraction': float(channel_energy.topk(n).values.sum()/total) if total > 0 else None,
                  'all_zero': bool(total == 0)}
    return statistics, token_norm.cpu().numpy(), channel_energy.cpu().numpy()
