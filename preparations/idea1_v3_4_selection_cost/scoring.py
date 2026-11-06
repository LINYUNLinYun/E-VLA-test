"""Use the existing signed projections, with deterministic legal-mask ties."""
import numpy as np
from preparations.idea1_v3_4_E2_adjacency_200.common import combinations


def score_and_choose(gradients, k, aggregation='layer_rms_sum'):
    g = np.asarray(gradients)
    if g.shape != (4, 17) or not np.isfinite(g).all():
        raise ValueError('Expected four finite signed gradients for 17 layers')
    masks = combinations(k, 'spaced')
    layer_score = np.sqrt(np.mean(np.square(g), axis=0))
    if aggregation == 'layer_rms_sum':
        scores = layer_score[masks].sum(axis=1)
    elif aggregation == 'signed_sum_rms':
        directional_sums = g[:, masks].sum(axis=2)
        scores = np.sqrt(np.mean(np.square(directional_sums), axis=0))
    else:
        raise ValueError(aggregation)
    index = int(np.argmin(scores))
    return (masks[index] + 1).tolist(), layer_score, scores


def refresh_due(completed_fulls, stride):
    if stride not in (1, 2, 3) or completed_fulls < 0:
        raise ValueError((completed_fulls, stride))
    return completed_fulls % stride == 0
