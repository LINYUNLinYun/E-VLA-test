"""Explicit, CPU-only initial states and intervals for partial flow sampling."""
import numpy as np


def nodes(start_time, steps):
    if not 0 < start_time <= 1 or steps not in (1, 2):
        raise ValueError('0 < start_time <= 1 and NFE in (1,2) required')
    q = [float(start_time)*(1-i/steps) for i in range(steps+1)]
    return dict(q=q, nodes=q, timesteps=q[:-1], dt=[b-a for a,b in zip(q,q[1:])], NFE=steps)


def mix(prior, seed, start_time, noise_scale):
    prior = np.asarray(prior, dtype=np.float32)
    if prior.shape != (50,32) or not np.isfinite(prior).all():
        raise ValueError('Finite H50 x D32 prior required')
    if not 0 < start_time <= 1 or not 0 <= noise_scale <= 1:
        raise ValueError('Invalid warm-start parameters')
    epsilon = np.random.default_rng(seed).standard_normal(prior.shape, dtype=np.float32)
    return ((1-start_time)*prior + noise_scale*start_time*epsilon).astype(np.float32)
