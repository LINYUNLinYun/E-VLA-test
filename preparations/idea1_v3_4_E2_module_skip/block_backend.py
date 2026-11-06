"""Reuse the published whole-block path without modifying its source."""
from preparations.idea1_v3_4_v12_compensation.backend import encoder
from .backend import _residual


def prepare_block(backend, obs, instruction, skip=(), capture=False,
                  history=None, older=None, ratio=0., reference=False):
    from scripts.spec.triton import pi0_infer as k
    original = k.transformer_encoder
    captured = {}

    def wrapped(w, b, seq, **kwargs):
        # The old encoder's capture=True only saves its historical fixed MASK.
        # Dynamic block selection needs every L1..17; trace actual native states.
        _, traces = encoder(k, w, b, seq, skip=skip, capture=False,
                               history=history, older=older, ratio=ratio,
                               reference=reference, traced=range(1,18) if capture else ())
        if capture:
            captured.update({layer:_residual(values['output'], values['input'])
                             for layer, values in traces.items()})

    k.transformer_encoder = wrapped
    try:
        c, timing = backend.prepare(obs, instruction, skip)
    finally:
        k.transformer_encoder = original
    if capture:
        c['residuals'] = captured
    return c, timing
