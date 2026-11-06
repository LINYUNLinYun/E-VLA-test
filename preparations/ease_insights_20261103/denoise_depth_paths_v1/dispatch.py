"""Prospective observed AE dispatch for exact D2/D3/D4/D10 adapter schedules.

This is CPU-tested preparation only. No GPU numerical gate, model construction,
scientific manifest, runnable service or quality experiment is provided here.
"""
from preparations.ease_insights_20261103.module_tradeoff_v1.ae_dispatch import checked_skip


def sample_with_dispatch(adapter,dispatch,context,seed,steps,ae_skip=()):
    skip=checked_skip(ae_skip)
    if type(steps) is not int or steps not in (2,3,4,10) or (steps==10 and skip):
        raise ValueError('Full D10 keeps AE18; future Fast uses adapter Euler D2/D3/D4')
    with dispatch.use(skip) as observed:
        result=adapter.sample(context,seed,dict(kind='native',m=None,num_steps=steps),False)
    if len(result)!=6 or result[3].get('actual_NFE')!=steps:raise RuntimeError('Actual adapter NFE differs')
    kept=[l for l in range(1,19) if l not in skip]
    if (observed.attempted!=list(range(1,19))*steps or observed.executed!=kept*steps
        or observed.bypassed!=list(skip)*steps or not observed.original_restored):
        raise RuntimeError('Actual ordered AE visits differ from the adapter schedule')
    receipt=dict(actual_NFE=steps,attempted_layers_per_step=list(range(1,19)),executed_layers_per_step=kept,bypassed_layers_per_step=list(skip),
        actual_layer_calls=len(observed.executed),attempted_layer_calls=len(observed.attempted),bypassed_layer_calls=len(observed.bypassed),
        original_helper_restored=True,scope='Observed eager helper dispatch, not FLOPs, latency or quality')
    return result,receipt
