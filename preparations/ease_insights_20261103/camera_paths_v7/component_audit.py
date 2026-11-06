"""Exact component routes and compensation receipts for real policy calls."""
from .common import FIXED_LAYERS,validate_condition


def expected(config,full,history_steps,step):
    validate_condition(config);kind=config['component']
    skipped=list(FIXED_LAYERS) if kind is not None and not full else []
    has_older=kind is not None and not full and len(history_steps)==2
    ratio=(step-history_steps[-1])/(history_steps[-1]-history_steps[-2]) if has_older else 0.
    masks={k:skipped if k==kind else [] for k in ('attn','ffn')}
    routes=[]
    for layer in range(1,19):
        attn=layer<18 and layer not in masks['attn'];ffn=layer<18 and layer not in masks['ffn']
        routes.append(dict(layer=layer,projection='qkv' if attn or layer==18 else 'kv',attn=attn,ffn=ffn,
            patch_attn=layer in masks['attn'],patch_ffn=layer in masks['ffn']))
    return dict(skip_attn=masks['attn'],skip_ffn=masks['ffn'],
        captured={kind:list(range(1,18))} if kind is not None and full else {},
        patched=masks,ratio=ratio,has_older=has_older,routes=routes,current_observation_kv=True,
        last_layer='historical QKV-only L18',capture_source='FP32 difference of native BF16 hidden states',
        original_encoder_helper_restored=True,grad_enabled=False,selector_calls=0,reference_only=False)


def validate_components(row,config,full,history_steps=None,step=None):
    route=row['route'];history=route['history_steps'] if history_steps is None else history_steps
    step=row['step'] if step is None else step
    audit=expected(config,full,history,step)
    if (route.get('component_execution_audit')!=audit or row['timing'].get('component_execution_audit')!=audit
        or route.get('component_reference_only') is not False or row['timing'].get('component_reference_only') is not False):
        raise ValueError('Actual component execution/capture/patch/history or reference-only identity differs')
    return audit
