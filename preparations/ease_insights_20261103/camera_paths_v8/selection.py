"""No-model, no-training role-specific visual edge selection.

These strategies retain128 locations per camera and all nonvisual keys. They
change AE attention edges, not prefix storage or QK/V work. Their quality and
GPU arithmetic are untested. No experiment launches from this module.
"""

def choose(query,visual_keys,mode,source_layer):
    import torch
    if mode not in ('language_query','state_query'):raise ValueError('Unknown query role')
    if (query.ndim!=2 or query.shape[0]<1 or query.shape[1]!=256 or visual_keys.shape!=(512,256)
        or query.device!=visual_keys.device or query.dtype not in (torch.float32,torch.bfloat16)
        or visual_keys.dtype not in (torch.float32,torch.bfloat16)):
        raise ValueError('Need current role queries and512 native visual keys with256 channels')
    # This explicitly defined FP32 affinity is a selection heuristic; it is not
    # asserted equal to the native BF16 attention table.
    scores=torch.softmax((query.float()@visual_keys.float().T)*(256**-.5),dim=-1).mean(dim=0)
    if not torch.isfinite(scores).all():raise ValueError('Nonfinite role affinity')
    indices=[]
    for view in range(2):
        order=torch.argsort(scores[view*256:(view+1)*256],descending=True,stable=True)[:128]
        indices.append(order.sort().values+view*256)
    kept=torch.cat(indices);mask=torch.zeros(512,dtype=torch.bool,device=query.device);mask[kept]=True
    return dict(mask=mask,scores=scores,kept_visual_indices=kept.cpu().tolist(),mode=mode,
        source_layer=source_layer,query_rows=int(query.shape[0]),visual_keys_per_camera=[128,128],
        tie_rule='Lower original visual index first at equal FP32 score',selection_formula='Mean-query softmax(FP32 QK^T/sqrt256) over512visual keys',
        original_absolute_positions_retained=True,prefix_compute_reduction_claimed=False)


def language_query(context):
    prefix=context['seq'];buffers=context['runtime'].buffers;q=buffers['encoder_Q'];key=context['K']
    if prefix<=512 or q.shape!=(prefix*8,256) or key.shape[0]!=18 or key.shape[1]!=prefix:
        raise ValueError('Unexpected current VLM Q/K layout')
    return choose(q[512*8:prefix*8],key[17,:512],'language_query',18)


def state_query(query,key,prefix):
    if query.shape!=(408,256) or key.shape!=(prefix+51,256) or prefix<=512:
        raise ValueError('Unexpected first AE-layer full attention layout')
    # Native layout interleaves8 heads per token. First8 rows are the state
    # token, while the remaining400 rows are noisy action-token queries.
    return choose(query[:8],key[:512],'state_query',1)
