"""CPU-derived features with reused float32 casts and per-layer norms.

The arithmetic definitions and output arrays match the original host pooling.
This changes neither the native model nor stored full activation tensors.
"""
import numpy as np


def pool(result):
    pooled,stats={},{}
    names=list(result['vision'])
    sizes=[result['vision'][name]['projected'].shape[1] for name in names]
    prefix=np.asarray(result['prefix_mask'][0],bool)
    groups={};begin=0
    for name,size in zip(names,sizes):
        group=np.zeros_like(prefix);group[begin:begin+size]=prefix[begin:begin+size]
        if group.any():
            groups[name]=group
            for kind,array in result['vision'][name].items():
                value=np.asarray(array,dtype=np.float32)
                pooled[f'vision/{name}/{kind}']=(value.mean(axis=-2).squeeze(-2)
                    if value.ndim==4 else value.mean(axis=-2))
        begin+=size
    visual=np.zeros_like(prefix);visual[:begin]=prefix[:begin]
    language=np.zeros_like(prefix);language[begin:]=prefix[begin:]
    groups.update(vision=visual,language=language)
    vlm=result['vlm']
    for group,mask in groups.items():
        if not mask.any():
            continue
        arrays={kind:np.asarray(array[:,:,mask,:],np.float32) for kind,array in vlm.items()}
        for kind,value in arrays.items():
            pooled[f'vlm/{group}/{kind}']=value.mean(axis=(1,2))
        hidden,attention=arrays['hidden'],arrays['post_attention']
        def norm(value):
            return np.sqrt(np.sum(value*value,axis=(1,2,3),dtype=np.float64))
        hidden_norm=norm(hidden)
        input_norm=np.maximum(hidden_norm[:-1],1e-12)
        stats[f'vlm/{group}/block_relative_norm']=norm(hidden[1:]-hidden[:-1])/input_norm
        stats[f'vlm/{group}/attn_relative_norm']=norm(arrays['attn_update'])/input_norm
        stats[f'vlm/{group}/ffn_relative_norm']=norm(arrays['ffn_update'])/np.maximum(norm(attention),1e-12)
        stats[f'vlm/{group}/block_cosine']=np.sum(hidden[1:]*hidden[:-1],axis=(1,2,3),dtype=np.float64)/np.maximum(hidden_norm[1:]*hidden_norm[:-1],1e-12)
    return pooled,stats
