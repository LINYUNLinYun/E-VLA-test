import numpy as np

def pool(result):
    pooled,stats={},{}
    names=list(result['vision'])
    sizes=[result['vision'][name]['projected'].shape[1] for name in names]
    prefix=np.asarray(result['prefix_mask'][0],bool)
    groups={}; begin=0
    for name,size in zip(names,sizes):
        group=np.zeros_like(prefix);group[begin:begin+size]=prefix[begin:begin+size]
        if group.any():
            groups[name]=group
            for kind,array in result['vision'][name].items():
                a=np.asarray(array,dtype=np.float32)
                pooled[f'vision/{name}/{kind}']=a.mean(axis=-2).squeeze(-2) if a.ndim==4 else a.mean(axis=-2)
        begin+=size
    visual=np.zeros_like(prefix);visual[:begin]=prefix[:begin]
    language=np.zeros_like(prefix);language[begin:]=prefix[begin:]
    groups.update(vision=visual,language=language)
    vlm=result['vlm']
    for group,mask in groups.items():
        if not mask.any():
            continue
        for kind,array in vlm.items():
            a=np.asarray(array[:,:,mask,:],np.float32)
            pooled[f'vlm/{group}/{kind}']=a.mean(axis=(1,2))
        h=np.asarray(vlm['hidden'][:,:,mask,:],np.float32)
        attention=np.asarray(vlm['post_attention'][:,:,mask,:],np.float32)
        def norm(a):
            return np.sqrt(np.sum(a*a,axis=(1,2,3),dtype=np.float64))
        stats[f'vlm/{group}/block_relative_norm']=norm(h[1:]-h[:-1])/np.maximum(norm(h[:-1]),1e-12)
        stats[f'vlm/{group}/attn_relative_norm']=norm(np.asarray(vlm['attn_update'][:,:,mask,:],np.float32))/np.maximum(norm(h[:-1]),1e-12)
        stats[f'vlm/{group}/ffn_relative_norm']=norm(np.asarray(vlm['ffn_update'][:,:,mask,:],np.float32))/np.maximum(norm(attention),1e-12)
        stats[f'vlm/{group}/block_cosine']=np.sum(h[1:]*h[:-1],axis=(1,2,3),dtype=np.float64)/np.maximum(norm(h[1:])*norm(h[:-1]),1e-12)
    return pooled,stats
