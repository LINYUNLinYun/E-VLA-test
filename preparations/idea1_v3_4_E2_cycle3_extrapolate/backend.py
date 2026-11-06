import torch
import triton
from preparations.idea1_v3_4_v12_compensation.backend import encoder,difference

def capture(hidden):
    assert len(hidden)==19;out={}
    for i in range(17):
        r=torch.empty_like(hidden[i],dtype=torch.float32)
        difference[(triton.cdiv(r.numel(),1024),)](hidden[i+1],hidden[i],r,r.numel(),1024)
        out[i+1]=r
    return out

def prepare_patch(backend,obs,instruction,skip,last,previous,ratio,reference=False):
    from scripts.spec.triton import pi0_infer as k
    original=k.transformer_encoder
    def wrapped(w,b,seq,**kwargs):
        encoder(k,w,b,seq,skip=skip,history=last,older=previous,ratio=ratio,reference=reference)
    k.transformer_encoder=wrapped
    try:return backend.prepare(obs,instruction,skip)
    finally:k.transformer_encoder=original
