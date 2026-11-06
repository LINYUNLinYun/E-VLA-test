"""One-time fixed vision/transform export; no environment or closed-loop reruns."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OUT,select,dump,sha

def key(side):return sha_text(side['path'])[:20]
def sha_text(x):
    import hashlib
    return hashlib.sha256(x.encode()).hexdigest()

def main():
    from preparations.idea1_v3.safe_qk import install
    OUT.mkdir(parents=True,exist_ok=True);install(str(OUT/'export_qk_receipt.json'))
    from preparations.idea1_v3_4.backend import Backend
    from preparations.kv_reuse_probe.probe import load_observation
    from scripts.spec.triton import pi0_infer as k
    from preparations.idea1_v2_2.schedules import flow_schedule
    pairs=select();dump(OUT/'selection.json',pairs)
    b=Backend();exported=[];tick=time.perf_counter()
    unique={}
    for p in pairs:
        for side in ('current','previous','previous2'):
            if p[side] is not None:unique[key(p[side])]=p[side]
    for index,(identity,side) in enumerate(unique.items()):
        dest=OUT/'inputs'/f'{identity}.pt'
        if dest.exists():continue
        obs,prompt=load_observation(side);c,t=b.prepare(obs,prompt,(),encode=False)
        r=c['runtime'];w,buf=r.weights,r.buffers
        k.layer_norm_matmul_n256_1152_2048_bias(buf['vision_x'],w['vision_final_norm_w'],w['vision_final_norm_b'],w['encoder_multi_modal_projector_w'],w['encoder_multi_modal_projector_b'],buf['encoder_x'],buf['vision_x_norm'])
        item=dict(x=buf['encoder_x'].detach().cpu().clone(),state=c['state'].detach().cpu().clone(),
            encoder_rope=buf['encoder_rope_weights'].detach().cpu().clone(),decoder_rope=buf['decoder_rope_weights'].detach().cpu().clone(),
            noise=torch.from_numpy(np.random.default_rng(side['seed']).standard_normal((50,32),dtype=np.float32)).to(torch.bfloat16),side=side,
            preparation=t)
        zeros=np.zeros((50,32),np.float32);offset=b.decode(c,torch.from_numpy(zeros).cuda())
        matrix=np.zeros((7,7),np.float32)
        for j in range(7):
            a=zeros.copy();a[:,j]=1
            decoded=b.decode(c,torch.from_numpy(a).cuda());matrix[j]=decoded[0]-offset[0]
            assert np.allclose(decoded-offset,matrix[j][None],atol=1e-6,rtol=1e-5)
        item.update(decode_matrix=torch.from_numpy(matrix),decode_offset=torch.from_numpy(offset))
        # This check catches a hidden non-affine decoder or padded-coordinate use.
        test=np.random.default_rng(1414).standard_normal((50,32),dtype=np.float32)
        actual=b.decode(c,torch.from_numpy(test).cuda());approx=test[:,:7]@matrix+offset
        assert np.allclose(actual,approx,atol=2e-6,rtol=1e-5),np.max(np.abs(actual-approx))
        if not (OUT/'model.pt').exists():
            nodes=flow_schedule('pi0',None,2);biases,outw,outb,_=b.adapter.schedule(r,nodes)
            weights={name:v.detach().cpu().clone() for name,v in w.items() if name.startswith(('encoder_attn','encoder_ffn','decoder_'))}
            model=dict(weights=weights,biases=[v.detach().cpu().clone() for v in biases],outw=[v.detach().cpu().clone() for v in outw],outb=[v.detach().cpu().clone() for v in outb],protocol=b.protocol)
            torch.save(model,OUT/'model.pt');dump(OUT/'model_identity.json',dict(sha256=sha(OUT/'model.pt'),protocol=b.protocol,export_s=time.perf_counter()-tick))
        # Native comparisons are diagnostics only, never pooled with PyTorch data.
        if any(p['e2'] and key(p['current'])==identity for p in pairs):
            c,_=b.prepare(obs,prompt,());a,raw,_=b.sample(c,side['seed'],2)
            item.update(native_full_raw=raw.detach().cpu(),native_full_actions=a,
                native_K=c['K'].detach().cpu(),native_V=c['V'].detach().cpu())
        dest.parent.mkdir(parents=True,exist_ok=True);torch.save(item,dest)
        exported.append(dict(key=identity,path=str(dest),sha256=sha(dest)))
        if index%10==0:print(json.dumps(dict(export=index+1,total=len(unique),elapsed_s=time.perf_counter()-tick)),flush=True)
        del c,r,w,buf,item
    records=[dict(key=identity,path=str(OUT/'inputs'/f'{identity}.pt'),sha256=sha(OUT/'inputs'/f'{identity}.pt')) for identity in unique]
    dump(OUT/'export_complete.json',dict(inputs=records,total=len(records),wall_s=time.perf_counter()-tick,time=time.time()))
    print('EXPORT_COMPLETE',flush=True)

if __name__=='__main__':main()
