import argparse,json,time
import numpy as np
import torch
from .common import OUT,SUITES,dump
from .export import key
from .torch_model import Model

def setup():
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False

def timing(fn):
    torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize()
    return out,(time.perf_counter()-t)*1000

def main():
    setup();model=Model(OUT/'model.pt');pairs=json.loads((OUT/'selection.json').read_text());checks=[]
    for suite in SUITES:
        pair=next(p for p in pairs if p['case']['suite']==suite and p['e2'])
        item=model.item(OUT/'inputs'/f"{key(pair['current'])}.pt")
        with torch.no_grad():
            ref,wall=timing(lambda:model.forward(item));g1=model.forward(item,gates=torch.ones(17,device='cuda'))
            native=item['native_full_raw'];max_native=float((ref-native).abs().max());rms_native=float((ref[:5,:7]-native[:5,:7]).square().mean().sqrt())
            assert torch.allclose(ref,g1,atol=1e-5,rtol=1e-5)
            for layer in (1,3,11,17):
                gate=torch.ones(17,device='cuda');gate[layer-1]=0
                a=model.forward(item,gates=gate);b=model.forward(item,skip=(layer,))
                assert torch.allclose(a,b,atol=1e-5,rtol=1e-5),(layer,float((a-b).abs().max()))
        gates=torch.ones(17,device='cuda',requires_grad=True)
        raw,fwd_grad=timing(lambda:model.forward(item,gates=gates))
        yy=raw[:5,:7];grads=[]
        _,backward=timing(lambda:grads.append(torch.autograd.grad(yy[0,0],gates,retain_graph=True)[0]))
        g=grads[0].detach()
        fd=[]
        for layer in (1,3,11,17):
            plus=torch.ones(17,device='cuda');minus=plus.clone();plus[layer-1]+=.001;minus[layer-1]-=.001
            with torch.no_grad():delta=(model.forward(item,gates=plus)[0,0]-model.forward(item,gates=minus)[0,0])/.002
            fd.append(dict(layer=layer,grad=float(g[layer-1]),finite_difference=float(delta),abs_difference=float((g[layer-1]-delta).abs())))
        assert all(abs(v['grad']-v['finite_difference'])<.003+.03*abs(v['finite_difference']) for v in fd),fd
        checks.append(dict(suite=suite,forward_ms=wall,forward_with_graph_ms=fwd_grad,one_vjp_ms=backward,
            native_raw_max_abs=max_native,native_prefix7_rms=rms_native,finite_difference=fd,passed=True,
            max_memory_mb=torch.cuda.max_memory_allocated()/1024**2))
        del raw,yy,gates,item;torch.cuda.empty_cache()
        print(json.dumps(checks[-1]),flush=True)
    dump(OUT/'engineering_probe.json',dict(passed=True,checks=checks))

if __name__=='__main__':main()
