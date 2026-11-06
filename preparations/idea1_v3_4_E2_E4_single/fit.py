import argparse,json,time
import numpy as np
import torch
from .common import OUT,LAYERS,dump
from .export import key
from .probe import setup
from .torch_model import Model,hadamard

def scale_fit(x,y):
    mx=x.mean(0);my=y.mean(0)
    s=(y.std(0,correction=0)/x.std(0,correction=0).clamp_min(1e-6)).clamp(.1,10)
    return dict(scale=s,bias=my-mx*s)

def lowrank_fit(x,y,seed):
    # Ridge affine residual mapping, fixed regularization and randomized SVD.
    mx=x.mean(0);res=y-x;mr=res.mean(0);xc=x-mx;rc=res-mr
    xtx=xc.T@xc;lam=.001*xtx.diag().mean()
    xtx.diagonal().add_(lam)
    A=torch.linalg.solve(xtx,xc.T@rc)
    torch.manual_seed(seed);u,s,v=torch.svd_lowrank(A,q=40,niter=2)
    return dict(left=u[:,:32]*s[:32],right=v[:,:32].T,input_mean=mx,residual_mean=mr,ridge=float(lam),
        full_fit_residual_mse=float((xc@A-rc).square().mean()),svd='randomized q40 niter2')

def main():
    setup();model=Model(OUT/'model.pt');data={l:dict(x=[],y=[],shift=[]) for l in LAYERS}
    pairs=[p for p in json.loads((OUT/'selection.json').read_text()) if p['split']=='calibration']
    t=time.perf_counter()
    with torch.no_grad():
        for j,p in enumerate(pairs):
            item=model.item(OUT/'inputs'/f"{key(p['current'])}.pt")
            full=model.encode(item,trace=True);shift=model.encode(item,skip=(3,4,9,11),trace=True)
            n=len(item['x']);idx=torch.cat((torch.linspace(0,511,32).round().long(),torch.linspace(512,n-1,min(16,n-512)).round().long())).cuda()
            for l in LAYERS:
                for name,value in [('x',full['inputs'][l-1]),('y',full['outputs'][l-1]),('shift',shift['inputs'][l-1])]:
                    data[l][name].append(value[idx].cpu())
            if j%10==0:print(json.dumps(dict(fit_features=j+1,total=len(pairs),wall_s=time.perf_counter()-t)),flush=True)
        models={};fitlog=[]
        for l in LAYERS:
            x,y,shift=[torch.cat(data[l][k]).cuda() for k in ('x','y','shift')]
            base=scale_fit(x,y);rotated=scale_fit(hadamard(x),hadamard(y));linear=lowrank_fit(x,y,340400+l);shiftlinear=lowrank_fit(shift,y,340500+l)
            models[l]=dict(channel=base,hadamard=rotated,linear=linear,shiftlinear=shiftlinear)
            fitlog.append(dict(layer=l,tokens=len(x),ridge=linear['ridge'],full_fit_residual_mse=linear['full_fit_residual_mse']))
            print(json.dumps(dict(fit_layer=l,wall_s=time.perf_counter()-t)),flush=True)
            del x,y,shift,data[l]
        def cpu(obj):
            if torch.is_tensor(obj):return obj.cpu()
            if isinstance(obj,dict):return {k:cpu(v) for k,v in obj.items()}
            return obj
        torch.save(cpu(models),OUT/'adapters.pt')
    dump(OUT/'fit_complete.json',dict(calibration_requests=len(pairs),rows=fitlog,wall_s=time.perf_counter()-t,time=time.time()))

if __name__=='__main__':main()
