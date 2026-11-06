import argparse,json,time
import numpy as np
import torch
from .common import OUT,LAYERS,dump,metrics
from .export import key
from .probe import setup,timing
from .torch_model import Model,hadamard

def todevice(obj,device='cuda'):
    if torch.is_tensor(obj):return obj.to(device)
    if isinstance(obj,dict):return {k:todevice(v,device) for k,v in obj.items()}
    return obj

def make_patch(name,adapter,old=None,old2=None,ratio=0):
    if name=='channel':return lambda x:x*adapter['channel']['scale']+adapter['channel']['bias']
    if name=='hadamard':return lambda x:hadamard(hadamard(x)*adapter['hadamard']['scale']+adapter['hadamard']['bias'])
    if name=='reuse':return lambda x:x+old
    if name=='extrapolate':return lambda x:x+old+(0 if old2 is None else ratio*(old-old2))
    kind='shiftlinear' if name.startswith('shift') else 'linear';r=int(name.split('_r')[-1]);a=adapter[kind]
    return lambda x:x+((x-a['input_mean'])@a['left'][:,:r])@a['right'][:r]+a['residual_mean']

NAMES=('skip','channel','hadamard','reuse','extrapolate','linear_r8','linear_r32')

def run(model,adapters,p):
    d=OUT/'E4'/f"sample{p['id']:03d}";d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():return
    item=model.item(OUT/'inputs'/f"{key(p['current'])}.pt")
    olditem=model.item(OUT/'inputs'/f"{key(p['previous'])}.pt")
    old2item=None if p['previous2'] is None else model.item(OUT/'inputs'/f"{key(p['previous2'])}.pt")
    with torch.no_grad():
        full=model.encode(item,trace=True);shift=model.encode(item,skip=(3,4,9,11),trace=True)
        old=model.encode(olditem,trace=True);old2=None if old2item is None else model.encode(old2item,trace=True)
        ref=model.sample(item,full);reff=ref.cpu().numpy();actions=model.decode(item,ref)
        ratio=0 if old2 is None else (p['current']['step']-p['previous']['step'])/(p['previous']['step']-p['previous2']['step'])
        arrays=dict(full_raw=reff,full_actions=actions);rows=[];hiddenrows=[];speed=[]
        for l in LAYERS:
            hist=old['outputs'][l-1]-old['inputs'][l-1]
            hist2=None if old2 is None else old2['outputs'][l-1]-old2['inputs'][l-1]
            names=list(NAMES)
            for name in names:
                patch=None if name=='skip' else make_patch(name,adapters[l],hist,hist2,ratio)
                c=model.encode(item,skip=(l,),patch=None if patch is None else (l,patch),prefix=full,start=l-1)
                raw=model.sample(item,c);arr=raw.cpu().numpy();decoded=model.decode(item,raw)
                arrays[f'L{l}_{name}_raw']=arr;arrays[f'L{l}_{name}_actions']=decoded
                for h in (5,50):rows.append(dict(id=p['id'],**p['case'],split=p['split'],layer=l,candidate=name,horizon=h,
                    has_second_history=old2 is not None,age_steps=p['current']['step']-p['previous']['step'],
                    **metrics(actions,decoded,reff,arr,h)))
            # Assess the same single-layer patch on shifted inputs. No multi-layer
            # downstream action experiment is hidden in this robustness check.
            for domain,x in [('full',full['inputs'][l-1]),('shifted',shift['inputs'][l-1])]:
                target=full['outputs'][l-1]
                for name in (*NAMES,'shiftlinear_r8','shiftlinear_r32'):
                    patch=None if name=='skip' else make_patch(name,adapters[l],hist,hist2,ratio)
                    y=x if patch is None else patch(x)
                    hiddenrows.append(dict(id=p['id'],**p['case'],split=p['split'],layer=l,domain=domain,candidate=name,
                        hidden_mse=float((y-target).square().mean()),hidden_relative_l2=float((y-target).norm()/target.norm())))
            # Complete paths for timing; shared Full prefixes above are only a
            # scientific computation saving. Fixed one observation/task for cost.
            if p['e2'] and p['case']['task']==0:
                funcs={name:(lambda patch=make_patch(name,adapters[l],hist,hist2,ratio):model.forward(item,skip=(l,),patch=(l,patch))) for name in NAMES if name!='skip'}
                funcs['skip']=lambda:model.forward(item,skip=(l,))
                def refresh():
                    cc=model.encode(item,trace=True)
                    rr=(cc['outputs'][l-1]-cc['inputs'][l-1]).clone()
                    return model.sample(item,cc),rr
                funcs['full_refresh']=refresh
                for fn in funcs.values():fn()
                for rep in range(3):
                    order=list(funcs)
                    if rep%2:order.reverse()
                    for name in order:
                        _,wall=timing(funcs[name]);speed.append(dict(id=p['id'],**p['case'],layer=l,candidate=name,rep=rep,model_wall_ms=wall))
            if l%5==0:print(json.dumps(dict(E4_sample=p['id'],layer=l)),flush=True)
    np.savez_compressed(d/'arrays.npz',**arrays)
    dump(d/'metrics.json',dict(rows=rows,hidden=hiddenrows,timing=speed,has_second_history=old2 is not None,extrapolation_ratio=ratio))
    dump(d/'complete.json',dict(id=p['id'],time=time.time(),layers=17,candidates=list(NAMES)))

def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);args=a.parse_args()
    setup();model=Model(OUT/'model.pt');adapters=todevice(torch.load(OUT/'adapters.pt',map_location='cpu',weights_only=False))
    pairs=json.loads((OUT/'selection.json').read_text())
    for i,p in enumerate(pairs):
        if i%args.shards!=args.shard:continue
        tick=time.perf_counter();run(model,adapters,p);print(json.dumps(dict(E4=p['id'],case=p['case'],wall_s=time.perf_counter()-tick)),flush=True)

if __name__=='__main__':main()
