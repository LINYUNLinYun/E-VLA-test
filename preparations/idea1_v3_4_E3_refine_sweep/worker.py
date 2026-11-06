import json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from .common import OUT,sha,tag,metrics


def run(b,stage,ids):
    m=json.loads((OUT/'manifest.json').read_text());steps=2 if stage==1 else 1
    spec=m if stage==1 else json.loads((OUT/'stage2_manifest.json').read_text())
    grid=spec['grid'];fresh='fresh2' if stage==1 else 'fresh1';completed=[]
    for item in m['selected']:
        if item['id'] not in ids:continue
        dest=OUT/f'stage{stage}'/f"sample{item['id']:03d}";dest.mkdir(parents=True,exist_ok=True)
        if (dest/'complete.json').exists():completed.append(item['id']);continue
        for field,hashfield in [('input','input_sha256'),('shadow_arrays','shadow_arrays_sha256'),('reference_arrays','reference_sha256')]:
            if sha(item[field])!=item[hashfield]:raise ValueError('Frozen data changed '+item[field])
        with np.load(item['input']) as z:
            obs={k:z[k] for k in z.files if k not in ('instruction','seed')};prompt=str(z['instruction']);assert int(z['seed'])==item['seed']
        with np.load(item['shadow_arrays']) as z:initial=z['v14_d2_raw'].copy();original=z['v14_d2'].copy()
        with np.load(item['reference_arrays']) as z:old={k:z[k].copy() for k in z.files}
        c,prep=b.prepare(obs,prompt,());arrays={};rows=[];parity=[];timings={};warmup_s=0.
        for noise,seed in [('e0',item['seed']),('e1',item['refine_seed'])]:
            ref,rawref,_=b.sample(c,seed,10);rr=rawref.float().cpu().numpy()
            equal=np.allclose(ref,old['full_d10_'+noise],atol=m['atol'],rtol=m['rtol']) and np.allclose(rr,old['full_d10_'+noise+'_raw'],atol=m['atol'],rtol=m['rtol'])
            parity.append(dict(check='Full-D10',noise=noise,passed=bool(equal),raw_max_abs=float(np.abs(rr-old['full_d10_'+noise+'_raw']).max())))
            if not equal:raise RuntimeError('Full reference differs; do not pool')
            arrays['ref_'+noise]=ref;arrays['ref_'+noise+'_raw']=rr
            names=[fresh]+(['fresh2'] if stage==2 else [])+[tag(q) for q in grid]
            qmap={tag(q):q for q in grid}
            def sample(name):
                b.torch.cuda.synchronize();tick=time.perf_counter()
                if name.startswith('fresh'):a,r,cost=b.sample(c,seed,int(name[-1]))
                else:a,r,cost=b.warm_sample(c,initial,seed,qmap[name],steps)
                b.torch.cuda.synchronize();wall=(time.perf_counter()-tick)*1000
                return a,r.float().cpu().numpy(),dict(wall_ms=wall,**cost)
            tick=time.perf_counter()
            for name in names:sample(name)
            warmup_s+=time.perf_counter()-tick
            saved={};costs={name:[] for name in names}
            for rep in range(3):
                for name in (names if rep%2==0 else names[::-1]):
                    a,r,cost=sample(name);costs[name].append(cost)
                    if name in saved:
                        if not np.array_equal(a,saved[name][0]) or not np.array_equal(r,saved[name][1]):raise RuntimeError('Candidate is not repeatable')
                    else:saved[name]=(a,r)
            for name,(a,r) in saved.items():arrays[name+'_'+noise]=a;arrays[name+'_'+noise+'_raw']=r
            q1=saved[tag(1.)];ff=saved[fresh]
            if not (np.array_equal(q1[0],ff[0]) and np.array_equal(q1[1],ff[1])):raise RuntimeError('q=1 does not equal fresh same-NFE')
            parity.append(dict(check='q1_equals_fresh',noise=noise,passed=True,NFE=steps))
            for name in ['fresh2']+([tag(.3)] if stage==1 else []):
                oldname=('full_d2_' if name=='fresh2' else 'draft_q03_')+noise
                a,r=saved[name];equal=np.allclose(a,old[oldname],atol=m['atol'],rtol=m['rtol']) and np.allclose(r,old[oldname+'_raw'],atol=m['atol'],rtol=m['rtol'])
                parity.append(dict(check=oldname,passed=bool(equal)))
                if not equal:raise RuntimeError('Archived fresh/q03 path differs')
            arrays['draft_'+noise]=original;arrays['draft_'+noise+'_raw']=initial
            timings[noise]=costs
        for noise in ('e0','e1'):
            names=[k[:-3] for k in arrays if k.endswith('_'+noise) and not k.startswith('ref_')]
            for name in names:
                a=arrays[name+'_'+noise];r=arrays[name+'_'+noise+'_raw']
                costs=timings[noise].get(name,[])
                for reference in ('e0','e1'):
                    for h in (5,50):
                        rows.append(dict(id=item['id'],suite=item['case']['suite'],task=item['case']['task'],split=item['split'],stage=stage,candidate=name,noise=noise,reference=reference,horizon=h,
                            ae_ms=float(np.mean([t['ae_ms'] for t in costs])) if costs else 0.,wall_ms=float(np.mean([t['wall_ms'] for t in costs])) if costs else 0.,
                            **metrics(a,r,arrays['ref_'+reference],arrays['ref_'+reference+'_raw'],h)))
        for name,a in arrays.items():assert a.shape==((50,32) if name.endswith('_raw') else (50,7)) and np.isfinite(a).all(),name
        np.savez_compressed(dest/'actions.npz',**arrays)
        atomic(dest/'metrics.json',dict(selection=item,grid=grid,steps=steps,prepare=prep,warmup_s=warmup_s,timings=timings,rows=rows,parity=parity,original_draft_noise='e0'))
        atomic(dest/'complete.json',dict(id=item['id'],manifest_hash=m['manifest_hash'],stage_spec_sha256=sha(OUT/'manifest.json' if stage==1 else OUT/'stage2_manifest.json')))
        completed.append(item['id'])
    return dict(completed=completed)
