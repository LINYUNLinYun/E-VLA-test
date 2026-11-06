"""Same observed Full trajectory: single-layer/suffix, old versus current oracle."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic,write_rows
from preparations.idea1_v3.policy import V14
from preparations.kv_reuse_probe.probe import load_observation,action_metrics,write_csv


def layer_stats(torch,full,candidate):
    rows=[]
    for key in ('K','V'):
        for region,sl in (('all',slice(None)),('vision',slice(0,512)),('language',slice(512,None))):
            a=full[key][:,sl].float().reshape(18,-1); b=candidate[key][:,sl].float().reshape(18,-1)
            if a.shape[1]==0: continue
            an=a.norm(dim=1); bn=b.norm(dim=1)
            data=torch.stack([(a*b).sum(dim=1)/(an*bn),(a-b).norm(dim=1)/an,bn/an],dim=1).cpu().numpy()
            rows += [dict(kv=key,region=region,layer=i+1,cosine=float(x[0]),relative_l2=float(x[1]),rms_ratio=float(x[2])) for i,x in enumerate(data)]
    return rows


def run(backend,pairs,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True); torch=backend.torch
    done=0
    for pair_id,p in enumerate(pairs):
        dest=out/f'pair{pair_id:03d}'; dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists(): done+=1; continue
        obs,prompt=load_observation(p['current']); oldobs,oldprompt=load_observation(p['previous'])
        if oldprompt!=prompt: raise ValueError('Mechanism pair prompt mismatch')
        seed=p['current']['seed']
        old,_=backend.prepare(oldobs,prompt,()); full,_=backend.prepare(obs,prompt,()); draft,_=backend.prepare(obs,prompt,V14)
        stats=[]
        for name,c in (('v14',draft),('old',old)):
            stats += [dict(candidate=name,**r) for r in layer_stats(torch,full,c)]
        write_csv(dest/'layer_metrics.csv',stats)
        rows=[]; values={}; timings=[]
        # Explicit immutable snapshots; modify only the selected KV layers.
        candidates=[('v14',None,None)]
        candidates += [(f'old_single_L{i}',old,[i-1]) for i in range(1,19)]
        candidates += [(f'old_suffix_L{i}',old,list(range(i-1,18))) for i in range(12,19)]
        candidates += [('all_old',old,list(range(18)))]
        candidates += [(f'oracle_suffix_L{i}',full,list(range(i-1,18))) for i in (12,13,16)]
        for d in (2,10):
            reference,rawref,_=backend.sample(full,seed,d)
            values[f'full_D{d}']=reference; values[f'full_D{d}_raw']=rawref.float().cpu().numpy()
            for name,source,layers in candidates:
                context=draft
                if source is not None:
                    context=dict(draft)
                    for key in ('K','V'):
                        context[key]=draft[key].clone(); context[key][layers]=source[key][layers]
                actions,raw,cost=backend.sample(context,seed,d)
                values[f'{name}_D{d}']=actions
                values[f'{name}_D{d}_raw']=raw.float().cpu().numpy()
                rows += [dict(candidate=name,D=d,**r) for r in action_metrics(reference,actions)]
                del context
        # Request speed: common shape warmed above, paths AB/BA, three repetitions.
        # Preparation of old Full is charged separately; periodic refresh is in rollout.
        for rep in range(3):
            names=['v14','old16','old13','old12','full_d2','full_d10']
            if rep%2: names.reverse()
            for name in names:
                torch.cuda.synchronize(); tick=time.perf_counter()
                if name.startswith('old'):
                    c,t=backend.prepare_hybrid(obs,prompt,old,int(name[3:])); d=2
                else:
                    c,t=backend.prepare(obs,prompt,V14 if name=='v14' else ()); d=10 if name=='full_d10' else 2
                _,_,cost=backend.sample(c,seed,d); torch.cuda.synchronize()
                timings.append(dict(path=name,rep=rep,wall_ms=(time.perf_counter()-tick)*1000,prepare=t,sample=cost))
                del c
        write_csv(dest/'action_metrics.csv',rows); np.savez_compressed(dest/'actions.npz',**values)
        atomic(dest/'timings.json',timings); atomic(dest/'pair.json',p)
        atomic(dest/'complete.json',dict(pair_id=pair_id,selection_sha256=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()))
        done+=1
        print(json.dumps(dict(mechanism_complete=done,total=len(pairs))),flush=True)
    return dict(completed=done,total=len(pairs))
