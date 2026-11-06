import time
from pathlib import Path
from preparations.idea1_v3.io import atomic
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v3.policy import V14


def run(backend,pairs,output):
    b=backend; b.reuse_allowed=True; rows=[]
    for index,p in enumerate(pairs):
        oldobs,prompt=load_observation(p['previous']); obs,current_prompt=load_observation(p['current'])
        assert prompt==current_prompt
        old,_=b.prepare(oldobs,prompt,()); seed=p['current']['seed']
        def measure(name):
            b.torch.cuda.synchronize(); tick=time.perf_counter()
            if name.startswith('old'): c,prep=b.prepare_hybrid(obs,prompt,old,int(name[3:])); steps=2
            else: c,prep=b.prepare(obs,prompt,V14 if name=='v14' else ()); steps=10 if name=='full10' else 2
            a,r,cost=b.sample(c,seed,steps); b.torch.cuda.synchronize()
            return dict(path=name,wall_ms=(time.perf_counter()-tick)*1000,prepare=prep,sample=cost)
        names=['full10','v14','old15','old16','old17']
        for name in names: measure(name)
        for rep in range(3):
            for name in names if rep%2==0 else names[::-1]: rows.append(dict(pair=index,episode=p['episode'],rep=rep,**measure(name)))
        atomic(Path(output),dict(completed_pairs=index+1,total=len(pairs),rows=rows))
    return dict(completed_pairs=len(pairs))
