import argparse,json,random,time,uuid
from pathlib import Path
import numpy as np
from .common import OUT,casekey,directory
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock

class InitializationMismatch(RuntimeError):pass

def reference(c):
    p=OUT/'initial_reference'/f'{casekey(c)}.json'
    return json.loads(p.read_text()) if p.exists() else None

def source(m,c,name):
    d=directory(name,c)
    if (d/'complete.json').exists():return d,False
    ref=reference(c);reuse=m['reuse'].get(casekey(c),{})
    if ref is not None and ref['reuse_valid'] and name in reuse:return Path(reuse[name]['directory']),True
    return None,None

def run_case(m,c,index,port,client,state_path):
    from preparations.idea1_v3_4 import rollout
    from preparations.idea1_v3_1.rollout import Environments
    from preparations.idea1_v2_1.runtime_client import rpc,input_digest
    raw_rpc=rpc;first=False
    def isolated_rpc(port,request,*args,**kwargs):
        nonlocal first
        if request['op']=='predict' and not first:
            actual=input_digest(request['obs'],request['instruction']);p=OUT/'initial_reference'/f'{casekey(c)}.json';ref=reference(c)
            if ref is None:
                old=m['old_initial_hashes'].get(casekey(c));reuse_valid=old is None or old==actual
                ref=dict(case=c,input_hash=actual,reuse_valid=reuse_valid,old_input_hash=old,time=time.time())
                atomic(p,ref)
                if old is not None and not reuse_valid:atomic(OUT/'reuse_rejected'/f'{casekey(c)}.json',ref)
            if ref['input_hash']!=actual:raise InitializationMismatch(f"Initial observation mismatch for {casekey(c)}")
            first=True
        return raw_rpc(port,dict(request,client=client),*args,**kwargs)
    rollout.rpc=isolated_rpc
    cfgs=m['conditions'];offset=index%len(cfgs);order=cfgs[offset:]+cfgs[:offset]
    while True:
        ref=reference(c);reuse=m['reuse'].get(casekey(c),{})
        pending=[cfg for cfg in order if source(m,c,cfg['id'])[0] is None]
        if not pending:break
        if ref is None and reuse:pending=[cfg for cfg in pending if cfg['id'] not in reuse]
        for cfg in pending:
            if source(m,c,cfg['id'])[0] is not None:continue
            d=directory(cfg['id'],c)
            for attempt in range(2):
                if d.exists():d.rename(d.with_name(d.name+'.technical.'+uuid.uuid4().hex))
                random.seed(7);np.random.seed(7);pool=Environments(reuse=False);first=False
                atomic(state_path,dict(status='evaluating',case=c,condition=cfg['id'],time=time.time()))
                try:
                    r=rollout.episode(d,port,c,cfg,pool,m['manifest_hash'])
                    print(json.dumps(dict(condition=cfg['id'],case=c,success=r['success'],actions=r['actions'])),flush=True)
                    break
                except Exception as exc:
                    d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(exc),attempt=attempt+1,time=time.time()))
                    if attempt:raise
                finally:pool.close()
    records=[];hashes=[]
    for cfg in cfgs:
        d,reused=source(m,c,cfg['id']);r=json.loads((d/'result.json').read_text());identity=json.loads((d/'identity.json').read_text())
        assert r['valid'] and r['config']==cfg
        h=json.loads((d/'infer.jsonl').open().readline())['input_hash'];hashes.append(h)
        records.append(dict(r,directory=str(d),reused=reused,initial_input_hash=h,init_vector_sha256=identity['init_vector_sha256']))
    assert len(set(hashes))==1
    assert len({r['init_vector_sha256'] for r in records})==1
    atomic(OUT/'case_done'/f'{casekey(c)}.json',dict(case=c,records=records,initial_input_hash=hashes[0],time=time.time()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--actor',type=int,required=True);p.add_argument('--suite',required=True);p.add_argument('--task',type=int,required=True);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());state=OUT/'workers'/f'gpu{a.gpu}'/f'actor{a.actor}.json'
    for index,c in enumerate(m['cases']):
        if c['suite']!=a.suite or c['task']!=a.task:continue
        done=OUT/'case_done'/f'{casekey(c)}.json'
        if done.exists():continue
        try:
            with lock(OUT/'case_claims'/f'{casekey(c)}.lock',blocking=False):
                if not done.exists():run_case(m,c,index,a.port,f'gpu{a.gpu}-actor{a.actor}',state)
        except BlockingIOError:continue
    atomic(state,dict(status='group_complete',suite=a.suite,task=a.task,time=time.time()))

if __name__=='__main__':main()
