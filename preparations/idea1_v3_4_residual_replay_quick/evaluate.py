import argparse,json,random,uuid
from pathlib import Path
import numpy as np
from .common import OUT,key,directory
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);a=p.parse_args()
    from preparations.idea1_v3_4 import rollout
    from preparations.idea1_v3_1.rollout import Environments
    from preparations.idea1_v2_1.runtime_client import rpc,input_digest
    m=json.loads((OUT/'manifest.json').read_text());state=dict(first=True,case=None)
    def checked_rpc(port,req,*args,**kwargs):
        if req['op']=='predict' and state['first']:
            actual=input_digest(req['obs'],req['instruction']);c=state['case'];p=OUT/'initial_reference'/f'{key(c)}.json'
            if p.exists():assert json.loads(p.read_text())['input_hash']==actual,'Initial observation mismatch'
            else:atomic(p,dict(case=c,input_hash=actual,reference_compatible=actual==m['references'][key(c)]['initial_input_hash']))
            state['first']=False
        return rpc(port,req,*args,**kwargs)
    rollout.rpc=checked_rpc
    for index,c in enumerate(m['cases']):
        done=OUT/'case_done'/f'{key(c)}.json'
        if done.exists():continue
        try:
            with lock(OUT/'claims'/f'{key(c)}.lock',blocking=False):
                if done.exists():continue
                cfgs=m['conditions'];cfgs=cfgs[index%2:]+cfgs[:index%2]
                for cfg in cfgs:
                    d=directory(cfg['id'],c)
                    if (d/'complete.json').exists():continue
                    for attempt in range(2):
                        if d.exists():d.rename(d.with_name(d.name+'.technical.'+uuid.uuid4().hex))
                        random.seed(7);np.random.seed(7);pool=Environments(reuse=False);state.update(first=True,case=c)
                        try:
                            r=rollout.episode(d,a.port,c,cfg,pool,m['manifest_hash']);print(json.dumps(dict(condition=cfg['id'],case=c,success=r['success'])),flush=True);break
                        except Exception as exc:
                            d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(exc),attempt=attempt+1))
                            if attempt:raise
                        finally:pool.close()
                ref=json.loads((OUT/'initial_reference'/f'{key(c)}.json').read_text())
                if ref['reference_compatible']:reference=m['references'][key(c)]['directory']
                else:
                    cfg=m['reference_config'];d=directory(cfg['id'],c);random.seed(7);np.random.seed(7);pool=Environments(reuse=False);state.update(first=True,case=c)
                    try:rollout.episode(d,a.port,c,cfg,pool,m['manifest_hash'])
                    finally:pool.close()
                    reference=str(d)
                atomic(done,dict(case=c,reference_directory=reference,reference_reused=ref['reference_compatible']))
        except BlockingIOError:continue
if __name__=='__main__':main()
