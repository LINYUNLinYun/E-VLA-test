import argparse,json,time,uuid
from pathlib import Path
import numpy as np
from .common import OUT,ck,directory,source,claim_case,finish_case,register
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_selection_200.evaluate import identity
from preparations.idea1_v3_4_E2_selection_200.finish import prefix

def header(d,r=None):
    d=Path(d);r=json.loads((d/'result.json').read_text()) if r is None else r
    qs=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
    with np.load(d/'arrays.npz') as z:return dict(result=r,identity=identity(d),first_mask=qs[0]['route'].get('selected_skip',[]),raw0=z['raw_0'].copy(),actions0=z['actions_0'].copy(),fast_input=qs[1]['input_hash'] if len(qs)>1 else None)

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--actor',type=int,required=True);p.add_argument('--group',required=True);p.add_argument('--phase',required=True);a=p.parse_args()
    from preparations.idea1_v3_4 import rollout
    from preparations.idea1_v2_1.runtime_client import rpc
    raw_rpc=rpc;client=f'gpu{a.gpu}-actor{a.actor}'
    rollout.rpc=lambda port,request,*args,**kwargs:raw_rpc(port,dict(request,client=client),*args,**kwargs)
    m=json.loads((OUT/'manifest.json').read_text());pool=Fresh();state=OUT/'workers'/f'gpu{a.gpu}'/f'actor{a.actor}.json'
    try:
        while True:
            c=claim_case(a.group,client)
            if c is None:break
            cfgs=[cfg for cfg in m['conditions'] if cfg['mode']=='verified'];offset=m['cases'].index(c)%len(cfgs);cfgs=cfgs[offset:]+cfgs[:offset]
            expected=identity(Path(m['controls'][ck(c)]['directory']));records=[];control=None
            for cfg in cfgs:
                d,reused=source(m,cfg,c)
                if d is None:
                    d=directory(cfg['id'],c);counter=OUT/'technical_attempts'/cfg['id']/f'{ck(c)}.json';used=json.loads(counter.read_text())['attempts'] if counter.exists() else 0
                    if used>=2:raise RuntimeError(f"Technical retry exhausted {cfg['id']} {ck(c)}")
                    for attempt in range(used,2):
                        if d.exists():d.rename(d.with_name(d.name+'.invalid.'+uuid.uuid4().hex))
                        atomic(counter,dict(attempts=attempt+1,time=time.time()));atomic(state,dict(status='evaluating',group=a.group,case=c,condition=cfg['id'],time=time.time()))
                        try:r=rollout.episode(d,a.port,c,cfg,pool,m['manifest_hash']);break
                        except Exception as e:
                            d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(e),attempt=attempt+1,time=time.time()))
                            if attempt or 'OutOfMemoryError' in repr(e) or 'single_actor_required' in repr(e):raise
                else:r=json.loads((d/'result.json').read_text())
                actual=identity(d);assert {k:v for k,v in actual.items() if k!='first_input_hash'}=={k:v for k,v in expected.items() if k!='first_input_hash'}
                register(cfg,c,d,r,reused);records.append(dict(condition=cfg['id'],directory=str(d),reused=reused))
                if control is None:control=header(d,r)
                print(json.dumps(dict(condition=cfg['id'],case=c,success=r['success'],actions=r['actions'],reused=reused)),flush=True)
            atomic(OUT/'case_done'/f'{ck(c)}.json',dict(case=c,records=records,gpu=a.gpu,client=client,time=time.time()));finish_case(a.group,c)
    finally:pool.close()
    atomic(state,dict(status='group_complete',group=a.group,time=time.time()))

if __name__=='__main__':main()
