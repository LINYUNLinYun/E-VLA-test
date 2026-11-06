import argparse,json,time,uuid
from pathlib import Path
from .common import OUT,ck,source,claim_case,finish_case,register
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v3.io import atomic

def identity(d):
    a=json.loads((d/'identity.json').read_text());a['first_input_hash']=json.loads((d/'infer.jsonl').open().readline())['input_hash']
    return {k:a[k] for k in ('init_vector_sha256','bddl_sha256','controller','prompt','env_seed','settling','first_input_hash')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--actor',type=int,required=True);p.add_argument('--group',required=True);p.add_argument('--phase',required=True);a=p.parse_args()
    from preparations.idea1_v3_4 import rollout
    from preparations.idea1_v2_1.runtime_client import rpc
    raw_rpc=rpc;client=f'gpu{a.gpu}-actor{a.actor}'
    def isolated(port,request,*args,**kwargs):return raw_rpc(port,dict(request,client=client),*args,**kwargs)
    rollout.rpc=isolated;m=json.loads((OUT/'manifest.json').read_text());pool=Fresh();state=OUT/'workers'/f'gpu{a.gpu}'/f'actor{a.actor}.json'
    try:
        while True:
            c=claim_case(a.group,client)
            if c is None:break
            cfgs=[cfg for cfg in m['conditions'] if cfg['phase']=='main200' or c in m['small_cases']];index=m['cases'].index(c);offset=index%len(cfgs);cfgs=cfgs[offset:]+cfgs[:offset]
            expected=identity(Path(m['baseline']['v14_jacobian_spaced_extrapolate'][ck(c)]['directory']));records=[]
            for cfg in cfgs:
                d,reused=source(m,cfg,c)
                if d is None:
                    from .common import directory
                    d=directory(cfg['id'],c)
                    counter=OUT/'technical_attempts'/cfg['id']/f'{ck(c)}.json'
                    used=json.loads(counter.read_text())['attempts'] if counter.exists() else 0
                    if used>=2:raise RuntimeError(f"Technical retry budget exhausted: {cfg['id']} {ck(c)}")
                    for attempt in range(used,2):
                        if d.exists():d.rename(d.with_name(d.name+'.invalid.'+uuid.uuid4().hex))
                        atomic(counter,dict(attempts=attempt+1,time=time.time()))
                        atomic(state,dict(status='evaluating',group=a.group,case=c,condition=cfg['id'],time=time.time()))
                        try:
                            r=rollout.episode(d,a.port,c,cfg,pool,m['manifest_hash'])
                            print(json.dumps(dict(condition=cfg['id'],case=c,success=r['success'],actions=r['actions'],wall_s=r['wall_s'])),flush=True);break
                        except Exception as e:
                            d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(e),attempt=attempt+1,time=time.time()))
                            if attempt or 'OutOfMemoryError' in repr(e) or 'single_actor_required' in repr(e):raise
                else:r=json.loads((d/'result.json').read_text())
                actual=identity(d)
                assert {k:v for k,v in actual.items() if k!='first_input_hash'}=={k:v for k,v in expected.items() if k!='first_input_hash'},dict(case=c,condition=cfg['id'],expected=expected,actual=actual)
                register(cfg,c,d,r);records.append(dict(condition=cfg['id'],directory=str(d),initial_hash=actual['first_input_hash'],baseline_initial_hash=expected['first_input_hash'],initial_matches_baseline=actual['first_input_hash']==expected['first_input_hash']))
            atomic(OUT/'case_done'/a.phase/f'{ck(c)}.json',dict(case=c,records=records,gpu=a.gpu,client=client,time=time.time()));finish_case(a.group,c)
    finally:pool.close()
    atomic(state,dict(status='group_complete',group=a.group,time=time.time()))

if __name__=='__main__':main()
