import argparse,json,time,uuid
from .common import OUT,REFERENCE,claim,finish_case
from preparations.idea1_v3_4.rollout import episode
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v3.io import atomic

def identity(directory):
    a=json.loads((directory/'identity.json').read_text());a['first_input_hash']=json.loads((directory/'infer.jsonl').read_text().splitlines()[0])['input_hash']
    return {k:a[k] for k in ['init_vector_sha256','bddl_sha256','controller','prompt','env_seed','settling','first_input_hash']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());pool=Fresh()
    try:
        while True:
            case=claim(args.gpu)
            if case is None:break
            index=m['cases'].index(case);configs=m['conditions'];offset=index%len(configs);reference=identity(REFERENCE/'v14_cosine'/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}")
            for cfg in configs[offset:]+configs[:offset]:
                d=OUT/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
                if not (d/'complete.json').exists():
                    for attempt in range(2):
                        if d.exists():d.rename(d.with_name(d.name+'.invalid.'+uuid.uuid4().hex))
                        try:
                            r=episode(d,args.port,case,cfg,pool,m['manifest_hash'])
                            print(json.dumps(dict(condition=cfg['id'],case=case,success=r['success'],actions=r['actions'],wall_s=r['wall_s'])),flush=True);break
                        except Exception as e:
                            d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(e),attempt=attempt+1,time=time.time()))
                            if attempt:raise
                actual=identity(d)
                if reference is None:reference=actual
                assert actual==reference,dict(case=case,condition=cfg['id'],reference=reference,actual=actual)
            atomic(OUT/'identities'/f"{case['suite']}_{case['task']}_{case['init']}.json",dict(passed=True,case=case,conditions=len(configs),gpu=args.gpu,time=time.time()))
            finish_case(case)
    finally:pool.close()
if __name__=='__main__':main()
