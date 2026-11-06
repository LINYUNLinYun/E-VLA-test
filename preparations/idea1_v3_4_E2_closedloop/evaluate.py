import argparse,json,uuid
from .common import OUT,SUITES
from preparations.idea1_v3_4.rollout import episode
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v3.io import atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=4);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());pool=Environments(reuse=True)
    try:
        for i,case in enumerate(m['cases']):
            owner=((case['task']-1)//2+SUITES.index(case['suite'])+case['init']-19)%a.shards
            if owner!=a.shard:continue
            configs=m['conditions'];offset=i%len(configs)
            for cfg in configs[offset:]+configs[:offset]:
                d=OUT/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
                if (d/'complete.json').exists():continue
                for attempt in range(2):
                    if d.exists():d.rename(d.with_name(d.name+'.invalid.'+uuid.uuid4().hex))
                    try:
                        r=episode(d,a.port,case,cfg,pool,m['manifest_hash']);print(json.dumps(dict(condition=cfg['id'],case=case,success=r['success'],actions=r['actions'])),flush=True);break
                    except Exception as exc:
                        d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(exc),attempt=attempt+1))
                        if attempt:raise
    finally:pool.close()

if __name__=='__main__':main()
