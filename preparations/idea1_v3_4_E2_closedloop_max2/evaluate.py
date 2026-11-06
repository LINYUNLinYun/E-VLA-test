import argparse,json,uuid
from .common import OUT,claim,finish_case
from preparations.idea1_v3_4.rollout import episode
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v3.io import atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());pool=Environments(reuse=True)
    try:
        while True:
            case=claim(a.gpu)
            if case is None:break
            i=m['cases'].index(case);configs=m['conditions'];offset=i%len(configs)
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
            finish_case(case)
    finally:pool.close()

if __name__=='__main__':main()
