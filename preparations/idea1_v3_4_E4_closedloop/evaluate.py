import argparse,json,uuid
from .common import OUT
from preparations.idea1_v3_4.rollout import episode
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3.io import atomic

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());pool=Environments(reuse=True)
    try:
        for index,case in enumerate(m['cases']):
            key=f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"
            done=OUT/'case_done'/f'{key}.json'
            if done.exists():continue
            try:
                with lock(OUT/'claims'/f'{key}.lock',blocking=False):
                    if done.exists():continue
                    configs=m['conditions'];offset=index%len(configs)
                    for cfg in configs[offset:]+configs[:offset]:
                        d=OUT/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
                        if (d/'complete.json').exists():continue
                        for attempt in range(2):
                            if d.exists():d.rename(d.with_name(d.name+'.invalid.'+uuid.uuid4().hex))
                            try:
                                result=episode(d,a.port,case,cfg,pool,m['manifest_hash'])
                                print(json.dumps(dict(condition=cfg['id'],case=case,success=result['success'],actions=result['actions'])),flush=True);break
                            except Exception as exc:
                                d.mkdir(parents=True,exist_ok=True);atomic(d/'technical_error.json',dict(error=repr(exc),attempt=attempt+1))
                                if attempt:raise
                    atomic(done,dict(case=case,port=a.port))
            except BlockingIOError:continue
    finally:pool.close()

if __name__=='__main__':main()
