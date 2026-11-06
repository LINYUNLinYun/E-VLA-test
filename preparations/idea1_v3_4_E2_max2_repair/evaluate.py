import argparse,json,random
import numpy as np
from .common import OUT
from preparations.idea1_v3_4.rollout import episode
from preparations.idea1_v3_1.rollout import Environments
class Fresh(Environments):
    def __init__(self):super().__init__(reuse=False)
    def acquire(self,suite,task):
        random.seed(7);np.random.seed(7)
        return super().acquire(suite,task)
def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--shard',type=int,required=True);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());case=m['cases'][a.shard];configs=m['conditions'];configs=configs[a.shard:]+configs[:a.shard];pool=Fresh()
    try:
        for c in configs:
            d=OUT/c['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            if (d/'complete.json').exists():continue
            r=episode(d,a.port,case,c,pool,m['manifest_hash']);print(json.dumps(dict(condition=c['id'],case=case,success=r['success'])),flush=True)
    finally:pool.close()
if __name__=='__main__':main()
