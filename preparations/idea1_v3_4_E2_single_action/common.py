import itertools,json
import numpy as np
from preparations.idea1_v3_4_E2_E4_single.common import ROOT,OUT as OLD,sha,dump
OUT=ROOT/'results/idea1_v3_4_E2_single_action'
PLAN=ROOT/'docs/experiments/idea1/idea1_v3_4_E2_single_action_plan.md'
STEPS=(0,2,4)
COMBOS=np.array([c for c in itertools.combinations(range(17),4) if not any({i,i+1,i+2}<=set(c) for i in range(15))])
def weights(seed=340710):return np.random.default_rng(seed).choice([-1.,1.],(4,7)).astype(np.float32)
def mask(score):return COMBOS[np.argmin(np.asarray(score)[COMBOS].sum(1))]
def references(p):
    with np.load(OLD/'E2'/f"sample{p['id']:03d}"/'arrays.npz') as z:
        full=z['full_raw'].copy();J=z['jacobian'].copy()
        delta=np.stack([z[f'L{l}_g0'][:5,:7]-full[:5,:7] for l in range(1,18)],-1)
    return full,J,delta
def write_csv(path,rows):
    import csv
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
