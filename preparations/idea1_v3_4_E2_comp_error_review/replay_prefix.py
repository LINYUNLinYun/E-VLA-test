"""Diagnostic action replay only; never substitutes original outcomes."""
import argparse,json,random
from pathlib import Path
import numpy as np
from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
from preparations.idea1_v2_1.runtime_client import official_init_states,input_digest

ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'results/idea1_v3_4_E2_cycle3_extrapolate/v14_jacobian_extrapolate/episodes/libero_spatial/task07/init19'
    qs=[json.loads(s) for s in (source/'infer.jsonl').read_text().splitlines()]
    with np.load(source/'arrays.npz') as z:actions=np.concatenate([z[f'actions_{i}'][:5] for i in range(4)])
    pool=Fresh();records=[]
    try:
        for repeat in range(2):
            env,definition,bddl,selected,created=pool.acquire('libero_spatial',7)
            try:
                vector=np.asarray(official_init_states(selected,7)[19]);random.seed(7);np.random.seed(7);env.seed(7);env.reset();obs=env.set_init_state(vector.copy())
                for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                for step in range(21):
                    if step%5==0:
                        model_obs={k:np.asarray(v).copy() for k,v in obs.items() if isinstance(v,np.ndarray)}
                        digest=input_digest(model_obs,definition.language);records.append(dict(repeat=repeat,step=step,input_hash=digest,old_hash=qs[step//5]['input_hash'],eef_pos=np.asarray(obs['robot0_eef_pos']).tolist()))
                        np.savez(out/f'repeat{repeat}_step{step:02d}.npz',**model_obs)
                    if step<20:obs,_,_,_=env.step(actions[step].tolist())
            finally:pool.release('libero_spatial',7)
    finally:pool.close()
    (out/'records.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records),flush=True)
if __name__=='__main__':main()
