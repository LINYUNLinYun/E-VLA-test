"""Same-service Full-D10 references for predetermined candidate inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .common import OUT,directory,sha
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import rpc

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--suite',required=True);p.add_argument('--task',type=int,required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());case=dict(suite=a.suite,task=a.task,init=19);dst=OUT/'profiles'/a.suite/f'task{a.task:02d}'
    if (dst/'complete.json').exists():return
    start=time.time();rpc(a.port,dict(op='reset_all'));result=[]
    for cfg in [c for c in m['conditions'] if c['mode']=='verified']:
        d=directory(cfg['id'],case);qs=[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()][:15];bank=[]
        for q in qs:
            f=Path(q['input_path']);assert sha(f)==q['input_file_sha256']
            with np.load(f) as z:obs={k:z[k].copy() for k in z.files if k not in ('instruction','seed')};prompt=str(z['instruction'].item());seed=int(z['seed'])
            bank.append(dict(obs=obs,instruction=prompt,seed=seed,suite=a.suite))
        for i in range(20):rpc(a.port,dict(op='profile_full',**bank[i%len(bank)]))
        expected={};timings={i:[] for i in range(len(bank))}
        for trial in range(2):
            for i in (range(len(bank)) if trial==0 else reversed(range(len(bank)))):
                r=rpc(a.port,dict(op='profile_full',**bank[i]));timings[i].append(r['timing']['policy_wall_ms'])
                if i not in expected:expected[i]=(r['raw'].copy(),r['actions'].copy())
                else:assert np.allclose(r['raw'],expected[i][0],atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],expected[i][1],atol=1e-5,rtol=1e-5)
        for i,q in enumerate(qs):result.append(dict(condition=cfg['id'],case=case,replan=i,input_hash=q['input_hash'],seed=q['seed'],executed_length=q['executed_length'],candidate_policy_ms=q['timing']['policy_wall_ms'],candidate_selector_capture_ms=q['timing'].get('selector_capture_ms',0.),full_reference_ms=float(np.mean(timings[i])),full_reference_trials=timings[i],request_category=q['route']['request_category'],gpu=a.gpu))
    atomic(dst/'complete.json',dict(passed=True,rows=result,gpu=a.gpu,unique_inputs_per_condition={cfg['id']:len({r['input_hash'] for r in result if r['condition']==cfg['id']}) for cfg in m['conditions'] if cfg['mode']=='verified'},wall_s=time.time()-start,setup_per_condition=20,measured_passes=2,time=time.time()))

if __name__=='__main__':main()
