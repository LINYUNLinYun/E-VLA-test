"""Exercise the selected four-direction fused backend in sync and async modes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v3_4_selection_async.common import OUT
from preparations.idea1_v3_4_selection_async.runtime import samples


def run(port,out):
    m=json.loads((OUT/'optimized_manifest.json').read_text());cfg=m['condition'];out=Path(out)
    out.mkdir(parents=True,exist_ok=True);checks=[]
    for si in (0,2):
        sample=samples()[si];obs,prompt=load_observation(sample['current']);seed=sample['current']['seed'];got={}
        for asynchronous in (False,True):
            rpc(port,dict(op='configure',config=dict(cfg,asynchronous=asynchronous)));rows=[]
            for index in range(3):
                r=rpc(port,dict(op='predict',obs=obs,instruction=prompt,suite=sample['suite'],seed=seed+index));rows.append(r)
                rpc(port,dict(op='ack',executed=r['actions'][:r['execute_length']],
                    quaternions=np.tile([0.,0.,0.,1.],(r['execute_length'],1))))
            terminal=rpc(port,dict(op='finish'));got[asynchronous]=(rows,terminal)
        for i,(a,b) in enumerate(zip(got[False][0],got[True][0])):
            passed=bool(np.allclose(a['raw'],b['raw'],atol=1e-5,rtol=1e-5) and
                np.allclose(a['actions'],b['actions'],atol=1e-5,rtol=1e-5) and
                a['route']['draft_skip']==b['route']['draft_skip'] and a['route']['path']==b['route']['path'] and a['execute_length']==b['execute_length'])
            checks.append(dict(sample=si,request=i,passed=passed,raw_max_abs=float(np.max(np.abs(a['raw']-b['raw'])))))
        def gradients(value):
            result={}
            for row in value[0]+[value[1]]:
                result.update({k:v for k,v in row['arrays'].items() if k.startswith('gate_gradient_job')})
            return result
        a=gradients(got[False]);b=gradients(got[True]);assert a.keys()==b.keys()
        for key in a:
            checks.append(dict(sample=si,key=key,passed=bool(np.allclose(a[key],b[key],atol=1e-4,rtol=1e-3)),
                gradient_max_abs=float(np.max(np.abs(a[key]-b[key])))))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,time=time.time())
    atomic(out/'complete.json',result);assert result['passed'];rpc(port,dict(op='reset_all'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.port,a.output)
