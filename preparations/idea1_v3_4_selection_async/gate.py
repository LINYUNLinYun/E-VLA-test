"""Same observations/noise and acknowledged actions, sync versus async policy."""
import argparse,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v3.io import atomic
from .common import conditions,OUT
from .runtime import samples


def run(port,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);checks=[];saved={}
    # Fixed bank samples cover distinct prompts and suites, with unchanged input.
    for si in (0,2):
        sample=samples()[si];obs,prompt=load_observation(sample['current']);base_seed=sample['current']['seed']
        for q in (4,2,1):
            template=next(c for c in conditions() if c['id']==('v12_p3_q4_serial_async' if q==4 else f'v12_p3_q{q}_layer'))
            outputs={}
            for asynchronous in (False,True):
                rpc(port,dict(op='configure',config=dict(template,asynchronous=asynchronous)))
                rows=[]
                for index in range(3):
                    r=rpc(port,dict(op='predict',obs=obs,instruction=prompt,suite=sample['suite'],seed=base_seed+index))
                    rows.append(r)
                    rpc(port,dict(op='ack',executed=r['actions'][:r['execute_length']],
                        quaternions=np.tile(np.array([0.,0.,0.,1.]),(r['execute_length'],1))))
                terminal=rpc(port,dict(op='finish'));outputs[asynchronous]=(rows,terminal)
            for index,(a,b) in enumerate(zip(outputs[False][0],outputs[True][0])):
                check=dict(sample=si,q=q,request=index,raw_max_abs=float(np.max(np.abs(a['raw']-b['raw']))),
                    actions_max_abs=float(np.max(np.abs(a['actions']-b['actions']))),
                    raw_passed=bool(np.allclose(a['raw'],b['raw'],atol=1e-5,rtol=1e-5)),
                    actions_passed=bool(np.allclose(a['actions'],b['actions'],atol=1e-5,rtol=1e-5)),
                    path_passed=a['route']['path']==b['route']['path'],
                    execute_passed=a['execute_length']==b['execute_length'],
                    skip_passed=a['route']['draft_skip']==b['route']['draft_skip'])
                check['passed']=all(v for k,v in check.items() if k.endswith('_passed'));checks.append(check)
            def gradients(value):
                rows,terminal=value;d={}
                for row in rows+[terminal]:
                    for key,array in row['arrays'].items():
                        if key.startswith('gate_gradient_'):d[key]=array
                return d
            aa=gradients(outputs[False]);bb=gradients(outputs[True]);assert aa.keys()==bb.keys()
            for key in aa:
                checks.append(dict(sample=si,q=q,kind=key,gradient_max_abs=float(np.max(np.abs(aa[key]-bb[key]))),
                    passed=bool(np.allclose(aa[key],bb[key],atol=1e-4,rtol=1e-3))))
                saved[f'sample{si}_q{q}_{key}']=bb[key]
            atomic(out/'checks.json',dict(passed=all(c['passed'] for c in checks),checks=checks,time=time.time()))
            print(json.dumps(dict(sample=si,q=q,passed=all(c['passed'] for c in checks))),flush=True)
    np.savez(out/'gradients.npz',**saved)
    result=dict(passed=all(c['passed'] for c in checks),checks=checks,time=time.time())
    atomic(out/'complete.json',result);assert result['passed'];rpc(port,dict(op='reset_all'));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',required=True,type=int);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.port,a.output)
