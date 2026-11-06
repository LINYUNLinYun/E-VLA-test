"""Auxiliary same-input B1 module/request timing, never promotion evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.cli import verify_gpu
from .dispatch import OUT,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',choices=['pi0','xiaomi'],required=True);a=p.parse_args()
    device_identity=verify_gpu(a.gpu)
    leases=json.loads((OUT/'external_gpu_leases.json').read_text())
    if leases.get(str(a.gpu),{}).get('active'):raise ValueError('Externally leased GPU')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    bankpath=ROOT/'results/idea1_v3_1/input_bank.pkl';bank=pickle.loads(bankpath.read_bytes());dest=OUT/'request_bank';dest.mkdir(exist_ok=True)
    load_start=time.perf_counter()
    if a.model=='pi0':
        from preparations.idea1_v3.safe_qk import install
        install(str(dest/'qk_receipt.json'))
        from .backends import PiBackend
        backend=PiBackend();paths=[('full_d10',(),10),('full_d2',(),2),('v14_d2',(3,4,9,11),2),('v14_d1',(3,4,9,11),1)]
    else:
        from .backends import XiaomiBackend
        from .policy import XMASK
        backend=XiaomiBackend();paths=[('full_d2',(),2),('skip12_d2',XMASK,2)]
    load_s=time.perf_counter()-load_start
    torch=backend.torch;rows=[]
    def measure(item,name,skip,steps):
        torch.cuda.synchronize();start=time.perf_counter();c,prep=backend.prepare(item['obs'],item['instruction'],skip);actions,raw,cost=backend.sample(c,item['seed'],steps);torch.cuda.synchronize()
        elapsed=(time.perf_counter()-start)*1000
        return c,raw,dict(path=name,wall_ms=elapsed,prepare=prep,sample=cost,decoded_sha256=hashlib.sha256(actions.tobytes()).hexdigest())
    # Untimed model warmup is separate; measured raw wall still retains any
    # prompt/input-shape setup encountered by the fixed bank itself.
    warm=time.perf_counter()
    seen=set()
    for item in bank:
        if item['instruction'] in seen:continue
        seen.add(item['instruction'])
        for name,skip,steps in paths:measure(item,name,skip,steps)
        fc,_,_=measure(item,*next(path for path in paths if path[0]=='full_d2'))
        dc,dr,_=measure(item,*next(path for path in paths if path[0] in ('v14_d2','skip12_d2')))
        backend.verify(fc,dc,dr,item['seed'],item['suite']);torch.cuda.synchronize()
        del fc,dc,dr
    warm_s=time.perf_counter()-warm
    for index,item in enumerate(bank):
        full=None;draft=None;raw=None;results=[]
        torch.cuda.synchronize();setup_start=time.perf_counter()
        common,_=backend.prepare(item['obs'],item['instruction'],())
        torch.cuda.synchronize();common_setup_ms=(time.perf_counter()-setup_start)*1000
        del common
        # Always reconstruct Full then draft on the SAME observation/noise.
        # This is a request/module benchmark, not a historical-policy route.
        for name,skip,steps in paths:
            c,r,result=measure(item,name,skip,steps);results.append(result)
            if name=='full_d2':full=c
            if name in ('v14_d2','skip12_d2'):draft=c;raw=r.detach().clone()
        torch.cuda.synchronize();start=time.perf_counter();hats,timing=backend.verify(full,draft,raw,item['seed'],item['suite']);torch.cuda.synchronize()
        results.append(dict(path='verify_same_input_draft2',wall_ms=(time.perf_counter()-start)*1000,components=timing,reconstruction_sha256=hashlib.sha256(np.asarray(hats).tobytes()).hexdigest()))
        rows.append(dict(index=index,input_hash=item['input_hash'],suite=item['suite'],seed=item['seed'],source=item['source'],common_input_setup_ms=common_setup_ms,measurements=results))
        del full,draft,raw,c,r,hats
    names=[x['path'] for x in rows[0]['measurements']]
    atomic(dest/f'{a.model}_gpu{a.gpu}.json',dict(model=a.model,physical_gpu=a.gpu,device_identity=device_identity,B=1,n=len(bank),model_load_s=load_s,warmup_s=warm_s,
        backend_protocol=backend.protocol,environment=dict(CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],torch_version=torch.__version__,cuda_version=torch.version.cuda,python_executable=__import__('sys').executable),
        scope='Auxiliary same-observation same-seed backend request/module timings only. Each input receives one separate common Full preparation first, whose wall is reported separately; this prevents first-path prompt construction bias. Measured paths then include their own native transforms/vision/VLM/sample. Full KV is built from that same observation, not prior closed-loop history. Not a full-policy route, closed-loop speed or promotion gate.',
        common_input_setup_mean_ms=float(np.mean([r['common_input_setup_ms'] for r in rows])),
        bank_sha256=hashlib.sha256(bankpath.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        summary={name:dict(mean_ms=float(np.mean([next(m for m in r['measurements'] if m['path']==name)['wall_ms'] for r in rows]))) for name in names},rows=rows))


if __name__=='__main__':main()
