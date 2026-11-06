import hashlib,json,time
import numpy as np
import torch
from .common import ROOT,OUT,MASK,STATIC,selection
from .backend import Backend
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_E4_single.fit import scale_fit,lowrank_fit
from preparations.idea1_v3_4_E2_E4_single.torch_model import hadamard
from preparations.kv_reuse_probe.probe import load_observation

def cpu(value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    return value

def run():
    begin=time.perf_counter();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    from preparations.idea1_v3.safe_qk import install
    install(str(OUT/'calibration_qk_receipt.json'))
    b=Backend();train=[p for p in selection() if p['split']=='calibration'];valid=[p for p in selection() if p['split']=='validation']
    cache={p['id']:load_observation(p['current']) for p in train+valid};targets={};indices={};methods={};rows=[]
    with torch.inference_mode():
        for p in train:
            obs,prompt=cache[p['id']];c,_=b.prepare_residual(obs,prompt,traced=MASK)
            n=c['seq'];idx=torch.cat((torch.linspace(0,511,32).round().long(),torch.linspace(512,n-1,min(16,n-512)).round().long())).cuda()
            indices[p['id']]=idx
            targets[p['id']]={l:c['traces'][l]['output'][idx].float().cpu() for l in MASK}
        for kind in STATIC:
            params={};methods[kind]=params
            for layer in MASK:
                xs=[];ys=[]
                for p in train:
                    obs,prompt=cache[p['id']]
                    c,_=b.prepare_residual(obs,prompt,MASK,static=(kind,params),rank=32,traced=(layer,))
                    xs.append(c['traces'][layer]['input'][indices[p['id']]].float().cpu());ys.append(targets[p['id']][layer])
                x,y=torch.cat(xs).cuda(),torch.cat(ys).cuda()
                if kind=='channel':a=scale_fit(x,y)
                elif kind=='hadamard':a=scale_fit(hadamard(x),hadamard(y))
                else:a=lowrank_fit(x,y,341200+layer)
                params[layer]=a
                row=dict(method=kind,layer=layer,tokens=len(x),wall_s=time.perf_counter()-begin)
                rows.append(row);atomic(OUT/'calibration_status.json',dict(status='fitting',completed_layers=len(rows),target_layers=18,latest=row,time=time.time()))
                print(json.dumps(row),flush=True)
                del x,y,c,xs,ys
        b.adapters=methods;b.rank=32;validation=[]
        for p in valid:
            obs,prompt=cache[p['id']];seed=p['current']['seed']
            full,_=b.prepare_residual(obs,prompt);ref,_,_=b.sample(full,seed,2)
            for name,rank in [('pure_v12',None),('linear_r8',8),('linear_r32',32)]:
                c,_=b.prepare_residual(obs,prompt,MASK,kind=None if rank is None else 'linear',rank=rank)
                actions,_,_=b.sample(c,seed,2)
                validation.append(dict(id=p['id'],candidate=name,translation_mm=float(np.linalg.norm(actions[:5,:3]-ref[:5,:3],axis=-1).mean()*50),
                    gripper_sign=float(np.mean((actions[:5,6]<0)!=(ref[:5,6]<0)))))
        scores={name:{k:float(np.mean([r[k] for r in validation if r['candidate']==name])) for k in ('translation_mm','gripper_sign')} for name in ('pure_v12','linear_r8','linear_r32')}
        eligible=[r for r in (8,32) if scores[f'linear_r{r}']['gripper_sign']<=scores['pure_v12']['gripper_sign']]
        rank=min(eligible,key=lambda r:(scores[f'linear_r{r}']['translation_mm'],r)) if eligible else min((8,32),key=lambda r:(scores[f'linear_r{r}']['gripper_sign'],scores[f'linear_r{r}']['translation_mm'],r))
        data=dict(methods=cpu(methods),selected_rank=rank,calibration_ids=[p['id'] for p in train],validation_ids=[p['id'] for p in valid],mask=list(MASK))
        temporary=OUT/'adapters.tmp.pt';torch.save(data,temporary);temporary.replace(OUT/'adapters.pt')
        receipt=dict(passed=True,selected_rank=rank,scores=scores,validation_requests=validation,fit_layers=rows,
            calibration_ids=data['calibration_ids'],validation_ids=data['validation_ids'],
            adapter_sha256=hashlib.sha256((OUT/'adapters.pt').read_bytes()).hexdigest(),wall_s=time.perf_counter()-begin,protocol=b.protocol)
        atomic(OUT/'calibration.json',receipt);atomic(OUT/'calibration_status.json',dict(status='complete',rank=rank,time=time.time()))

if __name__=='__main__':run()
