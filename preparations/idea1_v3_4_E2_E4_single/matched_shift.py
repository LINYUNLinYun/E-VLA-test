"""Matched randomized-SVD seeds for the input-distribution robustness control."""
import json,time
import numpy as np
import torch
from .common import ROOT,OUT,LAYERS,dump,sha
from .export import key
from .torch_model import Model
from .probe import setup
from .fit import lowrank_fit
from .e4 import todevice,make_patch
from .finish import write_csv,balanced

def main():
    setup();directory=OUT/'matched_shift';directory.mkdir(exist_ok=True)
    plan=ROOT/'docs/experiments/idea1/idea1_v3_4_E4_matched_shift_plan.md'
    pairs=json.loads((OUT/'selection.json').read_text());cal=[p for p in pairs if p['split']=='calibration']
    manifest=dict(plan_sha256=sha(plan),selection_sha256=sha(OUT/'selection.json'),main_manifest_sha256=sha(OUT/'manifest.json'),
        calibration_ids=[p['id'] for p in cal],seed_rule='340400+layer, identical to Full-input fit',rank=[8,32],
        sources={name:sha(Path(__file__).parent/name) for name in ('matched_shift.py','fit.py','torch_model.py','e4.py')},time=time.time())
    dump(directory/'manifest.json',manifest)
    model=Model(OUT/'model.pt');original=todevice(torch.load(OUT/'adapters.pt',map_location='cpu',weights_only=False))
    data={l:dict(x=[],y=[]) for l in LAYERS};start=time.perf_counter()
    with torch.no_grad():
        for p in cal:
            item=model.item(OUT/'inputs'/f"{key(p['current'])}.pt")
            full=model.encode(item,trace=True);shift=model.encode(item,skip=(3,4,9,11),trace=True)
            n=len(item['x']);idx=torch.cat((torch.linspace(0,511,32).round().long(),torch.linspace(512,n-1,min(16,n-512)).round().long())).cuda()
            for l in LAYERS:
                data[l]['x'].append(shift['inputs'][l-1][idx].cpu());data[l]['y'].append(full['outputs'][l-1][idx].cpu())
        matched={};controls=[]
        for l in LAYERS:
            a=lowrank_fit(torch.cat(data[l]['x']).cuda(),torch.cat(data[l]['y']).cuda(),340400+l)
            matched[l]=dict(linear=original[l]['linear'],shiftlinear=a)
            if l<=3:
                other=original[l]['linear']
                # SVD factor signs are not identifiable: compare represented maps.
                differences={k:float((a[k]-other[k]).abs().max()) for k in ('input_mean','residual_mean')}
                for rank in (8,32):
                    differences[f'map_rank{rank}']=float((a['left'][:,:rank]@a['right'][:rank]-other['left'][:,:rank]@other['right'][:rank]).abs().max())
                controls.append(dict(layer=l,differences=differences,passed=all(v<1e-5 for v in differences.values())))
                assert controls[-1]['passed'],controls[-1]
            del data[l]
        def cpu(x):
            if torch.is_tensor(x):return x.cpu()
            if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
            return x
        torch.save(cpu(matched),directory/'adapters.pt')
        rows=[]
        for j,p in enumerate(pairs):
            item=model.item(OUT/'inputs'/f"{key(p['current'])}.pt")
            full=model.encode(item,trace=True);shift=model.encode(item,skip=(3,4,9,11),trace=True)
            for l in LAYERS:
                target=full['outputs'][l-1]
                for domain,x in [('full',full['inputs'][l-1]),('shifted',shift['inputs'][l-1])]:
                    for name in ('skip','linear_r8','linear_r32','shiftlinear_r8','shiftlinear_r32'):
                        y=x if name=='skip' else make_patch(name,matched[l])(x)
                        rows.append(dict(id=p['id'],**p['case'],split=p['split'],layer=l,domain=domain,candidate=name,
                            hidden_mse=float((y-target).square().mean()),hidden_relative_l2=float((y-target).norm()/target.norm())))
            if j%20==0:print(json.dumps(dict(matched_shift=j+1,total=160,elapsed_s=time.perf_counter()-start)),flush=True)
    write_csv(directory/'requests.csv',rows)
    summary=[]
    for l in LAYERS:
        for domain in ('full','shifted'):
            for name in ('skip','linear_r8','linear_r32','shiftlinear_r8','shiftlinear_r32'):
                rs=[r for r in rows if r['layer']==l and r['domain']==domain and r['candidate']==name and r['split']=='heldout']
                summary.append(dict(layer=l,domain=domain,candidate=name,requests=len(rs),hidden_mse=balanced(rs,'hidden_mse'),hidden_relative_l2=balanced(rs,'hidden_relative_l2')))
    write_csv(directory/'heldout_summary.csv',summary)
    assert len(rows)==160*17*2*5 and all(np.isfinite(r['hidden_mse']) for r in rows)
    dump(directory/'audit.json',dict(passed=True,calibration_requests=80,total_requests=160,heldout_requests=48,
        negative_controls=controls,rows=len(rows),matched_seed=True,wall_s=time.perf_counter()-start,time=time.time()))

from pathlib import Path
if __name__=='__main__':main()
