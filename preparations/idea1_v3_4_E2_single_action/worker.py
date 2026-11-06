import argparse,json,time
import numpy as np
import torch
from scipy.stats import spearmanr
from .common import OUT,OLD,STEPS,weights,mask,references,dump,write_csv
from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_E4_single.probe import setup,timing
from preparations.idea1_v3_4_E2_E4_single.export import key
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dtype',choices=['fp32','bf16'],required=True);args=parser.parse_args()
    setup();model=tm.Model(OLD/'model.pt');dtype=args.dtype
    if dtype=='bf16':
        for k,v in model.w.items():model.w[k]=v.to(torch.bfloat16)
        for k in ('biases','outw','outb'):setattr(model,k,[v.to(torch.bfloat16) for v in getattr(model,k)])
        torch.cuda.empty_cache();tm.rms=lambda x:(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)
    pairs=[p for p in json.loads((OLD/'selection.json').read_text()) if p['e2'] and p['case']['task'] in (0,2)]
    methods=[('forward',-1,0),('full35',-1,35)]+[(name,s,n) for s in STEPS for name,n in [('project1',1),('project4',4),('exact7',7)]]
    for p in pairs:
        directory=OUT/dtype/f"sample{p['id']:03d}";directory.mkdir(parents=True,exist_ok=True)
        if (directory/'complete.json').exists():continue
        item=model.item(OLD/'inputs'/f"{key(p['current'])}.pt");rawref,J,delta=references(p)
        if dtype=='bf16':
            for k in ('x','state','encoder_rope','decoder_rope','noise'):item[k]=item[k].to(torch.bfloat16)
        R=torch.as_tensor(weights(),device='cuda');data=[];arrays={};checks=[];quality=[]
        def call(method,step,n):
            g=torch.ones(17,device='cuda',requires_grad=n>0);t=time.perf_counter()
            with torch.set_grad_enabled(n>0):raw=model.forward(item,gates=g if n else None)
            torch.cuda.synchronize();fw=(time.perf_counter()-t)*1000
            if n:
                y=raw[:5,:7].float().reshape(-1) if method=='full35' else raw[step,:7].float()
                values=R[:n]@y if method.startswith('project') else y
                grads=[torch.autograd.grad(values[j],g,retain_graph=j<n-1)[0].detach() for j in range(n)]
                G=torch.stack(grads).float();score=G.square().sum(0).sqrt();grad=G.cpu().numpy();selected=mask(score.cpu().numpy())
            else:grad=np.empty((0,17),np.float32);selected=np.array([],int)
            torch.cuda.synchronize()
            return raw.detach().float().cpu().numpy(),grad,selected,fw
        # Warm actual methods/shapes before measurements.
        for method,s,n in methods:call(method,s,n)
        for rep in range(3):
            order=methods if rep%2==0 else list(reversed(methods))
            for method,s,n in order:
                (raw,G,chosen,fw),total=timing(lambda:call(method,s,n))
                data.append(dict(id=p['id'],**p['case'],dtype=dtype,method=method,step=s+1,rep=rep,vjp=n,total_ms=total,forward_ms=fw,added_ms=total-fw,mask_1based=str((chosen+1).tolist())))
                if rep==0:
                    name=f'{method}_step{s+1}';arrays[name+'_gradient']=G;arrays[name+'_raw']=raw
                    if dtype=='fp32':assert np.allclose(raw,rawref,atol=1e-5,rtol=1e-5)
                    if n:
                        exact=J.reshape(35,17) if method=='full35' else J[s]
                        expected=weights()[:n]@exact if method.startswith('project') else exact
                        if dtype=='fp32':assert np.allclose(G,expected,atol=1e-4,rtol=1e-3),(name,np.max(np.abs(G-expected)))
                        checks.append(dict(method=method,step=s+1,gradient_relative_error=float(np.linalg.norm(G-expected)/np.linalg.norm(expected)),raw_prefix7_rms=float(np.sqrt(np.mean((raw[:5,:7]-rawref[:5,:7])**2)))))
                        score=np.linalg.norm(G,axis=0);a5=np.linalg.norm(delta.reshape(35,17),axis=0)
                        for target_step in STEPS:
                            a=np.linalg.norm(delta[target_step],axis=0)
                            quality.append(dict(id=p['id'],**p['case'],dtype=dtype,method=method,step=s+1,target_step=target_step+1,
                                rho_step=float(spearmanr(score,a).statistic),rho_R5=float(spearmanr(score,a5).statistic),selected_step_effect_ratio=float(a[chosen].mean()/a.mean())))
        assert len(data)==len(methods)*3 and np.isfinite([r['total_ms'] for r in data]).all()
        write_csv(directory/'latency.csv',data);write_csv(directory/'quality.csv',quality);dump(directory/'checks.json',checks)
        np.savez_compressed(directory/'arrays.npz',**arrays);dump(directory/'complete.json',dict(passed=True,id=p['id'],dtype=dtype,time=time.time()))
        print(json.dumps(dict(dtype=dtype,completed=p['id'],time=time.time())),flush=True)
if __name__=='__main__':main()
