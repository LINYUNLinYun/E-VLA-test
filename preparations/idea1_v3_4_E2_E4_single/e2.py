import argparse,json,time
import numpy as np
import torch
from scipy.stats import spearmanr
from .common import OUT,LAYERS,dump,metrics
from .export import key
from .probe import setup,timing
from .torch_model import Model

GROUPS={'translation':slice(0,3),'rotation':slice(3,6),'gripper':slice(6,7)}

def safe_corr(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if np.std(a)<1e-15 or np.std(b)<1e-15:return None
    return float(spearmanr(a,b).statistic)

def run(model,p):
    d=OUT/'E2'/f"sample{p['id']:03d}";d.mkdir(parents=True,exist_ok=True)
    if (d/'complete.json').exists():return
    item=model.item(OUT/'inputs'/f"{key(p['current'])}.pt")
    with torch.no_grad():
        full=model.encode(item,trace=True);ref=model.sample(item,full)
        _,fwdms=timing(lambda:model.forward(item))
        reff=ref.cpu().numpy();actions=model.decode(item,ref)
        cos=[];resnorm=[]
        for inp,y in zip(full['inputs'][:17],full['outputs'][:17]):
            cos.append(float(torch.nn.functional.cosine_similarity(inp,y,dim=-1).mean()))
            resnorm.append(float((y-inp).norm()/inp.norm()))
    gates=torch.ones(17,device='cuda',requires_grad=True)
    raw,graphms=timing(lambda:model.forward(item,gates=gates))
    target=raw[:5,:7].reshape(-1);jac=[]
    tick=time.perf_counter()
    for i in range(35):jac.append(torch.autograd.grad(target[i],gates,retain_graph=i<34)[0].detach())
    torch.cuda.synchronize();jacms=(time.perf_counter()-tick)*1000
    J=torch.stack(jac).reshape(5,7,17).cpu().numpy()
    del raw,target,jac,gates
    arrays=dict(full_raw=reff,full_actions=actions,jacobian=J);rows=[];rankrows=[];deltas={}
    with torch.no_grad():
        for layer in LAYERS:
            for alpha in (.9,.5,0.):
                gates=torch.ones(17,device='cuda');gates[layer-1]=alpha
                result=model.forward(item,gates=gates)
                if alpha==0:
                    actual=model.forward(item,skip=(layer,))
                    assert torch.allclose(result,actual,atol=1e-5,rtol=1e-5),(p['id'],layer,float((result-actual).abs().max()))
                arr=result.cpu().numpy();name=f'L{layer}_g{alpha:g}';arrays[name]=arr
                decoded=model.decode(item,result);arrays[name+'_actions']=decoded
                delta=arr[:5,:7]-reff[:5,:7];pred=(alpha-1)*J[:,:,layer-1]
                deltas[layer,alpha]=delta
                for group,sl in GROUPS.items():
                    a,b=delta[:,sl].reshape(-1),pred[:,sl].reshape(-1)
                    an,bn=np.linalg.norm(a),np.linalg.norm(b)
                    rows.append(dict(id=p['id'],**p['case'],layer=layer,gate=alpha,group=group,
                        actual_rms=float(np.sqrt(np.mean(a*a))),predicted_rms=float(np.sqrt(np.mean(b*b))),
                        relative_vector_error=float(np.linalg.norm(a-b)/max(an,1e-12)),
                        direction_cosine=float(np.dot(a,b)/max(an*bn,1e-12)),
                        layer_cosine=cos[layer-1],relative_residual=resnorm[layer-1],
                        **metrics(actions,decoded,reff,arr)))
        for alpha in (.9,.5,0.):
            for group,sl in GROUPS.items():
                actual=np.array([np.sqrt(np.mean(deltas[l,alpha][:,sl]**2)) for l in LAYERS])
                predicted=np.sqrt(np.mean(J[:,sl,:]**2,axis=(0,1)))*abs(alpha-1)
                low=np.argsort(predicted)[:4];high=np.argsort(actual)[-4:]
                rankrows.append(dict(id=p['id'],**p['case'],gate=alpha,group=group,
                    jacobian_spearman=safe_corr(actual,predicted),cosine_spearman=safe_corr(actual,1-np.array(cos)),
                    residual_spearman=safe_corr(actual,resnorm),low4_actual_mean=float(actual[low].mean()),
                    all_actual_mean=float(actual.mean()),low4_in_actual_high4=int(len(set(low)&set(high)))))
        projection=[]
        for group,sl in GROUPS.items():
            matrix=J[:,sl,:].reshape(-1,17);actual=np.array([np.linalg.norm(deltas[l,0.][:,sl]) for l in LAYERS])
            for n in (2,4,8):
                rng=np.random.default_rng(340200+n)
                R=rng.standard_normal((n,len(matrix)))/np.sqrt(n)
                norms=np.linalg.norm(R@matrix,axis=0);exact=np.linalg.norm(matrix,axis=0)
                projection.append(dict(id=p['id'],**p['case'],group=group,projections=n,
                    norm_relative_error=float(np.linalg.norm(norms-exact)/max(np.linalg.norm(exact),1e-12)),
                    rank_vs_exact=safe_corr(norms,exact),rank_vs_skip=safe_corr(norms,actual)))
    np.savez_compressed(d/'arrays.npz',**arrays)
    dump(d/'metrics.json',dict(rows=rows,ranks=rankrows,projections=projection,
        forward_ms=fwdms,graph_forward_ms=graphms,backward35_ms=jacms,jacobian_total_ms=graphms+jacms))
    dump(d/'complete.json',dict(id=p['id'],time=time.time(),gate_zero_bypass_checks=17))

def main():
    a=argparse.ArgumentParser();a.add_argument('--shard',type=int,default=0);a.add_argument('--shards',type=int,default=1);args=a.parse_args()
    setup();model=Model(OUT/'model.pt')
    pairs=[p for p in json.loads((OUT/'selection.json').read_text()) if p['e2']]
    for i,p in enumerate(pairs):
        if i%args.shards!=args.shard:continue
        tick=time.perf_counter();run(model,p);print(json.dumps(dict(E2=p['id'],case=p['case'],wall_s=time.perf_counter()-tick)),flush=True)

if __name__=='__main__':main()
