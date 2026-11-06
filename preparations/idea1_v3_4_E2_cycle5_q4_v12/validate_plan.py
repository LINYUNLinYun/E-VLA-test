import json,tempfile
from pathlib import Path
import numpy as np
import torch
from .common import OUT,CONDITIONS,choose,legal
from .model import calculate,projection_weights
from preparations.idea1_v3.io import atomic

def main():
    from preparations.idea1_v3_4_E2_comp_error_cost.verify import TinyFull
    from preparations.idea1_v3_4_E2_cycle3.common import coefficients
    torch.set_num_threads(4);torch.manual_seed(312);model=TinyFull()
    item=dict(x=torch.randn(4,8,dtype=torch.float64),encoder_rope=None);g=torch.ones(17,dtype=torch.float64,requires_grad=True)
    W=torch.as_tensor(projection_weights(4),dtype=torch.float64)
    assert np.array_equal(projection_weights(4)[0],coefficients()) and np.linalg.matrix_rank(W.numpy().reshape(4,-1))==4
    raw,grads=calculate(model,item,W,g)
    for j in range(4):
        rr,gg=calculate(model,item,W[j:j+1],g);assert torch.equal(rr,raw) and torch.equal(gg[0],grads[j])
    eps=1e-4;fd=np.empty((4,17))
    with torch.no_grad():
        for layer in range(17):
            plus,minus=g.detach().clone(),g.detach().clone();plus[layer]+=eps;minus[layer]-=eps
            delta=(model.forward(item,gates=plus)[:5,:7]-model.forward(item,gates=minus)[:5,:7])/(2*eps)
            fd[:,layer]=(delta[None]*W).sum((1,2)).numpy()
    assert np.allclose(fd,grads.numpy(),atol=1e-8,rtol=1e-5)
    from . import policy as policy
    from preparations.idea1_v3_4_E2_cycle5_spaced import policy as parent
    from preparations.idea1_v3_4_E2_cycle5_spaced.validate_plan import Backend
    class Fake:
        q=4;rule='spaced'
        def __init__(self,b):pass
        def clear(self):pass
        def run(self,c,seed,coeff,k):
            grads=np.cos(np.arange(68).reshape(4,17)+seed).astype(np.float32);score=np.sqrt(np.mean(grads**2,axis=0))
            return choose(score,k,'spaced'),grads,score,dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)
    policy.Selector=Fake
    parent.prepare_full=lambda b,obs,instruction,trace:(dict(hidden_trace=[torch.zeros(2,2)]*19) if trace else {},dict(nodes=[]))
    parent.capture=lambda hidden:{i:torch.zeros(2,2) for i in range(1,18)}
    parent.prepare_patch=lambda b,obs,instruction,skip,last,previous,ratio:({},dict(nodes=[]))
    from .finish import audit_episode
    checks=[]
    with tempfile.TemporaryDirectory(prefix='q4-cycle5-') as tmp:
        for cfg in CONDITIONS:
            pol=policy.Policy(Backend());pol.configure(cfg);qs=[];arrays={};d=Path(tmp)/cfg['id'];d.mkdir()
            for i in range(15):
                r=dict(pol.predict({},'test','libero_spatial',100+i));full=i%5==0
                assert r['route']['steady_vjp_count']==(4 if full else 0) and legal(r['route']['selected_skip'],6,'spaced')
                actions=r.pop('actions');arrays[f'actions_{i}']=actions;arrays[f'raw_{i}']=r.pop('raw')
                for name,value in r.pop('arrays').items():arrays[f'{name}_{i}']=value
                r['timing']['nodes']=[0]*(11 if full else 3)
                qs.append(dict(r,replan=i,input_hash='synthetic',executed_length=5,rpc_ms=0.));pol.acknowledge(actions[:5],None)
            (d/'infer.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs));np.savez(d/'arrays.npz',**arrays)
            (d/'identity.json').write_text(json.dumps(dict(init_vector_sha256='synthetic',bddl_sha256='synthetic',controller={},prompt='test',env_seed=7,settling=10)))
            _,a=audit_episode(cfg,d,dict(predictions=15,actions=75));assert len(a['blocks'])==2
            checks.append(dict(condition=cfg['id'],requests=15,passed=True))
    result=dict(passed=True,finite_difference_max_abs=float(abs(fd-grads.numpy()).max()),independent_directions=True,projection_rank=4,route_and_audit=checks)
    atomic(OUT/'cpu_gate.json',result);print(json.dumps(result))

if __name__=='__main__':main()
