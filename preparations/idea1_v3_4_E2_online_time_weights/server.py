import argparse,json,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,OLD,SUITES,CONDITIONS,coefficients
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

class Policy:
    def __init__(self,b):
        from .selector import Selector
        self.b=b;self.selector=Selector(b);self.config=CONDITIONS[0];self.reset()
    def reset(self):self.pending=None;self.round=0;self.step=0;self.counts=dict(full=0,draft=0,verify=0,selector=0)
    def configure(self,config):
        self.selector.clear();self.config=config;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        from scripts.spec.triton import pi0_infer as k
        tick=time.perf_counter();c,t=self.b.prepare(obs,instruction,(),encode=False)
        skip,grad,score,cost=self.selector.run(c,seed,coefficients(self.config));t.update(cost)
        r=c['runtime'];buf=r.buffers
        _,t['vlm_ms']=self.b.adapter.measure(lambda:k.transformer_encoder(r.weights,buf,c['seq'],vlm_layer_indices=tuple(i for i in range(18) if i+1 not in skip)))
        c.update(K=buf['encoder_K'][:,:c['seq']].clone(),V=buf['encoder_V'][:,:c['seq']].clone())
        actions,raw,cost=self.b.sample(c,seed,2);t.update(cost)
        t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms']
        self.counts['draft']+=1;self.counts['selector']+=1
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=dict(gate_gradient=grad,layer_score=score),execute_length=5,
                    route=dict(path='online_gradient_pruned',skip=skip,alpha=self.config['alpha'],direction=self.config['direction']),timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None)
        self.pending=result;self.round+=1;return result
    def acknowledge(self,executed,quaternions):
        if self.pending is None:raise ValueError('No proposal')
        assert np.array_equal(np.asarray(executed,np.float32),self.pending['actions'][:len(executed)])
        self.step+=len(executed);self.pending=None;return {}

def preflight(b,pol,out):
    import torch
    from .selector import current_item,calculate
    from preparations.idea1_v3_4_E2_E4_single.export import key
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_single_action.common import mask
    pairs=json.loads((OLD/'selection.json').read_text());checks=[]
    for suite in SUITES:
        pair=next(p for p in pairs if p['case']['suite']==suite and p['case']['task']==0 and p['e2'])
        obs,prompt=load_observation(pair['current']);seed=pair['current']['seed'];c,_=b.prepare(obs,prompt,(),encode=False)
        item=current_item(c,seed);arch=torch.load(OLD/'inputs'/f"{key(pair['current'])}.pt",map_location='cpu',weights_only=False)
        assert all(torch.equal(item[k].cpu(),arch[k].to(item[k].dtype)) for k in ['x','state','encoder_rope','decoder_rope','noise'])
        for cfg in CONDITIONS:
            chosen,grad,score,cost=pol.selector.run(c,seed,coefficients(cfg))
            W=torch.as_tensor(coefficients(cfg),device='cuda');g=torch.ones(17,device='cuda',requires_grad=True)
            raw,expected=calculate(pol.selector.model,item,W,g)
            actual_raw=pol.selector.raw.float().cpu();actual_grad=torch.as_tensor(grad)
            passed=bool(torch.allclose(raw.float().cpu(),actual_raw,atol=1e-5,rtol=1e-5) and torch.allclose(expected.cpu(),actual_grad,atol=1e-4,rtol=1e-3) and chosen==(mask(abs(expected.cpu().numpy()))+1).tolist())
            checks.append(dict(suite=suite,condition=cfg['id'],passed=passed,raw_max_abs=float((raw.float().cpu()-actual_raw).abs().max()),gradient_max_abs=float((expected.cpu()-actual_grad).abs().max()),skip=chosen,selector_hot_ms=cost['selector_hot_ms']))
            atomic(out/'gate.json',dict(passed=all(x['passed'] for x in checks),checks=checks));assert passed,checks[-1]
        # Native execution stays on the frozen Triton branch.
        c,_=b.prepare(obs,prompt,());aa,rr,_=b.sample(c,seed,2)
        assert np.allclose(aa,arch['native_full_actions'],atol=1e-5,rtol=1e-5) and torch.allclose(rr.float().cpu(),arch['native_full_raw'].float(),atol=1e-5,rtol=1e-5)
        pol.selector.clear();print(json.dumps(dict(preflight_suite=suite,passed=True)),flush=True)
    atomic(out/'gate.json',dict(passed=True,checks=checks,native_reference=True,input_identity=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    b=TritonBackend();pol=Policy(b);preflight(b,pol,out)
    protocol=dict(b.protocol,experiment='E2_online_time_weights',D=2,selection='current-observation Full single VJP every replan',conditions=CONDITIONS,graph=True,deterministic=True)
    atomic(out/'protocol.json',protocol)
    if args.preflight_only:return
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',args.port));srv.listen(8);print('READY',flush=True)
        while True:
            con,_=srv.accept()
            with con:
                try:
                    req=pickle.loads(recv_exact(con,struct.unpack('!I',recv_exact(con,4))[0]));op=req.pop('op')
                    if op=='health':res=dict(protocol=protocol)
                    elif op=='configure':res=pol.configure(**req)
                    elif op=='predict':res=pol.predict(**req)
                    elif op=='ack':res=pol.acknowledge(**req)
                    else:raise ValueError(op)
                    res['ok']=True
                except Exception:res=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(res,protocol=5);con.sendall(struct.pack('!I',len(data))+data)
if __name__=='__main__':main()
