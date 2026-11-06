import argparse,json,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,CONDITIONS,SUITES,random_mask
from .cosine import Cosine,calculate
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact
from preparations.idea1_v3_4_E2_online_time_weights.server import Policy as GradientPolicy,preflight as gradient_preflight

class Policy(GradientPolicy):
    def __init__(self,b):
        super().__init__(b);self.cosine=Cosine(b);self.config=CONDITIONS[0];self.episode_skip=None
    def configure(self,config):
        self.selector.clear();self.cosine.clear();self.config=config;self.episode_skip=None;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        from scripts.spec.triton import pi0_infer as k
        tick=time.perf_counter();method=self.config['method'];arrays={}
        c,t=self.b.prepare(obs,instruction,(),encode=False)
        if method=='cosine':
            skip,score,cost=self.cosine.run(c,seed);arrays['layer_score']=score
        else:
            start=time.perf_counter()
            if method=='random_online' or self.episode_skip is None:self.episode_skip=random_mask(self.config['random_seed'],seed)
            skip=self.episode_skip;ms=(time.perf_counter()-start)*1000;cost=dict(selector_total_ms=ms,selector_hot_ms=ms,selector_capture_ms=0.)
        t.update(cost);r=c['runtime'];buf=r.buffers
        _,t['vlm_ms']=self.b.adapter.measure(lambda:k.transformer_encoder(r.weights,buf,c['seq'],vlm_layer_indices=tuple(i for i in range(18) if i+1 not in skip)))
        c.update(K=buf['encoder_K'][:,:c['seq']].clone(),V=buf['encoder_V'][:,:c['seq']].clone());actions,raw,cost=self.b.sample(c,seed,2);t.update(cost)
        t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms'];self.counts['draft']+=1;self.counts['selector']+=1
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=arrays,execute_length=5,route=dict(path=method,skip=skip),timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None)
        self.pending=result;self.round+=1;return result

def preflight(b,pol,out):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_single_action.common import mask
    from preparations.idea1_v3_4_E2_online_time_weights.selector import current_item
    gradient_preflight(b,pol,out)
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text());checks=[]
    for suite in SUITES:
        pair=next(p for p in pairs if p['e2'] and p['case']['suite']==suite and p['case']['task']==0)
        obs,prompt=load_observation(pair['current']);c,_=b.prepare(obs,prompt,(),encode=False);item=current_item(c,pair['current']['seed'])
        chosen,score,cost=pol.cosine.run(c,pair['current']['seed']);expected=calculate(pol.cosine.model,item).cpu().numpy()
        passed=bool(np.allclose(score,expected,atol=1e-5,rtol=1e-5) and chosen==(mask(expected)+1).tolist());assert passed
        checks.append(dict(suite=suite,passed=passed,max_abs=float(abs(score-expected).max()),skip=chosen,hot_ms=cost['selector_hot_ms']));pol.cosine.clear()
        for root in [341101,341102,341103]:
            a=random_mask(root,pair['current']['seed']);assert a==random_mask(root,pair['current']['seed']) and len(a)==4 and not any({i,i+1,i+2}<=set(a) for i in range(1,16))
    atomic(out/'cosine_gate.json',dict(passed=True,checks=checks,random_legal_deterministic=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    b=TritonBackend();pol=Policy(b);preflight(b,pol,out);protocol=dict(b.protocol,experiment='E2_selection_baselines',D=2,conditions=CONDITIONS)
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
