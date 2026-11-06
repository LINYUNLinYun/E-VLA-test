import argparse,json,pickle,socket,struct,time,traceback
import numpy as np
from .common import ROOT,OUT
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

class Policy:
    def __init__(self,b):self.b=b;self.config={};self.reset()
    def reset(self):
        self.pending=None;self.round=0;self.step=0;self.counts=dict(full=0,draft=0,verify=0)
    def configure(self,config):self.config=config;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        tick=time.perf_counter();mask=self.config['skip'];c,t=self.b.prepare(obs,instruction,mask)
        a,r,cost=self.b.sample(c,seed,2);t.update(cost);t['policy_wall_ms']=(time.perf_counter()-tick)*1000
        self.counts['draft' if mask else 'full']+=1
        result=dict(actions=np.asarray(a,np.float32),raw=r.float().cpu().numpy(),arrays={},execute_length=5,
            route=dict(path='pruned' if mask else 'full',skip=mask),timing=t,seed=seed,round=self.round,step=self.step,
            counts=dict(self.counts),full_cache_age=None)
        self.pending=result;self.round+=1;return result
    def acknowledge(self,executed,quaternions):
        if self.pending is None:raise ValueError('No proposal')
        assert np.array_equal(np.asarray(executed,np.float32),self.pending['actions'][:len(executed)])
        self.step+=len(executed);self.pending=None;return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from pathlib import Path
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_E4_single.export import key
    b=TritonBackend();pol=Policy(b);m=json.loads((OUT/'manifest.json').read_text());checks=[]
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text())
    for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
        pair=next(p for p in pairs if p['case']['suite']==suite and p['case']['task']==0 and p['e2'])
        obs,prompt=load_observation(pair['current']);seed=pair['current']['seed']
        for cfg in m['conditions']:
            c,_=b.prepare(obs,prompt,cfg['skip']);aa,rr,_=b.sample(c,seed,2);bb,ss,_=b.sample(c,seed,2)
            assert np.array_equal(aa,bb) and b.torch.equal(rr,ss)
        c,_=b.prepare(obs,prompt,());aa,rr,_=b.sample(c,seed,2)
        ref=b.torch.load(ROOT/'results/idea1_v3_4_E2_E4_single/inputs'/f"{key(pair['current'])}.pt",map_location='cpu',weights_only=False)
        diff=float((rr.float().cpu()-ref['native_full_raw'].float()).abs().max())
        passed=bool(np.allclose(aa,ref['native_full_actions'],atol=1e-5,rtol=1e-5) and b.torch.allclose(rr.float().cpu(),ref['native_full_raw'].float(),atol=1e-5,rtol=1e-5))
        checks.append(dict(suite=suite,passed=passed,raw_max_abs=diff));atomic(out/'gate.json',dict(passed=all(r['passed'] for r in checks),checks=checks))
        if not passed:raise RuntimeError('Native reference mismatch')
    protocol=dict(b.protocol,experiment='E2_max2_reset_repair',D=2,masks=m['masks'])
    atomic(out/'protocol.json',protocol)
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',a.port));srv.listen(8)
        print('READY',flush=True)
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
