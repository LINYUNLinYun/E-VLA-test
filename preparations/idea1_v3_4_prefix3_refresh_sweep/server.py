import argparse,json,os,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from .common import OUT,OLD,MASK
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact
from preparations.idea1_v3_4_residual_replay_quick.backend import Backend

def gate(b,out):
    from preparations.idea1_v3_4_residual_replay_quick.server import selection
    from preparations.kv_reuse_probe.probe import load_observation
    checks=[];arrays={};torch=b.torch
    for p in selection()[::10]:
        s=p['case']['suite'];obs,prompt=load_observation(p['current']);oldobs,oldprompt=load_observation(p['previous']);seed=p['current']['seed']
        native,_=b.prepare(obs,prompt,());a,r,_=b.sample(native,seed,10)
        cap,_=b.prepare_replay(obs,prompt,capture_layers=MASK);aa,rr,_=b.sample(cap,seed,10)
        passed=all(torch.equal(native[k],cap[k]) for k in ('K','V')) and np.array_equal(a,aa) and torch.equal(r,rr)
        checks.append(dict(suite=s,kind='full_capture',passed=bool(passed)));assert passed
        hist,_=b.prepare_replay(oldobs,oldprompt,capture_layers=MASK)
        c,_=b.prepare_replay(obs,prompt,skip=MASK,history=hist['residuals']);a,r,_=b.sample(c,seed,2);aa,rr,_=b.sample(c,seed,2)
        with np.load(OLD/'workers/gpu0/gate_actions.npz') as ref:expected=ref[s+'_prefix3_reuse_raw'];expected_a=ref[s+'_prefix3_reuse']
        passed=bool(np.array_equal(a,aa) and torch.equal(r,rr) and np.allclose(expected,r.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(expected_a,a,atol=1e-5,rtol=1e-5))
        checks.append(dict(suite=s,kind='prior_reference_and_repeat',passed=passed));assert passed
        arrays[s+'_raw']=r.float().cpu().numpy();arrays[s]=a
    np.savez(out/'gate_actions.npz',**arrays);atomic(out/'gate.json',dict(passed=True,checks=checks,atol=1e-5,rtol=1e-5))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    b=Backend();pol=Policy(b);gate(b,out)
    protocol=dict(b.protocol,experiment='prefix3_refresh_sweep',skip=MASK,periods=[2,4,6],server_pid=os.getpid(),service_load_and_gate_s=time.perf_counter()-start)
    atomic(out/'protocol.json',protocol)
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',a.port));srv.listen(8);print('READY',flush=True)
        while True:
            con,_=srv.accept()
            with con:
                try:
                    req=pickle.loads(recv_exact(con,struct.unpack('!I',recv_exact(con,4))[0]));op=req.pop('op')
                    if op=='health':r=dict(protocol=protocol)
                    elif op=='configure':r=pol.configure(**req)
                    elif op=='predict':r=pol.predict(**req)
                    elif op=='ack':r=pol.acknowledge(**req)
                    else:raise ValueError(op)
                    r['ok']=True
                except Exception:r=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(r,protocol=5);con.sendall(struct.pack('!I',len(data))+data)
if __name__=='__main__':main()
