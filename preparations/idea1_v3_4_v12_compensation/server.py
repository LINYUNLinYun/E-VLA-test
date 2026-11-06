import argparse,hashlib,json,os,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,SUITES,MASK,STATIC
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def selection():
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text())
    return [next(p for p in pairs if p['case']['suite']==s and p['case']['task']==t and p['e2']) for s in SUITES for t in range(10)]

def gate(b,out):
    from preparations.kv_reuse_probe.probe import load_observation
    checks=[];arrays={};torch=b.torch
    def check(name,a,r,aa,rr):
        diff=float((r.float()-rr.float()).abs().max().cpu())
        passed=bool(np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5))
        checks.append(dict(name=name,passed=passed,raw_max_abs=diff,decoded_max_abs=float(np.abs(a-aa).max())))
        if not passed:raise RuntimeError('Engineering comparison failed: '+name)
    for p in selection()[::10]:
        s=p['case']['suite'];obs,prompt=load_observation(p['current']);seed=p['current']['seed']
        hist=[]
        for which in ('previous2','previous'):
            oo,pp=load_observation(p[which]);c,_=b.prepare_residual(oo,pp,capture=True);hist.append(c['residuals'])
        for name,skip,D in [('full',(),10),('v12',MASK,2)]:
            native,_=b.prepare(obs,prompt,skip);a,r,_=b.sample(native,seed,D)
            current,_=b.prepare_residual(obs,prompt,skip,capture=name=='full');aa,rr,_=b.sample(current,seed,D)
            check(s+'_'+name,a,r,aa,rr)
            kv=all(torch.equal(native[k],current[k]) for k in ('K','V'))
            checks.append(dict(name=s+'_'+name+'_kv',passed=kv));assert kv
            if name=='full':
                # A diagnostic trace is used only in this bounded engineering check.
                from scripts.spec.triton import pi0_infer as k
                original=k.transformer_encoder;trace=[]
                def traced(w,buff,seq,**kwargs):return original(w,buff,seq,hidden_trace=trace,**kwargs)
                k.transformer_encoder=traced
                try:b.prepare(obs,prompt,())
                finally:k.transformer_encoder=original
                exact=all(torch.equal(current['residuals'][l],trace[l].float()-trace[l-1].float()) for l in MASK)
                checks.append(dict(name=s+'_captured_residual',passed=exact));assert exact
            arrays[s+'_'+name+'_raw']=rr.float().cpu().numpy();arrays[s+'_'+name]=aa
        for kind in STATIC:
            c,_=b.prepare_residual(obs,prompt,MASK,kind=kind);a,r,_=b.sample(c,seed,2)
            cc,_=b.prepare_residual(obs,prompt,MASK,kind=kind);aa,rr,_=b.sample(cc,seed,2)
            check(s+'_'+kind+'_repeat',a,r,aa,rr)
            arrays[s+'_'+kind+'_raw']=rr.float().cpu().numpy();arrays[s+'_'+kind]=aa
        ratio=(p['current']['step']-p['previous']['step'])/(p['previous']['step']-p['previous2']['step'])
        for name,older in [('reuse',None),('extrapolate',hist[0])]:
            c,_=b.prepare_residual(obs,prompt,MASK,history=hist[1],older=older,ratio=ratio,reference=True)
            a,r,_=b.sample(c,seed,2)
            cc,_=b.prepare_residual(obs,prompt,MASK,history=hist[1],older=older,ratio=ratio)
            aa,rr,_=b.sample(cc,seed,2);check(s+'_'+name,a,r,aa,rr)
            aaa,rrr,_=b.sample(cc,seed,2);check(s+'_'+name+'_repeat',aa,rr,aaa,rrr)
            arrays[s+'_'+name+'_raw']=rr.float().cpu().numpy();arrays[s+'_'+name]=aa
    np.savez(out/'gate_actions.npz',**arrays)
    atomic(out/'gate.json',dict(passed=all(c['passed'] for c in checks),checks=checks,atol=1e-5,rtol=1e-5))

def benchmark(b,out):
    from .timing import benchmark as run
    return run(b,out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from .backend import Backend
    from .policy import Policy
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    b=Backend();b.load_adapters();pol=Policy(b);gate(b,out)
    protocol=dict(b.protocol,experiment='v12_compensation',adapter_sha256=hashlib.sha256((OUT/'adapters.pt').read_bytes()).hexdigest(),selected_rank=b.rank,conditions=json.loads((OUT/'manifest.json').read_text())['conditions'],
        service_load_and_gate_s=time.perf_counter()-begin,server_pid=os.getpid())
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
                    elif op=='benchmark':res=benchmark(b,out)
                    else:raise ValueError(op)
                    res['ok']=True
                except Exception:res=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(res,protocol=5);con.sendall(struct.pack('!I',len(data))+data)

if __name__=='__main__':main()
