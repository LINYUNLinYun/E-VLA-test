import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback
import numpy as np
from preparations.idea1_v2_1.runtime_client import recv_exact
from preparations.idea1_v3.io import atomic


def gate(backend,out):
    from preparations.kv_reuse_probe.probe import load_observation,mixed_context
    from preparations.idea1_v3.policy import V14
    root=Path(__file__).resolve().parents[2]
    pairs=json.loads((root/'results/kv_reuse_probe/run_20261024_d2_gpu2/selection.json').read_text())
    seen=set(); selected=[]
    for p in pairs:
        suite=p['episode'].split('/')[0]
        if suite not in seen: selected.append(p); seen.add(suite)
    checks=[]; arrays={}; torch=backend.torch
    for idx,p in enumerate(selected):
        obs,instruction=load_observation(p['current']); oldobs,oldinstruction=load_observation(p['previous'])
        if instruction!=oldinstruction: raise ValueError('Gate prompt changed')
        seed=p['current']['seed']
        old,_=backend.prepare(oldobs,instruction,())
        full,_=backend.prepare(obs,instruction,())
        v14,_=backend.prepare(obs,instruction,V14)
        for name,c,steps in (('full10',full,10),('v14_2',v14,2),('full2',full,2)):
            a,r,t=backend.sample(c,seed,steps); aa,rr,tt=backend.sample(c,seed,steps)
            checks.append(dict(input=idx,path=name,repeat=bool(np.array_equal(a,aa) and torch.equal(r,rr))))
            arrays[f'{idx}_{name}_raw']=r.float().cpu().numpy(); arrays[f'{idx}_{name}']=a
        for cutoff in (12,13,14,15,16,17):
            reference=mixed_context(v14,old,cutoff)
            hybrid,_=backend.prepare_hybrid(obs,instruction,old,cutoff)
            kv_equal=all(torch.equal(reference[key],hybrid[key]) for key in ('K','V'))
            a,r,_=backend.sample(reference,seed,2); aa,rr,_=backend.sample(hybrid,seed,2)
            rawmax=float((r.float()-rr.float()).abs().max().cpu()); maxabs=float(np.abs(a-aa).max())
            checks.append(dict(input=idx,cutoff=cutoff,kv_equal=kv_equal,raw_max_abs=rawmax,decoded_max_abs=maxabs,
                passed=bool(kv_equal and np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r,rr,atol=1e-5,rtol=1e-5))))
        a,r,_=backend.sample(full,seed,2)
        aa,rr,_=backend.warm_sample(full,np.zeros((50,32)),seed,q=1.,steps=2)
        checks.append(dict(input=idx,warm_q1_raw_max_abs=float((r.float()-rr.float()).abs().max().cpu()),
            passed=bool(torch.equal(r,rr) and np.array_equal(a,aa))))
        del old,full,v14,reference,hybrid
    passed=all(c.get('passed',c.get('repeat',False)) for c in checks)
    np.savez(out/'gate_actions.npz',**arrays)
    result=dict(passed=passed,checks=checks,atol=1e-5,rtol=1e-5,selection=selected)
    atomic(out/'gate.json',result)
    if not passed: raise RuntimeError('Same-service native/truncation gate failed')
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,required=True); p.add_argument('--output',required=True); a=p.parse_args()
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tick=time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3_4_followup.backend import Backend
    from preparations.idea1_v3_4_followup.policy import Policy
    b=Backend(); pol=Policy(b)
    pol.protocol=dict(b.protocol,experiment='idea1_v3_4_l12_14_first',server_pid=os.getpid(),service_load_s=time.perf_counter()-tick,
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    gate(b,out)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); server.bind(('127.0.0.1',a.port)); server.listen(16)
        print('READY',flush=True)
        while True:
            conn,_=server.accept()
            with conn:
                try:
                    req=pickle.loads(recv_exact(conn,struct.unpack('!I',recv_exact(conn,4))[0])); op=req.pop('op')
                    if op=='health': result=dict(protocol=pol.protocol)
                    elif op=='configure': result=pol.configure(**req)
                    elif op=='predict': result=pol.predict(**req)
                    elif op=='ack': result=pol.acknowledge(**req)
                    elif op=='shadow': result=pol.shadow()
                    elif op=='bench':
                        from .bench import run
                        result=run(b,**req)
                    elif op=='mechanism':
                        from .mechanism import run
                        result=run(b,**req)
                    else: raise ValueError(op)
                    result['ok']=True
                except Exception: result=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(result,protocol=5); conn.sendall(struct.pack('!I',len(data))+data)

if __name__=='__main__': main()
