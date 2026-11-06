import argparse,json,os,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OLD,OUT
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def isolation_gate(b,out,m):
    from preparations.idea1_v3_4_v12_compensation.server import selection
    from preparations.kv_reuse_probe.probe import load_observation
    items=[load_observation(p['current']) for p in selection()[::10]];checks=[]
    configs=[next(c for c in m['conditions'] if c['id']==name) for name in ('periodic_reuse','periodic_extrapolate','periodic_hadamard','periodic_linear')]
    for pair in range(2):
        cfgs=configs[2*pair:2*pair+2];inputs=items[2*pair:2*pair+2];expected=[]
        for actor in range(2):
            pol=Policy(b);pol.configure(cfgs[actor]);raw=[]
            for i in range(6):
                obs,prompt=inputs[actor];r=pol.predict(obs,prompt,'gate',1000+actor*100+i)
                raw.append((r['raw'].copy(),r['actions'].copy()));pol.acknowledge(r['actions'][:5],None)
            expected.append(raw)
        policies=[Policy(b),Policy(b)]
        for i,p in enumerate(policies):p.configure(cfgs[i])
        for step in range(6):
            for actor in range(2):
                obs,prompt=inputs[actor];r=policies[actor].predict(obs,prompt,'gate',1000+actor*100+step)
                raw,actions=expected[actor][step]
                passed=bool(np.allclose(raw,r['raw'],atol=1e-5,rtol=1e-5) and np.allclose(actions,r['actions'],atol=1e-5,rtol=1e-5))
                checks.append(dict(pair=pair,actor=actor,step=step,passed=passed,raw_max_abs=float(np.abs(raw-r['raw']).max())))
                policies[actor].acknowledge(r['actions'][:5],None)
    passed=all(r['passed'] for r in checks)
    atomic(out/'isolation_gate.json',dict(passed=passed,checks=checks))
    return passed

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3_4_v12_compensation.backend import Backend
    from preparations.idea1_v3_4_v12_compensation.server import gate
    b=Backend();b.load_adapters();m=json.loads((OUT/'manifest.json').read_text())
    gate(b,out);isolated=isolation_gate(b,out,m)
    protocol=dict(b.protocol,experiment='v12_compensation_200',conditions=m['conditions'],selected_rank=b.rank,
        adapter_sha256=m['adapter_sha256'],server_pid=os.getpid(),service_load_and_gate_s=time.perf_counter()-start,
        environment_workers=2 if isolated else 1,service_mode='serial B1 inference; isolated policy clients')
    atomic(out/'protocol.json',protocol);policies={}
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',a.port));srv.listen(16)
        print('READY',flush=True)
        while True:
            con,_=srv.accept()
            with con:
                try:
                    req=pickle.loads(recv_exact(con,struct.unpack('!I',recv_exact(con,4))[0]));op=req.pop('op');client=req.pop('client','default')
                    if op=='health':res=dict(protocol=protocol)
                    else:
                        if client not in policies:policies[client]=Policy(b)
                        pol=policies[client]
                        if op=='configure':res=pol.configure(**req)
                        elif op=='predict':res=pol.predict(**req)
                        elif op=='ack':res=pol.acknowledge(**req)
                        else:raise ValueError(op)
                    res['ok']=True
                except Exception:res=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(res,protocol=5);con.sendall(struct.pack('!I',len(data))+data)

if __name__=='__main__':main()
