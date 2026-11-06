import argparse,json,os,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact
from .common import OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3_4.backend import Backend
    from preparations.idea1_v3_4.server import gate
    from preparations.kv_reuse_probe.probe import load_observation
    b=Backend();g=gate(b,out);checks=[]
    for item in g['selection']:
        obs,prompt=load_observation(item['current']);c,_=b.prepare(obs,prompt,());seed=item['current']['seed']
        a,r,_=b.sample(c,seed,1);aa,rr,_=b.warm_sample(c,np.full((50,32),.123,np.float32),seed,1.,1)
        checks.append(dict(episode=item['episode'],passed=bool(np.array_equal(a,aa) and b.torch.equal(r,rr))))
    atomic(out/'one_step_gate.json',dict(passed=all(c['passed'] for c in checks),checks=checks))
    if not all(c['passed'] for c in checks):raise RuntimeError('D1/q1 equality failed')
    protocol=dict(backend=b.protocol,pid=os.getpid(),load_gate_s=time.perf_counter()-tick)
    atomic(out/'protocol.json',protocol)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('127.0.0.1',args.port));server.listen(8)
        print('READY',flush=True)
        while True:
            conn,_=server.accept()
            with conn:
                try:
                    req=pickle.loads(recv_exact(conn,struct.unpack('!I',recv_exact(conn,4))[0]));op=req.pop('op')
                    if op=='health':result=dict(protocol=protocol)
                    elif op=='run':
                        from .worker import run
                        result=run(b,**req)
                    else:raise ValueError(op)
                    result['ok']=True
                except Exception:result=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(result,protocol=5);conn.sendall(struct.pack('!I',len(data))+data)


if __name__=='__main__':main()
