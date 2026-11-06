"""Sequential localhost-only trusted RPC; an exclusive lease protects each GPU."""
import argparse, pickle, socket, struct, traceback, os
from pathlib import Path
from common import atomic_json, start_ticks,gpu_lease
from runtime_client import legacy
def serve(a):
    from runtime_models import NativePolicy
    policy=NativePolicy(a.model,a.output_dir)
    atomic_json(Path(a.output_dir)/'service_lease.json',dict(pid=os.getpid(),start_ticks=start_ticks(os.getpid()),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),port=a.port,protocol=policy.protocol))
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); server.bind(('127.0.0.1',a.port)); server.listen(8)
        print('READY',flush=True)
        while True:
            conn,_=server.accept()
            with conn:
                try:
                    n=struct.unpack('!I',legacy.recv_exact(conn,4))[0]
                    if n>128*1024*1024: raise ValueError('Oversized request')
                    req=pickle.loads(legacy.recv_exact(conn,n)); op=req.pop('op','predict')
                    if op=='health': result=dict(ok=True,protocol=policy.protocol,calls=policy.calls)
                    elif op=='reset': result=policy.reset() or {'ok':True}
                    elif op in ('predict','parity'): result=getattr(policy,op)(**req)
                    else: raise ValueError('Unsupported operation')
                    result.setdefault('ok',True)
                except Exception: result=dict(ok=False,error=traceback.format_exc())
                payload=pickle.dumps(result,protocol=5); conn.sendall(struct.pack('!I',len(payload))+payload)
def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--port',type=int,required=True); p.add_argument('--output-dir',required=True); p.add_argument('--launch',action='store_true'); a=p.parse_args()
    if not a.launch: print({'dry_run':True,'model':a.model,'port':a.port}); return
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu is None or ',' in gpu: raise ValueError('One GPU plus an exclusive lease is required')
    with gpu_lease(gpu): serve(a)
if __name__=='__main__': main()
