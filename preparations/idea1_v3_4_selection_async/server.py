"""Persistent B1 service; only selector work uses a CUDA Graph."""
import argparse,json,pickle,socket,struct,time,traceback
from pathlib import Path
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact
from .runtime import setup
from .policy import Policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True)
    p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    backend=setup(out);policy=Policy(backend)
    protocol=dict(backend.protocol,experiment='selection_async',cuda_graph=True,
        native_inference_graph=False,verify_graph=False,verification_times=[.10,.05],
        projection_seed=340610,tau=.10,environment_workers=1)
    atomic(out/'protocol.json',protocol)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',a.port));server.listen(8);print('READY',flush=True)
        while True:
            connection,_=server.accept();stop=False
            with connection:
                try:
                    request=pickle.loads(recv_exact(connection,struct.unpack('!I',recv_exact(connection,4))[0]))
                    operation=request.pop('op')
                    if operation=='health':result=dict(protocol=protocol)
                    elif operation=='configure':result=policy.configure(**request)
                    elif operation=='predict':result=policy.predict(**request)
                    elif operation=='ack':result=policy.acknowledge(**request)
                    elif operation=='finish':result=policy.finish()
                    elif operation=='reset_all':policy.clear();result={}
                    elif operation=='stop':policy.clear();result={};stop=True
                    else:raise ValueError(operation)
                    result['ok']=True
                except Exception:result=dict(ok=False,error=traceback.format_exc())
                payload=pickle.dumps(result,protocol=5)
                connection.sendall(struct.pack('!I',len(payload))+payload)
            if stop:return


if __name__=='__main__':main()
