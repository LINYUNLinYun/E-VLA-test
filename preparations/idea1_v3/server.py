"""One model resident per worker; execution acknowledgement precedes replanning."""
import argparse
import json
import pickle
import socket
import struct
import traceback

from preparations.idea1_v2_1.runtime_client import recv_exact
from preparations.idea1_v3.policy import Policy


def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,required=True)
    p.add_argument('--config',required=True); p.add_argument('--launch',action='store_true'); a=p.parse_args()
    if not a.launch: print(json.dumps(dict(dry_run=True,config=json.loads(a.config)))); return
    policy=Policy(json.loads(a.config))
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',a.port)); server.listen(8)
        print('READY',flush=True)
        while True:
            conn,_=server.accept()
            with conn:
                try:
                    n=struct.unpack('!I',recv_exact(conn,4))[0]
                    if n>128*1024*1024: raise ValueError('oversized RPC')
                    req=pickle.loads(recv_exact(conn,n)); op=req.pop('op')
                    if op=='health': result=dict(protocol=policy.protocol)
                    elif op=='reset': result=policy.reset()
                    elif op=='ack': result=policy.acknowledge(**req)
                    elif op=='predict': result=policy.predict(**req)
                    else: raise ValueError('unknown operation')
                    result['ok']=True
                except Exception: result=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(result,protocol=5)
                conn.sendall(struct.pack('!I',len(data))+data)

if __name__=='__main__': main()
