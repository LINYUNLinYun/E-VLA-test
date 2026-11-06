"""Dedicated E2/E3 server, independent of the running E1/P0 revision."""
import argparse
import json
import pickle
import socket
import struct
import traceback

from preparations.idea1_v2_1.runtime_client import recv_exact


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--config',required=True); parser.add_argument('--launch',action='store_true')
    args=parser.parse_args(); config=json.loads(args.config)
    if not args.launch: print(json.dumps(dict(dry_run=True,config=config))); return
    if config['experiment']=='E2':
        from preparations.idea1_v3.pi05 import Pi05Policy
        policy=Pi05Policy(config)
    elif config['experiment']=='E3':
        from preparations.idea1_v3.history import HistoryPolicy
        policy=HistoryPolicy(config)
    else: raise ValueError('branch server only handles E2/E3')
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',args.port)); server.listen(8)
        print('READY',flush=True)
        while True:
            connection,_=server.accept()
            with connection:
                try:
                    size=struct.unpack('!I',recv_exact(connection,4))[0]
                    if size>128*1024*1024: raise ValueError('oversized request')
                    request=pickle.loads(recv_exact(connection,size)); op=request.pop('op')
                    if op=='health': result=dict(protocol=policy.protocol)
                    elif op=='reset': result=policy.reset()
                    elif op=='ack': result=policy.acknowledge(**request)
                    elif op=='predict': result=policy.predict(**request)
                    else: raise ValueError('unknown operation')
                    result['ok']=True
                except Exception: result=dict(ok=False,error=traceback.format_exc())
                payload=pickle.dumps(result,protocol=5)
                connection.sendall(struct.pack('!I',len(payload))+payload)

if __name__=='__main__': main()
