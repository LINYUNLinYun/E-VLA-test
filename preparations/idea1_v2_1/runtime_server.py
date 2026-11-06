"""Trusted localhost RPC: uint32 big-endian frame length followed by pickle dict."""
import argparse
import pickle
import socket
import struct
import traceback
import hashlib
from pathlib import Path
from runtime_models import NativePolicy

SERVER_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def receive(conn, count):
    data=bytearray()
    while len(data)<count:
        part=conn.recv(count-len(data))
        if not part: raise EOFError
        data.extend(part)
    return bytes(data)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['xiaomi-robotic0','db-cogact'])
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    policy=NativePolicy(args.model,args.output_dir)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',args.port)); server.listen(8)
        print(f'READY {args.model} {args.port}',flush=True)
        while True:
            conn,_=server.accept()
            with conn:
                while True:
                    try:
                        length=struct.unpack('>I',receive(conn,4))[0]
                        if length>64*1024*1024: raise ValueError('RPC frame too large')
                        req=pickle.loads(receive(conn,length)); op=req.pop('op','predict')
                        if op=='health': result={'ok':True,'protocol':policy.protocol,'calls':policy.calls}
                        elif op=='reset': result=policy.reset()
                        elif op=='predict': result=policy.predict(**req)
                        elif op=='parity': result=policy.parity(**req)
                        elif op=='diagnostics':
                            result={'ok':True,'layers':[{'layer':i+1,'instance_forward':repr(vars(layer).get('forward')),
                                'attributes':list(vars(layer))} for i,layer in enumerate(policy.layers)]}
                        elif op=='reload_runtime':
                            import importlib
                            import runtime_models
                            importlib.reload(runtime_models)
                            policy.__class__=runtime_models.NativePolicy
                            policy.record_runtime_epoch()
                            result={'ok':True,'loaded_code_hashes':runtime_models.LOADED_CODE_HASHES}
                        else: raise ValueError(op)
                    except EOFError: break
                    except Exception:
                        result={'ok':False,'error':traceback.format_exc()}
                        print(result['error'],flush=True)
                    result['loaded_server_sha256']=SERVER_SOURCE_SHA256
                    payload=pickle.dumps(result,protocol=5)
                    conn.sendall(struct.pack('>I',len(payload))+payload)


if __name__=='__main__': main()
