"""Persistent B1 service with all CUDA Graph construction explicitly forbidden."""
import argparse
import gc
import json
import os
import pickle
import socket
import struct
import time
import traceback
from pathlib import Path
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact
from .common import OUT, verify_frozen


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    for name in ('DRAFT_GRAPH','VERIFY_GRAPH','VERIFY_SEMANTICS_GRAPH'):
        os.environ['SPEC_TRITON_'+name]='0'
    os.environ['SPEC_TRITON_VERIFY_EXACT_INPUT']='1'
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    graph_attempts=[]
    class ForbiddenGraph:
        def __init__(self,*a,**kw):
            graph_attempts.append(traceback.format_stack())
            atomic(out/'forbidden_graph_attempt.json',dict(attempts=graph_attempts))
            raise RuntimeError('This experiment forbids all CUDA Graph capture')
    torch.cuda.CUDAGraph=ForbiddenGraph
    from preparations.idea1_v3.safe_qk import install
    from preparations.idea1_v3.policy import TritonBackend
    from .policy import Policy
    from .benchmark import run
    manifest=json.loads((OUT/'manifest.json').read_text());verify_frozen(manifest)
    begin=time.perf_counter();install(str(out/'qk_receipt.json'));backend=TritonBackend()
    model_load_s=time.perf_counter()-begin
    atomic(out/'startup.json',dict(model_load_s=model_load_s,pid=os.getpid(),time=time.time()))
    benchmark=run(backend,OUT/'benchmark')
    assert benchmark['equivalence']['1'] and not graph_attempts
    protocol=dict(backend.protocol,experiment='selection_cost',model_load_s=model_load_s,
        service_load_and_gate_s=time.perf_counter()-begin,environment_workers=1,cuda_graph=False,
        graph_construction_guard=True,projections=4,projection_seed=340610,
        verify_graph=False,verification_times=[.10,.05],tau=.10)
    atomic(out/'protocol.json',protocol)
    policy=Policy(backend)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        server.bind(('127.0.0.1',args.port));server.listen(8);print('READY',flush=True)
        while True:
            connection,_=server.accept()
            with connection:
                stop=False
                try:
                    request=pickle.loads(recv_exact(connection,struct.unpack('!I',recv_exact(connection,4))[0]))
                    operation=request.pop('op')
                    if operation=='health':result=dict(protocol=protocol)
                    elif operation=='configure':result=policy.configure(**request)
                    elif operation=='predict':result=policy.predict(**request)
                    elif operation=='ack':result=policy.acknowledge(**request)
                    elif operation=='reset_all':
                        assert policy.pending is None
                        policy.selector.clear();policy.reset();gc.collect();torch.cuda.empty_cache();result={}
                    elif operation=='stop':result={};stop=True
                    else:raise ValueError(operation)
                    assert not graph_attempts
                    result['ok']=True
                except Exception:result=dict(ok=False,error=traceback.format_exc())
                payload=pickle.dumps(result,protocol=5)
                connection.sendall(struct.pack('!I',len(payload))+payload)
                if stop:return


if __name__=='__main__':main()
