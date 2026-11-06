import argparse,gc,json,os,pickle,socket,struct,time,traceback
from pathlib import Path
from .common import CONDITIONS
from .policy import Policy
from .preflight import preflight
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    from preparations.idea1_v3.policy import TritonBackend
    start=time.perf_counter();install(str(out/'qk_receipt.json'));b=TritonBackend();actors=preflight(b,out,(out/'single_actor_required.json').exists())
    protocol=dict(b.protocol,experiment='E2_cycle5_q4_v12',conditions=CONDITIONS,environment_workers=actors,service_mode='serial B1; isolated same-task policy clients',service_load_and_gate_s=time.perf_counter()-start,server_pid=os.getpid(),schedule=['Full-D10']+['Fast-D2']*4,jacobian='original residual gate, four shared-forward Full-D10 VJPs; original residual score; period5, V12, spaced')
    atomic(out/'protocol.json',protocol);policies={};group_prompt=None;resource_failed=False
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',a.port));srv.listen(16);print('READY',flush=True)
        while True:
            con,_=srv.accept()
            with con:
                try:
                    req=pickle.loads(recv_exact(con,struct.unpack('!I',recv_exact(con,4))[0]));op=req.pop('op');client=req.pop('client','default')
                    if op=='health':res=dict(protocol=protocol)
                    elif resource_failed:raise RuntimeError('single_actor_required: restart service with one environment')
                    elif op=='reset_all':
                        assert all(p.pending is None for p in policies.values())
                        for pol in policies.values():pol.selector.clear();pol.reset()
                        policies.clear();gc.collect();torch.cuda.empty_cache();group_prompt=None;res={}
                    else:
                        if client not in policies:policies[client]=Policy(b)
                        pol=policies[client]
                        if op=='configure':res=pol.configure(**req)
                        elif op=='predict':
                            if group_prompt is None:group_prompt=req['instruction']
                            assert req['instruction']==group_prompt,'Two environments must share task/prompt'
                            res=pol.predict(**req)
                        elif op=='ack':res=pol.acknowledge(**req)
                        else:raise ValueError(op)
                    res['ok']=True
                except torch.cuda.OutOfMemoryError:
                    resource_failed=True
                    atomic(out/'single_actor_required.json',dict(reason='Runtime two-client CUDA allocation OOM; bounded restart with one actor'))
                    res=dict(ok=False,error=traceback.format_exc())
                except Exception:res=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(res,protocol=5);con.sendall(struct.pack('!I',len(data))+data)

if __name__=='__main__':main()
