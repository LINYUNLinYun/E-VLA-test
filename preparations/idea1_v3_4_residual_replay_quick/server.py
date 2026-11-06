import argparse,json,os,pickle,socket,struct,time,traceback
from pathlib import Path
import numpy as np
from .common import OUT,OLD,MASKS
from .backend import Backend
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def selection():
    from preparations.idea1_v3_4_v12_compensation.server import selection as select
    return select()

def gate(b,out):
    from preparations.kv_reuse_probe.probe import load_observation
    checks=[];arrays={};torch=b.torch
    def check(name,a,r,aa,rr):
        passed=bool(np.allclose(a,aa,atol=1e-5,rtol=1e-5) and torch.allclose(r.float(),rr.float(),atol=1e-5,rtol=1e-5))
        checks.append(dict(name=name,passed=passed,raw_max_abs=float((r.float()-rr.float()).abs().max().cpu())))
        if not passed:raise RuntimeError('Gate failed: '+name)
    for p in selection()[::10]:
        suite=p['case']['suite'];obs,prompt=load_observation(p['current']);oldobs,oldprompt=load_observation(p['previous']);seed=p['current']['seed']
        old,_=b.prepare_replay(oldobs,oldprompt,capture_layers=range(1,18));history=old['residuals']
        full,_=b.prepare(obs,prompt,());a,r,_=b.sample(full,seed,10)
        captured,_=b.prepare_replay(obs,prompt,capture_layers=range(1,18));aa,rr,_=b.sample(captured,seed,10)
        check(suite+'_full',a,r,aa,rr)
        assert all(torch.equal(full[k],captured[k]) for k in ('K','V'))
        full2,raw2,_=b.sample(full,seed,2)
        for name,mask in MASKS.items():
            native,_=b.prepare(obs,prompt,mask);a,r,_=b.sample(native,seed,2)
            bare,_=b.prepare_replay(obs,prompt,skip=mask);aa,rr,_=b.sample(bare,seed,2)
            check(suite+'_'+name+'_skip',a,r,aa,rr)
            assert all(torch.equal(native[k],bare[k]) for k in ('K','V'))
            identity,_=b.prepare_replay(obs,prompt,skip=mask,history=captured['residuals']);aa,rr,_=b.sample(identity,seed,2)
            check(suite+'_'+name+'_same_input_identity',full2,raw2,aa,rr)
            ref,_=b.prepare_replay(obs,prompt,skip=mask,history=history,reference=True);a,r,_=b.sample(ref,seed,2)
            fast,_=b.prepare_replay(obs,prompt,skip=mask,history=history);aa,rr,_=b.sample(fast,seed,2)
            check(suite+'_'+name+'_past_history',a,r,aa,rr)
            arrays[suite+'_'+name+'_raw']=rr.float().cpu().numpy();arrays[suite+'_'+name]=aa
            if name=='v12_reuse_reference':
                with np.load(OLD/'workers/gpu0/gate_actions.npz') as previous:
                    expected=previous[suite+'_reuse_raw'];expected_a=previous[suite+'_reuse']
                check(suite+'_old_v12_reference',expected_a,torch.as_tensor(expected,device='cuda'),aa,rr)
    np.savez(out/'gate_actions.npz',**arrays)
    atomic(out/'gate.json',dict(passed=True,checks=checks,atol=1e-5,rtol=1e-5))

def benchmark(b,out):
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_v12_compensation.timing import processes
    rows=[];resources=[];start=time.perf_counter();warm_s=0.;gpu=int(out.name[3:])
    names=['full_no_capture']+[prefix+name for name in MASKS for prefix in ('full_','draft_')]
    cache_bytes={}
    for index,p in enumerate(selection()):
        resources.append(dict(time=time.time(),processes=processes(gpu)))
        obs,prompt=load_observation(p['current']);oldobs,oldprompt=load_observation(p['previous']);seed=p['current']['seed']
        old,_=b.prepare_replay(oldobs,oldprompt,capture_layers=range(1,18));history=old['residuals']
        for name,mask in MASKS.items():cache_bytes[name]=max(cache_bytes.get(name,0),sum(history[l].numel()*history[l].element_size() for l in mask))
        def run(name):
            tick=time.perf_counter();isfull=name.startswith('full_')
            variant=None if name=='full_no_capture' else name.split('_',1)[1];mask=() if variant is None else MASKS[variant]
            c,t=b.prepare_replay(obs,prompt,skip=() if isfull else mask,capture_layers=mask if isfull else (),history=None if isfull else history)
            _,_,cost=b.sample(c,seed,10 if isfull else 2);b.torch.cuda.synchronize()
            return dict(policy_wall_ms=(time.perf_counter()-tick)*1000,**t,**cost)
        tick=time.perf_counter()
        for _ in range(3):
            for name in names:run(name)
        warm_s+=time.perf_counter()-tick
        for repeat in range(2):
            offset=index%len(names);order=names[offset:]+names[:offset]
            if repeat:order=list(reversed(order))
            for name in order:rows.append(dict(id=p['id'],suite=p['case']['suite'],task=p['case']['task'],path=name,repeat=repeat,**run(name)))
        resources.append(dict(time=time.time(),processes=processes(gpu)))
        atomic(out/'timing_progress.json',dict(completed=index+1,target=40,time=time.time()))
    exclusive=all(len(r['processes'])==1 for r in resources)
    atomic(out/'benchmark.json',dict(rows=rows,warmup_calls=840,warmup_s=warm_s,wall_s=time.perf_counter()-start,cache_bytes=cache_bytes,
        exclusive=exclusive,resources=resources,history_source='fixed Full bank for timing only'))
    return dict(exclusive=exclusive,measurements=len(rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    b=Backend();pol=Policy(b);gate(b,out)
    protocol=dict(b.protocol,experiment='residual_replay_quick',masks=MASKS,server_pid=os.getpid(),service_load_and_gate_s=time.perf_counter()-start)
    atomic(out/'protocol.json',protocol)
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',a.port));srv.listen(8);print('READY',flush=True)
        while True:
            con,_=srv.accept()
            with con:
                try:
                    req=pickle.loads(recv_exact(con,struct.unpack('!I',recv_exact(con,4))[0]));op=req.pop('op')
                    if op=='health':r=dict(protocol=protocol)
                    elif op=='configure':r=pol.configure(**req)
                    elif op=='predict':r=pol.predict(**req)
                    elif op=='ack':r=pol.acknowledge(**req)
                    elif op=='benchmark':r=benchmark(b,out)
                    else:raise ValueError(op)
                    r['ok']=True
                except Exception:r=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(r,protocol=5);con.sendall(struct.pack('!I',len(data))+data)
if __name__=='__main__':main()
