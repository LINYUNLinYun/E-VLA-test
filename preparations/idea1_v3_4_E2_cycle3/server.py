import argparse,json,pickle,socket,struct,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,SUITES,CONDITIONS,choose,legal
from .policy import Policy
from .selection import prepare_full,cosine_scores
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def preflight(b,pol,out):
    import torch
    from preparations.idea1_v3_4_E2_online_time_weights.server import preflight as gradient_preflight
    from preparations.kv_reuse_probe.probe import load_observation
    gradient_preflight(b,pol,out)
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text());checks=[]
    for suite in SUITES:
        pair=next(p for p in pairs if p['e2'] and p['case']['suite']==suite and p['case']['task']==0);obs,prompt=load_observation(pair['current']);seed=pair['current']['seed']
        c,_=b.prepare(obs,prompt,());aa,rr,_=b.sample(c,seed,10);reference=rr.float().cpu().numpy().copy()
        c,_=prepare_full(b,obs,prompt,True);bb,ss,_=b.sample(c,seed,10);assert np.allclose(aa,bb,atol=1e-5,rtol=1e-5) and np.allclose(reference,ss.float().cpu().numpy(),atol=1e-5,rtol=1e-5)
        scores=cosine_scores(c['hidden_trace']).cpu().numpy();assert np.isfinite(scores).all() and all(legal(choose(scores,k),k) for k in [4,6])
        checks.append(dict(suite=suite,passed=True,raw_max_abs=float(abs(reference-ss.float().cpu().numpy()).max()),skip4=choose(scores,4),skip6=choose(scores,6)))
        if suite==SUITES[0]:
            first={}
            for cfg in CONDITIONS:
                pol.configure(cfg);routes=[]
                for index in range(4):
                    result=pol.predict(obs,prompt,suite,seed+index);routes.append(result['route']);pol.acknowledge(result['actions'][:5],None)
                    if index==0:
                        assert np.allclose(result['raw'],reference,atol=1e-5,rtol=1e-5)
                        first[cfg['id']]=result['actions'].copy()
                    assert legal(result['route']['selected_skip'],cfg['skip_count'])
                    if index in [1,2] and cfg['method'] in ['cosine','jacobian']:
                        assert result['timing']['selector_total_ms']==0 and result['route']['mask_age_env_steps']==index*5
                assert [r['path'] for r in routes]==['full','fast','fast','full'] and [r['D'] for r in routes]==[10,2,2,10]
                if cfg['method'] in ['cosine','jacobian','random_fixed']:assert routes[0]['selected_skip']==routes[1]['skip']==routes[2]['skip']
            assert all(np.allclose(x,next(iter(first.values())),atol=1e-5,rtol=1e-5) for x in first.values())
            pol.selector.clear()
    atomic(out/'cycle_gate.json',dict(passed=True,checks=checks,full10_trace_parity=True,schedule=True,fast_no_probe=True,budgets=[14,12]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    b=TritonBackend();pol=Policy(b);preflight(b,pol,out);protocol=dict(b.protocol,experiment='E2_cycle3',schedule=['Full-D10','Fast-D2','Fast-D2'],period=3,conditions=CONDITIONS,verifier=False)
    atomic(out/'protocol.json',protocol)
    if args.preflight_only:return
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind(('127.0.0.1',args.port));srv.listen(8);print('READY',flush=True)
        while True:
            con,_=srv.accept()
            with con:
                try:
                    req=pickle.loads(recv_exact(con,struct.unpack('!I',recv_exact(con,4))[0]));op=req.pop('op')
                    if op=='health':res=dict(protocol=protocol)
                    elif op=='configure':res=pol.configure(**req)
                    elif op=='predict':res=pol.predict(**req)
                    elif op=='ack':res=pol.acknowledge(**req)
                    else:raise ValueError(op)
                    res['ok']=True
                except Exception:res=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(res,protocol=5);con.sendall(struct.pack('!I',len(data))+data)
if __name__=='__main__':main()
