import argparse,json,pickle,socket,struct,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,SUITES,CONDITIONS,choose,legal
from .policy import Policy
from .backend import capture,prepare_patch
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full,cosine_scores
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def preflight(b,pol,out):
    import torch
    from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.server import preflight as gradient_preflight
    from preparations.kv_reuse_probe.probe import load_observation
    gradient_preflight(b,pol,out)
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text());checks=[]
    for suite in SUITES:
        p=next(p for p in pairs if p['e2'] and p['case']['suite']==suite and p['case']['task']==0);obs,prompt=load_observation(p['current']);seed=p['current']['seed']
        c,_=prepare_full(b,obs,prompt,trace=True);full_action,full_raw,_=b.sample(c,seed,2);full_raw=full_raw.float().cpu().numpy().copy()
        residuals,ms=b.adapter.measure(lambda:capture(c['hidden_trace']));assert set(residuals)==set(range(1,18))
        score=cosine_scores(c['hidden_trace']).cpu().numpy();older={k:v*.95 for k,v in residuals.items()}
        masks=[choose(score,4),choose(score,6),[3,14,16,17],[2,3,5,14,16,17]]
        for skip in masks:
            cp,_=prepare_patch(b,obs,prompt,skip,residuals,None,0.);aa,rr,_=b.sample(cp,seed,2)
            passed=bool(np.allclose(full_raw,rr.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(full_action,aa,atol=1e-5,rtol=1e-5));assert passed,(suite,skip,float(abs(full_raw-rr.float().cpu().numpy()).max()))
            for ratio in [1/3,2/3]:
                cp,_=prepare_patch(b,obs,prompt,skip,residuals,older,ratio);aa,rr,_=b.sample(cp,seed,2);actual=rr.float().cpu().numpy().copy()
                cr,_=prepare_patch(b,obs,prompt,skip,residuals,older,ratio,reference=True);bb,ss,_=b.sample(cr,seed,2)
                assert np.allclose(actual,ss.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(aa,bb,atol=1e-5,rtol=1e-5)
        checks.append(dict(suite=suite,passed=True,same_observation_reconstructs_full=True,extrapolation_reference=True,captured_layers=17,capture_ms=ms));print(json.dumps(checks[-1]),flush=True)
        if suite==SUITES[0]:
            for cfg in CONDITIONS:
                pol.configure(cfg);routes=[]
                for index in range(6):
                    result=pol.predict(obs,prompt,suite,seed+index);route=result['route'];routes.append(route)
                    assert legal(route['selected_skip'],cfg['skip_count']) and route['captured_layers']==list(range(1,18))
                    if index in [1,2]:assert not route['has_second_history'] and route['extrapolation_ratio']==0
                    if index in [4,5]:assert route['has_second_history'] and np.isclose(route['extrapolation_ratio'],(index-3)/3)
                    if index%3:assert route['patch_layers']==route['skip'] and result['timing']['selector_total_ms']==0
                    pol.acknowledge(result['actions'][:5],None)
                assert [r['path'] for r in routes]==['full','fast','fast','full','fast','fast']
                assert all(set(h['residuals'])==set(range(1,18)) for h in pol.history)
            pol.configure(CONDITIONS[0])
    atomic(out/'patch_gate.json',dict(passed=True,checks=checks,own_history=True,all17_capture=True,first_reuse_fallback=True,age_ratios=[1/3,2/3],reference_history_is_synthetic_for_numeric_gate=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    b=TritonBackend();pol=Policy(b);preflight(b,pol,out);protocol=dict(b.protocol,experiment='E2_cycle3_extrapolate',schedule=['Full-D10','Fast-D2','Fast-D2'],period=3,conditions=CONDITIONS,patch='last+(age/gap)*(last-previous); initial hold',captured_layers=list(range(1,18)),jacobian_model_steps=10)
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
