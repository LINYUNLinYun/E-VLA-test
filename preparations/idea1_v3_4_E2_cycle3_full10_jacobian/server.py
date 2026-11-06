import argparse,json,pickle,socket,struct,traceback
from pathlib import Path
import numpy as np
from .common import ROOT,OUT,SUITES,choose,legal,coefficients
from .policy import Policy
from .model import calculate,current_item
from preparations.idea1_v3.io import atomic
from preparations.idea1_v2_1.runtime_client import recv_exact

def preflight(b,pol,out):
    import torch
    from preparations.idea1_v3_4_E2_E4_single.export import key
    from preparations.kv_reuse_probe.probe import load_observation
    old=ROOT/'results/idea1_v3_4_E2_E4_single';pairs=json.loads((old/'selection.json').read_text());checks=[]
    for suite in SUITES:
        pair=next(p for p in pairs if p['e2'] and p['case']['suite']==suite and p['case']['task']==0);obs,prompt=load_observation(pair['current']);seed=pair['current']['seed']
        c,_=b.prepare(obs,prompt,());native_actions,native_raw,_=b.sample(c,seed,10);native_raw=native_raw.float().cpu().numpy().copy()
        item=current_item(c,seed);arch=torch.load(old/'inputs'/f"{key(pair['current'])}.pt",map_location='cpu',weights_only=False)
        assert all(torch.equal(item[k].cpu(),arch[k].to(item[k].dtype)) for k in ['x','state','encoder_rope','decoder_rope','noise'])
        chosen,grad,score,cost=pol.selector.run(c,seed,coefficients(),4)
        model=pol.selector.model;assert len(model.biases)==len(model.outw)==len(model.outb)==10
        W=torch.as_tensor(coefficients(),device='cuda');g=torch.ones(17,device='cuda',requires_grad=True)
        expected_raw,expected_grad=calculate(model,item,W,g);actual_raw=pol.selector.raw.float().cpu().numpy().copy()
        passed=bool(np.allclose(actual_raw,expected_raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(grad,expected_grad.cpu().numpy(),atol=1e-4,rtol=1e-3) and chosen==choose(abs(expected_grad.cpu().numpy()),4))
        assert passed and np.isfinite(actual_raw).all() and np.isfinite(grad).all()
        for k in [4,6]:assert legal(choose(score,k),k)
        replay_actions=b.decode(c,pol.selector.raw);delta=replay_actions[:5]-native_actions[:5]
        checks.append(dict(suite=suite,passed=passed,NFE=10,layers=18,raw_graph_eager_max=float(abs(actual_raw-expected_raw.float().cpu().numpy()).max()),gradient_graph_eager_max=float(abs(grad-expected_grad.cpu().numpy()).max()),native_replay_raw_max=float(abs(native_raw-actual_raw).max()),native_replay_prefix7_rms=float(np.sqrt(np.mean((native_raw[:5,:7]-actual_raw[:5,:7])**2))),native_replay_translation_max_mm=float(np.linalg.norm(delta[:,:3],axis=1).max()*1000),native_replay_gripper_sign_flips=int(np.sum((native_actions[:5,6]<0)!=(replay_actions[:5,6]<0))),selector_hot_ms=cost['selector_hot_ms'],selector_capture_ms=cost['selector_capture_ms'],max_memory_mb=torch.cuda.max_memory_allocated()/2**20))
        atomic(out/'gate.json',dict(passed=True,checks=checks,native_replay_parity_is_diagnostic=True));print(json.dumps(checks[-1]),flush=True)
        pol.selector.clear()
        if suite==SUITES[0]:
            for budget in [14,12]:
                cfg=dict(id=f'v{budget}_jacobian',budget=budget,skip_count=18-budget,method='jacobian',model='pi0',mode='periodic',period=3,full_steps=10,fast_steps=2,H=50,R=5,random_seed=341101,jacobian_model_steps=10)
                pol.configure(cfg);routes=[]
                for index in range(4):
                    result=pol.predict(obs,prompt,suite,seed+index);routes.append(result['route']);assert result['route']['jacobian_model_steps']==10
                    assert legal(result['route']['selected_skip'],18-budget)
                    if index==0:assert np.allclose(result['raw'],native_raw,atol=1e-5,rtol=1e-5)
                    if index in [1,2]:assert result['timing']['selector_total_ms']==0 and result['route']['skip']==routes[0]['selected_skip']
                    pol.acknowledge(result['actions'][:5],None)
                assert [r['path'] for r in routes]==['full','fast','fast','full']
                pol.selector.clear()
    atomic(out/'gate.json',dict(passed=True,checks=checks,NFE=10,full_path=True,schedule=True,fast_no_probe=True,native_execution_unchanged=True,native_replay_parity_is_diagnostic=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--preflight-only',action='store_true');args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    import torch
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    b=TritonBackend();pol=Policy(b);preflight(b,pol,out);protocol=dict(b.protocol,experiment='E2_cycle3_full10_jacobian',schedule=['Full-D10','Fast-D2','Fast-D2'],period=3,jacobian_model_steps=10,jacobian_full_layers=18,jacobian_output_steps=[1,2,3,4,5],jacobian_projection='one VJP; temporal exponential; seven +1 weights')
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
