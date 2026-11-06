import argparse
import json
import os
import pickle
import socket
import struct
import time
import traceback
from preparations.idea1_v2_1.runtime_client import recv_exact
from .metadata_r2 import Policy


def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,required=True); p.add_argument('--model',required=True); p.add_argument('--output',required=True); a=p.parse_args()
    start=time.perf_counter()
    if a.model=='pi0':
        from preparations.idea1_v3.safe_qk import install
        install(a.output+'/qk_receipt.json')
        from .backends import PiBackend
        backend=PiBackend()
    else:
        from .backends import XiaomiBackend
        backend=XiaomiBackend()
    policy=Policy(backend)
    policy.protocol.update(server_pid=os.getpid(),service_load_s=time.perf_counter()-start)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1); server.bind(('127.0.0.1',a.port)); server.listen(16)
        print('READY',flush=True)
        while True:
            conn,_=server.accept()
            with conn:
                try:
                    req=pickle.loads(recv_exact(conn,struct.unpack('!I',recv_exact(conn,4))[0])); op=req.pop('op')
                    if op=='health': result=dict(protocol=policy.protocol)
                    elif op=='configure': result=policy.configure(**req)
                    elif op=='reset': result=policy.reset()
                    elif op=='ack': result=policy.acknowledge(**req)
                    elif op=='predict': result=policy.predict(**req)
                    elif op=='probe':
                        obs,instruction,suite,seed=req['obs'],req['instruction'],req['suite'],req['seed']
                        from .metadata_probe import check
                        metadata_check=check(backend,obs,instruction,suite,seed)
                        from .policy import XMASK,V14
                        import numpy as np
                        checks=[]
                        for skip in ((),XMASK if a.model=='xiaomi' else V14):
                            c,t=backend.prepare(obs,instruction,skip); aa,rr,cost=backend.sample(c,seed,2)
                            dd,ss,cost2=backend.sample(c,seed,2)
                            checks.append(dict(skip=list(skip),repeat_max_abs=float(np.abs(aa-dd).max()),passed=bool(np.array_equal(aa,dd)),NFE=2))
                            if a.model=='pi0':
                                # Original released eager sampler, same explicit
                                # noise/input/mask as the adapter path.
                                from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
                                native=backend.adapter.backend
                                noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
                                inputs=prepare_triton_inputs_from_transformed(transformed=native.transformed,device='cuda',action_horizon=50,action_dim=32,noise=noise)
                                images,state,initial=inputs; buf=native.runtime.buffers
                                buf['observation_images_normalized'].copy_(images); buf['observation_state_normalized'].copy_(state); buf['diffusion_noise'].copy_(initial)
                                native.record(tuple(skip),2); backend.torch.cuda.synchronize()
                                native_raw=buf['diffusion_noise'].float().cpu().numpy().copy()
                                native_actions=backend.decode(c,backend.torch.as_tensor(native_raw,device='cuda'))
                                diff=float(np.max(np.abs(native_actions-aa)))
                                raw_diff=float(np.max(np.abs(native_raw-rr.float().cpu().numpy())))
                                checks.append(dict(skip=list(skip),native_decoded_max_abs=diff,native_raw_max_abs=raw_diff,passed=bool(np.allclose(native_actions,aa,atol=1e-5,rtol=1e-5) and np.allclose(native_raw,rr.float().cpu().numpy(),atol=1e-5,rtol=1e-5))))
                        if a.model=='pi0':
                            one,raw,cost=backend.sample(c,seed,1)
                            checks.append(dict(D1_nodes=cost.get('nodes'),NFE=1,finite=bool(np.isfinite(one).all()),passed=bool(np.isfinite(one).all())))
                        else:
                            reference=backend.native.native_reference(obs,instruction,seed=seed,skip=XMASK,num_steps=2)
                            diff=float(np.max(np.abs(reference-aa)))
                            checks.append(dict(native_masked_max_abs=diff,passed=bool(np.allclose(reference,aa,atol=1e-5,rtol=1e-5))))
                        from .cross_restart import check as cross_check
                        cross=cross_check(backend)
                        result=dict(passed=all(v['passed'] for v in checks) and metadata_check['passed'] and cross['passed'],checks=checks,metadata_equivalence=metadata_check,cross_restart=cross)
                    else: raise ValueError(op)
                    result['ok']=True
                except Exception: result=dict(ok=False,error=traceback.format_exc())
                data=pickle.dumps(result,protocol=5); conn.sendall(struct.pack('!I',len(data))+data)


if __name__=='__main__': main()
