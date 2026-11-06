"""One real-input D1 adapter vs released native one-Euler reference."""
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.cli import verify_gpu


def main():
    from .dispatch import OUT
    device=verify_gpu(0);os.environ['CUDA_VISIBLE_DEVICES']='0'
    from preparations.idea1_v3.safe_qk import install
    install(str(OUT/'request_bank/d1_qk_receipt.json'))
    from .backends import PiBackend
    from .policy import V14
    from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
    source=OUT/'pi0_parent/episodes/libero_spatial/task00/init06/input0.npz';saved=np.load(source)
    obs={k:saved[k] for k in saved.files if k not in ('instruction','seed')};instruction=str(saved['instruction']);seed=int(saved['seed'])
    backend=PiBackend();c,_=backend.prepare(obs,instruction,V14);actions,raw,cost=backend.sample(c,seed,1)
    native=backend.adapter.backend;noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
    images,state,initial=prepare_triton_inputs_from_transformed(transformed=native.transformed,device='cuda',action_horizon=50,action_dim=32,noise=noise)
    buffers=native.runtime.buffers;buffers['observation_images_normalized'].copy_(images);buffers['observation_state_normalized'].copy_(state);buffers['diffusion_noise'].copy_(initial)
    native.record(tuple(V14),1);backend.torch.cuda.synchronize()
    expected=buffers['diffusion_noise'].float().cpu().numpy().copy();decoded=backend.decode(c,backend.torch.as_tensor(expected,device='cuda'))
    raw_array=raw.float().cpu().numpy();passed=np.allclose(expected,raw_array,atol=1e-5,rtol=1e-5) and np.allclose(decoded,actions,atol=1e-5,rtol=1e-5)
    arrays=OUT/'request_bank/D1_native_euler_arrays.npz';np.savez(arrays,adapter_raw=raw_array,native_raw=expected,adapter_decoded=actions,native_decoded=decoded)
    atomic(OUT/'D1_native_euler_probe.json',dict(time=time.time(),passed=bool(passed),physical_gpu=0,device=device,input_path=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tolerance=dict(atol=1e-5,rtol=1e-5),arrays_path=str(arrays),arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest(),
        output_sha256={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in dict(adapter_raw=raw_array,native_raw=expected,adapter_decoded=actions,native_decoded=decoded).items()},
        scope='Released native.record(skipV14,num_steps=1) vs adapter.sample D1, same real observation and explicit noise; exactly one Euler step1->0, no environment episode',
        raw_exact=bool(np.array_equal(expected,raw_array)),decoded_exact=bool(np.array_equal(decoded,actions)),raw_max_abs=float(np.max(np.abs(expected-raw_array))),decoded_max_abs=float(np.max(np.abs(decoded-actions))),adapter_nodes=cost['nodes']))
    if not passed:raise ValueError('Native D1 numeric gate failed')


if __name__=='__main__':main()
