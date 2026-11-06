"""Original scientific rollout plus environment-only render binding/parity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v2_1.runtime_client import official_init_states
from .egl_mapping import mapping


def gate():
    root=Path(__file__).resolve().parents[2]; out=root/'results/idea1_v3_3'
    expected=int(os.environ['IDEA1_RENDER_PHYSICAL_GPU']); index=int(os.environ['MUJOCO_EGL_DEVICE_ID']); devices=mapping()
    leases=json.loads((out/'external_gpu_leases.json').read_text()) if (out/'external_gpu_leases.json').exists() else {}
    if devices.get(index)!=expected or leases.get(str(expected),{}).get('active'): raise ValueError('EGL device is not the assigned currently authorized physical GPU')
    receipt=out/f'egl_parity_gpu{expected}.json'
    if receipt.exists() and json.loads(receipt.read_text()).get('passed'): return
    source=out/'xiaomi_full_d2/episodes/libero_spatial/task00/init00/input0.npz'
    saved=np.load(source,allow_pickle=False); pool=Environments(False)
    try:
        env,definition,bddl,selected,created=pool.acquire('libero_spatial',0)
        vector=np.asarray(official_init_states(selected,0)[0]); random.seed(7);np.random.seed(7);env.seed(7);env.reset();obs=env.set_init_state(vector.copy())
        for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
        checks={k:dict(equal=bool(np.array_equal(saved[k],obs[k])),max_abs=float(np.max(np.abs(saved[k].astype(float)-np.asarray(obs[k]).astype(float))))) for k in saved.files if k not in ('instruction','seed')}
        passed=all(r['equal'] for r in checks.values())
        atomic(receipt,dict(passed=passed,time=time.time(),physical_gpu=expected,egl_index=index,mapping=devices,env={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID')},
            reference=str(source),reference_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checks=checks,scope='Same official init/seed/10 settling; raw RGB and state parity after environment-only render-device correction; no policy call.'))
        if not passed: raise ValueError('Initial observation differs across EGL bindings')
    finally: pool.close()


if __name__=='__main__':
    gate()
    from . import rollout
    physical=int(os.environ['IDEA1_RENDER_PHYSICAL_GPU'])
    context=dict(engineering_revision='egl-physical-binding-v3',model_gpu=physical,render_gpu=physical,egl_index=int(os.environ['MUJOCO_EGL_DEVICE_ID']),
        worker_pid=os.getpid(),CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],EGL_to_physical=mapping(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    original_atomic=rollout.atomic
    current_server=None
    def enriched_atomic(path,value):
        global current_server
        path=Path(path)
        if path.name=='identity.json':
            current_server=value['protocol']['server_pid'];value=dict(value,engineering_binding=context)
        elif path.name=='result.json':value=dict(value,engineering_binding=context)
        elif path.name=='complete.json':
            side=dict(context,model_server_pid=current_server,identity_sha256=hashlib.sha256((path.parent/'identity.json').read_bytes()).hexdigest(),result_sha256=hashlib.sha256((path.parent/'result.json').read_bytes()).hexdigest(),scientific_manifest_hash=value['manifest_hash'])
            original_atomic(path.parent/'engineering_revision.json',side)
        original_atomic(path,value)
        if path.name=='native_probe.json':
            copy=path.with_name(f'native_probe.gpu{physical}.worker{os.getpid()}.json')
            original_atomic(copy,dict(value,worker_binding=context,model_server_pid=current_server))
    rollout.atomic=enriched_atomic
    rollout.main()
