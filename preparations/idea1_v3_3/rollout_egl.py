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
    if devices.get(index)!=expected or expected==1: raise ValueError('EGL device is not the assigned allowed physical GPU')
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
    from .rollout import main
    main()
