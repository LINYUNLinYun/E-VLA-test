from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from functools import lru_cache

from preparations.idea1_v3_1.common import ROOT, SUITES, LIMITS, PYTHONS, atomic_json, environment

CHECKPOINT = ROOT / 'shared/checkpoints/openpi/openpi-assets/checkpoints/pi05_libero'
REVISION = 'pi05-representation-v1'
PROTOCOL = dict(revision=REVISION, model='pi05_libero', backend='native OpenPI JAX',
                checkpoint=str(CHECKPOINT), H=10, R=5, D=10, mask=[], env_seed=7,
                settling_steps=10, control_hz=20, limits=LIMITS,
                init_ids=[0, 1, 2], uniform_points=12, event_points=8, target_points=20)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def request_seed(suite, task, init, request):
    key = f'{REVISION}/{suite}/{task}/{init}/{request}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little')


def episode_path(root, suite, task, init):
    return Path(root) / 'episodes' / suite / f'task{task:02d}' / f'init{init:02d}'


def worker_env(gpu, render=False):
    env = environment()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION='.70', TF_NUM_INTRAOP_THREADS='1',
               TF_NUM_INTEROP_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if render:
        # EGL device ordinals are not necessarily physical CUDA ordinals.
        devices = egl_devices()
        selected = next(i for i, physical in devices.items() if physical == gpu)
        env.update(MUJOCO_EGL_DEVICE_ID=str(selected), IDEA1_RENDER_PHYSICAL_GPU=str(gpu))
        # EGL validates its ordinal against the visible-list length in robosuite.
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, dict.fromkeys((gpu, selected))))
    return env


@lru_cache(maxsize=1)
def egl_devices():
    # The controller may itself have CUDA_VISIBLE_DEVICES set. Query physical
    # EGL mapping in a clean child so that this mask cannot remap device IDs.
    env=dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    code='import json; from preparations.idea1_v3_3.egl_mapping import mapping; print(json.dumps(mapping()))'
    result=json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=ROOT,env=env,text=True))
    return {int(k):int(v) for k,v in result.items()}


def process_identity():
    return dict(pid=os.getpid(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
