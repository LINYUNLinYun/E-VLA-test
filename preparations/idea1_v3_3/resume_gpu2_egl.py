"""Retain own live model service; resume interrupted condition with EGL fix."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v2_1.runtime_client import rpc
from .dispatch import ROOT,OUT,PY
from .egl_mapping import worker_environment


def main():
    gpu,port,serverpid=2,9342,32889
    wd=OUT/'workers/gpu2'
    env=worker_environment(environment(gpu),gpu)
    with (wd/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        health=rpc(port,dict(op='health'))
        if health['protocol']['server_pid']!=serverpid: raise ValueError('Retained model service identity differs')
        identity=json.loads((wd/'identity.json').read_text()); identity.update(pid=os.getpid(),engineering_revision='egl-physical-binding-v1',retained_model_pid=serverpid,environment={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_EGL_DEVICE_ID')})
        atomic(wd/'identity.json',identity)
        state=json.loads((OUT/'campaign_state.json').read_text()); state['workers']['2']=dict(pid=os.getpid(),model='xiaomi'); atomic(OUT/'campaign_state.json',state)
        while True:
            m=json.loads((OUT/'manifest.json').read_text()); claimed=None; handle=None
            for cfg in m['conditions']:
                if cfg['model']!='xiaomi': continue
                directory=OUT/'conditions'/cfg['id']; directory.mkdir(parents=True,exist_ok=True)
                if (directory/'done.json').exists() or (directory/'failed.json').exists(): continue
                candidate=(directory/'owner.lock').open('a')
                try: fcntl.flock(candidate,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: candidate.close();continue
                claimed=cfg;handle=candidate;break
            if claimed is None: break
            directory=OUT/'conditions'/claimed['id']; atomic(directory/'running.json',dict(gpu=gpu,pid=os.getpid(),config=claimed,started=time.time(),engineering_revision='egl-physical-binding-v1'))
            try:
                command=[str(PY['env']),'-m','preparations.idea1_v3_3.rollout_egl','--manifest',str(OUT/'manifest.json'),'--condition',claimed['id'],'--output',str(OUT),'--port',str(port)]
                tick=time.perf_counter()
                with (directory/'rollout.log').open('a') as log: subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                atomic(directory/'done.json',dict(config=claimed,gpu=gpu,wall_s=time.perf_counter()-tick,finished=time.time(),engineering_revision='egl-physical-binding-v1'))
            except Exception as exc:
                atomic(directory/'failed.json',dict(error=repr(exc),gpu=gpu,engineering_revision='egl-physical-binding-v1'));raise
            finally:handle.close()
        atomic(wd/'finished.json',dict(time=time.time(),engineering_revision='egl-physical-binding-v1'))
        os.kill(serverpid,15)


if __name__=='__main__': main()
