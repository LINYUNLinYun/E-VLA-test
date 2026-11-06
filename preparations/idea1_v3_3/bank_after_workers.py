"""Use naturally retired screen GPUs for bounded auxiliary B1 bank timing."""
import json
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment
from .dispatch import OUT,ROOT,PY
from .campaign import alive,retired_gpu


def main():
    gatepath=OUT/'cost_selection_gate.json';gate=json.loads(gatepath.read_text());gate.update(pi0_screen_ready=False,auxiliary_bank_pending=True);atomic(gatepath,gate)
    state=json.loads((OUT/'campaign_state.json').read_text());assigned={0:('pi0',state['workers']['0']['pid']),2:('xiaomi',state['workers']['2']['pid'])}
    children={};done={};started=time.time()
    try:
        while len(done)<len(assigned):
            if time.time()-started>3600:raise TimeoutError('Auxiliary bank bounded waiting/running')
            for gpu,(model,owner) in assigned.items():
                if gpu in done:continue
                if gpu not in children:
                    if alive(owner):continue
                    wd=OUT/'workers'/f'gpu{gpu}';identity=wd/'identity.json';finished=wd/'finished.json'
                    if json.loads(identity.read_text())['pid']!=owner or not finished.exists() or finished.stat().st_mtime<identity.stat().st_mtime:raise RuntimeError('Screen worker did not finish normally')
                    retired_gpu(gpu,9340+gpu)
                    env=environment(gpu)
                    if model=='pi0':env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
                    log=(OUT/f'request_bank_{model}_gpu{gpu}.log').open('a')
                    child=subprocess.Popen([str(PY[model]),'-m','preparations.idea1_v3_3.request_bank','--gpu',str(gpu),'--model',model],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                    children[gpu]=(child,time.time());log.close()
                else:
                    child,tick=children[gpu]
                    if time.time()-tick>600:child.terminate();raise TimeoutError('Auxiliary request bank exceeded10minutes')
                    if child.poll() is not None:
                        if child.returncode:raise RuntimeError(f'Bank {model} GPU{gpu} failed; isolated log retained')
                        retired_gpu(gpu,9340+gpu);done[gpu]=dict(model=model,finished=time.time(),wall_s=time.time()-tick)
            atomic(OUT/'request_bank_status.json',dict(time=time.time(),assigned=assigned,processes={str(g):p.pid for g,(p,_) in children.items()},done=done,scope='Auxiliary100 real fixed observations, no science cases or promotion cost changes'))
            time.sleep(5)
    except Exception as exc:
        atomic(OUT/'request_bank_error.json',dict(time=time.time(),error=repr(exc)))
        for child,_ in children.values():
            if child.poll() is None:child.terminate();child.wait(timeout=30)
        raise
    finally:
        gate=json.loads(gatepath.read_text());gate.update(pi0_screen_ready=True,auxiliary_bank_pending=False);atomic(gatepath,gate)


if __name__=='__main__':main()
