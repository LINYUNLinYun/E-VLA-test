"""Drain pure-D2 parent condition then start strict-c1 before any formal E2."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from preparations.idea1_v3.io import atomic,append
from .metadata_boundary import children_active
from .dispatch import ROOT,OUT,PY
from .campaign import launch


def main():
    owner,server,gpu,cid=38508,38717,0,'pi0_pure_d2';start=time.monotonic()
    while True:
        n=len(list((OUT/cid/'episodes').glob('*/task*/init*/complete.json')))
        atomic(OUT/'c1_boundary_state.json',dict(status='draining_pure80',valid=n,time=time.time()))
        if n==80 and not children_active(owner): break
        if time.monotonic()-start>3600: raise TimeoutError('C1 boundary drain')
        time.sleep(5)
    wd=OUT/'workers/gpu0'; history=wd/'history/pre_strict_c1';history.mkdir(parents=True,exist_ok=True)
    for name in ('identity.json','loaded_protocol.json'):
        (history/name).write_bytes((wd/name).read_bytes())
    for pid in (owner,server):
        if b'preparations.idea1_v3_3.' not in Path(f'/proc/{pid}/cmdline').read_bytes(): raise ValueError('PID identity changed')
    handoff=OUT/'worker_handoffs/gpu0_strict_c1.json'
    atomic(handoff,dict(gpu=0,old_pid=owner,new_worker=None,expires=time.time()+120))
    os.kill(server,signal.SIGTERM);os.kill(owner,signal.SIGKILL)
    running=json.loads((OUT/'conditions'/cid/'running.json').read_text());atomic(OUT/'conditions'/cid/'done.json',dict(config=running['config'],gpu=0,wall_s=time.time()-running['started'],finished=time.time(),completion='80 valid cases; deliberate c1 boundary drain'))
    append(OUT/'campaign_events.jsonl',dict(event='strict_c1_boundary',gpu=0,completed=cid,valid=80,time=time.time()))
    time.sleep(3)
    state=json.loads((OUT/'campaign_state.json').read_text());state['workers']['0']=dict(pid=launch(0,'pi0'),model='pi0');atomic(OUT/'campaign_state.json',state)
    atomic(handoff,dict(gpu=0,old_pid=owner,new_worker=state['workers']['0'],expires=time.time()+120))
    atomic(OUT/'c1_boundary_state.json',dict(status='strict_c1_launched',worker=state['workers']['0'],time=time.time()))


if __name__=='__main__':main()
