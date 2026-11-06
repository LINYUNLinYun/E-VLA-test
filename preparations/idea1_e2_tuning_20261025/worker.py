"""Two explicitly assigned serial B1 workers, with local C/G ownership guard."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_3.dispatch import PY
from preparations.idea1_v3_3.egl_mapping import worker_environment
from .design import ROOT,OUT

UUIDS={2:'GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25',3:'GPU-555cde36-5e41-bc44-0c69-ea47358fd703'}


def snapshot(gpu):
    xml=subprocess.check_output(['nvidia-smi','-i',str(gpu),'-q','-x'],text=True);node=ET.fromstring(xml).find('gpu')
    if node.findtext('uuid')!=UUIDS[gpu]:raise ValueError('Physical GPU UUID changed')
    return dict(time=time.time(),uuid=UUIDS[gpu],processes=[dict(pid=int(p.findtext('pid')),type=p.findtext('type')) for p in node.findall('processes/process_info')])


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[2,3],required=True);a=p.parse_args();g=a.gpu;port=9430+g
    os.sched_setaffinity(0,set(range(64,80) if g==2 else range(80,96)))
    assignment=json.loads((OUT/'resource_assignment.json').read_text())
    if assignment['devices'][str(g)]['uuid']!=UUIDS[g] or not assignment['devices'][str(g)]['authorized']:raise ValueError('No explicit current assignment')
    first=snapshot(g)
    if first['processes']:raise RuntimeError('Assigned GPU has C/G owner; do not start')
    with socket.socket() as s:s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
    wd=OUT/'workers'/f'gpu{g}';wd.mkdir(parents=True,exist_ok=True);server=client=None;expected={};last_good=time.time();success=False
    env=environment(g);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
    atomic(wd/'identity.json',dict(pid=os.getpid(),gpu=g,uuid=UUIDS[g],port=port,CPU_affinity=sorted(os.sched_getaffinity(0)),initial_snapshot=first))
    try:
        with (wd/'service.log').open('a') as log:server=subprocess.Popen([str(PY['pi0']),'-m','preparations.idea1_e2_tuning_20261025.server','--model','pi0','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        start=time.time()
        while True:
            if server.poll() is not None:raise RuntimeError('Model service exited')
            ready_snapshot=snapshot(g);ready_C={r['pid'] for r in ready_snapshot['processes'] if 'C' in r['type']}
            if any('G' in r['type'] for r in ready_snapshot['processes']) or len(ready_C)>1:raise RuntimeError('Foreign C/G during service startup')
            if ready_C:
                if 'C' in expected and ready_C!={expected['C']}:raise RuntimeError('Unexpected C/G replacement during startup')
                expected['C']=next(iter(ready_C))
            try:health=rpc(port,dict(op='health'));break
            except (ConnectionError,OSError):
                if time.time()-start>600:raise TimeoutError('Service readiness')
                time.sleep(1)
        atomic(wd/'loaded_protocol.json',dict(health,load_s=time.time()-start))
        with (wd/'rollout.log').open('a') as log:client=subprocess.Popen([str(PY['env']),'-m','preparations.idea1_e2_tuning_20261025.rollout','--gpu',str(g),'--port',str(port),'--shard',str(g-2)],cwd=ROOT,env=worker_environment(env,g),stdout=log,stderr=subprocess.STDOUT)
        while client.poll() is None:
            current=snapshot(g);rows=current['processes']
            for kind in ['C','G']:
                candidates={r['pid'] for r in rows if kind in r['type'] and (kind=='G' or 'G' not in r['type'])}
                if len(candidates)>1:raise RuntimeError('Foreign C/G process entered assigned GPU')
                if candidates and kind not in expected:expected[kind]=next(iter(candidates))
            if any(r['pid'] not in set(expected.values()) for r in rows):raise RuntimeError('Unexpected C/G PID; stop only this experiment')
            last_good=time.time();atomic(wd/'resource_status.json',dict(current,expected_host_context_pids=expected,local_server_pid=server.pid,local_rollout_pid=client.pid,guard_scope='Initially empty assigned GPU, one own model C and own renderer G registered; any additional/replaced PID aborts. Never kills external processes.'))
            time.sleep(5)
        if client.returncode:raise RuntimeError('Rollout exited; inspect isolated log')
        success=True
    except Exception as exc:
        atomic(wd/'alert.json',dict(time=time.time(),error=repr(exc),last_good_resource_time=last_good))
        # Resource-contaminated recently completed attempts cannot qualify cost.
        if 'Foreign' in str(exc) or 'Unexpected C/G' in str(exc):
            for done in OUT.glob('*/episodes/*/task*/init08/complete.json'):
                if done.stat().st_mtime>=last_good and json.loads(done.with_name('result.json').read_text())['engineering_binding']['model_gpu']==g:done.parent.rename(done.parent.with_name('init08.invalid.resource.'+uuid.uuid4().hex))
        raise
    finally:
        for child in [client,server]:
            if child is not None and child.poll() is None:
                child.terminate()
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:child.kill();child.wait()
        if success:atomic(wd/'finished.json',dict(time=time.time(),success=True,final_snapshot=snapshot(g)))


if __name__=='__main__':main()
