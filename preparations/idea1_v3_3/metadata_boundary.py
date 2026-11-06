"""Finish current baseline80 before switching only own pi0 service processes."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from preparations.idea1_v3.io import atomic,append
from .dispatch import ROOT,OUT,PY
from .campaign import launch,alive


def children_active(pid):
    rows=subprocess.check_output(['ps','-eo','ppid,pid,stat,args'],text=True).splitlines()
    return any(parts[0]==str(pid) and parts[2][0]!='Z' and 'preparations.idea1_v3_3.rollout' in line for line in rows if len(parts:=line.split(None,3))==4)


def main():
    assigned={0:(32615,'pi0_parent'),1:(32616,'pi0_full_d10')}
    start=time.monotonic()
    while True:
        counts={gpu:len(list((OUT/cid/'episodes').glob('*/task*/init*/complete.json'))) for gpu,(pid,cid) in assigned.items()}
        atomic(OUT/'metadata_boundary_state.json',dict(status='draining_baseline80',time=time.time(),counts=counts,pids={str(g):p for g,(p,c) in assigned.items()}))
        if all(counts[g]==80 and not children_active(pid) for g,(pid,cid) in assigned.items()): break
        if time.monotonic()-start>3600: raise TimeoutError('Metadata boundary exceeds bounded 1h drain')
        time.sleep(5)
    parent=json.loads((OUT/'manifest.json').read_text())
    for path,expected in parent['source_files'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected: raise ValueError('Original frozen source changed')
    files=['metadata_r2.py','metadata_probe.py','cross_restart.py','server_r2.py','dispatch_r2.py','test_metadata_r2.py','metadata_boundary.py']
    receipt=dict(execution_revision='v33-noise-r2',parent_manifest_hash=parent['manifest_hash'],scientific_behavior='Original frozen Policy invoked unchanged; metadata fields only',
        original_source_files=parent['source_files'],successor_source_files={str(Path(__file__).with_name(name).relative_to(ROOT)):hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files},
        kalman_asset=dict(path='preparations/idea1_v3_3/kalman_frozen.json',sha256=hashlib.sha256(Path(__file__).with_name('kalman_frozen.json').read_bytes()).hexdigest()),
        prior_completed_conditions=['pi0_parent','pi0_full_d10'],CPU_validation='10 targeted tests passed; includes old/new actions/raw/routes/ack state and Kalman accepted/rejected noise identity',
        real_input_validation='Mandatory first R2 case runs five-request old/new policy replay including Kalman and native raw/decoded gates before scientific actions',time=time.time())
    atomic(OUT/'metadata_revision_r2.json',receipt)
    for gpu,(pid,cid) in assigned.items():
        wd=OUT/'workers'/f'gpu{gpu}'; history=wd/'history'/'pre_noise_r2'; history.mkdir(parents=True,exist_ok=True)
        for name in ('identity.json','loaded_protocol.json'):
            if (wd/name).exists(): (history/name).write_bytes((wd/name).read_bytes())
        protocol=json.loads((wd/'loaded_protocol.json').read_text())['protocol']; serverpid=protocol['server_pid']
        for ownpid in (serverpid,pid):
            command=Path(f'/proc/{ownpid}/cmdline').read_bytes()
            if b'preparations.idea1_v3_3.' not in command: raise ValueError('PID identity changed')
        os.kill(serverpid,signal.SIGTERM); os.kill(pid,signal.SIGKILL)
        running=json.loads((OUT/'conditions'/cid/'running.json').read_text())
        atomic(OUT/'conditions'/cid/'done.json',dict(config=running['config'],gpu=gpu,wall_s=time.time()-running['started'],finished=time.time(),completion='80 valid cases; parent dispatch deliberately drained for metadata-only revision'))
        append(OUT/'campaign_events.jsonl',dict(event='metadata_boundary',gpu=gpu,old_dispatch_pid=pid,old_server_pid=serverpid,condition=cid,valid=80))
    time.sleep(3)
    state=json.loads((OUT/'campaign_state.json').read_text())
    for gpu in assigned: state['workers'][str(gpu)]=dict(pid=launch(gpu,'pi0'),model='pi0')
    atomic(OUT/'campaign_state.json',state)
    with (OUT/'campaign.log').open('a') as log:
        supervisor=subprocess.Popen([str(PY['pi0']),'-m','preparations.idea1_v3_3.campaign'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    atomic(OUT/'metadata_boundary_state.json',dict(status='r2_launched',time=time.time(),supervisor_pid=supervisor.pid,workers=state['workers']))


if __name__=='__main__': main()
