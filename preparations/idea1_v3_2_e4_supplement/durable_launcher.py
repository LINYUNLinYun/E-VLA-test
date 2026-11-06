"""Detached module launch with exact process identity and duplicate protection.

This launcher never allocates, scans, or selects a GPU. The target retains all
of its own idle, UUID, parity, and execution-deadline checks.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time


def identity(pid):
    base=Path('/proc')/str(pid)
    try:
        raw=(base/'stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        args=[v.decode(errors='replace') for v in (base/'cmdline').read_bytes().split(b'\0') if v]
        return dict(pid=pid,start_ticks=fields[19],state=fields[0],ppid=int(fields[1]),
                    argv=args,cwd=str((base/'cwd').resolve()),session_id=os.getsid(pid))
    except (FileNotFoundError,ProcessLookupError):return None


def instances(module):
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:current=identity(int(entry.name))
        except PermissionError:continue
        if not current or current['state']=='Z':continue
        args=current['argv']
        if any(args[i]=='-m' and args[i+1]==module for i in range(len(args)-1)):found.append(current)
    return found


def check_empty(module,port):
    found=instances(module)
    if found:raise RuntimeError('Target module already alive; no launch: '+json.dumps(found))
    with socket.socket() as probe:
        try:probe.bind(('127.0.0.1',port))
        except OSError as error:raise RuntimeError('Target localhost port is occupied; no launch: '+str(port)) from error


def write_receipt(path,value):
    temporary=path.with_name(path.name+'.temporary.%d'%os.getpid())
    with temporary.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    temporary.replace(path)


def launch(python,module,arguments,cwd,port,receipt_path,log_path):
    python=Path(python).resolve();cwd=Path(cwd).resolve();receipt_path=Path(receipt_path).resolve();log_path=Path(log_path).resolve()
    if not python.is_file() or not cwd.is_dir():raise ValueError('Existing interpreter and cwd required')
    if not module or any(not (c.isalnum() or c in '._') for c in module):raise ValueError('Explicit dotted Python module required')
    if not 1<=port<=65535:raise ValueError('Valid target port required')
    if receipt_path==log_path:raise ValueError('Receipt and runtime log must differ')
    if receipt_path.exists() or log_path.exists():raise FileExistsError('New launch receipt and log paths required')
    # Serialize all launches of this module, even when they request distinct
    # output directories. The kernel releases this CPU-only lock on exit.
    key=hashlib.sha256(module.encode()).hexdigest()
    lock_path=Path(tempfile.gettempdir())/('codex-durable-module-'+key+'.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_empty(module,port)
        receipt_path.parent.mkdir(parents=True,exist_ok=True);log_path.parent.mkdir(parents=True,exist_ok=True)
        with receipt_path.open('x') as stream:json.dump(dict(status='reserved',launcher_pid=os.getpid()),stream)
        command=[str(python),'-m',module]+list(arguments)
        receipt=dict(status='launching',launcher=identity(os.getpid()),launcher_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            command=command,cwd=str(cwd),module=module,checked_port=port,log=str(log_path),receipt=str(receipt_path),
            started_unix=time.time(),start_new_session=True,stdin='DEVNULL',stdout='independent_file',stderr='STDOUT',
            inherited_device_environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','JAX_PLATFORMS')},
            target_owns_gpu_and_deadline_gates=True)
        write_receipt(receipt_path,receipt)
        try:
            with log_path.open('xb',buffering=0) as log:
                child=subprocess.Popen(command,cwd=cwd,env=dict(os.environ),stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
            current=identity(child.pid)
            if not current or current['session_id']!=child.pid:raise RuntimeError('Detached child session identity unavailable')
            receipt.update(status='launched_not_readiness_verified',child=current,child_pid=child.pid,
                detached_session_id=current['session_id'],child_poll_at_receipt=child.poll(),launch_completed_unix=time.time())
            write_receipt(receipt_path,receipt)
            return receipt
        except Exception as error:
            receipt.update(status='launch_failure',error=repr(error),launch_completed_unix=time.time());write_receipt(receipt_path,receipt);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python',required=True);p.add_argument('--module',required=True);p.add_argument('--cwd',required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--receipt',required=True);p.add_argument('--log',required=True)
    p.add_argument('arguments',nargs=argparse.REMAINDER);a=p.parse_args()
    args=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    result=launch(a.python,a.module,args,a.cwd,a.port,a.receipt,a.log)
    print(json.dumps(dict(status=result['status'],pid=result['child_pid'],start_ticks=result['child']['start_ticks'],receipt=result['receipt'],log=result['log'])))


if __name__=='__main__':main()
