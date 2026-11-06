"""Optional bounded queue-wait scan; explicit --launch and an idle assigned GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from preparations.idea1_v3_1.common import ROOT, atomic_json, environment
from preparations.idea1_v3_1.server import rpc


def run(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    deadline=datetime.datetime(2026,10,22,5,7,tzinfo=datetime.timezone.utc).timestamp()
    if deadline-time.time()<240:
        raise TimeoutError('Insufficient original E1 budget for optional scan')
    busy=subprocess.run(['nvidia-smi','-i',str(args.gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],
        capture_output=True,text=True,check=True).stdout.strip()
    if busy:
        raise RuntimeError(f'Assigned GPU is occupied: {busy}')
    command=[sys.executable,'-m','preparations.idea1_v3_2.batch_e1','--service','--bank',args.bank,
        '--manifest',args.manifest,'--output',args.output,'--port',str(args.port),'--launch']
    receipt=dict(controller_pid=os.getpid(),command=command,status='starting',gpu=args.gpu,
        initial_e1_deadline_utc='2026-10-22T05:08:00Z',scan_deadline_utc='2026-10-22T05:07:00Z')
    path=output/'wait_scan_controller.json'
    atomic_json(path,receipt)
    env=environment(args.gpu)
    env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    server=None
    with (output/'service.log').open('x') as stream:
        try:
            server=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            receipt['service_pid']=server.pid
            atomic_json(path,receipt)
            while True:
                if time.time()>deadline-150 or server.poll() is not None:
                    raise RuntimeError('Service load failed or remaining budget too short')
                try:
                    protocol=rpc(args.port,dict(op='health'))['protocol']
                    assert protocol['server_pid']==server.pid
                    break
                except OSError:
                    time.sleep(1)
            from .batch_rescue import run as rescue
            args.arm_spec=['approximate_b2_wait0:2:0','approximate_b4_wait5:4:5','rescue_b1_after:1:0']
            args.original_controller_state='Fresh standalone wait-scan controller; no suspended original controller involved.'
            rescue(args)
            receipt['status']='completed'
        except BaseException as exc:
            receipt.update(status='failed',error=repr(exc))
            raise
        finally:
            if server is not None:
                if server.poll() is None: server.terminate()
                try: server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
                receipt['service_exit_code']=server.returncode
            atomic_json(path,receipt)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',type=int,default=3)
    parser.add_argument('--port',type=int,default=19333)
    parser.add_argument('--bank',default='results/idea1_v3_1/input_bank.pkl')
    parser.add_argument('--manifest',default='results/idea1_v3_1/manifest_v2.json')
    parser.add_argument('--diagnostics',default='results/idea1_v3_2/e1_20261022T0421_batch/batch_diagnostics.json')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if args.launch: run(args)
    else: print(json.dumps(dict(dry_run=True,**vars(args))))
