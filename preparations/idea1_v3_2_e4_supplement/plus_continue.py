"""Execution-only continuation of frozen Plus matrix; no environment changes.

Runs all350 perturbed cases, then resumes original40's identical plan. A child
technical failure stops with evidence, never silently becomes a policy failure.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .native_envs import ROOT,sha,write_json
from .evaluation import protocol_identity


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--extension-receipt',required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--port',type=int,default=19143)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    root=Path(a.root);manifest=root/'manifest.json';protocol=protocol_identity(manifest)
    extension=Path(a.extension_receipt)
    if not extension.exists(): raise ValueError('Explicit extension receipt is required')
    # Retain the historical manifest deadline unchanged; this is execution-only.
    receipt=dict(status='starting',manifest_hash=protocol['manifest_hash'],historical_manifest_deadline=protocol['deadline_unix'],
        deadline_unix=a.deadline_unix,extension_receipt=str(extension),extension_receipt_sha256=sha(extension),
        pid=os.getpid(),created_unix=time.time(),source_sha256=sha(__file__),stages=[])
    if not a.launch: print(json.dumps(receipt));return
    directory=root/('continuation_%d'%int(time.time()));directory.mkdir(parents=True,exist_ok=False)
    for relative,expected in protocol['source_files'].items():
        path=Path(relative);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=expected: raise ValueError('Frozen runtime source mismatch: '+str(path))
    base=['/policies/LIBERO-plus/.venv/bin/python','-m','preparations.idea1_v3_2_e4_supplement.run_plus',
        '--protocol',str(manifest),'--port',str(a.port),'--gpu','3','--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--video','--launch']
    order=[str(suite*10+index) for index in range(10) for suite in range(4)]
    commands=[('perturbed350',base+['--output',str(root/'perturbed350'),'--max-new-cases','350']),
        ('original40',base+['--output',str(root/'original40'),'--baseline','--indices']+order+['--max-new-cases','36'])]
    write_json(directory/'receipt.json',receipt)
    for name,command in commands:
        if time.time()>=a.deadline_unix:
            receipt['status']='deadline_reached';break
        env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1')
        with (directory/(name+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
            row=dict(stage=name,pid=process.pid,command=command,started_unix=time.time(),log=str(directory/(name+'.log')))
            receipt['stages'].append(row);receipt['status']='running';write_json(directory/'receipt.json',receipt)
            # Child controller owns/cleans its episode children at the same
            # deadline. Allow its documented15s cleanup grace, no broad kill.
            try: code=process.wait(timeout=max(1,a.deadline_unix-time.time())+30)
            except subprocess.TimeoutExpired:
                process.terminate()
                try: process.wait(timeout=15)
                except subprocess.TimeoutExpired: process.kill();process.wait()
                code=124
        row.update(exit_code=code,completed_unix=time.time(),reaped=True);write_json(directory/'receipt.json',receipt)
        if code:
            receipt['status']='technical_failure_or_deadline';break
    else: receipt['status']='complete'
    receipt['completed_unix']=time.time();write_json(directory/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__': main()
