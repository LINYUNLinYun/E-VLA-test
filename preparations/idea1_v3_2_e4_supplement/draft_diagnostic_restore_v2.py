"""Resume eight restore arms into capture_v2, preserving old valid runs and failures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from .draft_diagnostic_audit import DEFAULT_OUT,ROOT,read
from . import draft_diagnostic_restore as old
from .draft_diagnostic_noise_capture import captured_environment,clean_counterfactual
from .libero_plus_env import PLUS,isolated_child_environment
from preparations.idea1_v3_1.common import atomic_json


def run(a,row):
    base_class,base_observations=old.NativeLiberoPlusEnv,old.observations;instances=[]
    captured=captured_environment(base_class)
    class Tracking(captured):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);instances.append(self)
    old.NativeLiberoPlusEnv=Tracking
    old.observations=clean_counterfactual
    try:old.run(a,row)
    finally:
        old.NativeLiberoPlusEnv=base_class;old.observations=base_observations
        path=a.directory/'restore8'/('case%04d'%row['source_index'])/'result.json'
        if path.exists():
            result=read(path);result['capture_version']='v2_operator_input'
            sources=[Path(__file__),Path(__file__).with_name('draft_diagnostic_noise_capture.py')]
            result['v2_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
            if instances and hasattr(instances[0],'_diagnostic_noise_capture'):
                result['noise_capture']=instances[0]._diagnostic_noise_capture.metadata()
            atomic_json(path,result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root-directory',type=Path,default=DEFAULT_OUT)
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--deadline-unix',type=float,default=0);p.add_argument('--child-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    cases=old.selected(a.root_directory);a.directory=a.root_directory/'capture_v2';a.directory.mkdir(exist_ok=True)
    if not a.launch:print(json.dumps(dict(dry_run=True,cases=len(cases),reuse_old_complete=True,new_root=str(a.directory))));return
    if not a.gpu_uuid or time.time()>=a.deadline_unix:raise ValueError('Explicit UUID/deadline required')
    if a.child_index is not None:run(a,cases[a.child_index]);return
    reused=[]
    for index,row in enumerate(cases):
        if time.time()>=a.deadline_unix:raise TimeoutError('Restore_v2 deadline')
        source=a.root_directory/'restore8'/('case%04d'%row['source_index'])/'result.json'
        if source.exists() and read(source).get('status')=='complete':
            reused.append(dict(source_index=row['source_index'],path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            atomic_json(a.directory/'reused_restore_results.json',reused);continue
        target=a.directory/'restore8'/('case%04d'%row['source_index'])/'result.json'
        if target.exists():
            if read(target).get('status') in ('complete','actual_reproduction_gate_failed'):continue
            raise RuntimeError('New v2 incomplete run requires inspection: '+str(target))
        env=isolated_child_environment(row['case'],a.directory/'restore8/config_plus',gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        command=[str(PLUS/'.venv/bin/python'),'-m',__package__+'.draft_diagnostic_restore_v2','--root-directory',str(a.root_directory),
            '--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--launch']
        with (a.directory/('restore_case%04d.log'%row['source_index'])).open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=process.wait()
            if code:raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':main()
