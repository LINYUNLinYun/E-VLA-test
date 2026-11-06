"""Posthoc four-arm completion for Object tasks0/1 duration x gripper decomposition."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .draft_diagnostic_audit import ROOT,DEFAULT_OUT,read
from . import draft_diagnostic_protocol as frozen
from .libero_plus_env import isolated_child_environment
from preparations.idea1_v3_2.transfer_client import rpc
from preparations.idea1_v3_1.common import atomic_json


def make_plan(reference_directory):
    original=read(Path(reference_directory)/'protocol_plan.json');episodes=[]
    for task in (0,1):
        base=[r for r in original['episodes'] if r['head']=='draft_object' and r['case']['task_index']==task and r['init_index']==0 and r['prompt']=='legacy_filename']
        assert len(base)==2
        for settle in ('five_open','ten_zero'):
            episodes.append(dict(base[0],episode_index=len(episodes),settle=settle,
                reference_episode_indices={r['settle']:r['episode_index'] for r in base}))
    return dict(kind='posthoc_object_settle_duration_by_gripper',posthoc=True,count=4,episodes=episodes,seed=7,cap=600,H=50,R=5,video=False,
        reference_directory=str(reference_directory),selection='Previously fixed Object native task0/1, init0, legacy filename prompt; no reselection.',
        settle=dict(five_open=dict(steps=5,action=[0.]*6+[-1.]),ten_zero=dict(steps=10,action=[0.]*7)),
        reused_cells=dict(five_zero='original40 init0/new/legacy',ten_open='original40 init0/old/legacy'),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(frozen.__file__))},
        interpretation='Posthoc local 2tasks x duration5/10 x gripper0/-1, two cells reused from40, four new episodes. No suite score or preplanned test claim.')


def child(a,plan,row):
    references=[]
    for index in row['reference_episode_indices'].values():
        directory=a.reference_directory/('episode%03d'%index);result=read(directory/'result.json')
        if result.get('status')!='complete':raise RuntimeError('Completed original40 reference required')
        references.append((directory,result))
    health=rpc(a.port,dict(op='health'))['protocol']
    for _,result in references:
        for key in ('gpu_uuid','server_pid','manifest_hash'):
            if result['service_identity'][key]!=health[key]:raise RuntimeError('Original40 same-model/device/server identity gate failed')
    with np.load(references[0][0]/'reset_physical.npz',allow_pickle=False) as archive:
        expected={k:archive[k].copy() for k in archive.files}
    for directory,_ in references[1:]:
        with np.load(directory/'reset_physical.npz',allow_pickle=False) as archive:
            if set(expected)!=set(archive.files) or not all(np.array_equal(value,archive[k]) for k,value in expected.items()):
                raise RuntimeError('Reused duration/gripper references do not share exact reset state')
    original_class=frozen.NativeLiberoPlusEnv;gate=dict(passed=False)
    class GatedEnv(original_class):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            original_set=self.env.set_init_state
            def checked_set(vector):
                obs=original_set(vector);actual=dict(init_vector=np.asarray(vector).copy(),**frozen.state_snapshot(self))
                equal={k:bool(np.array_equal(v,actual[k])) for k,v in expected.items()}
                gate.update(passed=all(equal.values()),reset_physical_equal=equal,
                    reference_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for directory,_ in references for path in (directory/'result.json',directory/'reset_physical.npz')},
                    same_service_identity=dict(gpu_uuid=health['gpu_uuid'],server_pid=health['server_pid'],manifest_hash=health['manifest_hash']))
                if not gate['passed']:raise RuntimeError('New posthoc reset does not match reused original40 reset state')
                return obs
            self.env.set_init_state=checked_set
    frozen.NativeLiberoPlusEnv=GatedEnv
    try:frozen.run_episode(a,plan,row)
    finally:
        frozen.NativeLiberoPlusEnv=original_class
        directory=a.output/('episode%03d'%row['episode_index'])
        if directory.exists():atomic_json(directory/'reference_gate.json',gate)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT/'posthoc_object_settle4')
    p.add_argument('--reference-directory',type=Path,default=DEFAULT_OUT/'original_protocol40')
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--deadline-unix',type=float,default=0);p.add_argument('--child-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    plan=make_plan(a.reference_directory)
    if not a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        if any(a.output.glob('episode*/result.json')):raise RuntimeError('Cannot refresh posthoc plan after episodes')
        atomic_json(a.output/'protocol_plan.json',plan);print(json.dumps(dict(dry_run=True,count=4,posthoc=True)));return
    if not a.gpu_uuid or time.time()>=a.deadline_unix:raise ValueError('Explicit device/deadline required')
    a.output.mkdir(parents=True,exist_ok=True)
    if read(a.output/'protocol_plan.json')!=plan:raise RuntimeError('Posthoc frozen plan mismatch')
    if a.child_index is not None:child(a,plan,plan['episodes'][a.child_index]);return
    for row in plan['episodes']:
        if time.time()>=a.deadline_unix:raise TimeoutError('Posthoc deadline')
        result=a.output/('episode%03d'%row['episode_index'])/'result.json'
        if result.exists():
            if read(result).get('status')=='complete':continue
            raise RuntimeError('Incomplete posthoc episode requires inspection')
        env=isolated_child_environment(row['case'],a.output/'config_original',gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        command=[str(ROOT/'policies/libero-eval-v21/.venv/bin/python'),'-m',__package__+'.draft_diagnostic_settle_split',
            '--output',str(a.output),'--reference-directory',str(a.reference_directory),'--port',str(a.port),'--gpu',str(a.gpu),
            '--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(row['episode_index']),'--launch']
        with (a.output/('episode%03d.log'%row['episode_index'])).open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=process.wait()
            if code:raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':main()
