"""Serial same-resident-service engineering80 after bounded Xiaomi science.

Called synchronously by stage at a complete condition boundary: unlike an
external SIGSTOP watcher this cannot race the stage's service teardown.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v2_1.runtime_client import rpc
from .dispatch import OUT,ROOT,PY
from .egl_mapping import worker_environment
from .manifest import cases
from .cost_analysis import profile,load_nested
import numpy as np


def run(gpu,port,stage_manifest,stage_owner):
    root=OUT/'engineering_cost_rebaseline';target=root/f'xiaomi_pure_cost_gpu{gpu}';receipt=root/f'xiaomi_gpu{gpu}_comparison.json'
    if receipt.exists():
        if len(load_nested(target))!=80:raise ValueError('Engineering receipt lacks complete80')
        return
    parent=json.loads((OUT/'manifest.json').read_text());cfg=dict(next(c for c in parent['conditions'] if c['id']=='xiaomi_pure_d2'),id=target.name)
    health=rpc(port,dict(op='health'));source=Path(__file__)
    manifest=dict(stage='engineering_Xiaomi_pure_cost',engineering_only=True,conditions=[cfg],cases=parent['cases'],
        scientific_denominator_increment=0,budget_episodes=80,actual_model_gpu=gpu,actual_render_gpu=gpu,
        retained_model_pid=health['protocol']['server_pid'],parent_manifest_hash=parent['manifest_hash'],
        bounded_stage_manifest=str(stage_manifest),bounded_stage_manifest_sha256=hashlib.sha256(Path(stage_manifest).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),service_history='Bounded tau1.6 first actual case used native probe; engineering pure has no probe, same resident model, no concurrent science RPC.')
    manifest['manifest_hash']=digest(manifest);mp=root/f'xiaomi_gpu{gpu}_manifest.json';atomic(mp,manifest)
    boundary=root/f'xiaomi_gpu{gpu}_boundary_receipt.json'
    atomic(boundary,dict(time=time.time(),state='engineering80_running',gpu=gpu,port=port,owner=stage_owner,
        mechanism='Synchronous stage hook after all owned bounded scientific cases complete; stage cannot launch another rollout or tear down service until return.',
        no_concurrent_scientific_RPC=True,resident_model_pid=manifest['retained_model_pid'],model_reload=False,scientific_denominator_increment=0))
    command=[str(PY['env']),'-m','preparations.idea1_v3_3.rollout_bound','--manifest',str(mp),'--condition',cfg['id'],'--output',str(root),'--port',str(port)]
    with (root/f'xiaomi_gpu{gpu}_rollout.log').open('a') as log:
        subprocess.run(command,cwd=ROOT,env=worker_environment(environment(gpu),gpu),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
    rows=load_nested(target)
    if len(rows)!=80:raise ValueError('Xiaomi engineering80 incomplete')
    comparisons=[]
    for case in cases('xiaomi'):
        tail=Path('episodes')/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        old,new=OUT/'xiaomi_pure_d2'/tail,target/tail
        with np.load(old/'arrays.npz') as a,np.load(new/'arrays.npz') as b:
            fields=sorted(k for k in a.files if k.startswith('actions_'));other=sorted(k for k in b.files if k.startswith('actions_'))
            exact=fields==other and all(np.array_equal(a[k],b[k]) for k in fields)
        comparisons.append(dict(case=case,decoded_all_exact=exact,old_result_sha256=hashlib.sha256((old/'result.json').read_bytes()).hexdigest(),
            new_result_sha256=hashlib.sha256((new/'result.json').read_bytes()).hexdigest(),success_equal=json.loads((old/'result.json').read_text())['success']==json.loads((new/'result.json').read_text())['success']))
    atomic(receipt,dict(time=time.time(),gpu=gpu,engineering_only=True,scientific_denominator_increment=0,profile=profile(rows),checks=comparisons,
        decoded_all_exact=all(c['decoded_all_exact'] for c in comparisons),success_all_equal=all(c['success_equal'] for c in comparisons)))
    state=json.loads(boundary.read_text());state.update(state='engineering80_complete',finished=time.time(),valid=80);atomic(boundary,state)

