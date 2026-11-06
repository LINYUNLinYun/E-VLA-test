"""Additional authorized GPU2 pure80 to repair the physical-device baseline."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v2_1.runtime_client import rpc
from .dispatch import OUT,ROOT,PY
from .campaign import retired_gpu
from .egl_mapping import worker_environment
from .cost_analysis import profile,load_nested
from .manifest import cases


def main():
    gpu=2;port=9362;root=OUT/'engineering_cost_rebaseline';wd=root/'xiaomi_fresh_gpu2_service';wd.mkdir(exist_ok=True)
    leases=json.loads((OUT/'external_gpu_leases.json').read_text())
    if leases.get('2',{}).get('active'):raise ValueError('External GPU2 lease')
    before=retired_gpu(gpu,port);retired_gpu(gpu,9352)
    env=environment(gpu);server=None;success=False
    try:
        with (wd/'service.log').open('a') as log:
            server=subprocess.Popen([str(PY['xiaomi']),'-m','preparations.idea1_v3_3.server','--model','xiaomi','--port',str(port),'--output',str(wd)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        start=time.time()
        while True:
            if server.poll() is not None:raise RuntimeError('Fresh Xiaomi service exited')
            try:health=rpc(port,dict(op='health'));break
            except (ConnectionError,OSError):
                if time.time()-start>600:raise TimeoutError('Fresh Xiaomi readiness')
                time.sleep(1)
        atomic(wd/'loaded_protocol.json',dict(health,load_s=time.time()-start))
        input0=OUT/'xiaomi_pure_d2/episodes/libero_spatial/task00/init00/input0.npz'
        with np.load(input0) as saved:
            obs={k:saved[k] for k in saved.files if k not in ('instruction','seed')};instruction=str(saved['instruction']);seed=int(saved['seed'])
        native=rpc(port,dict(op='probe',obs=obs,instruction=instruction,suite='libero_spatial',seed=seed));atomic(wd/'native_probe.json',native)
        if not native['passed']:raise ValueError('Fresh Xiaomi native gate failed')
        parent=json.loads((OUT/'manifest.json').read_text());cfg=dict(next(c for c in parent['conditions'] if c['id']=='xiaomi_pure_d2'),id='xiaomi_pure_cost_gpu2')
        source=Path(__file__);sources=[source,source.with_name('server.py'),source.with_name('backends.py'),source.with_name('rollout_bound.py'),source.with_name('cost_analysis.py')]
        m=dict(stage='engineering_Xiaomi_pure_cost_gpu2',engineering_only=True,budget_episodes=80,total_engineering_valid_budget=400,scientific_denominator_increment=0,
            conditions=[cfg],cases=parent['cases'],actual_model_gpu=2,actual_render_gpu=2,correct_EGL_index=1,device_uuid='GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25',
            server_pid=server.pid,loaded_protocol=health['protocol'],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            parent_manifest_hash=parent['manifest_hash'],input0_sha256=hashlib.sha256(input0.read_bytes()).hexdigest(),
            service_history='Fresh isolated Xiaomi service; native Full/skip12 D2 repeated same real spatial task0 init0 probe before engineering, outside policy wall. No scientific RPC shares this service.',
            authorization='Root authorized additional80 after correcting tau.8 physicalGPU2 vs periodicGPU3. First320 engineering valid unchanged; new total400, science denominator0.')
        m['manifest_hash']=digest(m);manifest=root/'xiaomi_gpu2_manifest.json';atomic(manifest,m)
        atomic(root/'xiaomi_gpu2_boundary_receipt.json',dict(time=time.time(),state='engineering80_running',gpu=2,port=port,driver_pid=os.getpid(),server_pid=server.pid,old_owner98822_exited=True,old_C_G_absent=True,old_RPC9352_free=True,nvidia_smi_before=before,
            no_concurrent_scientific_RPC=True,model_reload=True,manifest_hash=m['manifest_hash']))
        command=[str(PY['env']),'-m','preparations.idea1_v3_3.rollout_bound','--manifest',str(manifest),'--condition',cfg['id'],'--output',str(root),'--port',str(port)]
        with (root/'xiaomi_gpu2_rollout.log').open('a') as log:subprocess.run(command,cwd=ROOT,env=worker_environment(env,gpu),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        rows=load_nested(root/cfg['id'])
        if len(rows)!=80:raise ValueError('Fresh pure80 incomplete')
        checks=[]
        for case in cases('xiaomi'):
            tail=Path('episodes')/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}";old,new=OUT/'xiaomi_pure_d2'/tail,root/cfg['id']/tail
            with np.load(old/'arrays.npz') as a,np.load(new/'arrays.npz') as b:
                fields=sorted(k for k in a.files if k.startswith('actions_'));other=sorted(k for k in b.files if k.startswith('actions_'));exact=fields==other and all(np.array_equal(a[k],b[k]) for k in fields)
            checks.append(dict(case=case,decoded_all_exact=exact,old_result_sha256=hashlib.sha256((old/'result.json').read_bytes()).hexdigest(),new_result_sha256=hashlib.sha256((new/'result.json').read_bytes()).hexdigest(),success_equal=json.loads((old/'result.json').read_text())['success']==json.loads((new/'result.json').read_text())['success']))
        atomic(root/'xiaomi_gpu2_comparison.json',dict(time=time.time(),gpu=2,engineering_only=True,scientific_denominator_increment=0,profile=profile(rows),checks=checks,decoded_all_exact=all(c['decoded_all_exact'] for c in checks),success_all_equal=all(c['success_equal'] for c in checks)))
        success=True
    except Exception as exc:
        atomic(root/'xiaomi_gpu2_fresh_error.json',dict(time=time.time(),error=repr(exc)));raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:server.wait(timeout=30)
            except subprocess.TimeoutExpired:server.kill();server.wait()
        if success:
            after=retired_gpu(gpu,port)
            atomic(root/'xiaomi_gpu2_fresh_finished.json',dict(time=time.time(),valid=80,server_exited=True,all_own_C_G_exited=True,RPC9362_free=True,nvidia_smi_after=after))


if __name__=='__main__':main()
