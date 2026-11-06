"""Serial case blocks, each all4 arms same physical model/render service."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v3_3.egl_mapping import mapping
from .design import OUT
from .episode import episode


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);a=p.parse_args()
    m=json.loads((OUT/'manifest.json').read_text());configs={c['id']:c for c in m['conditions']};devices=mapping();index=int(os.environ['MUJOCO_EGL_DEVICE_ID'])
    if devices.get(index)!=a.gpu:raise ValueError('EGL is not the assigned model physicalGPU')
    binding=dict(model_gpu=a.gpu,render_gpu=a.gpu,EGL_index=index,CVD=os.environ['CUDA_VISIBLE_DEVICES'],EGL_mapping=devices,rollout_pid=os.getpid())
    probed=False
    indices=[i for i in range(40) if (i//4)%a.shards==a.shard]
    for i in indices:
        item=m['schedule'][i];case=item['case'];ref=OUT/'case_references'/case['suite']/f"task{case['task']:02d}_init08.json"
        with lock(str(ref)+'.lock'):
            block=json.loads(ref.read_text()) if ref.exists() else {}
            for cid in item['condition_order']:
                d=OUT/cid/'episodes'/case['suite']/f"task{case['task']:02d}"/'init08'
                if (d/'complete.json').exists():continue
                for attempt in range(2):
                    if d.exists():d.rename(d.with_name(d.name+'.invalid.'+uuid.uuid4().hex))
                    pool=Environments(reuse=False)
                    try:
                        result=episode(d,a.port,case,configs[cid],pool,m['manifest_hash'],probe=not probed,block=block,binding=binding)
                        atomic(ref,dict(block,case=case,physical_gpu=a.gpu,manifest_hash=m['manifest_hash']));probed=True
                        print(json.dumps(dict(condition=cid,case=case,success=result['success'],actions=result['actions'],policy_wall_s=result['policy_wall_s'])),flush=True);break
                    except Exception as exc:
                        atomic(d/'technical_error.json',dict(time=time.time(),attempt=attempt+1,error=repr(exc)))
                        if attempt==1:raise
                    finally:pool.close()
    atomic(OUT/'workers'/f'gpu{a.gpu}'/'rollout_finished.json',dict(time=time.time(),case_indices=indices,episodes=4*len(indices)))


if __name__=='__main__':main()
