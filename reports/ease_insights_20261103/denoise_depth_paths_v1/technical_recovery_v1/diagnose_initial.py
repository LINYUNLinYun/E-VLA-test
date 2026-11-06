"""Recreate only the rejected initial observation; no VLA or policy action runs."""
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
from preparations.ease_insights_20261103.denoise_depth_paths_v1 import common
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v2_1.runtime_client import official_init_states,input_digest
from preparations.idea1_v2_2.common import seed

OUT=Path(__file__).resolve().parent
STUDY=common.ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'


def main():
    m=common.read_manifest(STUDY/'manifest.json',verify_sources=True);scene=dict(suite='libero_goal',task=3,init=45)
    ref=STUDY/'initial_observations/libero_goal/task03/init45.json';expected=json.loads(ref.read_text())
    class Seeded(Environments):
        def acquire(self,suite,task):
            random.seed(7);np.random.seed(7);return super().acquire(suite,task)
    pool=Seeded(reuse=False)
    try:
        env,definition,bddl,selected,_=pool.acquire(scene['suite'],scene['task']);vector=np.asarray(official_init_states(selected,scene['task'])[scene['init']])
        random.seed(7);np.random.seed(7);env.seed(7);env.reset();obs=env.set_init_state(vector.copy())
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        current={k:np.asarray(v) for k,v in obs.items() if isinstance(v,np.ndarray)}
        actual=input_digest(current,definition.language);generation_seed=seed(scene['suite'],scene['task'],scene['init'],0)
        physical=dict(init_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),bddl_sha256=hashlib.sha256(bddl.read_bytes()).hexdigest(),
            prompt=definition.language,env_seed=7,settling=10,policy_action_limit=common.LIMITS[scene['suite']],control_hz=float(env.env.control_freq))
        source=STUDY/'dense_d2/episodes/libero_goal/task03/init45/input_0000.npz';differences=[]
        with np.load(source,allow_pickle=False) as saved:
            keys=set(saved.files)-{'instruction','seed'}
            if keys!=set(current):differences.append(dict(key='field_set',expected=sorted(keys),actual=sorted(current)))
            for k in sorted(keys&set(current)):
                a,b=saved[k],current[k];same=a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)
                if not same:
                    details=dict(key=k,expected_shape=list(a.shape),actual_shape=list(b.shape),expected_dtype=str(a.dtype),actual_dtype=str(b.dtype))
                    if a.shape==b.shape and a.dtype.kind in 'buif' and b.dtype.kind in 'buif':
                        details.update(max_abs=float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max()),changed_values=int(np.count_nonzero(a!=b)))
                    differences.append(details)
        passed=actual==expected['input_hash'] and generation_seed==expected['seed'] and physical==expected['physical_assets'] and not differences
        stamp=str(time.time_ns());np.savez_compressed(OUT/('initial_'+stamp+'.npz'),**current,instruction=np.asarray(definition.language),seed=np.asarray(generation_seed))
        result=dict(status='pass' if passed else 'mismatch',time=time.time(),manifest_hash=m['manifest_hash'],case=scene,
            actual_input_hash=actual,expected_input_hash=expected['input_hash'],noise_seed=generation_seed,physical_assets_equal=physical==expected['physical_assets'],
            differences=differences,VLA_model_started=False,VLA_inference_performed=False,policy_actions_executed=0,settling_actions_executed=10,
            scope='Exact original reset/settling/input contract in a fresh environment process on the same assigned EGL device. Original scientific records and expected first input remain unchanged.',
            source_sha256=common.sha(__file__))
        common.atomic_json(OUT/('diagnostic_'+stamp+'.json'),result);print(json.dumps(result),flush=True)
    finally:pool.close()


if __name__=='__main__':main()
