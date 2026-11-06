"""Post-hoc single-case mechanism, separate from fixed representative plots."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT
from .rules_c1 import pose_angle


def run():
    case=dict(suite='libero_goal',task=9,init=7);tail=Path('episodes/libero_goal/task09/init07')
    names=['E2_a30_b0.1_c10','E2_a30_b0.1_c10_no_c'];dirs=[OUT/n/tail for n in names]
    records=[[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()] for d in dirs]
    telemetry=[[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()] for d in dirs]
    results=[json.loads((d/'result.json').read_text()) for d in dirs]
    with np.load(dirs[0]/'arrays.npz') as a,np.load(dirs[1]/'arrays.npz') as b:
        first=next((i for i in range(min(len(records[0]),len(records[1]))) if not np.array_equal(a[f'actions_{i}'],b[f'actions_{i}'])),None)
    if first is None:raise ValueError('Expected single-case divergence was not found')
    x,y=records[0][first],records[1][first];step=x['step'];assert step==y['step']
    assert x['input_hash']==y['input_hash'] and x['seed']==y['seed'] and x['verification_seed']==y['verification_seed']
    same_states=all(telemetry[0][i]==telemetry[1][i] for i in range(step+1))
    assert same_states and x['route']['gate_blocked']==['c_pose'] and y['route']['gate_blocked']==[]
    assert x['route']['path']=='v14_verified' and y['route']['path']=='kalman_verified'
    angle=pose_angle([s['eef_quat'] for s in telemetry[0][max(1,step-4):step+1]])
    output=dict(time=time.time(),posthoc_single_case=True,case=case,conditions=names,first_proposal_divergence_replan=first,step=step,
        all_prior_full_proposal_arrays_exact=True,all_prior_telemetry_exact=same_states,same_input_hash=x['input_hash'],same_seed=x['seed'],same_verification_seed=x['verification_seed'],
        c_world_pose_angle_degrees=angle,theta_c=10,c_operator='<',divergent_requests=[dict(route=r['route'],counts=r['counts'],executed_length=r['executed_length'],draft_noise_hash=r.get('draft_noise_hash'),verify_noise_hash=r.get('verify_noise_hash')) for r in (x,y)],
        results=[dict(success=r['success'],actions=r['actions'],requests=r['predictions'],termination=r['termination']) for r in results],
        sources=[dict(directory=str(d),sha256={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['identity.json','result.json','arrays.npz','infer.jsonl','telemetry.jsonl']}) for d in dirs],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Mechanism evidence for this one identified case after the paired outcome difference was observed. Not a representative sample, population effect, or deployment threshold calibration; formal80 remains unchanged.')
    atomic(OUT/'E2_no_c_case_mechanism_receipt.json',output)
    return output


if __name__=='__main__':run()
