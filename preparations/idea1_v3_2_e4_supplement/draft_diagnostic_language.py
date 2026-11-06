"""Four metadata-selected Plus language cases, three prompts on identical saved real observations."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .draft_diagnostic_audit import DEFAULT_OUT,ROOT,HEADS,read,dump,csvout
from preparations.idea1_v3_2.transfer_client import rpc


def action_difference(first,second):
    a,b=np.asarray(first,np.float32),np.asarray(second,np.float32)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid paired actions')
    delta=a-b
    xyz=np.linalg.norm(delta[:5,:3],axis=1);rot=delta[:5,3:6]
    result=dict(bitwise_equal=bool(np.array_equal(a,b)),max_abs=float(np.max(np.abs(delta))),rms=float(np.sqrt(np.mean(delta**2))),
        first5_max_abs=float(np.max(np.abs(delta[:5]))),first5_rms=float(np.sqrt(np.mean(delta[:5]**2))),
        xyz_first5_l2_mean=float(np.mean(xyz)),xyz_first5_l2_max=float(np.max(xyz)),xyz_first1_l2=float(xyz[0]),
        xyz_first5_l2_by_step=xyz.tolist(),rotation_first5_l2_mean=float(np.linalg.norm(rot,axis=1).mean()),
        gripper_max_abs=float(np.max(np.abs(delta[:,6]))),gripper_first5_sign_disagreements=int(np.sum((a[:5,6]>0)!=(b[:5,6]>0))),
        gripper_first5_abs_mean=float(np.mean(np.abs(delta[:5,6]))),
        units='Decoded native controller action units; xyz/rotation/gripper separated. No physical EEF distance conversion.')
    for axis,column in zip('xyz',range(3)):
        result['rotation_'+axis+'_first5_rms']=float(np.sqrt(np.mean(rot[:,column]**2)))
        result['rotation_'+axis+'_first5_max_abs']=float(np.max(np.abs(rot[:,column])))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=DEFAULT_OUT)
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu-uuid');p.add_argument('--launch',action='store_true');a=p.parse_args()
    plan=read(a.directory/'paired_plan.json');cases=[r for r in plan['cases'] if r['category']=='Language Instructions']
    if not a.launch:print(json.dumps(dict(dry_run=True,cases=len(cases),calls=6*len(cases),video=False)));return
    health=rpc(a.port,dict(op='health'))['protocol']
    if not a.gpu_uuid or health['gpu_uuid']!=a.gpu_uuid or not health['recorded_gpu3_gate']['passed']:
        raise RuntimeError('Explicit same-device strict-gated service required')
    output=a.directory/'language_same_observation';output.mkdir(exist_ok=False)
    ledger=read(ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json');summaries=[]
    for row in cases:
        entry=next(e for e in ledger['selected_completed'] if e['group']=='perturbed350' and e['source_index']==row['source_index'] and e['mode']==row['head'])
        source=(ROOT/entry['path']).parent/'initial_native_observation.npz'
        with np.load(source,allow_pickle=False) as archive:obs={k:archive[k].copy() for k in archive.files if k!='instruction'}
        prompts=dict(actual=row['actual_instruction'],mother_bddl=row['mother_bddl_instruction'],legacy_filename=row['legacy_instruction'])
        responses={}
        with (output/(row['head']+'_responses.jsonl')).open('x') as stream:
            for repeat,arm in enumerate(('actual','mother_bddl','legacy_filename','legacy_filename','mother_bddl','actual')):
                response=rpc(a.port,dict(op='predict',mode=row['head'],mother_suite=row['case']['mother_suite'],
                    observation=obs,instruction=prompts[arm],seed=7,category='preflight'))
                if response['manifest_hash']!=health['manifest_hash']:raise RuntimeError('Service identity changed')
                stream.write(json.dumps(dict(arm=arm,repeat=repeat,instruction=prompts[arm],response=response))+'\n')
                if arm in responses:
                    if not np.array_equal(np.asarray(response['actions'],np.float32),np.asarray(responses[arm]['actions'],np.float32)):
                        raise RuntimeError('Same-input same-device repeat action differs')
                    if not np.array_equal(np.asarray(response['raw_actions'],np.float32),np.asarray(responses[arm]['raw_actions'],np.float32)):
                        raise RuntimeError('Same-input same-device repeat raw action differs')
                responses[arm]=response
        for first,second in (('actual','mother_bddl'),('mother_bddl','legacy_filename'),('actual','legacy_filename')):
            summaries.append(dict(head=row['head'],source_index=row['source_index'],first=first,second=second,
                first_instruction=prompts[first],second_instruction=prompts[second],**action_difference(responses[first]['actions'],responses[second]['actions'])))
    csvout(output/'action_differences.csv',summaries)
    dump(output/'receipt.json',dict(status='complete',cases=len(cases),gpu_uuid=a.gpu_uuid,server_pid=health['server_pid'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol='same saved real native observation, three actual prompts, ABC-CBA repeat check, each head separately',
        limitation='Action sensitivity alone does not establish closed-loop success causality.',video=False))


if __name__=='__main__':main()
