"""Bounded actual-environment interface gate; never imports or calls a model."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from .evaluate_transfer import actual_record,make_adapter,verify_registered_initialization
from .transfer_registration import load_registration


def selected_cases(reg,environment):
    cases=[c for c in reg['cases'] if c['environment']==environment]
    if environment!='simpler':return [cases[0],cases[0]]
    coke=[c for c in cases if c['task']=='google_robot_pick_coke_can']
    move=[c for c in cases if c['task']=='google_robot_move_near']
    return [coke[i] for i in (0,24,25,49)]+[move[0],move[-1]]


def physical_probes(environment):
    axes=(0,1) if environment=='pusht' else (0,1,2,3,4,5,6)
    return [(axis,sign,np.eye(7)[axis]*sign*(1. if axis==6 else .2))
            for axis in axes for sign in (1,-1)]


def write(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))


def save_raw(path,raw):
    np.savez_compressed(str(path),**raw)
    return {k:dict(shape=list(np.asarray(v).shape),sha256=hashlib.sha256(np.asarray(v).tobytes()).hexdigest()) for k,v in raw.items()}


def run(reg,auth,environment,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    reference=json.loads(Path(reg['input_files']['source_reference']['path']).read_text())
    result=dict(status='running',environment=environment,registration_hash=reg['registration_hash'],
        model_calls=0,quality_requests=0,scientific_samples=0,rows=[],source_files=reg['source_files'],
        note='Actual interface probe only; no task-success optimization or scientific denominator',
        gpu_uuid=auth['gpu_uuid'] if environment=='simpler' else None)
    start=time.monotonic()
    try:
        for index,case in enumerate(selected_cases(reg,environment)):
            if time.monotonic()-start>540:raise TimeoutError('Bounded9minute gate deadline')
            env=None
            row=dict(case_id=case['case_id'],fresh_instance=index,reset_options=case.get('reset_options'),probes=[])
            result['rows'].append(row)
            try:
                env=make_adapter(case,reference,auth)
                if environment=='calvin':
                    env.reset_sequence(case['sequence']['initial_condition']);env.start_subtask(case['sequence']['tasks'][0])
                elif environment=='simpler':env.reset(case['seed'],case['reset_options'])
                else:env.reset(case['seed'])
                row.update(initial=actual_record(env),info=env.info,registered_initialization=verify_registered_initialization(case,env),
                    raw=save_raw(output/('initial_%02d.npz'%index),env.raw()))
                # Probe only first selected case, with both signs for every
                # physical channel. These sequential dynamics are not SR data.
                if index==0:
                    for axis,sign,action in physical_probes(environment):
                        before=actual_record(env);success,terminal,audit=env.step(action);after=actual_record(env)
                        row['probes'].append(dict(axis=axis,sign=sign,source_action=action.tolist(),before=before,after=after,
                            controller_audit=audit,oracle=bool(success),terminal=bool(terminal),
                            raw=save_raw(output/('probe_%d_%s.npz'%(axis,'pos' if sign>0 else 'neg')),env.raw())))
                row['status']='complete'
            finally:
                if env is not None:env.close()
            write(output/'receipt.json',result)
        if environment!='simpler':
            a,b=result['rows']
            key='robot_obs' if environment=='calvin' else 'native_state'
            x=np.asarray(a['initial']['native_observation_fields'][key]);y=np.asarray(b['initial']['native_observation_fields'][key])
            result['fresh_initial_comparison']=dict(max_abs_state_difference=float(np.max(np.abs(x-y))),
                initial_raw_hashes_equal=a['raw']==b['raw'],state_equal=bool(np.allclose(x,y,atol=1e-7,rtol=0)))
            if not result['fresh_initial_comparison']['state_equal']:raise RuntimeError('Fresh duplicate initial state differs')
        result['status']='complete_actual_interface_gate_requires_parent_review'
    except Exception:
        result.update(status='technical_failure_stop_no_quality',error=traceback.format_exc())
    result['wall_s']=time.monotonic()-start
    write(output/'receipt.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',required=True)
    p.add_argument('--authorization',required=True);p.add_argument('--environment',choices=['calvin','simpler','pusht'],required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    reg=load_registration(a.registration,verify_sources=True);auth=json.loads(Path(a.authorization).read_text())
    if auth['registration_hash']!=reg['registration_hash'] or auth['gpu']!=3 or auth['gpu_uuid']!='GPU-555cde36-5e41-bc44-0c69-ea47358fd703':
        raise PermissionError('Exact staged Bv2/GPU3 identity required')
    result=run(reg,auth,a.environment,a.output)
    print(json.dumps(dict(status=result['status'],environment=a.environment,rows=len(result['rows']),wall_s=result['wall_s'],quality_requests=0)))
    if result['status'].startswith('technical_failure'):raise SystemExit(1)


if __name__=='__main__':main()
