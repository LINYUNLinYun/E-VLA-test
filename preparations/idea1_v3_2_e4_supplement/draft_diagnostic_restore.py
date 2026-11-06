"""Eight fixed Plus restore arms, reusing old actual outcomes only after exact reproduction gates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from .draft_diagnostic_audit import ROOT,DEFAULT_OUT,read,dump
from .draft_diagnostic_protocol import OBS_KEYS,hashes
from .draft_diagnostic_inputs import physical,observations
from .draft_diagnostic_service import strict_compare,comparisons_accepted
from .libero_plus_env import NativeLiberoPlusEnv,isolated_child_environment,PLUS
from .native_envs import digest
from preparations.idea1_v3_2.transfer_client import rpc
from preparations.idea1_v3_1.common import atomic_json


def selected(directory):
    return [r for r in read(Path(directory)/'paired_plan.json')['cases'] if r['category'] in ('Language Instructions','Sensor Noise')]


LANGUAGE_SCOPE=dict(draft_goal='put/place is a direct synonym; same bowl/stove referents',
    draft_spatial='Descriptive dark dish/flat dish/dessert dish correspond broadly to bowl/plate/ramekin but make grounding more ambiguous.',
    draft_object='Added children/snack narrative; intended milk-to-basket target retained.',
    draft_10='Book described as bound printed pages; wheeled utility cart for caddy risks semantic/reference drift. Not a strict synonym claim.')


def gate_reference(env,row,port):
    ledgerpath=ROOT/'reports/idea1_libero_plus_half_final_20261022T2124Z/audit_ledger.json'
    entry=next(e for e in read(ledgerpath)['selected_completed'] if e['group']=='perturbed350' and e['source_index']==row['source_index'] and e['mode']==row['head'])
    resultpath=ROOT/entry['path'];old=read(resultpath);directory=resultpath.parent
    with (directory/'infer.jsonl').open() as stream:first=json.loads(next(stream))
    with np.load(directory/'initial_native_observation.npz',allow_pickle=False) as archive:
        expected={key:archive[key].copy() for key in OBS_KEYS}
    obs={key:np.asarray(env.obs[key]).copy() for key in OBS_KEYS}
    obs_equal={key:bool(np.array_equal(obs[key],expected[key])) for key in OBS_KEYS}
    checks=dict(observation_bitwise_equal=obs_equal,
        init_vector_sha256_equal=env.info['initial_vector_sha256']==old['native_info']['initial_vector_sha256'],
        instruction_equal=env.info['instruction']==old['native_info']['instruction'],
        case_equal=env.case==old['case'],old_actual_success=old['success'],old_actual_steps=old['steps'],
        references_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (ledgerpath,resultpath,directory/'initial_native_observation.npz',directory/'infer.jsonl')},
        physical_scope='Old run records complete pre-settle init vector hash plus five observation fields, not a complete post-settle simulator snapshot. New intervention uses one unchanged simulator state.')
    checks['passed']=all(obs_equal.values()) and checks['init_vector_sha256_equal'] and checks['instruction_equal'] and checks['case_equal']
    if not checks['passed']:return checks,obs
    response=rpc(port,dict(op='predict',mode=row['head'],mother_suite=row['case']['mother_suite'],observation=obs,
        instruction=env.info['instruction'],seed=int(first['seed']),category='preflight'))
    comparisons=dict(actions=strict_compare(response['actions'],first['actions'],(50,7)),
        raw_actions=strict_compare(response['raw_actions'],first['response']['raw_actions'],(50,32)))
    checks.update(comparisons=comparisons,input_hash_equal=response['input_hash']==first['response']['input_hash'])
    checks['passed']=checks['passed'] and comparisons_accepted(comparisons) and checks['input_hash_equal']
    return checks,obs


def run(a,row):
    out=a.directory/'restore8'/('case%04d'%row['source_index']);out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('draft_diagnostic_inputs.py'),Path(__file__).with_name('draft_diagnostic_language.py'),Path(__file__).with_name('draft_diagnostic_state.py')]
    receipt=dict(status='running',case=row,steps=0,requests=0,video=False,actual_arm_reexecuted=False,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources});env=None
    try:
        health=rpc(a.port,dict(op='health'))['protocol']
        if health['gpu_uuid']!=a.gpu_uuid or not health['recorded_gpu3_gate']['passed']:raise RuntimeError('Same-device gated draft service required')
        env=NativeLiberoPlusEnv(row['case'],render=True,gpu=a.gpu,gpu_uuid=a.gpu_uuid);env.reset()
        gate,actual=gate_reference(env,row,a.port);receipt['actual_reproduction_gate']=gate
        if not gate['passed']:
            receipt.update(status='actual_reproduction_gate_failed',success=None);return
        before=physical(env);np.savez_compressed(out/'actual_physical.npz',**before)
        np.savez_compressed(out/'actual_observation.npz',**actual)
        instruction=row['mother_bddl_instruction'] if row['category']=='Language Instructions' else env.info['instruction']
        if row['category']=='Language Instructions':receipt['language_semantic_scope']=LANGUAGE_SCOPE[row['head']]
        if row['category']=='Sensor Noise':
            receipt['old_wrapper_noise']=env.env.noise;env.env.noise=0;observations(env)
        after=physical(env);np.savez_compressed(out/'intervention_physical.npz',**after)
        obs={key:np.asarray(env.obs[key]).copy() for key in OBS_KEYS};np.savez_compressed(out/'intervention_observation.npz',**obs)
        receipt.update(physical_bitwise_equal={key:bool(np.array_equal(value,after[key])) for key,value in before.items()},
            instruction=instruction,actual_instruction=env.info['instruction'],initial_observation_hashes=hashes(obs),
            gpu_uuid=a.gpu_uuid,server_pid=health['server_pid'],manifest_hash=health['manifest_hash'])
        if not all(receipt['physical_bitwise_equal'].values()):raise RuntimeError('Physical state changed during input intervention')
        if row['category']=='Language Instructions' and hashes(actual)!=hashes(obs):raise RuntimeError('Prompt-only observation changed')
        if row['category']=='Sensor Noise' and not np.array_equal(actual['robot0_eye_in_hand_image'],obs['robot0_eye_in_hand_image']):
            raise RuntimeError('Noise intervention changed wrist camera')
        rpc(a.port,dict(op='reset'));success=False;case_hash=digest(row['case'])
        with (out/'infer.jsonl').open('x') as inference,(out/'telemetry.jsonl').open('x') as telemetry:
            while receipt['steps']<600 and not success:
                if time.time()>=a.deadline_unix:raise TimeoutError('Restore diagnostic deadline')
                obs={key:np.asarray(env.obs[key]) for key in OBS_KEYS}
                seed=int(digest(['plus-evaluation-v1',case_hash,receipt['requests']])[:8],16)
                response=rpc(a.port,dict(op='predict',mode=row['head'],mother_suite=row['case']['mother_suite'],
                    observation=obs,instruction=instruction,seed=seed,category='quality'))
                if response['manifest_hash']!=health['manifest_hash']:raise RuntimeError('Model service changed')
                actions=np.asarray(response['actions'],np.float32)
                if actions.shape!=(50,7) or not np.isfinite(actions).all():raise RuntimeError('Invalid actions')
                inference.write(json.dumps(dict(step=receipt['steps'],request=receipt['requests'],seed=seed,response=response))+'\n');inference.flush()
                receipt['requests']+=1
                for action in actions[:min(5,600-receipt['steps'])]:
                    _,audit=env.step_native(action);receipt['steps']+=1;success=audit['success']
                    telemetry.write(json.dumps(dict(step=receipt['steps'],actual=env.actual(),executed_command=audit))+'\n')
                    if success:break
                atomic_json(out/'result.json',receipt)
        receipt.update(status='complete',success=bool(success),termination='success' if success else 'step_cap',
            paired_effect=int(success)-int(gate['old_actual_success']),
            claim_scope='One preregistered case per head/category. Closed-loop restoration compared with old actual after exact observation/action reproduction; no suite-wide causal estimate.')
    except BaseException:
        receipt.update(status='technical_failure',success=None,error=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        atomic_json(out/'result.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=DEFAULT_OUT)
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--deadline-unix',type=float,default=0);p.add_argument('--child-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    cases=selected(a.directory)
    if not a.launch:print(json.dumps(dict(dry_run=True,cases=len(cases),new_episodes=8,old_actual_reused_only_after_gate=True,video=False)));return
    if not a.gpu_uuid or time.time()>=a.deadline_unix:raise ValueError('Explicit device/deadline required')
    if a.child_index is not None:run(a,cases[a.child_index]);return
    for index,row in enumerate(cases):
        if time.time()>=a.deadline_unix:raise TimeoutError('Restore8 deadline')
        result=a.directory/'restore8'/('case%04d'%row['source_index'])/'result.json'
        if result.exists():
            if read(result).get('status') in ('complete','actual_reproduction_gate_failed'):continue
            raise RuntimeError('Existing incomplete restore requires inspection: '+str(result))
        env=isolated_child_environment(row['case'],a.directory/'restore8/config_plus',gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        command=[str(PLUS/'.venv/bin/python'),'-m',__package__+'.draft_diagnostic_restore',
            '--directory',str(a.directory),'--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--launch']
        with (a.directory/('restore_case%04d.log'%row['source_index'])).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=child.wait()
            if code:raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':main()
