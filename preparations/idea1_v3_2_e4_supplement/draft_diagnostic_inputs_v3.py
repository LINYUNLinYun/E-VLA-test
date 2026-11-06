"""Fresh/fresh robot/background pairs and same-frame captured clean/noisy pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from .draft_diagnostic_audit import DEFAULT_OUT,ROOT,read,dump
from .draft_diagnostic_protocol import OBS_KEYS,hashes
from .draft_diagnostic_inputs import physical,observations,robot_intervention
from .draft_diagnostic_background_v3 import background_intervention
from .draft_diagnostic_noise_capture import captured_environment,clean_counterfactual
from .draft_diagnostic_language import action_difference
from .draft_diagnostic_state import state_difference
from .libero_plus_env import NativeLiberoPlusEnv,PLUS,isolated_child_environment
from preparations.idea1_v3_2.transfer_client import rpc


def run(a,row):
    out=a.root_directory/'capture_v3/physical_input_pairs'/('case%04d'%row['source_index']);out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__)]+[Path(__file__).with_name(name+'.py') for name in ('draft_diagnostic_noise_capture','draft_diagnostic_inputs','draft_diagnostic_language','draft_diagnostic_state','draft_diagnostic_background_v3','draft_diagnostic_texture_upload')]
    receipt=dict(status='running',case=row,video=False,closed_loop=False,capture_version='v3',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources});env=None
    try:
        health=rpc(a.port,dict(op='health'))['protocol']
        if health['gpu_uuid']!=a.gpu_uuid or not health['recorded_gpu3_gate']['passed']:raise RuntimeError('Strict same-device service required')
        cls=captured_environment(NativeLiberoPlusEnv)
        env=cls(row['case'],render=True,gpu=a.gpu,gpu_uuid=a.gpu_uuid);env.reset()
        if row['category']=='Sensor Noise':actual={k:np.asarray(env.obs[k]).copy() for k in OBS_KEYS}
        else:actual=observations(env)
        before=physical(env)
        if row['category']=='Sensor Noise':
            counter={k:v for k,v in clean_counterfactual(env).items() if k in OBS_KEYS}
            details=dict(kind='official_operator_same_frame_input_capture',capture=env._diagnostic_noise_capture.metadata(),
                expected_only_agentview_changed=True)
        elif row['category']=='Robot Initial States':
            counter,details=robot_intervention(env)
            details['observation_sampling']='fresh before and after arm-qpos intervention; joint visual/proprioception change'
        elif row['category']=='Background Textures':
            counter,details=background_intervention(env,row['case'])
            details['observation_sampling']='fresh before and after texture-only intervention'
        else:raise ValueError('Use saved-observation language client')
        after=physical(env);equal={k:bool(np.array_equal(v,after[k])) for k,v in before.items()}
        proprio_equal={k:bool(np.array_equal(actual[k],counter[k])) for k in OBS_KEYS if 'image' not in k}
        image_equal={k:bool(np.array_equal(actual[k],counter[k])) for k in OBS_KEYS if 'image' in k}
        if row['category']=='Robot Initial States':
            valid=details['nonrobot_qpos_qvel_bitwise_equal'] and all(v for k,v in equal.items() if k not in ('qpos','qvel','contacts'))
        else:valid=all(equal.values()) and all(proprio_equal.values())
        if row['category']=='Sensor Noise':valid=valid and image_equal['robot0_eye_in_hand_image']
        if row['category']=='Background Textures':valid=valid and details['intervention_nonempty']
        np.savez_compressed(out/'actual_observation.npz',**actual);np.savez_compressed(out/'counterfactual_observation.npz',**counter)
        np.savez_compressed(out/'physical_before.npz',**before);np.savez_compressed(out/'physical_after.npz',**after)
        receipt.update(physical_equal=equal,nonimage_observation_bitwise_equal=proprio_equal,image_bitwise_equal=image_equal,
            intervention=details,valid_single_factor_input_pair=bool(valid),state_geometry=state_difference(actual,counter),
            actual_observation_hashes=hashes(actual),counterfactual_observation_hashes=hashes(counter),gpu_uuid=a.gpu_uuid,service_pid=health['server_pid'])
        if not valid:
            receipt.update(status='intervention_gate_failed',reason='Exact physical/proprioception/ownership/nonempty gate failed; no model requests')
            return
        responses={}
        with (out/'responses.jsonl').open('x') as stream:
            for arm in ('actual','counterfactual','counterfactual','actual'):
                response=rpc(a.port,dict(op='predict',mode=row['head'],mother_suite=row['case']['mother_suite'],
                    observation=actual if arm=='actual' else counter,instruction=row['actual_instruction'],seed=7,category='preflight'))
                if response['manifest_hash']!=health['manifest_hash']:raise RuntimeError('Service identity changed')
                if arm in responses:
                    for key in ('actions','raw_actions'):
                        if not np.array_equal(np.asarray(response[key],np.float32),np.asarray(responses[arm][key],np.float32)):
                            raise RuntimeError('Exact same-input repeat differs')
                responses[arm]=response;stream.write(json.dumps(dict(arm=arm,response=response))+'\n')
        receipt.update(status='complete',actions=action_difference(responses['actual']['actions'],responses['counterfactual']['actions']),
            limitation='Input sensitivity at one true state, not proof of closed-loop failure causality. Visual_preencoder RMS/L2 from first diagnostic service is invalid and excluded.')
    except BaseException:
        receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        dump(out/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root-directory',type=Path,default=DEFAULT_OUT)
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--deadline-unix',type=float,default=0);p.add_argument('--child-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    cases=[r for r in read(a.root_directory/'paired_plan.json')['cases'] if r['category']!='Language Instructions']
    if not a.launch:print(json.dumps(dict(dry_run=True,cases=len(cases),calls=48,version='capture_v3')));return
    if not a.gpu_uuid or time.time()>=a.deadline_unix:raise ValueError('Explicit UUID/deadline required')
    if a.child_index is not None:run(a,cases[a.child_index]);return
    reused=[];version_directory=a.root_directory/'capture_v3';version_directory.mkdir(exist_ok=True)
    for index,row in enumerate(cases):
        if time.time()>=a.deadline_unix:raise TimeoutError('Inputs_v2 deadline')
        previous=a.root_directory/'capture_v2/physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
        if previous.exists() and read(previous).get('status')=='complete':
            reused.append(dict(source_index=row['source_index'],path=str(previous),sha256=hashlib.sha256(previous.read_bytes()).hexdigest()))
            dump(version_directory/'reused_input_results.json',reused);continue
        result=a.root_directory/'capture_v3/physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
        if result.exists():
            if read(result).get('status') in ('complete','intervention_gate_failed'):continue
            raise RuntimeError('Existing v2 technical failure requires inspection')
        directory=a.root_directory/'capture_v3';directory.mkdir(exist_ok=True)
        env=isolated_child_environment(row['case'],directory/'physical_input_pairs/config_plus',gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        command=[str(PLUS/'.venv/bin/python'),'-m',__package__+'.draft_diagnostic_inputs_v3','--root-directory',str(a.root_directory),
            '--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--launch']
        with (directory/('physical_input_case%04d.log'%row['source_index'])).open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);code=process.wait()
            if code:raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':main()

