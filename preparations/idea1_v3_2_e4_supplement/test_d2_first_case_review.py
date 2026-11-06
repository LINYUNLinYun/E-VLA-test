import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import d2_first_case_review as review


def write(path,value):path.write_text(json.dumps(value))


def fixture(root,success=True):
    old=root/'old';old.mkdir();directory=root/'attempt';directory.mkdir()
    obs=dict(agentview_image=np.zeros((4,5,3),np.uint8),robot0_eye_in_hand_image=np.zeros((4,5,3),np.uint8),
             robot0_eef_pos=np.zeros(3),robot0_eef_quat=np.array([0.,0.,0.,1.]),robot0_gripper_qpos=np.zeros(2))
    for dest in (old,directory):np.savez_compressed(dest/'initial_native_observation.npz',**obs,instruction='real BDDL')
    write(old/'result.json',dict(native_info=dict(initial_vector_sha256='vector')))
    write(old/'infer.jsonl',dict(seed=7))
    case=dict(environment='libero_plus',mother_suite='libero_goal',native_max_steps=600);case_hash=review.digest(case)
    job=dict(case=case,case_hash=case_hash,group='perturbed350',source_index=0,video=False,
             paired_results=[dict(mode='full_d10',**review.bound(old/'result.json'))],initial_observation=review.bound(old/'initial_native_observation.npz'),first_input=review.bound(old/'infer.jsonl'))
    initial=review.initial_comparison(obs,'real BDDL',job['initial_observation']['path']);initial['initial_vector_sha256_equal']=True
    write(directory/'initial_comparison.json',initial)
    state=np.zeros(10);np.savez_compressed(directory/'initial_physical_state.npz',state=state)
    steps=1 if success else 600;requests=(steps+4)//5
    result=dict(status='complete',success=success,manifest_hash='manifest',case_hash=case_hash,case=case,mode='v14_d2',gpu_uuid=review.UUID,
        group='perturbed350',source_index=0,H=50,R=5,D=2,steps=steps,requests=requests,initial_comparison=initial,paired_results=job['paired_results'],
        initial_physical_state_sha256=review.value_hash(state),termination='native_success_oracle' if success else 'step_cap',
        native_info=dict(native_hz=20,settle_steps=5,max_steps=600,instruction='real BDDL',initial_vector_sha256='vector'))
    write(directory/'result.json',result);actions=np.zeros((50,7));actions[0,0]=1.2
    infers=[]
    for index in range(requests):
        seed=int(review.digest(['plus-evaluation-v1',case_hash,index])[:8],16)
        response=dict(seed=seed,noise_hash=review.value_digest(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)),noise_used_in_inference=True,
            manifest_hash='manifest',protocol_hash='manifest',mode='v14_d2',category='quality',execute_length=5,raw_actions=np.zeros((50,32)).tolist(),
            observed_calls=dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0,ae_euler_updates=2),
            timing=dict(actual_NFE=2,nodes=[1.,.5,0.]),route=dict(mode='v14_d2',fresh_prefix=True,full_vision=True,skip_vlm=[3,4,9,11],kv_cross_request=False,fallback=False,verify=False),
            input_hash=review.input_digest(review.native_libero_raw(obs),'real BDDL'))
        infers.append(dict(request=index,step=index*5,seed=seed,actions=actions.tolist(),response=response,input_hashes={k:review.value_hash(v) for k,v in obs.items()}))
    (directory/'infer.jsonl').write_text('\n'.join(json.dumps(r) for r in infers)+'\n')
    actual=dict(eef_xyz_world=obs['robot0_eef_pos'].tolist(),eef_quaternion_xyzw=obs['robot0_eef_quat'].tolist(),gripper_qpos=obs['robot0_gripper_qpos'].tolist())
    telemetry=[dict(step=0,actual=actual,executed_command=None)]
    for index in range(steps):
        raw=actions[index%5];clipped=np.clip(raw,-1,1)
        telemetry.append(dict(step=index+1,actual=actual,executed_command=dict(raw_action=raw.tolist(),native_action=clipped.tolist(),
             clipped_components=np.flatnonzero(raw!=clipped).tolist(),success=success and index==steps-1)))
    (directory/'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in telemetry)+'\n')
    return directory/'result.json',job,dict(manifest_hash='manifest')


class FirstCaseReview(unittest.TestCase):
    def test_policy_success_and_failure_both_technically_approve(self):
        for success in (True,False):
            with tempfile.TemporaryDirectory() as temporary:
                path,job,manifest=fixture(Path(temporary),success)
                report=review.audit_episode(path,job,manifest)
                self.assertTrue(report['passed']);self.assertEqual(report['success'],success)

    def test_route_noise_clip_and_video_corruption_fail(self):
        for corruption in ('route','noise','clip','video','seed','action'):
            with tempfile.TemporaryDirectory() as temporary:
                path,job,manifest=fixture(Path(temporary));directory=path.parent
                if corruption=='video':(directory/'video.mp4').write_bytes(b'forbidden')
                elif corruption in ('clip','action'):
                    rows=review.jsonl(directory/'telemetry.jsonl')
                    rows[1]['executed_command']['native_action' if corruption=='clip' else 'raw_action'][0]=1.1
                    (directory/'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                else:
                    rows=review.jsonl(directory/'infer.jsonl')
                    if corruption=='route':rows[0]['response']['observed_calls']['ae_layer_calls']=180
                    elif corruption=='seed':rows[0]['seed']+=1
                    else:rows[0]['response']['noise_hash']='0'*64
                    (directory/'infer.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                with self.assertRaises(ValueError):review.audit_episode(path,job,manifest)

    def test_exclusive_marker_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'continue.json';expected=dict(manifest_hash='fixed',first_result_sha256='hash',continue_remaining=True)
            self.assertEqual(review.write_marker(path,expected),'created_exact_continuation')
            self.assertEqual(review.write_marker(path,expected),'existing_exact_continuation_verified')
            with self.assertRaises(ValueError):review.write_marker(path,dict(expected,first_result_sha256='other'))
            self.assertEqual(json.loads(path.read_text()),expected)

    def test_relative_watcher_output_exact_parse(self):
        root=Path.cwd()/'results/experiment'
        current=dict(argv='python -m preparations.idea1_v3_2_e4_supplement.plus_v14_d2_gpu3_v2 --stage watch --output results/experiment --launch ',cwd=str(Path.cwd()))
        review.verify_watcher_command(current,root)
        with self.assertRaises(ValueError):review.verify_watcher_command(current,root/'other')

    def test_wrong_strict_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);array_path=root/'gate.npz'
            arrays={key:np.zeros((50,32) if key.endswith('raw') else (50,7),np.float32) for key in
                ('current_D10_raw','historical_raw','current_D10_decoded','historical_decoded','native_D2_raw','wrapped_D2_raw','native_D2_decoded','wrapped_D2_decoded')}
            arrays['current_D10_raw'][0,31]=.03125;np.savez_compressed(array_path,**arrays)
            protocol=dict(server_pid=123,manifest_hash='m',gpu_uuid=review.UUID,required_gpu_index=3,no_video=True,strict_same_gpu_gate=True,legacy_gate_passed=True,D=2,H=50,R=5,modes=['v14_d2'])
            check=dict(arrays=review.bound(array_path),historical=dict(raw=dict(passed=True),decoded=dict(passed=True)),d2_native_parity=dict(passed=True))
            gate=dict(passed=True,strict_atol=1e-5,strict_rtol=0,quality_requests=0,required_gpu_uuid=review.UUID,no_video=True,
                checks=[dict(check,saved=dict(mode=mode)) for mode in ('full_d10','v14_d10')])
            write(root/'service.json',dict(server_pid=123,manifest_hash='m',status='ready',protocol=protocol,gate=gate,
                final_preCUDA_guard=dict(passed=True,sample=dict(idle=True,uuid=review.UUID,index=3))))
            with patch.object(review,'rpc',return_value=dict(protocol=protocol)),self.assertRaisesRegex(ValueError,'numerical gate'):
                review.audit_service(root,dict(service_pid=123),dict(manifest_hash='m'),19163)


if __name__=='__main__':unittest.main()
