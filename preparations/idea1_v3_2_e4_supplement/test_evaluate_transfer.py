"""CPU-only B registration/resume/RPC/telemetry contracts, no real policy."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np

from .transfer_registration import build_registration,check_registration,digest,MODES,SCOPE,official_google_initializations
from .evaluate_transfer import (run_attempt,request_seed,launch_gate,validate_response,
    complete_results,next_attempt,summarize,counter_namespace,decode_video,actual_record,verify_registered_initialization)
from preparations.idea1_v2_1.runtime_client import input_digest,value_digest


class FakeAdapter:
    def __init__(self,*unused):
        self.info={'cpu_test_fixture':True};self.steps=0;self.resets=0
        self.native=SimpleNamespace(obs={'robot_obs':np.zeros(15),'scene_obs':np.zeros(24)})
    def reset_sequence(self,condition):self.resets+=1
    def start_subtask(self,task):return 'perform '+task
    def raw(self):return dict(base_rgb=np.zeros((4,4,3),dtype=np.uint8),
        wrist_rgb=np.zeros((2,2,3),dtype=np.uint8),state8=np.zeros(8,dtype=np.float32))
    def telemetry(self):return dict(eef_xyz_world=[self.steps*.001,0.,.4],gripper_width_m=.04)
    def step(self,action):
        self.steps+=1
        return True,False,dict(native_action=np.asarray(action).tolist(),actual_controller='CPU test fixture')
    def close(self):self.closed=True


class BDriverTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        canonical=[dict(index=i,initial_condition={'index':i},tasks=['a','b','c','d','e']) for i in range(1000)]
        sequence=self.root/'sequences.json';sequence.write_text(json.dumps(dict(sequences=canonical,sequence_sha256=digest(canonical))))
        source=self.root/'source.json';source.write_text(json.dumps(dict(robot_base_world_matrix=np.eye(4).tolist(),
            actual_eef_world_xyz=[.1,.2,.3],actual_eef_world_quaternion_xyzw=[1.,0,0,0],actual_gripper_qpos=[.02,-.02])))
        policy=self.root/'policy.json';payload={'family':'libero_transfer','cpu_test_fixture':True}
        payload['manifest_hash']=digest(payload);policy.write_text(json.dumps(payload))
        self.registration=build_registration(sequence,source,policy)
        self.case=self.registration['cases'][1]  # no preregistered video for the fast fixture
        self.service=dict(service_manifest_hash='s'*64,gpu_uuid='GPU-CPU-test-not-real',gpu=3)
        self.args=SimpleNamespace(port=1,deadline_unix=time.time()+1000,save_observations=False)
        self.requests=[]
    def tearDown(self):self.temp.cleanup()

    def response(self,request):
        mode=request['mode'];draft=mode.startswith('draft_');reg=self.registration
        case=next(c for c in reg['cases'] if c['case_id']==request['case_id'])
        noise=np.random.default_rng(request['seed']).standard_normal((50,32),dtype=np.float32)
        return dict(ok=True,manifest_hash=self.service['service_manifest_hash'],registration_hash=reg['registration_hash'],
            counter_namespace=counter_namespace(reg),mode=mode,execute_length=5,
            actions=np.zeros((50,7)).tolist(),raw_actions=np.zeros((50,32)).tolist(),
            input_hash=input_digest(request['raw'],request['instruction']),noise_hash=value_digest(noise),
            noise_used_in_inference=not draft,observed_calls=dict(vision_encoder_calls=1,vlm_prefix_calls=0 if draft else 1,
                ae_layer_calls=0 if draft else 180,learned_head_calls=1 if draft else 0,verify_calls=0),
            route=dict(fresh_prefix=True,fallback=False,verify=False,kv_cross_request=False),
            attribution=dict(scope=SCOPE,case_id=case['case_id'],case_hash=digest(case),environment=case['environment'],
                mode=mode,service_manifest_hash=self.service['service_manifest_hash']))
    def rpc(self,port,request):
        self.requests.append(request)
        return self.response(request) if request['op']=='predict' else {'ok':True}
    def directory(self,mode='full_d10',attempt=0):
        return self.root/'run'/'episodes'/self.case['environment']/self.case['case_id']/mode/('attempt%03d'%attempt)
    def run_fixture(self,mode='full_d10'):
        self.adapter=FakeAdapter()
        return run_attempt(self.args,self.registration,self.case,mode,self.directory(mode),self.service,{},
            request=self.rpc,adapter_factory=lambda *args:self.adapter)

    def test_expanded_independent_matrix(self):
        r=self.registration;self.assertEqual((len(r['cases']),r['condition_count']),(300,1800))
        self.assertEqual([c['sequence']['index'] for c in r['cases'][:100]],list(range(100)))
        self.assertEqual(sum(c['environment']=='simpler' for c in r['cases']),100)
        self.assertEqual(sum(c['environment']=='pusht' for c in r['cases']),100)
        self.assertEqual(len([c for c in r['cases'] if c['video']]),4)
        self.assertIsNone(r['denoising']['draft_goal']);self.assertEqual(r['denoising']['v14_d10'],10)
        broken=copy.deepcopy(r);broken['cases'][0]['max_steps_per_subtask']=1
        broken['registration_hash']=digest({k:v for k,v in broken.items() if k!='registration_hash'})
        with self.assertRaises(ValueError):check_registration(broken)
        broken=copy.deepcopy(r);broken['scope']='native_LIBERO_Plus'
        broken['registration_hash']=digest({k:v for k,v in broken.items() if k!='registration_hash'})
        with self.assertRaises(ValueError):check_registration(broken)

    def test_static_official_google_options_have_real_coverage(self):
        options,audit=official_google_initializations()
        coke=options['google_robot_pick_coke_can'];move=options['google_robot_move_near']
        self.assertEqual(len({r['geometry_key'] for r in coke}),50)
        self.assertEqual({r['official_orientation'] for r in coke},{'lr_switch','upright'})
        self.assertEqual(len({tuple(r['official_xy']) for r in coke}),25)
        self.assertEqual(len({r['official_episode_id'] for r in move}),50)
        self.assertEqual(len({r['geometry_key'] for r in move}),10)
        for triplet in range(5):
            rows=[r for r in move if r['triplet_index']==triplet]
            self.assertEqual(len(rows),10)
            self.assertEqual([sum(r['layout_index']==layout for r in rows) for layout in (0,1)],[5,5])
            self.assertEqual(len({(r['source_object_index'],r['target_object_index']) for r in rows}),6)
        self.assertFalse(audit['runtime_measured_initial_poses'])
        self.assertEqual(audit['move_near']['unique_object_geometries'],10)

    def test_actual_object_geometry_is_logged_not_just_requested_options(self):
        adapter=FakeAdapter()
        obj=SimpleNamespace(name='native-object',pose=SimpleNamespace(p=np.array([.1,.2,.3]),q=np.array([1.,0,0,0])))
        adapter.native.native=SimpleNamespace(episode_objs=[obj],urdf_version='native-seeded-visual',
                                            episode_source_obj=obj,episode_target_obj=obj)
        actual=actual_record(adapter)['actual']
        self.assertEqual(actual['object_poses_world']['native-object']['xyz'],[.1,.2,.3])
        self.assertEqual(actual['native_urdf_visual_version'],'native-seeded-visual')

    def test_native_reset_must_honor_registered_options_before_quality(self):
        coke=next(c for c in self.registration['cases'] if c.get('task')=='google_robot_pick_coke_can')
        engine=SimpleNamespace(obj_init_options=coke['reset_options']['obj_init_options'])
        env=SimpleNamespace(native=SimpleNamespace(native=engine))
        self.assertEqual(verify_registered_initialization(coke,env)['status'],'registered_native_options_honored')
        engine.obj_init_options=dict(init_xy=[0,0],orientation='lr_switch')
        with self.assertRaises(ValueError):verify_registered_initialization(coke,env)
        move=next(c for c in self.registration['cases'] if c.get('task')=='google_robot_move_near')
        init=move['initialization'];ids=init['object_ids']
        engine=SimpleNamespace(episode_model_ids=ids,episode_source_obj=SimpleNamespace(name=ids[init['source_object_index']]),
                               episode_target_obj=SimpleNamespace(name=ids[init['target_object_index']]))
        env=SimpleNamespace(native=SimpleNamespace(native=engine,reset_info={'episode_id':init['official_episode_id']}))
        self.assertTrue(verify_registered_initialization(move,env)['not_50_unique_geometry'])
        env.native.reset_info['episode_id']=59
        with self.assertRaises(ValueError):verify_registered_initialization(move,env)

    def test_plus_service_or_missing_grant_cannot_launch(self):
        r=self.registration;authorization=dict(scope=SCOPE,allow_quality=True,registration_hash=r['registration_hash'],
            service_manifest_hash=self.service['service_manifest_hash'],model_reference_manifest_hash=r['policy_reference']['manifest_hash'],
            gpu=3,gpu_uuid=self.service['gpu_uuid'],modes=list(MODES),environments=['calvin','simpler','pusht'],deadline_unix=time.time()+500)
        protocol=dict(manifest_hash=self.service['service_manifest_hash'],family='libero_transfer',
            model_reference_manifest_hash=r['policy_reference']['manifest_hash'],gpu_uuid=self.service['gpu_uuid'],legacy_gate_passed=True,
            authorized_scientific_registrations=[r['registration_hash']],counter_namespace=counter_namespace(r))
        launch_gate(r,authorization,{'protocol':protocol},'calvin',time.time()+100)
        with self.assertRaises(PermissionError):launch_gate(dict(r,schema=1),authorization,{'protocol':protocol},'calvin',time.time()+100)
        plus=dict(protocol,authorized_scientific_registrations=[])
        with self.assertRaises(PermissionError):launch_gate(r,authorization,{'protocol':plus},'calvin',time.time()+100)
        with self.assertRaises(PermissionError):launch_gate(r,dict(authorization,allow_quality=False),{'protocol':protocol},'calvin',time.time()+100)

    def test_chain_state_carry_complete_and_mode_independent_noise(self):
        result=self.run_fixture();self.assertEqual(result['status'],'complete')
        self.assertEqual((result['completed_subtasks'],result['native_steps'],result['requests']),(5,5,5))
        self.assertEqual(self.adapter.resets,1);self.assertTrue(self.adapter.closed)
        full=[r['seed'] for r in self.requests if r['op']=='predict'];self.requests=[]
        draft=self.run_fixture('draft_goal')
        self.assertEqual(full,[r['seed'] for r in self.requests if r['op']=='predict'])
        self.assertIsNone(draft['denoising'])
        rows=[json.loads(line) for line in (self.directory()/'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),6);self.assertIn('gripper_width_m',rows[-1]['actual'])
        self.assertTrue(rows[-1]['oracle_success']);self.assertIsNotNone(rows[-1]['executed_command'])

    def test_full_case_resume_and_artifact_integrity(self):
        result=self.run_fixture();selected,attempts=complete_results(self.root/'run',self.registration)
        self.assertEqual(len(selected),1);self.assertEqual(len(attempts),1)
        self.assertEqual(next_attempt(self.directory().parent).name,'attempt001')
        (self.directory()/'telemetry.jsonl').write_text('corrupted')
        with self.assertRaises(ValueError):complete_results(self.root/'run',self.registration)

    def test_duplicate_complete_is_not_cherrypicked(self):
        self.run_fixture();shutil.copytree(self.directory(),self.directory(attempt=1))
        with self.assertRaises(ValueError):complete_results(self.root/'run',self.registration)

    def test_technical_failure_not_policy_zero(self):
        def broken(port,request):
            if request['op']=='predict':raise ConnectionError('CPU fixture transport failure')
            return {'ok':True}
        result=run_attempt(self.args,self.registration,self.case,'full_d10',self.directory(),self.service,{},
            request=broken,adapter_factory=lambda *args:FakeAdapter())
        self.assertEqual(result['status'],'technical_failure');self.assertIsNone(result['success'])
        self.assertEqual(result['attempted_quality_requests'],1);self.assertEqual(result['quality_responses'],0)
        selected,attempts=complete_results(self.root/'run',self.registration);report=summarize(self.registration,selected,attempts)
        self.assertEqual(report['technical_failure_attempts'],1);self.assertEqual(report['valid_complete_conditions'],0)
        self.assertIsNone(report['metrics']['calvin_chains']['full_d10']['success_rate'])

    def test_wrong_input_or_route_hash_is_invalid(self):
        self.run_fixture();request=next(r for r in self.requests if r['op']=='predict');response=self.response(request)
        response['input_hash']='wrong'
        with self.assertRaises(ValueError):validate_response(response,self.registration,'full_d10',request['raw'],request['instruction'],request['seed'],self.service['service_manifest_hash'],self.case)
        response=self.response(request);response['observed_calls']['verify_calls']=1
        with self.assertRaises(ValueError):validate_response(response,self.registration,'full_d10',request['raw'],request['instruction'],request['seed'],self.service['service_manifest_hash'],self.case)

    def test_video_full_decode_and_framecount_required(self):
        calls=[]
        def runner(command,**kwargs):
            calls.append(command);return SimpleNamespace(returncode=0,stdout=b'6\n',stderr=b'')
        self.assertTrue(decode_video('fixture.mp4',6,runner)['full_stream_decode_passed'])
        self.assertIn('-xerror',calls[0]);self.assertIn('-count_frames',calls[1])
        self.assertFalse(decode_video('fixture.mp4',7,runner)['full_stream_decode_passed'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'CPU ffmpeg tools unavailable')
    def test_tiny_preregistered_video_actual_full_decode(self):
        try:import imageio
        except ImportError:self.skipTest('CPU imageio unavailable')
        self.case=self.registration['cases'][0]
        result=self.run_fixture()
        self.assertEqual(result['status'],'complete',result.get('video_error'))
        self.assertTrue(result['video']['full_stream_decode_passed']);self.assertEqual(result['video']['decoded_frames'],6)


if __name__=='__main__':unittest.main()
