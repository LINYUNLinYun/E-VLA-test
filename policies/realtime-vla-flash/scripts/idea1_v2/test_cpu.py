import tempfile
import unittest
from pathlib import Path
import numpy as np
import json
import os
import contextlib
from types import SimpleNamespace
from unittest.mock import patch
from .core import Kalman, Router, noise, write_json
from .kalman import predictions, equal_score
from .policy import Policy

def state(step,d=.2,contact=False,grasp=False,displacement=0,progress=0):
    return {'step':step,'primary_objects':['box'],'objects':{'box':{'type':'object'}},
            'eef_object_distances':{'box':d},'object_displacements':{'box':displacement},
            'robot_contacts':{'box':contact},'robot_grasps':{'box':grasp},'goal_progress':progress}

class Tests(unittest.TestCase):
    def test_discovery_cooldown_reuses_snapshot_without_hardware_query(self):
        from . import runner
        with tempfile.TemporaryDirectory() as d:
            with patch.object(runner,'NEW',Path(d)),patch.object(runner.time,'time',return_value=1000),\
                 patch.object(runner.subprocess,'run',side_effect=[SimpleNamespace(stdout='uuid_busy, 123\n'),
                                                                    SimpleNamespace(stdout='0, uuid_free, 0\n1, uuid_busy, 0\n')]) as query:
                first=runner.discover();second=runner.discover()
                self.assertEqual(first,(['0'],1030));self.assertEqual(second,first)
                self.assertEqual(query.call_count,2) # Two commands in one hardware scan.
                second[0].append('mutated')
                self.assertEqual(runner.discover(),first)
                self.assertEqual(query.call_count,2)

    def test_discovery_30_second_boundary_and_manifest(self):
        from . import runner
        from .__main__ import manifest
        with tempfile.TemporaryDirectory() as d:
            with patch.object(runner,'NEW',Path(d)),patch.object(runner.time,'time',return_value=1000) as now,\
                 patch.object(runner.subprocess,'run',side_effect=[SimpleNamespace(stdout=''),SimpleNamespace(stdout='0, uuid_free, 0\n'),
                                                                    SimpleNamespace(stdout='uuid_free, 123\n'),SimpleNamespace(stdout='0, uuid_free, 500\n')]) as query:
                self.assertEqual(runner.discover(),(['0'],1030))
                now.return_value=1029.999
                self.assertEqual(runner.discover(),(['0'],1030));self.assertEqual(query.call_count,2)
                now.return_value=1030
                self.assertEqual(runner.discover(),([],1060));self.assertEqual(query.call_count,4)
                status=json.loads((Path(d)/'status/gpu_discovery.json').read_text())
                self.assertEqual(status['interval_seconds'],30);self.assertEqual(status['next_scan_time'],1060)
            self.assertEqual(manifest(Path('/old'))['gpu_discovery_interval_seconds'],30)

    def test_discovery_then_smoke_uses_cached_idle_card(self):
        from . import runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'attempt';campaign=Path(d)/'campaign'
            write_json(root/'protocol/execution_manifest.json',{'code_hashes':{}})
            args=SimpleNamespace(output=root,source=Path('/old'),gpu='0',retry_invalid=False,fix_note='')
            hardware=[]
            def child(cmd,**kw):
                if cmd[0]=='nvidia-smi':
                    hardware.append(cmd)
                    return SimpleNamespace(stdout='' if '--query-compute-apps=gpu_uuid,pid' in cmd else '0, uuid_free, 0\n')
                return SimpleNamespace(returncode=0)
            with patch.object(runner,'NEW',campaign),patch.object(runner.time,'time',return_value=1000),\
                 patch.object(runner.subprocess,'run',side_effect=child),\
                 patch.object(runner,'lease',return_value=contextlib.nullcontext()),\
                 patch.object(runner,'server',side_effect=lambda *a:contextlib.nullcontext(8765)),\
                 patch.object(runner,'episode',return_value={'directory':'/dummy'}),patch.object(runner,'baseline_compare'):
                self.assertEqual(runner.discover()[0],['0'])
                runner.smoke(args)
                self.assertEqual(len(hardware),2)
            self.assertTrue(json.loads((root/'smoke/gates.json').read_text())['passed'])

    def test_full_matrix_analysis_and_repeat_cpu_only(self):
        from .analysis import run
        from .runner import ALT
        from .core import SUITES
        import csv
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'new';source=Path(d)/'old';directory=root/'fixture'
            directory.mkdir(parents=True)
            infer={'client_roundtrip_ms':4,'idea1_v2':{'cuda_policy_ms':1,'prepare_ms':2,'sample_actions_ms':3,'prompt_graph_rebuild':False}}
            (directory/'infer.jsonl').write_text(json.dumps(infer)+'\n')
            current={(s,t,i):dict(success=True,directory=str(directory),infer_path='infer.jsonl',actual_actions=5,full_rounds=0,full_actions=0)
                     for s in SUITES for t in range(10) for i in range(5)}
            for s in SUITES:
                for condition in ('Full18_D3_R5_FORMAL','V14_A03_D3_R5_FORMAL'):
                    write_json(source/'closed_loop'/condition/s/'run/episode_log.json',
                               [dict(task_id=t,episode_idx=i,success=True) for t in range(10) for i in range(5)])
            with patch('scripts.idea1_v2.analysis.completed',side_effect=lambda r,c:current if c==ALT else {}),\
                 patch('scripts.idea1_v2.diagnostics.run'):
                run(root,source);run(root,source)
            with (root/'analysis/success_tasks.csv').open() as stream:tasks=list(csv.DictReader(stream))
            with (root/'analysis/success_bootstrap.csv').open() as stream:bootstrap=list(csv.DictReader(stream))
            self.assertEqual(len(tasks),80);self.assertEqual(len(bootstrap),10)
            self.assertTrue(all(float(r['difference_pp_ci025'])==0 for r in bootstrap))

    def test_diagnostic_thresholds_and_unknown_grasp(self):
        from .diagnostics import detect
        baseline=[dict(step=i,eef_pos=[0,0,0]) for i in range(8)]
        states=[dict(step=i,eef_pos=[.021 if i>=2 else 0,0,0],semantic_phase='transport',
                     primary_objects=['box'],robot_grasps={'box':i<3}) for i in range(8)]
        phase,events=detect(states,baseline)
        self.assertEqual(phase,'transport')
        self.assertEqual(events[0]['event_step'],2)
        self.assertEqual(events[1]['event_step'],3)
        states[4]['robot_grasps']['box']=None
        self.assertEqual(len(detect(states,baseline)[1]),1)
        states[3]['eef_pos']=[0,0,0]
        self.assertEqual(detect(states[:5],baseline)[1],[])

    def test_incomplete_audit_does_not_claim_complete(self):
        from .audit import run
        with tempfile.TemporaryDirectory() as d:
            result=run(Path(d))
            self.assertFalse(result['experiments_complete'])
            self.assertEqual(result['E2']['recorded_valid'],0)
            self.assertEqual(len(result['E2']['missing_keys']),200)
            self.assertTrue(result['failed_tasks_are_valid'])
            self.assertNotEqual(result['E3']['status'],'deferred_by_preregistered_gate')

    def test_report_and_render_repeat_without_cuda(self):
        from .media import run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.dict(__import__('sys').modules,{'torch':None,'libero':None}):
                run(root,'render');run(root,'render')
            report=(root/'report.md').read_text()
            self.assertEqual(report.count('## '),4)
            self.assertIn('E2尚无有效闭环结果',report)
            self.assertTrue((root/'media/gallery.html').exists())

    def test_event_clips_cpu_encoding_and_provenance_resume(self):
        import shutil
        import subprocess
        from .diagnostics import clips
        if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):self.skipTest('ffmpeg unavailable')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);video=root/'source.mp4'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','color=c=blue:s=32x32:r=20',
                            '-frames:v','8','-c:v','libx264','-pix_fmt','yuv420p',str(video)],check=True)
            case={'condition':'test','suite':'libero_object','task':0,'init':0}
            write_json(root/'media/cases.json',{'cases':[case]})
            write_json(root/'analysis/failure_diagnostics.json',{'records':[dict(case,event_step=4,event='grasp_loss_candidate',full_video=str(video),new_video=str(video))]})
            result=clips(root);self.assertEqual(len(result),2)
            self.assertEqual(result[0]['start_step'],0);self.assertEqual(result[0]['end_step'],7)
            self.assertEqual(clips(root),result)

    def test_fixed_cycle_reset(self):
        r=Router();self.assertEqual([r.choose()[0] for _ in range(7)],['V14','V14','Full','V14','V14','Full','V14'])
        r.reset();self.assertEqual(r.choose()[0],'V14')

    def test_phase_history_boundaries_latch(self):
        r=Router(True)
        self.assertEqual(r.choose([state(0)])[0],'Full')
        self.assertEqual(r.choose([state(1),state(2)])[0],'V14')
        self.assertEqual(r.choose([state(3,d=.08)])[0],'Full')
        self.assertEqual(r.choose([state(4),state(5),state(6)])[0],'V14')
        self.assertEqual(r.choose([state(7,grasp=True)])[1],'precision_latch')
        self.assertEqual(r.choose([state(8)])[0],'Full')
        r.reset();self.assertEqual(r.choose([{'step':0}])[1],'unknown')

    def test_recent_contact_and_missing_history(self):
        r=Router(True)
        self.assertEqual(r.choose([state(0),state(1),state(2,contact=True)])[0],'Full')
        self.assertEqual(r.choose([state(i) for i in range(3,8)])[0],'Full')
        self.assertEqual(r.choose([state(8)])[0],'V14')
        with self.assertRaises(ValueError):r.choose([state(10)])

    def test_kalman_copy_covariance_analytic(self):
        k=Kalman(1e-4,1e-3)
        for _ in range(20):k.observe([.2,-.1,.3,0,0,0,0])
        x,P=k.x.copy(),k.P.copy();mean,cov=k.forecast()
        self.assertTrue(np.array_equal(x,k.x));self.assertTrue(np.array_equal(P,k.P))
        self.assertTrue(np.allclose(mean[49],(np.linalg.matrix_power(k.F,50)@x)[:3]))
        self.assertTrue(np.allclose(cov,cov.transpose(0,2,1)))
        self.assertGreaterEqual(np.linalg.eigvalsh(k.P).min(),-1e-10)

    def test_future_leakage_and_short_prefix(self):
        actions=np.tile([.2,0,0,0,0,0,0],(7,1))
        records=[{'environment_action_step':0,'_actions':actions},{'environment_action_step':5,'_actions':actions}]
        first=list(predictions(records,1e-4,1e-3))
        actions[5:,:3]=.9
        second=list(predictions(records,1e-4,1e-3))
        self.assertTrue(np.array_equal(first[0][1],second[0][1]))
        self.assertTrue(np.array_equal(first[1][1],second[1][1]))
        self.assertTrue(np.array_equal(first[0][1],np.zeros((50,3))))

    def test_noise_identity_and_exclusive(self):
        self.assertTrue(np.array_equal(noise(0,0),noise(0,0)))
        self.assertFalse(np.array_equal(noise(0,0),noise(0,1)))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'f.json';write_json(p,{},exclusive=True)
            with self.assertRaises(FileExistsError):write_json(p,{},exclusive=True)

    def test_policy_duplicate_and_precision(self):
        from .core import WEIGHT
        class Backend:
            weight_hash=WEIGHT
            calls=0
            length=3
            def prepare(self,*a):pass
            def infer(self,skip,steps):
                self.calls+=1;self.skip=skip
                return np.zeros((50,32)),np.zeros((50,7)),{'cuda_policy_ms':1}
        with tempfile.TemporaryDirectory() as d:
            backend=Backend();p=Policy(backend,'libero_object','test',Path(d))
            obs={'__trace_task_id__':0,'__trace_episode_idx__':0,'__trace_infer_id__':0,'__executed_steps__':0,
                 'observation/image':np.zeros((2,2,3),np.uint8),'observation/wrist_image':np.zeros((2,2,3),np.uint8),
                 'observation/state':np.asarray([.123456789012345],np.float64),'prompt':'test'}
            a=p.infer(obs);a['actions'][0,0]=10
            self.assertEqual(p.infer(obs)['actions'][0,0],0);self.assertEqual(backend.calls,1)
            with np.load(next(Path(d).rglob('*.npz'))) as saved:self.assertEqual(saved['robot_state'].dtype,np.float64)
            for r in (1,2):obs['__trace_infer_id__']=r;p.infer(obs)
            self.assertEqual(backend.skip,())
            obs['observation/state'][0]=.2
            with self.assertRaises(ValueError):p.infer(obs)

    def test_global_step_not_previous_chunk_length(self):
        from .core import WEIGHT
        class Backend:
            weight_hash=WEIGHT
            def prepare(self,*a):pass
            def infer(self,*a):return np.zeros((50,32)),np.zeros((50,7)),{'cuda_policy_ms':1}
        with tempfile.TemporaryDirectory() as d:
            p=Policy(Backend(),'libero_object','test',Path(d))
            obs={'__trace_task_id__':0,'__trace_episode_idx__':0,'__trace_infer_id__':0,'__executed_steps__':5,
                 '__v2_global_step__':0,'observation/image':np.zeros((1,1,3),np.uint8),
                 'observation/wrist_image':np.zeros((1,1,3),np.uint8),'observation/state':np.zeros(8),'prompt':'test'}
            p.infer(obs)
            obs.update(__trace_infer_id__=1,__v2_global_step__=5);p.infer(obs)
            obs.update(__trace_infer_id__=2,__v2_global_step__=10)
            self.assertEqual(p.infer(obs)['policy_timing']['frame_idx'],10)

    def test_ledger_budget_invalid_retry_and_valid_immutability(self):
        from .runner import Ledger
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json')
            token=ledger.reserve('key','E2',280,56)
            with self.assertRaises(RuntimeError):ledger.reserve('key','E2',280,56)
            ledger.finish(token,'invalid')
            with self.assertRaises(RuntimeError):ledger.reserve('key','E2',280,56,retry=True)
            second=ledger.reserve('key','E2',280,56,retry=True,fix_note='fixed transport')
            ledger.finish(second,'valid',10,2)
            with self.assertRaises(RuntimeError):ledger.reserve('key','E2',280,56,retry=True,fix_note='again')
            with self.assertRaises(RuntimeError):ledger.reserve('huge','E2',66000,13201)

    def test_crash_reconcile_and_durable_validation(self):
        from .runner import Ledger
        with tempfile.TemporaryDirectory() as d:
            ledger=Ledger(Path(d)/'ledger.json')
            token=ledger.reserve('key','E2',280,56)
            ledger.annotate(token,pid=999999999)
            self.assertEqual(ledger.reconcile('owner crashed'),[token])
            self.assertEqual(json.loads(ledger.path.read_text())['entries'][0]['predictions'],56)
            token=ledger.reserve('key','E2',280,56,retry=True,fix_note='repair')
            output=Path(d)/'episode'
            write_json(output/'validated.json',{'valid':True,'directory':str(output),'actual_actions':3,'actual_predictions':1,'success':False})
            ledger.annotate(token,pid=999999999,directory=str(output))
            ledger.reconcile('durable artifact recovered')
            self.assertEqual(json.loads(ledger.path.read_text())['entries'][-1]['status'],'valid')

    def test_worker_commands_and_orchestration_environment(self):
        from .runner import env,command
        args=SimpleNamespace(source=Path('/old'),output=Path('/new'),retry_invalid=True,fix_note='fixed')
        self.assertEqual(env('2')['IDEA1_V2_ORCHESTRATED'],'1')
        cmd=command(args,'worker',suite='libero_object',gpu='2')
        self.assertIn('--launch',cmd);self.assertIn('--retry-invalid',cmd);self.assertIn('--fix-note',cmd)

    def test_unknown_grasp_latches_and_future_states_rejected(self):
        r=Router(True)
        unknown=state(0,grasp=True);unknown['eef_object_distances']={'box':None}
        self.assertEqual(r.choose([unknown])[1],'unknown')
        self.assertEqual(r.choose([state(1),state(2)])[1],'precision_latch')

    def test_smoke_subprocess_sequence_without_gpu(self):
        from . import runner
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write_json(root/'protocol/execution_manifest.json',{'code_hashes':{}})
            args=SimpleNamespace(output=root,source=Path('/old'),gpu='0',retry_invalid=False,fix_note='')
            calls=[]
            def episode(*a):
                calls.append((a[3],a[4],a[5],a[6],a[7],a[8]));return {'directory':'/dummy'}
            with patch.object(runner,'discover',return_value=(['0'],999)),patch.object(runner,'lease',return_value=contextlib.nullcontext()),\
                 patch.object(runner,'server',side_effect=lambda *a:contextlib.nullcontext(8765)),patch.object(runner,'episode',side_effect=episode),\
                 patch.object(runner,'baseline_compare'),patch.object(runner.subprocess,'run') as child:
                runner.smoke(args)
            self.assertEqual(len(calls),12)
            self.assertEqual(sum(c[-1]=='baseline' for c in calls),8)
            self.assertEqual(sum(c[-1]=='E2' for c in calls),4)
            self.assertIn('parity',child.call_args.args[0])
            self.assertTrue(json.loads((root/'smoke/gates.json').read_text())['passed'])

    def test_launch_gates_no_subprocess(self):
        from . import runner
        from .__main__ import manifest
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);args=SimpleNamespace(output=root,source=Path('/old'),mode='alternating')
            write_json(root/'protocol/execution_manifest.json',manifest(args.source))
            frozen=manifest(args.source)
            with patch('scripts.idea1_v2.__main__.manifest',return_value=frozen),patch.object(runner.subprocess,'Popen') as child:
                with self.assertRaises(FileNotFoundError):runner.launch(args)
                child.assert_not_called()
            write_json(root/'smoke/gates.json',{'passed':True,'code_hashes':manifest(args.source)['code_hashes']})
            args.mode='phase'
            with patch('scripts.idea1_v2.__main__.manifest',return_value=frozen),patch.object(runner.subprocess,'Popen') as child:
                runner.launch(args);child.assert_not_called()
            self.assertEqual(json.loads((root/'E3_deferred.json').read_text())['status'],'deferred_by_preregistered_gate')

    def test_worker_frozen_task_init_order_without_environment(self):
        from . import runner
        with tempfile.TemporaryDirectory() as d:
            args=SimpleNamespace(output=Path(d),source=Path('/old'),mode='alternating',suite='libero_object',gpu='0',retry_invalid=False)
            calls=[]
            with patch.object(runner,'lease',return_value=contextlib.nullcontext()),\
                 patch.object(runner,'server',return_value=contextlib.nullcontext(8765)),\
                 patch.object(runner,'episode',side_effect=lambda *a:calls.append((a[4],a[5]))):
                runner.worker(args)
            self.assertEqual(calls,[(t,i) for t in range(10) for i in range(5)])

    def test_measured_xyz_media_with_synthetic_fixture(self):
        from .trajectories import render
        from .runner import ALT
        from .core import SUITES
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/'source';root=Path(d)/'output'
            for suite in SUITES:
                for condition in ('Full18_D3_R5_FORMAL','V14_A03_D3_R5_FORMAL'):
                    directory=source/'closed_loop'/condition/suite/'run'
                    episodes=[]
                    if suite=='libero_object':
                        episodes=[{'task_id':0,'episode_idx':0,'success':condition=='Full18_D3_R5_FORMAL',
                                   'telemetry_path':'telemetry.jsonl','video_path':'full.mp4'}]
                        directory.mkdir(parents=True,exist_ok=True)
                        with (directory/'telemetry.jsonl').open('w') as stream:
                            for t in range(3):stream.write(json.dumps({'eef_pos':[t*.01,0,0]})+'\n')
                    write_json(directory/'episode_log.json',episodes)
            directory=root/'closed_loop'/ALT/'libero_object/task00/init00/attempt00'
            result={'task_suite_name':'libero_object','task_id':0,'episode_idx':0,'success':True,
                    'valid':True,
                    'directory':str(directory),'telemetry_path':'telemetry.jsonl','infer_path':'infer.jsonl','video_path':'new.mp4'}
            write_json(directory/'validated.json',result)
            with (directory/'telemetry.jsonl').open('w') as stream:
                for t in range(3):stream.write(json.dumps({'eef_pos':[t*.01,.01,0]})+'\n')
            with (directory/'infer.jsonl').open('w') as stream:stream.write(json.dumps({'frame_idx':1,'idea1_v2':{'model':'Full'}})+'\n')
            result=render(root,source)
            self.assertEqual(result['selected_cases'],1)
            self.assertTrue(next((root/'figures').rglob('*.png')).is_file())
            cases=json.loads((root/'media/cases.json').read_text())['cases']
            self.assertEqual(cases[0]['rule'],'v14_failure_new_success_lexicographic')

if __name__=='__main__':unittest.main()
