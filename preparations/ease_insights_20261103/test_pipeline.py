"""CPU-only supervisor fixtures: no real prepares, scheduler, model, or GPU calls."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from preparations.ease_insights_20261103.pipeline import (
    FLASH_PYTHON,MODULE,Pipeline,Paths,SchedulerWatch,append_jobs,marker_status,
    proc_identity,sha,write)

SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
VLM=['full_d10','fixed_v12_jacobian','fixed_v12_action_mse','fixed_v12_cosine']

class NoResourceWatch:
    def observe(self,state):return dict(status='live',process=dict(pid=999,argv=[]))
    def ensure(self,state,queue,job_state,allow_restart=True):return dict(status='live',pid=999)

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='ease-pipeline-fixture-')
        self.root=Path(self.temp.name)
        self.paths=Paths.defaults(self.root,draft_plan=self.root/'draft_plan',reports=self.root/'reports')
        write(self.paths.queue,dict(jobs=[],phase='fixture',created=17))
        write(self.root/'draft_discovery_manifest.json',dict(jobs=['discovery_full_libero_spatial'],source_suites=SUITES))
        write(self.root/'vlm/manifest.json',dict(manifest_hash='frozen-vlm',conditions=[dict(id=c) for c in VLM]))
        write(self.paths.scheduler_output/'jobs_state.json',{})
    def tearDown(self):self.temp.cleanup()
    def engine(self,runner=None):return Pipeline(self.paths,command_runner=runner,scheduler_watch=NoResourceWatch())
    def complete(self,engine,name):
        for path,kind,expected in engine.discovery_specs(name):
            write(path,dict(status='complete',**expected) if kind=='draft' else dict(complete=True,**expected))
    def job(self,name):return dict(id=name,argv=['python3','fixture.py','--id',name],complete_file=str(self.root/(name+'.done')),depends_on=[])
    def test_append_is_idempotent_preserves_existing_and_top_level(self):
        a,b=self.job('a'),self.job('b')
        write(self.paths.queue,dict(jobs=[a],phase='untouched',created=17))
        first=append_jobs(self.paths.queue,[b]);second=append_jobs(self.paths.queue,[b])
        q=json.loads(self.paths.queue.read_text())
        self.assertEqual(q,dict(jobs=[a,b],phase='untouched',created=17))
        self.assertEqual(first['added_ids'],['b']);self.assertEqual(second['added_ids'],[])
        changed=copy.deepcopy(b);changed['argv'].append('different')
        with self.assertRaises(ValueError):append_jobs(self.paths.queue,[changed])
        self.assertEqual(json.loads(self.paths.queue.read_text()),q)
    def test_duplicate_completion_target_refused(self):
        a=self.job('a');write(self.paths.queue,dict(jobs=[a]))
        b=dict(a,id='another-id')
        with self.assertRaises(ValueError):append_jobs(self.paths.queue,[b])
    def test_incomplete_never_freezes(self):
        calls=[];engine=self.engine(lambda argv,log:calls.append(argv))
        status=engine.tick()
        self.assertEqual(calls,[])
        self.assertEqual(status['branches'],dict(draft='waiting_discovery',vlm='waiting_discovery'))
    def test_none_stops_vlm_branch_without_confirmation(self):
        calls=[]
        def runner(argv,log):
            calls.append(argv)
            out=Path(argv[argv.index('--output')+1])
            write(out/'selection.json',dict(complete_discovery_episodes=160,selection=dict(status='none',selected=None)))
            return 0
        engine=self.engine(runner);self.complete(engine,'vlm')
        engine.tick();engine.tick()
        self.assertEqual(engine.state['branches']['vlm']['phase'],'needs_scientific_review')
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],str(FLASH_PYTHON))
        self.assertTrue(calls[0][2].endswith('.analyze'))
        self.assertEqual(json.loads(self.paths.queue.read_text())['jobs'],[])
    def test_historical_failed_job_outside_queue_is_ignored(self):
        write(self.paths.scheduler_output/'jobs_state.json',{'removed_old_vlm_job':dict(status='failed')})
        engine=self.engine(lambda argv,log:0);status=engine.tick()
        self.assertEqual(status['branches']['vlm'],'waiting_discovery')
    def test_current_failure_requires_attention_and_no_retry(self):
        engine=self.engine();spec=engine.discovery_specs('vlm')[0]
        job=dict(self.job('current'),complete_file=str(spec[0]))
        write(self.paths.queue,dict(jobs=[job]));write(self.paths.scheduler_output/'jobs_state.json',{'current':dict(status='failed')})
        engine.tick();engine.tick()
        self.assertEqual(engine.state['branches']['vlm']['phase'],'needs_attention')
    def test_vlm_prepare_builds_four_task_shards(self):
        calls=[];selection=self.root/'vlm/analysis/selection.json';write(selection,dict(selection_hash='chosen'))
        def runner(argv,log):
            calls.append(argv)
            out=Path(argv[argv.index('--output')+1])
            write(out/'manifest.json',dict(total_confirmation_episodes=600,cases_per_condition=200,confirmation_inits=[31,32,33,34,35],
                  planned_conditions=['full_d10','fixed_v12_jacobian','fixed_v12_cosine'],manifest_hash='confirmation-frozen'))
            return 0
        engine=self.engine(runner)
        engine.state['branches']['draft']['phase']='needs_scientific_review'
        engine.state['branches']['vlm'].update(phase='prepare_confirmation',selection=str(selection),selection_hash='chosen',selection_sha256=sha(selection))
        engine.tick();jobs=engine.state['branches']['vlm']['jobs']
        self.assertEqual(len(jobs),12);self.assertEqual(sum(j['expected_episodes'] for j in jobs),600)
        for j in jobs:
            self.assertEqual(j['argv'][0],str(FLASH_PYTHON))
            self.assertTrue(j['argv'][2].endswith('.confirm_runner'))
            self.assertEqual(j['argv'][j['argv'].index('--shards')+1],'4')
            self.assertIn('of04_complete.json',j['complete_file'])
        engine.tick();engine.tick()
        self.assertEqual(len(json.loads(self.paths.queue.read_text())['jobs']),12)
        self.assertEqual(len(calls),1)
    def test_draft_none_adopts_valid_frozen_output_without_reprepare(self):
        engine=self.engine(lambda *args:self.fail('Must not rerun preparation'))
        self.complete(engine,'draft')
        plan=self.paths.draft_plan;write(plan/'discovery_audit.json',dict(status='complete'))
        write(plan/'selection.json',dict(status='none',discovery_conditions=360,
            discovery_manifest_sha256=sha(self.root/'draft_discovery_manifest.json'),discovery_audit_sha256=sha(plan/'discovery_audit.json')))
        write(plan/'jobs_draft.json',dict(jobs=[],expected_episodes=0,selection_sha256=sha(plan/'selection.json')))
        engine.tick()
        self.assertEqual(engine.state['branches']['draft']['phase'],'needs_scientific_review')

    def test_enqueue_requires_previously_stored_selection_sha(self):
        engine=self.engine();selection=self.root/'chosen.json';write(selection,dict(choice='frozen'))
        engine.state['branches']['draft']['phase']='needs_scientific_review'
        engine.state['branches']['vlm'].update(phase='enqueue_confirmation',selection=str(selection),jobs=[self.job('confirmation')])
        engine.tick()
        self.assertEqual(engine.state['branches']['vlm']['phase'],'needs_attention')
        self.assertEqual(json.loads(self.paths.queue.read_text())['jobs'],[])
    def test_enqueue_rejects_changed_vlm_selection(self):
        engine=self.engine();selection=self.root/'chosen.json';write(selection,dict(choice='frozen'))
        original_sha=sha(selection)
        engine.state['branches']['draft']['phase']='needs_scientific_review'
        engine.state['branches']['vlm'].update(phase='enqueue_confirmation',selection=str(selection),selection_sha256=original_sha,jobs=[self.job('confirmation')])
        write(selection,dict(choice='changed'))
        engine.tick()
        self.assertEqual(engine.state['branches']['vlm']['phase'],'needs_attention')
        self.assertEqual(json.loads(self.paths.queue.read_text())['jobs'],[])
    def test_vlm_freeze_records_file_sha(self):
        def runner(argv,log):
            out=Path(argv[argv.index('--output')+1])
            write(out/'selection.json',dict(complete_discovery_episodes=160,selection_hash='logical-selection-hash',selection=dict(status='selected',selected=dict(pair=['fixed_v12_jacobian','fixed_v12_cosine']))))
            return 0
        engine=self.engine(runner);self.complete(engine,'vlm');engine.tick()
        record=engine.state['branches']['vlm']
        self.assertEqual(record['phase'],'prepare_confirmation')
        self.assertEqual(record['selection_sha256'],sha(record['selection']))
        self.assertNotEqual(record['selection_sha256'],record['selection_hash'])
    def test_complete_draft_counts_do_not_hide_initial_pairing_mismatch(self):
        def runner(argv,log):
            out=Path(argv[argv.index('--output')+1])
            if Path(argv[1]).name=='analyze.py':
                write(out/'audit.json',dict(complete_conditions=400,uncompleted=[],initial_pairing_mismatches=[dict(case='mismatch')]))
            elif Path(argv[1]).name=='heldout_statistics.py':
                write(out/'statistics.json',dict(status='complete'))
            return 0
        engine=self.engine(runner)
        engine.state['branches']['vlm']['phase']='needs_scientific_review'
        engine.state['branches']['draft'].update(phase='analyze_confirmation',selection=str(self.root/'selection.json'),expected_confirmation_episodes=400)
        engine.tick()
        record=engine.state['branches']['draft']
        self.assertEqual(record['phase'],'needs_attention')
        self.assertIn('pairing audit',record['error'])
        self.assertFalse((self.paths.output/'draft_report_ready.json').exists())
        self.assertTrue((self.paths.reports/'draft_confirmation_analysis/audit.json').exists())

class SchedulerFixture(Fixture):
    def live(self,pid=42):
        return dict(pid=pid,status='live',start_ticks='12345',argv=['python3','-m',MODULE,
            '--queue',str(self.paths.queue),'--output',str(self.paths.scheduler_output),'--max-gpus','2','--idle-seconds','0'])
    def test_live_scheduler_never_restarted_even_with_stale_heartbeat(self):
        write(self.root/'continuation.json',dict(scheduler_pid=42,time=0))
        watch=SchedulerWatch(self.paths,42,probe=lambda pid:self.live(pid),spawn=lambda *args:self.fail('Live scheduler restart forbidden'))
        queue=dict(jobs=[self.job('pending')]);state={}
        result=watch.ensure(state,queue,{})
        self.assertEqual(result['status'],'live');self.assertEqual(state['pid'],42)
    def test_uncertain_observation_never_restarted(self):
        watch=SchedulerWatch(self.paths,42,probe=lambda pid:dict(pid=pid,status='unknown'),spawn=lambda *args:self.fail('Uncertain scheduler restart forbidden'))
        result=watch.ensure({},dict(jobs=[self.job('pending')]),{})
        self.assertEqual(result['status'],'needs_attention')
    def test_dead_scheduler_restarted_once_with_exact_resource_options(self):
        spawned=[]
        class Process:pid=999
        watch=SchedulerWatch(self.paths,42,probe=lambda pid:dict(pid=pid,status='absent'),spawn=lambda argv,log:(spawned.append(argv) or Process()))
        state={};queue=dict(jobs=[self.job('pending')])
        result=watch.ensure(state,queue,{})
        self.assertEqual(result['status'],'restarted');self.assertEqual(len(spawned),1)
        self.assertEqual(spawned[0][spawned[0].index('--max-gpus')+1],'2')
        self.assertEqual(spawned[0][spawned[0].index('--idle-seconds')+1],'0')
        second=watch.ensure(state,queue,{})
        self.assertEqual(second['status'],'needs_attention');self.assertEqual(len(spawned),1)
    def test_live_orphan_job_prevents_scheduler_restart(self):
        watch=SchedulerWatch(self.paths,42,probe=lambda pid:(dict(pid=pid,status='absent') if pid==42 else dict(pid=pid,status='live',argv=['worker'])),spawn=lambda *args:self.fail('Orphan job must block restart'))
        queue=dict(jobs=[self.job('running'),self.job('pending')])
        result=watch.ensure({},queue,{'running':dict(status='running',pid=55)})
        self.assertEqual(result['status'],'needs_attention')
    def test_proc_identity_reads_cmdline_and_start_ticks(self):
        proc=self.root/'proc/123';proc.mkdir(parents=True)
        fields=['S']+['0']*18+['777']+['0']*5
        (proc/'stat').write_text('123 (python3) '+' '.join(fields))
        (proc/'cmdline').write_bytes(b'python3\0-m\0module\0')
        result=proc_identity(123,self.root/'proc')
        self.assertEqual(result['status'],'live');self.assertEqual(result['start_ticks'],'777')
        self.assertEqual(result['argv'],['python3','-m','module'])

if __name__=='__main__':unittest.main()
