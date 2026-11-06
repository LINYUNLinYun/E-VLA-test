import json, tempfile, unittest,csv
from pathlib import Path
from common import conditions, planned, digest, Ledger, atomic_json, sha, seed
from runtime_data import native_limits
from runtime_analysis import bootstrap, audit,analyze,native_control

class PipelineTests(unittest.TestCase):
    def manifest(self,extension=False):
        return dict(conditions=conditions(extension),protocol_hashes={m:digest(m) for m in ('pi0','xiaomi-robotic0','db-cogact')},observations={m:[] for m in ('pi0','xiaomi-robotic0','db-cogact')})
    def test_complete_frozen_matrix(self):
        for extension,count in ((False,38),(True,54)):
            m=self.manifest(extension); tasks=list(planned(m)); self.assertEqual(len(tasks),count*200); self.assertEqual(len({r['key'] for r in tasks}),count*200)
            self.assertEqual(len(list(planned(m,True))),count*40)
            self.assertEqual({r['init'] for r in tasks},{6,7,8,9,10})
            self.assertEqual(sum(len(c['aliases']) for c in m['conditions']),2)
    def test_seed_not_condition_dependent(self):
        self.assertEqual(seed('libero_goal',1,6,2),seed('libero_goal',1,6,2)); self.assertNotEqual(seed('libero_goal',1,6,2),seed('libero_goal',1,6,3))
    def test_reserve_no_duplicate_valid_failure(self):
        with tempfile.TemporaryDirectory() as t:
            l=Ledger(t); r=next(planned(self.manifest())); token=l.reserve(r); self.assertIsNone(l.reserve(r)); l.update(token,status='valid',success=False); self.assertIsNone(l.reserve(r))
    def test_crash_commit_recovery(self):
        with tempfile.TemporaryDirectory() as t:
            l=Ledger(t); r=next(planned(self.manifest())); token=l.reserve(r); d=Path(t)/'episode'; d.mkdir(); atomic_json(d/'validated.json',dict(valid=True,episode_key=r['key'],success=False)); l.update(token,pid=999999999,start_ticks='0',directory=str(d))
            self.assertIsNone(l.reserve(r)); self.assertEqual(l.read()['entries'][0]['status'],'valid'); self.assertFalse(l.read()['entries'][0]['success'])
    def test_invalid_is_not_silently_retried(self):
        with tempfile.TemporaryDirectory() as t:
            l=Ledger(t); r=next(planned(self.manifest())); token=l.reserve(r); l.update(token,status='invalid')
            with self.assertRaises(RuntimeError): l.reserve(r)
    def test_bootstrap_exact_constant(self):
        values={(s,t,i):.25 for s in ('libero_spatial','libero_object','libero_goal','libero_10') for t in range(10) for i in range(6,11)}
        self.assertEqual(bootstrap(values),(25.,25.)); values.pop(next(iter(values))); self.assertEqual(bootstrap(values),(None,None))
    def test_native_action_limits_parsed(self): self.assertEqual(native_limits()['limits']['libero_10'],520)
    def test_audit_rejects_empty_present_files(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t); atomic_json(root/'condition_manifest.json',self.manifest()); atomic_json(root/'episode_ledger.json',dict(entries=[]))
            result,_=audit(root,False); self.assertFalse(result['complete']); self.assertEqual(result['valid'],0); self.assertEqual(len(result['missing']),7600)
            with self.assertRaises(RuntimeError): audit(root)
    def test_analyze_entire_matrices_with_missing_data_truthfully(self):
        for extension in (False,True):
            with tempfile.TemporaryDirectory() as t:
                root=Path(t); atomic_json(root/'condition_manifest.json',self.manifest(extension)); atomic_json(root/'episode_ledger.json',dict(entries=[]))
                analyze(root)
                self.assertTrue((root/'report.md').exists()); self.assertFalse(json.loads((root/'completion_audit.json').read_text())['complete'])
                self.assertEqual(native_control(self.manifest(extension)['conditions'],'db-cogact','full')['schedule']['num_steps'],10)
                with (root/'paired_comparisons.csv').open() as stream: data=list(csv.DictReader(stream))
                e1=next(x for x in data if x['comparison']=='E1 xiaomi-robotic0_skip12_short_runs_m70 interaction')
                self.assertEqual(json.loads(e1['terms']),[['xiaomi-robotic0_skip12_short_runs_m70',1],['xiaomi-robotic0_skip12_short_runs_u50',-1],['xiaomi-robotic0_full_m70',-1],['xiaomi-robotic0_full_u50',1]])
                e2=next(x for x in data if x['comparison']=='E2 xiaomi-robotic0 interaction')
                self.assertEqual(json.loads(e2['terms']),[['xiaomi-robotic0_skip12_short_runs_u50',1],['xiaomi-robotic0_skip12_short_runs_d5',-1],['xiaomi-robotic0_full_u50',-1],['xiaomi-robotic0_full_d5',1]])
if __name__=='__main__': unittest.main()
