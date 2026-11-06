from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from common import atomic_json, sha
import benchmark_light_revision as revision
import benchmark_light_analysis as analysis


class LightTests(unittest.TestCase):
    def test_exact_mixed_scope_and_no_db_execution(self):
        counts = {'pi0': 16, 'xiaomi-robotic0': 18, 'db-cogact': 20}
        self.assertEqual(sum(counts[m]*32*sum(revision.POLICY[m]) for m in counts), 22528)
        self.assertEqual({m:32*revision.POLICY[m][1] for m in counts},
                         {'pi0':160, 'xiaomi-robotic0':160, 'db-cogact':640})
        with patch.object(revision, 'verify_revision', return_value={}):
            with self.assertRaises(ValueError): revision.run('/unused', 'db-cogact')
            with self.assertRaises(ValueError): revision.authorize('/unused', 'db-cogact', 'benchmark')

    def test_six_independent_calls_one_warmup_guard_outside(self):
        for model in ('pi0', 'xiaomi-robotic0'):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); trace=[]; phases=[]
                class Policy: calls=0
                policy=Policy()
                class Monitor:
                    def check(self,label): trace.append(label.rsplit(':',1)[-1]); return len(trace)
                def transport(**kwargs):
                    trace.append('transport'); phases.append(kwargs['category']); policy.calls+=1
                    return dict(noise_hash='same')
                def save(path,result,**kwargs):
                    self.assertEqual(kwargs['calls_after'], kwargs['calls_before']+1)
                    self.assertEqual(kwargs['phase'], phases[-1]); return result
                manifest=dict(conditions=[dict(id='c',model=model,skip=[],schedule={})],
                              observations={model:[dict(index=0)]})
                with patch.object(revision,'read',return_value=manifest), \
                     patch.object(revision,'load_observation',return_value=({},'prompt',7)), \
                     patch.object(revision.original,'benchmark_transport',return_value=transport), \
                     patch.object(revision.original,'save_prediction',side_effect=save):
                    revision.point_loop(root,model,policy,Monitor(),lambda label:trace.append('seal'))
                self.assertEqual(trace,['before','transport','after']*6+['seal'])
                self.assertEqual(phases,['warmup']+['measurement']*5)

    def test_db_inheritance_binds_selection_completion_and_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent=Path(tmp); origin=parent/'benchmark_revisions/all_process_v2'
            attempt=origin/'attempts/db-cogact/token'
            atomic_json(origin/'revision_manifest.json',{})
            atomic_json(attempt/'revision_complete.json',dict(passed=True))
            selection=dict(model='db-cogact',attempt=str(attempt),
                revision_manifest_sha256=sha(origin/'revision_manifest.json'),
                completion_sha256=sha(attempt/'revision_complete.json'))
            atomic_json(origin/'selected/db-cogact.json',selection)
            with patch('benchmark_all_process_revision.verify_revision',return_value=dict(source_attempt=str(parent))):
                bound=revision.inherited_db(parent)
                self.assertEqual(bound['attempt'],str(attempt))
                atomic_json(attempt/'revision_complete.json',dict(passed=False))
                with self.assertRaises(ValueError): revision.inherited_db(parent)
                atomic_json(attempt/'revision_complete.json',dict(passed=True))
                selection['model']='pi0'; atomic_json(origin/'selected/db-cogact.json',selection)
                with self.assertRaises(ValueError): revision.inherited_db(parent)

    def test_analysis_rejects_missing_sample_wrong_phase_and_nonminimal(self):
        with tempfile.TemporaryDirectory() as tmp:
            revision_root=Path(tmp); root=revision_root/'attempts/pi0/token'
            condition=dict(id='c',model='pi0'); points=[dict(seed=7,content_hash='input') for _ in range(32)]
            manifest=dict(conditions=[condition],observations={'pi0':points})
            atomic_json(revision_root/'condition_manifest.json',manifest)
            atomic_json(revision_root/'revision_manifest.json',{})
            directory=root/'benchmark/pi0'; atomic_json(directory/'complete.json',dict(conditions=1,points=32,warmup=1,measurement=5,exclusive=True))
            atomic_json(root/'revision_complete.json',{})
            atomic_json(revision_root/'selected/pi0.json',dict(model='pi0',attempt=str(root),
                revision_manifest_sha256=sha(revision_root/'revision_manifest.json'),completion_sha256=sha(root/'revision_complete.json')))
            samples={}; sequence=0
            for point in range(32):
                for repeat in range(6):
                    name=f'p{point:02d}_r{repeat:02d}_c'; sequence+=1; pre=sequence
                    samples[sequence]=dict(label=name+':before',monotonic_end=0)
                    sequence+=1; samples[sequence]=dict(label=name+':after',monotonic_start=3)
                    row=dict(condition='c',point=point,repeat=repeat,independent_forward=True,input_hash='input',noise_hash='noise',
                        phase='warmup' if repeat==0 else 'measurement',calls_before=0,calls_after=1,
                        audit=dict(instrumentation_mode='minimal'),timing=dict(measurement_mode='minimal_action_generation',shared_condition=False,
                        action_generation_ms=1,server_total_ms=2,client_roundtrip_ms=3),noise_path=str(root/'native/pi0/noise'),server_prediction_path=str(root/'native/pi0/raw'),
                        all_process_guard=dict(before_sequence=pre,after_sequence=sequence,started=1,ended=2))
                    atomic_json(directory/'samples'/f'{name}.json',row)
            proof=dict(scientific_complete_sha256=sha(directory/'complete.json'))
            identity=dict(required_models=['pi0'])
            with patch.object(analysis,'verify_revision',return_value=identity), \
                 patch.object(analysis,'verify_attempt_events',return_value=({},proof,samples)), \
                 patch.object(analysis,'authorize',return_value=({'gate_receipt':str(root/'gate_complete.json')},None)), \
                 patch('runtime_client.check_prediction'),patch('runtime_data.audited_protocol',return_value={}), \
                 patch.object(analysis,'TOTAL_CALLS',192):
                _,result=analysis.verify_benchmark(revision_root)
                self.assertEqual(result['pi0']['measurement_denominator_per_condition'],160)
                target=directory/'samples/p00_r01_c.json'; original=analysis.read(target)
                for field,value in [('phase','warmup'),('audit',dict(instrumentation_mode='full'))]:
                    atomic_json(target,{**original,field:value})
                    with self.assertRaises(ValueError): analysis.verify_benchmark(revision_root)
                atomic_json(target,original); target.unlink()
                with self.assertRaises(ValueError): analysis.verify_benchmark(revision_root)


if __name__=='__main__': unittest.main()
