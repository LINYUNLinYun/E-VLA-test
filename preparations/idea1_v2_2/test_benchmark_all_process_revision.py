from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import contextmanager
import os
import types

from common import atomic_json, sha
import benchmark_all_process_revision as revision


class RevisionTests(unittest.TestCase):
    def test_same_model_phase_lease_and_selected_publish_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            identity={'required_models':['pi0']}
            with patch.object(revision,'verify_revision',return_value=identity),patch.object(revision,'_run') as execute:
                with revision.exclusive(root/'leases/pi0.benchmark.lock'):
                    with self.assertRaisesRegex(RuntimeError,'Active owner'): revision.run(root,'pi0')
                execute.assert_not_called()
            path=root/'selected/pi0.json'
            revision.publish_selection(path, {'original':True})
            with self.assertRaises(FileExistsError): revision.publish_selection(path, {'replacement':True})
            self.assertEqual(revision.read(path), {'original':True})

    def test_monitor_fault_retains_invalid_attempt_no_complete_selection(self):
        for fault in ('graphics unsupported','monitor died','pipe timeout','parent identity changed','periodic graphics'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                for name in ('condition_manifest.json','runtime_acceptance.json'):
                    atomic_json(root/name,{})
                (root/'protocols').mkdir();(root/'parity').mkdir()
                approval=root/'approval.json';atomic_json(approval,{})
                class Scanner:
                    uuid='GPU-test'
                    def __init__(self,**kwargs):pass
                    def close(self):pass
                class Binding:
                    uuid='GPU-test';pid=17;local={'pid':1}
                    def __init__(self,*args,**kwargs):pass
                    def bind(self,torch):pass
                class Monitor:
                    def __init__(self,*args):pass
                    def check(self,label):raise RuntimeError(fault)
                    def flush(self,*args,**kwargs):return dict(error=fault,events=[])
                @contextmanager
                def lease(*args,**kwargs):yield 7
                with patch.object(revision,'authorize',return_value=({'uuid':'GPU-test'},approval)),\
                     patch.object(revision,'verify_revision',return_value=dict(isolation_claim='sampled',observer_effect='possible')),\
                     patch.object(revision,'gpu_lease',lease),patch.object(revision,'Scanner',Scanner),\
                     patch.object(revision,'CausalIdentity',Binding),patch.object(revision,'Monitor',Monitor),\
                     patch.dict('sys.modules',{'torch':types.SimpleNamespace()}),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'}):
                    with self.assertRaisesRegex(RuntimeError,fault): revision._run(root,'pi0',True)
                attempt=next((root/'attempts/pi0').iterdir())
                self.assertFalse((attempt/'gate_complete.json').exists())
                self.assertTrue((attempt/'invalid_attempt.json').exists())
                self.assertFalse((root/'selected/pi0.json').exists())
    def test_per_model_phase_authority_no_global_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_json(root/'revision_manifest.json', {'kind':'new'})
            identity=dict(required_models=['pi0','xiaomi-robotic0'],revision_code_hashes={'guard':'hash'})
            receipt=dict(schema=1,accepted=True,model='xiaomi-robotic0',phase='allocation_gate',reviewer_model='gpt-6.1-sol',
                reasoning_effort='medium',evidence=['source'],revision_manifest_sha256=sha(root/'revision_manifest.json'),
                revision_code_hashes=identity['revision_code_hashes'],uuid='GPU-test')
            atomic_json(root/'benchmark_revision_acceptance.json',receipt)
            with patch.object(revision,'verify_revision',return_value=identity):
                with self.assertRaises(FileNotFoundError): revision.authorize(root,'pi0','allocation_gate')
                atomic_json(root/'acceptance/pi0.allocation_gate.json',receipt)
                with self.assertRaisesRegex(ValueError,'per-model'): revision.authorize(root,'pi0','allocation_gate')
                atomic_json(root/'acceptance/xiaomi-robotic0.allocation_gate.json',receipt)
                revision.authorize(root,'xiaomi-robotic0','allocation_gate')
                atomic_json(root/'acceptance/xiaomi-robotic0.benchmark.json',receipt)
                with self.assertRaisesRegex(ValueError,'per-model'): revision.authorize(root,'xiaomi-robotic0','benchmark')

    def test_guard_is_outside_timed_transport_and_canonical_25_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            trace=[]
            class Policy: calls=0
            policy=Policy()
            class Monitor:
                def check(self,label):
                    trace.append(label.rsplit(':',1)[-1]);return len(trace)
            def transport(**kwargs):
                trace.append('transport');policy.calls+=1
                return dict(noise_hash='noise')
            condition=dict(id='c',model='pi0',skip=[],schedule={})
            manifest=dict(conditions=[condition],observations={'pi0':[dict(index=0)]})
            def save(path,result,**kwargs):
                self.assertEqual(kwargs['calls_after'],kwargs['calls_before']+1)
                self.assertLessEqual(kwargs['all_process_guard']['started'],kwargs['all_process_guard']['ended'])
                return result
            with patch.object(revision,'read',return_value=manifest),patch.object(revision,'load_observation',return_value=({},'prompt',7)),\
                 patch.object(revision.original,'benchmark_transport',return_value=transport),patch.object(revision.original,'save_prediction',side_effect=save):
                revision.point_loop(root,'pi0',policy,Monitor(),lambda label:trace.append('seal'))
            self.assertEqual(trace, ['before','transport','after']*25+['seal'])
            self.assertEqual(policy.calls,25)


if __name__ == '__main__': unittest.main()
