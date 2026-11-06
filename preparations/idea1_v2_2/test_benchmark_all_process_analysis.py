import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

from common import atomic_json,sha
import benchmark_all_process_analysis as analysis


class AnalysisTests(unittest.TestCase):
    def test_missing_or_reused_per_call_boundary_rejected(self):
        samples={1:dict(label='call:before',monotonic_end=1.),2:dict(label='call:after',monotonic_start=1.2)}
        guard=dict(before_sequence=1,after_sequence=2,started=1.01,ended=1.1)
        seen=set()
        analysis.verify_call_boundary(samples,guard,'call',seen)
        with self.assertRaisesRegex(ValueError,'reused'): analysis.verify_call_boundary(samples,guard,'call',seen)
        with self.assertRaises(KeyError): analysis.verify_call_boundary({1:samples[1]},guard,'call',set())

    def test_complete_gate_evidence_and_corruption_gap_graphics_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            revision=Path(directory)
            root=revision/'attempts/pi0/attempt'
            atomic_json(revision/'revision_manifest.json', {'kind':'new'})
            approval=revision/'acceptance/pi0.allocation_gate.json'
            atomic_json(approval,{'approved':'fixture'})
            local={'pid':7,'start_ticks':'s','pid_namespace':'ns'}
            execution=dict(model='pi0',phase='allocation_gate',revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                authorization_path=str(approval),authorization_sha256=sha(approval),uuid='GPU-test',nvml_pid=17,local=local)
            atomic_json(root/'execution_receipt.json',execution)
            labels=['empty_baseline','context_created','context_stable','challenge_allocated','challenge_released','causal_identity_bound']
            events=[dict(label=label,processes={} if i==0 else {'17':128 if i==3 else 1},graphics={},uuid='GPU-test',local=local) for i,label in enumerate(labels)]
            atomic_json(root/'binding.json',{'events':events})
            sample=lambda sequence,label,start:dict(uuid='GPU-test',processes={'17':1},graphics={},local=local,
                sequence=sequence,label=label,monotonic_start=start,monotonic_end=start+.01)
            rows=[sample(1,'monitor_start',1),sample(2,'final',1.1)]
            chunk=root/'monitor/0000_final.json'
            proof=dict(model='pi0',phase='allocation_gate',passed=True,uuid='GPU-test',revision_manifest_sha256=execution['revision_manifest_sha256'],
                execution_receipt_sha256=sha(root/'execution_receipt.json'),binding_sha256=sha(root/'binding.json'))
            def publish(values):
                atomic_json(chunk,dict(error=None,events=values,sequence=values[-1]['sequence']))
                proof['monitor_chunks']=[dict(path=str(chunk),sha256=sha(chunk))]
                atomic_json(root/'gate_complete.json',proof)
            publish(rows)
            with patch.object(analysis,'authorize',return_value=({'uuid':'GPU-test'},approval)):
                analysis.verify_attempt_events(revision,root,'pi0','allocation_gate')
                for changed in (dict(rows[1],graphics={'99':1}),dict(rows[1],monotonic_start=2,monotonic_end=2.01),dict(rows[1],sequence=3)):
                    publish([rows[0],changed])
                    with self.assertRaises((RuntimeError,ValueError)):
                        analysis.verify_attempt_events(revision,root,'pi0','allocation_gate')
                publish(rows)
                chunk.unlink()
                with self.assertRaises((FileNotFoundError,ValueError)): analysis.verify_attempt_events(revision,root,'pi0','allocation_gate')
                execution['phase']='benchmark'
                atomic_json(root/'execution_receipt.json',execution)
                proof.update(phase='benchmark',execution_receipt_sha256=sha(root/'execution_receipt.json'))
                proof['monitor_chunks']=[]
                labels=['startup']+[f'point{i:02d}' for i in range(32)]+['final']
                for position,label in enumerate(labels):
                    path=root/'monitor'/f'{position:04d}_{label}.json'
                    row=sample(position+1,'monitor_start' if position==0 else label,1+position*.1)
                    atomic_json(path,dict(error=None,events=[row],sequence=position+1))
                    proof['monitor_chunks'].append(dict(path=str(path),sha256=sha(path)))
                atomic_json(root/'revision_complete.json',proof)
                analysis.verify_attempt_events(revision,root,'pi0','benchmark')
                broken=Path(proof['monitor_chunks'][10]['path'])
                value=analysis.read(broken)
                value['events'][0].update(monotonic_start=100,monotonic_end=100.01)
                atomic_json(broken,value)
                proof['monitor_chunks'][10]['sha256']=sha(broken)
                atomic_json(root/'revision_complete.json',proof)
                with self.assertRaisesRegex(RuntimeError,'gap'):
                    analysis.verify_attempt_events(revision,root,'pi0','benchmark')

    def test_foreign_or_explicit_failed_attempt_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            revision=Path(directory)
            with self.assertRaisesRegex(ValueError,'invalid or belongs'):
                analysis.verify_attempt_events(revision, revision/'elsewhere','pi0','benchmark')
            root=revision/'attempts/pi0/attempt'
            atomic_json(root/'invalid_attempt.json',{'passed':False})
            with self.assertRaisesRegex(ValueError,'invalid or belongs'):
                analysis.verify_attempt_events(revision, root,'pi0','benchmark')


if __name__ == '__main__': unittest.main()
