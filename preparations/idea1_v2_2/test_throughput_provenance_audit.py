"""Real-byte receipt history and ledger/session provenance boundary tests."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from common import atomic_json, planned, sha
import throughput_provenance_audit as auditor
import final_provenance_analysis as final_analysis


def signatures(revision, model='pi0'):
    revision = Path(revision)
    revision_hash = sha(revision/'revision_manifest.json')
    for phase in ('source', 'gate', 'pilot'):
        receipt = dict(schema=1, accepted=True, reviewer_model='gpt-6.1-sol', reasoning_effort='medium',
            model=model, phase=phase, revision_manifest_sha256=revision_hash, evidence=['review'])
        if phase != 'source':
            receipt['phase_receipt_sha256'] = sha(revision/f'{model}.{phase}_ready.json')
        if phase == 'pilot':
            receipt['evidence'] = ['../../../episodes/pi0_full/libero_spatial']
            atomic_json(revision/'acceptance'/f'{model}.pilot.initial_signature.json', receipt)
            receipt['evidence'] = ['../../episodes/pi0_full/libero_spatial']
        atomic_json(revision/'acceptance'/f'{model}.{phase}.json', receipt)
    directory = revision/'acceptance'
    correction = dict(schema=1, kind='display_evidence_path_correction', reviewer_model='gpt-6.1-sol', reasoning_effort='medium',
        scientific_acceptance_unchanged=True, original_receipt=f'{model}.pilot.initial_signature.json',
        corrected_receipt=f'{model}.pilot.json', original_receipt_sha256=sha(directory/f'{model}.pilot.initial_signature.json'),
        corrected_receipt_sha256=sha(directory/f'{model}.pilot.json'),
        phase_receipt_sha256=sha(revision/f'{model}.pilot_ready.json'),
        change={'from': '../../../episodes/pi0_full/libero_spatial', 'to': '../../episodes/pi0_full/libero_spatial'})
    atomic_json(directory/f'{model}.pilot.evidence_path_correction.json', correction)
    return correction


class ReceiptHistoryTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        atomic_json(root/'revision_manifest.json', dict(revision=1))
        atomic_json(root/'pi0.gate_ready.json', dict(exact=True))
        atomic_json(root/'pi0.pilot_ready.json', dict(pilot=True))
        correction = signatures(root)
        return root, correction

    def test_initial_and_current_resolve_to_actual_preserved_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, correction = self.fixture(directory)
            index = auditor.ReceiptIndex(root, {'pi0'})
            for field in ('original_receipt_sha256', 'corrected_receipt_sha256'):
                self.assertEqual(index.resolve(correction[field], 'pi0', 'pilot')['phase'], 'pilot')
            self.assertTrue(index.equivalent(correction['original_receipt_sha256'], correction['corrected_receipt_sha256'], 'pi0', 'pilot'))
            with self.assertRaises(ValueError): index.resolve('unknown', 'pi0', 'pilot')
            with self.assertRaises(ValueError): index.resolve(correction['original_receipt_sha256'], 'db-cogact', 'pilot')
            with self.assertRaises(ValueError): index.resolve(correction['original_receipt_sha256'], 'pi0', 'source')

    def test_preserved_original_tamper_rejected_even_after_indexing(self):
        with tempfile.TemporaryDirectory() as directory:
            root, correction = self.fixture(directory)
            index = auditor.ReceiptIndex(root, {'pi0'})
            old = root/'acceptance'/correction['original_receipt']
            atomic_json(old, dict(forged=True))
            with self.assertRaises(ValueError): index.resolve(correction['original_receipt_sha256'], 'pi0', 'pilot')
            with self.assertRaises(ValueError): auditor.ReceiptIndex(root, {'pi0'})

    def test_ready_drift_and_undeclared_scientific_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, _ = self.fixture(directory)
            atomic_json(root/'pi0.pilot_ready.json', dict(pilot='changed'))
            with self.assertRaises(ValueError): auditor.ReceiptIndex(root, {'pi0'})
        with tempfile.TemporaryDirectory() as directory:
            root, correction = self.fixture(directory)
            current = root/'acceptance'/'pi0.pilot.json'
            forged = auditor.read(current)
            forged['release'] = 'undeclared changed approval scope'
            atomic_json(current, forged)
            correction['corrected_receipt_sha256'] = sha(current)
            atomic_json(root/'acceptance'/'pi0.pilot.evidence_path_correction.json', correction)
            with self.assertRaises(ValueError): auditor.ReceiptIndex(root, {'pi0'})


def ledger_fixture(directory):
    attempt = Path(directory)
    revision = attempt/'throughput_revisions'/'two_env_v1'
    condition = dict(id='pi0_full', model='pi0', schedule={'kind': 'flow', 'm': .5, 'num_steps': 2},
                     mask='full', skip=[], extension=False)
    manifest = dict(protocol_hashes={'pi0': 'protocol'}, conditions=[condition])
    atomic_json(attempt/'condition_manifest.json', manifest)
    atomic_json(revision/'revision_manifest.json', dict(kind='two_env_sequential_server_execution_revision', source_attempt=str(attempt)))
    revision_hash = sha(revision/'revision_manifest.json')
    sessions = {phase: revision/'sessions'/f'pi0_{phase}_fixture' for phase in ('gate', 'pilot', 'run')}
    rows = list(planned(manifest))[:3]
    entries, candidates = [], []
    for index, row in enumerate(rows):
        phase, slot = ('pilot', index) if index < 2 else ('run', 0)
        token, pid = f'{100+index}-{index}', 100+index
        directory = attempt/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/token
        inference = dict(call=index+1 if phase == 'pilot' else 1, category='formal', input_hash='input',
            noise_hash='noise', raw_prediction_hash='raw', seed=index, loaded_code_hashes={},
            noise_path=str(directory/'noise.npz'), server_prediction_path=str(directory/'raw.npz'),
            audit=dict(NFE=2, actual_NFE=2, schedule=condition['schedule'], skip=[], cache_key='key', instrumentation_mode='instrumented_audit'))
        directory.mkdir(parents=True)
        with (directory/'infer.jsonl').open('w') as stream:
            stream.write(json.dumps(inference)+'\n')
        server_row = dict(inference, prediction_path=inference['server_prediction_path'])
        server_row.pop('server_prediction_path')
        ledger_path = sessions[phase]/'server'/'prediction_ledger.jsonl'
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with ledger_path.open('a') as stream:
            stream.write(json.dumps(server_row)+'\n')
        atomic_json(directory/'validated.json', dict(valid=True, video_verified=True, episode_key=row['key'],
            model='pi0', suite=row['suite'], task=row['task'], init=row['init'], success=False,
            artifacts={'infer.jsonl': sha(directory/'infer.jsonl')}))
        entry = dict(row, token=token, pid=pid, start_ticks=str(pid*10), status='valid',
            directory=str(directory), validated_sha256=sha(directory/'validated.json'), success=False,
            execution_revision=str(revision), execution_revision_sha256=revision_hash,
            execution_mode='two_process_envs_sequential_server', throughput_session=str(sessions[phase]), env_slot=slot)
        entries.append(entry)
        if index < 2:
            candidate = dict(slot=slot, token=token, key=row['key'], candidate_directory=str(directory),
                escrow_directory=str(sessions['pilot']/'escrow'/token), validated_sha256=entry['validated_sha256'], success=False)
            candidates.append(candidate)
            atomic_json(sessions['pilot']/f'pilot{slot}.json', candidate)
    atomic_json(revision/'pi0.gate_ready.json', dict(session=str(sessions['gate'])))
    atomic_json(revision/'pi0.pilot_ready.json', dict(model='pi0', phase='pilot', revision_manifest_sha256=revision_hash,
        session=str(sessions['pilot']), results=candidates))
    atomic_json(revision/'pi0.pilot_session.json', dict(session=str(sessions['pilot']), revision_manifest_sha256=revision_hash))
    correction = signatures(revision)
    for entry in entries:
        entry['throughput_acceptance_sha256'] = correction['original_receipt_sha256'] if entry['env_slot'] == 1 or 'pilot' in entry['throughput_session'] else correction['corrected_receipt_sha256']
    for phase, session in sessions.items():
        signature = {'gate': sha(revision/'acceptance/pi0.source.json'),
            'pilot': sha(revision/'acceptance/pi0.gate.json'), 'run': correction['original_receipt_sha256']}[phase]
        atomic_json(session/'execution_receipt.json', dict(schema=1, model='pi0', phase=phase, environments=2,
            revision_manifest_sha256=revision_hash, phase_acceptance_sha256=signature,
            frozen_runtime_protocol={}, server_pid=200, manager_pid=201, gpu=0, port=123))
        atomic_json(session/'server/native_protocol.json', {})
        atomic_json(session/'server/service_lease.json', dict(protocol={}, pid=200, gpu='0', port=123))
    atomic_json(attempt/'episode_ledger.json', dict(entries=entries))
    return attempt, revision, manifest, entries


class LedgerBoundaryTests(unittest.TestCase):
    def run_audit(self, attempt, manifest, final=False):
        with ExitStack() as stack:
            stack.enter_context(patch('runtime_data.verify_frozen', return_value=manifest))
            stack.enter_context(patch('runtime_data.verify_parity', return_value={}))
            stack.enter_context(patch('runtime_data.verify_acceptance', return_value={}))
            stack.enter_context(patch.object(auditor, 'verify_revision', return_value=({'source_attempt': str(attempt), 'code_hashes': {}}, manifest)))
            stack.enter_context(patch.object(auditor, 'verify_gate'))
            return auditor.audit(attempt, partial=not final)

    def test_actual_initial_session_and_current_run_row_both_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt, _, manifest, _ = ledger_fixture(directory)
            index = self.run_audit(attempt, manifest)
            self.assertTrue(index['partial'])
            self.assertEqual(len(index['throughput_entries']), 3)
            self.assertEqual(len(index['sessions']), 3)
            with self.assertRaises(ValueError): self.run_audit(attempt, manifest, final=True)

    def test_unknown_sha_wrong_session_and_extra_pilot_candidate_rejected(self):
        for mutation in ('unknown_sha', 'wrong_session', 'extra_pilot', 'wrong_model', 'duplicate_key'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                attempt, _, manifest, entries = ledger_fixture(directory)
                if mutation == 'unknown_sha': entries[2]['throughput_acceptance_sha256'] = 'unknown'
                elif mutation == 'wrong_session': entries[2]['throughput_session'] += '_other'
                elif mutation == 'extra_pilot': entries[2]['throughput_session'] = entries[0]['throughput_session']
                elif mutation == 'wrong_model': entries[2]['model'] = 'db-cogact'
                elif mutation == 'duplicate_key': entries.append(copy.deepcopy(entries[2]))
                atomic_json(attempt/'episode_ledger.json', dict(entries=entries))
                with self.assertRaises(ValueError): self.run_audit(attempt, manifest)

    def test_snapshot_recheck_rejects_indexed_bytes_or_server_prefix_drift(self):
        for mutation in ('receipt', 'server_prefix'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                attempt, revision, manifest, _ = ledger_fixture(directory)
                index = self.run_audit(attempt, manifest)
                auditor.verify_index_snapshot(index)
                if mutation == 'receipt':
                    atomic_json(revision/'acceptance/pi0.source.json', {'changed': True})
                else:
                    prefix = next(item['server_prediction_ledger_prefix'] for item in index['sessions'].values() if 'server_prediction_ledger_prefix' in item)
                    atomic_json(prefix['path'], {'changed': True})
                with self.assertRaises(ValueError): auditor.verify_index_snapshot(index)


class FinalWrapperTests(unittest.TestCase):
    def run_wrapper(self, root, drift=False):
        parent, benchmark, view = root/'parent', root/'benchmark', root/'view'
        atomic_json(parent/'episode_ledger.json', {'entries': []})
        atomic_json(benchmark/'revision_manifest.json', {'source_attempt': str(parent)})
        source = root/'scientific_fixture.py'
        source.write_text('# immutable scientific analysis fixture\n')
        module = types.SimpleNamespace(__file__=str(source))
        def create_view(revision, output):
            Path(output).mkdir()
            atomic_json(Path(output)/'benchmark_revision_view_receipt.json', {'routes': {}})
            atomic_json(parent/'benchmark_revision_index.json', {'analysis_view': str(output)})
        def analyze(output):
            for name in ('analysis_input_receipt.json', 'analysis_sources.json', 'completion_audit.json'):
                atomic_json(Path(output)/name, {'scientific_fixture': True})
        module.analyze = analyze
        module.audit = lambda *args, **kwargs: ({'complete': True}, {})
        index = dict(partial=False, ledger_snapshot_sha256=sha(parent/'episode_ledger.json'))
        with patch.object(final_analysis.throughput, 'audit', return_value=index) as audit, \
             patch.object(final_analysis.throughput, 'verify_index_snapshot', side_effect=ValueError('drift') if drift else None) as recheck, \
             patch.object(final_analysis.benchmark, 'create_view', side_effect=create_view), \
             patch.object(final_analysis.benchmark, 'install_analysis_route'), \
             patch.dict('sys.modules', {'runtime_analysis': module}):
            if drift:
                with self.assertRaises(ValueError): final_analysis.finalize(benchmark, view)
                self.assertFalse((view/'final_analysis_provenance_receipt.json').exists())
            else:
                receipt = final_analysis.finalize(benchmark, view)
                self.assertTrue(receipt['complete'])
                self.assertTrue(receipt['independent_final_C_acceptance_required'])
                self.assertEqual(receipt['analysis_input_receipt_sha256'], sha(view/'analysis_input_receipt.json'))
                self.assertIn('throughput_provenance_index_sha256', auditor.read(view/'benchmark_revision_view_receipt.json'))
            audit.assert_called_once_with(parent, partial=False)
            recheck.assert_called_once_with(index)

    def test_final_wrapper_joins_receipts_without_changing_original_analysis_receipt(self):
        with tempfile.TemporaryDirectory() as directory: self.run_wrapper(Path(directory))

    def test_post_analysis_provenance_drift_cannot_seal_final(self):
        with tempfile.TemporaryDirectory() as directory: self.run_wrapper(Path(directory), drift=True)


if __name__ == '__main__':
    unittest.main()
