import copy
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from common import Ledger, atomic_json, digest, sha
import retry_history_audit as history
import retry_history_transition as transition


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'attempt';self.root.mkdir()
        self.row=dict(key='0'*63+'2',condition='c',model='pi0',suite='s',task=0,init=6,protocol_hash='p',schedule={'steps':2},mask='full')
        self.old=dict(self.row,token='999999999-1',pid=999999999,start_ticks='old',status='reserved')
        atomic_json(self.root/'condition_manifest.json',dict(manifest_hash='canonical'))
        atomic_json(self.root/'episode_ledger.json',dict(entries=[self.old]))
        self.source=self.root/'source_acceptance.json'
        atomic_json(self.source,dict(schema=1,accepted=True,reviewer_model='gpt-6.1-sol',reasoning_effort='medium',
            source_hashes={'retry_history_transition.py':sha(transition.__file__)}))
        self.patches=[patch.object(transition,'planned',return_value=[self.row]),patch.object(history,'planned',return_value=[self.row]),
            patch('runtime_data.verify_frozen',return_value={})]
        for item in self.patches:item.start();self.addCleanup(item.stop)

    def ledger(self):return json.loads((self.root/'episode_ledger.json').read_text())
    def set_entries(self,entries):atomic_json(self.root/'episode_ledger.json',dict(entries=entries))
    def resolve(self,receipts=()):return history.resolve(history.manifest_identity(self.root/'condition_manifest.json'),(self.root/'episode_ledger.json').read_bytes(),receipts)

    def approval(self,phase):
        plan,_=transition.plan(self.root,self.row['key'],self.old['token'],phase)
        path=self.root/f'{phase}_approval.json'
        atomic_json(path,dict(schema=1,accepted=True,authorized_by='root',phase=phase,target_single_owner=True,
            attempt=str(self.root),
            key=self.row['key'],old_token=self.old['token'],expected_chain_sha256=plan['chain_sha256'],
            manifest_file_sha256=plan['manifest_file_sha256'],source_sha256=sha(transition.__file__),
            diagnosis='Technical owner exit before episode entry; actual exit cause unknown',
            source_acceptance=dict(path=str(self.source),sha256=sha(self.source))))
        return path

    def recover_authorize(self):
        first=transition.execute(self.root,self.row['key'],self.old['token'],self.approval('transition_only'),self.root/'retry_recovery/transition')
        approval=self.approval('authorize_retry')
        def cli(command,**kwargs):
            Ledger(self.root).authorize_retry(self.row['key'],command[command.index('--diagnosis')+1]);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
        with patch.object(transition.subprocess,'run',side_effect=cli):
            second=transition.execute(self.root,self.row['key'],self.old['token'],approval,self.root/'retry_recovery/authorization',
                phase='authorize_retry',transition_receipt=first)
        return [dict(path=str(second),sha256=sha(second))]

    def valid(self,token='222-2'):
        return dict(self.row,token=token,pid=222,start_ticks='new',status='valid',success=False,
            directory=str(self.root/'episodes/c/s/task00/init06'/token),validated_sha256='validsha')

    def test_native_state_only_then_original_retry_selects_only_new_valid_failure(self):
        references=self.recover_authorize()
        rows=self.ledger()['entries'];self.assertEqual(len(rows),1);self.assertEqual(rows[0]['status'],'invalid')
        self.set_entries(rows+[self.valid()])
        resolved=self.resolve(references)
        self.assertEqual(resolved['valid_by_key'][self.row['key']]['token'],'222-2')
        self.assertFalse(resolved['valid_by_key'][self.row['key']]['success'])
        self.assertEqual(resolved['unresolved'],[])
        self.assertEqual(len(resolved['history_index']['keys'][self.row['key']]['attempts']),2)

    def test_actual_original_retry_cli_launches_only_on_temp_ledger(self):
        first=transition.execute(self.root,self.row['key'],self.old['token'],self.approval('transition_only'),self.root/'retry_recovery/transition')
        second=transition.execute(self.root,self.row['key'],self.old['token'],self.approval('authorize_retry'),self.root/'retry_recovery/authorization',
            phase='authorize_retry',transition_receipt=first)
        receipt=json.loads(second.read_text())
        self.assertIn('--launch',receipt['original_cli']['command'])
        self.assertEqual(receipt['original_cli']['exit_code'],0)
        self.set_entries(self.ledger()['entries']+[self.valid()])
        self.assertFalse(self.resolve([dict(path=str(second),sha256=sha(second))])['unresolved'])

    def test_phase_index_appends_and_preserves_old_bytes_without_authorizing_next_step(self):
        first=transition.execute(self.root,self.row['key'],self.old['token'],self.approval('transition_only'),self.root/'retry_recovery/transition')
        original=(self.root/'retry_recovery_index.json').read_bytes()
        first_ref=history.recovery_index_reference(self.root)
        self.assertEqual(Path(first_ref['path']).read_bytes(),original)
        self.assertEqual(history.recovery_references(self.root),[])
        approval=self.approval('authorize_retry')
        def cli(command,**kwargs):
            Ledger(self.root).authorize_retry(self.row['key'],command[command.index('--diagnosis')+1]);return types.SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
        with patch.object(transition.subprocess,'run',side_effect=cli):
            second=transition.execute(self.root,self.row['key'],self.old['token'],approval,self.root/'retry_recovery/authorization','authorize_retry',first)
        self.assertEqual(Path(first_ref['path']).read_bytes(),original)
        self.assertEqual(len(history.recovery_references(self.root)),1)
        self.assertEqual(len(self.ledger()['entries']),1)
        with self.assertRaises(ValueError):transition.register_receipt(self.root,second,'authorize_retry')
        index=json.loads((self.root/'retry_recovery_index.json').read_text())
        index['entries'][0]['phase']='unknown'
        atomic_json(self.root/'retry_recovery_index.json',index)
        with self.assertRaisesRegex(ValueError,'index phase'):history.recovery_references(self.root)

    def test_duplicate_valid_token_post_valid_and_fourth_attempt_rejected(self):
        for rows in ([self.valid(),self.valid('333-3')],[self.valid(),dict(self.old,token='new')],
                     [self.old,self.old],[dict(self.old,token=str(i),status='invalid') for i in range(4)]):
            self.set_entries(rows)
            with self.assertRaises(ValueError):self.resolve()

    def test_missing_empty_wrong_token_or_drifted_retry_authority_rejected(self):
        self.set_entries([dict(self.old,status='invalid'),self.valid()])
        with self.assertRaises(ValueError):self.resolve()
        self.set_entries([dict(self.old,status='invalid',retry_diagnosis=' ',retry_authorized_at=1),self.valid()])
        with self.assertRaises(ValueError):self.resolve()
        self.set_entries([self.old]);references=self.recover_authorize();self.set_entries(self.ledger()['entries']+[self.valid()])
        receipt=Path(references[0]['path']);value=json.loads(receipt.read_text());value['old_token']='wrong';atomic_json(receipt,value)
        with self.assertRaises(ValueError):self.resolve(references)

    def test_cli_launch_log_hash_and_cross_attempt_authority_semantics(self):
        references=self.recover_authorize();self.set_entries(self.ledger()['entries']+[self.valid()])
        receipt=Path(references[0]['path']);saved=json.loads(receipt.read_text())
        changed=copy.deepcopy(saved);changed['original_cli']['command'].remove('--launch');atomic_json(receipt,changed)
        with self.assertRaisesRegex(ValueError,'CLI command'):self.resolve([dict(path=str(receipt),sha256=sha(receipt))])
        atomic_json(receipt,saved)
        Path(saved['original_cli']['stdout']['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'evidence bytes changed'):self.resolve([dict(path=str(receipt),sha256=sha(receipt))])

    def test_dangling_episode_path_refused_without_write(self):
        expected=self.root/'episodes/c/s/task00/init06'/self.old['token']
        expected.parent.mkdir(parents=True);expected.symlink_to(self.root/'missing')
        before=(self.root/'episode_ledger.json').read_bytes()
        with self.assertRaises(ValueError):transition.plan(self.root,self.row['key'],self.old['token'])
        self.assertEqual((self.root/'episode_ledger.json').read_bytes(),before)

    def test_reserved_and_unrecovered_invalid_are_unresolved_not_directory_recovered(self):
        for status in ('reserved','invalid'):
            self.set_entries([dict(self.old,status=status)])
            resolved=self.resolve()
            self.assertFalse(resolved['valid_by_key']);self.assertEqual(len(resolved['unresolved']),1)

    def test_unknown_scientific_key_schedule_mask_or_protocol_rejected(self):
        for fields in ({'key':'unknown'},{'schedule':{'steps':3}},{'mask':'other'},{'protocol_hash':'other'}):
            self.set_entries([dict(self.old,**fields)])
            with self.assertRaises(ValueError):self.resolve()

    def test_live_authorized_or_artifact_target_preconditions_are_zero_write(self):
        for change in ({'retry_diagnosis':'already'},{'directory':'somewhere'},{'validated_sha256':'exists'},{'status':'valid'}):
            self.set_entries([dict(self.old,**change)]);before=(self.root/'episode_ledger.json').read_bytes()
            with self.assertRaises(ValueError):transition.plan(self.root,self.row['key'],self.old['token'])
            self.assertEqual((self.root/'episode_ledger.json').read_bytes(),before)
        self.set_entries([self.old]);before=(self.root/'episode_ledger.json').read_bytes()
        with patch.object(transition,'start_ticks',return_value='old'):
            with self.assertRaises(ValueError):transition.plan(self.root,self.row['key'],self.old['token'])
        self.assertEqual((self.root/'episode_ledger.json').read_bytes(),before)

    def test_changed_target_authority_refuses_before_native_reserve(self):
        approval=self.approval('transition_only')
        self.set_entries([dict(self.old,start_ticks='different')]);before=(self.root/'episode_ledger.json').read_bytes()
        with patch.object(transition.Ledger,'reserve') as native:
            with self.assertRaises(ValueError):transition.execute(self.root,self.row['key'],self.old['token'],approval,self.root/'retry_recovery/refuse')
            native.assert_not_called()
        self.assertEqual((self.root/'episode_ledger.json').read_bytes(),before)

    def test_native_unexpected_exception_or_return_preserved_not_rolled_back(self):
        approval=self.approval('transition_only')
        for name,options in [('wrong',dict(side_effect=RuntimeError('different'))),('return',dict(return_value='new-token'))]:
            with patch.object(transition.Ledger,'reserve',**options):
                with self.assertRaises(RuntimeError):transition.execute(self.root,self.row['key'],self.old['token'],approval,self.root/f'retry_recovery/{name}')
            self.assertTrue((self.root/f'retry_recovery/{name}/failure.json').exists())


if __name__=='__main__':unittest.main()
