import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from common import atomic_json,sha
import independent_media_audit as original
import independent_media_history_audit as controller
import retry_history_audit as history
from test_independent_media_audit import MediaTests


class MediaHistoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=MediaTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.root,self.base=self.fixture.attempt,self.fixture.evidence
        self.evidence=self.root/'media_audits/history'
        self.patch=patch.object(history,'planned',return_value=[self.fixture.row]);self.patch.start();self.addCleanup(self.patch.stop)
        controller.prepare(self.root,self.evidence,self.base)
        atomic_json(self.evidence/'source_acceptance.json',dict(schema=1,accepted=True,reviewer_model='gpt-6.1-sol',reasoning_effort='medium',
            source_manifest_sha256=sha(self.evidence/'source_manifest.json')))

    def test_original_receipt_reused_without_decode_and_final_real_ledger_join(self):
        self.fixture.run_audit()
        receipt=self.base/'receipts/key1.json';before=receipt.read_bytes()
        with patch.object(original,'command_log') as decode:
            partial=controller.audit(self.root,self.evidence,keys=['key1'])
            self.assertEqual(original.read(partial)['new_decodes'],0)
            decode.assert_not_called()
            final=controller.audit(self.root,self.evidence,final=True)
        proof=controller.verify_final_index(self.root,final)
        self.assertEqual(proof['covered_count'],1)
        self.assertEqual(receipt.read_bytes(),before)
        self.assertEqual(original.read(final)['ledger_snapshot_sha256'],sha(self.root/'episode_ledger.json'))

    def test_new_decode_uses_original_function_and_own_history_index(self):
        index=controller.audit(self.root,self.evidence,max_items=1)
        data=original.read(index)
        self.assertEqual(data['new_decodes'],1)
        self.assertTrue((self.base/'receipts/key1.json').exists())
        with self.assertRaisesRegex(ValueError,'Partial'):
            controller.verify_final_index(self.root,index)

    def test_token_validation_source_and_output_drift_reject_reuse(self):
        self.fixture.run_audit()
        receipt=self.base/'receipts/key1.json';saved=original.read(receipt)
        for field,value in [('token','different'),('validated_sha256','different')]:
            changed=copy.deepcopy(saved);changed['binding'][field]=value;atomic_json(receipt,changed)
            with self.assertRaisesRegex(ValueError,'tuple changed'):controller.audit(self.root,self.evidence,keys=['key1'])
        changed=copy.deepcopy(saved);changed['source_manifest_sha256']='different';atomic_json(receipt,changed)
        with self.assertRaisesRegex(ValueError,'source/acceptance'):controller.audit(self.root,self.evidence,keys=['key1'])
        atomic_json(receipt,saved)
        Path(saved['commands'][1]['stdout']['path']).write_text('frame=2\nprogress=end\n')
        with self.assertRaises(ValueError):controller.audit(self.root,self.evidence,keys=['key1'])

    def test_current_video_drift_and_projected_ledger_final_rejected(self):
        index=controller.audit(self.root,self.evidence,final=True)
        with (self.fixture.directory/'video.mp4').open('ab') as stream:stream.write(b'changed')
        with self.assertRaises(ValueError):controller.verify_final_index(self.root,index)
        data=original.read(index)
        atomic_json(self.root/'episode_ledger.json',dict(entries=[]))
        with self.assertRaisesRegex(ValueError,'real complete current ledger'):controller.verify_final_index(self.root,index)


if __name__=='__main__':unittest.main()
