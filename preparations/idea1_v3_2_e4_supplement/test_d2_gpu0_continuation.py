import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import d2_gpu0_continuation as continuation

PARENT=Path('results/idea1_v3_2_e4_supplement/libero_plus_v14_d2_quality_gpu01_20261023_run2/manifest.json')


class Continuation(unittest.TestCase):
    def test_real_frozen39_plus_pending58_partition_no_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'continuation';m=continuation.register(PARENT,root)
            self.assertEqual(len(m['jobs']),58);self.assertEqual(len(m['parent_gpu1_reused_complete']),39)
            old=continuation.protocol_identity(PARENT);odd=continuation.v3.shard_jobs(old,1)
            key=lambda j:(j['group'],j['source_index'])
            pending={key(j) for j in m['jobs']};done={key(j) for j in m['parent_gpu1_reused_complete']}
            self.assertFalse(pending&done);self.assertEqual(pending|done,{key(j) for j in odd})
            self.assertIn(('perturbed350',154),pending);self.assertEqual(m['assigned_gpus'],{'0':continuation.diagnostic.CARDS[0]})
            self.assertEqual(continuation.verify(root/'manifest.json')['manifest_hash'],m['manifest_hash'])
            self.assertEqual(m['deadline_unix'],1792762731)

    def test_parent_running_waits_without_gpu_and_technical_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);watcher_path=root/'watcher.json';owner=dict(pid=111,start_ticks='fixed')
            watcher=dict(manifest_hash=continuation.PARENT,owner=owner,status='executing_remaining_D2_only')
            watcher_path.write_text(json.dumps(watcher));m=dict(parent_gpu0_watcher_path=str(watcher_path),parent_gpu0_watcher=owner)
            with patch.object(continuation,'alive',return_value=True),patch.object(continuation,'sample_gpu0') as sample:
                self.assertFalse(continuation.parent_readiness(m)['ready']);sample.assert_not_called()
            watcher['status']='technical_failure';watcher_path.write_text(json.dumps(watcher))
            with self.assertRaisesRegex(RuntimeError,'technical failure'):continuation.parent_readiness(m)
            watcher['status']='executing_remaining_D2_only';watcher_path.write_text(json.dumps(watcher))
            with patch.object(continuation,'alive',return_value=False),self.assertRaisesRegex(RuntimeError,'without normal'):continuation.parent_readiness(m)

    def test_gpu_sample_queries_only_assigned_uuid0(self):
        outputs=['','0, %s, NVIDIA GeForce RTX 4090, 1 MiB, 0 %%'%continuation.diagnostic.CARDS[0]]
        with patch.object(continuation.subprocess,'check_output',side_effect=outputs) as tool:
            self.assertTrue(continuation.sample_gpu0()['idle'])
            self.assertTrue(all(call.args[0][2]==continuation.diagnostic.CARDS[0] for call in tool.call_args_list))


if __name__=='__main__':unittest.main()
