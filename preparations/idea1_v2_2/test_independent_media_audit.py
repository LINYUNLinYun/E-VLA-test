"""CPU-only temporary video fixtures; never reads a real experiment video."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import independent_media_audit as media
from common import atomic_json, sha


class MediaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.attempt = Path(self.temp.name)/'attempt'
        self.attempt.mkdir()
        self.evidence = self.attempt/'media_audits/test'
        self.row = dict(key='key1', condition='condition', model='pi0', suite='suite', task=0, init=6,
            protocol_hash='protocol', schedule={'num_steps': 2}, mask='full')
        self.manifest = dict(manifest_hash='canonical')
        atomic_json(self.attempt/'condition_manifest.json', self.manifest)
        self.directory = self.attempt/'episodes/condition/suite/task00/init06/token'
        self.directory.mkdir(parents=True)
        subprocess.run(['/usr/bin/ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=black:s=16x16:r=20',
            '-frames:v', '3', '-c:v', 'libx264', '-threads', '1', str(self.directory/'video.mp4')], check=True, capture_output=True)
        self.result = dict(model='pi0', suite='suite', task=0, init=6, success=True, valid=True, kind='formal',
            directory=str(self.directory), video='video.mp4', actions=2, frames=3)
        atomic_json(self.directory/'result.json', self.result)
        self.validated = dict(self.result, episode_key='key1', artifacts={name:sha(self.directory/name) for name in ('result.json', 'video.mp4')})
        atomic_json(self.directory/'validated.json', self.validated)
        self.entry = dict(self.row, token='token', status='valid', success=True, directory=str(self.directory),
            validated_sha256=sha(self.directory/'validated.json'))
        atomic_json(self.attempt/'episode_ledger.json', dict(entries=[self.entry]))
        with patch.object(media, 'planned', return_value=iter([self.row])):
            media.prepare(self.attempt, self.evidence)
        atomic_json(self.evidence/'source_acceptance.json', dict(schema=1, accepted=True, reviewer_model='gpt-6.1-sol',
            reasoning_effort='medium', source_manifest_sha256=sha(self.evidence/'source_manifest.json')))

    def run_audit(self, **kwargs):
        with patch.object(media, 'planned', return_value=iter([self.row])):
            return media.audit(self.attempt, self.evidence, **kwargs)

    def test_actual_full_decode_and_strict_resume_final(self):
        index = media.read(self.run_audit(max_items=1))
        self.assertTrue(index['partial'])
        self.assertEqual(index['covered_count'], 1)
        receipt = self.evidence/'receipts/key1.json'
        original = receipt.read_bytes()
        final = media.read(self.run_audit(final=True))
        self.assertTrue(final['complete'])
        self.assertEqual(final['new_decodes'], 0)
        self.assertEqual(receipt.read_bytes(), original)

    def republish_metadata(self):
        atomic_json(self.directory/'result.json', self.result)
        self.validated['artifacts']['result.json'] = sha(self.directory/'result.json')
        atomic_json(self.directory/'validated.json', self.validated)
        self.entry['validated_sha256'] = sha(self.directory/'validated.json')
        atomic_json(self.attempt/'episode_ledger.json', dict(entries=[self.entry]))

    def test_valid_task_failure_is_included(self):
        self.result['success'] = self.validated['success'] = self.entry['success'] = False
        self.republish_metadata()
        index = media.read(self.run_audit())
        self.assertEqual(index['covered_count'], 1)

    def test_republished_tuple_cannot_reuse_old_receipt(self):
        self.run_audit()
        self.result['extra'] = 'changed metadata'
        self.republish_metadata()
        for options in ({'keys': ['key1']}, {'final': True}):
            with self.assertRaisesRegex(ValueError, 'stale canonical ledger tuple'):
                self.run_audit(**options)

    def test_metadata_frame_contradiction_rejected(self):
        self.validated['frames'] = 4
        self.republish_metadata()
        with self.assertRaisesRegex(ValueError, 'action/frame'):
            self.run_audit()

    def test_nonzero_decode_not_pass_despite_final_progress(self):
        probe = b'{"streams":[{"codec_type":"video","nb_read_frames":"3"}]}'
        results = [(dict(exit_code=0), probe, b''), (dict(exit_code=1), b'frame=3\nprogress=end\n', b'bad payload')]
        with patch.object(media, 'command_log', side_effect=results):
            with self.assertRaisesRegex(ValueError, 'decode failed'):
                self.run_audit()
        self.assertFalse(media.read(self.evidence/'receipts/key1.json')['passed'])

    def test_tampered_video_and_stale_resume_rejected(self):
        self.run_audit()
        with (self.directory/'video.mp4').open('ab') as stream: stream.write(b'changed')
        with self.assertRaises(ValueError): self.run_audit(keys=['key1'])
        with self.assertRaises(ValueError): self.run_audit(final=True)

    def test_truncated_video_is_failed_and_not_reusable(self):
        video = self.directory/'video.mp4'
        video.write_bytes(video.read_bytes()[:100])
        self.validated['artifacts']['video.mp4'] = sha(video)
        atomic_json(self.directory/'validated.json', self.validated)
        self.entry['validated_sha256'] = sha(self.directory/'validated.json')
        atomic_json(self.attempt/'episode_ledger.json', dict(entries=[self.entry]))
        with self.assertRaises(ValueError): self.run_audit()
        self.assertFalse(media.read(self.evidence/'receipts/key1.json')['passed'])
        with self.assertRaises(ValueError): self.run_audit(keys=['key1'])

    def test_short_actual_frame_count_rejected(self):
        self.result.update(actions=3, frames=4)
        atomic_json(self.directory/'result.json', self.result)
        self.validated.update(actions=3, frames=4)
        self.validated['artifacts']['result.json'] = sha(self.directory/'result.json')
        atomic_json(self.directory/'validated.json', self.validated)
        self.entry['validated_sha256'] = sha(self.directory/'validated.json')
        atomic_json(self.attempt/'episode_ledger.json', dict(entries=[self.entry]))
        with self.assertRaisesRegex(ValueError, 'decoded count'): self.run_audit()

    def test_duplicate_key_and_escape_rejected(self):
        atomic_json(self.attempt/'episode_ledger.json', dict(entries=[self.entry]*2))
        with self.assertRaisesRegex(ValueError, 'Duplicate'): self.run_audit()
        atomic_json(self.attempt/'episode_ledger.json', dict(entries=[dict(self.entry, directory=str(self.attempt))]))
        with self.assertRaisesRegex(ValueError, 'canonical physical'): self.run_audit()

    def test_partial_cannot_finalize_or_cross_attempt_resume(self):
        self.run_audit()
        with patch.object(media, 'planned', return_value=iter([self.row, dict(self.row, key='missing')])):
            with self.assertRaises(ValueError): media.audit(self.attempt, self.evidence, final=True)
        with self.assertRaisesRegex(ValueError, 'direct child'):
            media.verify_source(self.attempt.parent, self.evidence)

    def test_source_binary_and_receipt_log_drift_rejected(self):
        real = media.tools()
        with patch.object(media, 'tools', return_value=dict(real, ffmpeg=dict(real['ffmpeg'], sha256='changed'))):
            with self.assertRaisesRegex(ValueError, 'source/tool'): self.run_audit()
        identity = media.read(self.evidence/'source_manifest.json')
        identity['source_sha256'] = 'changed'
        with patch.object(media, 'read', side_effect=lambda path: identity if Path(path).name == 'source_manifest.json' else json.loads(Path(path).read_text())):
            with self.assertRaises(ValueError): self.run_audit()
        self.run_audit()
        receipt = media.read(self.evidence/'receipts/key1.json')
        Path(receipt['commands'][1]['stdout']['path']).write_text('frame=3\nprogress=continue\n')
        with self.assertRaisesRegex(ValueError, 'log changed'): self.run_audit(keys=['key1'])

    def test_frame_parser_and_timeout_fail_closed(self):
        probe = b'{"streams":[{"codec_type":"video","nb_read_frames":"3"}]}'
        for output in (b'frame=3\nprogress=continue\n', b'frame=2\nprogress=end\n'):
            with self.assertRaises(ValueError): media.frame_counts(probe, output)
        with patch.object(media.subprocess, 'run', side_effect=subprocess.TimeoutExpired('decode', 1)):
            record, _, _ = media.command_log(['decode'], self.directory/'timeout')
            self.assertEqual(record['exit_code'], 'timeout')

    def test_immutable_receipt_publish(self):
        path = self.directory/'immutable.json'
        media.unique_write(path, {'one': 1})
        with self.assertRaises(FileExistsError): media.unique_write(path, {'two': 2})
        self.assertEqual(media.read(path), {'one': 1})


if __name__ == '__main__':
    unittest.main()
