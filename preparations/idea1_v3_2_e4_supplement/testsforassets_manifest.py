"""CPU-only acquisition safety tests; no network or model construction."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from . import assets, manifest


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = headers or {}
        self.url = 'https://example.test/pinned'


class AcquisitionSafety(unittest.TestCase):
    def item(self, directory, data=b'abcdef'):
        return dict(repository='owner/model', revision='a'*40, path='model.safetensors',
                    size=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    url='https://example.test/pinned', target=str(Path(directory)/'model.safetensors'))

    def test_revision_and_path_guard(self):
        for revision, filename in [('main','a.json'), ('a'*40,'../escape'), ('a'*40,'/absolute')]:
            with self.assertRaises(ValueError):
                assets.resolve_url('owner/repo', revision, filename)
        with self.assertRaises(ValueError):
            manifest.safe_target('/tmp/example', '../escape')

    def test_metadata_cap(self):
        with patch.object(assets.urllib.request, 'urlopen', return_value=Response(b'12345')):
            with self.assertRaises(ValueError):
                assets.bounded_read('https://example.test/meta', maximum=4)
        with self.assertRaises(ValueError):
            assets.bounded_read('http://example.test/meta')

    def test_weight_never_small_text(self):
        row = dict(rfilename='model.safetensors', size=10, lfs={'sha256':'b'*64})
        self.assertFalse(assets.file_record('o/r', 'a'*40, row)['metadata_only'])

    def test_verified_download_and_existing(self):
        with tempfile.TemporaryDirectory() as d:
            item = self.item(d)
            with patch.object(manifest.urllib.request, 'urlopen', return_value=Response(b'abcdef')):
                self.assertEqual(manifest.download_file(item)['status'], 'downloaded_verified')
            self.assertEqual(manifest.download_file(item)['status'], 'verified_existing')

    def test_mismatch_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            item = self.item(d); Path(item['target']).write_bytes(b'wrong')
            with self.assertRaises(FileExistsError):
                manifest.download_file(item)
            self.assertEqual(Path(item['target']).read_bytes(), b'wrong')

    def test_resume_requires_exact_range(self):
        with tempfile.TemporaryDirectory() as d:
            item = self.item(d); target = Path(item['target'])
            target.with_name(target.name+'.part').write_bytes(b'abc')
            identity = {k:item[k] for k in ('repository','revision','path','size','sha256','url')}
            target.with_name(target.name+'.download_identity.json').write_text(json.dumps(identity))
            with patch.object(manifest.urllib.request, 'urlopen', return_value=Response(b'def', 206, {'Content-Range':'bytes 3-5/6'})) as remote:
                manifest.download_file(item)
                self.assertEqual(remote.call_args.args[0].headers['Range'], 'bytes=3-')
            self.assertEqual(target.read_bytes(), b'abcdef')

    def test_bad_sha_preserves_partial(self):
        with tempfile.TemporaryDirectory() as d:
            item = self.item(d)
            with patch.object(manifest.urllib.request, 'urlopen', return_value=Response(b'xxxxxx')):
                with self.assertRaises(ValueError):
                    manifest.download_file(item)
            self.assertFalse(Path(item['target']).exists())
            self.assertEqual(Path(item['target']+'.part').read_bytes(), b'xxxxxx')

    def test_manifest_digest_and_grant(self):
        with tempfile.TemporaryDirectory() as d:
            plan = dict(target_root=d, files=[], repositories=[{'repository':'owner/repo'}])
            plan['manifest_hash'] = manifest.payload_digest(plan)
            p = Path(d)/'manifest.json'; p.write_text(json.dumps(plan))
            self.assertEqual(manifest.load(p), plan)
            with self.assertRaises(ValueError):
                manifest.execute(plan, Path(d)/'receipt.json', [])
            plan['files'] = ['tampered']; p.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                manifest.load(p)

    def test_archive_member_safety(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'assets.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('assets/object.xml', b'abc')
            self.assertEqual(assets.inspect_evaluation_zip(p)['total_uncompressed_bytes'], 3)
            destination = Path(d)/'destination'; destination.mkdir()
            assets.extract_evaluation_zip(p, destination, Path(d)/'extraction.json')
            self.assertEqual((destination/'assets/object.xml').read_bytes(), b'abc')
            with self.assertRaises(FileExistsError):
                assets.extract_evaluation_zip(p, destination, Path(d)/'second.json')
            with self.assertRaises(ValueError):
                assets.inspect_evaluation_zip(p, maximum_uncompressed=2)
            p = Path(d)/'escape.zip'
            with zipfile.ZipFile(p, 'w') as z:
                z.writestr('assets/../../escape', b'x')
            with self.assertRaises(ValueError):
                assets.inspect_evaluation_zip(p)
            wrapped = Path(d)/'wrapped.zip'
            with zipfile.ZipFile(wrapped, 'w') as z:
                z.writestr('official/wrapper/assets/a.xml', b'abc')
            with self.assertRaises(ValueError):
                assets.inspect_evaluation_zip(wrapped)
            self.assertEqual(assets.inspect_evaluation_zip(wrapped, wrapper_prefix='official/wrapper/')['member_count'], 1)


if __name__ == '__main__':
    unittest.main()
