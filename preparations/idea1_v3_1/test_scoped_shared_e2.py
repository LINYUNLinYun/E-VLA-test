from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import atomic_json,read_manifest
from .reverify_shared_e2_scoped import scoped_path_type,verify
from .same_process_parity import sha
from . import audit_shared_e2 as original


class ScopedTests(unittest.TestCase):
    def setUp(self):
        self.manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        self.variants = tuple(v["id"] for v in self.manifest["e2"]["variants"])

    def test_quality_keys_ignored_but_profile_extras_not_hidden(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for arm in self.variants:
                for scene in self.manifest["e2"]["episodes"]:
                    atomic_json(root/"pi05_full"/arm/scene["suite"]/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"/"complete.json",{})
            scoped = scoped_path_type(root,self.variants)
            condition = scoped(root/"pi05_full")
            pattern = "*/*/task*/init*/complete.json"
            before = set(condition.glob(pattern))
            self.assertEqual(len(before),32)
            for task in range(10):
                atomic_json(root/"pi05_full/episodes/libero_spatial"/f"task{task:02d}"/"init00/complete.json",{})
            self.assertEqual(set(condition.glob(pattern)),before)
            self.assertEqual(len(list(Path(root/"pi05_full").glob(pattern))),42)
            self.assertEqual(len(list(condition.glob("episodes/*/task*/init*/complete.json"))),10)
            atomic_json(root/"pi05_full/full_io/libero_spatial/task09/init09/complete.json",{})
            self.assertEqual(len(list(condition.glob(pattern))),33)

    def test_other_roots_and_patterns_not_scoped(self):
        with tempfile.TemporaryDirectory() as root:
            atomic_json(Path(root)/"other/episodes/suite/task00/init00/complete.json",{})
            scoped = scoped_path_type(root,self.variants)
            self.assertEqual(len(list(scoped(Path(root)/"other").glob("*/*/task*/init*/complete.json"))),1)

    def test_source_and_validation_identity_must_match_original(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            atomic_json(root/"validation.json",{})
            atomic_json(root/"audit.json",dict(passed=True,video_verified=True,audit_source_sha256="wrong",validation_receipt_sha256=sha(root/"validation.json")))
            with self.assertRaises(AssertionError):
                verify(root,self.manifest)
            atomic_json(root/"audit.json",dict(passed=True,video_verified=True,audit_source_sha256=sha(original.__file__),validation_receipt_sha256="changed"))
            with self.assertRaises(AssertionError):
                verify(root,self.manifest)


if __name__ == "__main__":
    unittest.main()
