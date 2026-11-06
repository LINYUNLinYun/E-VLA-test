"""Lease protocol tests use files and mocks, never a physical GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import batch_resources


class BatchLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.registry = self.root / "leases.json"
        self.registry.write_text(json.dumps({"2": {"active": False, "owner": "previous"}}))

    def enter(self):
        return batch_resources.exclusive_gpu("GPU-test", self.root / "run", leases_path=self.registry)

    @patch("preparations.idea2.layer_bench.require_idle_gpu", return_value={"processes": []})
    @patch.object(batch_resources, "_gpu_index", return_value="2")
    def test_claim_and_release_own_lease(self, index, idle):
        with self.enter() as receipt:
            self.assertEqual(receipt["status"], "active")
            self.assertTrue(json.loads(self.registry.read_text())["2"]["active"])
        self.assertFalse(json.loads(self.registry.read_text())["2"]["active"])
        self.assertEqual(idle.call_count, 2)
        self.assertEqual(json.loads((self.root / "run/gpu_lease.json").read_text())["status"], "released")

    @patch("preparations.idea2.layer_bench.require_idle_gpu")
    @patch.object(batch_resources, "_gpu_index", return_value="2")
    def test_foreign_active_lease_refused_without_gpu_admission(self, index, idle):
        original = {"2": {"active": True, "owner": "another session"}}
        self.registry.write_text(json.dumps(original))
        with self.assertRaisesRegex(RuntimeError, "active lease"):
            with self.enter():
                self.fail("Must not admit a reserved GPU")
        self.assertEqual(json.loads(self.registry.read_text()), original)
        idle.assert_not_called()

    @patch("preparations.idea2.layer_bench.require_idle_gpu", side_effect=[{}, RuntimeError("busy")])
    @patch.object(batch_resources, "_gpu_index", return_value="2")
    def test_external_startup_race_releases_own_claim(self, index, idle):
        with self.assertRaisesRegex(RuntimeError, "busy"):
            with self.enter():
                self.fail("Must not run on busy GPU")
        self.assertFalse(json.loads(self.registry.read_text())["2"]["active"])

    @patch("preparations.idea2.layer_bench.require_idle_gpu", return_value={})
    @patch.object(batch_resources, "_gpu_index", return_value="2")
    def test_exception_releases_lease(self, index, idle):
        with self.assertRaisesRegex(ValueError, "experiment failed"):
            with self.enter():
                raise ValueError("experiment failed")
        self.assertFalse(json.loads(self.registry.read_text())["2"]["active"])

    @patch("preparations.idea2.layer_bench.require_idle_gpu", return_value={})
    @patch.object(batch_resources, "_gpu_index", return_value="2")
    def test_changed_owner_never_overwritten(self, index, idle):
        foreign = {"2": {"active": True, "owner": "new owner", "token": "foreign"}}
        with self.assertRaisesRegex(RuntimeError, "ownership changed"):
            with self.enter():
                self.registry.write_text(json.dumps(foreign))
        self.assertEqual(json.loads(self.registry.read_text()), foreign)


if __name__ == "__main__":
    unittest.main()
