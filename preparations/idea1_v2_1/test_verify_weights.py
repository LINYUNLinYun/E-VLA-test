import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("verify_weights", Path(__file__).with_name("verify_weights.py"))
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)


class WeightVerificationTests(unittest.TestCase):
    def fixture(self, base):
        revision = "a" * 40
        checkpoint = base / revision
        checkpoint.mkdir()
        payload = b"tiny fake safetensors bytes, not loaded"
        (checkpoint / "one.safetensors").write_bytes(payload)
        index_bytes = json.dumps({"weight_map": {"layer.weight": "one.safetensors"}}).encode()
        index = checkpoint / "model.safetensors.index.json"
        index.write_bytes(index_bytes)
        blob = hashlib.sha1(b"blob " + str(len(index_bytes)).encode() + b"\0" + index_bytes).hexdigest()
        data = {"sha": revision, "id": "test/repo", "siblings": [
            {"rfilename": index.name, "blobId": blob},
            {"rfilename": "one.safetensors", "size": len(payload), "lfs": {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}]}
        hub = base / "hub_file_info.json"
        hub.write_text(json.dumps(data))
        return checkpoint, hub, revision

    def test_exact_bytes_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, hub, revision = self.fixture(Path(directory))
            self.assertTrue(verify.verify(checkpoint, hub, revision)["ok"])
            with self.assertRaises(ValueError):
                verify.verify(checkpoint, hub, "b" * 40)

    def test_missing_size_hash_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, hub, revision = self.fixture(Path(directory))
            shard = checkpoint / "one.safetensors"
            original = shard.read_bytes()
            shard.write_bytes(b"x" * len(original))
            self.assertFalse(verify.verify(checkpoint, hub, revision)["ok"])
            shard.write_bytes(b"short")
            self.assertIn("size mismatch", verify.verify(checkpoint, hub, revision)["shards"][0]["error"])
            shard.unlink()
            self.assertIn("missing", verify.verify(checkpoint, hub, revision)["shards"][0]["error"])

    def test_index_revision_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, hub, revision = self.fixture(Path(directory))
            (checkpoint / "model.safetensors.index.json").write_text('{"weight_map": {"wrong": "one.safetensors"}}')
            with self.assertRaises(ValueError):
                verify.verify(checkpoint, hub, revision)

    def test_path_traversal_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "weights"
            root.mkdir()
            (base / "outside.safetensors").write_bytes(b"outside")
            (root / "escape.safetensors").symlink_to(base / "outside.safetensors")
            for name in ("../outside.safetensors", "/outside.safetensors", "escape.safetensors", "a\\b", "./a"):
                with self.assertRaises(ValueError):
                    verify.confined_file(root, name)


if __name__ == "__main__":
    unittest.main()
