import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pi0_prompt_cache_order_revision as order


class OrderTests(unittest.TestCase):
    def identity(self, root):
        return dict(reference_order=list(range(8)), cached_order=list(range(8)),
            gate_controller=dict(source=str(order.HERE), sha256=order.sha(order.HERE),
                scope='gate_only_order_control', failed_revision=str(root),
                failed_revision_manifest_sha256=order.sha(root/'revision_manifest.json')))

    def test_exact_order_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'revision_manifest.json').touch()
            identity = self.identity(root)
            order.validate_order(identity)
            for key in ('reference_order', 'cached_order'):
                bad = copy.deepcopy(identity)
                bad[key] = [0, 2, 4, 6, 1, 3, 5, 7]
                with self.assertRaises(ValueError): order.validate_order(bad)
            identity['gate_controller']['sha256'] = 'unknown'
            with self.assertRaises(ValueError): order.validate_order(identity)

    def test_gate_validates_before_any_original_calls(self):
        with patch.object(order.original, 'verify_revision', return_value=({}, {})), patch.object(order.original, 'gate') as gpu:
            with self.assertRaises(ValueError): order.gate('/unused', 0)
            gpu.assert_not_called()


if __name__ == '__main__':
    unittest.main()
