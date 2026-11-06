"""Read-only tests against preserved real gate bytes, never model calls."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from common import ROOT
from throughput_provenance_audit_v3 import ReceiptIndex
import pi0_prompt_cache_failed_audit as failed
from pi0_prompt_cache_audit_helpers_v3 import verify_gate

P = ROOT/'policies/realtime-vla-flash/results/idea1_v2_2_sampling_steps/run_20261018T024042Z'
R = P/'throughput_revisions/pi0_prompt_cache_v1'
O = P/'throughput_revisions/pi0_prompt_cache_order_v2'


@unittest.skipUnless((O/'pi0.gate_ready.json').exists(), 'Preserved actual gates unavailable')
class FailedAuditTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((P/'condition_manifest.json').read_text())
        self.ledger = json.loads((P/'episode_ledger.json').read_text())
        self.declaration = failed.declared_failures({str(R), str(O)})[str(R)]

    def index(self):
        index = ReceiptIndex(R, set())
        index.add_receipt(R/'acceptance/pi0.source.json', 'pi0', 'source')
        return index

    def test_actual_failed_sixteen_outputs_recomputed(self):
        index = self.index()
        failed.verify_failed(index, self.manifest, self.ledger, self.declaration)
        self.assertIn(str(R/'independent_gate_failure.json'), index.files)

    def test_every_formal_status_rejected(self):
        for status in ('valid', 'reserved', 'invalid'):
            ledger = copy.deepcopy(self.ledger)
            ledger['entries'].append(dict(status=status, execution_revision=str(R)))
            with self.assertRaisesRegex(ValueError, 'formal ledger'):
                failed.verify_failed(self.index(), self.manifest, ledger, self.declaration)

    def test_unknown_or_tampered_report_rejected(self):
        with patch.object(failed, 'sha', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'Unknown or changed'):
                failed.declared_failures({str(R), str(O)})
        bad = dict(self.declaration, report_sha256='changed')
        with self.assertRaisesRegex(ValueError, 'declaration drift'):
            failed.verify_failed(self.index(), self.manifest, self.ledger, bad)

    def test_actual_order_controlled_gate(self):
        ready = json.loads((O/'pi0.gate_ready.json').read_text())
        execution = json.loads((Path(ready['session'])/'execution_receipt.json').read_text())
        index = ReceiptIndex(O, set())
        verify_gate(index, 'pi0', self.manifest, execution['frozen_runtime_protocol'])

    def test_pretend_pass_or_extra_session_rejected(self):
        original_exists = Path.exists
        with patch.object(Path, 'exists', lambda path: True if path == R/'pi0.gate_ready.json' else original_exists(path)):
            with self.assertRaisesRegex(ValueError, 'successful phase'):
                failed.verify_failed(self.index(), self.manifest, self.ledger, self.declaration)
        original_glob = Path.glob
        def extra(path, pattern):
            paths = list(original_glob(path, pattern))
            if path == R/'sessions' and pattern == '*/execution_receipt.json':
                paths.append(R/'sessions/unapproved_pilot/execution_receipt.json')
            return iter(paths)
        with patch.object(Path, 'glob', extra):
            with self.assertRaisesRegex(ValueError, 'unexpected sessions'):
                failed.verify_failed(self.index(), self.manifest, self.ledger, self.declaration)

    def test_unknown_failed_kind_rejected(self):
        original_read = failed.read
        def changed(path):
            value = original_read(path)
            if Path(path) == R/'revision_manifest.json':
                value['kind'] = 'unknown'
            return value
        with patch.object(failed, 'read', changed):
            with self.assertRaisesRegex(ValueError, 'Unknown failed'):
                failed.declared_failures({str(R), str(O)})


if __name__ == '__main__':
    unittest.main()
