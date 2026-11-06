import json
from pathlib import Path
import tempfile
import unittest
from scripts.experiment11.audit_next_round import ledger, verify_hashes
from scripts.experiment11.call_budget import CallBudget
from scripts.experiment10.common import sha256


class AuditGuardTests(unittest.TestCase):
    def test_unfinished_call_cannot_pass_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'calls.sqlite'
            budget=CallBudget(path,2)
            call=budget.reserve({'kind':'test'})
            with self.assertRaisesRegex(RuntimeError,'unfinished'):
                ledger(path,2)
            budget.finish(call,'failed',{'reason':'recorded failure'})
            rows,statuses=ledger(path,2)
            self.assertEqual(statuses,{'failed':1})
            self.assertEqual(len(rows),1)
            with self.assertRaisesRegex(RuntimeError,'limit changed'):
                ledger(path,3)

    def test_changed_output_does_not_pass_green_component_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'data.csv';output.write_text('original')
            marker=root/'audit.json'
            marker.write_text(json.dumps({'complete':True,'outputs':{'data.csv':sha256(output)}}))
            verify_hashes(marker)
            output.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Hash mismatch'):
                verify_hashes(marker)


if __name__=='__main__':
    unittest.main()
