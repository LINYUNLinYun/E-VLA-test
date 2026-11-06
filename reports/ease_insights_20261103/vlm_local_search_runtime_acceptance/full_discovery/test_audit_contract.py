"""Small CPU-only fail-closed contract checks; never opens discovery artifacts."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
AUDIT_PATH = Path(__file__).with_name("audit.py")
SPEC = importlib.util.spec_from_file_location("full_discovery_audit_contract", AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def fake_full_records():
    records = {}
    for condition in AUDIT.CONDITION_IDS:
        for suite in AUDIT.SUITES:
            for task in range(10):
                for initial in (36, 37, 38):
                    case = (suite, task, initial)
                    success = not (condition == "jac_anchor" and suite == "libero_spatial" and task == 0 and initial == 37)
                    records[condition, case] = {"success": success}
    return records


class FullDiscoveryAuditContract(unittest.TestCase):
    def test_jac_120_cannot_pass_discovery_720_completeness(self):
        records = {("jac_anchor", (suite, task, initial)): {"success": True}
                   for suite in AUDIT.SUITES for task in range(10) for initial in (36, 37, 38)}
        summary = AUDIT.stage_summary(dict(expected=720, records=records, complete=False,
            missing=[{"condition": condition} for condition in AUDIT.CONDITION_IDS if condition != "jac_anchor"],
            invalid=[], unexpected=[]))
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["valid"], 120)
        self.assertEqual(AUDIT.final_status(summary["passed"], []), "incomplete")

    def test_scientific_failure_is_preserved_but_not_an_acceptance_gate(self):
        records = fake_full_records()
        summary = AUDIT.stage_summary(dict(expected=720, records=records, complete=True,
            missing=[], invalid=[], unexpected=[]))
        false_outcomes = [(condition, case) for (condition, case), record in records.items()
                          if record["success"] is False]
        self.assertTrue(summary["passed"])
        self.assertEqual(false_outcomes, [("jac_anchor", ("libero_spatial", 0, 37))])
        self.assertEqual(AUDIT.final_status(summary["passed"], []), "pass")

    def test_frozen_gate_uses_full_native_name(self):
        names = AUDIT.gate_names(AUDIT.CONDITION_IDS, AUDIT.SUITES)
        self.assertEqual(len(names), 332)
        self.assertIn("libero_spatial/full_native", names)
        self.assertNotIn("libero_spatial/full_d10/actual_execution", names)
        self.assertIn("libero_spatial/local_long/actual_execution", names)

    def test_first_case_request_validator_imports_without_running_audit(self):
        module, path = AUDIT.request_auditor()
        self.assertTrue(path.is_file())
        self.assertTrue(callable(module.audit_requests))


if __name__ == "__main__":
    unittest.main(verbosity=2)
