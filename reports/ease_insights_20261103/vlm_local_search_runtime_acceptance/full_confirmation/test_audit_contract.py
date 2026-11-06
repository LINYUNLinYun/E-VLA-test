"""Small CPU-only contracts using in-memory confirmation records only."""
import importlib.util
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
AUDIT_PATH = Path(__file__).with_name("audit.py")
SPEC = importlib.util.spec_from_file_location("full_confirmation_audit_contract", AUDIT_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def fake_manifest():
    ids = AUDIT.EXPECTED_DISCOVERY_CONDITIONS
    return dict(phase="local_search_confirmation_v1",
        planned_conditions=list(AUDIT.EXPECTED_PLANNED_CONDITIONS),
        confirmation_inits=list(AUDIT.CONFIRMATION_INITS),
        cases_per_condition=AUDIT.EXPECTED_CASES_PER_CONDITION,
        total_confirmation_episodes=AUDIT.EXPECTED_TOTAL_CASES,
        conditions=[dict(id=condition) for condition in ids])


def fake_scenes():
    return [dict(suite=suite, task=task, init=init)
            for suite in AUDIT.SUITES for task in range(10) for init in AUDIT.CONFIRMATION_INITS]


def fake_full_records():
    records = {}
    for condition in AUDIT.EXPECTED_PLANNED_CONDITIONS:
        for case in fake_scenes():
            key = (case["suite"], case["task"], case["init"])
            success = not (condition == "jac_anchor" and case ==
                           dict(suite="libero_spatial", task=0, init=39))
            records[condition, key] = {"success": success}
    return records


class FullConfirmationAuditContract(unittest.TestCase):
    def test_jac_200_cannot_pass_confirmation_600_completeness(self):
        records = {}
        for case in fake_scenes():
            key = (case["suite"], case["task"], case["init"])
            records["jac_anchor", key] = {"success": True}
        audit = dict(expected=600, records=records, scenes=fake_scenes(), complete=False,
                     missing=[{"condition": condition} for condition in ("full_d10", "local_long")],
                     invalid=[], unexpected=[])
        summary = AUDIT.stage_summary(audit, fake_manifest())
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["valid"], 200)
        self.assertEqual(AUDIT.final_status(summary["passed"], []), "incomplete")

    def test_final_stage_check_record_assembles_without_duplicate_keys(self):
        check = AUDIT.complete_stage_check_record(dict(passed=True, expected=600, valid=600,
            complete=True, per_condition={"full_d10": 200, "jac_anchor": 200, "local_long": 200}))
        self.assertEqual(check["name"], "complete_600_confirmation_receipts")
        self.assertTrue(check["passed"])
        self.assertEqual(check["valid"], 600)

    def test_success_failure_is_preserved_without_gating_valid_confirmation(self):
        records = fake_full_records()
        summary = AUDIT.stage_summary(dict(expected=600, records=records, scenes=fake_scenes(), complete=True,
            missing=[], invalid=[], unexpected=[]), fake_manifest())
        false_outcomes = [(condition, case) for (condition, case), record in records.items()
                          if record["success"] is False]
        self.assertTrue(summary["passed"])
        self.assertEqual(false_outcomes, [("jac_anchor", ("libero_spatial", 0, 39))])
        self.assertEqual(AUDIT.final_status(summary["passed"], []), "pass")

    def test_three_episode_conditions_still_use_six_condition_332_gate(self):
        self.assertEqual(tuple(fake_manifest()["planned_conditions"]), AUDIT.EXPECTED_PLANNED_CONDITIONS)
        names = AUDIT.import_discovery_helpers()[0].gate_names(
            AUDIT.EXPECTED_DISCOVERY_CONDITIONS, AUDIT.SUITES)
        self.assertEqual(len(names), 332)
        self.assertIn("libero_spatial/full_native", names)
        self.assertNotIn("libero_spatial/full_d10/actual_execution", names)
        self.assertIn("libero_spatial/local_long/actual_execution", names)

    def test_confirmation_marker_is_not_discovery_marker(self):
        self.assertEqual(AUDIT.CONFIRMATION_MARKER_NAME, "confirm_shard00of01_complete.json")
        self.assertNotEqual(AUDIT.CONFIRMATION_MARKER_NAME, "discovery_shard00of01_complete.json")

    def test_request_validator_import_is_read_only(self):
        module, path = AUDIT.import_discovery_helpers()[0].request_auditor()
        self.assertTrue(path.is_file())
        self.assertTrue(callable(module.audit_requests))


if __name__ == "__main__":
    unittest.main(verbosity=2)
