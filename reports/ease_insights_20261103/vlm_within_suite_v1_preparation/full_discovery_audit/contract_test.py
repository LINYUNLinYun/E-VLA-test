"""Small synthetic contracts for the future 432-case acceptance entry point."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit


class Frozen432Contracts(unittest.TestCase):
    def synthetic_stage(self, missing_one=False):
        scenes = audit.scenes()
        records = {}
        for condition in audit.CONDITIONS:
            for index, case in enumerate(scenes):
                # A valid policy failure is deliberately included in this fixture.
                records[condition, audit.key(case)] = {"success": not (condition == "full_d10" and index == 0)}
        missing = []
        if missing_one:
            records.pop((audit.CONDITIONS[-1], audit.key(scenes[-1])))
            missing = [dict(condition=audit.CONDITIONS[-1], case=scenes[-1])]
        return dict(expected=432, scenes=scenes, records=records, missing=missing,
                    invalid=[], unexpected=[], complete=not missing_one)

    def test_full_432_matrix_and_valid_false_outcome(self):
        stage = self.synthetic_stage()
        result = audit.stage_summary(stage)
        self.assertTrue(result["passed"], result["reasons"])
        self.assertEqual(result["valid"], 432)
        self.assertEqual(result["per_condition"], {condition: 72 for condition in audit.CONDITIONS})
        self.assertFalse(stage["records"]["full_d10", audit.key(stage["scenes"][0])]["success"])

    def test_incomplete_matrix_is_not_pass(self):
        result = audit.stage_summary(self.synthetic_stage(missing_one=True))
        self.assertFalse(result["passed"])
        self.assertIn("missing entries remain", result["reasons"])

    def test_completion_marker_requires_exactly_72(self):
        marker = dict(complete=True, manifest_hash=audit.MANIFEST_HASH,
                      condition="local_goal", completed_after=72)
        self.assertTrue(audit.marker_check(marker, "local_goal", audit.MANIFEST_HASH)["passed"])
        marker["completed_after"] = 71
        self.assertFalse(audit.marker_check(marker, "local_goal", audit.MANIFEST_HASH)["passed"])

    def test_raw_manifest_file_sha_pin_rejects_byte_mismatch(self):
        self.assertTrue(audit.raw_manifest_pin(audit.RAW_MANIFEST_SHA)["passed"])
        self.assertFalse(audit.raw_manifest_pin("0" * 64)["passed"])

    def test_completion_marker_requires_nonempty_in_scope_logs_path(self):
        root = Path("/synthetic/discovery")
        marker = {"logs": ""}
        self.assertFalse(audit.marker_run_path(marker, root, "full_d10")["passed"])
        marker["logs"] = str(root / "full_d10" / "runs" / "synthetic_run")
        self.assertTrue(audit.marker_run_path(marker, root, "full_d10")["passed"])
        marker["logs"] = str(root / "jac_anchor" / "runs" / "wrong_condition")
        self.assertFalse(audit.marker_run_path(marker, root, "full_d10")["passed"])

    def test_new_libero90_category_mapping_is_12_tasks_and_36_cases_each(self):
        manifest = dict(categories=list(audit.CATEGORIES), tasks=list(audit.TASKS),
                         task_categories={str(task): audit.CATEGORIES[0 if index < 12 else 1]
                                           for index, task in enumerate(audit.TASKS)})
        result = audit.category_summary(manifest, audit.scenes())
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(result["tasks_per_category"], {audit.CATEGORIES[0]: 12, audit.CATEGORIES[1]: 12})
        self.assertEqual(result["cases_per_category"], {audit.CATEGORIES[0]: 36, audit.CATEGORIES[1]: 36})

    def test_full_marker_verifier_propagates_missing_empty_and_wrong_condition_logs(self):
        root = Path("/synthetic/discovery")
        for invalid in (None, "", str(root / "jac_anchor" / "runs" / "wrong_condition")):
            with self.subTest(logs=invalid):
                markers = {cid: dict(complete=True, manifest_hash=audit.MANIFEST_HASH,
                                     condition=cid, completed_after=72,
                                     logs=str(root / cid / "runs" / "synthetic_run"))
                           for cid in audit.CONDITIONS}

                def read_artifact(path):
                    path = Path(path)
                    cid = path.parents[2].name if path.name == "result.json" else path.parent.name
                    return markers[cid]

                with patch.object(audit, "read", side_effect=read_artifact), patch.object(audit, "sha", return_value="0" * 64):
                    self.assertTrue(audit.verify_markers(root, {"manifest_hash": audit.MANIFEST_HASH})["passed"])
                    if invalid is None:
                        markers["full_d10"].pop("logs")
                    else:
                        markers["full_d10"]["logs"] = invalid
                    result = audit.verify_markers(root, {"manifest_hash": audit.MANIFEST_HASH})
                self.assertFalse(result["passed"])
                self.assertFalse(result["conditions"]["full_d10"]["passed"])
                self.assertEqual([row["condition"] for row in result["issues"]], ["full_d10"])

    def test_final_report_assembly_accepts_all_named_sections(self):
        report = audit.assemble_report(audit.MANIFEST_HASH, audit.DEFAULT_ROOT, 1.0, [], [], stage={"passed": True},
                                       episode_evidence={"receipt_count": 432}, serving_runs={"passed": True})
        self.assertEqual(report["manifest_hash"], audit.MANIFEST_HASH)
        self.assertEqual(report["stage"], {"passed": True})
        self.assertEqual(report["episode_evidence"]["receipt_count"], 432)
        self.assertEqual(report["serving_runs"], {"passed": True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
