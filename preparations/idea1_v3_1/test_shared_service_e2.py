import copy
import unittest
from pathlib import Path
import tempfile

from .shared_service_e2 import client_commands,fixed_checks
from .common import read_manifest
from .common import atomic_json
from .audit_shared_e2 import verify,adoption_evidence


class SharedE2Tests(unittest.TestCase):
    def test_original_arms_workers_and_commands(self):
        m = read_manifest("results/idea1_v3_1/manifest_v2.json")
        self.assertEqual([v["id"] for v in m["e2"]["variants"]],["full_io","lean_io","reuse_env","two_workers"])
        for v in m["e2"]["variants"]:
            commands = client_commands("manifest","root",9700,v)
            self.assertEqual(len(commands),v["workers"])
            self.assertTrue(all("preparations.idea1_v3_1.rollout" in c and "--variant" in c and v["id"] in c for c in commands))
            self.assertTrue(all("--environment-reuse" not in c for c in commands))

    def reference(self):
        return [dict(input_hash=str(i),noise_hash="noise",seed=i,actions=[[0.]*7 for _ in range(10)]) for i in range(5)]

    def test_identical_controls_and_raw_not_claimed(self):
        r = self.reference()
        result = fixed_checks(r,copy.deepcopy(r))
        self.assertTrue(result["passed"])
        self.assertFalse(result["raw_actions_available"])

    def test_input_or_noise_or_shape_change_rejected(self):
        r = self.reference()
        for key,value in (("input_hash","changed"),("noise_hash","changed"),("actions",[[0.]*7])):
            other = copy.deepcopy(r)
            other[0][key] = value
            with self.assertRaises(AssertionError):
                fixed_checks(r,other)

    def test_numerical_drift_fails_without_tolerance_change(self):
        r = self.reference()
        other = copy.deepcopy(r)
        other[0]["actions"][0][0] = .001
        self.assertFalse(fixed_checks(r,other)["passed"])

    def test_independent_audit_rejects_incomplete_and_changed_source(self):
        manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        with tempfile.TemporaryDirectory() as root:
            p = Path(root)/"validation.json"
            atomic_json(p,dict(status="starting"))
            with self.assertRaises(AssertionError):
                verify(root,manifest,False)
            atomic_json(p,dict(status="completed",server_exit_code=0,manifest_hash=manifest["manifest_hash"],
                condition="pi05_full",helper_source_sha256="changed"))
            with self.assertRaises(AssertionError):
                verify(root,manifest,False)

    def test_adoption_nonvacuous_and_negative_outcomes_are_not_integrity_errors(self):
        def pair(matched=1,outcome=True):
            return dict(scene=["fixture",0,0],same_official_initial=True,same_outcome=outcome,matched_requests=matched,
                paired_action_checks=[dict(passed=True)]*matched,
                first_input_deltas=dict(robot0_eef_pos=dict(max_abs=0.,different_fraction=0.),agentview_image=dict(max_abs=0.,different_fraction=0.)))
        def report(p):
            return dict(fixed_controls_pass=True,comparisons=[dict(variant="two_workers",pairs=[p]*8)])
        self.assertTrue(adoption_evidence(report(pair()))[0]["evidence_supports_consideration"])
        self.assertFalse(adoption_evidence(report(pair(matched=0)))[0]["evidence_supports_consideration"])
        self.assertFalse(adoption_evidence(report(pair(outcome=False)))[0]["evidence_supports_consideration"])


if __name__ == "__main__":
    unittest.main()
