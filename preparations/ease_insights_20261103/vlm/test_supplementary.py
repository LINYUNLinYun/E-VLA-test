"""Small CPU-only fixtures; synthetic outcomes are never research evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import analyze_supplementary as supplementary
from .analyze import key
from .common import SUITES, conditions, digest, sha


class SupplementaryTests(unittest.TestCase):
    def setUp(self):
        self.ids = [c["id"] for c in conditions()[:3]]
        self.scenes = [dict(suite=s, task=t, init=i) for s in SUITES for t in range(10) for i in range(31, 36)]

    def fixture(self, folder):
        root = folder / "confirmation"
        root.mkdir()
        directions = [dict(**{k: spec[k] for k in ("suite", "better", "worse")}, discovery_difference=.1)
            for spec in supplementary.FAMILY[:2]]
        selection = dict(complete_discovery_episodes=160, independent_validation_launched=False,
            selection=dict(status="selected", selected=dict(pair=self.ids[1:], directions=directions)))
        selection["selection_hash"] = digest(selection)
        selected = folder / "selection.json"
        selected.write_text(json.dumps(selection))
        manifest = dict(phase="independent_confirmation", confirmation_output=str(root),
            discovery_reference=dict(root=str(folder / "discovery")), conditions=conditions(),
            planned_conditions=self.ids, suites=list(SUITES), tasks=list(range(10)), confirmation_inits=list(range(31, 36)),
            total_confirmation_episodes=600, source_files={}, checkpoints={}, preregistered_hypotheses=directions,
            selection_evidence=dict(path=str(selected), sha256=sha(selected), selection_hash=selection["selection_hash"], frozen_selection=selection["selection"]),
            official_init_availability=dict(cases=[dict(suite=s, task=t, init_sha256={str(i): "fixture-init" for i in range(31, 36)}) for s in SUITES for t in range(10)]))
        manifest["manifest_hash"] = digest(manifest)
        (root / "manifest.json").write_text(json.dumps(manifest))
        registration = dict(version="prospective-supplemental-goal-v1", registered_unix=1000.,
            family=copy.deepcopy(supplementary.FAMILY), original_primary_directions=directions,
            supplementary_direction={k: supplementary.FAMILY[2][k] for k in ("suite", "better", "worse")},
            confirmation_manifest_sha256=sha(root / "manifest.json"), confirmation_manifest_hash=manifest["manifest_hash"],
            selection_sha256=sha(selected), discovery_inits=[30], validation_inits=list(range(31, 36)),
            new_gpu_episodes=0, planned_matrix_unchanged=True, model_or_runtime_changed=False,
            original_frozen_selection_changed=False, heldout_outcomes_read=False, heldout_evidence_files_at_registration=[])
        registered = root / "supplementary_goal_contrast.json"
        registered.write_text(json.dumps(registration))
        case, cfg = self.scenes[0], conditions()[0]
        directory = root / cfg["id"] / "episodes" / case["suite"] / "task00/init31"
        directory.mkdir(parents=True)
        identity = dict(manifest_hash=manifest["manifest_hash"], case=case, config=cfg, init_vector_sha256="fixture-init",
            protocol=dict(manifest_hash=manifest["manifest_hash"], condition_id=cfg["id"], real_input_parity_passed=True, gpu_uuid="GPU-fixture"))
        result = dict(valid=True, manifest_hash=manifest["manifest_hash"], case=case, condition=cfg["id"], config=cfg,
            success=True, predictions=1, actions=5, counts=dict(full=1, draft=0), wall_s=1.)
        for name, obj in (("identity.json", identity), ("result.json", result),
            ("complete.json", dict(valid=True, manifest_hash=manifest["manifest_hash"]))):
            (directory / name).write_text(json.dumps(obj))
        os.utime(directory / "identity.json", (1100., 1100.))
        (directory / "infer.jsonl").write_text(json.dumps(dict(route=dict(refresh=True), executed_length=5,
            timing=dict(policy_wall_ms=1.), rpc_ms=2.)) + "\n")
        (directory / "telemetry.jsonl").write_text("{}\n")
        (directory / "arrays.npz").write_bytes(b"opaque synthetic fixture")
        receipt = dict(valid=True, manifest_hash=manifest["manifest_hash"], case=case, condition=cfg["id"], success=True,
            gpu_uuid="GPU-fixture", initial_gate=dict(passed=True, manifest_hash=manifest["manifest_hash"],
                case=case, input_hash="paired-fixture", seed=1),
            artifacts={name: sha(directory / name) for name in ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz")})
        (directory / "episode_receipt.json").write_text(json.dumps(receipt))
        return root, manifest, registered, directory

    def records(self):
        records = {}
        for cid in self.ids:
            for case in self.scenes:
                # Original Spatial direction fails; fixed Goal and Long succeed.
                success = cid == "full_d10" or (cid == "fixed_v12_jacobian" if case["suite"] == "libero_goal" else cid == "fixed_v12_action_mse")
                records[cid, key(case)] = dict(condition=cid, case=case, success=success, actions=5, predictions=1,
                    full_requests=1, fast_requests=0, policy_ms=[1.], full_policy_ms=[1.], fast_policy_ms=[], rpc_ms=[2.],
                    episode_wall_s=1., full_executed_actions=5, fast_executed_actions=0, gpu_uuid="GPU-fixture")
        return records

    def test_holm_three_fixed_family(self):
        self.assertEqual(supplementary.primary.holm([.04, .01, .03]), [.06, .03, .06])
        self.assertEqual([row["suite"] for row in supplementary.FAMILY], ["libero_spatial", "libero_10", "libero_goal"])

    def test_partial_is_only_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, registered, _ = self.fixture(Path(folder))
            with patch.object(supplementary, "REGISTRATION_SHA256", sha(registered)):
                report = supplementary.analyze(root, replicates=100)
            self.assertEqual((report["status"], report["valid_episodes"]), ("incomplete", 1))
            self.assertEqual(report["original_primary_analysis"]["conclusion"], "incomplete")
            self.assertEqual(report["supplementary_goal_long_analysis"]["conclusion"], "incomplete")
            self.assertTrue(all(row["inference"] is None and row["exact_mcnemar_two_sided"] is None for row in report["contrasts"]))
            self.assertTrue(report["prospective_registration"]["accepted"])

    def test_full_fixed_directions_keep_primary_failure_and_separate_holm(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, registered, _ = self.fixture(Path(folder))
            with patch.object(supplementary, "REGISTRATION_SHA256", sha(registered)), \
                patch.object(supplementary, "load_episodes", return_value=(self.records(), [], [], [])), \
                patch.object(supplementary.primary, "audit_identities", return_value=[]):
                report = supplementary.analyze(root, replicates=100)
            self.assertTrue(report["complete"])
            self.assertEqual(report["original_primary_analysis"]["conclusion"], "both_prespecified_directions_not_reproduced")
            self.assertEqual(report["supplementary_goal_long_analysis"]["conclusion"], "supports_complementary_fixed_paths_on_the_two_prespecified_suites")
            self.assertEqual([row["inference"]["exact_mcnemar_one_sided"] for row in report["contrasts"]], [1., 2**-50, 2**-50])
            self.assertEqual([row["inference"]["holm_adjusted_p"] for row in report["contrasts"]], [1., 3 * 2**-50, 3 * 2**-50])
            self.assertEqual(report["original_primary_analysis"]["primary_directions"][1]["inference"]["holm_adjusted_p"], 2**-49)
            for row in report["contrasts"]:
                self.assertEqual((len(row["by_task"]), len(row["by_init"]), len(row["by_case"])), (10, 5, 50))
                self.assertEqual(len(row["inference"]["leave_one_task_out"]), 10)
                self.assertEqual(len(row["inference"]["leave_one_init_out"]), 5)
            self.assertIn("Original Spatial/Long conclusion: both_prespecified_directions_not_reproduced", supplementary.markdown(report))

    def test_one_missing_full_episode_blocks_all_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, registered, _ = self.fixture(Path(folder))
            records = self.records()
            del records["full_d10", key(self.scenes[-1])]
            with patch.object(supplementary, "REGISTRATION_SHA256", sha(registered)), \
                patch.object(supplementary, "load_episodes", return_value=(records, [dict(condition="full_d10")], [], [])), \
                patch.object(supplementary.primary, "audit_identities", return_value=[]):
                report = supplementary.analyze(root, replicates=100)
            self.assertEqual(report["valid_episodes"], 599)
            self.assertFalse(report["complete"])
            self.assertTrue(all(row["n_paired"] == 50 and row["inference"] is None for row in report["contrasts"]))

    def test_registration_or_bound_evidence_drift_rejected(self):
        for drift in ("registration", "manifest", "selection", "direction"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as folder:
                root, manifest, registered, _ = self.fixture(Path(folder))
                pinned = sha(registered)
                target = {"registration": registered, "manifest": root / "manifest.json",
                    "selection": Path(manifest["selection_evidence"]["path"]), "direction": registered}[drift]
                if drift == "direction":
                    obj = json.loads(registered.read_text())
                    obj["family"][2]["better"], obj["family"][2]["worse"] = obj["family"][2]["worse"], obj["family"][2]["better"]
                    registered.write_text(json.dumps(obj))
                    pinned = sha(registered)  # Still reject an internally changed direction.
                else:
                    target.write_text(target.read_text() + "\n")
                with patch.object(supplementary, "REGISTRATION_SHA256", pinned), self.assertRaises(ValueError):
                    supplementary.analyze(root, replicates=100)

    def test_earliest_retry_identity_prevents_prospective_label(self):
        with tempfile.TemporaryDirectory() as folder:
            root, manifest, registered, directory = self.fixture(Path(folder))
            retry = directory.with_name("init31.attempt0")
            retry.mkdir()
            (retry / "identity.json").write_bytes((directory / "identity.json").read_bytes())
            os.utime(retry / "identity.json", (900., 900.))
            with patch.object(supplementary, "REGISTRATION_SHA256", sha(registered)), \
                patch.object(supplementary, "load_episodes", return_value=(self.records(), [], [], [])), \
                patch.object(supplementary.primary, "audit_identities", return_value=[]):
                report = supplementary.analyze(root, replicates=100)
            self.assertFalse(report["prospective_registration"]["accepted"])
            self.assertEqual(report["prospective_registration"]["first_original_heldout_identity"]["raw_identity_mtime_unix"], 900.)
            self.assertFalse(report["supplementary_goal_long_analysis"]["prospective_label"])
            self.assertEqual(report["supplementary_goal_long_analysis"]["conclusion"], "prospective_label_refused; descriptive_only")
            self.assertEqual(report["original_primary_analysis"]["conclusion"], "both_prespecified_directions_not_reproduced")

    def test_original_identity_audit_cannot_be_bypassed_with_updated_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root, _, registered, directory = self.fixture(Path(folder))
            identity = json.loads((directory / "identity.json").read_text())
            identity["config"] = conditions()[1]
            (directory / "identity.json").write_text(json.dumps(identity))
            receipt = json.loads((directory / "episode_receipt.json").read_text())
            receipt["artifacts"]["identity.json"] = sha(directory / "identity.json")
            (directory / "episode_receipt.json").write_text(json.dumps(receipt))
            with patch.object(supplementary, "REGISTRATION_SHA256", sha(registered)):
                report = supplementary.analyze(root, replicates=100)
            self.assertEqual(report["valid_episodes"], 0)
            self.assertTrue(report["invalid"])
            self.assertFalse(report["complete"])


if __name__ == "__main__":
    unittest.main()
