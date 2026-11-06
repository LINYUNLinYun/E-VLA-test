"""Synthetic CPU checks; never reads the active confirmation case tree."""
import copy
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_rebuild", HERE / "rebuild.py")
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)


def accepted_report():
    return dict(revision="full-local-search-confirmation-read-only-acceptance-v1", status="pass",
        root=str(stats.CASE_ROOT), manifest_hash=stats.MANIFEST_HASH, manifest_file_sha256=stats.MANIFEST_SHA,
        failed_checks=[], stage=dict(passed=True, complete=True, expected=600, valid=600, scenes=200,
            planned_conditions=list(stats.CONDITIONS), init_ids=list(stats.INITS),
            per_condition={cid: 200 for cid in stats.CONDITIONS}, missing=[], missing_count=0,
            invalid=[], invalid_count=0, unexpected=[], unexpected_count=0, reasons=[]),
        checks=[dict(passed=True) for _ in range(4)], source_identity=dict(passed=True),
        frozen_selection=dict(passed=True, errors=[]), requests=dict(passed=True, cases_checked=600, failed_case_requests=[]),
        initial_input_pairing=dict(passed=True, cases_checked=200, failed_cases=[]),
        discovery_reference=dict(passed=True, issues=[]), serving_runs=dict(passed=True, issues=[]),
        scope=dict(cases_checked=600, gpu_used=False, scientific_selection_or_statistics=False))


def synthetic_records():
    records = {}
    for cid in stats.CONDITIONS:
        for scene in stats.cases():
            suite, task, init = scene
            # Every primary task has a +1 difference: winners 5/5, losers 4/5.
            success = True
            if cid == "local_long" and suite == "libero_object" and init == 39:
                success = False
            if cid == "jac_anchor" and suite == "libero_10" and init == 39:
                success = False
            records[cid, scene] = success
    return records


def synthetic_directions():
    return [dict(suite=suite, better=a, worse=b) for suite, a, b in stats.DIRECTIONS]


class FrozenContract(unittest.TestCase):
    def test_acceptance_complete_gate_and_zero_issues(self):
        good = accepted_report()
        stats.acceptance_gate(good)
        mutations = [("status", "incomplete"), ("manifest_hash", "0" * 64), ("failed_checks", [dict(issue="bad")])]
        for name, value in mutations:
            report = copy.deepcopy(good)
            report[name] = value
            with self.subTest(field=name), self.assertRaises(stats.GateError):
                stats.acceptance_gate(report)
        for name, value in (("valid", 599), ("missing_count", 1), ("invalid", ["bad"]), ("unexpected_count", 1)):
            report = copy.deepcopy(good)
            report["stage"][name] = value
            with self.subTest(stage=name), self.assertRaises(stats.GateError):
                stats.acceptance_gate(report)

    def test_incomplete_report_stops_before_any_outcome_or_manifest_open(self):
        incomplete = accepted_report()
        incomplete["status"] = "incomplete"
        source = dict(path="synthetic-acceptance", sha256="0" * 64)
        with mock.patch.object(stats, "read_bound_json", return_value=(incomplete, source)) as read_json, \
                mock.patch.object(stats, "load_outcomes") as load, mock.patch.object(stats, "source_identity") as sources:
            result = stats.run("0" * 64, "1" * 64)
        self.assertEqual(result["status"], "incomplete")
        self.assertFalse(result["raw_outcomes_opened"])
        self.assertEqual(read_json.call_count, 1)
        self.assertEqual(read_json.call_args.args[0], stats.ACCEPTANCE)
        load.assert_not_called()
        sources.assert_not_called()
        malformed = accepted_report()
        malformed["stage"] = []
        with mock.patch.object(stats, "read_bound_json", return_value=(malformed, source)):
            malformed_result = stats.run("0" * 64, "1" * 64)
        self.assertEqual(malformed_result["status"], "incomplete")
        self.assertFalse(malformed_result["raw_outcomes_opened"])

    def test_signflip_ties_and_exact_mcnemar(self):
        self.assertEqual(stats.sign_flip([0] * 10)["extreme_assignments"], 1024)
        self.assertEqual(stats.sign_flip([1] * 10)["extreme_assignments"], 1)
        tied = stats.sign_flip([1, -1] + [0] * 8)
        self.assertEqual(tied["extreme_assignments"], 768)
        values = [-2, 0, 3, 1, -1, 0, 2, -3, 1, 1]
        reference = sum(sum(s * v for s, v in zip(signs, values)) >= sum(values)
                        for signs in itertools.product((-1, 1), repeat=10))
        self.assertEqual(stats.sign_flip(values)["extreme_assignments"], reference)
        self.assertEqual(stats.binomial_tail(0, 0), 1.)
        self.assertEqual(stats.binomial_tail(10, 0), 1 / 1024)
        self.assertEqual(stats.binomial_tail(10, 0, two_sided=True), 1 / 512)
        self.assertEqual(stats.binomial_tail(3, 1), 5 / 16)
        self.assertEqual(stats.holm_two([.04, .01]), [.04, .02])
        self.assertEqual(stats.holm_two([.03, .03]), [.06, .06])

    def test_bootstrap_seed_order_quantile_and_golden_fixture(self):
        values = [1, 0, -1, 1, 1, 0, 0, -1, 0, 1] * 5
        result = stats.independent_bootstrap(values)
        # Standalone reference formulation of the frozen NumPy API contract.
        # It imports no scientific project helper and uses only synthetic data.
        np = stats.np
        sampled_case = np.random.default_rng(20261104).choice(np.array(values, dtype=np.float64), size=(20000, 50), replace=True).mean(axis=1)
        task_means = np.array([sum(values[t * 5:(t + 1) * 5]) / 5 for t in range(10)], dtype=np.float64)
        sampled_task = np.random.default_rng(20261104).choice(task_means, size=(20000, 10), replace=True).mean(axis=1)
        for key, draws in (("case", sampled_case), ("task_cluster", sampled_task)):
            self.assertEqual(result[key]["low"], float(np.quantile(draws, .025)))
            self.assertEqual(result[key]["high"], float(np.quantile(draws, .975)))
            self.assertEqual(result[key]["seed"], 20261104)
            self.assertEqual(result[key]["replicates"], 20000)
        self.assertAlmostEqual(result["case"]["low"], 0., places=12)
        self.assertAlmostEqual(result["case"]["high"], .4, places=12)
        self.assertAlmostEqual(result["task_cluster"]["low"], .08, places=12)
        self.assertAlmostEqual(result["task_cluster"]["high"], .32, places=12)

    def test_complete_synthetic_success_matrix_counts_and_verdict(self):
        report = stats.rebuild_statistics(synthetic_records(), synthetic_directions())
        self.assertEqual(len(report["condition_summaries"]), 135)
        self.assertEqual([r["successes"] for r in report["condition_summaries"][:3]], [200, 190, 190])
        self.assertEqual(report["conclusion"], "complementarity_reproduced")
        for row in report["primary_directions"]:
            self.assertEqual((row["n_paired"], row["a_only"], row["b_only"], row["both_success"], row["both_failure"]), (50, 10, 0, 40, 0))
            inf = row["inference"]
            self.assertEqual(inf["primary_task_block"]["task_difference_counts"], [1] * 10)
            self.assertEqual(inf["primary_holm2_p"], 1 / 512)
            self.assertEqual(inf["sensitivity_holm2_p"], 1 / 512)
            self.assertTrue(inf["direction_reproduced"])
            self.assertTrue(all(r["difference"] == .2 for r in inf["leave_one_task_out"]))
        self.assertEqual(stats.differences(report, copy.deepcopy(report)), [])
        altered = copy.deepcopy(report)
        altered["primary_directions"][1]["inference"]["primary_holm2_p"] = .9
        differences = stats.differences(report, altered)
        self.assertEqual(len(differences), 1)
        self.assertEqual(differences[0]["path"], "/primary_directions/1/inference/primary_holm2_p")
        missing = copy.deepcopy(report)
        del missing["condition_summaries"][0]["successes"]
        self.assertEqual(stats.differences(report, missing)[0]["issue"], "supervisor field missing")
        tie_records = {key: True for key in synthetic_records()}
        self.assertEqual(stats.rebuild_statistics(tie_records, synthetic_directions())["conclusion"], "not_reproduced")

    def test_usable_integer_boundaries_and_failures(self):
        report = stats.rebuild_statistics(synthetic_records(), synthetic_directions())
        self.assertTrue(all(g["usable"] for g in report["overall_candidate_eligibility"].values()))
        # Full=200 means the relative floor is exactly 170; 169 must fail.
        for count, expected in ((170, True), (169, False), (150, False)):
            records = synthetic_records()
            for index, scene in enumerate(stats.cases()):
                records["jac_anchor", scene] = index < count
            gate = stats.rebuild_statistics(records, synthetic_directions())["overall_candidate_eligibility"]["jac_anchor"]
            self.assertEqual(gate["usable"], expected)
        for count, expected in ((150, True), (149, False)):
            records = synthetic_records()
            for index, scene in enumerate(stats.cases()):
                records["full_d10", scene] = index < 160
                records["jac_anchor", scene] = index < count
            gate = stats.rebuild_statistics(records, synthetic_directions())["overall_candidate_eligibility"]["jac_anchor"]
            self.assertEqual(gate["usable"], expected)

    def test_supervisor_evidence_binds_the_same_raw_matrix(self):
        selected = dict(path="synthetic-selection.json", sha256=stats.SELECTION_SHA, selection_hash="frozen-selection")
        manifest = dict(selection_evidence=selected)
        paths = [stats.receipt_path(stats.CASE_ROOT, cid, scene).resolve() for cid in stats.CONDITIONS for scene in stats.cases()]
        accepted = {path: "0" * 64 for path in paths}
        report = dict(revision="ease-vlm-local-search-confirmation-analysis-v1", complete=True, expected_episodes=600,
            valid_episodes=600, missing=[], invalid=[], unexpected=[], manifest_hash=stats.MANIFEST_HASH,
            overall_candidate_eligibility={cid: {} for cid in stats.CONDITIONS[1:]},
            selection_hash="frozen-selection", analysis_protocol=dict(bootstrap_replicates=20000, bootstrap_seed=20261104,
            alpha=.05, no_reselection=True), evidence=dict(
            manifest=dict(path=str(stats.CASE_ROOT / "manifest.json"), sha256=stats.MANIFEST_SHA),
            selection=dict(path=str(stats.project_path(selected["path"])), sha256=stats.SELECTION_SHA),
            analysis_source=dict(path=str(stats.ROOT / stats.ANALYSIS_PATH), sha256=stats.ANALYSIS_SHA),
            episodes=[dict(path=str(path), sha256=sha) for path, sha in accepted.items()]))
        stats.supervisor_gate(report, manifest, accepted)
        for kind in ("hash", "missing", "seed", "incomplete", "extra_candidate"):
            changed = copy.deepcopy(report)
            if kind == "hash":
                changed["evidence"]["episodes"][0]["sha256"] = "1" * 64
            elif kind == "missing":
                changed["evidence"]["episodes"].pop()
            elif kind == "seed":
                changed["analysis_protocol"]["bootstrap_seed"] = 1
            elif kind == "incomplete":
                changed["complete"] = False
            else:
                changed["overall_candidate_eligibility"]["extra_path"] = {}
            with self.subTest(kind=kind), self.assertRaises(stats.GateError):
                stats.supervisor_gate(changed, manifest, accepted)
    def test_synthetic_600_raw_receipt_result_hash_binding(self):
        records = synthetic_records()
        manifest = dict(conditions=[dict(id=cid) for cid in stats.CONDITIONS])
        accepted = {}
        # All writes stay in the assigned statistics directory and are removed.
        with tempfile.TemporaryDirectory(prefix="synthetic_", dir=HERE) as temp:
            root = Path(temp)
            for (cid, scene), success in records.items():
                path = stats.receipt_path(root, cid, scene)
                path.parent.mkdir(parents=True)
                case = dict(zip(("suite", "task", "init"), scene))
                payload = dict(valid=True, manifest_hash=stats.MANIFEST_HASH, condition=cid, case=case, success=success)
                result = dict(payload, config=dict(id=cid))
                (path.parent / "result.json").write_text(json.dumps(result))
                receipt = dict(payload, artifacts={"result.json": stats.file_sha(path.parent / "result.json")})
                path.write_text(json.dumps(receipt))
                accepted[path.resolve()] = stats.file_sha(path)
            recovered, evidence = stats.load_outcomes(root, manifest, accepted)
            self.assertEqual(recovered, records)
            self.assertEqual(len(evidence), 600)
            expected_paths = list(accepted)
            rows = [dict(path=str(p), sha256=sha) for p, sha in accepted.items()]
            self.assertEqual(stats.evidence_map(rows, expected_paths, "synthetic"), accepted)
            with self.assertRaises(stats.GateError):
                stats.evidence_map(rows + rows[:1], expected_paths, "synthetic")
            path = expected_paths[0]
            original_receipt = path.read_text()
            original_result = (path.parent / "result.json").read_text()
            (path.parent / "result.json").write_text(original_result + " ")
            with self.assertRaises(stats.GateError):
                stats.load_outcomes(root, manifest, accepted)
            (path.parent / "result.json").write_text(original_result)
            path.write_text(original_receipt + " ")
            with self.assertRaises(stats.GateError):
                stats.load_outcomes(root, manifest, accepted)
            # Even if a malformed result's hash is bound, success must be bool.
            result = json.loads(original_result)
            result["success"] = 1
            (path.parent / "result.json").write_text(json.dumps(result))
            receipt = json.loads(original_receipt)
            receipt["artifacts"]["result.json"] = stats.file_sha(path.parent / "result.json")
            path.write_text(json.dumps(receipt))
            accepted[path] = stats.file_sha(path)
            with self.assertRaises(stats.GateError):
                stats.load_outcomes(root, manifest, accepted)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FrozenContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = dict(status="pass" if result.wasSuccessful() else "failed", tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), active_case_tree_read=False,
        fixture="synthetic only; 600 tiny receipt/result pairs under assigned directory",
        gpu_used=False, model_or_environment_loaded=False,
        numpy_version=stats.np.__version__, rebuild_sha256=stats.file_sha(HERE / "rebuild.py"),
        test_sha256=stats.file_sha(__file__))
    (HERE / "contract_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
