"""Synthetic contracts for the unused frozen confirmation360 audit."""
import sys
import copy
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit


class Confirmation360Contracts(unittest.TestCase):
    def manifest(self):
        payload = dict(phase="within_suite_confirmation_v1", suites=["libero_90"],
            tasks=list(audit.TASKS), categories=list(audit.CATEGORIES), task_categories=dict(audit.TASK_CATEGORIES),
            confirmation_inits=list(audit.INITS), planned_conditions=["full_d10", "jac_anchor", "local_goal"],
            conditions=[{"id": cid} for cid in audit.CONDITIONS_ALL], cases_per_condition=120,
            total_confirmation_episodes=360,
            case_table=[dict(suite="libero_90", task=t) for t in audit.TASKS],
            science={"policy_action_limit": 400, "settling_actions": 10})
        payload["manifest_hash"] = audit.canonical_manifest_hash(payload)
        return payload

    def test_exact_360_universe_and_valid_false_outcome(self):
        stage_cases = audit.cases()
        records = {(cid, audit.key(case)): {"success": True}
                   for cid in ("full_d10", "jac_anchor", "local_goal") for case in stage_cases}
        records["full_d10", audit.key(stage_cases[0])]["success"] = False
        stage = dict(expected=360, records=records, scenes=stage_cases, missing=[], invalid=[], unexpected=[], complete=True)
        result = audit.stage_summary(stage, ["full_d10", "jac_anchor", "local_goal"])
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(result["per_condition"], {"full_d10": 120, "jac_anchor": 120, "local_goal": 120})

    def test_incomplete_360_matrix_is_nonpass(self):
        stage_cases = audit.cases()
        records = {(cid, audit.key(case)): {} for cid in ("full_d10", "jac_anchor", "local_goal") for case in stage_cases}
        records.pop(("local_goal", audit.key(stage_cases[-1])))
        stage = dict(expected=360, records=records, scenes=stage_cases, missing=[{}], invalid=[], unexpected=[], complete=False)
        self.assertFalse(audit.stage_summary(stage, ["full_d10", "jac_anchor", "local_goal"])["passed"])

    def test_confirmation_matrix_and_explicit_raw_manifest_pin(self):
        manifest = self.manifest()
        self.assertTrue(audit.confirmation_matrix(manifest, manifest["manifest_hash"], "a" * 64)["passed"])
        self.assertFalse(audit.raw_manifest_pin("0" * 64, "a" * 64)["passed"])
        bad = dict(manifest, total_confirmation_episodes=359)
        self.assertFalse(audit.confirmation_matrix(bad, manifest["manifest_hash"], "a" * 64)["passed"])

    def test_frozen_pair_and_opposite_category_directions_bind_without_ranking(self):
        manifest = self.manifest()
        protocol, design = {"path": "protocol", "sha256": "p"}, {"path": "design", "sha256": "d"}
        manifest.update(protocol_evidence=protocol, design_evidence=design)
        choice = {"status": "selected", "selected": {"paths": ["jac_anchor", "local_goal"],
            "pair": ["jac_anchor", "local_goal"], "directions": [
            {"category": audit.CATEGORIES[0], "better": "jac_anchor", "worse": "local_goal"},
            {"category": audit.CATEGORIES[1], "better": "local_goal", "worse": "jac_anchor"}]}}
        selection = {"revision": "ease-vlm-within-suite-selection-v1", "discovery_manifest_hash": audit.DISCOVERY_HASH,
            "complete_discovery_episodes": 432, "expected_discovery_episodes": 432,
            "independent_validation_launched": False, "protocol_evidence": protocol, "design_evidence": design,
            "selection": choice, "evidence": {"synthetic": True}}
        selection["selection_hash"] = audit.digest(selection)
        manifest["planned_conditions"] = ["full_d10", *choice["selected"]["paths"]]
        manifest["preregistered_hypotheses"] = choice["selected"]["directions"]
        manifest["selection_evidence"] = {"selection_hash": selection["selection_hash"],
            "frozen_selection": choice, "evidence": selection["evidence"]}
        self.assertTrue(audit.selection_contract(manifest, {"manifest_hash": audit.DISCOVERY_HASH}, selection)["passed"])
        manifest["preregistered_hypotheses"] = []
        self.assertFalse(audit.selection_contract(manifest, {"manifest_hash": audit.DISCOVERY_HASH}, selection)["passed"])

    def test_confirmation_categories_are_60_cases_each(self):
        result = audit.category_summary(self.manifest(), audit.cases())
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(result["cases_per_category"], {audit.CATEGORIES[0]: 60, audit.CATEGORIES[1]: 60})

    def test_marker_requires_exact_120_and_nonempty_condition_run_path(self):
        marker = {"complete": True, "manifest_hash": "m", "condition": "jac_anchor", "completed_after": 120}
        self.assertTrue(audit.marker_check(marker, "jac_anchor", "m")["passed"])
        self.assertFalse(audit.marker_check(dict(marker, completed_after=119), "jac_anchor", "m")["passed"])
        self.assertFalse(audit.marker_run_path(dict(logs=""), "/synthetic/root", "jac_anchor")["passed"])
        inside = "/synthetic/root/jac_anchor/runs/run1"
        self.assertTrue(audit.marker_run_path(dict(logs=inside), "/synthetic/root", "jac_anchor")["passed"])

    def test_report_assembly_has_no_duplicate_named_sections(self):
        report = audit.assemble_report("m", "raw", audit.DEFAULT_ROOT, 1.0, [], [],
                                       frozen_binding={"passed": True}, serving_runs={"passed": True})
        self.assertEqual(report["frozen_binding"], {"passed": True})
        self.assertEqual(report["serving_runs"], {"passed": True})

    def resource_sources(self):
        discovery = {name:{name + '/original.py':'a'*64} for name in
                     ('source_files','analysis_sources','local_runtime_sources')}
        additions = {name:'b'*64 for name in audit.RESERVOIR_SOURCES}
        manifest = copy.deepcopy(discovery)
        for name in ('source_files','local_runtime_sources'):
            manifest[name].update(additions)
        return manifest, discovery, additions

    def test_only_exact_resource_additions_preserve_all_original_pins(self):
        manifest, discovery, additions = self.resource_sources()
        before = copy.deepcopy(manifest)
        result = audit.resource_source_extension_contract(manifest, discovery, additions)
        self.assertTrue(result['passed'], result['issues'])
        self.assertEqual(manifest, before)
        self.assertEqual(result['base_projection']['source_files'], discovery['source_files'])

    def test_missing_or_additional_resource_module_is_rejected(self):
        for mode in ('missing','extra'):
            manifest, discovery, additions = self.resource_sources()
            if mode == 'missing':
                manifest['source_files'].pop(audit.RESERVOIR_SOURCES[0])
            else:
                manifest['local_runtime_sources']['unapproved.py']='c'*64
            self.assertFalse(audit.resource_source_extension_contract(manifest, discovery, additions)['passed'])

    def test_changed_original_science_source_is_not_hidden_by_resource_projection(self):
        manifest, discovery, additions = self.resource_sources()
        manifest['analysis_sources']['analysis_sources/original.py']='c'*64
        self.assertFalse(audit.resource_source_extension_contract(manifest, discovery, additions)['passed'])

    def test_resource_hash_differs_or_unknown_addition_set_is_rejected(self):
        manifest, discovery, additions = self.resource_sources()
        manifest['source_files'][audit.RESERVOIR_SOURCES[0]]='c'*64
        self.assertFalse(audit.resource_source_extension_contract(manifest, discovery, additions)['passed'])
        manifest, discovery, additions = self.resource_sources()
        additions['other.py']='d'*64
        self.assertFalse(audit.resource_source_extension_contract(manifest, discovery, additions)['passed'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
