import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .common import atomic_json,read_manifest
from .same_process_parity import sha
from .dataset_bundle_v3 import plan,collect,summarize,create_view,profile_provenance,audit


class BundleV3Tests(unittest.TestCase):
    def setUp(self):
        self.m = read_manifest("results/idea1_v3_1/manifest_v2.json")
        self.conditions = [c for c in self.m["conditions"] if c["model"]=="pi0"][:3]

    def profiles(self,root):
        root = Path(root)
        variants = self.m["e2"]["variants"]
        validation = dict(manifest_hash=self.m["manifest_hash"],condition="pi0_full",arms=[dict(variant=v["id"],wall_s=10+i) for i,v in enumerate(variants)],
            helper_source_sha256=sha(Path(__file__).with_name("shared_service_e2_generic.py")))
        audit = dict(passed=True,manifest_hash=self.m["manifest_hash"],counts={v["id"]:8 for v in variants},
            adoption_evidence=[dict(variant=v["id"],evidence_supports_consideration=True) for v in variants if v["id"]!="full_io"])
        atomic_json(root/"validation.json",validation)
        audit.update(validation_receipt=str(root/"validation.json"),validation_receipt_sha256=sha(root/"validation.json"),
            audit_source_sha256=sha(Path(__file__).with_name("audit_shared_e2_generic.py")))
        atomic_json(root/"audit.json",audit)
        provenance = dict(validation_path=str(root/"validation.json"),validation_sha256=sha(root/"validation.json"),audit_path=str(root/"audit.json"),audit_sha256=sha(root/"audit.json"),
            registration_source_sha256=sha(Path(__file__).with_name("register_shared_e2_generic.py")))
        targets = []
        for arm,v in zip(validation["arms"],variants):
            target = root/"pi0_full"/f"throughput_profile_{v['id']}.json"
            atomic_json(target,dict(manifest_hash=self.m["manifest_hash"],condition="pi0_full",provenance=provenance,
                source_kind="derived_from_authenticated_shared_service_validation",wall_s=arm["wall_s"],workers=v["workers"],variant=v["id"],
                newly_completed_episodes=8,completed_before=0,completed_after=8))
            targets.append(str(target))
        atomic_json(root/"shared_registration.json",dict(manifest_hash=self.m["manifest_hash"],provenance=provenance,derived_timing_targets=targets,benchmark_aliases=[]))

    def episodes(self,root,c,count=40,successes=30):
        for index in range(count):
            suite,task = self.m["suites"][index//10],index%10
            p = Path(root)/c["id"]/"episodes"/suite/f"task{task:02d}"/"init00"
            atomic_json(p/"complete.json",dict(valid=True,manifest_hash=self.m["manifest_hash"]))
            atomic_json(p/"result.json",dict(valid=True,manifest_hash=self.m["manifest_hash"],condition=c["id"],model=c["model"],
                suite=suite,task=task,init=0,success=index<successes,furthest_phase="approach"))

    def test_completed_old_pending_new_no_dropped_or_duplicate_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            old,new = Path(directory)/"old",Path(directory)/"new"
            self.profiles(new)
            self.episodes(old,self.conditions[0],successes=36)
            self.episodes(old,self.conditions[1],successes=30)
            index = plan(self.m,old,{"pi0":str(new)})
            by_id = {r["condition"]:r for r in index["conditions"]}
            self.assertEqual(by_id[self.conditions[0]["id"]]["actual_directory"],str(old/self.conditions[0]["id"]))
            self.assertEqual(by_id[self.conditions[2]["id"]]["actual_directory"],str(new/self.conditions[2]["id"]))
            self.assertEqual({a["kind"] for a in index["resume_alias_plan"]},{"completed condition directory alias","completed episodes alias; preserve E2 arms/benchmark in runtime condition directory"})
            self.assertFalse((new/"pi0_full/episodes").exists())
            bundle = Path(directory)/"bundle"
            create_view(index,bundle,self.m)
            self.assertFalse((new/"pi0_full/episodes").exists())  # Never activate real resume mappings.
            self.assertEqual(sum(map(len,collect(index,self.m).values())),80)
            # Simulate a separately authorized resume alias, then confirm no
            # recursion/drop/duplication even when a whole condition is aliased.
            (new/"pi0_full/episodes").symlink_to(old/"pi0_full/episodes",target_is_directory=True)
            (new/self.conditions[1]["id"]).symlink_to(old/self.conditions[1]["id"],target_is_directory=True)
            self.assertEqual(sum(map(len,collect(index,self.m).values())),80)
            self.episodes(new,self.conditions[2],successes=25)
            self.assertEqual(sum(map(len,collect(index,self.m).values())),120)
            self.assertEqual(index["profile_origins"]["pi0"]["registration_sha256"],sha(new/"shared_registration.json"))

    def test_partial_boundary_and_distinct_duplicate_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            old,new = Path(directory)/"old",Path(directory)/"new"
            self.profiles(new)
            self.episodes(old,self.conditions[0],count=2)
            with self.assertRaisesRegex(ValueError,"boundary"):
                plan(self.m,old,{"pi0":str(new)})
            self.episodes(old,self.conditions[0])
            self.episodes(new,self.conditions[0])
            with self.assertRaisesRegex(ValueError,"Split/duplicate"):
                plan(self.m,old,{"pi0":str(new)})

    def test_frozen_quality_formula_and_derived_timing_tamper(self):
        from .analyze import summarize as frozen
        with tempfile.TemporaryDirectory() as directory:
            old,new = Path(directory)/"old",Path(directory)/"new"
            self.profiles(new)
            self.episodes(old,self.conditions[0],successes=36)
            self.episodes(old,self.conditions[1],successes=25)
            index = plan(self.m,old,{"pi0":str(new)})
            actual = summarize(index,self.m)["quality"]
            expected = frozen(self.m,old)["quality"]
            for a,b in zip(actual,expected,strict=True):
                self.assertEqual({k:v for k,v in a.items() if k not in ("actual_directory","episode_source")},b)
            target = new/"pi0_full/throughput_profile_two_workers.json"
            row = json.loads(target.read_text())
            atomic_json(target,dict(row,wall_s=1.))
            with self.assertRaises(AssertionError):
                profile_provenance(new,"pi0",self.m)

    def test_audit_keyword_binding_original_gate_paths_and_creation_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)/"bundle"
            bundle.mkdir()
            original = Path(directory)/"original"
            index = dict(conditions=[],profile_origins={},original_root=str(original),source_sha256="retained-creation-version")
            atomic_json(bundle/"index.json",index)
            before = sha(bundle/"index.json")
            captured = {}
            def scientific(manifest,root,gate_paths=None,check_videos=True,analysis_report=None):
                self.assertIsInstance(gate_paths,dict)
                self.assertIs(check_videos,False)
                captured.update(gate_paths=gate_paths,check_videos=check_videos,root=root,analysis_report=analysis_report)
                return dict(errors=[],missing_requirements=[])
            with patch("preparations.idea1_v3_1.dataset_bundle_v3.validate"), \
                 patch("preparations.idea1_v3_1.dataset_bundle_v3.collect",return_value={}), \
                 patch("preparations.idea1_v3_1.final_audit_v2.audit",side_effect=scientific):
                report = audit(index,bundle,self.m,check_videos=False)
            self.assertEqual(captured["gate_paths"],{m:str(original/f"{m}_vision_same_process_parity.json") for m in ("pi0","pi05")})
            self.assertEqual(report["dataset_creation_source_sha256"],"retained-creation-version")
            self.assertEqual(report["dataset_source_sha256"],sha(Path(__file__).with_name("dataset_bundle_v3.py")))
            self.assertEqual(sha(bundle/"index.json"),before)
            self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
