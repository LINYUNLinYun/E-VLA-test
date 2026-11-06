"""Synthetic screen data; no live dispatches, models, launches or GPU queries."""
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from . import controller
from .common import read_manifest
from .confirmation_dispatch import main,plan,verify,write_new


class ConfirmationDispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"view"
        self.root.mkdir()
        self.manifest = read_manifest("results/idea1_v3_1/manifest_v2.json",verify_sources=True)
        fulls = [c for c in self.manifest["conditions"] if not c["skip"]]
        self.groups = [[c for c in self.manifest["conditions"] if c["model"]=="pi0" and len(c["skip"])==b] for b in (3,6)]
        reference = copy.deepcopy(self.groups[0][0])
        reference.update(id="pi0_historical_reference",reference=True)
        self.reference = reference
        self.manifest["conditions"] = fulls+self.groups[0]+self.groups[1]+[reference]
        self.conditions = {c["id"]:c for c in self.manifest["conditions"]}
        for c in fulls:
            self.complete(c,34)

    def tearDown(self):
        self.temp.cleanup()

    def complete(self,condition,score,inits=(0,)):
        for init in inits:
            for index,(suite,task) in enumerate((s,t) for s in self.manifest["suites"] for t in self.manifest["tasks"]):
                path = self.root/condition["id"]/"episodes"/suite/f"task{task:02d}"/f"init{init:02d}"
                path.mkdir(parents=True,exist_ok=True)
                common = dict(suite=suite,task=task,init=init,manifest_hash=self.manifest["manifest_hash"])
                result = dict(common,condition=condition["id"],valid=True,success=index<score,init_vector_sha256=f"{suite}/{task}/{init}")
                identity = dict(common,condition=condition,init_vector_sha256=result["init_vector_sha256"])
                for name,row in (("result.json",result),("identity.json",identity),("complete.json",dict(common,valid=True))):
                    (path/name).write_text(json.dumps(row))

    def first_path(self,cid,name="result.json"):
        return self.root/cid/"episodes"/self.manifest["suites"][0]/"task00"/"init00"/name

    def close_first(self):
        for condition,score in zip(self.groups[0],(36,35,34,1,38)):
            self.complete(condition,score)
        return plan(self.manifest,self.root)

    def test_progressive_union_exact_frozen_promotion_four_of_five_not_closed(self):
        for condition,score in zip(self.groups[0][:4],(36,35,34,1)):
            self.complete(condition,score)
        self.complete(self.reference,40)
        early = plan(self.manifest,self.root)
        self.assertEqual(early["closed_budgets"],[])
        self.assertEqual(set(early["selected_conditions"]),{m+"_full" for m in ("pi0","pi05","xiaomi-robotic0","db-cogact")})
        self.complete(self.groups[0][4],38)
        first = plan(self.manifest,self.root)
        self.assertEqual(first["closed_budgets"][0]["decision"]["best"],self.groups[0][4]["id"])
        self.assertNotIn(self.reference["id"],first["selected_conditions"])
        for condition,score in zip(self.groups[1],(10,20,30,2,15)):
            self.complete(condition,score)
        final = plan(self.manifest,self.root)
        self.assertEqual(verify(first,self.manifest,self.root)["passed"],True)
        self.assertEqual(first["closed_budgets"][0],final["closed_budgets"][0])
        union = set(early["selected_conditions"])|set(first["selected_conditions"])|set(final["selected_conditions"])
        self.assertEqual(union,set(controller.promotion(self.manifest,self.root)["selected_conditions"]))
        self.assertFalse(final["global_promotion"])

    def test_confirmation_extra160_inits_leave_screen_choice_and_hashes_identical(self):
        receipt = self.close_first()
        winner = receipt["closed_budgets"][0]["decision"]["best"]
        for cid in ("pi0_full",winner):
            self.complete(self.conditions[cid],3,inits=(1,2,3,4))
        self.assertEqual(plan(self.manifest,self.root),receipt)
        self.assertTrue(verify(receipt,self.manifest,self.root)["passed"])

    def test_hash_full_baseline_source_and_foreign_real_path_rejected(self):
        receipt = self.close_first()
        altered = copy.deepcopy(receipt)
        altered["sources"]["helper_sha256"] = "changed"
        with self.assertRaises(AssertionError):
            verify(altered,self.manifest,self.root)
        path = self.first_path("pi0_full")
        original = path.read_text()
        row = json.loads(original)
        row["note"] = "same scientific score, changed artifact"
        path.write_text(json.dumps(row))
        with self.assertRaises(AssertionError):
            verify(receipt,self.manifest,self.root)
        path.write_text(original)
        old = self.root/"pi0_full"
        foreign = Path(self.temp.name)/"foreign_full"
        shutil.copytree(old,foreign)
        old.rename(self.root/"saved_original_full")
        old.symlink_to(foreign,target_is_directory=True)
        with self.assertRaises(AssertionError):
            verify(receipt,self.manifest,self.root)

    def test_wrong_scene_invalid_marker_and_initial_identity_rejected(self):
        receipt = self.close_first()
        marker = self.first_path("pi0_full","complete.json")
        old = marker.read_text()
        row = json.loads(old)
        row["valid"] = False
        marker.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            verify(receipt,self.manifest,self.root)
        marker.write_text(old)
        candidate = self.groups[0][0]["id"]
        path = self.first_path(candidate)
        row = json.loads(path.read_text())
        row["init_vector_sha256"] = "foreign-initial"
        path.write_text(json.dumps(row))
        with self.assertRaises(AssertionError):
            plan(self.manifest,self.root)
        self.complete(self.conditions[candidate],36)
        source = self.first_path(candidate).parent
        unexpected = source.parent.parent/"task99"/"init00"
        shutil.copytree(source,unexpected)
        for name in ("result.json","identity.json"):
            p = unexpected/name
            row = json.loads(p.read_text())
            row["task"] = 99
            p.write_text(json.dumps(row))
        with self.assertRaises(ValueError):
            plan(self.manifest,self.root)

    def test_append_only_not_global_and_dryrun_no_files_or_resources(self):
        result = plan(self.manifest,self.root)
        output = Path(self.temp.name)/"dispatch.json"
        write_new(output,result)
        with self.assertRaises(FileExistsError):
            write_new(output,result)
        with self.assertRaises(ValueError):
            write_new(Path(self.temp.name)/"promotion.json",result)
        dry = Path(self.temp.name)/"never-created.json"
        with patch("sys.argv",["dispatch","--manifest","fake","--root",str(self.root),"--output",str(dry)]), \
             patch("preparations.idea1_v3_1.confirmation_dispatch.read_manifest",return_value=self.manifest), \
             patch("subprocess.Popen",side_effect=AssertionError("launch")), \
             patch("subprocess.check_output",side_effect=AssertionError("resource query")),patch("sys.stdout",io.StringIO()):
            main()
        self.assertFalse(dry.exists())

    def test_progressive_union_against_actual_all91_frozen_promotion(self):
        manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        conditions = manifest["conditions"]
        self.assertEqual(len(conditions),91)
        scores = {c["id"]:(40 if c["reference"] else 34 if not c["skip"] else 20+i%15) for i,c in enumerate(conditions)}
        complete = {c["id"] for c in conditions if not c["skip"]}
        complete.update(c["id"] for c in conditions if c["model"]=="pi0" and len(c["skip"])==3)
        def records(root,cid,inits=(0,),manifest_hash=None):
            return [dict(success=i<scores[cid]) for i in range(40)] if cid in complete else []
        def evidence(root,c,m):
            return dict(condition=c["id"],initials={"fixed40":"same official keys"},files=[])
        with patch.object(controller,"completed",side_effect=records), \
             patch("preparations.idea1_v3_1.confirmation_dispatch.queue.completed_quality",side_effect=lambda root,c,m,stage:c["id"] in complete), \
             patch("preparations.idea1_v3_1.confirmation_dispatch.screen_evidence",side_effect=evidence):
            early = plan(manifest,self.root)
            complete.update(c["id"] for c in conditions)
            final = plan(manifest,self.root)
            frozen = controller.promotion(manifest,self.root)
            union = set(early["selected_conditions"])|set(final["selected_conditions"])
            self.assertEqual(union,set(frozen["selected_conditions"]))
            self.assertEqual(early["closed_budgets"][0],final["closed_budgets"][0])
            self.assertFalse(any(c["id"] in union for c in conditions if c["reference"]))


if __name__=="__main__":
    unittest.main()
