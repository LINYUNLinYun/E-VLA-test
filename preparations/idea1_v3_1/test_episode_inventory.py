"""Mock-provenance CPU inventory fixtures; no real all-episode generation."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .dataset_bundle_v3 import quality_rows
from .episode_inventory import build,check_links,inventory
from .same_process_parity import sha


def save(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(row))


class EpisodeInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"data & (saved)"
        self.original,self.adopted = self.root/"original",self.root/"adopted"
        full = dict(id="pi0_full",model="pi0",skip=[],keep=[1,2],horizon=10,execute_length=5,reference=False)
        mask = dict(full,id="pi0_mask",skip=[2],keep=[1])
        benchmark = dict(full,id="pi0_benchmark_only",benchmark_only=True)
        self.manifest = dict(manifest_hash="fixture",suites=["suite_a","suite_b","suite_c","suite_d"],tasks=list(range(10)),
            screen_inits=[0],confirm_inits=[0,1,2,3,4],conditions=[full,mask],benchmark_references=[benchmark],
            e2=dict(episodes=[dict(suite=s,task=0,init=i) for s in ("suite_a","suite_b","suite_c","suite_d") for i in (0,1)],
                variants=[dict(id=a) for a in ("full_io","lean_io","reuse_env","two_workers")]))
        self.promotion = dict(manifest_hash="fixture",selected_conditions=[full["id"]])
        for c in (full,mask):
            for suite in self.manifest["suites"]:
                for task in self.manifest["tasks"]:
                    for init in self.manifest["confirm_inits"] if c==full else [0]:
                        self.episode(self.original/c["id"]/"episodes",c,suite,task,init,video=task==0 and init==0)
        for root in (self.adopted,self.original):
            for arm in ("full_io","lean_io","reuse_env","two_workers"):
                for scene in self.manifest["e2"]["episodes"]:
                    self.episode(root/full["id"]/arm,full,**scene,video=arm=="full_io")
        alias = self.root/"explicit alias"/full["id"]
        alias.parent.mkdir(parents=True)
        alias.symlink_to(self.original/full["id"],target_is_directory=True)
        mappings = []
        for c in (full,mask,benchmark):
            directory = alias if c==full else self.original/c["id"]
            mappings.append(dict(condition=c["id"],model=c["model"],actual_directory=str(directory),episode_source=str(directory/"episodes")))
        origin = dict(root=str(self.adopted))
        for field in ("registration","validation","audit"):
            path = self.adopted/(field+".json")
            save(path,dict(manifest_hash="fixture",scope="Synthetic provenance; not scientific evidence"))
            origin[field+"_path"],origin[field+"_sha256"] = str(path),sha(path)
        self.index = dict(manifest_hash="fixture",original_root=str(self.original),conditions=mappings,profile_origins=dict(pi0=origin))
        def collect(index,manifest):
            definitions = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
            return {r["condition"]:quality_rows(r["episode_source"],definitions[r["condition"]],manifest) for r in index["conditions"]}
        # Only the existing dataset/provenance guard is mocked. The production
        # quality_rows collector and ALL new episode identity/link checks run.
        self.collect = patch("preparations.idea1_v3_1.episode_inventory.collect_quality",side_effect=collect)
        self.collect.start()
    def tearDown(self):
        self.collect.stop()
        self.temp.cleanup()

    def episode(self,directory,condition,suite,task,init,video=False):
        path = directory/suite/f"task{task:02d}"/f"init{init:02d}"
        common = dict(manifest_hash="fixture",suite=suite,task=task,init=init)
        save(path/"result.json",dict(common,valid=True,condition=condition["id"],model=condition["model"],success=task%2==0,
            init_vector_sha256=f"{suite}/{task}/{init}",actions=5,predictions=1,wall_s=.12345678901234568,video=video))
        save(path/"identity.json",dict(common,condition=condition,init_vector_sha256=f"{suite}/{task}/{init}"))
        save(path/"complete.json",dict(common,valid=True,video_decoded=False))
        with (path/"telemetry.jsonl").open("x") as stream:
            for step in range(6):
                stream.write(json.dumps(dict(step=step,eef_pos=[.12345678901234568+step*1e-9,.2,.3],sim_time_s=step*.05))+"\n")
        if video:
            (path/"video.mp4").write_bytes(b"Recorded fixture bytes; intentionally not decoded")
        return path

    def test_expected200_vs40_benchmark_exclusion_and_archive_denominators(self):
        rows,counts = inventory(self.index,self.manifest,self.promotion,self.root/"not-created")
        self.assertEqual(counts["expected_quality_by_condition"],dict(pi0_full=200,pi0_mask=40))
        self.assertEqual(counts["counts_by_category"],dict(quality=240,adopted_e2_profile=32,archived_not_adopted=32))
        self.assertEqual(len(rows),304)
        self.assertEqual(len({r["actual_episode_directory"] for r in rows}),304)
        self.assertFalse(any(r["condition"]=="pi0_benchmark_only" for r in rows))
        self.assertTrue(any(not r["video_recorded"] and r["xyz_raw_url"] for r in rows))
        self.assertTrue(counts["archived_not_adopted_excluded_from_quality_and_adopted_denominators"])

    def test_resolved_alias_dedup_invalid_and_unfinished_exclusion(self):
        old = self.original/"pi0_full"/"full_io"
        old.rename(self.root/"preserved old profile outside original scan")
        old.symlink_to(self.adopted/"pi0_full"/"full_io",target_is_directory=True)
        bad = self.original/"pi0_full"/"lean_io"/"suite_a"/"task00"/"init00.invalid.fixture"
        save(bad/"result.json",dict(garbage=True))
        save(bad/"complete.json",dict(garbage=True))
        unfinished = self.original/"pi0_full"/"lean_io"/"suite_a"/"task00"/"init99"
        save(unfinished/"result.json",dict(garbage=True))
        rows,counts = inventory(self.index,self.manifest,self.promotion,self.root/"not-created")
        self.assertEqual(counts["counts_by_category"]["archived_not_adopted"],24)
        self.assertEqual(len(rows),296)

    def test_identity_actions_missing_telemetry_and_partial_quality_fail(self):
        path = self.original/"pi0_full"/"episodes"/"suite_a"/"task00"/"init00"
        result = json.loads((path/"result.json").read_text())
        for field,value in (("success",1),("predictions",2),("actions",5.0)):
            altered = dict(result,**{field:value})
            save(path/"result.json",altered)
            with self.assertRaises(AssertionError):
                inventory(self.index,self.manifest,self.promotion,self.root/"never")
        save(path/"result.json",result)
        (path/"telemetry.jsonl").rename(path/"kept-telemetry")
        with self.assertRaises(AssertionError):
            inventory(self.index,self.manifest,self.promotion,self.root/"never")
        (path/"kept-telemetry").rename(path/"telemetry.jsonl")
        (path/"complete.json").rename(path/"kept-completion")
        with self.assertRaises(AssertionError):
            inventory(self.index,self.manifest,self.promotion,self.root/"never")

    def test_new_inventory_full_precision_links_hashes_html_video_preload_none(self):
        index_path,promotion_path = self.root/"index.json",self.root/"promotion.json"
        save(index_path,self.index)
        save(promotion_path,self.promotion)
        output = Path(self.temp.name)/"new inventory"
        result = build(index_path,promotion_path,self.manifest,output)
        self.assertFalse(result["scientific_audit_performed"])
        self.assertFalse(result["video_decode_audit_performed"])
        self.assertEqual(len(result["artifacts_sha256"]),4)
        data = json.loads((output/"episodes.json").read_text())
        self.assertEqual(len(data["episodes"]),304)
        with (output/"episodes.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows),304)
        self.assertEqual(float(rows[0]["first_eef_x_m"]),.12345678901234568)
        self.assertEqual(float(rows[0]["wall_s"]),.12345678901234568)
        self.assertTrue(any("%26" in r["xyz_raw_url"] and "%28" in r["xyz_raw_url"] for r in rows))
        document = (output/"videos.html").read_text()
        recorded = sum(r["video_recorded"] for r in data["episodes"])
        self.assertEqual(document.count('preload="none"'),recorded)
        self.assertIn("archived_not_adopted",document)
        for name,digest in result["artifacts_sha256"].items():
            self.assertEqual(sha(output/name),digest)
        self.assertTrue(all(r["telemetry_sha256"] and r["result_sha256"] for r in rows))
        with self.assertRaises(FileExistsError):
            build(index_path,promotion_path,self.manifest,output)
        with self.assertRaises(AssertionError):
            check_links(output,["missing.mp4"])

    def test_source_provenance_hash_and_selection_identity_rejected(self):
        index_path,promotion_path = self.root/"index.json",self.root/"promotion.json"
        altered = copy.deepcopy(self.index)
        altered["profile_origins"]["pi0"]["audit_sha256"] = "changed"
        save(index_path,altered)
        save(promotion_path,self.promotion)
        output = self.root/"never-created"
        with self.assertRaises(AssertionError):
            build(index_path,promotion_path,self.manifest,output)
        self.assertFalse(output.exists())
        wrong = dict(self.promotion,manifest_hash="foreign")
        with self.assertRaises(AssertionError):
            inventory(self.index,self.manifest,wrong,output)


if __name__=="__main__":
    unittest.main()
