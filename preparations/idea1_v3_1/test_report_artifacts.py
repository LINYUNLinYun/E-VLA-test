"""Synthetic CPU tests; no live dataset reports, simulators or GPUs."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from urllib.parse import unquote

import numpy as np

from .common import atomic_json,read_manifest
from .report_artifacts import DISTANCE_CAVEAT,audit_documents,audit_links,build,collect,distances,plot_quality,quality_tables,trajectory


class ReportArtifactTests(unittest.TestCase):
    def fixture(self,root):
        manifest = dict(manifest_hash="fixture",suites=["suite_a","suite_b","suite_c","suite_d"],tasks=[0],
            screen_inits=[0],confirm_inits=[1,2,3,4],benchmark_references=[],conditions=[
                dict(id="pi0_full",model="pi0",skip=[],keep=[1,2],layout="full"),
                dict(id="pi0_mask",model="pi0",skip=[2],keep=[1],layout="last")])
        mappings = [dict(condition=c["id"],model=c["model"],actual_directory=str(root/c["id"])) for c in manifest["conditions"]]
        return manifest,dict(manifest_hash="fixture",conditions=mappings)

    def episode(self,root,cid,suite,init=0,success=True,phase="approach",video=True,xyz=None,times=None):
        path = root/cid/"episodes"/suite/"task00"/f"init{init:02d}"
        path.mkdir(parents=True)
        atomic_json(path/"result.json",dict(valid=True,success=success,condition=cid,manifest_hash="fixture",suite=suite,
            task=0,init=init,init_vector_sha256=f"official-{suite}-{init}",furthest_phase=phase))
        atomic_json(path/"complete.json",dict(valid=True,manifest_hash="fixture"))
        if xyz is not None:
            with (path/"telemetry.jsonl").open("x") as stream:
                stream.write("".join(json.dumps(dict(step=i,sim_time_s=t,eef_pos=p))+"\n" for i,(p,t) in enumerate(zip(xyz,times))))
        if video:
            (path/"video.mp4").touch()
        return path

    def test_known_xyz_mm_same_step_vs_actual_time_no_padding(self):
        full = dict(xyz=np.array([[0,0,0],[.003,.004,0],[.006,.008,0]]),times=np.array([0.,.05,.1]),steps=np.array([0,1,2]))
        variant = dict(xyz=np.array([[.003,.004,0],[.006,.008,0]]),times=np.array([.05,.1]),steps=np.array([0,1]))
        step,timed,coverage = distances(full,variant)
        self.assertEqual([r["distance_mm"] for r in step],[5.,5.])
        self.assertEqual([r["distance_mm"] for r in timed],[0.,0.])
        self.assertEqual([(r["variant_step"],r["full_step"]) for r in timed],[(0,1),(1,2)])
        self.assertEqual(coverage["common_start_s"],.05)
        self.assertEqual(coverage["same_step_points"],2)
        self.assertAlmostEqual(coverage["variant_duration_s"],.05)
        self.assertIn(DISTANCE_CAVEAT,coverage["interpretation"])
        variant["times"] = np.array([.2,.25])
        self.assertEqual(distances(full,variant)[1],[])
        self.assertIsNone(distances(full,variant)[2]["common_end_s"])

    def test_explicit_alias_directory_ignores_invalid_and_unrelated_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest,index = self.fixture(root)
            actual = root/"real condition & video"
            row = self.episode(root,"pi0_full","suite_a")
            (root/"pi0_full").rename(actual)
            (root/"pi0_full").symlink_to(actual,target_is_directory=True)
            invalid = actual/"episodes"/"suite_a"/"task00"/"init01.invalid.fixture"
            invalid.mkdir()
            atomic_json(invalid/"result.json",dict(untrusted=True))
            defs,mappings,data = collect(index,manifest,{})
            self.assertEqual(len(data["pi0_full"]),1)
            self.assertEqual(data["pi0_full"][("suite_a",0,0)]["condition"],"pi0_full")
            index["conditions"].append(index["conditions"][0])
            with self.assertRaises(AssertionError):
                collect(index,manifest,{})

    def test_partial_matching_full_phase_proportions_and_separate_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest,index = self.fixture(root)
            self.episode(root,"pi0_full","suite_a",success=True)
            self.episode(root,"pi0_mask","suite_a",success=False,phase="contact")
            self.episode(root,"pi0_mask","suite_b",success=False,phase="approach")
            self.episode(root,"pi0_mask","suite_c",init=1,success=True)
            defs,_,data = collect(index,manifest,{})
            quality,phases = quality_tables(manifest,defs,data)
            screen = next(r for r in quality if r["condition"]=="pi0_mask" and r["suite"]=="ALL" and r["scope"]=="screen_init0")
            self.assertEqual((screen["successes"],screen["denominator"],screen["expected"]),(0,2,4))
            self.assertEqual(screen["matched_full_denominator"],1)
            self.assertEqual(screen["exact_matching_full_retention"],0.)
            self.assertFalse(screen["complete"])
            self.assertFalse(screen["complete_matching_full"])
            combined = next(r for r in quality if r["condition"]=="pi0_mask" and r["suite"]=="ALL" and r["scope"]=="combined_init0_4")
            self.assertEqual((combined["successes"],combined["denominator"],combined["expected"]),(1,3,20))
            observed = [r for r in phases if r["condition"]=="pi0_mask" and r["suite"]=="ALL" and r["scope"]=="screen_init0"]
            self.assertEqual({r["progress_phase"]:r["proportion"] for r in observed},{"approach":.5,"contact":.5})
            self.assertTrue(all("PROGRESS" in r["interpretation"] and "not failure cause" in r["interpretation"] for r in observed))

    def test_missing_paired_full_produces_issue_not_fabricated_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest,index = self.fixture(root)
            self.episode(root,"pi0_mask","suite_a",xyz=[[1,2,3]],times=[0.])
            result = build(index,manifest,root/"report",["pi0_mask"],plots=False)
            self.assertEqual(len(result["media"]),0)
            self.assertEqual(len(result["missing_evidence"]),4)
            self.assertTrue(result["completion_is_not_implied_by_report"])
            self.assertEqual(result["quality_complete_conditions"]["screen_init0"],0)
            self.assertFalse(list((root/"report").glob("*_same_step_mm.csv")))

    def test_media_escaping_link_audit_hashes_and_actual_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"data & media (fixture)"
            root.mkdir()
            manifest,index = self.fixture(root)
            for suite in manifest["suites"]:
                self.episode(root,"pi0_full",suite,xyz=[[0,0,0],[.003,.004,0]],times=[0.,.05])
                self.episode(root,"pi0_mask",suite,xyz=[[.003,.004,0],[.006,.008,0]],times=[0.,.05])
            output = Path(directory)/"report"
            result = build(index,manifest,output,["pi0_mask"],plots=True)
            self.assertEqual(len(result["media"]),4)
            self.assertEqual(result["missing_evidence"],[])
            self.assertTrue(any("%26" in link and "%28" in link for link in result["relative_links_checked"]))
            for item in result["media"]:
                self.assertEqual(set(item["links"]),{"Full video","Variant video","Full telemetry","Variant telemetry","XYZ and distance PNG"})
                self.assertEqual(item["coverage"]["nearest_actual_time_points"],2)
            self.assertEqual(audit_links(output,result["relative_links_checked"]),result["relative_links_checked"])
            self.assertEqual(audit_documents(output),result["relative_links_checked"])
            self.assertIn("artifact_index.json",[p.name for p in output.iterdir()])
            with self.assertRaises(FileExistsError):
                build(index,manifest,output,["pi0_mask"],plots=False)
            with self.assertRaises(AssertionError):
                audit_links(output,["missing.mp4"])

    def test_wrong_scene_manifest_and_nonfinite_telemetry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest,index = self.fixture(root)
            path = self.episode(root,"pi0_full","suite_a",xyz=[[0,0,float("nan")]],times=[0.])
            with self.assertRaises(AssertionError):
                trajectory(path/"telemetry.jsonl",{})
            row = json.loads((path/"result.json").read_text())
            row["manifest_hash"] = "wrong"
            atomic_json(path/"result.json",row)
            with self.assertRaises(AssertionError):
                collect(index,manifest,{})

    def test_benchmark_only_not_quality_91_conditions_not_93(self):
        manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        definitions = {c["id"]:c for c in manifest["conditions"]+manifest["benchmark_references"]}
        self.assertEqual(len(definitions),93)
        self.assertEqual(len(manifest["conditions"]),91)
        keys = [(s,t,i) for s in manifest["suites"] for t in manifest["tasks"] for i in [0,1,2,3,4]]
        data = {cid:{key:dict(success=not c.get("benchmark_only",False),init_vector_sha256=str(key),furthest_phase="fixture")
            for key in keys} for cid,c in definitions.items()}
        quality,phases = quality_tables(manifest,definitions,data)
        excluded = {c["id"] for c in manifest["benchmark_references"]}
        self.assertTrue(all(r["condition"] not in excluded for r in quality+phases))
        for scope,expected in (("screen_init0",40),("combined_init0_4",200)):
            all_rows = [r for r in quality if r["scope"]==scope and r["suite"]=="ALL"]
            self.assertEqual(len(all_rows),91)
            self.assertEqual(sum(r["complete"] for r in all_rows),91)
            self.assertTrue(all(r["expected"]==expected for r in all_rows))

    def test_empty_conditions_stay_in_csv_but_create_no_quality_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest,index = self.fixture(root)
            definitions,mappings,data = collect(index,manifest,{})
            quality,_ = quality_tables(manifest,definitions,data)
            self.assertEqual(len(quality),20)
            self.assertEqual(plot_quality(root,quality),[])
            self.episode(root,"pi0_full","suite_a")
            definitions,mappings,data = collect(index,manifest,{})
            quality,_ = quality_tables(manifest,definitions,data)
            paths = plot_quality(root,quality)
            self.assertEqual(len(paths),2)
            self.assertTrue(all("budget0" in p.name for p in paths))
            self.assertTrue(any(r["condition"]=="pi0_mask" and r["denominator"]==0 for r in quality))


if __name__=="__main__":
    unittest.main()
