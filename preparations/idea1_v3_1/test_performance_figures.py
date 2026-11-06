"""Synthetic saved-audit chains and CPU Agg rendering; no live plot generation."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from .common import read_manifest
from .performance_figures import ARMS,MODELS,TIMERS,build,e2_values,vision_values
from .same_process_parity import sha


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))
    return sha(path)


def summary(values):
    return {m:{p:float(v) for p in ("p50","p95","p99")} for m,v in zip(TIMERS,values)}


class PerformanceFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")
        self.definitions = {c["id"]:c for c in self.manifest["conditions"]+self.manifest["benchmark_references"]}
        self.e2,self.vision = self.fixtures()
    def tearDown(self):
        self.temp.cleanup()

    def fixtures(self,unstable=False):
        identity = self.manifest["manifest_hash"]
        e2 = dict(models=[],scope="Previously audited synthetic fixture")
        for model in MODELS:
            directory = self.root/"e2"/model
            walls = dict(zip(ARMS,(100.,80.,70.,50.)))
            speeds = {arm:100./v for arm,v in walls.items()}
            validation = dict(manifest_hash=identity,condition=model+"_full",status="completed",server_exit_code=0,warmup=20,
                protocol=dict(server_pid=123,condition_id=model+"_full"),server_pid=123,gpu="0",
                arms=[dict(variant=a,wall_s=walls[a]) for a in ARMS])
            val_sha = save(directory/"validation.json",validation)
            audit = dict(passed=True,video_verified=True,manifest_hash=identity,counts={a:8 for a in ARMS},
                summary=dict(complete_episodes=32,fixed_controls_pass=True),fixed_control_max_abs=0.,
                validation_receipt_sha256=val_sha,validation_receipt=str(directory/"validation.json"),speedups=speeds,
                adoption_evidence=[dict(variant=a,evidence_supports_consideration=True,matched_scenes=8,matched_requests_total=32) for a in ARMS[1:]])
            audit_sha = save(directory/"audit.json",audit)
            e2["models"].append(dict(model=model,audit_path=str(directory/"audit.json"),audit_sha256=audit_sha,validation_sha256=val_sha,
                eight_episode_wall_s=walls,speedup_vs_full_io=speeds,fixed_control_max_abs=0.,paired_requests={a:32 for a in ARMS[1:]},
                two_workers_speedup_vs_lean_io=1.6,two_workers_speedup_vs_reuse_env=1.4,lean_io_time_reduction_fraction=.2))
        vision = dict(manifest_hash=identity,models=[],scientific_scope="Already audited independent timers")
        for model in ("pi0","pi05"):
            directory = self.root/"vision"/model
            candidates = [next(c for c in self.manifest["conditions"] if c["model"]==model and len(c["skip"])==b and not c["reference"])["id"] for b in (3,6,9,12)]
            roles = [("native_full_before",model+"_full",summary((100,98,12))),
                ("matched_all27_full",model+"_full_vision_scan_reference",summary((105,103,14)))]+[
                ("candidate",cid,summary((101-b*.1,99-b*.1,12-b*.2))) for cid,b in zip(candidates,(3,6,9,12))]+[
                ("native_full_after",model+"_full",summary((140,136,17) if unstable and model=="pi0" else (102,100,12.2)))]
            gpu = dict(uuid="GPU-test-same-card",name="fixture",assigned_gpu="0")
            banks = [dict(path="fixture-bank-not-reread",sha256="prior-audit-bank-sha",observations=100)]
            phases,executions = [],[]
            for i,(role,cid,stats) in enumerate(roles):
                path = directory/f"phase{i}"/cid/"benchmark.json"
                benchmark = dict(manifest_hash=identity,condition=self.definitions[cid],banks=banks,warmup=20,samples=100,summary=stats,
                    rows=[{m:stats[m]["p50"] for m in TIMERS} for _ in range(100)])
                digest = save(path,benchmark)
                phases.append(dict(condition=cid,role=role,summary=stats,benchmark_path=str(path),artifacts={"benchmark.json":digest}))
                executions.append(dict(condition=cid,role=role,gpu_before=gpu,gpu_after=gpu,artifact_hashes={"benchmark.json":digest}))
            cohort = dict(manifest_hash=identity,model=model,status="completed",require_four_budgets=True,gpu_identity=gpu,phases=executions)
            cohort_sha = save(directory/"cohort.json",cohort)
            before,after,matched = roles[0][2],roles[-1][2],roles[1][2]
            drift = {m:abs(after[m]["p50"]/before[m]["p50"]-1) for m in TIMERS}
            flags = {m:v>.1 for m,v in drift.items()}
            stable = not any(flags[m] for m in ("request_roundtrip_ms","vision_ms"))
            comparison = dict(primary_statistic="P50",primary_drift_metrics=["request_roundtrip_ms","vision_ms"],unstable=not stable,
                drift_threshold=.1,native_full_after_p50_drift_fraction=drift,drift_exceeds_threshold_by_metric=flags,candidates=[
                    dict(condition=cid,native_before_p50_speedup={m:before[m]["p50"]/stats[m]["p50"] for m in TIMERS},
                        matched_full_p50_speedup={m:matched[m]["p50"]/stats[m]["p50"] for m in TIMERS}) for role,cid,stats in roles if role=="candidate"])
            audit = dict(passed=True,stable_for_primary_comparison=stable,manifest_hash=identity,gpu_identity=gpu,final_vision_cohort=True,
                vision_budget_coverage=dict(observed=[3,6,9,12],required=[3,6,9,12],complete=True),comparison=comparison,
                cohort_path=str(directory),cohort_receipt_sha256=cohort_sha,banks=banks,phases=phases)
            digest = save(directory/"audit.json",audit)
            vision["models"].append(dict(model=model,source=str(directory/"audit.json"),source_sha256=digest,gpu=gpu,stable=stable,
                comparison=comparison,native_before=before,native_after=after))
        return e2,vision

    def test_hand_known_e2_ratios_and_authenticated_source_numbers(self):
        rows,increments,proofs = e2_values(self.e2,self.manifest,{})
        self.assertEqual(len(rows),16)
        self.assertEqual(next(r["speedup_vs_full_io"] for r in rows if r["model"]=="pi0" and r["arm"]=="two_workers"),2.)
        self.assertEqual(increments[0]["two_workers_vs_lean_io"],1.6)
        self.assertEqual(increments[0]["two_workers_vs_reuse_env"],1.4)
        self.assertIn("legacy",proofs[0]["gpu"]["uuid"])
        changed = copy.deepcopy(self.e2)
        changed["models"][0]["speedup_vs_full_io"]["two_workers"] = 9.
        with self.assertRaises(AssertionError):
            e2_values(changed,self.manifest,{})

    def test_native_matched_ratio_arithmetic_and_exact_four_budgets(self):
        rows,proofs = vision_values(self.vision,self.manifest,{})
        self.assertEqual(len(rows),24)
        row = next(r for r in rows if r["model"]=="pi0" and r["skip"]==3 and r["metric"]=="request_roundtrip_ms")
        self.assertAlmostEqual(row["native_before_p50_speedup"],100/100.7)
        self.assertAlmostEqual(row["matched_full_p50_speedup"],105/100.7)
        self.assertEqual({r["skip"] for r in rows},{3,6,9,12})
        self.assertTrue(all(p["stable"] for p in proofs))

    def test_changed_audit_hash_manifest_and_incomplete_budget_rejected(self):
        changed = copy.deepcopy(self.e2)
        changed["models"][0]["audit_sha256"] = "forged"
        with self.assertRaises(AssertionError):
            e2_values(changed,self.manifest,{})
        wrong = dict(self.manifest,manifest_hash="foreign")
        with self.assertRaises(AssertionError):
            vision_values(self.vision,wrong,{})
        record = self.vision["models"][0]
        path = Path(record["source"])
        audit = json.loads(path.read_text())
        audit["vision_budget_coverage"]["complete"] = False
        record["source_sha256"] = save(path,audit)
        with self.assertRaises(AssertionError):
            vision_values(self.vision,self.manifest,{})

    def test_source_ratio_tampering_and_stability_marker_mismatch_rejected(self):
        record = self.vision["models"][0]
        path = Path(record["source"])
        audit = json.loads(path.read_text())
        audit["comparison"]["candidates"][0]["native_before_p50_speedup"]["vision_ms"] = 999.
        record["comparison"] = audit["comparison"]
        record["source_sha256"] = save(path,audit)
        with self.assertRaises(AssertionError):
            vision_values(self.vision,self.manifest,{})
        e2,vision = self.fixtures()
        vision["models"][0]["stable"] = False
        with self.assertRaises(AssertionError):
            vision_values(vision,self.manifest,{})

    def test_unstable_scientific_result_retained_not_discarded(self):
        e2,vision = self.fixtures(unstable=True)
        rows,proofs = vision_values(vision,self.manifest,{})
        self.assertFalse(next(p["stable"] for p in proofs if p["model"]=="pi0"))
        self.assertEqual(len(rows),24)
        self.assertTrue(all(not r["stable"] for r in rows if r["model"]=="pi0"))
        self.assertTrue(all(r["stable"] for r in rows if r["model"]=="pi05"))

    def test_cpu_agg_new_artifacts_csv_precision_and_no_fresh_audit_claim(self):
        e2,vision = self.fixtures(unstable=True)
        e2_path,vision_path = self.root/"e2-findings.json",self.root/"performance-findings.json"
        save(e2_path,e2)
        save(vision_path,vision)
        output = self.root/"new-figures"
        result = build(e2_path,vision_path,self.manifest,output)
        self.assertEqual(len(result["artifacts_sha256"]),5)
        self.assertEqual(result["unstable_cohorts"],["pi0"])
        self.assertFalse(result["normal_gain_conclusion_authorized"])
        self.assertIn("NOT a new raw audit",result["validation_scope"])
        self.assertTrue((output/"plot_index.json").is_file())
        for name,digest in result["artifacts_sha256"].items():
            self.assertEqual(sha(output/name),digest)
        with (output/"vision_native_and_matched_raw_ratios.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        first = next(r for r in rows if r["model"]=="pi0" and r["skip"]=="3" and r["metric"]=="request_roundtrip_ms")
        self.assertEqual(float(first["native_before_p50_speedup"]),100/100.7)
        with self.assertRaises(FileExistsError):
            build(e2_path,vision_path,self.manifest,output)


if __name__=="__main__":
    unittest.main()
