import copy
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch
import contextlib

import numpy as np

from .common import ROOT,atomic_json,read_manifest
from .same_process_parity import sha
from . import performance_cohort as cohort


class CohortTests(unittest.TestCase):
    def setUp(self):
        self.m = read_manifest("results/idea1_v3_1/manifest_v2.json")
        self.manifest_path = str((ROOT/"results/idea1_v3_1/manifest_v2.json").resolve())
        self.candidate = next(c for c in self.m["conditions"] if c["model"]=="pi0" and len(c["skip"])==3)["id"]

    def inputs(self,root):
        bank = Path(root)/"bank.pkl"
        records = [dict(obs={"state":np.asarray([i],np.float32)},instruction="fixture",suite=self.m["suites"][i%4],seed=i+17) for i in range(100)]
        bank.write_bytes(pickle.dumps(records))
        gate = Path(root)/"original_gate.json"
        atomic_json(gate,dict(fixture="CPU gate verifier is mocked; no model/GPU loaded"))
        return bank,gate

    def planning(self,root,launch=False):
        bank,gate = self.inputs(root)
        with patch.object(cohort,"authenticate_vision_gate",return_value=dict(passed=True)):
            p = cohort.make_plan(self.m,self.manifest_path,Path(root)/"cohort","pi0",[self.candidate],[str(bank)],gpu="0",launch=launch,vision_parity=str(gate))
        return p,bank,gate

    def phase_files(self,p,phase,after_vision=3.):
        conditions = {c["id"]:c for c in self.m["conditions"]+self.m["benchmark_references"]}
        c = conditions[phase["condition"]]
        protocol = dict(c,manifest_hash=self.m["manifest_hash"],condition_id=c["id"],server_pid=5000+phase["index"],
            real_input_parity_passed=False,H=c["horizon"],D=c["denoising_steps"],R=5,vision_depth=27,vision_tokens=256,vision_compression="off",
            checkpoint=self.m["checkpoints"]["pi0"]["path"],
            openpi_sources={f"policies/openpi/src/openpi/models/{name}.py":sha(ROOT/f"policies/openpi/src/openpi/models/{name}.py") for name in ("pi0","siglip","gemma")})
        rows = []
        vision = after_vision if phase["role"]=="native_full_after" else (2. if phase["role"]=="candidate" else 3.)
        for expected in p["expected_inputs"]:
            rows.append(dict(expected,noise_hash="a"*64,request_roundtrip_ms=8. if phase["role"]=="candidate" else 10.,
                server_request_ms=7. if phase["role"]=="candidate" else 8.,vision_ms=vision,
                vision_details=dict(cameras=3,shapes=[[1,256,2048]]*3,total_tokens=768),
                parity_actions=np.full((c["horizon"],7),9. if phase["role"]=="candidate" else phase["index"],np.float32).tolist() if expected["index"]<2 else None))
        b = dict(manifest_hash=self.m["manifest_hash"],condition=c,banks=p["banks"],warmup=20,samples=100,protocol=protocol,rows=rows,
            summary={key:dict(zip(("p50","p95","p99"),np.percentile([r[key] for r in rows],[50,95,99]).tolist())) for key in cohort.TIMERS})
        directory = Path(phase["phase_root"])/c["id"]
        atomic_json(directory/"benchmark.json",b)
        atomic_json(directory/"loaded_protocol.json",dict(protocol,real_input_parity_passed=True))
        atomic_json(directory/"condition_parity.json",dict(passed=True,max_abs=0.,noise_paired=True,manifest_hash=self.m["manifest_hash"],condition_id=c["id"],banks=p["banks"]))
        phase.update(server_pid=protocol["server_pid"],artifact_hashes={name:sha(directory/name) for name in ("benchmark.json","loaded_protocol.json","condition_parity.json")})
        return directory

    def completed(self,root,after_vision=3.):
        p,bank,gate = self.planning(root,launch=True)
        identity = dict(uuid="GPU-FIXTURE",name="fixture",pci_bus_id="0000:00:00.0",assigned_gpu="0")
        p.update(status="completed",gpu_identity=identity)
        for phase in p["phases"]:
            phase.update(status="completed",controller_pid=4000+phase["index"],exit_code=0,wall_s=20.,gpu_before=identity,gpu_after=identity)
            self.phase_files(p,phase,after_vision)
        atomic_json(Path(p["root"])/"cohort.json",p)
        return p

    def audited(self,p):
        with patch.object(cohort,"authenticate_vision_gate",return_value=dict(passed=True)):
            return cohort.audit(p["root"],self.m)

    def test_dry_run_zero_gpu_or_child_queries(self):
        with tempfile.TemporaryDirectory() as root:
            bank,gate = self.inputs(root)
            argv = ["performance_cohort","run","--manifest",self.manifest_path,"--root",str(Path(root)/"new"),"--model","pi0",
                "--candidate",self.candidate,"--bank",str(bank),"--gpu","0","--vision-parity",str(gate)]
            with patch("sys.argv",argv),patch("subprocess.check_output",side_effect=AssertionError("No GPU query allowed")), \
                 patch("subprocess.Popen",side_effect=AssertionError("No launch allowed")),contextlib.redirect_stdout(io.StringIO()):
                cohort.main()
            self.assertFalse((Path(root)/"new").exists())

    def test_phase_order_ports_unique_candidates_and_budget_gate(self):
        with tempfile.TemporaryDirectory() as root:
            p,bank,gate = self.planning(root)
            self.assertEqual([r["role"] for r in p["phases"]],["native_full_before","matched_all27_full","candidate","native_full_after"])
            self.assertEqual([r["port"] for r in p["phases"]],list(range(11000,11004)))
            for candidates,flag,vision in (([self.candidate,self.candidate],False,str(gate)),([self.candidate],True,str(gate)),([self.candidate],False,None)):
                with self.assertRaises(ValueError):
                    cohort.make_plan(self.m,self.manifest_path,Path(root)/"new","pi0",candidates,[str(bank)],require_four_budgets=flag,vision_parity=vision)
            four = [next(c["id"] for c in self.m["conditions"] if c["model"]=="pi0" and len(c["skip"])==n) for n in (3,6,9,12)]
            full = cohort.make_plan(self.m,self.manifest_path,Path(root)/"final","pi0",four,[str(bank)],require_four_budgets=True,vision_parity=str(gate))
            self.assertEqual(len(full["phases"]),7)

    def test_real_schema_same_pid_and_counterfactual_candidate_outputs_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.completed(root)
            result = self.audited(p)
            self.assertTrue(result["passed"])
            self.assertTrue(result["stable_for_primary_comparison"])
            self.assertFalse(result["final_vision_cohort"])
            self.assertEqual(result["comparison"]["candidates"][0]["native_before_p50_speedup"]["request_roundtrip_ms"],1.25)
            self.assertTrue(all(p["affirmative_same_pid_proof"]["proof"].startswith("affirmative same-PID") for p in result["phases"]))
            self.assertTrue(all(p["original_benchmark_parity_snapshot"] is False for p in result["phases"]))

    def test_small_vision_drift_flags_with_stable_rpc_no_auto_retry(self):
        with tempfile.TemporaryDirectory() as root:
            p = self.completed(root,after_vision=3.6)
            result = self.audited(p)
            self.assertTrue(result["passed"])
            self.assertFalse(result["stable_for_primary_comparison"])
            self.assertTrue(result["comparison"]["drift_exceeds_threshold_by_metric"]["vision_ms"])
            self.assertFalse(result["comparison"]["drift_exceeds_threshold_by_metric"]["request_roundtrip_ms"])

    def test_changed_pid_identity_noise_summary_or_bank_rejected(self):
        for change in ("pid","condition","noise","summary","bank","gpu_uuid"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as root:
                p = self.completed(root)
                phase = p["phases"][2]
                directory = Path(phase["phase_root"])/phase["condition"]
                if change=="gpu_uuid":
                    phase["gpu_after"] = dict(phase["gpu_after"],uuid="GPU-OTHER")
                    atomic_json(Path(p["root"])/"cohort.json",p)
                    with self.assertRaises(AssertionError):
                        self.audited(p)
                    continue
                if change=="pid":
                    path = directory/"loaded_protocol.json"
                    row = json.loads(path.read_text())
                    row["server_pid"] += 1
                else:
                    path = directory/"benchmark.json"
                    row = json.loads(path.read_text())
                    if change=="condition":
                        row["condition"]["skip"] = []
                    elif change=="noise":
                        row["rows"][0]["noise_hash"] = "b"*64
                    elif change=="summary":
                        row["summary"]["vision_ms"]["p50"] = 1.
                    else:
                        row["banks"][0]["sha256"] = "wrong"
                atomic_json(path,row)
                phase["artifact_hashes"][path.name] = sha(path)
                atomic_json(Path(p["root"])/"cohort.json",p)
                with self.assertRaises(AssertionError):
                    self.audited(p)

    def test_launch_stops_first_nonzero_records_own_pid_and_preserves(self):
        class Child:
            pid = 123
            def wait(self):
                return 7
        calls = []
        def popen(command,**kwargs):
            calls.append(command)
            return Child()
        with tempfile.TemporaryDirectory() as root:
            p,_,_ = self.planning(root,launch=True)
            identity = dict(uuid="GPU-MOCK",assigned_gpu="0")
            self.assertEqual(cohort.execute(p,popen=popen,query=lambda gpu:identity),7)
            self.assertEqual(len(calls),1)
            receipt = json.loads((Path(p["root"])/"cohort.json").read_text())
            self.assertEqual(receipt["phases"][0]["controller_pid"],123)
            self.assertEqual(receipt["phases"][0]["exit_code"],7)
            self.assertEqual(receipt["phases"][1]["status"],"pending")
            with self.assertRaises(FileExistsError):
                cohort.execute(p,popen=popen,query=lambda gpu:identity)

    def test_assigned_card_only_query_and_gate_bank_authentication(self):
        with patch("subprocess.check_output",return_value="GPU-FIXTURE, card, 0000:01:00.0\n") as query:
            self.assertEqual(cohort.gpu_identity("2")["assigned_gpu"],"2")
        self.assertIn("--id=2",query.call_args.args[0])
        with tempfile.TemporaryDirectory() as root:
            p,bank,gate = self.planning(root)
            atomic_json(gate,dict(banks=p["banks"]))
            p["vision_parity"]["sha256"] = sha(gate)
            with patch("preparations.idea1_v3_1.final_audit_v2.verify_gate",return_value=dict(passed=True)):
                self.assertTrue(cohort.authenticate_vision_gate(p,self.m)["passed"])
                wrong = copy.deepcopy(p["banks"])
                wrong[0]["sha256"] = "different"
                atomic_json(gate,dict(banks=wrong))
                p["vision_parity"]["sha256"] = sha(gate)
                with self.assertRaises(AssertionError):
                    cohort.authenticate_vision_gate(p,self.m)

    def test_coverage_registrar_never_copies_full_or_overwrites_and_keeps_witnesses(self):
        from . import register_performance_cohort as registrar
        with tempfile.TemporaryDirectory() as root:
            p = self.completed(root)
            fresh = self.audited(p)
            atomic_json(Path(p["root"])/"audit.json",fresh)
            runtime = Path(root)/"runtime"
            protected = runtime/"pi0_full/benchmark.json"
            protected.parent.mkdir(parents=True)
            protected.write_bytes(b"original Full is never touched")
            original = sha(protected)
            with patch.object(cohort,"authenticate_vision_gate",return_value=dict(passed=True)):
                registration = registrar.prepare(p["root"],runtime,self.m)
            self.assertEqual(len(registration["copies"]),1)
            alias = registration["copies"][0]
            self.assertIn(self.candidate,alias["target"])
            self.assertEqual(len(alias["origin"]["origin_witnesses"]),2)
            self.assertFalse(alias["origin"]["fresh_full_references_copied"])
            self.assertIn("Legacy",alias["origin"]["warning"])
            registrar.write(registration)
            self.assertEqual(sha(protected),original)
            self.assertEqual(sha(alias["target"]),alias["origin"]["source_sha256"])
            self.assertFalse((runtime/"pi0_full_vision_scan_reference/benchmark.json").exists())
            with self.assertRaises(FileExistsError):
                registrar.write(registration)
            with patch.object(cohort,"authenticate_vision_gate",return_value=dict(passed=True)):
                repeated = registrar.prepare(p["root"],runtime,self.m)
            self.assertEqual(repeated["copies"],[])
            self.assertEqual(len(repeated["skipped"]),1)


if __name__ == "__main__":
    unittest.main()
