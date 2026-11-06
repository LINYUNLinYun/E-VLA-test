import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .common import ROOT, SUITES, atomic_json, read_manifest
from .same_process_parity import compare, representatives, sha
from .final_audit_v2 import audit, verify_gate,benchmark_parity_proof,schema_protocol_check,selected_best_frontiers


class GateTests(unittest.TestCase):
    def setUp(self):
        self.manifest = read_manifest("results/idea1_v3_1/manifest_v2.json")

    def fixture(self,root):
        from .benchmark import load_banks
        from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
        records = [dict(obs={"state":np.array([i],np.float32)},instruction="actual fixture",suite=s,seed=i+10)
                   for i,s in enumerate([SUITES[0],*SUITES])]
        bank = Path(root)/"bank.pkl"
        bank.write_bytes(pickle.dumps(records))
        records,banks = load_banks([str(bank)])
        source_paths = [ROOT/"preparations/idea1_v3_1"/n for n in ("same_process_parity.py","vision.py","models.py")]
        source_paths += [ROOT/"policies/openpi/src/openpi/models"/n for n in ("pi0.py","siglip.py","gemma.py")]
        def metric(shape):
            return dict(shape=shape,max_abs=0.,rmse=0.,exact=True,passed=True)
        camera = dict(encoded=metric([1,256,1152]),projected=metric([1,256,2048]),
            blocks={f"block{i:02d}":{k:metric([1,256,1152]) for k in ("sa","+sa","mlp","+mlp")} for i in range(27)})
        mode = dict(raw=metric([1,50,32]),decoded=metric([50,7]),cameras={k:copy.deepcopy(camera) for k in ("base_0_rgb","left_wrist_0_rgb","right_wrist_0_rgb")})
        checks = []
        indices = representatives(records)
        for i in indices:
            r = records[i]
            noise = np.random.default_rng(r["seed"]).standard_normal((1,50,32),dtype=np.float32)
            checks.append(dict(bank_index=i,suite=r["suite"],seed=r["seed"],input_hash=input_digest(r["obs"],r["instruction"]),
                noise_hash=value_digest(noise),prepared_hash="fixture",modes={k:copy.deepcopy(mode) for k in ("native","native_retrace","selected_v2")}))
        t,times = np.float32(1),[]
        while t>=np.float32(.05):
            times.append(float(t))
            t = np.float32(t+np.float32(-.1))
        return dict(passed=True,model="pi0",manifest_hash=self.manifest["manifest_hash"],atol=1e-5,rtol=1e-5,
            checkpoint=self.manifest["checkpoints"]["pi0"],source_hashes={str(p.resolve()):sha(p) for p in source_paths},
            banks=banks,representative_indices=indices,checks=checks,sampler=dict(NFE=10,num_steps=10,time_values=times,final_time=float(t),scope="inferred"),
            cross_process_controls=dict(native_vs_native=[],native_vs_selected=[]))

    def test_positive_fixture_and_negative_proofs(self):
        with tempfile.TemporaryDirectory() as root:
            gate = self.fixture(root)
            p = Path(root)/"gate.json"
            atomic_json(p,gate)
            self.assertEqual(verify_gate(p,self.manifest,"pi0")["representatives"],5)
            changes = [lambda g:g.update(model="pi05"),
                lambda g:g["source_hashes"].update({next(iter(g["source_hashes"])):"wrong"}),
                lambda g:g["banks"][0].update(sha256="changed"),
                lambda g:g["sampler"].update(NFE=9),
                lambda g:g["checks"][0]["modes"]["selected_v2"]["cameras"]["base_0_rgb"]["blocks"].pop("block26"),
                lambda g:g["checks"].pop(),
                lambda g:g["checks"][0]["modes"].pop("native_retrace")]
            for change in changes:
                broken = copy.deepcopy(gate)
                change(broken)
                atomic_json(p,broken)
                with self.assertRaises((AssertionError,ValueError)):
                    verify_gate(p,self.manifest,"pi0")

    def test_missing_or_only_one_gate_cannot_pass_or_remove_error(self):
        base = dict(passed=True,errors=["pi0 native/matched real vision parity: AssertionError: "],missing_requirements=[],limitations=[])
        with tempfile.TemporaryDirectory() as root, patch("preparations.idea1_v3_1.final_audit.audit",side_effect=lambda *a:copy.deepcopy(base)):
            result = audit(self.manifest,root)
            self.assertFalse(result["passed"])
            self.assertIn(base["errors"][0],result["errors"])
            with patch("preparations.idea1_v3_1.final_audit_v2.verify_gate",side_effect=[dict(model="pi0"),AssertionError("missing pi05")]):
                result = audit(self.manifest,root)
            self.assertFalse(result["passed"])
            self.assertEqual(result["explicit_gate_replacements"],[])
            self.assertIn(base["errors"][0],result["errors"])

    def test_selection_and_finite_metrics(self):
        with self.assertRaises(ValueError):
            representatives([dict(suite=SUITES[0])])
        with self.assertRaises(AssertionError):
            compare([float("nan")],[0])

    def test_only_exact_proven_error_can_be_reclassified(self):
        exact = "pi0 native/matched real vision parity: AssertionError: "
        base = dict(passed=False,errors=[exact,"unrelated sampler failure"],missing_requirements=["missing quality"],limitations=[])
        control = dict(input_paired=True,noise_paired=True)
        gates = [dict(path="pi0_gate",cross_process_controls=dict(native_vs_native=[control])),
                 dict(path="pi05_gate",cross_process_controls=dict(native_vs_native=[]))]
        with tempfile.TemporaryDirectory() as root, patch("preparations.idea1_v3_1.final_audit.audit",return_value=base), \
             patch("preparations.idea1_v3_1.final_audit_v2.verify_gate",side_effect=gates), \
             patch("preparations.idea1_v3_1.final_audit_v2.old_cross_process_observation",return_value=dict(checks=[dict(passed=False)])):
            result = audit(self.manifest,root)
        self.assertFalse(result["passed"])
        self.assertEqual(result["errors"],["unrelated sampler failure"])
        self.assertEqual(result["missing_requirements"],["missing quality"])
        self.assertEqual(len(result["explicit_gate_replacements"]),1)

    def test_preparity_snapshot_requires_affirmative_or_explicit_limited_proof(self):
        c = self.manifest["conditions"][0]
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)/c["id"]
            bank = Path(root)/"bank"
            bank.write_bytes(b"trusted fixture")
            banks = [dict(path=str(bank),sha256=sha(bank))]
            protocol = dict(server_pid=123,real_input_parity_passed=False,manifest_hash=self.manifest["manifest_hash"],condition_id=c["id"])
            atomic_json(directory/"benchmark.json",dict(protocol=protocol,condition=c,manifest_hash=self.manifest["manifest_hash"],warmup=20,samples=100,rows=[{}]*100,banks=banks))
            atomic_json(directory/"loaded_protocol.json",dict(protocol,real_input_parity_passed=True))
            parity = dict(passed=True,max_abs=0.,manifest_hash=self.manifest["manifest_hash"],condition_id=c["id"],banks=banks)
            atomic_json(directory/"condition_parity.json",parity)
            proof = benchmark_parity_proof(protocol,c,self.manifest,root)
            self.assertIn("same-PID",proof["proof"])
            self.assertFalse(protocol["real_input_parity_passed"])
            atomic_json(directory/"condition_parity.json",dict(parity,passed=False))
            with self.assertRaises(AssertionError):
                benchmark_parity_proof(protocol,c,self.manifest,root)
            atomic_json(directory/"loaded_protocol.json",dict(protocol,server_pid=456,real_input_parity_passed=True))
            proof = benchmark_parity_proof(protocol,c,self.manifest,root)
            self.assertEqual(proof["historical_server_pid"],123)
            self.assertEqual(proof["current_loaded_pid"],456)
            self.assertIn("weaker",proof["limitation"])

    def test_exact_original_shard_digest_and_negative_identity(self):
        from preparations.idea1_v2_1.runtime_models import digest
        with tempfile.TemporaryDirectory() as root:
            weight = Path(root)/"weight"
            weight.write_bytes(b"abc")
            receipt = Path(root)/"receipt.json"
            atomic_json(receipt,dict(ok=True,shards=[dict(path=str(weight),actual_sha256=sha(weight),expected_bytes=3)]))
            c = dict(id="fixture_full",model="fixture",horizon=10,denoising_steps=2,depth=36,execute_length=10,skip=[],schedule={})
            m = dict(manifest_hash="fixture",checkpoints=dict(fixture=dict(path=str(weight))))
            shards = [dict(path=str(weight),sha256=sha(weight),bytes=3)]
            p = dict(manifest_hash="fixture",condition_id=c["id"],server_pid=123,real_input_parity_passed=True,
                condition=c,H=10,D=2,R=10,depth=36,checkpoint=str(weight),checkpoint_verification_source=str(receipt),
                checkpoint_shards=shards,checkpoint_sha256=digest(shards))
            schema_protocol_check(p,c,m,root,[])
            with self.assertRaises(AssertionError):
                schema_protocol_check(dict(p,checkpoint_sha256="wrong"),c,m,root,[])

    def test_frontier_only_best_not_all_near_floor_masks(self):
        candidates = [c for c in self.manifest["conditions"] if c["model"]=="xiaomi-robotic0" and len(c["skip"])==24]
        with tempfile.TemporaryDirectory() as root:
            for cid,successes in [("xiaomi-robotic0_full",40)]+[(c["id"],n) for c,n in zip(candidates,[27,26,25,24,0])]:
                index = 0
                for suite in self.manifest["suites"]:
                    for task in self.manifest["tasks"]:
                        p = Path(root)/cid/"episodes"/suite/f"task{task:02d}"/"init00"
                        atomic_json(p/"complete.json",dict(valid=True,manifest_hash=self.manifest["manifest_hash"]))
                        atomic_json(p/"result.json",dict(valid=True,manifest_hash=self.manifest["manifest_hash"],condition=cid,suite=suite,task=task,init=0,success=index<successes))
                        index += 1
            obligations,observations = selected_best_frontiers(root,self.manifest)
        self.assertEqual(len(obligations),1)
        self.assertEqual(obligations[0]["condition"],candidates[0]["id"])
        self.assertEqual(observations[0]["screen_relative_retention"],27/40)


if __name__ == "__main__":
    unittest.main()
