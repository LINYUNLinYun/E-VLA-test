"""Synthetic CPU fixtures; none of these outcomes are research evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .common import SUITES, conditions, digest, sha
from .analyze import key
from .analyze_confirmation import analyze, bootstrap, direction_analysis, holm


class ConfirmationAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.ids=[c["id"] for c in conditions()[:3]]
        self.scenes=[dict(suite=s,task=t,init=i) for s in SUITES for t in range(10) for i in range(31,36)]
        self.direction=dict(suite=SUITES[0],better=self.ids[1],worse=self.ids[2],discovery_difference=.2)
        self.manifest=dict(manifest_hash="synthetic-only",tasks=list(range(10)),confirmation_inits=list(range(31,36)))

    def records(self,positive_tasks):
        return {(cid,key(c)):dict(success=cid==self.ids[1] and c["task"] in positive_tasks)
            for cid in self.ids for c in self.scenes}

    def test_exact_holm_and_stable_positive_direction(self):
        self.assertEqual(holm([.04,.03]),[.06,.06])
        row=direction_analysis(self.direction,self.manifest,self.records(range(10)),self.scenes,True,replicates=300)
        self.assertEqual((row["a_only"],row["b_only"]),(50,0))
        self.assertEqual(row["inference"]["exact_mcnemar_one_sided"],2**-50)
        self.assertTrue(row["inference"]["stable_positive_direction"])
        self.assertEqual(row["task_direction_consistency"]["positive"],10)
        self.assertEqual(row["init_direction_consistency"]["positive"],5)

    def test_task_concentrated_gain_remains_uncertain(self):
        row=direction_analysis(self.direction,self.manifest,self.records([0]),self.scenes,True,replicates=1000)
        self.assertGreater(row["success_difference"],0)
        self.assertEqual(row["inference"]["confidence_intervals"]["task_cluster"]["low"],0)
        self.assertFalse(row["inference"]["stable_positive_direction"])
        self.assertFalse(row["inference"]["positive_after_any_one_task_or_init_removed"])

    def test_bootstrap_reproducible_and_incomplete_has_no_inference(self):
        values=[1]*5+[0]*45;tasks=[t for t in range(10) for _ in range(5)]
        self.assertEqual(bootstrap(values,tasks,1,2,300),bootstrap(values,tasks,1,2,300))
        row=direction_analysis(self.direction,self.manifest,self.records(range(10)),self.scenes,False)
        self.assertIsNone(row["inference"])
        self.assertIsNone(row["exact_mcnemar_two_sided"])

    def test_no_selection_and_all_contention_records_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);selection=dict(complete_discovery_episodes=160,independent_validation_launched=False,
                selection=dict(status="none",selected=None))
            selection["selection_hash"]=digest(selection)
            path=folder/"selection.json";path.write_text(json.dumps(selection))
            contention=folder/"resource_contention";contention.mkdir()
            record=dict(observed_at=1,multi_gpu_processes={"861820":["GPU-0","GPU-3"]})
            (contention/"observed.json").write_text(json.dumps(record))
            report=analyze(folder/"confirmation",path,contention)
            self.assertEqual(report["status"],"no_selection")
            self.assertEqual(report["resource_contention"]["known_multi_gpu_processes"],record["multi_gpu_processes"])
            self.assertEqual(report["resource_contention"]["all_known_records"][0]["records"],[record])

    def fixture(self,folder):
        root=folder/"confirmation";root.mkdir()
        directions=[self.direction,dict(suite=SUITES[1],better=self.ids[2],worse=self.ids[1],discovery_difference=.2)]
        selection=dict(complete_discovery_episodes=160,independent_validation_launched=False,
            selection=dict(status="selected",selected=dict(pair=self.ids[1:],directions=directions)))
        selection["selection_hash"]=digest(selection)
        selected=folder/"selection.json";selected.write_text(json.dumps(selection))
        m=dict(phase="independent_confirmation",confirmation_output=str(root),discovery_reference=dict(root=str(folder/"discovery")),
            conditions=conditions(),planned_conditions=self.ids,suites=list(SUITES),tasks=list(range(10)),confirmation_inits=list(range(31,36)),
            total_confirmation_episodes=600,source_files={},checkpoints={},preregistered_hypotheses=directions,
            selection_evidence=dict(path=str(selected),sha256=sha(selected),selection_hash=selection["selection_hash"],frozen_selection=selection["selection"]),
            official_init_availability=dict(cases=[dict(suite=s,task=t,init_sha256={str(i):"fixture-init" for i in range(31,36)}) for s in SUITES for t in range(10)]))
        m["manifest_hash"]=digest(m);(root/"manifest.json").write_text(json.dumps(m))
        case=self.scenes[0];cfg=conditions()[0]
        d=root/cfg["id"]/"episodes"/case["suite"]/"task00/init31";d.mkdir(parents=True)
        identity=dict(manifest_hash=m["manifest_hash"],case=case,config=cfg,init_vector_sha256="fixture-init",
            protocol=dict(manifest_hash=m["manifest_hash"],condition_id=cfg["id"],real_input_parity_passed=True,gpu_uuid="GPU-test"))
        result=dict(valid=True,manifest_hash=m["manifest_hash"],case=case,condition=cfg["id"],config=cfg,success=True,
            predictions=1,actions=5,counts=dict(full=1,draft=0),wall_s=1)
        for name,obj in (("identity.json",identity),("result.json",result),("complete.json",dict(valid=True,manifest_hash=m["manifest_hash"]))):
            (d/name).write_text(json.dumps(obj))
        (d/"infer.jsonl").write_text(json.dumps(dict(route=dict(refresh=True),executed_length=5,timing=dict(policy_wall_ms=1),rpc_ms=2))+"\n")
        (d/"telemetry.jsonl").write_text("{}\n");(d/"arrays.npz").write_bytes(b"synthetic opaque fixture")
        receipt=dict(valid=True,manifest_hash=m["manifest_hash"],case=case,condition=cfg["id"],success=True,gpu_uuid="GPU-test",
            initial_gate=dict(passed=True,manifest_hash=m["manifest_hash"],case=case,input_hash="paired-fixture",seed=1),
            artifacts={name:sha(d/name) for name in ("identity.json","result.json","complete.json","infer.jsonl","telemetry.jsonl","arrays.npz")})
        (d/"episode_receipt.json").write_text(json.dumps(receipt))
        return root,d,receipt

    def test_partial_receipts_are_incomplete_and_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,d,_=self.fixture(Path(folder))
            report=analyze(root)
            self.assertEqual(report["valid_episodes"],1)
            self.assertEqual(report["status"],"incomplete")
            self.assertTrue(all(row["inference"] is None for row in report["primary_directions"]))
            (d/"arrays.npz").write_bytes(b"changed")
            report=analyze(root)
            self.assertEqual(report["valid_episodes"],0);self.assertTrue(report["invalid"])

    def test_matching_hash_does_not_hide_wrong_original_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root,d,receipt=self.fixture(Path(folder))
            identity=json.loads((d/"identity.json").read_text());identity["case"]=dict(identity["case"],init=30)
            (d/"identity.json").write_text(json.dumps(identity))
            receipt["artifacts"]["identity.json"]=sha(d/"identity.json")
            (d/"episode_receipt.json").write_text(json.dumps(receipt))
            report=analyze(root)
            self.assertEqual(report["valid_episodes"],0);self.assertTrue(report["invalid"])

    def test_complete_synthetic_confirmation_uses_both_frozen_directions(self):
        with tempfile.TemporaryDirectory() as folder:
            root,_,_=self.fixture(Path(folder));records={}
            for cid in self.ids:
                for case in self.scenes:
                    success=cid=="full_d10" or case["suite"] not in SUITES[:2] or (
                        (case["suite"]==SUITES[0] and cid==self.ids[1]) or
                        (case["suite"]==SUITES[1] and cid==self.ids[2]))
                    records[cid,key(case)]=dict(condition=cid,case=case,success=success,actions=5,predictions=1,
                        full_requests=1,fast_requests=0,policy_ms=[1.],full_policy_ms=[1.],fast_policy_ms=[],rpc_ms=[2.],
                        episode_wall_s=1.,full_executed_actions=5,fast_executed_actions=0,gpu_uuid="GPU-test")
            with patch("preparations.ease_insights_20261103.vlm.analyze_confirmation.load_episodes",return_value=(records,[],[],[])), \
                 patch("preparations.ease_insights_20261103.vlm.analyze_confirmation.audit_identities",return_value=[]):
                report=analyze(root,replicates=300)
            self.assertEqual(report["status"],"complete")
            self.assertEqual(report["valid_episodes"],600)
            self.assertEqual(report["conclusion"],"supports_complementary_fixed_paths_on_the_two_prespecified_suites")
            self.assertEqual([r["inference"]["holm_adjusted_p"] for r in report["primary_directions"]],[2**-49]*2)


if __name__ == "__main__":unittest.main()
