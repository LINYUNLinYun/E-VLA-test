"""CPU-only confirmation isolation tests; synthetic data is never experimental evidence."""
import copy
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from .common import SUITES, conditions, digest, read_manifest, cases as discovery_cases
from .confirm_common import PHASE, cases, planned_config
from .prepare_confirmation import prepare, reject_heldout_data, verify_selection
from .recovery import run_with_retries


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.manifest=dict(conditions=conditions(),suites=list(SUITES),tasks=list(range(10)),discovery_inits=[30],
            total_discovery_episodes=160,source_files={},checkpoints={})
        self.manifest["manifest_hash"]=digest(self.manifest)
        self.pair=[c["id"] for c in conditions()[1:3]]
        choice=dict(status="selected",selected=dict(pair=self.pair,directions=[
            dict(suite=SUITES[0],better=self.pair[0],worse=self.pair[1],discovery_difference=.2),
            dict(suite=SUITES[1],better=self.pair[1],worse=self.pair[0],discovery_difference=.2)]))
        self.selection=dict(discovery_manifest_hash=self.manifest["manifest_hash"],complete_discovery_episodes=160,
            selection=choice,evidence=dict(synthetic_test_only=True),independent_validation_launched=False)
        self.selection["selection_hash"]=digest(self.selection)
        self.report=dict(complete=True,valid_episodes=160,expected_episodes=160,selection=choice,evidence=self.selection["evidence"])

    def test_unfrozen_or_changed_selection_rejected(self):
        missing=copy.deepcopy(self.selection);missing.pop("selection_hash")
        with self.assertRaises(ValueError):verify_selection(missing,self.manifest,self.report)
        changed=copy.deepcopy(self.selection);changed["selection"]["selected"]["directions"][0]["suite"]=SUITES[2]
        with self.assertRaises(ValueError):verify_selection(changed,self.manifest,self.report)
        changed.pop("selection_hash");changed["selection_hash"]=digest(changed)
        with self.assertRaises(ValueError):verify_selection(changed,self.manifest,self.report)

    def test_six_hundred_cases_and_stage_isolation(self):
        m=dict(self.manifest,phase=PHASE,confirmation_inits=list(range(31,36)),planned_conditions=["full_d10",*self.pair])
        self.assertEqual(sum(len(cases(m,"confirm")) for _ in m["planned_conditions"]),600)
        self.assertEqual({c["init"] for c in cases(m,"confirm")},set(range(31,36)))
        all_shards=[c for i in range(3) for c in cases(m,"confirm",i,3)]
        self.assertEqual(len(all_shards),200);self.assertEqual(len({tuple(c.values()) for c in all_shards}),200)
        with self.assertRaises(ValueError):cases(m,"discovery")
        with self.assertRaises(ValueError):cases(self.manifest,"confirm")
        with self.assertRaises(ValueError):discovery_cases(m,"confirm")
        with self.assertRaises(ValueError):planned_config(m,conditions()[3]["id"])

    def test_preexisting_heldout_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"fixed_v12_jacobian/episodes/libero_spatial/task00/init31"
            p.mkdir(parents=True);reject_heldout_data([root])
            (p/"result.json").write_text("{}")
            with self.assertRaises(ValueError):reject_heldout_data([root])

    def test_completed_success_resumes_without_inference(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"init31";p.mkdir();(p/"complete.json").write_text('{"valid":true}')
            def forbidden():raise AssertionError("Completed case must not run inference")
            result=run_with_retries(p,3,forbidden,lambda:dict(valid=True,success=True),lambda:None)
            self.assertTrue(result["success"])

    def test_prepare_freezes_all_definitions_but_only_three_planned_conditions(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);discovery=root/"discovery";discovery.mkdir();output=root/"confirmation"
            (discovery/"manifest.json").write_text(json.dumps(self.manifest))
            selected=discovery/"selection.json";selected.write_text(json.dumps(self.selection))
            with patch("preparations.ease_insights_20261103.vlm.analyze.analyze",return_value=self.report):
                result=prepare(discovery,selected,output)
                m=read_manifest(output/"manifest.json",verify_sources=True)
                self.assertEqual(m["planned_conditions"],["full_d10",*self.pair])
                self.assertEqual(len(m["conditions"]),4)
                self.assertEqual(m["preregistered_hypotheses"],self.selection["selection"]["selected"]["directions"])
                self.assertEqual(m["total_confirmation_episodes"],600)
                self.assertFalse(result["gpu_launched"])
                self.assertTrue(prepare(discovery,selected,output)["reused"])
                from .confirm_runner import run
                args=Namespace(output=str(output),condition="full_d10",stage="confirm",shard_index=0,shards=1,
                    gpu_uuid="GPU-NOT-ACCESSED",port=21999,launch=False,preflight_only=False)
                commands=run(args)
                self.assertIn("preparations.ease_insights_20261103.vlm.server",commands["server"])
                self.assertIn("preparations.ease_insights_20261103.vlm.confirm_worker",commands["worker"])
                self.assertEqual(commands["episodes"],200)
                args.launch=True
                with patch("preparations.ease_insights_20261103.vlm.confirm_runner.recover_case",return_value=None), \
                     patch("preparations.ease_insights_20261103.vlm.confirm_worker.completed",return_value=True), \
                     patch("preparations.ease_insights_20261103.vlm.confirm_runner.resolve_binding",side_effect=AssertionError("No GPU for complete cases")):
                    resumed=run(args)
                self.assertEqual(resumed["already_completed"],200)
                self.assertTrue((output/"full_d10/confirm_shard00of01_complete.json").exists())
            selected.write_text(json.dumps({**self.selection,"selection_hash":"changed"}))
            with self.assertRaises(ValueError):prepare(discovery,selected,output)

    def test_prepare_without_frozen_selection_creates_no_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);discovery=root/"discovery";discovery.mkdir();output=root/"confirmation"
            (discovery/"manifest.json").write_text(json.dumps(self.manifest))
            selected=discovery/"selection.json";selected.write_text('{"selection":{"status":"incomplete_discovery"}}')
            with self.assertRaises(ValueError):prepare(discovery,selected,output)
            self.assertFalse(output.exists())


if __name__ == "__main__":unittest.main()
