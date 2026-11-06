"""CPU-only protocol, parity, negative-result and dry-run regression checks."""
import copy
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from . import shared_service_e2_generic as runner
from .audit_shared_e2_generic import adoption_evidence,verify,scoped_artifacts
from .common import atomic_json,digest,read_manifest,seed
from .final_audit_v2 import schema_protocol_check
from .same_process_parity import sha

MANIFEST = "results/idea1_v3_1/manifest_v2.json"


class GenericE2Tests(unittest.TestCase):
    def setUp(self):
        self.manifest = read_manifest(MANIFEST,verify_sources=True)

    def controls(self,horizon):
        return [dict(bank_index=i,input_hash=f"input{i}",noise_hash=f"noise{i}",seed=i,
            actions=np.zeros((horizon,7)).tolist()) for i in range(5)]

    def test_full_native_protocols_and_horizons(self):
        for model,D,H in (("pi0",10,50),("xiaomi-robotic0",2,10),("db-cogact",10,16)):
            c = runner.full_condition(self.manifest,model)
            self.assertEqual((c["denoising_steps"],c["horizon"]),(D,H))
            self.assertEqual(c["schedule"],dict(kind="native",m=None,num_steps=D))
            self.assertTrue(runner.fixed_checks(self.controls(H),self.controls(H),H)["passed"])
            self.assertFalse(runner.fixed_checks(self.controls(H),self.controls(H),H)["raw_actions_available"])
            for variant in self.manifest["e2"]["variants"]:
                commands = runner.client_commands("manifest","out",9711,variant,model)
                self.assertEqual(len(commands),variant["workers"])
                for command in commands:
                    self.assertEqual(command[command.index("--condition")+1],model+"_full")
                    self.assertEqual(command[command.index("--stage")+1],"profile")
                    self.assertNotIn("--environment-reuse",command)
        self.assertEqual(runner.full_condition(self.manifest,"db-cogact")["execute_length"],
            dict(libero_spatial=12,libero_object=16,libero_goal=16,libero_10=15))

    def test_reject_changed_budget_mask_schedule_and_control_keys(self):
        for field,value in (("denoising_steps",2),("horizon",10),("skip",[4]),("schedule",dict(kind="flow",m=.5,num_steps=10))):
            manifest = copy.deepcopy(self.manifest)
            c = next(c for c in manifest["conditions"] if c["id"]=="pi0_full")
            c[field] = value
            with self.assertRaises(AssertionError):
                runner.full_condition(manifest,"pi0")
        base = self.controls(16)
        for key,value in (("bank_index",99),("input_hash",""),("noise_hash","different"),("seed",99),("actions",[[0.]*7])):
            changed = copy.deepcopy(base)
            changed[0][key] = value
            with self.assertRaises(AssertionError):
                runner.fixed_checks(base,changed,16)
        changed = copy.deepcopy(base)
        changed[0]["actions"][0][0] = float("nan")
        with self.assertRaises(AssertionError):
            runner.fixed_checks(base,changed,16)
        changed = copy.deepcopy(base)
        changed[0]["actions"][0][0] = .001
        self.assertFalse(runner.fixed_checks(base,changed,16)["passed"])

    def test_dryrun_never_loads_model_or_queries_gpu(self):
        for model in runner.MODELS:
            with patch("sys.argv",["runner","--model",model,"--manifest",MANIFEST,"--output","unused","--port","9711"]), \
                 patch.object(runner,"run",side_effect=AssertionError("launch")), \
                 patch.object(runner.subprocess,"check_output",side_effect=AssertionError("GPU query")), \
                 redirect_stdout(io.StringIO()) as output:
                runner.main()
            row = json.loads(output.getvalue())
            self.assertTrue(row["dry_run"])
            self.assertEqual(row["episodes"],32)
            self.assertEqual(row["condition"],model+"_full")

    def test_assigned_gpu_only_and_busy_rejection(self):
        with patch.object(runner.subprocess,"check_output",return_value="GPU-id") as query:
            with self.assertRaises(RuntimeError):
                runner.assert_assigned_gpu_free("0")
            self.assertIn("-i",query.call_args.args[0])
            self.assertIn("0",query.call_args.args[0])

    def test_correct_torch_digest_schema_per_backend(self):
        from preparations.idea1_v2_1.runtime_models import digest as native_digest
        for model in ("xiaomi-robotic0","db-cogact"):
            c = runner.full_condition(self.manifest,model)
            with tempfile.TemporaryDirectory() as root:
                asset = Path(root)/"shard"
                asset.touch()
                shards = [dict(path=str(asset),sha256=sha(asset),bytes=0)]
                protocol = dict(condition=c,condition_id=c["id"],manifest_hash=self.manifest["manifest_hash"],
                    server_pid=123,real_input_parity_passed=True,H=c["horizon"],D=c["denoising_steps"],
                    R=c["execute_length"],depth=c["depth"],checkpoint=self.manifest["checkpoints"][model]["path"],
                    checkpoint_verification_source="mock",checkpoint_shards=shards,checkpoint_sha256=native_digest(shards))
                receipt = dict(ok=True,shards=[dict(path=str(asset),actual_sha256=sha(asset),expected_bytes=0)])
                self.assertNotEqual(native_digest(shards),digest(shards))
                with patch("preparations.idea1_v3_1.final_audit.load",return_value=receipt):
                    schema_protocol_check(protocol,c,self.manifest,root,[])
                    protocol["checkpoint_sha256"] = digest(shards)
                    with self.assertRaises(AssertionError):
                        schema_protocol_check(protocol,c,self.manifest,root,[])

    def test_first_capture_each_arm_uses_unchanged_policy(self):
        from . import models,server
        from preparations.idea1_v2_1.runtime_client import input_digest
        scene = self.manifest["e2"]["episodes"][0]
        request_seed = seed(scene["suite"],scene["task"],scene["init"],0)
        obs = dict(agentview_image=np.zeros((2,2,3),np.uint8),robot0_eef_pos=np.zeros(3))
        for model in runner.MODELS:
            calls = []
            class FakePolicy:
                protocol = {}
            policy = FakePolicy()
            # Use a closure rather than adding behavior to the native factory.
            horizon = runner.full_condition(self.manifest,model)["horizon"]
            policy.predict = lambda **request:(calls.append(request) or dict(actions=np.zeros((horizon,7))))
            with tempfile.TemporaryDirectory() as root:
                def fake_main():
                    wrapped = models.load(runner.full_condition(self.manifest,model))
                    for variant in self.manifest["e2"]["variants"]:
                        for _ in range(2):
                            wrapped.predict(obs=obs,instruction="task",seed=request_seed,
                                _validation_phase="arm:"+variant["id"],_validation_control=False)
                with patch.object(models,"load",return_value=policy),patch.object(server,"main",side_effect=fake_main), \
                     patch.object(runner,"gpu_identity",return_value=dict(uuid="GPU-test")), \
                     patch.dict("os.environ",CUDA_VISIBLE_DEVICES="0"),patch("sys.argv",["test"]):
                    runner.service(SimpleNamespace(manifest=MANIFEST,model=model,output=root,port=9711))
                files = list(Path(root).rglob("observation.npz"))
                self.assertEqual(len(files),4)
                self.assertEqual(len(calls),8)
                self.assertTrue(all(not any(k.startswith("_validation") for k in r) for r in calls))
                for path in files:
                    identity = json.loads(path.with_name("identity.json").read_text())
                    self.assertEqual(identity["input_hash"],input_digest(obs,"task"))
                    self.assertEqual(identity["npz_sha256"],sha(path))

    def test_negative_outcome_valid_but_adoption_requires_first_pairing(self):
        pair = dict(scene=["fixture",0,0],same_official_initial=True,same_outcome=False,matched_requests=1,
            paired_action_checks=[dict(passed=True)],first_input_deltas=dict(
                robot0_eef_pos=dict(max_abs=0.,different_fraction=0.),agentview_image=dict(max_abs=0.,different_fraction=0.)))
        summary = dict(fixed_controls_pass=True,comparisons=[dict(variant="reuse_env",pairs=[pair]*8)])
        self.assertFalse(adoption_evidence(summary)[0]["evidence_supports_consideration"])
        pair["same_outcome"] = True
        self.assertTrue(adoption_evidence(summary)[0]["evidence_supports_consideration"])
        self.assertFalse(adoption_evidence(summary,{"reuse_env":{("fixture",0,0):False}})[0]["evidence_supports_consideration"])
        pair.update(matched_requests=0,paired_action_checks=[])
        self.assertFalse(adoption_evidence(summary)[0]["evidence_supports_consideration"])

    def test_negative_episodes_remain_valid_raw_data_for_all_backends(self):
        from . import final_audit
        for model in runner.MODELS:
            c = runner.full_condition(self.manifest,model)
            scene = self.manifest["e2"]["episodes"][0]
            common = dict(scene,manifest_hash=self.manifest["manifest_hash"])
            with tempfile.TemporaryDirectory() as root:
                path = Path(root)/scene["suite"] / "task00" / "init00"
                path.mkdir(parents=True)
                atomic_json(path/"complete.json",dict(valid=True,manifest_hash=self.manifest["manifest_hash"]))
                identity = dict(common,condition=c,init_vector_sha256="official-fixture",env_seed=7,control_hz=20,
                    protocol={},controller=dict(input_min=-1,input_max=1,output_min=[-.05]*3+[-.5]*3,output_max=[.05]*3+[.5]*3))
                atomic_json(path/"identity.json",identity)
                atomic_json(path/"result.json",dict(common,condition=c["id"],valid=True,success=False,
                    init_vector_sha256="official-fixture",predictions=1,actions=1,video=False,save_observations=False))
                actions = np.zeros((c["horizon"],7))
                audit = dict(NFE=c["denoising_steps"],vision_skip=[],original_scanned_ae=True,skip=[],schedule=c["schedule"])
                inference = dict(request=0,step=0,seed=seed(scene["suite"],0,0,0),manifest_hash=self.manifest["manifest_hash"],
                    condition_id=c["id"],actions=actions.tolist(),executed_length=1,audit=audit,input_hash="input",noise_hash="noise")
                with (path/"infer.jsonl").open("x") as stream:
                    stream.write(json.dumps(inference)+"\n")
                telemetry = [dict(step=i,eef_pos=[0.]*3,sim_time_s=i/20,host_time_s=float(i),
                    phase_interpretation="fixture",semantic_phase="unknown") for i in range(2)]
                telemetry[1]["executed_action"] = [0.]*6+[1. if model=="db-cogact" else 0.]
                with (path/"telemetry.jsonl").open("x") as stream:
                    stream.write("".join(json.dumps(row)+"\n" for row in telemetry))
                with patch.object(final_audit,"protocol_check"):
                    result = final_audit.audit_episode(path,c,self.manifest,False)
                self.assertTrue(result["valid"])
                self.assertFalse(result["success"])

    def test_incomplete_audit_and_registration_overwrite_rejected(self):
        from .register_shared_e2_generic import write
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"validation.json"
            atomic_json(path,dict(status="starting"))
            with self.assertRaises(AssertionError):
                verify(root,self.manifest,False)
            target = Path(root)/"throughput_profile_reuse_env.json"
            target.touch()
            with self.assertRaises(FileExistsError):
                write([(target,{})],dict(benchmark_aliases=[]),root)
            self.assertEqual(target.read_bytes(),b"")

    def test_future_quality_artifacts_do_not_change_e2_scope(self):
        condition = runner.full_condition(self.manifest,"pi0")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant in self.manifest["e2"]["variants"]:
                for scene in self.manifest["e2"]["episodes"]:
                    key = Path(scene["suite"])/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
                    arm = root/condition["id"]/variant["id"]/key
                    arm.mkdir(parents=True)
                    (arm/"complete.json").touch()
                    first = root/"first_inputs"/variant["id"]/key
                    first.mkdir(parents=True)
                    (first/"observation.npz").touch()
                    if variant["video"]=="all":
                        (arm/"video.mp4").touch()
            before = scoped_artifacts(root,condition,self.manifest)
            self.assertEqual(tuple(map(len,before)),(32,32,8))
            quality = root/condition["id"]/"episodes"/"libero_spatial"/"task00"/"init02"
            quality.mkdir(parents=True)
            for filename in ("complete.json","result.json","video.mp4"):
                (quality/filename).touch()
            after = scoped_artifacts(root,condition,self.manifest)
            self.assertEqual(before,after)
            # Unexpected keys inside an E2 arm still remain visible to exactness checks.
            unexpected = root/condition["id"]/"lean_io"/"libero_spatial"/"task00"/"init99"
            unexpected.mkdir(parents=True)
            (unexpected/"complete.json").touch()
            self.assertNotEqual(before,scoped_artifacts(root,condition,self.manifest))


if __name__=="__main__":
    unittest.main()
