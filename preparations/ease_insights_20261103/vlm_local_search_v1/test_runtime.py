"""CPU-only protocol/isolation tests. Synthetic cases are not scientific evidence."""
import copy
from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch
import numpy as np

from .common import (SUITES, MASKS, CONDITIONS, DISCOVERY_PHASE, CONFIRMATION_PHASE,
    conditions, cases, validate_condition, planned_config, read_manifest, digest, sha, validate_output)
from .policy import Policy
from preparations.ease_insights_20261103.vlm.policy import Policy as FrozenPolicy
from preparations.ease_insights_20261103.vlm.test_protocol import FakeBackend
from preparations.ease_insights_20261103.vlm.recovery import run_with_retries, CompletedEpisodeFinalizationError


def manifest(root, stage="discovery"):
    root = Path(root)
    protocol = root/"synthetic_protocol.json"; protocol.write_text('{"synthetic_test_only":true}')
    design = root/"synthetic_design.json"; design.write_text('{"synthetic_test_only":true}')
    value = dict(phase=DISCOVERY_PHASE if stage=="discovery" else CONFIRMATION_PHASE,
        conditions=conditions(), suites=list(SUITES), tasks=list(range(10)), output_root=str(root),
        planned_conditions=list(CONDITIONS) if stage=="discovery" else ["full_d10", "jac_anchor", "local_goal"],
        cases_per_condition=120 if stage=="discovery" else 200,
        protocol_evidence=dict(path=str(protocol),sha256=sha(protocol)), design_evidence=dict(path=str(design),sha256=sha(design)),
        local_runtime_sources={},source_files={},checkpoints={"pi0":{"sha256":"checkpoint","identity_artifacts":{}}})
    if stage=="discovery": value.update(discovery_inits=[36,37,38],total_discovery_episodes=720)
    else: value.update(confirmation_inits=list(range(39,44)),total_confirmation_episodes=600)
    value["manifest_hash"] = digest(value)
    return value


class RuntimeTests(unittest.TestCase):
    def test_original_policy_mechanics_are_identical(self):
        for name in ("predict", "acknowledge", "reset", "__init__"):
            self.assertIs(getattr(Policy,name),getattr(FrozenPolicy,name))
        self.assertIsNot(Policy.configure,FrozenPolicy.configure)

    def test_all_candidates_own_exact_mask_history_noise_schedule(self):
        noise = []
        for cfg in conditions()[1:]:
            backend=FakeBackend(); policy=Policy(backend);policy.configure(cfg);rows=[]
            for index in range(6):
                row=policy.predict(index,"p","s",100+index);rows.append(row);policy.acknowledge(row["actions"][:5])
            self.assertEqual(backend.steps,[10,2,2,10,2,2])
            mask=tuple(cfg["skip"])
            self.assertEqual(backend.calls[0]["capture_layers"],mask)
            self.assertEqual(backend.calls[3]["capture_layers"],mask)
            self.assertEqual([backend.calls[i]["skip"] for i in (1,2,4,5)],[mask]*4)
            self.assertEqual(backend.calls[4]["history"],dict.fromkeys(mask,3))
            self.assertEqual(backend.calls[4]["older"],dict.fromkeys(mask,0))
            self.assertEqual([backend.calls[i]["ratio"] for i in (4,5)],[1/3,2/3])
            noise.append([r["model_noise_sha256"] for r in rows])
            policy.configure(cfg);self.assertEqual(policy.history,[])
        self.assertTrue(all(row==noise[0] for row in noise))

    def test_configuration_and_legacy_namespace_isolation(self):
        from preparations.ease_insights_20261103.vlm.common import validate_condition as old_validate
        for cfg in conditions()[1:]:
            with self.assertRaises(ValueError):old_validate(cfg)
        for name,value in (("skip",[1,2,3,4,5,6]),("D",3),("period",3),("R",12),("H",10),("id","fixed_v12_jacobian")):
            cfg=conditions()[1];cfg[name]=value
            with self.assertRaises(ValueError):validate_condition(cfg)
        for mask in MASKS.values():
            self.assertEqual(len(mask),6);self.assertEqual(tuple(sorted(set(mask))),mask)
            self.assertTrue(set(mask)<=set(range(1,18)))

    def test_case_universes_stage_locks_and_shards(self):
        with tempfile.TemporaryDirectory() as root:
            m=manifest(root)
            self.assertEqual(len(cases(m)),120)
            self.assertEqual({c["init"] for c in cases(m)},{36,37,38})
            merged=[c for shard in range(7) for c in cases(m,"discovery",shard,7)]
            self.assertEqual(len(merged),120);self.assertEqual(len({tuple(c.values()) for c in merged}),120)
            with self.assertRaises(ValueError):cases(m,"confirm")
            m=manifest(root,"confirm")
            self.assertEqual(len(cases(m)),200)
            with self.assertRaises(ValueError):planned_config(m,"local_spatial")
            m["confirmation_inits"][0]=38
            with self.assertRaises(ValueError):cases(m)

    def test_manifest_changes_and_moved_output_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);m=manifest(root);path=root/"manifest.json";path.write_text(json.dumps(m))
            self.assertEqual(read_manifest(path)["manifest_hash"],m["manifest_hash"])
            with self.assertRaises(ValueError):validate_output(root/"other",m)
            m["conditions"][1]["skip"]=[1,2,3,4,5,6];m.pop("manifest_hash");m["manifest_hash"]=digest(m);path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):read_manifest(path)

    def test_probe_dynamic_counts_and_identical_noise(self):
        from .benchmark import run
        rows=[dict(obs=0,instruction="p",suite=s,seed=123,input_hash=s) for s in SUITES]
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);m=manifest(root);m["bank"]={"path":"synthetic","sha256":"bank"}
            with patch("preparations.idea1_v3_1.benchmark.load_banks",return_value=(rows,[{"sha256":"bank"}])):
                result=run(FakeBackend(),m,root)
            self.assertEqual(result["total_warmup_requests"],144)
            self.assertEqual(result["total_measured_requests"],288)
            for suite in result["suites"]:
                blocks=suite["conditions"];ref=[r["model_noise_sha256"] for r in blocks[0]["measured"]]
                self.assertTrue(all([r["model_noise_sha256"] for r in b["measured"]]==ref for b in blocks))

    def test_runner_infers_phase_and_skips_gpu_for_completed_cases(self):
        from .runner import run
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);m=manifest(root);(root/"manifest.json").write_text(json.dumps(m))
            args=Namespace(output=str(root),condition="jac_anchor",stage=None,shard_index=0,shards=1,gpu_uuid="GPU-TEST",port=21999,launch=False,preflight_only=False)
            result=run(args);self.assertEqual(result["episodes"],120)
            self.assertIn("preparations.ease_insights_20261103.vlm_local_search_v1.server",result["server"])
            self.assertIn("preparations.ease_insights_20261103.vlm_local_search_v1.worker",result["worker"])
            args.launch=True
            with patch("preparations.ease_insights_20261103.vlm_local_search_v1.runner.recover_case",return_value=None), \
                patch("preparations.ease_insights_20261103.vlm_local_search_v1.worker.completed",return_value=True), \
                patch("preparations.ease_insights_20261103.vlm_local_search_v1.runner.resolve_binding",side_effect=AssertionError("Must not query GPU")):
                result=run(args)
            self.assertEqual(result["already_completed"],120)
            self.assertTrue((root/"jac_anchor/discovery_shard00of01_complete.json").exists())

    def test_valid_policy_failure_is_not_rerun(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root);(p/"complete.json").write_text('{"valid":true}')
            def forbidden():raise AssertionError("Never rerun complete science")
            result=run_with_retries(p,3,forbidden,lambda:dict(valid=True,success=False),lambda:None)
            self.assertFalse(result["success"])
            with self.assertRaises(CompletedEpisodeFinalizationError):
                run_with_retries(p,3,forbidden,lambda:(_ for _ in ()).throw(ValueError("Receipt bug")),lambda:None)

    def test_initial_gate_pairs_condition_independent_inputs(self):
        from .worker import initial_gate
        scene=dict(suite=SUITES[0],task=0,init=36)
        with tempfile.TemporaryDirectory() as root:
            a=initial_gate(root,"m","jac_anchor",scene,"input",100)
            b=initial_gate(root,"m","local_goal",scene,"input",100)
            self.assertEqual(a,b)
            with self.assertRaises(ValueError):initial_gate(root,"m","local_goal",scene,"changed",100)

    def test_prior_use_detects_legacy_alias_and_full(self):
        from .prepare import prior_use_audit,semantic_match
        self.assertEqual(semantic_match(dict(id="fixed_v12_jacobian",model="pi0",skip=list(MASKS["jac_anchor"]))),["jac_anchor"])
        self.assertEqual(semantic_match(dict(id="anything",model="pi0",skip=[])),["full_d10"])
        self.assertEqual(semantic_match(dict(id="anything",model="pi05",skip=[])),[])
        self.assertEqual(semantic_match(dict(model="pi0",mode="full")),["full_d10"])
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/"old/episodes/libero_spatial/task00/init36";p.mkdir(parents=True)
            (p/"identity.json").write_text(json.dumps(dict(case=dict(suite=SUITES[0],task=0,init=36),config=dict(id="fixed_v12_jacobian",model="pi0",skip=list(MASKS["jac_anchor"])))))
            with self.assertRaises(ValueError):prior_use_audit([root],[36])
            self.assertTrue(prior_use_audit([root],[39])["passed"])
            (p/"identity.json").write_text(json.dumps(dict(case=dict(suite=SUITES[0],task=0,init=36),config=dict(model="pi0",mode="unknown"))))
            with self.assertRaises(ValueError):prior_use_audit([root],[36])

    def test_real_legacy_receipt_adapted_in_temporary_fixture(self):
        from .common import ROOT
        from .worker import finalize, completed, episode_dir
        from preparations.idea1_v2_2.common import seed
        source=ROOT/"results/ease_insights_20261103/vlm_confirmation/fixed_v12_jacobian/episodes/libero_spatial/task00/init31"
        self.assertTrue((source/"episode_receipt.json").exists(), "Pinned real receipt fixture unavailable")
        before={p.name:sha(p) for p in source.iterdir() if p.is_file()}
        scene=dict(suite="libero_spatial",task=0,init=36)
        with tempfile.TemporaryDirectory(prefix="synthetic-local-search-receipt-") as root:
            root=Path(root);m=manifest(root);cfg=conditions()[1]
            directory=episode_dir(root,cfg["id"],scene);directory.mkdir(parents=True)
            for name in ("identity.json","result.json","complete.json","infer.jsonl","telemetry.jsonl","arrays.npz","input_0000.npz"):
                shutil.copy2(source/name,directory/name)
            identity=json.loads((directory/"identity.json").read_text())
            m["checkpoints"]["pi0"]["sha256"]=identity["protocol"]["checkpoint_sha256"]
            m["official_init_availability"]=dict(cases=[dict(suite=scene["suite"],task=scene["task"],
                init_sha256={"36":identity["init_vector_sha256"]},bddl_sha256=identity["bddl_sha256"],prompt=identity["prompt"])])
            m.pop("manifest_hash");m["manifest_hash"]=digest(m)
            # The historical observation/actions are only a CPU serialization
            # fixture. Rename identity and regenerate declared seeds for init36;
            # this adaptation never represents a new scientific observation.
            identity.update(case=scene,config=cfg,manifest_hash=m["manifest_hash"])
            identity["protocol"].update(condition_id=cfg["id"],config=cfg,manifest_hash=m["manifest_hash"],qk_reference_passed=True)
            (directory/"identity.json").write_text(json.dumps(identity))
            result=json.loads((directory/"result.json").read_text());result.update(case=scene,condition=cfg["id"],config=cfg,manifest_hash=m["manifest_hash"])
            (directory/"result.json").write_text(json.dumps(result))
            (directory/"complete.json").write_text(json.dumps(dict(valid=True,manifest_hash=m["manifest_hash"])))
            rows=[json.loads(line) for line in (directory/"infer.jsonl").read_text().splitlines()]
            for index,row in enumerate(rows):
                row["seed"]=seed(scene["suite"],scene["task"],scene["init"],index)
                row["model_noise_sha256"]=hashlib.sha256(np.random.default_rng(row["seed"]).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest()
            with np.load(directory/"input_0000.npz",allow_pickle=False) as saved: inputs={k:saved[k] for k in saved.files}
            inputs["seed"]=np.asarray(rows[0]["seed"]);np.savez_compressed(directory/"input_0000.npz",**inputs)
            rows[0]["input_file_sha256"]=sha(directory/"input_0000.npz");rows[0]["input_path"]=str(directory/"input_0000.npz")
            (directory/"infer.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            receipt=finalize(root,cfg["id"],scene,m)
            self.assertEqual(receipt["success"],result["success"])
            self.assertEqual(receipt["executed_steps"],result["actions"])
            self.assertIn("input_0000.npz",receipt["artifacts"])
            self.assertTrue(completed(root,cfg["id"],scene,m["manifest_hash"]))
            identity["bddl_sha256"]="tampered";(directory/"identity.json").write_text(json.dumps(identity))
            with self.assertRaises(ValueError):finalize(root,cfg["id"],scene,m)
            with self.assertRaises(ValueError):completed(root,cfg["id"],scene,m["manifest_hash"])
        self.assertEqual(before,{p.name:sha(p) for p in source.iterdir() if p.is_file()})

    def test_new_proposal_mapping_is_by_mask_not_desired_direction(self):
        from .prepare import read_design, DEFAULT_DESIGN
        design,mapping=read_design(DEFAULT_DESIGN)
        self.assertEqual(len(mapping),4)
        self.assertEqual(mapping["local_spatial"]["design_id"],"fixed_v12_restore14_skip13")
        self.assertTrue(design["proposal_only"])
        self.assertIn("no observed preference",mapping["local_goal"]["label_meaning"])


if __name__=="__main__":unittest.main()
