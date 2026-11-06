"""CPU synthetic checks of category selection, exact inference and artifact binding."""
import copy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from .common import (ROOT, CONDITIONS, TASKS, TASK_CATEGORIES, CATEGORIES, SUITES, GATE_SUITES,
    DISCOVERY_PHASE, CONFIRMATION_PHASE, SCIENCE_FILES, SHARED_ANALYSIS_FILES,
    conditions, cases, validate_manifest, atomic_json, digest, sha)
from .selection import select_pair, usable, analyze_discovery, freeze, read_selection, completed_service_costs, key
from .analyze import task_block_signflip, bootstrap, direction_analysis, apply_verdict, check_selection, analyze_confirmation
from .prepare import prepare, prior_use_audit, read_design, read_protocol, accepted_cross_service_reference, pin_normalization_asset


def fixture_records(inits=(36,37,38), selected=False):
    manifest=dict(planned_conditions=list(CONDITIONS),conditions=conditions(),suites=list(SUITES),tasks=list(TASKS),
        categories=list(CATEGORIES),task_categories=dict(TASK_CATEGORIES),discovery_inits=list(inits))
    records={}
    for cid in CONDITIONS:
        for index,task in enumerate(TASKS):
            for init in inits:
                success=cid in ("full_d10","jac_anchor","local_goal")
                if cid=="jac_anchor" and task in TASKS[12:14]:success=False
                if cid=="local_goal" and task in TASKS[:2]:success=False
                records[cid,("libero_90",task,init)]=dict(success=success)
    costs=dict(valid_services=1,median_full_over_policy_cycle={c:1. if c=="full_d10" else 1.6 for c in CONDITIONS})
    return manifest,records,costs


def fixture_manifest(root,confirmation=False,selection=None):
    protocol=Path(__file__).with_name("protocol.json")
    from .prepare import DEFAULT_DESIGN
    local={str((Path(__file__).parent/n).relative_to(ROOT)):sha(Path(__file__).parent/n) for n in SCIENCE_FILES}
    local.update({n:sha(ROOT/n) for n in SHARED_ANALYSIS_FILES})
    value=dict(phase=CONFIRMATION_PHASE if confirmation else DISCOVERY_PHASE,output_root=str(root),
        conditions=conditions(),planned_conditions=["full_d10","jac_anchor","local_goal"] if confirmation else list(CONDITIONS),
        suites=list(SUITES),tasks=list(TASKS),categories=list(CATEGORIES),task_categories=dict(TASK_CATEGORIES),
        case_table=[dict(suite="libero_90",task=t) for t in TASKS],science=dict(policy_action_limit=400,settling_actions=10),
        cases_per_condition=120 if confirmation else 72,
        protocol_evidence=dict(path=str(protocol),sha256=sha(protocol)),
        design_evidence=dict(path=str(DEFAULT_DESIGN),sha256=sha(DEFAULT_DESIGN)),
        analysis_sources=local,source_files=local,local_runtime_sources={},
        checkpoints=dict(pi0=dict(sha256="CPU-FIXTURE",identity_artifacts={})),
        official_init_availability=dict(cases=[dict(suite="libero_90",task=t,prompt="fixture libero_90",bddl_sha256="fixture-bddl",
            init_sha256={str(i):"fixture-vector-%d"%i for i in range(36,44)}) for t in TASKS]))
    if confirmation:
        selected=read_selection(selection)
        value.update(confirmation_inits=list(range(39,44)),total_confirmation_episodes=360,
            selection_evidence=dict(path=str(selection),sha256=sha(selection),selection_hash=selected["selection_hash"],frozen_selection=selected["selection"]),
            preregistered_hypotheses=selected["selection"]["selected"]["directions"])
    else:value.update(discovery_inits=[36,37,38],total_discovery_episodes=432)
    value["manifest_hash"]=digest(value);atomic_json(root/"manifest.json",value)
    return value


def write_episode(root, manifest, cfg, case, success):
    """Tiny artifacts satisfy the real receipt/identity validator, with no simulator."""
    directory = root / cfg["id"] / "episodes" / case["suite"] / ("task%02d" % case["task"]) / ("init%02d" % case["init"])
    directory.mkdir(parents=True, exist_ok=True)
    from .common import runtime_revision
    from preparations.idea1_v2_1.runtime_client import input_digest
    from preparations.idea1_v2_2.common import seed
    noise_seed = seed(case["suite"], case["task"], case["init"], 0)
    observation = dict(state=np.asarray([case["task"], case["init"]], dtype=np.float32))
    prompt = "fixture " + case["suite"]
    np.savez(directory / "input_0000.npz", **observation, instruction=prompt, seed=noise_seed)
    reference = root / "initial_observations" / case["suite"] / ("task%02d" % case["task"]) / ("init%02d.json" % case["init"])
    initial = dict(passed=True, manifest_hash=manifest["manifest_hash"], case=case,
        input_hash=input_digest(observation, prompt), seed=noise_seed, reference=str(reference))
    if not reference.exists():
        atomic_json(reference, initial)
    protocol = dict(manifest_hash=manifest["manifest_hash"], condition_id=cfg["id"],
        gpu_uuid="CPU-FIXTURE-NO-GPU", real_input_parity_passed=True, qk_reference_passed=True,
        config=cfg, checkpoint_sha256=manifest["checkpoints"]["pi0"]["sha256"])
    identity = dict(manifest_hash=manifest["manifest_hash"], case=case, config=cfg,
        protocol=protocol, init_vector_sha256="fixture-vector-%d" % case["init"],
        bddl_sha256="fixture-bddl", prompt=prompt, env_seed=7, settling=10, policy_action_limit=400)
    result = dict(valid=True, manifest_hash=manifest["manifest_hash"], case=case, condition=cfg["id"],
        config=cfg, success=bool(success), predictions=1, counts=dict(full=1, draft=0), actions=5, wall_s=.2, policy_action_limit=400, settling_actions=10)
    atomic_json(directory / "identity.json", identity)
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "complete.json", dict(valid=True, manifest_hash=manifest["manifest_hash"]))
    row = dict(route=dict(refresh=True), executed_length=5, timing=dict(policy_wall_ms=1.), rpc_ms=1.5,
        input_file_sha256=sha(directory / "input_0000.npz"), input_hash=initial["input_hash"], seed=noise_seed)
    (directory / "infer.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "telemetry.jsonl").write_text("{}\n")
    (directory / "arrays.npz").write_bytes(b"CPU receipt-hash fixture; no array decoding required by analysis")
    files = ("identity.json", "result.json", "complete.json", "infer.jsonl", "telemetry.jsonl", "arrays.npz", "input_0000.npz")
    atomic_json(directory / "episode_receipt.json", dict(valid=True, manifest_hash=manifest["manifest_hash"],
        condition=cfg["id"], case=case, success=bool(success), gpu_uuid="CPU-FIXTURE-NO-GPU",
        initial_gate=initial, technical_runtime_revision=runtime_revision(manifest), artifacts={name: sha(directory / name) for name in files}))


def populate(root,manifest):
    confirmation="confirmation_inits" in manifest
    for config in manifest["conditions"]:
        if config["id"] not in manifest["planned_conditions"]:continue
        for scene in cases(manifest):
            cid=config["id"];task=scene["task"];init=scene["init"]
            if confirmation:
                success=not(init==39 and ((cid=="jac_anchor" and TASK_CATEGORIES[str(task)]==CATEGORIES[1]) or
                    (cid=="local_goal" and TASK_CATEGORIES[str(task)]==CATEGORIES[0])))
            else:
                success=cid in ("full_d10","jac_anchor","local_goal")
                if cid=="jac_anchor" and task in TASKS[12:14]:success=False
                if cid=="local_goal" and task in TASKS[:2]:success=False
            write_episode(root,manifest,config,scene,success)


def cost_service(root,manifest,complete=True):
    directory=root/"full_d10"/"runs"/"cpu-fixture";directory.mkdir(parents=True,exist_ok=True)
    shared=dict(manifest_hash=manifest["manifest_hash"])
    atomic_json(directory/"protocol.json",dict(shared,condition_id="full_d10",real_input_parity_passed=True,qk_reference_passed=True,
        gpu_uuid="CPU-FIXTURE",config=manifest["conditions"][0],checkpoint_sha256=manifest["checkpoints"]["pi0"]["sha256"]))
    atomic_json(directory/"gate.json",dict(shared,passed=True));atomic_json(directory/"cross_service.json",dict(shared,passed=True))
    groups=[]
    for suite in GATE_SUITES:
        blocks=[]
        for cid in CONDITIONS:
            measured=[dict(round=r,seed=r,model_noise_sha256=str(r),refresh=cid=="full_d10" or r%3==0,
                policy_wall_ms=10. if cid=="full_d10" or r%3==0 else 4.) for r in range(6,18)]
            blocks.append(dict(condition=cid,warmup=[{}]*6,measured=measured))
        groups.append(dict(suite=suite,conditions=blocks))
    atomic_json(directory/"fixed_input_cost.json",dict(shared,suites=groups))
    atomic_json(directory/"result.json",dict(shared,complete=complete,condition="full_d10",total_wall_s=2.,closed_loop_wall_s=1.))


class ScienceTests(unittest.TestCase):
    def setUp(self):self.manifest,self.records,self.costs=fixture_records()

    def test_fixed_candidate_protocol_and_case_shards(self):
        self.assertEqual(len(read_design()["tasks"]),24);self.assertEqual(read_protocol()["execution"]["new_scientific_episode_cap"],792)
        with tempfile.TemporaryDirectory() as d:
            manifest=fixture_manifest(Path(d));validate_manifest(manifest)
            self.assertEqual(len(cases(manifest)),72)
            first=cases(manifest,shard_index=0,shards=2);second=cases(manifest,shard_index=1,shards=2)
            self.assertEqual(len(first),36);self.assertEqual(set(map(key,first))&set(map(key,second)),set())
            self.assertEqual({tuple(sorted(c)) for c in first},{("init","suite","task")})
            manifest["task_categories"][str(TASKS[0])]=CATEGORIES[1]
            with self.assertRaises(ValueError):validate_manifest(manifest)

    def test_normalization_pin_preserved_and_tamper_rejected(self):
        from .common import read_manifest
        with tempfile.TemporaryDirectory(prefix="within-suite-norm-") as d:
            root=Path(d);asset=root/"norm_stats.json";asset.write_text('{"mean":[0.0]}\n')
            historical=root/"preflight.json"
            atomic_json(historical,dict(normalization_asset=dict(path=str(asset),sha256=sha(asset))))
            artifacts={str(historical):sha(historical),"preserved-artifact":"preserved-sha"}
            checkpoint=dict(sha256="CPU-FIXTURE",identity_artifacts=dict(artifacts))
            expected=sha(asset);pin_normalization_asset(checkpoint,historical)
            self.assertEqual(checkpoint["identity_artifacts"],dict(artifacts,**{str(asset):expected}))
            # The ordinary manifest source check now directly covers the small norm file.
            del checkpoint["identity_artifacts"]["preserved-artifact"]
            manifest=fixture_manifest(root/"discovery");manifest["checkpoints"]["pi0"]=checkpoint
            manifest.pop("manifest_hash");manifest["manifest_hash"]=digest(manifest)
            path=root/"discovery/manifest.json";atomic_json(path,manifest)
            read_manifest(path,verify_sources=True)
            asset.write_text('{"mean":[1.0]}\n')
            with self.assertRaisesRegex(ValueError,"Normalization asset SHA differs"):
                pin_normalization_asset(checkpoint,historical)
            with self.assertRaisesRegex(ValueError,"checkpoint identity artifact changed"):
                read_manifest(path,verify_sources=True)

    def test_explicit_old_reference_binding(self):
        reference=accepted_cross_service_reference()
        self.assertEqual(set(reference),{"path","sha256","manifest_hash","reference_manifest","source_protocol"})
        with patch("preparations.ease_insights_20261103.vlm_within_suite_v1.prepare.sha",return_value="changed"):
            with self.assertRaises(ValueError):accepted_cross_service_reference()

    def test_category_crossing_and_lexical_rank(self):
        decision=select_pair(self.manifest,self.records,self.costs,True)
        self.assertEqual(decision["status"],"selected");self.assertEqual(decision["selected"]["paths"],["jac_anchor","local_goal"])
        self.assertEqual([d["category"] for d in decision["selected"]["directions"]],list(CATEGORIES))
        self.assertEqual(len(decision["all_pair_category_comparisons"]),20)
        self.assertEqual(len(decision["all_ordered_category_crossings"]),20)
        for cid,case in list(self.records):
            if cid=="local_object":self.records[cid,case]=copy.deepcopy(self.records["local_goal",case])
        self.manifest["planned_conditions"]=list(reversed(CONDITIONS))
        self.assertEqual(select_pair(self.manifest,self.records,self.costs,True)["selected"]["pair"],["jac_anchor","local_goal"])

    def test_incomplete_cost_usability_and_single_task_gates(self):
        self.assertEqual(select_pair(self.manifest,{},self.costs,False)["status"],"incomplete_discovery")
        self.assertEqual(select_pair(self.manifest,self.records,{},True)["status"],"missing_valid_cost_probe")
        self.costs["median_full_over_policy_cycle"]["jac_anchor"]=1.149
        self.assertEqual(select_pair(self.manifest,self.records,self.costs,True)["status"],"no_candidate")
        self.assertTrue(usable(62,72,72));self.assertFalse(usable(61,72,72))
        self.assertTrue(usable(54,60,72));self.assertFalse(usable(53,60,72))
        self.costs["median_full_over_policy_cycle"]["jac_anchor"]=1.6
        for (cid,(_,task,init)),record in self.records.items():
            if cid in ("jac_anchor","local_goal"):
                record["success"]=not((cid=="jac_anchor" and task==TASKS[12]) or (cid=="local_goal" and task==TASKS[0]))
        self.assertEqual(select_pair(self.manifest,self.records,self.costs,True)["status"],"no_candidate")

    def test_signflip_all4096_ties_and_independent_convolution(self):
        self.assertEqual(task_block_signflip([1]*12)["p_one_sided"],1/4096)
        self.assertEqual(task_block_signflip([0]*12)["p_one_sided"],1.)
        self.assertEqual(task_block_signflip([1]*6+[0]*6)["p_one_sided"],1/64)
        values=[-2,1,0,5,1,-1,2,0,3,1,0,1]
        distribution={0:1}
        for value in values:
            updated={}
            for count,n in distribution.items():
                for delta in (-value,value):updated[count+delta]=updated.get(count+delta,0)+n
            distribution=updated
        test=task_block_signflip(values)
        self.assertEqual(test["extreme_assignments"],sum(n for count,n in distribution.items() if count>=sum(values)))
        with self.assertRaises(ValueError):task_block_signflip([1]*10)

    def test_bootstrap12task_shape_seed_and_linear(self):
        values=[1]*30+[0]*30;task_ids=[t for t in range(12) for _ in range(5)]
        interval=bootstrap(values,task_ids)
        self.assertEqual(interval,bootstrap(values,task_ids));self.assertEqual(interval["case"]["seed"],20261105)
        self.assertEqual(interval["task_cluster"]["clusters"],12);self.assertEqual(interval["case"]["quantile_method"],"linear")
        self.assertGreater(interval["task_cluster"]["low"],0)
        with self.assertRaises(ValueError):bootstrap(values[:50],task_ids[:50])

    def test_freeze_requires_explicit_review_and_finite_prior_use(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(ValueError,"review"):
                prepare(root)
            atomic_json(root/"manifest.json",dict(tasks=list(TASKS),suites=["libero_90"],discovery_inits=[36,37,38]))
            audit=prior_use_audit([root],list(range(36,44)));self.assertTrue(audit["passed"])
            atomic_json(root/"identity.json",dict(case=dict(suite="libero_90",task=TASKS[0],init=36),success=False))
            audit=prior_use_audit([root],list(range(36,44)),raise_on_collision=False)
            self.assertFalse(audit["passed"]);self.assertEqual(len(audit["matching_existing_outcomes"]),1)
            with self.assertRaises(ValueError):prior_use_audit([root],list(range(36,44)))

    def test_complete_artifact_pipeline_cost_scope_frozen_pair_and_tamper(self):
        with tempfile.TemporaryDirectory(prefix="within-suite-cpu-") as d:
            base=Path(d);disc=base/"discovery";manifest=fixture_manifest(disc);populate(disc,manifest)
            cost_service(disc,manifest,complete=False)
            report=analyze_discovery(disc);self.assertTrue(report["complete"]);self.assertEqual(report["valid_episodes"],432)
            self.assertEqual(report["selection"]["status"],"missing_valid_cost_probe")
            with self.assertRaises(ValueError):freeze(disc/"analysis",report)
            cost_service(disc,manifest);report=analyze_discovery(disc)
            self.assertEqual(report["fixed_input_cost"]["valid_services"],1)
            path=freeze(disc/"analysis",report);self.assertEqual(freeze(disc/"analysis",report),path)
            confirm=base/"confirmation";heldout=fixture_manifest(confirm,True,path);populate(confirm,heldout)
            result=analyze_confirmation(confirm,base/"empty_contention")
            self.assertEqual(result["valid_episodes"],360);self.assertEqual(result["conclusion"],"complementarity_reproduced")
            for direction in result["primary_directions"]:
                self.assertEqual(direction["n_paired"],60);self.assertEqual(len(direction["by_task"]),12)
                self.assertEqual(direction["inference"]["primary_task_block"]["p_one_sided"],1/4096)
                self.assertEqual(direction["inference"]["primary_holm2_p"],2/4096)
            wrong=copy.deepcopy(heldout);wrong["preregistered_hypotheses"][0]["category"]=CATEGORIES[1]
            with self.assertRaises(ValueError):check_selection(wrong)
            victim=next(disc.glob("jac_anchor/episodes/*/task*/init*/infer.jsonl"));original=victim.read_text();victim.write_text(original+"{}\n")
            damaged=analyze_discovery(disc);self.assertFalse(damaged["complete"]);self.assertEqual(damaged["valid_episodes"],431)
            with self.assertRaises(ValueError):freeze(disc/"analysis",damaged)
            # Rehashing altered input artifacts cannot bypass the paired-input contract.
            victim.write_text(original)
            saved_input=victim.parent/"input_0000.npz"
            with np.load(saved_input,allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
            arrays["state"][0]+=1;np.savez(saved_input,**arrays)
            first=json.loads(original);first["input_file_sha256"]=sha(saved_input)
            victim.write_text(json.dumps(first)+"\n")
            receipt_path=victim.parent/"episode_receipt.json";receipt=json.loads(receipt_path.read_text())
            receipt["artifacts"].update({name:sha(victim.parent/name) for name in ("input_0000.npz","infer.jsonl")})
            atomic_json(receipt_path,receipt)
            damaged=analyze_discovery(disc);self.assertFalse(damaged["complete"]);self.assertEqual(damaged["valid_episodes"],431)
            self.assertTrue(any("Initial observation/noise differs" in row["error"] for row in damaged["invalid"]))
            gates=result["overall_candidate_eligibility"];gates["jac_anchor"]["usable"]=False
            self.assertEqual(apply_verdict(result["primary_directions"],gates),"not_reproduced")


if __name__=="__main__":unittest.main()
