"""CPU synthetic/import tests. These fixtures are not scientific observations."""
from argparse import Namespace
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from .common import (MODULE, SUITES, GATE_SUITES, TASKS, TASK_CATEGORIES, CATEGORIES,
    CONDITIONS, DISCOVERY_PHASE, conditions, digest, sha)
from .policy import Policy
from .rollout import episode, MAX_POLICY_ACTIONS, SETTLING_ACTIONS
from .worker import completed, episode_dir, finalize, initial_gate, verify_physical_identity
from preparations.ease_insights_20261103.vlm.policy import Policy as FrozenPolicy
from preparations.ease_insights_20261103.vlm.recovery import run_with_retries, CompletedEpisodeFinalizationError
from preparations.ease_insights_20261103.vlm.test_protocol import FakeBackend, Raw


def synthetic_manifest(root):
    root = Path(root)
    protocol = root/"synthetic_protocol.json"; protocol.write_text('{"synthetic_test_only":true}')
    design = root/"synthetic_design.json"; design.write_text('{"synthetic_test_only":true}')
    value = dict(phase=DISCOVERY_PHASE, conditions=conditions(), suites=list(SUITES), tasks=list(TASKS),
        categories=list(CATEGORIES), task_categories=TASK_CATEGORIES,
        case_table=[dict(suite="libero_90", task=t) for t in TASKS], output_root=str(root),
        planned_conditions=list(CONDITIONS), cases_per_condition=72, total_discovery_episodes=432,
        discovery_inits=[36,37,38], science=dict(policy_action_limit=400, settling_actions=10),
        protocol_evidence=dict(path=str(protocol),sha256=sha(protocol)),
        design_evidence=dict(path=str(design),sha256=sha(design)), local_runtime_sources={}, source_files={},
        checkpoints={"pi0":dict(sha256="synthetic-checkpoint",identity_artifacts={})}, technical_max_attempts=3)
    value["manifest_hash"] = digest(value)
    return value


class AuditedBackend(FakeBackend):
    def prepare_residual(self, obs, instruction, **kwargs):
        context, timing = super().prepare_residual(obs, instruction, **kwargs)
        skip = list(kwargs.get("skip", ()))
        context["execution_audit"] = dict(kv_only_layers=skip,
            qkv_layers=[i for i in range(1,19) if i not in skip],
            hidden_update_layers=[i for i in range(1,18) if i not in skip],
            captured_layers=list(kwargs.get("capture_layers", ())), patched_layers=skip,
            grad_enabled=False, selector_calls=0)
        return context, timing

    def sample(self, context, seed, steps):
        self.steps.append(steps)
        return np.zeros((50,7),np.float32), Raw(), dict(nodes=[1-i/steps for i in range(steps+1)])


class SyntheticEnvironment:
    def __init__(self, success_after=None):
        controller = SimpleNamespace(input_min=np.full(6,-1.), input_max=np.ones(6),
            output_max=np.asarray([.05,.05,.05,.5,.5,.5]), output_min=np.asarray([-.05,-.05,-.05,-.5,-.5,-.5]))
        self.env = SimpleNamespace(control_freq=20., robots=[SimpleNamespace(controller=controller)],
                                   sim=SimpleNamespace(data=SimpleNamespace(time=0.)))
        self.success_after = success_after
        self.steps = 0; self.resets = 0; self.seed_value = None

    def observation(self):
        return dict(robot0_eef_pos=np.asarray([float(self.steps),0.,0.]),
                    robot0_eef_quat=np.asarray([0.,0.,0.,1.]),
                    agentview_image=np.zeros((2,2,3),np.uint8))

    def seed(self, value): self.seed_value = value
    def reset(self): self.resets += 1; self.steps = 0
    def set_init_state(self, vector): self.vector = vector.copy(); return self.observation()

    def step(self, action):
        self.steps += 1; self.env.sim.data.time = self.steps/20.
        done = self.success_after is not None and self.steps >= SETTLING_ACTIONS+self.success_after
        return self.observation(), 0., done, {}


def synthetic_episode(root, condition="jac_anchor", task=TASKS[0], success_after=28):
    """Exercise serialization, real Policy methods and fake RPC/environment only."""
    root = Path(root); manifest = synthetic_manifest(root)
    cfg = next(c for c in conditions() if c["id"] == condition)
    scene = dict(suite="libero_90",task=task,init=36)
    bddl = root/"synthetic.bddl"; bddl.write_text("synthetic test only")
    vector = np.arange(8,dtype=np.float64)
    manifest["official_init_availability"] = dict(cases=[dict(suite="libero_90",task=task,
        init_sha256={"36":hashlib.sha256(vector.tobytes()).hexdigest()}, bddl_sha256=sha(bddl), prompt="synthetic prompt")])
    manifest.pop("manifest_hash"); manifest["manifest_hash"] = digest(manifest)
    env = SyntheticEnvironment(success_after)
    released = []
    pool = SimpleNamespace(acquire=lambda suite, task:(env, SimpleNamespace(language="synthetic prompt"),bddl,None,True),
                           release=lambda suite, task:released.append((suite,task)))
    backend = AuditedBackend(); policy = Policy(backend)
    protocol = dict(manifest_hash=manifest["manifest_hash"], condition_id=condition,config=cfg,
        real_input_parity_passed=True,qk_reference_passed=True,checkpoint_sha256="synthetic-checkpoint",
        policy_action_limit=400,settling_actions=10,gpu_uuid="GPU-SYNTHETIC")
    def rpc_call(port, request):
        if request["op"] == "configure": return policy.configure(request["config"])
        if request["op"] == "health": return dict(protocol=protocol)
        if request["op"] == "predict":
            # A real pickle RPC detaches the client response from pending state.
            return copy.deepcopy(policy.predict(request["obs"],request["instruction"],request["suite"],request["seed"]))
        if request["op"] == "ack": return policy.acknowledge(request["executed"],request["quaternions"])
        raise AssertionError(request["op"])
    gates = []
    def gate(identity, request):
        assets = verify_physical_identity(manifest,scene,identity)
        self_input = __import__("preparations.idea1_v2_1.runtime_client",fromlist=["input_digest"])
        gates.append(initial_gate(root,manifest["manifest_hash"],condition,scene,
            self_input.input_digest(request["obs"],request["instruction"]),request["seed"],assets=assets))
    directory = episode_dir(root,condition,scene)
    with patch(MODULE+".rollout.official_init_states",return_value=[vector]*44):
        result = episode(directory,21999,scene,cfg,pool,manifest["manifest_hash"],rpc_call=rpc_call,initial_input_gate=gate)
    return manifest, scene, result, env, backend, released, gates


class RuntimeTests(unittest.TestCase):
    def test_original_six_mask_mechanics_and_paired_noise(self):
        for name in ("predict","acknowledge","reset","__init__"):
            self.assertIs(getattr(Policy,name),getattr(FrozenPolicy,name))
        noises = []
        for cfg in conditions():
            backend = FakeBackend(); policy = Policy(backend); policy.configure(cfg); rows=[]
            for index in range(6):
                row=policy.predict(index,"p","libero_90",100+index); rows.append(row); policy.acknowledge(row["actions"][:5])
            self.assertEqual(backend.steps,[10]*6 if cfg["id"]=="full_d10" else [10,2,2,10,2,2])
            if cfg["skip"]:
                self.assertEqual(backend.calls[4]["history"],dict.fromkeys(tuple(cfg["skip"]),3))
                self.assertEqual(backend.calls[4]["older"],dict.fromkeys(tuple(cfg["skip"]),0))
                self.assertEqual([backend.calls[i]["ratio"] for i in (4,5)],[1/3,2/3])
            noises.append([row["model_noise_sha256"] for row in rows])
            policy.configure(cfg); self.assertEqual(policy.history,[])
        self.assertEqual(noises,[noises[0]]*6)

    def test_pure_runner_shape_and_launch_refusal(self):
        from .runner import run
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); manifest=synthetic_manifest(root)
            (root/"manifest.json").write_text(json.dumps(manifest))
            args=Namespace(output=str(root),condition="jac_anchor",stage=None,shard_index=0,shards=2,
                           gpu_uuid="GPU-SYNTHETIC",port=21999,launch=False,preflight_only=False)
            with patch(MODULE+".runner.resolve_binding",side_effect=AssertionError("No GPU query")):
                prepared=run(args)
                self.assertEqual(prepared["episodes"],36)
                self.assertTrue({"args","output","manifest","scenes","marker","logs","server","worker","plan"} <= set(prepared))
                self.assertIn(MODULE+".server",prepared["server"]); self.assertIn(MODULE+".worker",prepared["worker"])
                self.assertEqual(prepared["plan"]["identity"]["policy_action_limit"],400)
                json.dumps(prepared["plan"])
                args.launch=True
                with self.assertRaisesRegex(RuntimeError,"borrowed-resource"): run(args)
            self.assertFalse((root/"jac_anchor").exists())

    def test_gate_bank_is_finite_and_separate_from_evaluation(self):
        from .server import finite_bank_scope, finite_gate, finite_benchmark
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); manifest=synthetic_manifest(root)
            scope=finite_bank_scope(manifest)
            self.assertEqual(scope["gate_bank_suites"],list(GATE_SUITES))
            self.assertEqual(scope["evaluation_suites"],["libero_90"])
            self.assertFalse(scope["gate_covers_new_case_universe"])
            with patch("preparations.ease_insights_20261103.vlm_local_search_v1.gate.run",return_value=dict(passed=True)) as gate:
                self.assertTrue(finite_gate(None,manifest,root)["passed"]); gate.assert_called_once_with(None,manifest,root)
            with patch("preparations.ease_insights_20261103.vlm_local_search_v1.benchmark.run",return_value=dict(total_measured_requests=288)):
                self.assertIn("within_suite_scope",finite_benchmark(None,manifest,root))

    def test_uniform_400_policy_actions_plus_ten_settling_in_both_categories(self):
        from preparations.idea1_v3_1.common import LIMITS as old_limits
        original=dict(old_limits)
        for task in (TASKS[0],TASKS[12]):
            with tempfile.TemporaryDirectory() as root:
                manifest,scene,result,env,backend,released,gates=synthetic_episode(root,task=task,success_after=None)
                self.assertEqual((result["actions"],env.steps,result["predictions"]),(400,410,80))
                self.assertEqual((env.resets,env.seed_value),(1,7)); self.assertEqual(released,[("libero_90",task)])
                self.assertEqual(len(gates),1); self.assertFalse(result["success"])
                receipt=finalize(root,"jac_anchor",scene,manifest)
                self.assertFalse(receipt["success"]); self.assertEqual(receipt["executed_steps"],400)
                self.assertTrue(completed(root,"jac_anchor",scene,manifest["manifest_hash"]))
        self.assertEqual(old_limits,original); self.assertNotIn("libero_90",old_limits)
        for wrong in (399,520):
            with self.assertRaises(ValueError):
                episode("unused",1,dict(suite="libero_90"),conditions()[0],None,"m",max_policy_actions=wrong)

    def test_synthetic_receipt_history_actions_and_official_assets_tamper(self):
        with tempfile.TemporaryDirectory() as root:
            manifest,scene,result,env,backend,released,gates=synthetic_episode(root)
            directory=episode_dir(root,"jac_anchor",scene)
            receipt=finalize(root,"jac_anchor",scene,manifest)
            self.assertTrue(receipt["success"]); self.assertEqual(receipt["executed_steps"],28)
            self.assertFalse(receipt["ae_audit"]["actual_NFE_exported"])
            rows=[json.loads(s) for s in (directory/"infer.jsonl").read_text().splitlines()]
            rows[4]["route"]["history_steps"]=[0,14]
            (directory/"infer.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            with self.assertRaisesRegex(ValueError,"history"): finalize(root,"jac_anchor",scene,manifest)
            with self.assertRaises(ValueError): completed(root,"jac_anchor",scene,manifest["manifest_hash"])
            rows[4]["route"]["history_steps"]=[0,15]
            (directory/"infer.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
            telemetry=[json.loads(s) for s in (directory/"telemetry.jsonl").read_text().splitlines()]
            telemetry[1]["action"][0]=1.
            (directory/"telemetry.jsonl").write_text("".join(json.dumps(row)+"\n" for row in telemetry))
            with self.assertRaisesRegex(ValueError,"Executed action"): finalize(root,"jac_anchor",scene,manifest)
            identity=json.loads((directory/"identity.json").read_text()); identity["prompt"]="wrong"
            with self.assertRaises(ValueError): verify_physical_identity(manifest,scene,identity)

    def test_initial_physical_inputs_pair_across_conditions(self):
        scene=dict(suite="libero_90",task=TASKS[0],init=36)
        with tempfile.TemporaryDirectory() as root:
            assets=dict(prompt="p",bddl_sha256="b",init_vector_sha256="i",env_seed=7,settling=10,policy_action_limit=400,control_hz=20.)
            a=initial_gate(root,"m","jac_anchor",scene,"input",7,assets=assets)
            b=initial_gate(root,"m","local_goal",scene,"input",7,assets=assets)
            self.assertEqual(a,b)
            for mutation in (dict(assets,prompt="changed"),dict(assets,init_vector_sha256="changed")):
                with self.assertRaises(ValueError): initial_gate(root,"m","local_goal",scene,"input",7,assets=mutation)
            with self.assertRaises(ValueError): initial_gate(root,"m","local_goal",scene,"input",8,assets=assets)

    def test_failed_policy_outcome_is_never_scientifically_retried(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root); (path/"complete.json").write_text('{"valid":true}')
            def forbidden(): raise AssertionError("No retry of complete science")
            receipt=run_with_retries(path,3,forbidden,lambda:dict(valid=True,success=False),lambda:None)
            self.assertFalse(receipt["success"])
            with self.assertRaises(CompletedEpisodeFinalizationError):
                run_with_retries(path,3,forbidden,lambda:(_ for _ in ()).throw(ValueError("receipt repair")),lambda:None)

    def test_cross_service_records_new_and_reference_scientific_identities(self):
        from .runner import cross_service
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); manifest=synthetic_manifest(root)
            (root/"manifest.json").write_text(json.dumps(manifest))
            logs=root/"service1"; logs.mkdir(); np.savez(logs/"gate_actions.npz",synthetic=np.zeros((50,32),np.float32))
            receipt=cross_service(root,logs,manifest["manifest_hash"])
            self.assertTrue(receipt["passed"]); self.assertEqual(receipt["scientific_manifest_hash"],manifest["manifest_hash"])
            self.assertFalse(receipt["new_scientific_protocol_unchanged_from_reference"])
            logs2=root/"service2"; logs2.mkdir(); np.savez(logs2/"gate_actions.npz",synthetic=np.ones((50,32),np.float32))
            with self.assertRaises(RuntimeError): cross_service(root,logs2,manifest["manifest_hash"])

    def test_external_reference_pins_retain_original_manifest_and_protocol(self):
        from .runner import cross_service
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); manifest=synthetic_manifest(root)
            source=root/"historical_numerical_fixture"; source.mkdir()
            prior=source/"manifest.json"; prior.write_text(json.dumps(dict(manifest_hash="historical-scientific-identity")))
            protocol=source/"protocol.json"; protocol.write_text(json.dumps(dict(manifest_hash="historical-scientific-identity",
                real_input_parity_passed=True,qk_reference_passed=True,checkpoint_sha256="synthetic-checkpoint")))
            npz=source/"gate_actions.npz"; np.savez(npz,synthetic=np.zeros((50,32),np.float32))
            manifest["preflight"]=dict(cross_service_reference=dict(path=str(npz),sha256=sha(npz),
                manifest_hash="historical-scientific-identity",reference_manifest=dict(path=str(prior),sha256=sha(prior)),
                source_protocol=dict(path=str(protocol),sha256=sha(protocol))))
            manifest.pop("manifest_hash"); manifest["manifest_hash"]=digest(manifest)
            (root/"manifest.json").write_text(json.dumps(manifest))
            before={p.name:sha(p) for p in source.iterdir()}
            logs=root/"new_service"; logs.mkdir(); np.savez(logs/"gate_actions.npz",synthetic=np.zeros((50,32),np.float32))
            receipt=cross_service(root,logs,manifest["manifest_hash"])
            self.assertEqual(receipt["reference_identity"]["source_manifest_hash"],"historical-scientific-identity")
            self.assertEqual(receipt["scientific_manifest_hash"],manifest["manifest_hash"])
            self.assertEqual(before,{p.name:sha(p) for p in source.iterdir()})
            protocol.write_text('{"manifest_hash":"changed"}')
            with self.assertRaises(ValueError): cross_service(root,logs,manifest["manifest_hash"])

    def test_runtime_imports_do_not_load_torch_libero_or_rendering(self):
        code="import importlib,sys; [importlib.import_module('"+MODULE+".'+n) for n in ('policy','rollout','worker','server','runner')]; assert 'torch' not in sys.modules; assert 'libero' not in sys.modules; assert 'mujoco' not in sys.modules"
        subprocess.run([sys.executable,"-c",code],check=True,capture_output=True,text=True)


if __name__ == "__main__":
    unittest.main()
