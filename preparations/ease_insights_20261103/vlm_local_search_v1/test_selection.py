"""CPU-only fixtures for the frozen selection rule and artifact boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from .common import CONDITIONS, MASKS, SUITES, atomic_json, conditions, digest, sha
from .selection import (CONDITION_IDS, analyze_discovery, freeze, key, read_selection,
    select_pair, usable)


def fixture_records():
    manifest = dict(planned_conditions=list(CONDITIONS), conditions=conditions(), suites=list(SUITES),
        tasks=list(range(10)), discovery_inits=[36, 37, 38])
    records = {}
    for cid in CONDITIONS:
        for suite in SUITES:
            for task in range(10):
                for init in (36, 37, 38):
                    success = cid in ("full_d10", "jac_anchor", "local_goal")
                    if cid == "jac_anchor" and suite == "libero_goal" and task < 4:
                        success = False
                    if cid == "local_goal" and suite == "libero_spatial" and task < 4:
                        success = False
                    records[cid, (suite, task, init)] = dict(success=success)
    costs = dict(valid_services=1, median_full_over_policy_cycle={cid: 1 if cid == "full_d10" else 1.6 for cid in CONDITIONS})
    return manifest, records, costs


def fixture_protocol(base):
    """A private copy of the protocol with destinations inside a temporary tree."""
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    protocol["discovery"]["root"] = str(base / "discovery")
    protocol["confirmation"]["root"] = str(base / "confirmation")
    path = base / "fixture_protocol.json"
    atomic_json(path, protocol)
    return path


def fixture_design(base):
    source = base / "fake_historical_input.txt"
    source.write_text("CPU fixture only; no model or environment was run.\n")
    evidence = dict(path=str(source), sha256=sha(source))
    value = dict(proposal_only=True, new_initial_state_outcomes_read=False,
        constraints=dict(anchor=list(MASKS["jac_anchor"])), inputs=dict(fixture=evidence), generator=evidence,
        proposals=[dict(id=cid, skip=list(mask), proposal_suite="fixture_provenance")
            for cid, mask in MASKS.items() if cid != "jac_anchor"])
    value["receipt_hash"] = digest(value)
    path = base / "fixture_design.json"
    atomic_json(path, value)
    return path


def fixture_base_manifest(base):
    value = dict(checkpoints=dict(pi0=dict(sha256="fixture-checkpoint", identity_artifacts={})), bank={}, environment={}, noise=dict(condition_independent=True),
        science=dict(H=50, R=5, draft_D=2, refresh_D=10, retained_layers=12, depth=18),
        quality_environment_reuse={}, technical_max_attempts=1, preflight={}, source_files={})
    value["manifest_hash"] = digest(value)
    path = base / "fake_base_manifest.json"
    atomic_json(path, value)
    return path


def fixture_official():
    return dict(passed=True, cases=[dict(suite=s, task=t, bddl_sha256="fixture-bddl", prompt="fixture " + s,
        init_sha256={str(i): "fixture-vector-%d" % i for i in range(36, 44)}) for s in SUITES for t in range(10)])


def prepare_fixture(base, stage="discovery", selection=None):
    """Exercise the real prepare path; only external init/inventory reads are mocked."""
    from .prepare import prepare
    protocol = base / "fixture_protocol.json"
    design = base / "fixture_design.json"
    reference = base / "fake_base_manifest.json"
    if not protocol.exists():
        fixture_protocol(base)
        fixture_design(base)
        fixture_base_manifest(base)
    output = base / ("discovery" if stage == "discovery" else "confirmation")
    with patch("preparations.ease_insights_20261103.vlm_local_search_v1.prepare.prior_use_audit",
            return_value=dict(passed=True, fixture=True)), patch(
            "preparations.ease_insights_20261103.vlm_local_search_v1.prepare.subprocess.check_output",
            return_value=json.dumps(fixture_official()) + "\n"):
        prepare(output, stage, protocol, design, selection, base / "discovery", reference)
    return output, json.loads((output / "manifest.json").read_text())


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
        bddl_sha256="fixture-bddl", prompt=prompt, env_seed=7, settling=10)
    result = dict(valid=True, manifest_hash=manifest["manifest_hash"], case=case, condition=cfg["id"],
        config=cfg, success=bool(success), predictions=1, counts=dict(full=1, draft=0), actions=5, wall_s=.2)
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


def populate_stage(root, manifest, confirmation=False):
    inits = manifest["confirmation_inits" if confirmation else "discovery_inits"]
    _, discoveries, _ = fixture_records()
    for cfg in manifest["conditions"]:
        cid = cfg["id"]
        if cid not in manifest["planned_conditions"]:
            continue
        for suite in SUITES:
            for task in range(10):
                for init in inits:
                    case = dict(suite=suite, task=task, init=init)
                    if confirmation:
                        success = not (init == 39 and ((cid == "jac_anchor" and suite == "libero_goal") or
                            (cid == "local_goal" and suite == "libero_spatial")))
                    else:
                        success = discoveries[cid, key(case)]["success"]
                    write_episode(root, manifest, cfg, case, success)
        stage = "confirm" if confirmation else "discovery"
        atomic_json(root / cid / (stage + "_shard00of01_complete.json"),
            dict(complete=True, condition=cid, manifest_hash=manifest["manifest_hash"]))


def write_cost_service(root, manifest, complete=True):
    directory = root / "full_d10" / "runs" / "cpu_fixture"
    directory.mkdir(parents=True, exist_ok=True)
    shared = dict(manifest_hash=manifest["manifest_hash"])
    atomic_json(directory / "protocol.json", dict(shared, condition_id="full_d10", real_input_parity_passed=True, gpu_uuid="CPU-FIXTURE-NO-GPU",
        qk_reference_passed=True, config=manifest["conditions"][0], checkpoint_sha256=manifest["checkpoints"]["pi0"]["sha256"]))
    atomic_json(directory / "gate.json", dict(shared, passed=True))
    atomic_json(directory / "cross_service.json", dict(shared, passed=True))
    suites = []
    for suite in SUITES:
        blocks = []
        for cid in CONDITION_IDS:
            measured = [dict(round=r, seed=r, model_noise_sha256=str(r),
                refresh=cid == "full_d10" or r % 3 == 0,
                policy_wall_ms=10. if cid == "full_d10" or r % 3 == 0 else 4.) for r in range(6, 18)]
            blocks.append(dict(condition=cid, warmup=[{}] * 6, measured=measured))
        suites.append(dict(suite=suite, conditions=blocks))
    atomic_json(directory / "fixed_input_cost.json", dict(shared, suites=suites))
    atomic_json(directory / "result.json", dict(shared, complete=complete, condition="full_d10", total_wall_s=2., closed_loop_wall_s=1.))


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest, self.records, self.costs = fixture_records()

    def test_selects_opposite_usable_paths_and_reports_all_comparisons(self):
        value = select_pair(self.manifest, self.records, self.costs, True)
        self.assertEqual(value["status"], "selected")
        self.assertEqual(value["selected"]["paths"], ["jac_anchor", "local_goal"])
        self.assertEqual([d["suite"] for d in value["selected"]["directions"]], ["libero_spatial", "libero_goal"])
        self.assertEqual(len(value["all_pair_suite_comparisons"]), 40)
        self.assertEqual(len(value["all_ordered_suite_crossings"]), 120)

    def test_incomplete_and_missing_cost_cannot_select(self):
        self.assertEqual(select_pair(self.manifest, {}, self.costs, False)["status"], "incomplete_discovery")
        self.assertEqual(select_pair(self.manifest, self.records, {}, True)["status"], "missing_valid_cost_probe")

    def test_cost_threshold_blocks_qualified_pair(self):
        self.costs["median_full_over_policy_cycle"]["jac_anchor"] = 1.149
        self.assertEqual(select_pair(self.manifest, self.records, self.costs, True)["status"], "no_candidate")

    def test_single_task_gap_is_not_eligible(self):
        for case in list(self.records):
            cid, (suite, task, init) = case
            if cid in ("jac_anchor", "local_goal"):
                self.records[case]["success"] = not (task == 0 and (
                    (cid == "jac_anchor" and suite == "libero_goal") or
                    (cid == "local_goal" and suite == "libero_spatial")))
        self.assertEqual(select_pair(self.manifest, self.records, self.costs, True)["status"], "no_candidate")

    def test_rate_boundaries_are_exact(self):
        self.assertTrue(usable(102, 120, 120))
        self.assertFalse(usable(101, 120, 120))
        self.assertTrue(usable(90, 100, 120))
        self.assertFalse(usable(89, 100, 120))

    def test_lexical_tie_break_is_independent_of_candidate_insertion_order(self):
        for cid, case in list(self.records):
            if cid == "local_object":
                self.records[cid, case] = copy.deepcopy(self.records["local_goal", case])
        first = select_pair(self.manifest, self.records, self.costs, True)
        self.manifest["planned_conditions"] = list(reversed(self.manifest["planned_conditions"]))
        second = select_pair(self.manifest, self.records, self.costs, True)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["selected"]["pair"], ["jac_anchor", "local_goal"])

    def test_full_artifact_integration_and_tamper_gate(self):
        from .analyze import analyze_confirmation
        with tempfile.TemporaryDirectory(prefix="ease-local-search-cpu-") as directory:
            base = Path(directory)
            discovery, manifest = prepare_fixture(base)
            populate_stage(discovery, manifest)
            write_cost_service(discovery, manifest, complete=False)
            report = analyze_discovery(discovery)
            self.assertTrue(report["complete"])
            self.assertEqual(report["valid_episodes"], 720)
            self.assertEqual(report["selection"]["status"], "missing_valid_cost_probe")
            with self.assertRaises(ValueError):
                freeze(discovery / "analysis", report)
            write_cost_service(discovery, manifest)
            report = analyze_discovery(discovery)
            selection = freeze(discovery / "analysis", report)
            self.assertEqual(read_selection(selection)["selection"]["status"], "selected")
            self.assertEqual(freeze(discovery / "analysis", report), selection)
            confirmation, confirm_manifest = prepare_fixture(base, "confirm", selection)
            populate_stage(confirmation, confirm_manifest, confirmation=True)
            analysis = analyze_confirmation(confirmation, base / "empty_contention", replicates=20000)
            self.assertTrue(analysis["complete"])
            self.assertEqual(analysis["valid_episodes"], 600)
            self.assertEqual(analysis["conclusion"], "complementarity_reproduced")
            for direction in analysis["primary_directions"]:
                self.assertEqual(direction["inference"]["primary_task_block"]["p_one_sided"], 1/1024)
                self.assertEqual(direction["inference"]["primary_holm2_p"], 2/1024)
            # Tampered data is rejected even if its completion marker survives.
            victim = next(discovery.glob("jac_anchor/episodes/*/task*/init*/infer.jsonl"))
            original = victim.read_text()
            victim.write_text(original + "{}\n")
            damaged = analyze_discovery(discovery)
            self.assertFalse(damaged["complete"])
            self.assertEqual(damaged["valid_episodes"], 719)
            with self.assertRaises(ValueError):
                freeze(discovery / "analysis", damaged)
            # Even internally rehashed files cannot change the decoded initial
            # observation behind an unchanged comparison identity/reference.
            victim.write_text(original)
            first_input = victim.parent / "input_0000.npz"
            with np.load(first_input, allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            arrays["state"][0] += 1
            np.savez(first_input, **arrays)
            first = json.loads(original)
            first["input_file_sha256"] = sha(first_input)
            victim.write_text(json.dumps(first) + "\n")
            receipt_path = victim.parent / "episode_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["artifacts"].update({name: sha(victim.parent / name) for name in ("input_0000.npz", "infer.jsonl")})
            atomic_json(receipt_path, receipt)
            rehashed = analyze_discovery(discovery)
            self.assertFalse(rehashed["complete"])
            self.assertEqual(rehashed["valid_episodes"], 719)
            self.assertTrue(any("Initial observation/noise differs" in r["error"] for r in rehashed["invalid"]))
            edited_manifest = copy.deepcopy(manifest)
            edited_manifest["analysis_sources"][next(iter(edited_manifest["analysis_sources"]))] = "changed"
            edited_manifest.pop("manifest_hash")
            edited_manifest["manifest_hash"] = digest(edited_manifest)
            atomic_json(discovery / "manifest.json", edited_manifest)
            with self.assertRaisesRegex(ValueError, "analysis sources"):
                analyze_discovery(discovery)


if __name__ == "__main__":
    unittest.main()
