"""Isolated CPU broker tests: temporary files and fake science manifests only."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import gpu_resource_campaign as campaign


class SimulatedCrash(Exception):
    pass


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ease-campaign-cpu-")
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name) / "project"
        self.experiment = self.project / "results/ease_insights_20261103"
        self.experiment.mkdir(parents=True)
        self.python = self.project / "policies/realtime-vla-flash/.venv/bin/python"
        for name in campaign.SOURCE_NAMES:
            path = self.project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("CPU fixture source: " + name)
        for name, value in (("PROJECT", self.project), ("EXPERIMENT", self.experiment),
                            ("MODEL_PYTHON", self.python)):
            started = patch.object(campaign, name, value)
            started.start()
            self.addCleanup(started.stop)
        self.outputs = {stage: self.experiment / "vlm_local_search_v1" / directory
                        for stage, directory in (("discovery", "discovery"), ("confirm", "confirmation"))}
        conditions = ["full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long"]
        self.manifests = {}
        for stage, output in self.outputs.items():
            output.mkdir(parents=True)
            (output / "manifest.json").write_text("fake manifest; never import science")
            self.manifests[str(output)] = dict(
                phase="local_search_discovery_v1" if stage == "discovery" else "local_search_confirmation_v1",
                manifest_hash=campaign.digest(stage.encode()), planned_conditions=conditions, output_root=str(output))
        self.queue_path = self.experiment / "queue.json"
        self.state_path = self.experiment / "jobs_state.json"
        self.root = self.experiment / "campaign"
        self.reader_calls = []

    def manifest_reader(self, output):
        self.reader_calls.append(str(output))
        stage = next(stage for stage, path in self.outputs.items() if path == output)
        return copy.deepcopy(self.manifests[str(output)]), stage

    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(campaign.encode(value))

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_bytes())

    def plan(self):
        return dict(schema=campaign.PLAN_SCHEMA, campaign_root=str(self.root),
                    allowed_outputs=[str(self.outputs[stage]) for stage in ("discovery", "confirm")],
                    source_hashes={name: campaign.digest((self.project / name).read_bytes())
                                   for name in campaign.SOURCE_NAMES}, hold_until_explicit_release=True)

    def job(self, condition="full_d10", stage="discovery", wrapped=True, job_id=None):
        output = self.outputs[stage]
        inner = [str(self.python), "-m", campaign.RUNNER, "--output", str(output),
                 "--condition", condition, "--gpu-uuid", "{gpu_uuid}", "--port", "{port}", "--launch"]
        logical = job_id or stage + "_" + condition
        job = dict(id=logical, argv=list(inner),
                   complete_file=str(output / condition / f"{stage}_shard00of01_complete.json"),
                   depends_on=[], ready_files=[str(output / "manifest.json")],
                   manifest_hash=self.manifests[str(output)]["manifest_hash"],
                   scientific_phase=self.manifests[str(output)]["phase"])
        if wrapped:
            job["argv"] = [str(self.python), "-m", campaign.GATE, "--gpu-uuid", "{gpu_uuid}",
                           "--status-file", str(self.experiment / "gates" / (logical + ".json")), "--", *inner]
            job["launch_gate"] = dict(original_argv=list(inner))
            job["env"] = {"EASE_GPU_GATE_IDLE_SECONDS": "25", "EASE_GPU_GATE_POLL_SECONDS": "2"}
        return job

    def initial_queue(self):
        jobs = [self.job(condition) for condition in
                ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")]
        # A completed science row precedes slots in the old queue. Slots must
        # still be published first; prepare must not silently starve pending
        # noncampaign jobs, which have a separate rejection test below.
        future = self.job("full_d10", "confirm")
        future["ready_files"] = [str(self.experiment / "not-created-selection.json")]
        jobs.insert(0, future)
        return dict(schema="cpu-test-queue", jobs=jobs), {
            "discovery_jac_anchor": {"status": "complete"}, "confirm_full_d10": {"status": "complete"}}

    def prepare(self, queue=None, states=None, proofs=None):
        if queue is None:
            queue, initial_states = self.initial_queue()
            states = initial_states if states is None else states
        self.write(self.queue_path, queue)
        self.write(self.state_path, states or {})
        return campaign.prepare(self.root, self.queue_path, self.state_path,
                                manifest_reader=self.manifest_reader, waiting_gate_proofs=proofs)

    def deployed(self, candidates=None):
        self.root.mkdir()
        for directory in ("tasks", "claims", "outcomes", "transactions", "sessions"):
            (self.root / directory).mkdir()
        plan = self.plan()
        self.write(self.root / "plan.json", plan)
        slots = campaign.slot_jobs(plan)
        self.write(self.root / "deployed.json", dict(
            plan_sha256=campaign.digest((self.root / "plan.json").read_bytes()),
            slot_job_ids=[job["id"] for job in slots]))
        candidates = candidates if candidates is not None else [self.job("full_d10", "confirm")]
        queue = dict(schema="cpu-test-queue", jobs=slots + copy.deepcopy(candidates))
        states = {job["id"]: {"status": "running"} for job in slots}
        states.update({job["id"]: {"status": "pending"} for job in candidates})
        self.write(self.queue_path, queue)
        self.write(self.state_path, states)
        return queue, states, candidates

    def publish(self):
        return campaign.publish(self.root, self.queue_path, self.state_path, manifest_reader=self.manifest_reader)

    def commits(self):
        return list((self.root / "transactions").glob("dispatch_*/committed.json"))

    def test_unwrapper_preserves_exact_scientific_argv(self):
        plain, wrapped = self.job(wrapped=False), self.job()
        self.assertEqual(campaign.unwrapped(plain), plain["argv"])
        self.assertEqual(campaign.unwrapped(wrapped), plain["argv"])
        task = campaign.task_for(wrapped, self.plan(), "initial", self.manifest_reader)
        self.assertEqual(task["argv"], plain["argv"])
        self.assertEqual(task["env"], {})
        self.assertEqual(task["schema"], campaign.TASK_SCHEMA)
        self.assertEqual(task["activation_file"], str(self.root / "transactions/initial/committed.json"))

    def test_unknown_wrapper_interpreter_and_changed_original_evidence_are_rejected(self):
        variants = []
        for index, value in ((0, "/tmp/arbitrary-executable"), (3, "--other-gpu"),
                             (4, "GPU-other"), (5, "--other-status"), (7, "--not-separator")):
            changed = self.job()
            changed["argv"][index] = value
            variants.append(changed)
        changed = self.job(); changed["launch_gate"]["original_argv"][-1] = "--not-launch"
        variants.append(changed)
        for job in variants:
            with self.subTest(argv=job["argv"]), self.assertRaises(RuntimeError):
                campaign.unwrapped(job)

    def test_runner_argv_rejects_duplicate_unknown_missing_and_concrete_bindings(self):
        original = self.job(wrapped=False)["argv"]
        variants = [original + ["--launch"], original + ["--evil", "x"],
                    original + ["--condition", "local_goal"], original[:-1],
                    ["/tmp/other-python", *original[1:]]]
        for flag, value in (("--gpu-uuid", "GPU-concrete"), ("--port", "1234")):
            changed = list(original); changed[changed.index(flag) + 1] = value; variants.append(changed)
        for argv in variants:
            changed = self.job(wrapped=False); changed["argv"] = argv
            with self.subTest(argv=argv), self.assertRaises(RuntimeError):
                campaign.task_for(changed, self.plan(), "initial", self.manifest_reader)

    def test_task_rejects_manifest_condition_stage_shard_hash_path_and_science_env(self):
        original = self.job(wrapped=False)
        variants = []
        for field, value in (("manifest_hash", "wrong"), ("scientific_phase", "wrong_phase"),
                             ("complete_file", str(self.experiment / "wrong.json")),
                             ("env", {"SCIENTIFIC_SETTING": "changed"})):
            changed = copy.deepcopy(original); changed[field] = value; variants.append(changed)
        for flag, value in (("--condition", "unplanned"), ("--output", str(self.experiment / "unapproved"))):
            changed = copy.deepcopy(original); changed["argv"][changed["argv"].index(flag) + 1] = value; variants.append(changed)
        for flags in (("--stage", "confirm"), ("--shards", "2"), ("--shard-index", "1")):
            changed = copy.deepcopy(original); changed["argv"].extend(flags); variants.append(changed)
        for changed in variants:
            with self.subTest(job=changed), self.assertRaises((RuntimeError, ValueError)):
                campaign.task_for(changed, self.plan(), "initial", self.manifest_reader)
        noncanonical = copy.deepcopy(original)
        noncanonical["argv"][noncanonical["argv"].index("--output") + 1] = str(self.outputs["discovery"] / ".." / "discovery")
        with self.assertRaisesRegex(RuntimeError, "Noncanonical"):
            campaign.task_for(noncanonical, self.plan(), "initial", self.manifest_reader)

    def test_manifest_reader_failure_is_not_bypassed(self):
        def failed_reader(output):
            raise RuntimeError("Frozen source/checkpoint manifest validation failed")
        with self.assertRaisesRegex(RuntimeError, "Frozen source"):
            campaign.task_for(self.job(), self.plan(), "initial", failed_reader)

    def test_collect_skips_completed_and_unready_but_refuses_running_or_failed(self):
        complete = self.job("jac_anchor")
        waiting_dependency = self.job("local_goal"); waiting_dependency["depends_on"] = ["not_complete"]
        waiting_file = self.job("local_long"); waiting_file["ready_files"] = [str(self.experiment / "absent")]
        pending = self.job()
        queue = dict(jobs=[complete, waiting_dependency, waiting_file, pending])
        selected, tasks, proposed = campaign.collect(queue, {complete["id"]: {"status": "complete"}},
                                                     self.plan(), "initial", self.manifest_reader)
        self.assertEqual([job["id"] for job in selected], [pending["id"]])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(proposed["jobs"], [complete, waiting_dependency, waiting_file])
        for status in ("running", "failed", "interrupted"):
            with self.subTest(status=status), self.assertRaises(RuntimeError):
                campaign.collect(dict(jobs=[pending]), {pending["id"]: {"status": status}},
                                 self.plan(), "initial", self.manifest_reader)

    def test_collect_refuses_duplicate_ids_and_unhandled_downstream_dependencies(self):
        original = self.job()
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            campaign.collect(dict(jobs=[original, original]), {}, self.plan(), "initial", self.manifest_reader)
        downstream = dict(id="consumer", argv=["cpu-only"], depends_on=[original["id"]])
        with self.assertRaisesRegex(RuntimeError, "downstream"):
            campaign.collect(dict(jobs=[original, downstream]), {}, self.plan(), "initial", self.manifest_reader)

    def test_prepare_is_inactive_archival_only_and_places_slots_before_future_science(self):
        proposal = self.prepare()
        self.assertEqual(len(proposal["logical_jobs"]), 5)
        self.assertEqual(len(proposal["task_sha256"]), 5)
        self.assertEqual(self.queue_path.read_bytes(), (self.root / "queue.before.json").read_bytes())
        self.assertFalse((self.root / "transactions/initial/committed.json").exists())
        self.assertFalse((self.root / "deployed.json").exists())
        proposed = self.read(self.root / "queue.proposed.json")
        positions = {job["id"]: index for index, job in enumerate(proposed["jobs"])}
        self.assertTrue(all(positions[slot] < positions["confirm_full_d10"] for slot in proposal["slot_job_ids"]))
        old_ids = {job["id"] for job in proposal["logical_jobs"]}
        self.assertFalse(old_ids & positions.keys())
        self.assertEqual(len(proposed["jobs"]), len(positions))
        for task_id, digest in proposal["task_sha256"].items():
            path = self.root / "tasks" / (task_id + ".json")
            self.assertEqual(campaign.digest(path.read_bytes()), digest)
            self.assertEqual(self.read(path)["env"], {})

    def test_prepare_planning_gate_proof_does_not_write_scheduler_state(self):
        queue, states = self.initial_queue()
        selected = self.job()["id"]
        states[selected] = dict(status="running", pid=12345, preserved="unchanged")
        proof = dict(pid=12345, module=campaign.GATE, start_ticks=98765, children=[])
        proposal = self.prepare(queue, states, {selected: proof})
        self.assertEqual(self.read(self.state_path), states)
        self.assertEqual(proposal["logical_states_before"][selected], states[selected])
        self.assertEqual(proposal["waiting_gate_proofs"][selected], proof)

    def test_prepare_rejects_other_pending_work_instead_of_starving_it(self):
        queue, states = self.initial_queue()
        queue["jobs"].append(dict(id="other_pending_cpu_job", argv=["cpu-only"], depends_on=[]))
        with self.assertRaisesRegex(RuntimeError, "[Pp]ending|[Oo]ther|[Rr]emaining|[Cc]omplete"):
            self.prepare(queue, states)
        self.assertEqual(self.read(self.queue_path), queue)
        self.assertEqual(self.read(self.state_path), states)
        self.assertFalse((self.root / "transactions/initial/committed.json").exists())

    def test_invalid_waiting_gate_proof_cannot_convert_running_science(self):
        for field, value in (("module", campaign.RUNNER), ("children", [999]), ("pid", 42), ("start_ticks", 0)):
            with self.subTest(field=field):
                self.root = self.experiment / ("campaign_" + field)
                queue, states = self.initial_queue(); selected = self.job()["id"]
                states[selected] = dict(status="running", pid=12345)
                proof = dict(pid=12345, module=campaign.GATE, start_ticks=98765, children=[])
                proof[field] = value
                with self.assertRaisesRegex(RuntimeError, "waiting-gate"):
                    self.prepare(queue, states, {selected: proof})
                self.assertEqual(self.read(self.state_path), states)

    def test_append_only_json_and_raw_archives_reject_overwrite(self):
        path = self.experiment / "artifact.json"
        original = {"id": "first"}
        first = campaign.append_identical(path, original)
        self.assertEqual(first, campaign.append_identical(path, original))
        with self.assertRaisesRegex(RuntimeError, "Append-only"):
            campaign.append_identical(path, {"id": "second"})
        with self.assertRaisesRegex(RuntimeError, "Append-only"):
            campaign.append_raw_identical(path, b"different")
        self.assertEqual(self.read(path), original)

    def test_publish_stages_descriptors_then_queue_then_activation_and_is_idempotent(self):
        _, _, candidates = self.deployed()
        events = []
        original_append, original_durable = campaign.append_identical, campaign.durable

        def durable(path, value, *args, **kwargs):
            if Path(path) == self.queue_path:
                self.assertTrue(list((self.root / "tasks").glob("*.json")))
                self.assertFalse(self.commits())
                events.append("queue")
            return original_durable(path, value, *args, **kwargs)

        def append(path, value):
            if Path(path).name == "committed.json":
                self.assertFalse({job["id"] for job in candidates} & {job["id"] for job in self.read(self.queue_path)["jobs"]})
                self.assertIn("queue", events)
                events.append("activate")
            return original_append(path, value)

        with patch.object(campaign, "durable", side_effect=durable), patch.object(campaign, "append_identical", side_effect=append):
            result = self.publish()
        self.assertEqual(result["phase"], "dispatched")
        self.assertEqual(events, ["queue", "activate"])
        self.assertEqual(len(self.commits()), 1)
        task_bytes = {path.name: path.read_bytes() for path in (self.root / "tasks").glob("*.json")}
        self.assertEqual(self.publish()["phase"], "waiting_tasks")
        self.assertEqual(task_bytes, {path.name: path.read_bytes() for path in (self.root / "tasks").glob("*.json")})

    def test_slot_order_failed_slot_and_missing_slot_refuse_dispatch(self):
        queue, states, candidates = self.deployed()
        slots = self.read(self.root / "deployed.json")["slot_job_ids"]
        with self.assertRaisesRegex(RuntimeError, "precedes"):
            campaign.slot_capacity_proof(dict(jobs=[candidates[0], *queue["jobs"][:-1]]), states, slots, [candidates[0]["id"]])
        states[slots[0]]["status"] = "failed"
        with self.assertRaisesRegex(RuntimeError, "failed/exited"):
            campaign.slot_capacity_proof(queue, states, slots, [candidates[0]["id"]])
        with self.assertRaisesRegex(RuntimeError, "disappeared"):
            campaign.slot_capacity_proof(dict(jobs=queue["jobs"][1:]), {}, slots, [candidates[0]["id"]])

    def test_queue_cas_race_keeps_descriptors_inactive_and_does_not_overwrite(self):
        queue, _, _ = self.deployed()
        original_append = campaign.append_identical
        raced = copy.deepcopy(queue); raced["jobs"].append(dict(id="concurrent_cpu_row", argv=["cpu-only"]))

        def append(path, value):
            result = original_append(path, value)
            if Path(path).name == "transaction.json":
                self.write(self.queue_path, raced)
            return result

        with patch.object(campaign, "append_identical", side_effect=append), self.assertRaisesRegex(RuntimeError, "CAS changed"):
            self.publish()
        self.assertEqual(self.read(self.queue_path), raced)
        self.assertTrue(list((self.root / "tasks").glob("*.json")))
        self.assertFalse(self.commits())

    def test_state_race_before_and_after_cas_never_activates(self):
        for timing in ("before", "after"):
            with self.subTest(timing=timing):
                self.root = self.experiment / ("campaign_" + timing)
                queue, states, candidates = self.deployed()
                original_read, original_durable = campaign.read, campaign.durable
                state_reads = [0]

                def read(path):
                    if Path(path) == self.state_path:
                        state_reads[0] += 1
                        if timing == "before" and state_reads[0] == 2:
                            changed = copy.deepcopy(states); changed[candidates[0]["id"]]["status"] = "running"
                            self.write(self.state_path, changed)
                    return original_read(path)

                def durable(path, value, *args, **kwargs):
                    result = original_durable(path, value, *args, **kwargs)
                    if timing == "after" and Path(path) == self.queue_path:
                        changed = copy.deepcopy(states); changed[candidates[0]["id"]]["status"] = "running"
                        self.write(self.state_path, changed)
                    return result

                with patch.object(campaign, "read", side_effect=read), patch.object(campaign, "durable", side_effect=durable):
                    with self.assertRaisesRegex(RuntimeError, "Logical task began|Post-CAS"):
                        self.publish()
                self.assertFalse(self.commits())
                self.assertEqual(candidates[0]["id"] in {job["id"] for job in self.read(self.queue_path)["jobs"]}, timing == "before")

    def crash_after_cas(self, candidates=None):
        self.deployed(candidates)
        original_append = campaign.append_identical

        def append(path, value):
            if Path(path).name == "committed.json":
                raise SimulatedCrash("after queue CAS, before activation")
            return original_append(path, value)

        with patch.object(campaign, "append_identical", side_effect=append), self.assertRaises(SimulatedCrash):
            self.publish()
        self.assertFalse(self.commits())

    def test_crash_after_cas_recovers_activation_with_append_only_new_queue_rows(self):
        self.crash_after_cas()
        queue = self.read(self.queue_path)
        queue["jobs"].append(dict(id="later_cpu_job", argv=["cpu-only"]))
        self.write(self.queue_path, queue)
        result = self.publish()
        self.assertEqual(result["phase"], "waiting_tasks")
        self.assertEqual(len(result["recovered"]), 1)
        self.assertEqual(len(self.commits()), 1)
        self.assertEqual(self.read(self.queue_path), queue)
        commit = self.read(self.commits()[0])
        for task_id, digest in commit["task_sha256"].items():
            path = self.root / "tasks" / (task_id + ".json")
            self.assertEqual(campaign.digest(path.read_bytes()), digest)
            self.assertEqual(self.read(path)["activation_file"], str(self.commits()[0]))

    def test_crash_recovery_rejects_partial_removal_and_raced_original_task(self):
        candidates = [self.job("full_d10", "confirm"), self.job("local_goal", "confirm")]
        self.crash_after_cas(candidates)
        queue = self.read(self.queue_path); queue["jobs"].append(candidates[0])
        self.write(self.queue_path, queue)
        with self.assertRaisesRegex(RuntimeError, "partial"):
            self.publish()
        self.assertFalse(self.commits())
        queue["jobs"].pop(); self.write(self.queue_path, queue)
        states = self.read(self.state_path); states[candidates[0]["id"]]["status"] = "running"
        self.write(self.state_path, states)
        with self.assertRaisesRegex(RuntimeError, "ran during"):
            self.publish()
        self.assertFalse(self.commits())

    def test_recovery_rejects_tampered_task(self):
        self.crash_after_cas()
        task_path = next((self.root / "tasks").glob("*.json"))
        task = self.read(task_path); task["condition"] = "tampered"
        self.write(task_path, task)
        with self.assertRaisesRegex(RuntimeError, "task/activation"):
            self.publish()
        self.assertFalse(self.commits())

    def repin_recovery_task(self, task_id, task):
        path = self.root/'tasks'/(task_id+'.json')
        self.write(path, task)
        record_path = next((self.root/'transactions').glob('dispatch_*/transaction.json'))
        record = self.read(record_path)
        record['task_sha256'][task_id] = campaign.digest(path.read_bytes())
        self.write(record_path, record)

    def test_recovery_rejects_two_tasks_for_logical_a_omitting_b_with_legal_hashes(self):
        candidates = [self.job('full_d10', 'confirm'), self.job('local_goal', 'confirm')]
        self.crash_after_cas(candidates)
        record_path = next((self.root/'transactions').glob('dispatch_*/transaction.json'))
        record = self.read(record_path)
        first_id, second_id = sorted(record['task_sha256'])
        first = self.read(self.root/'tasks'/(first_id+'.json'))
        duplicate = copy.deepcopy(first); duplicate['id'] = second_id
        self.repin_recovery_task(second_id, duplicate)
        self.assertEqual(len(self.read(record_path)['task_sha256']), len(candidates))
        with self.assertRaisesRegex(RuntimeError, 'one-to-one'):
            campaign.recover_publications(self.root, self.read(self.queue_path), self.read(self.state_path),
                                          current_raw=self.queue_path.read_bytes(), manifest_reader=self.manifest_reader)
        self.assertFalse(self.commits())

    def test_recovery_rebuilds_full_scientific_descriptor_even_with_recomputed_hashes(self):
        changes = [('condition', 'local_long'), ('stage', 'discovery'), ('output', '/tmp/foreign-output'),
                   ('shard_index', 1), ('shards', 2), ('env', {'SCIENCE': 'changed'}),
                   ('manifest_hash', 'wrong'), ('complete_file', '/tmp/foreign-complete.json')]
        for field, value in changes:
            with self.subTest(field=field):
                self.root = self.experiment/('campaign_full_task_'+field)
                self.crash_after_cas()
                path = next((self.root/'tasks').glob('*.json'))
                task = self.read(path); original_argv = copy.deepcopy(task['argv'])
                task[field] = value
                self.repin_recovery_task(path.stem, task)
                self.assertEqual(self.read(path)['argv'], original_argv)
                with self.assertRaisesRegex(RuntimeError, 'task/activation binding'):
                    self.publish()
                self.assertFalse(self.commits())

    def test_recovery_rejects_prepended_queue_rows_after_cas(self):
        self.crash_after_cas()
        queue = self.read(self.queue_path)
        queue['jobs'].insert(0, dict(id='prepended_cpu_row', argv=['cpu-only']))
        self.write(self.queue_path, queue)
        with self.assertRaisesRegex(RuntimeError, 'Committed queue prefix or metadata'):
            self.publish()
        self.assertFalse(self.commits())

    def test_recovery_rejects_reordered_committed_queue_rows_after_cas(self):
        self.crash_after_cas()
        queue = self.read(self.queue_path); queue['jobs'].reverse()
        self.write(self.queue_path, queue)
        with self.assertRaisesRegex(RuntimeError, 'Committed queue prefix or metadata'):
            self.publish()
        self.assertFalse(self.commits())

    def test_recovery_rejects_nonjobs_queue_metadata_changes_after_cas(self):
        self.crash_after_cas()
        queue = self.read(self.queue_path); queue['schema'] = 'changed-top-level-identity'
        self.write(self.queue_path, queue)
        with self.assertRaisesRegex(RuntimeError, 'Committed queue prefix or metadata'):
            self.publish()
        self.assertFalse(self.commits())

    def test_recovery_rejects_tampered_queue_archive(self):
        self.crash_after_cas()
        archive = next((self.root / "transactions").glob("dispatch_*/queue.proposed.json"))
        changed = self.read(archive)
        changed["jobs"].append(dict(id="tampered_archive_row", argv=["cpu-only"]))
        self.write(archive, changed)
        with self.assertRaisesRegex(RuntimeError, "archive changed"):
            self.publish()
        self.assertFalse(self.commits())

    def test_uncommitted_changed_queue_is_superseded_without_activating_old_inbox(self):
        queue, _, candidates = self.deployed()
        original_durable = campaign.durable

        def crash_before_cas(path, value, *args, **kwargs):
            if Path(path) == self.queue_path:
                raise SimulatedCrash("queue not committed")
            return original_durable(path, value, *args, **kwargs)

        with patch.object(campaign, "durable", side_effect=crash_before_cas), self.assertRaises(SimulatedCrash):
            self.publish()
        old_transaction = next((self.root / "transactions").glob("dispatch_*"))
        old_task_names = {path.name for path in (self.root / "tasks").glob("*.json")}
        self.assertEqual(self.read(self.queue_path), queue)
        queue["jobs"].append(dict(id="new_cpu_row", argv=["cpu-only"]))
        self.write(self.queue_path, queue)
        result = self.publish()
        self.assertEqual(result["phase"], "dispatched")
        self.assertTrue((old_transaction / "superseded_uncommitted.json").is_file())
        self.assertFalse((old_transaction / "committed.json").exists())
        self.assertTrue(old_task_names < {path.name for path in (self.root / "tasks").glob("*.json")})
        self.assertFalse({job["id"] for job in candidates} & {job["id"] for job in self.read(self.queue_path)["jobs"]})
        self.assertIn("new_cpu_row", {job["id"] for job in self.read(self.queue_path)["jobs"]})
        self.assertEqual(len(self.commits()), 1)

    def test_pre_cas_format_only_change_supersedes_old_transaction_and_activates_once(self):
        queue, _, candidates = self.deployed()
        original_durable = campaign.durable

        def crash_before_cas(path, value, *args, **kwargs):
            if Path(path) == self.queue_path:
                raise SimulatedCrash('queue not committed')
            return original_durable(path, value, *args, **kwargs)

        with patch.object(campaign, 'durable', side_effect=crash_before_cas), self.assertRaises(SimulatedCrash):
            self.publish()
        old_transaction = next((self.root/'transactions').glob('dispatch_*'))
        old_task_ids = {path.stem for path in (self.root/'tasks').glob('*.json')}
        before = self.queue_path.read_bytes()
        self.queue_path.write_text(json.dumps(queue, separators=(',', ':'))+'\n')
        self.assertEqual(self.read(self.queue_path), queue)
        self.assertNotEqual(self.queue_path.read_bytes(), before)
        self.assertEqual(self.publish()['phase'], 'dispatched')
        self.assertTrue((old_transaction/'superseded_uncommitted.json').is_file())
        self.assertFalse((old_transaction/'committed.json').exists())
        self.assertEqual(len(self.commits()), 1)
        activated_ids = set(self.read(self.commits()[0])['task_sha256'])
        self.assertFalse(activated_ids & old_task_ids)
        self.assertEqual(len(activated_ids), len(candidates))
        self.assertEqual(self.publish()['phase'], 'waiting_tasks')
        self.assertEqual(len(self.commits()), 1)
        self.assertFalse((old_transaction/'committed.json').exists())

    def test_source_pin_and_deployment_plan_hash_are_required(self):
        self.deployed()
        deployment = self.read(self.root / "deployed.json")
        self.write(self.root / "deployed.json", dict(deployment, plan_sha256="wrong"))
        with self.assertRaisesRegex(RuntimeError, "Deployment/plan"):
            self.publish()
        self.write(self.root / "deployed.json", deployment)
        (self.project / campaign.SOURCE_NAMES[0]).write_text("changed source")
        with self.assertRaisesRegex(RuntimeError, "source pin changed"):
            self.publish()
        self.assertFalse(self.commits())

    def test_release_request_does_not_move_jobs_and_broker_exits(self):
        queue, _, _ = self.deployed()
        (self.root / "release").touch()
        with patch.object(campaign, "scientific_manifest", side_effect=AssertionError("science import forbidden")):
            self.assertEqual(self.publish(), dict(phase="release_requested", moved=0))
        self.assertEqual(self.read(self.queue_path), queue)
        with patch.object(campaign, "publish", return_value=dict(phase="release_requested", moved=0)), \
             patch.object(campaign.time, "sleep", side_effect=AssertionError("broker must exit")):
            campaign.launch_broker(self.root)
        self.assertEqual(self.read(self.root / "broker_status.json")["phase"], "release_requested")
        self.assertEqual(self.read(self.root / "broker_process.json")["pid"], os.getpid())

    def test_broker_attention_preserves_inbox_and_exits_without_retry_loop(self):
        self.deployed()
        with patch.object(campaign, "publish", side_effect=RuntimeError("dispatch needs review")), \
             patch.object(campaign.time, "sleep", side_effect=AssertionError("attention must exit")):
            campaign.launch_broker(self.root)
        status = self.read(self.root / "broker_status.json")
        self.assertEqual(status["phase"], "attention")
        self.assertIn("dispatch needs review", status["error"])
        self.assertFalse(self.commits())


if __name__ == "__main__":
    unittest.main()
