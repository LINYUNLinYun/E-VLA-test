"""CPU-only fakes for the one-shot port-retry helper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from preparations.ease_insights_20261103 import retry_resource_session_port as retry


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = retry.json_bytes(value)
    path.write_bytes(raw)
    return raw


def proc_stat(pid: int, state: str, parent: int, start: str) -> str:
    tail = [state, str(parent)] + ["0"] * 17 + [str(start)]
    return "%d (fake python) %s\n" % (pid, " ".join(tail))


class Fixture:
    def __init__(self, *, phases=("attention_holding", "idle_holding"), running_worker=True,
                 task_stage="discovery", attention_conditions=("local_goal", "local_spatial")):
        self.temp = tempfile.TemporaryDirectory()
        self.project = Path(self.temp.name).resolve()
        self.experiment = self.project / "results/ease_insights_20261103"
        self.campaign = self.experiment / "gpu_resource_campaign_v3"
        self.report = self.project / "reports/port_retry_helper"
        self.watch_dir = self.experiment / "gpu_release_watch"
        self.proc_root = self.project / "proc"
        self.boot_id = "test-boot-id"
        self.now = 1000.0
        self.port_busy = set()
        self.probe_calls = []
        self.task_stage = task_stage
        self.attention_conditions = attention_conditions
        self.source_hashes = {}
        for name in retry.PINNED_SOURCE_NAMES:
            source = self.project / name
            source.parent.mkdir(parents=True, exist_ok=True)
            raw = ("fake pinned source:" + name).encode()
            source.write_bytes(raw)
            self.source_hashes[name] = digest(raw)
        session_source = self.project / "preparations/ease_insights_20261103/gpu_resource_session.py"
        session_source.write_bytes(b"fake pinned session source")
        self.source_hashes["preparations/ease_insights_20261103/gpu_resource_session.py"] = digest(session_source.read_bytes())
        watch_source = self.project / "preparations/ease_insights_20261103/gpu_release_watch.py"
        watch_source.write_bytes(b"fake watcher source")
        science_source = self.project / "preparations/test_science_source.py"
        science_source.parent.mkdir(parents=True, exist_ok=True)
        science_source.write_bytes(b"fake frozen science source")
        self.science_source_hash = digest(science_source.read_bytes())

        self.slots = tuple(self._slot_spec(index) for index in range(2))
        self.outputs = [self.experiment / "vlm_local_search_v1" / stage
                        for stage in ("discovery", "confirmation")]
        self.manifest_hashes = {}
        plan = {"schema": retry.PLAN_SCHEMA, "campaign_root": str(self.campaign),
                "allowed_outputs": [str(path) for path in self.outputs], "task_schema": retry.TASK_SCHEMA,
                "source_hashes": self.source_hashes,
                "keeper_max_memory_mib": 512}
        plan_path = self.campaign / "plan.json"
        plan_raw = write_json(plan_path, plan)
        self.plan_sha = digest(plan_raw)
        self.plan = plan

        for output in self.outputs:
            confirmation = output.name == "confirmation"
            manifest = {"phase": "local_search_confirmation_v1" if confirmation else "local_search_discovery_v1",
                        "planned_conditions": ["full_d10", "local_goal", "local_spatial"],
                        "suites": ["libero_spatial"], "tasks": [0], "discovery_inits": [36],
                        "confirmation_inits": [39],
                        "source_files": {"preparations/test_science_source.py": self.science_source_hash}}
            manifest["manifest_hash"] = digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
            self.manifest_hashes[str(output)] = manifest["manifest_hash"]
            write_json(output / "manifest.json", manifest)

        self.campaign.mkdir(parents=True, exist_ok=True)
        (self.campaign / "tasks").mkdir(parents=True, exist_ok=True)
        (self.campaign / "outcomes").mkdir(parents=True, exist_ok=True)
        (self.campaign / "transactions/initial").mkdir(parents=True, exist_ok=True)
        for spec in self.slots:
            for subdir in ("resume_receipts",):
                (self.campaign / "sessions" / spec.name / subdir).mkdir(parents=True, exist_ok=True)
            (self.campaign / "resumes" / spec.name).mkdir(parents=True, exist_ok=True)
        self._write_process(500, "5000", 1, ("/usr/bin/python3", "-u", "-m", "preparations.ease_insights_20261103.scheduler"))
        self._write_process(900, "9000", 1, ("python", "-u", "-m", "preparations.ease_insights_20261103.gpu_release_watch"))
        (self.proc_root / "sys/kernel/random").mkdir(parents=True, exist_ok=True)
        (self.proc_root / "sys/kernel/random/boot_id").write_text(self.boot_id)

        self.tasks = []
        for index, spec in enumerate(self.slots):
            self._write_slot_proc(spec)
            self._write_binding(spec)
            self._write_keeper(spec)
            phase = phases[index]
            if phase == "attention_holding":
                task = self._make_task("task_original_%s" % spec.name, self.attention_conditions[index], spec)
                self._write_status(spec, phase, task)
                self._write_outcome(spec, task)
                self.tasks.append(task)
            elif phase == "running":
                task = self._make_task("task_running_%s" % spec.name, "local_spatial", spec)
                self._write_status(spec, phase, task)
                self._write_running_children(spec, task, with_worker=running_worker)
                self.tasks.append(task)
            elif phase == "idle_holding":
                self._write_status(spec, phase, None)
                self.tasks.append(None)
            else:
                raise AssertionError(phase)
        self._write_activation_for_tasks([task for task in self.tasks if task is not None])
        # Activation paths are already the initial transaction, which is the
        # source path serialized into each task descriptor.
        self._write_watcher(phases)
        self.baseline = self._write_history()

    def close(self):
        self.temp.cleanup()

    def _slot_spec(self, index):
        source = retry.DEFAULT_SLOTS[index]
        slot = source.name
        gpu = source.gpu_uuid
        port = source.port
        session_pid = 110 + index
        session_start = str(2000 + index)
        keeper_pid = 120 + index
        keeper_start = str(3000 + index)
        scheduler_pid = 500
        scheduler_start = "5000"
        nonce = ("a" if index == 0 else "b") * 32
        scheduler_argv = ("/usr/bin/python3", "-u", "-m", "preparations.ease_insights_20261103.scheduler",
            "--queue", str(self.experiment / "queue.json"), "--output", str(self.experiment / "scheduler"),
            "--max-gpus", "2", "--scan-seconds", "2", "--idle-seconds", "0")
        session_argv = ("/model/python", "-B", "-u", "-m", "preparations.ease_insights_20261103.gpu_resource_session",
            "--gpu-uuid", gpu, "--port", str(port), "--plan", str(self.campaign / "plan.json"),
            "--output", str(self.campaign / "sessions" / slot), "--launch")
        keeper_argv = ("/usr/bin/python3", "-m", "preparations.ease_insights_20261103.gpu_owned_keeper",
            "--gpu-uuid", gpu, "--control-fd", "6" if index == 0 else "5", "--ready-fd", "9",
            "--nonce", nonce, "--parent-pid", str(session_pid), "--parent-start-ticks", session_start,
            "--allocation-mib", "64")
        return retry.SlotSpec(slot, gpu, port, session_pid, session_start, scheduler_pid,
            session_argv, keeper_pid, keeper_start, nonce, keeper_argv,
            scheduler_start=scheduler_start, scheduler_argv=scheduler_argv)

    def _write_process(self, pid, start, ppid, argv, state="S"):
        directory = self.proc_root / str(pid)
        (directory / "task" / str(pid)).mkdir(parents=True, exist_ok=True)
        (directory / "stat").write_text(proc_stat(pid, state, ppid, str(start)))
        (directory / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
        (directory / "status").write_text("Name:\tfake\nUid:\t0\t0\t0\t0\n")
        (directory / "task" / str(pid) / "children").write_text("")

    def _write_children(self, pid, children):
        path = self.proc_root / str(pid) / "task" / str(pid) / "children"
        path.write_text(" ".join(str(value) for value in children))

    def _write_slot_proc(self, spec):
        self._write_process(spec.session_parent, spec.scheduler_start, 1, spec.scheduler_argv)
        self._write_process(spec.session_pid, spec.session_start, spec.session_parent, spec.session_argv, state="R")
        self._write_process(spec.keeper_pid, spec.keeper_start, spec.session_pid, spec.keeper_argv)
        self._write_children(spec.session_pid, [spec.keeper_pid])
        self._write_children(spec.keeper_pid, [])

    def _write_binding(self, spec):
        scheduler = {"pid": spec.session_parent, "start_ticks": spec.scheduler_start, "state": "S", "ppid": 1,
                     "argv": list(spec.scheduler_argv), "uid": 0, "children": [str(spec.session_pid)], "sigterm_default": True}
        session = {"pid": spec.session_pid, "start_ticks": spec.session_start, "state": "R", "ppid": spec.session_parent,
                   "argv": list(spec.session_argv), "uid": 0, "children": [str(spec.keeper_pid)], "sigterm_default": False}
        binding = {"job_id": "ease_gpu_resource_session_v1_" + spec.name[-2:], "scheduler": scheduler,
                   "session": session, "lease_path": "/tmp/embodied-benchmark-gpu-%s.lock" % spec.gpu_uuid,
                   "lease_device": 1, "lease_inode": spec.session_pid, "lease_fds": [4]}
        write_json(self.campaign / "sessions" / spec.name / "binding.json", binding)

    def _write_keeper(self, spec):
        ready = {"schema": "ease-gpu-owned-keeper-v1", "gpu_uuid": spec.gpu_uuid, "pid": spec.keeper_pid,
                 "proc_start_ticks": int(spec.keeper_start), "nonce": spec.nonce, "phase": "ready",
                 "context_created": True, "allocated_bytes": 64 * 1024 * 1024,
                 "nspid": [spec.keeper_pid], "pid_namespace_inode": 1, "total_memory_bytes": 1000,
                 "driver_version": 1}
        write_json(self.campaign / "sessions" / spec.name / "keeper.json", ready)

    def _make_task(self, task_id, condition, spec):
        output = self.outputs[0] if self.task_stage == "discovery" else self.outputs[1]
        stage = self.task_stage
        model_python = "/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.venv/bin/python"
        return {"schema": retry.TASK_SCHEMA, "id": task_id, "logical_job_id": "logical_" + task_id,
            "argv": [model_python, "-m", "preparations.ease_insights_20261103.vlm_local_search_v1.runner",
                     "--output", str(output), "--condition", condition, "--gpu-uuid", "{gpu_uuid}",
                     "--port", "{port}", "--launch"],
            "complete_file": str(output / condition / (stage + "_shard00of01_complete.json")),
            "manifest_hash": self.manifest_hashes[str(output)], "output": str(output), "condition": condition,
            "stage": stage, "shard_index": 0, "shards": 1, "env": {},
            "activation_file": str(self.campaign / "transactions/initial/committed.json")}

    def _write_activation_for_tasks(self, tasks):
        hashes = {}
        for task in tasks:
            raw = write_json(self.campaign / "tasks" / (task["id"] + ".json"), task)
            hashes[task["id"]] = digest(raw)
        write_json(self.campaign / "transactions/initial/committed.json",
                   {"schema": retry.ACTIVATION_SCHEMA, "plan_sha256": self.plan_sha,
                    "task_sha256": hashes})

    def _write_status(self, spec, phase, task):
        value = {"schema": "ease-gpu-resource-session-v1", "phase": phase, "gpu_uuid": spec.gpu_uuid,
                 "time": self.now, "session_pid": spec.session_pid}
        if task is not None:
            value["task_id"] = task["id"]
            if phase == "attention_holding":
                value["attention"] = {"task_id": task["id"], "error": retry.PORT_ERROR}
            else:
                value["logical_job_id"] = task["logical_job_id"]
        write_json(self.campaign / "sessions" / spec.name / "status.json", value)

    def _write_outcome(self, spec, task, error=None):
        task_path = self.campaign / "tasks" / (task["id"] + ".json")
        task_raw = task_path.read_bytes() if task_path.exists() else retry.json_bytes(task)
        outcome = {"schema": "ease-gpu-resource-session-v1", "phase": "attention", "task_id": task["id"],
            "task_sha256": digest(task_raw), "gpu_uuid": spec.gpu_uuid,
            "error": retry.PORT_ERROR if error is None else error, "time": self.now}
        write_json(self.campaign / "outcomes" / (digest(task["id"].encode()) + ".json"), outcome)

    def _write_running_children(self, spec, task, *, with_worker):
        output = Path(task["output"])
        logs = output / task["condition"] / "runs" / "discovery_shard00of01_123456"
        server_argv = (task["argv"][0], "-m", "preparations.ease_insights_20261103.vlm_local_search_v1.server",
            "--manifest", str(output / "manifest.json"), "--condition", task["condition"], "--port", str(spec.port),
            "--gpu-uuid", spec.gpu_uuid, "--output", str(logs))
        server_pid = 130 + int(spec.name[-2:]) * 10
        self._write_process(server_pid, str(4000 + server_pid), spec.session_pid, server_argv)
        children = [spec.keeper_pid, server_pid]
        if with_worker:
            worker_pid = server_pid + 1
            worker_argv = ("/root/nfs/embodied/benchmark/policies/libero-eval-v21/.venv/bin/python",
                "-m", "preparations.ease_insights_20261103.vlm_local_search_v1.worker",
                "--output", str(output), "--condition", task["condition"], "--port", str(spec.port),
                "--stage", "discovery", "--shard-index", "0", "--shards", "1")
            self._write_process(worker_pid, str(4000 + worker_pid), spec.session_pid, worker_argv)
            children.append(worker_pid)
        self._write_children(spec.session_pid, children)

    def _write_watcher(self, phases):
        source = self.project / "preparations/ease_insights_20261103/gpu_release_watch.py"
        source_sha = digest(source.read_bytes())
        command = "python -u -m preparations.ease_insights_20261103.gpu_release_watch"
        receipt = {"pid": 900, "proc_start_ticks": 9000, "source_sha256": source_sha,
                   "source_path": str(source), "python_executable": "python", "command_line": command,
                   "boot_id": self.boot_id, "run_id": "watch-run-1", "mode": "observe_only", "poll_seconds": 2.0}
        write_json(self.watch_dir / "process_receipt.json", receipt)
        devices = []
        for index, spec in enumerate(self.slots):
            if phases[index] == "running":
                count, memory = 3, 12000.0
            else:
                count, memory = 1, 460.0
            devices.append({"index": index, "gpu_uuid": spec.gpu_uuid, "memory_mib": memory,
                            "compute_process_count": count, "host_compute_pids": list(range(9000 + index * 4, 9000 + index * 4 + count)),
                            "state": "busy", "idle": False})
        devices.extend([
            {"index": 2, "gpu_uuid": "GPU-83fdf473-8633-29d3-4a42-e482cfdf0c25", "memory_mib": 0,
             "compute_process_count": 0, "host_compute_pids": [], "state": "idle", "idle": True},
            {"index": 3, "gpu_uuid": "GPU-555cde36-5e41-bc44-0c69-ea47358fd703", "memory_mib": 0,
             "compute_process_count": 0, "host_compute_pids": [], "state": "idle", "idle": True},
        ])
        status = {"pid": 900, "proc_start_ticks": 9000, "source_sha256": source_sha,
                  "run_id": "watch-run-1", "mode": "observe_only", "query_finished_at": self.now,
                  "heartbeat_at": self.now, "error": None,
                  "devices": devices}
        write_json(self.watch_dir / "status.json", status)

    def _write_history(self):
        history_dir = self.campaign / "transactions" / retry.HISTORY_NAME
        history_dir.mkdir(parents=True, exist_ok=True)
        hashes = {"task_port_retry_slot00": digest(b"task0"), "task_port_retry_slot01": digest(b"task1")}
        records = []
        for index, spec in enumerate(self.slots):
            records.append({"slot": spec.name, "gpu_uuid": spec.gpu_uuid, "port": spec.port,
                "session": {"pid": spec.session_pid, "start_ticks": spec.session_start,
                            "ppid": spec.session_parent, "argv": list(spec.session_argv)},
                "keeper": {"pid": spec.keeper_pid, "start_ticks": spec.keeper_start,
                           "ppid": spec.session_pid, "argv": list(spec.keeper_argv)},
                "keeper_ready": {"nonce": spec.nonce}})
        preparation = {"schema": "ease-explicit-technical-port-retry-v1", "phase": "prepared",
            "plan_sha256": self.plan_sha, "source_or_plan_changed": False,
            "task_sha256": hashes, "records": records}
        pub = {"phase": "awaiting_owned_session_resume_ack", "task_sha256": hashes,
               "resource_release_files_written": False}
        activation = {"schema": retry.ACTIVATION_SCHEMA, "plan_sha256": self.plan_sha, "task_sha256": hashes}
        prep_raw = write_json(history_dir / "preparation.json", preparation)
        pub_raw = write_json(history_dir / "published.json", pub)
        act_raw = write_json(history_dir / "committed.json", activation)
        expected = {"preparation.json": digest(prep_raw), "published.json": digest(pub_raw),
                    "committed.json": digest(act_raw)}
        for spec in self.slots:
            request = {"schema": retry.RESUME_SCHEMA, "action": "resume_after_attention",
                "plan_sha256": self.plan_sha, "session_pid": spec.session_pid,
                "proc_start_ticks": spec.session_start, "gpu_uuid": spec.gpu_uuid,
                "attention_sha256": digest(("old attention " + spec.name).encode()), "reason": "prior verified resume"}
            request_path = self.campaign / "resumes" / spec.name / (retry.OLD_REQUEST_ID + ".json")
            request_raw = write_json(request_path, request)
            receipt = {"schema": "ease-gpu-resource-resume-receipt-v1", "phase": "resumed",
                "request_id": retry.OLD_REQUEST_ID, "request_sha256": digest(request_raw),
                "plan_sha256": self.plan_sha, "session_pid": spec.session_pid,
                "proc_start_ticks": spec.session_start, "gpu_uuid": spec.gpu_uuid,
                "source_proof": {"verified": True, "plan_sha256": self.plan_sha,
                    "source_hashes": self.source_hashes,
                    "session_source_sha256": digest((self.project / "preparations/ease_insights_20261103/gpu_resource_session.py").read_bytes()),
                    "keeper": {"nonce": spec.nonce}}}
            receipt_raw = write_json(self.campaign / "sessions" / spec.name / "resume_receipts" /
                                     (retry.OLD_REQUEST_ID + ".json"), receipt)
            expected["resumes/%s/%s.json" % (spec.name, retry.OLD_REQUEST_ID)] = digest(request_raw)
            expected["sessions/%s/resume_receipts/%s.json" % (spec.name, retry.OLD_REQUEST_ID)] = digest(receipt_raw)
        return retry.Baseline(project=self.project, experiment=self.experiment, campaign=self.campaign,
            report_dir=self.report, watch_dir=self.watch_dir, proc_root=self.proc_root,
            expected_plan_sha256=self.plan_sha, history_hashes=expected, slots=self.slots)

    def controller(self):
        def bind_probe(port):
            self.probe_calls.append(port)
            return port not in self.port_busy
        return retry.PortRetryController(self.baseline, bind_probe=bind_probe,
                                         clock=lambda: self.now, clock_ns=lambda: 123456789)

    def change_status(self, spec, phase):
        task = next((task for task in self.tasks if task is not None and task["id"].endswith(spec.name)), None)
        self._write_status(spec, phase, task if phase == "running" else None)
        if phase in ("idle_holding", "attention_holding"):
            self._write_children(spec.session_pid, [spec.keeper_pid])
            self._write_watcher(tuple(phase if other.name == spec.name else "idle_holding" for other in self.slots))


class RetryResourceSessionPortTests(unittest.TestCase):
    def setUp(self):
        self.fixture = Fixture()

    def tearDown(self):
        self.fixture.close()

    def test_publish_changes_only_id_and_activation_and_activates_before_atomic_resume(self):
        f = self.fixture
        # Make both exact port failures targets.
        spec = f.slots[1]
        task = f._make_task("task_original_slot01", "local_spatial", spec)
        f._write_status(spec, "attention_holding", task)
        f._write_outcome(spec, task)
        f.tasks[1] = task
        f._write_activation_for_tasks([f.tasks[0], task])
        f._write_watcher(("attention_holding", "attention_holding"))
        controller = f.controller()
        events = []
        original = controller._publish_atomically_no_replace

        def tracked(path, payload, nonce):
            events.append(Path(path))
            return original(path, payload, nonce)

        controller._publish_atomically_no_replace = tracked
        result = controller.run(apply=True)
        self.assertEqual(result["phase"], "published_wait_ack")
        transaction = Path(result["transaction_dir"])
        committed = json.loads((transaction / "committed.json").read_bytes())
        self.assertEqual(set(committed), {"schema", "plan_sha256", "task_sha256"})
        task_events = [path for path in events if path.parent == f.campaign / "tasks"]
        activation_index = events.index(transaction / "committed.json")
        resume_indices = [index for index, path in enumerate(events) if path.parent.parent == f.campaign / "resumes"]
        self.assertEqual(len(task_events), 2)
        self.assertLess(max(events.index(path) for path in task_events), activation_index)
        self.assertTrue(resume_indices)
        self.assertGreater(min(resume_indices), activation_index)
        self.assertEqual(set(committed["task_sha256"]), {"task_original_slot00__port_retry1", "task_original_slot01__port_retry1"})
        for spec, old in zip(f.slots, f.tasks):
            task_id = old["id"] + "__port_retry1"
            path = f.campaign / "tasks" / (task_id + ".json")
            new = json.loads(path.read_bytes())
            self.assertEqual({key for key in old if old[key] != new[key]}, {"id", "activation_file"})
            self.assertEqual(new["activation_file"], str(transaction / "committed.json"))
            self.assertEqual(committed["task_sha256"][task_id], digest(path.read_bytes()))
            request_path = f.campaign / "resumes" / spec.name / (result["request_id"] + ".json")
            request = json.loads(request_path.read_bytes())
            self.assertEqual(set(request), retry.RESUME_FIELDS)
            attention = {"task_id": old["id"], "error": retry.PORT_ERROR}
            self.assertEqual(request["attention_sha256"], retry.attention_sha256(attention))
        self.assertEqual(result["phase"], "published_wait_ack")
        self.assertFalse((f.experiment / "scheduler/jobs_state.json").exists())
        with self.assertRaisesRegex(retry.ValidationError, "Single retry task ID already exists"):
            controller.inspect()

    def test_refuses_nonport_attention_even_when_outcome_matches(self):
        f = self.fixture
        spec, task = f.slots[0], f.tasks[0]
        f._write_status(spec, "attention_holding", task)
        f._write_outcome(spec, task, error="RuntimeError('model failure')")
        with self.assertRaisesRegex(retry.ValidationError, "exact address-in-use"):
            f.controller().inspect()

    def test_refuses_when_target_complete_marker_exists(self):
        f = self.fixture
        marker = Path(f.tasks[0]["complete_file"])
        write_json(marker, {"complete": True})
        with self.assertRaisesRegex(retry.ValidationError, "complete marker already exists"):
            f.controller().inspect()

    def test_refuses_target_condition_with_any_committed_episode_receipt(self):
        f = self.fixture
        task = f.tasks[0]
        receipt = Path(task["output"]) / task["condition"] / "episodes/suite/task00/init36/episode_receipt.json"
        write_json(receipt, {"valid": True})
        with self.assertRaisesRegex(retry.ValidationError, "already has committed episode receipts"):
            f.controller().inspect()

    def test_confirmation_full_and_path_admissions_need_no_full_discovery_receipt(self):
        for condition in ("full_d10", "local_goal"):
            with self.subTest(condition=condition):
                f = Fixture(task_stage="confirm", attention_conditions=(condition, "local_spatial"))
                try:
                    audited = f.controller().inspect()
                    target = audited["slots"][0]["target"]
                    self.assertEqual(target["old_task"]["condition"], condition)
                    self.assertEqual(target["old_task"]["stage"], "confirm")
                    self.assertFalse((f.outputs[1] / "full_d10/episodes").exists())
                finally:
                    f.close()

    def test_refuses_foreign_or_pid_reused_session(self):
        f = self.fixture
        spec = f.slots[0]
        process_stat = f.proc_root / str(spec.session_pid) / "stat"
        process_stat.write_text(proc_stat(spec.session_pid, "R", spec.session_parent, "999999"))
        with self.assertRaisesRegex(retry.ValidationError, "identity differs"):
            f.controller().inspect()

    def test_refuses_foreign_argv_even_when_pid_and_start_ticks_match(self):
        f = self.fixture
        spec = f.slots[0]
        f._write_process(spec.session_pid, spec.session_start, spec.session_parent, ("/tmp/foreign-session",))
        f._write_children(spec.session_pid, [spec.keeper_pid])
        with self.assertRaisesRegex(retry.ValidationError, "identity differs"):
            f.controller().inspect()

    def test_refuses_retry_of_retry(self):
        f = self.fixture
        spec = f.slots[0]
        old = f.tasks[0]
        task = dict(old, id="task_original__port_retry1")
        task["logical_job_id"] = "logical_retry1"
        task["activation_file"] = str(f.campaign / "transactions/initial/committed.json")
        # Keep all scientific fields and create a matching immutable activation.
        path = f.campaign / "tasks" / (task["id"] + ".json")
        raw = write_json(path, task)
        activation = retry.read_json(f.campaign / "transactions/initial/committed.json")
        activation["task_sha256"][task["id"]] = digest(raw)
        write_json(f.campaign / "transactions/initial/committed.json", activation)
        f._write_status(spec, "attention_holding", task)
        f._write_outcome(spec, task)
        with self.assertRaisesRegex(retry.ValidationError, "retry of a retry"):
            f.controller().inspect()

    def test_busy_target_port_waits_without_mutation(self):
        f = self.fixture
        f.port_busy.add(f.slots[0].port)
        controller = f.controller()
        result = controller.run(apply=True)
        self.assertEqual(result["phase"], "waiting")
        self.assertFalse(list((f.campaign / "transactions").glob("port_retry_helper_*")))
        self.assertFalse(list((f.campaign / "resumes").glob("*/port_retry_ready_*.json")))

    def test_valid_running_keeper_server_worker_is_preserved_and_busy_port_skipped(self):
        f = Fixture(phases=("attention_holding", "running"), running_worker=True)
        self.addCleanup(f.close)
        f.port_busy.add(f.slots[1].port)
        result = f.controller().inspect()
        self.assertEqual(result["slots"][1]["slot"]["phase"], "running")
        self.assertEqual(result["slots"][1]["slot"]["science_child_roles"]["worker"]["pid"], 141)
        self.assertNotIn(f.slots[1].port, f.probe_calls)

    def test_running_slot_that_becomes_idle_with_busy_port_is_rechecked_and_rejected(self):
        f = Fixture(phases=("attention_holding", "running"), running_worker=False)
        self.addCleanup(f.close)
        controller = f.controller()
        initial = controller.inspect()
        f.port_busy.add(f.slots[1].port)
        spec = f.slots[1]
        f._write_status(spec, "idle_holding", None)
        f._write_children(spec.session_pid, [spec.keeper_pid])
        f._write_watcher(("attention_holding", "idle_holding"))
        with self.assertRaisesRegex(retry.ValidationError, "port remains busy"):
            controller.apply(initial)
        self.assertIn(spec.port, f.probe_calls)
        self.assertFalse(list((f.campaign / "transactions").glob("port_retry_helper_*")))

    def test_default_mode_is_dry_run_and_creates_no_campaign_artifacts(self):
        f = self.fixture
        result = f.controller().run()
        self.assertEqual(result["phase"], "dry_run_ready")
        self.assertEqual(result["mode"], "dry_run")
        self.assertFalse(list((f.campaign / "transactions").glob("port_retry_helper_*")))
        self.assertFalse(list((f.campaign / "resumes").glob("*/port_retry_ready_*.json")))


if __name__ == "__main__":
    unittest.main()
