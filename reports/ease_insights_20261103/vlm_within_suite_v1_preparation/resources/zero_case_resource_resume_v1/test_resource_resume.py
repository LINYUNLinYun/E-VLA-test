"""Adversarial fake-data contracts for the external resume controller.

These tests use only temporary CPU fixtures. They never inspect live stage
records, query a GPU, invoke a model, or call ResourceResumeController.run().
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import resource_resume as rr


GPU0 = "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1"
GPU1 = "GPU-599477fb-0507-40dc-14df-d491f7a7b202"
CHILD_PID = 812345
CHILD_START = "900123"
FOREIGN_PID = 987654
PEER_PID = 823456


def cached_status(gpu_uuid: str, query: float, host_pids: list[int] | None = None,
                  count: int | None = None) -> dict:
    keeper = rr.GPU_PINS[gpu_uuid]["host_keeper_pid"]
    host_pids = host_pids if host_pids is not None else [keeper, FOREIGN_PID]
    count = count if count is not None else len(host_pids)
    return {
        "pid": rr.WATCH_PID,
        "proc_start_ticks": rr.WATCH_START_TICKS,
        "run_id": rr.WATCH_RUN_ID,
        "source_sha256": rr.WATCH_SOURCE_SHA256,
        "error": None,
        "poll_seconds": 2.0,
        "query_finished_at": query,
        "devices": [{"gpu_uuid": gpu_uuid, "compute_process_count": count,
                     "host_compute_pids": host_pids, "memory_mib": 12000}],
    }


def owner_status(gpu_uuid: str, query: float, memory_mib: float = 256) -> dict:
    keeper = rr.GPU_PINS[gpu_uuid]["host_keeper_pid"]
    status = cached_status(gpu_uuid, query, [keeper], 1)
    status["devices"][0]["memory_mib"] = memory_mib
    return status


def service_fixture(root: Path, *, log_text: str | None = None) -> tuple[Path, dict[str, set[str]], dict[str, set[str]], dict]:
    stage = root / "stage"
    for condition in rr.CONDITIONS:
        (stage / condition / "runs").mkdir(parents=True, exist_ok=True)
    condition = "full_d10"
    name = "discovery_shard00of01_fake-oom"
    run_dir = stage / condition / "runs" / name
    run_dir.mkdir()
    identity = service_identity(stage, run_dir, condition, GPU0, CHILD_PID)
    (run_dir / "identity.json").write_text(json.dumps(identity))
    if log_text is None:
        log_text = (
            "Traceback (most recent call last):\n"
            "  File \"server.py\", line 41, in start\n"
            "    backend = Pi0Inference(...)\n"
            "  File \"/opt/pi0_infer.py\", line 204, in __init__\n"
            "    self.model = load_model()\n"
            "RuntimeError: CUDA error: out of memory\n"
            "CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.\n"
            "For debugging consider passing CUDA_LAUNCH_BLOCKING=1\n"
            "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.\n"
        )
    (run_dir / "server.log").write_text(log_text)
    before = {cid: set() for cid in rr.CONDITIONS}
    after = {cid: set(rows) for cid, rows in before.items()}
    after[condition] = {name}
    markers = {cid: None for cid in rr.CONDITIONS}
    return stage, before, after, markers


def service_identity(stage: Path, run_dir: Path, condition: str,
                     gpu_uuid: str, controller_pid: int) -> dict:
    server_module = "preparations.ease_insights_20261103.vlm_within_suite_v1.server"
    worker_module = "preparations.ease_insights_20261103.vlm_within_suite_v1.worker"
    return {
        "manifest_hash": rr.MANIFEST_HASH,
        "condition": condition,
        "controller_pid": controller_pid,
        "stateful_workers_per_server": 1,
        "technical_runtime_revision": {
            "scientific_manifest_hash": rr.MANIFEST_HASH,
            "phase": "within_suite_discovery_v1",
        },
        "resource_execution_revision": {
            "scientific_manifest_hash": rr.MANIFEST_HASH,
            "controller_pid": controller_pid,
            "controller_source_sha256": rr.BORROWER_SHA256,
        },
        "server_command": [
            "python", "-m", server_module,
            "--condition", condition,
            "--gpu-uuid", gpu_uuid,
            "--manifest", str(stage / "manifest.json"),
            "--output", str(run_dir),
        ],
        "worker_command": [
            "python", "-m", worker_module,
            "--condition", condition,
            "--stage", "discovery",
            "--output", str(stage),
        ],
    }


def add_peer_run(stage: Path, *, condition: str = "local_goal", name: str = "discovery_shard00of01_peer",
                 gpu_uuid: str = GPU1, controller_pid: int = PEER_PID) -> Path:
    run_dir = stage / condition / "runs" / name
    run_dir.mkdir()
    (run_dir / "identity.json").write_text(json.dumps(
        service_identity(stage, run_dir, condition, gpu_uuid, controller_pid)))
    (run_dir / "server.log").write_text("peer startup\n")
    return run_dir


def add_complete_marker(stage: Path, condition: str) -> dict:
    path = stage / condition / "discovery_shard00of01_complete.json"
    marker = {"complete": True, "manifest_hash": rr.MANIFEST_HASH,
              "condition": condition, "completed_after": 72}
    path.write_text(json.dumps(marker))
    return {"path": str(path), "sha256": rr.sha256_file(path), "cases": 72}


class AvailabilityContracts(unittest.TestCase):
    def test_wait_window_requires_nine_distinct_samples_and_fifteen_seconds(self):
        window = rr.AvailabilityWindow(GPU0)
        self.assertFalse(window.ready)
        for query in range(100, 117, 2):
            window.consume(owner_status(GPU0, query), query + 1)
        self.assertEqual(len(window.samples), 9)
        self.assertEqual(window.elapsed, 16)
        self.assertTrue(window.ready)

    def test_busy_backwards_gap_and_stale_cache_reset_the_window(self):
        window = rr.AvailabilityWindow(GPU0)
        self.assertFalse(window.consume(owner_status(GPU0, 100), 101))
        self.assertFalse(window.consume(owner_status(GPU0, 102), 103))
        busy = cached_status(GPU0, 104, [rr.GPU_PINS[GPU0]["host_keeper_pid"], FOREIGN_PID], 2)
        self.assertFalse(window.consume(busy, 105))
        self.assertEqual(window.samples, [])
        self.assertFalse(window.consume(owner_status(GPU0, 110), 111))
        self.assertFalse(window.consume(owner_status(GPU0, 108), 109))
        self.assertEqual(len(window.samples), 1)
        self.assertFalse(window.consume(owner_status(GPU0, 120), 124))
        self.assertEqual(window.samples, [])

    def test_post_exit_foreign_requires_three_fresh_distinct_polls_and_shared_pid(self):
        observations = [
            (cached_status(GPU0, 1000), 1001),
            (cached_status(GPU0, 1000), 1002),  # same cached poll cannot count twice
            (cached_status(GPU0, 1002), 1003),
            (cached_status(GPU0, 1004), 1005),
        ]
        result = rr.persistent_extra_foreign_compute(observations, GPU0, 999, {CHILD_PID})
        self.assertIsNotNone(result)
        self.assertEqual(len(result["observations"]), 3)
        self.assertEqual(result["shared_unattributed_foreign_pids"], [FOREIGN_PID])

        changing = [
            (cached_status(GPU0, 1000, [rr.GPU_PINS[GPU0]["host_keeper_pid"], 1]), 1001),
            (cached_status(GPU0, 1002, [rr.GPU_PINS[GPU0]["host_keeper_pid"], 2]), 1003),
            (cached_status(GPU0, 1004, [rr.GPU_PINS[GPU0]["host_keeper_pid"], 3]), 1005),
        ]
        self.assertIsNone(rr.persistent_extra_foreign_compute(changing, GPU0, 999, {CHILD_PID}))

    def test_post_exit_wait_uses_cached_reader_and_two_second_cadence(self):
        queries = iter((1000, 1000, 1002, 1004))
        now = [1001.0]
        reads = []
        sleeps = []

        def status_reader():
            reads.append(True)
            return cached_status(GPU0, next(queries))

        def clock():
            return now[0]

        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += seconds

        observations = rr.wait_for_post_exit_foreign(
            status_reader, GPU0, 999, {CHILD_PID}, clock=clock, sleep=sleep)
        self.assertEqual(len(reads), 4)
        self.assertEqual(sleeps, [2.0, 2.0, 2.0])
        self.assertIsNotNone(rr.persistent_extra_foreign_compute(observations, GPU0, 999, {CHILD_PID}))


class FailureClassifierContracts(unittest.TestCase):
    def classify(self, temp: Path, log: str | None = None,
                 observations: list[tuple[dict, float]] | None = None) -> dict:
        stage, before, after, markers = service_fixture(temp, log_text=log)
        if observations is None:
            observations = [(cached_status(GPU0, t), t + 1) for t in (1000, 1002, 1004)]
        return rr.classify_zero_case_startup_oom(
            stage_root=stage, gpu_uuid=GPU0, child_pid=CHILD_PID,
            child_start_ticks=CHILD_START, child_returncode=1,
            exit_time=999, own_group_members=[], before_runs=before, after_runs=after,
            complete_before=markers, complete_after=markers,
            post_exit_statuses=observations, now=1005, assigned_registry_identity_ok=True)

    def test_exact_last_pi0_init_traceback_and_three_foreign_polls_is_retryable(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, _, _, _ = service_fixture(Path(tmp))
            run_dir = stage / "full_d10" / "runs" / "discovery_shard00of01_fake-oom"
            identity = json.loads((run_dir / "identity.json").read_text())
            log = (run_dir / "server.log").read_text()
            self.assertNotIn("stage", identity)
            self.assertEqual(identity["technical_runtime_revision"]["phase"], "within_suite_discovery_v1")
            self.assertTrue(log.rstrip().endswith(
                "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions."))
            result = self.classify(Path(tmp) / "classification")
        self.assertEqual(result["classification"], "retryable_zero_case_pi0_init_oom_with_foreign_compute")
        self.assertTrue(result["retryable"])

    def test_peer_progress_during_wait_is_validated_and_becomes_new_baseline(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            for condition in rr.CONDITIONS:
                (stage / condition / "runs").mkdir(parents=True, exist_ok=True)
            before_runs = {condition: set() for condition in rr.CONDITIONS}
            before_markers = {condition: None for condition in rr.CONDITIONS}
            add_peer_run(stage)
            add_complete_marker(stage, "local_spatial")
            after_runs = rr.snapshot_run_directories(stage)
            after_markers = rr.snapshot_complete_markers(stage)
            rr.assert_prior_progress_preserved(before_markers, after_markers,
                                               before_runs, after_runs)
            own_runs = rr.validate_added_run_directories(
                stage, before_runs=before_runs, after_runs=after_runs,
                gpu_uuid=GPU0, own_controller_pid=None,
                peer_pids={GPU1: PEER_PID})
        self.assertEqual(own_runs, [])
        self.assertIsNotNone(after_markers["local_spatial"])
        self.assertEqual(after_runs["local_goal"], {"discovery_shard00of01_peer"})

    def test_peer_progress_during_child_does_not_hide_own_zero_case_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, before, _, markers = service_fixture(Path(tmp))
            add_peer_run(stage)
            add_complete_marker(stage, "local_spatial")
            after = rr.snapshot_run_directories(stage)
            markers_after = rr.snapshot_complete_markers(stage)
            observations = [(cached_status(GPU0, t), t + 1) for t in (1000, 1002, 1004)]
            result = rr.classify_zero_case_startup_oom(
                stage_root=stage, gpu_uuid=GPU0, child_pid=CHILD_PID,
                child_start_ticks=CHILD_START, child_returncode=1, exit_time=999,
                own_group_members=[], before_runs=before, after_runs=after,
                complete_before=markers, complete_after=markers_after,
                post_exit_statuses=observations, now=1005,
                assigned_registry_identity_ok=True, peer_pids={GPU1: PEER_PID})
        self.assertEqual(result["classification"], "retryable_zero_case_pi0_init_oom_with_foreign_compute")

    def test_mutation_or_deletion_of_preexisting_progress_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            for condition in rr.CONDITIONS:
                (stage / condition / "runs").mkdir(parents=True, exist_ok=True)
            marker = add_complete_marker(stage, "local_spatial")
            before_markers = rr.snapshot_complete_markers(stage)
            before_runs = {condition: set() for condition in rr.CONDITIONS}
            after_runs = {condition: set() for condition in rr.CONDITIONS}
            after_runs["local_goal"] = {"preexisting"}
            after_markers = dict(before_markers)
            after_markers["local_spatial"] = None
            with self.assertRaises(rr.AttentionRequired):
                rr.assert_prior_progress_preserved(before_markers, after_markers,
                                                   before_runs, after_runs)
            after_markers["local_spatial"] = marker
            before_runs["local_goal"] = {"preexisting"}
            with self.assertRaises(rr.AttentionRequired):
                rr.assert_prior_progress_preserved(before_markers, after_markers,
                                                   before_runs, {condition: set() for condition in rr.CONDITIONS})

    def test_earlier_oom_does_not_override_last_non_oom_traceback(self):
        text = (
            "Traceback (most recent call last):\n"
            "  File \"server.py\", line 1, in start\n"
            "    Pi0Inference(...)\n"
            "  File \"pi0_infer.py\", line 5, in __init__\n"
            "RuntimeError: CUDA error: out of memory\n"
            "Traceback (most recent call last):\n"
            "  File \"server.py\", line 2, in start\n"
            "ValueError: later unrelated failure\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = self.classify(Path(tmp), log=text)
        self.assertEqual(result["classification"], "attention")
        self.assertEqual(result["reason"], "last_traceback_not_exact_pre_health_pi0_inference_oom")
        self.assertEqual(result["last_exception"], "ValueError: later unrelated failure")

    def test_unrecognized_exception_or_text_after_oom_is_not_an_allowed_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.classify(Path(tmp), log=service_fixture_log() + "UnexpectedError: tail\n")
        self.assertEqual(result["classification"], "attention")
        self.assertFalse(result["recognized_warning_tail"])

    def test_other_exception_header_in_last_traceback_is_rejected(self):
        log = service_fixture_log().replace(
            "RuntimeError: CUDA error: out of memory\n",
            "ValueError: unrelated failure\nRuntimeError: CUDA error: out of memory\n")
        with tempfile.TemporaryDirectory() as tmp:
            result = self.classify(Path(tmp), log=log)
        self.assertEqual(result["classification"], "attention")
        self.assertFalse(result["retryable"])
        self.assertEqual(result["traceback_exception_headers"],
                         ["ValueError: unrelated failure", "RuntimeError: CUDA error: out of memory"])

    def test_constructor_and_init_frame_cannot_be_combined_across_tracebacks(self):
        text = (
            "Traceback (most recent call last):\n"
            "  File \"server.py\", line 1, in start\n"
            "    Pi0Inference(...)\n"
            "ValueError: earlier failure\n"
            "Traceback (most recent call last):\n"
            "  File \"/opt/pi0_infer.py\", line 204, in __init__\n"
            "RuntimeError: CUDA error: out of memory\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            result = self.classify(Path(tmp), log=text)
        self.assertEqual(result["classification"], "attention")
        self.assertFalse(result["pi0_constructor_call"])
        self.assertTrue(result["pi0_infer_init_frame"])

    def test_extra_run_artifact_forces_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, before, after, markers = service_fixture(Path(tmp))
            run_dir = stage / "full_d10" / "runs" / "discovery_shard00of01_fake-oom"
            (run_dir / "protocol.json").write_text("{}")
            result = rr.classify_zero_case_startup_oom(
                stage_root=stage, gpu_uuid=GPU0, child_pid=CHILD_PID,
                child_start_ticks=CHILD_START, child_returncode=1, exit_time=999,
                own_group_members=[], before_runs=before, after_runs=after,
                complete_before=markers, complete_after=markers,
                post_exit_statuses=[], now=1000, assigned_registry_identity_ok=True)
        self.assertEqual(result["classification"], "attention")
        self.assertEqual(result["reason"], "startup_run_not_exact_zero_case_artifact_set")

    def test_any_health_protocol_worker_or_episode_progression_forces_attention(self):
        for token in ("READY", "health check passed", "protocol handshake", "worker started", "episode 0"):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as tmp:
                result = self.classify(Path(tmp), log=token + "\n" + service_fixture_log())
            self.assertEqual(result["classification"], "attention")
            self.assertEqual(result["reason"], "last_traceback_not_exact_pre_health_pi0_inference_oom")

    def test_single_foreign_sample_is_not_enough_for_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            observations = [(cached_status(GPU0, 1000), 1001)]
            result = self.classify(Path(tmp), observations=observations)
        self.assertEqual(result["classification"], "attention")
        self.assertEqual(result["reason"], "no_persistent_extra_foreign_compute_after_child_exit")

    def test_own_registry_identity_corruption_forces_attention_even_on_exit_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            stage, before, after, markers = service_fixture(Path(tmp))
            result = rr.classify_zero_case_startup_oom(
                stage_root=stage, gpu_uuid=GPU0, child_pid=CHILD_PID,
                child_start_ticks=CHILD_START, child_returncode=0, exit_time=999,
                own_group_members=[], before_runs=before, after_runs=after,
                complete_before=markers, complete_after=markers, post_exit_statuses=[],
                now=1000, assigned_registry_identity_ok=False)
        self.assertEqual(result["classification"], "attention")
        self.assertEqual(result["reason"], "assigned_gpu_registry_identity_changed")


def service_fixture_log() -> str:
    return (
        "Traceback (most recent call last):\n"
        "  File \"server.py\", line 41, in start\n"
        "    backend = Pi0Inference(...)\n"
        "  File \"/opt/pi0_infer.py\", line 204, in __init__\n"
        "    self.model = load_model()\n"
        "RuntimeError: CUDA error: out of memory\n"
    )


class JournalAndRegistryContracts(unittest.TestCase):
    @staticmethod
    def write_complete_attempt(resource_dir: Path, *, gpu_uuid: str = GPU0,
                               attempt_id: str = "1700000000000000000") -> Path:
        attempt_dir = resource_dir / "attempts" / gpu_uuid / attempt_id
        attempt_dir.mkdir(parents=True)
        argv = rr.expected_argv(gpu_uuid)
        predecessor = {"pid": 765432, "start_ticks": "876543"}
        child = {
            "schema": "zero-case-resource-resume-child-v1",
            "attempt_id": attempt_id,
            "time": 10.0,
            "pid": 876543,
            "start_ticks": "987654",
            "argv": argv,
            "log": str(attempt_dir / "borrowed_controller.log"),
            "gpu_uuid": gpu_uuid,
            "manifest_hash": rr.MANIFEST_HASH,
        }
        exit_receipt = {"attempt_id": attempt_id, "time": 20.0, "returncode": 0,
                        "pid": child["pid"], "start_ticks": child["start_ticks"]}
        intent = {
            "schema": "zero-case-resource-resume-intent-v1",
            "attempt_id": attempt_id,
            "gpu_uuid": gpu_uuid,
            "manifest_hash": rr.MANIFEST_HASH,
            "manifest_file_sha256": rr.MANIFEST_FILE_SHA256,
            "argv": argv,
            "predecessor": predecessor,
            "shared_registry_sha256_before": "a" * 64,
        }
        registry_receipt = {
            "attempt_id": attempt_id,
            "shared_registry_before_sha256": "a" * 64,
            "shared_registry_after_sha256": "b" * 64,
            "predecessor": predecessor,
            "child": {"pid": child["pid"], "start_ticks": child["start_ticks"]},
            "registry_updated": True,
        }
        assigned_row = {
            "gpu_uuid": gpu_uuid,
            "pid": child["pid"],
            "start_ticks": child["start_ticks"],
            "argv": argv,
            "actual_argv": argv,
            "log": child["log"],
            "resource_resume_attempt": attempt_id,
            "state": "exited",
            "assignment_exit_recorded": True,
            "exit_code": 0,
        }
        exit_registry = {
            "attempt_id": attempt_id,
            "registry_updated": True,
            "shared_registry_after_exit_sha256": "c" * 64,
            "child": {"pid": child["pid"], "start_ticks": child["start_ticks"]},
            "assigned_row": assigned_row,
        }
        for name, value in (("intent.json", intent), ("child_receipt.json", child),
                            ("exit.json", exit_receipt), ("registry_receipt.json", registry_receipt),
                            ("exit_registry_receipt.json", exit_registry)):
            (attempt_dir / name).write_text(json.dumps(value))
        classification = {
            "classification": "assignment_complete",
            "retryable": False,
            "reason": "borrower_exit_zero",
            "attempt_id": attempt_id,
            "gpu_uuid": gpu_uuid,
            "manifest_hash": rr.MANIFEST_HASH,
            "borrower_argv": argv,
            "predecessor": predecessor,
            "child_pid": child["pid"],
            "child_start_ticks": child["start_ticks"],
            "shared_registry_sha256_after_launch": "b" * 64,
            "shared_registry_sha256_after_exit": "c" * 64,
            "assigned_registry_row_after_exit": assigned_row,
            "child_receipt_sha256": rr.sha256_file(attempt_dir / "child_receipt.json"),
            "exit_receipt_sha256": rr.sha256_file(attempt_dir / "exit.json"),
            "exit_registry_receipt_sha256": rr.sha256_file(attempt_dir / "exit_registry_receipt.json"),
        }
        (attempt_dir / "classification.json").write_text(json.dumps(classification))
        return attempt_dir

    def test_complete_attempt_chain_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_dir = Path(tmp) / "resources"
            self.write_complete_attempt(resource_dir)
            controller = rr.ResourceResumeController(
                root=Path(tmp), stage_root=Path(tmp) / "stage", resource_dir=resource_dir)
            prior, terminal = controller._preflight_journal(GPU0)
        self.assertEqual(prior, [])
        self.assertEqual(terminal["classification"]["classification"], "assignment_complete")

    def test_complete_but_mismatched_intent_child_and_receipt_hashes_fail_closed(self):
        mutations = (
            ("intent.json", lambda row: row.update(gpu_uuid=GPU1)),
            ("intent.json", lambda row: row.update(manifest_hash="f" * 64)),
            ("intent.json", lambda row: row["predecessor"].update(start_ticks="000000")),
            ("child_receipt.json", lambda row: row.update(argv=["python", "different"])),
            ("child_receipt.json", lambda row: row.update(attempt_id="other-attempt")),
            ("classification.json", lambda row: row.update(gpu_uuid=GPU1)),
            ("classification.json", lambda row: row.update(manifest_hash="f" * 64)),
            ("classification.json", lambda row: row.update(borrower_argv=["python", "different"])),
            ("classification.json", lambda row: row["predecessor"].update(pid=1)),
            ("classification.json", lambda row: row.update(child_pid=1)),
            ("classification.json", lambda row: row.update(child_receipt_sha256="0" * 64)),
            ("classification.json", lambda row: row.update(exit_receipt_sha256="0" * 64)),
            ("classification.json", lambda row: row.update(exit_registry_receipt_sha256="0" * 64)),
        )
        for filename, mutate in mutations:
            with self.subTest(filename=filename, mutation=mutate), tempfile.TemporaryDirectory() as tmp:
                resource_dir = Path(tmp) / "resources"
                attempt_dir = self.write_complete_attempt(resource_dir)
                path = attempt_dir / filename
                row = json.loads(path.read_text())
                mutate(row)
                path.write_text(json.dumps(row))
                controller = rr.ResourceResumeController(
                    root=Path(tmp), stage_root=Path(tmp) / "stage", resource_dir=resource_dir)
                with self.assertRaises(rr.ManualReconcileRequired):
                    controller._preflight_journal(GPU0)

    def test_next_attempt_predecessor_must_be_previous_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_dir = Path(tmp) / "resources"
            attempt_dir = self.write_complete_attempt(resource_dir)
            load = lambda name: json.loads((attempt_dir / name).read_text())
            docs = {
                "intent": load("intent.json"),
                "child": load("child_receipt.json"),
                "exit_receipt": load("exit.json"),
                "classification": load("classification.json"),
                "registry_receipt": load("registry_receipt.json"),
                "exit_registry": load("exit_registry_receipt.json"),
            }
            args = dict(attempt_dir=attempt_dir, gpu_uuid=GPU0, **docs)
            rr.validate_attempt_journal_chain(
                **args, previous_child={"pid": 765432, "start_ticks": "876543"})
            with self.assertRaises(rr.ManualReconcileRequired):
                rr.validate_attempt_journal_chain(
                    **args, previous_child={"pid": 123456, "start_ticks": "654321"})

    def test_missing_complete_chain_receipt_still_requires_manual_reconcile(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_dir = Path(tmp) / "resources"
            attempt_dir = self.write_complete_attempt(resource_dir)
            (attempt_dir / "exit_registry_receipt.json").unlink()
            controller = rr.ResourceResumeController(
                root=Path(tmp), stage_root=Path(tmp) / "stage", resource_dir=resource_dir)
            with self.assertRaises(rr.ManualReconcileRequired):
                controller._preflight_journal(GPU0)

    def test_o_excl_intent_prevents_duplicate_attempt_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intent.json"
            rr.durable_exclusive_json(path, {"attempt_id": "one"})
            with self.assertRaises(FileExistsError):
                rr.durable_exclusive_json(path, {"attempt_id": "duplicate"})
            self.assertEqual(json.loads(path.read_text())["attempt_id"], "one")

    def test_exit_update_preserves_peer_row_and_checks_only_assigned_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "controller_processes.json"
            lock_path = Path(tmp) / "registry_rmw.lock"
            argv = rr.expected_argv(GPU0)
            child = {"pid": CHILD_PID, "start_ticks": CHILD_START,
                     "argv": argv, "log": "/tmp/attempt/borrowed_controller.log",
                     "attempt_id": "attempt-1"}
            own_row = {"gpu_uuid": GPU0, "pid": CHILD_PID, "start_ticks": CHILD_START,
                       "argv": argv, "actual_argv": argv, "log": child["log"],
                       "resource_resume_attempt": child["attempt_id"], "state": "Z"}
            peer_row = {"gpu_uuid": GPU1, "pid": 23456, "start_ticks": "654321",
                        "argv": rr.expected_argv(GPU1), "actual_argv": rr.expected_argv(GPU1),
                        "peer_update": "concurrent-during-assignment"}
            registry = {"manifest_hash": rr.MANIFEST_HASH,
                        "manifest_file_sha256": rr.MANIFEST_FILE_SHA256,
                        "time": 123.0, "processes": [own_row, peer_row]}
            registry_path.write_text(json.dumps(registry))
            ok, after_sha, updated, reason = rr.update_assigned_registry_after_exit(
                registry_path, GPU0, child, 1, 456.0, lock_path=lock_path)
            current = json.loads(registry_path.read_text())
        self.assertTrue(ok)
        self.assertIsNone(reason)
        self.assertEqual(current["processes"][0]["state"], "exited")
        self.assertEqual(current["processes"][0]["exit_code"], 1)
        self.assertEqual(current["processes"][1], peer_row)
        self.assertEqual(current["time"], 123.0)
        self.assertEqual(updated["pid"], CHILD_PID)
        self.assertEqual(len(after_sha), 64)

    def test_assigned_row_corruption_does_not_rewrite_shared_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            lock = Path(tmp) / "registry.lock"
            argv = rr.expected_argv(GPU0)
            child = {"pid": CHILD_PID, "start_ticks": CHILD_START,
                     "log": "/tmp/attempt.log", "attempt_id": "x"}
            registry = {"manifest_hash": rr.MANIFEST_HASH,
                        "manifest_file_sha256": rr.MANIFEST_FILE_SHA256,
                        "processes": [{"gpu_uuid": GPU0, "pid": CHILD_PID,
                                       "start_ticks": CHILD_START, "argv": argv,
                                       "actual_argv": argv, "log": "changed.log",
                                       "resource_resume_attempt": "x"}]}
            path.write_text(json.dumps(registry))
            before = path.read_bytes()
            ok, _, row, reason = rr.update_assigned_registry_after_exit(path, GPU0, child, 1, 10, lock_path=lock)
            after = path.read_bytes()
        self.assertFalse(ok)
        self.assertEqual(before, after)
        self.assertEqual(row["log"], "changed.log")
        self.assertEqual(reason, "assigned_gpu_registry_identity_changed_at_exit")


if __name__ == "__main__":
    unittest.main()
