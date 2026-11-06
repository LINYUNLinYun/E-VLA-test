"""CPU-only contract tests for borrowed_controller; all process/GPU edges are faked."""
from __future__ import annotations

from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103.vlm_within_suite_v1 import borrowed_controller as borrowed


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def put_json(path: Path, value) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(raw)
    return raw


class CampaignFixture:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.campaign = self.root / "campaign"
        self.tasks = self.campaign / "tasks"
        self.outcomes = self.campaign / "outcomes"
        self.transactions = self.campaign / "transactions"
        self.initial = self.transactions / "initial"
        self.retry_tx = self.transactions / "port_retry"
        self.initial.mkdir(parents=True)
        self.retry_tx.mkdir(parents=True)
        self.tasks.mkdir(parents=True)
        self.outcomes.mkdir(parents=True)
        self.base = self.task("task_base")
        self.retry = dict(self.base, id=self.base["id"] + "__port_retry1",
                          activation_file=str(self.retry_tx / "committed.json"))
        self.base["activation_file"] = str(self.initial / "committed.json")
        self.retry["activation_file"] = str(self.retry_tx / "committed.json")
        self._write_task(self.base)
        self._write_task(self.retry)
        put_json(self.initial / "committed.json", {
            "schema": "ease-gpu-resource-activation-v1", "plan_sha256": borrowed.remote.PLAN_SHA256,
            "task_sha256": {self.base["id"]: self.base_sha}})
        put_json(self.retry_tx / "committed.json", {
            "schema": "ease-gpu-resource-activation-v1", "plan_sha256": borrowed.remote.PLAN_SHA256,
            "task_sha256": {self.retry["id"]: self.retry_sha}})
        self._write_outcome(self.base, "attention", error=borrowed.remote.PORT_ERROR)
        self._write_outcome(self.retry, "complete")
        self._write_marker(self.base)
        self._write_marker(self.retry)

    def close(self):
        self.tmp.cleanup()

    def task(self, task_id):
        marker = self.root / "science" / "condition_a" / "discovery_shard00of01_complete.json"
        return {"schema": "ease-gpu-resource-task-v1", "id": task_id,
                "logical_job_id": "logical_a", "argv": ["python", "-m", "frozen.runner", "--condition", "a"],
                "complete_file": str(marker), "manifest_hash": "manifest-a", "output": str(marker.parent.parent),
                "condition": "condition_a", "stage": "discovery", "shard_index": 0, "shards": 1,
                "env": {}, "activation_file": ""}

    def _write_task(self, task):
        raw = put_json(self.tasks / (task["id"] + ".json"), task)
        if task["id"] == self.base["id"]:
            self.base_sha = digest_bytes(raw)
        if task["id"] == self.retry["id"]:
            self.retry_sha = digest_bytes(raw)

    def _outcome_path(self, task):
        return self.outcomes / (hashlib.sha256(task["id"].encode()).hexdigest() + ".json")

    def _write_outcome(self, task, phase, error=None):
        value = {"task_id": task["id"], "task_sha256": digest_bytes((self.tasks / (task["id"] + ".json")).read_bytes()),
                 "phase": phase, "error": error}
        put_json(self._outcome_path(task), value)

    def _write_marker(self, task):
        put_json(Path(task["complete_file"]), {"complete": True, "manifest_hash": task["manifest_hash"],
                                         "condition": task["condition"]})


class FakeHolderGuard:
    def __init__(self, proof=None):
        self.calls = []
        self.fail = False
        self.specs = {"GPU-test": object()}
        self.proof = proof or {"keeper_ready": {"pid": 12, "nonce": "n"},
                               "keeper": {"pid": 12, "start_ticks": "34"}, "binding": {"lease_path": "/tmp/fake"}}

    def quick(self, gpu_uuid):
        self.calls.append(gpu_uuid)
        if self.fail:
            raise RuntimeError("original holder identity lost")
        if gpu_uuid not in self.specs:
            raise RuntimeError("foreign GPU")
        return self.proof

    def full(self, require_broker_stopped=True):
        self.calls.append(("full", require_broker_stopped))
        return {"schema": "fake-holder-proof", "broker_stopped": require_broker_stopped}


class FakeChild:
    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False
        self.waited = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


class BorrowedControllerCpuContractTests(unittest.TestCase):
    def test_remote_fd_proof_requires_same_inode_and_write_flock(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            lease = root / "gpu.lock"
            lease.write_text("lease")
            device, inode = lease.stat().st_dev, lease.stat().st_ino
            proc = root / "proc" / "631907"
            (proc / "fd").mkdir(parents=True)
            (proc / "fdinfo").mkdir()
            (proc / "fd" / "4").symlink_to(lease)
            (proc / "fdinfo" / "4").write_text("pos:\t0\nflags:\t0100002\nlock:\t1: FLOCK ADVISORY WRITE 631907 00:01:1 0 EOF\n")
            binding = {"lease_path": str(lease), "lease_device": device, "lease_inode": inode, "lease_fds": [4]}
            proof = borrowed.verify_remote_lease(binding, 631907, root / "proc")
            self.assertEqual(proof, [{"fd": 4, "device": device, "inode": inode}])

    def test_remote_fd_proof_rejects_inode_mismatch_and_read_lock(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            lease, foreign = root / "gpu.lock", root / "foreign.lock"
            lease.write_text("lease"); foreign.write_text("foreign")
            proc = root / "proc" / "99"
            (proc / "fd").mkdir(parents=True); (proc / "fdinfo").mkdir()
            link = proc / "fd" / "4"
            link.symlink_to(foreign)
            (proc / "fdinfo" / "4").write_text("lock:\t1: FLOCK ADVISORY WRITE 99 00:01:1 0 EOF\n")
            binding = {"lease_path": str(lease), "lease_device": lease.stat().st_dev,
                       "lease_inode": lease.stat().st_ino, "lease_fds": [4]}
            with self.assertRaisesRegex(RuntimeError, "inode/device|no longer holds"):
                borrowed.verify_remote_lease(binding, 99, root / "proc")
            link.unlink(); link.symlink_to(lease)
            (proc / "fdinfo" / "4").write_text("lock:\t1: FLOCK ADVISORY READ 99 00:01:1 0 EOF\n")
            with self.assertRaisesRegex(RuntimeError, "exclusive FLOCK"):
                borrowed.verify_remote_lease(binding, 99, root / "proc")

    def test_one_same_science_completed_retry_resolves_old_attention(self):
        fixture = CampaignFixture()
        self.addCleanup(fixture.close)
        result = borrowed.campaign_completed(fixture.campaign)
        self.assertEqual(result["tasks"], 2)
        self.assertEqual(set(result["task_hashes"]), {fixture.base["id"], fixture.retry["id"]})

    def test_campaign_rejects_uncommitted_activation_uncompleted_outcome_and_task_tamper(self):
        for failure in ("uncommitted", "unfinished", "tampered"):
            with self.subTest(failure=failure):
                fixture = CampaignFixture()
                try:
                    if failure == "uncommitted":
                        (fixture.transactions / "pending_dispatch").mkdir()
                    elif failure == "unfinished":
                        fixture._write_outcome(fixture.retry, "attention", error=borrowed.remote.PORT_ERROR)
                    else:
                        changed = dict(fixture.base, logical_job_id="tampered")
                        put_json(fixture.tasks / (changed["id"] + ".json"), changed)
                    with self.assertRaises(RuntimeError):
                        borrowed.campaign_completed(fixture.campaign)
                finally:
                    fixture.close()

    def test_campaign_rejects_retry_with_changed_science_even_when_activation_matches(self):
        fixture = CampaignFixture()
        self.addCleanup(fixture.close)
        changed = dict(fixture.retry, argv=["python", "-m", "different.runner"])
        fixture._write_task(changed)
        activation = json.loads((fixture.retry_tx / "committed.json").read_text())
        activation["task_sha256"][changed["id"]] = fixture.retry_sha
        put_json(fixture.retry_tx / "committed.json", activation)
        fixture._write_outcome(changed, "complete")
        with self.assertRaisesRegex(RuntimeError, "changed science"):
            borrowed.campaign_completed(fixture.campaign)

    def test_campaign_rejects_multiple_retry_suffixes(self):
        """An old attention cannot be treated as resolved by ambiguous retry suffixes."""
        fixture = CampaignFixture()
        self.addCleanup(fixture.close)
        second = dict(fixture.base, id=fixture.base["id"] + "__port_retry2",
                      activation_file=str(fixture.transactions / "retry2" / "committed.json"))
        second_dir = fixture.transactions / "retry2"
        second_dir.mkdir()
        fixture._write_task(second)
        put_json(second_dir / "committed.json", {"schema": "ease-gpu-resource-activation-v1",
                 "plan_sha256": borrowed.remote.PLAN_SHA256, "task_sha256": {second["id"]: fixture.retry_sha}})
        # Replace expected hash with the actual retry2 descriptor hash.
        activation = json.loads((second_dir / "committed.json").read_text())
        activation["task_sha256"][second["id"]] = digest_bytes((fixture.tasks / (second["id"] + ".json")).read_bytes())
        put_json(second_dir / "committed.json", activation)
        fixture._write_outcome(second, "complete")
        fixture._write_marker(second)
        with self.assertRaises(RuntimeError):
            borrowed.campaign_completed(fixture.campaign)

    def test_borrowed_runtime_blocks_holder_and_scheduler_lifecycle_methods(self):
        guard = FakeHolderGuard()
        runtime = borrowed.BorrowedRuntime(guard, "GPU-test")
        self.assertTrue(runtime.keeper_live({"ready": guard.proof["keeper_ready"]}))
        guard.fail = True
        self.assertFalse(runtime.keeper_live({"ready": guard.proof["keeper_ready"]}))
        guard.fail = False
        for method in (runtime.start_keeper, runtime.close_keeper, runtime.bind_scheduler):
            with self.assertRaises(RuntimeError):
                method()
        self.assertEqual(guard.calls, ["GPU-test", "GPU-test"])

    def test_owner_loss_cleanup_only_touches_borrower_popen_handles(self):
        guard = FakeHolderGuard()
        session = borrowed.BorrowedSession(guard, "GPU-test", 49123, {"manifest_hash": "m"})
        child = FakeChild()
        holder_sentinel = FakeChild()
        session.runtime.owned_scientific_children = [child]
        guard.fail = True
        with self.assertRaisesRegex(RuntimeError, "holder identity lost"):
            session.check_keeper()
        session.runtime.reap_scientific_children()
        self.assertTrue(child.terminated)
        self.assertTrue(child.waited)
        self.assertFalse(holder_sentinel.terminated)
        self.assertFalse(holder_sentinel.waited)

    def test_optional_claim_skips_only_acquisition_conflict(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "claim.lock"
            with path.open("a") as held:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with borrowed.lock(path, optional=True) as acquired:
                    self.assertFalse(acquired)
            with self.assertRaises(BlockingIOError):
                with borrowed.lock(path, optional=True) as acquired:
                    self.assertTrue(acquired)
                    raise BlockingIOError("raised by work body")

    def test_broker_shared_borrowers_coexist_but_exclusive_publisher_is_excluded(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "broker.lock"
            with ExitStack() as borrowers:
                self.assertTrue(borrowers.enter_context(borrowed.lock(path, shared=True)))
                self.assertTrue(borrowers.enter_context(borrowed.lock(path, shared=True)))
                with self.assertRaises(BlockingIOError):
                    with borrowed.lock(path):
                        self.fail("exclusive publisher unexpectedly acquired a borrower-held lock")
            with path.open("a") as publisher:
                fcntl.flock(publisher.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    with borrowed.lock(path, shared=True):
                        self.fail("borrower unexpectedly passed a live exclusive publisher")

    def test_shared_broker_lock_opens_a_readable_fd(self):
        with tempfile.TemporaryDirectory() as raw_tmp:
            path = Path(raw_tmp) / "broker.lock"
            observed_access_modes = []
            original_flock = borrowed.fcntl.flock

            def capture_fd_mode(fd, operation):
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                observed_access_modes.append(flags & os.O_ACCMODE)
                return original_flock(fd, operation)

            with patch.object(borrowed.fcntl, "flock", new=capture_fd_mode):
                with borrowed.lock(path, shared=True) as acquired:
                    self.assertTrue(acquired)
            self.assertEqual(observed_access_modes, [os.O_RDWR])
            self.assertNotEqual(observed_access_modes[0], os.O_WRONLY)

    def _run_with_fake_science_modules(self, marker_value):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        new_root = Path(temp.name) / "new_science"
        fake_campaign = Path(temp.name) / "old_campaign"
        fake_campaign.mkdir()
        output = new_root / "discovery"
        condition = "local_goal"
        marker = output / condition / "discovery_shard00of01_complete.json"
        marker_raw = put_json(marker, marker_value)
        gpu_uuid = "GPU-test"
        holder = FakeHolderGuard()
        fake_common = types.SimpleNamespace(
            read_manifest=lambda path, verify_sources: {"manifest_hash": "m", "planned_conditions": [condition]},
            stage_for=lambda manifest: "discovery",
            cases=lambda manifest, stage: ["case0", "case1"])
        fake_runner = types.SimpleNamespace(run=lambda *args, **kwargs: self.fail("runner must not run for a completed marker"))
        fake_worker = types.SimpleNamespace()
        module_prefix = "preparations.ease_insights_20261103.vlm_within_suite_v1"
        fake_modules = {module_prefix + ".common": fake_common,
                        module_prefix + ".runner": fake_runner,
                        module_prefix + ".worker": fake_worker}
        package = sys.modules[module_prefix]
        args = types.SimpleNamespace(output=str(output), gpu_uuid=gpu_uuid, launch=True, check_holders=False)
        with ExitStack() as stack:
            stack.enter_context(patch.object(borrowed, "HolderGuard", return_value=holder))
            stack.enter_context(patch.object(borrowed, "NEW_ROOT", new_root))
            stack.enter_context(patch.object(borrowed, "CAMPAIGN", fake_campaign))
            stack.enter_context(patch.object(borrowed.legacy, "Science",
                                              side_effect=AssertionError("science must not be constructed")))
            for attribute, fake_module in (("common", fake_common), ("runner", fake_runner), ("worker", fake_worker)):
                stack.enter_context(patch.object(package, attribute, fake_module, create=True))
            stack.enter_context(patch.dict(sys.modules, fake_modules))
            result = borrowed.run(args)
        self.assertEqual(marker.read_bytes(), marker_raw)
        return result

    def test_valid_complete_marker_is_never_overwritten_or_rerun(self):
        result = self._run_with_fake_science_modules({"complete": True, "manifest_hash": "m",
            "condition": "local_goal", "completed_after": 2})
        self.assertTrue(result["results"][0]["unchanged"])

    def test_invalid_existing_marker_is_refused_without_overwrite(self):
        with self.assertRaisesRegex(RuntimeError, "Existing complete marker differs"):
            self._run_with_fake_science_modules({"complete": False, "manifest_hash": "m",
                "condition": "local_goal", "completed_after": 2})


if __name__ == "__main__":
    unittest.main()
