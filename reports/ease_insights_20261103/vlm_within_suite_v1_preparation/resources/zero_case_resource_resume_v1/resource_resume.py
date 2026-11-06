"""External CPU-only, fail-closed resume controller for zero-case model-init OOM.

This tool reads the pinned cached GPU watcher file. It never queries GPUs,
starts environments, changes science, or signals processes. A launch requires
the explicit ``run --approve-launch`` command plus the caller's exact shared
registry SHA and predecessor PID/start ticks. Importing this module is inert.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Callable, Mapping, Sequence
import uuid as uuidlib


ROOT = Path("/root/nfs/embodied/benchmark")
STAGE = ROOT / "results/ease_insights_20261103/vlm_within_suite_v1/discovery"
RESOURCE_DIR = ROOT / "reports/ease_insights_20261103/vlm_within_suite_v1_preparation/resources/zero_case_resource_resume_v1"
MANIFEST_PATH = STAGE / "manifest.json"
MANIFEST_HASH = "6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d"
MANIFEST_FILE_SHA256 = "e9dc20f236a74da7e10493488ea644f0050501159f0c2e390e77c8f2ab653ddb"
BORROWER_PATH = ROOT / "preparations/ease_insights_20261103/vlm_within_suite_v1/borrowed_controller.py"
BORROWER_SHA256 = "10b6a2cbb31e5cb7ebf654d985f0a215ae6d46a1564f865b6a76be87dc697e5b"
WATCH_SOURCE_PATH = ROOT / "preparations/ease_insights_20261103/gpu_release_watch.py"
WATCH_SOURCE_SHA256 = "4314b4b5d7eca4f88968b0d66da305a5cd2f6aff4829f363ae76b951b9495680"
WATCH_RECEIPT_PATH = ROOT / "results/ease_insights_20261103/gpu_release_watch/process_receipt.json"
WATCH_STATUS_PATH = ROOT / "results/ease_insights_20261103/gpu_release_watch/status.json"
WATCH_PID = 385842
WATCH_START_TICKS = "2618445988"
WATCH_RUN_ID = "060b0172-ba52-4c9b-9c39-03c7fb604d69"
POLL_SECONDS = 2.0
MAX_CACHE_AGE_SECONDS = 3.0
MAX_QUERY_GAP_SECONDS = 3.0
REQUIRED_DISTINCT_SAMPLES = 9
REQUIRED_AVAILABLE_SECONDS = 15.0
KEEPER_MEMORY_LIMIT_MIB = 512.0
MAX_AUTO_RESUMES = 2  # Resource-only retries; never changes per-case technical retry policy.
POST_EXIT_FOREIGN_SAMPLES = 3
POST_EXIT_MAX_POLLS = 12
MIN_POST_EXIT_SAMPLE_GAP_SECONDS = 1.0
MODEL_PYTHON = ROOT / "policies/realtime-vla-flash/.venv/bin/python"
EXPECTED_ARGV_PREFIX = (
    str(MODEL_PYTHON), "-B", "-u", "-m",
    "preparations.ease_insights_20261103.vlm_within_suite_v1.borrowed_controller",
    "--output", str(STAGE), "--gpu-uuid",
)
CONDITIONS = ("full_d10", "jac_anchor", "local_spatial", "local_object", "local_goal", "local_long")
OUTPUT_ROOT = ROOT / "results/ease_insights_20261103/vlm_within_suite_v1/discovery"
SHARED_PROCESS_REGISTRY = OUTPUT_ROOT / "controllers/controller_processes.json"

GPU_PINS = {
    "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1": {
        "slot": 0,
        "original_session_pid": 631907,
        "original_session_start_ticks": "2623408608",
        "original_keeper_pid": 632159,
        "original_keeper_start_ticks": "2623410211",
        "host_keeper_pid": 1975204,
    },
    "GPU-599477fb-0507-40dc-14df-d491f7a7b202": {
        "slot": 1,
        "original_session_pid": 631915,
        "original_session_start_ticks": "2623408624",
        "original_keeper_pid": 632160,
        "original_keeper_start_ticks": "2623410216",
        "host_keeper_pid": 1975205,
    },
}
SCHEDULER_PID = 600088
SCHEDULER_START_TICKS = "2622761021"


class ResumeError(RuntimeError):
    """A pinned resource condition failed; no launch should follow."""


class ManualReconcileRequired(ResumeError):
    """Durable launch intent or ambiguous child state forbids automatic retry."""


class AttentionRequired(ResumeError):
    """Failure did not meet the narrow zero-case model-init OOM contract."""


class StageAlreadyComplete(ResumeError):
    """All strict 72-case markers appeared before a borrower launch."""

    def __init__(self, markers: Mapping):
        super().__init__("All frozen condition markers became complete before launch")
        self.markers = dict(markers)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ResumeError(message)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fsync_directory(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def durable_exclusive_json(path: Path, value: Mapping) -> None:
    """Create a new journal file with O_EXCL and fsync file plus directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive name is deliberately retained if a partial intent was
        # written. A restart then fails closed for manual reconciliation.
        raise
    fsync_directory(path.parent)


def durable_replace_json(path: Path, value: Mapping) -> None:
    """Atomic durable JSON update, used only while holding the CPU RMW lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()) + "." + uuidlib.uuid4().hex)
    fd = os.open(str(temp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


@contextmanager
def cpu_file_lock(path: Path, *, nonblocking: bool = False):
    """NFS-safe CPU lock. This is never a GPU lease or holder lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        operation = fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)
        try:
            fcntl.flock(stream.fileno(), operation)
        except BlockingIOError as error:
            raise ResumeError("CPU singleton/registry lock is already held: " + str(path)) from error
        yield stream


def manifest_digest(manifest: Mapping) -> str:
    payload = dict(manifest)
    payload.pop("manifest_hash", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_pin(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else root / path


def _verify_pin_map(root: Path, pins: Mapping[str, str], label: str) -> dict:
    evidence = {}
    for name, expected in sorted(pins.items()):
        path = _resolve_pin(root, name)
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            raise ResumeError("Frozen %s changed: %s" % (label, path))
        evidence[str(path)] = actual
    return evidence


def verify_frozen_identity(root: Path = ROOT, stage_root: Path = STAGE) -> dict:
    """Verify manifest, frozen source maps, and small checkpoint identity pins.

    The large checkpoint weights are not reread: the saved original borrower
    and service protocol bind their checkpoint SHA. This check does hash all
    manifest-pinned source/evidence/normalization identity files.
    """
    root, stage_root = Path(root), Path(stage_root)
    manifest_path = stage_root / "manifest.json"
    raw_sha = sha256_file(manifest_path)
    if raw_sha != MANIFEST_FILE_SHA256:
        raise ResumeError("Frozen manifest file SHA changed")
    manifest = read_json(manifest_path)
    computed = manifest_digest(manifest)
    if manifest.get("manifest_hash") != MANIFEST_HASH or computed != MANIFEST_HASH:
        raise ResumeError("Frozen scientific manifest identity changed")
    if Path(manifest.get("output_root", "")).resolve() != stage_root.resolve():
        raise ResumeError("Manifest output root changed")

    evidence = {
        "source_files": _verify_pin_map(root, manifest.get("source_files", {}), "source_files"),
        "local_runtime_sources": _verify_pin_map(root, manifest.get("local_runtime_sources", {}), "local_runtime_sources"),
        "analysis_sources": _verify_pin_map(root, manifest.get("analysis_sources", {}), "analysis_sources"),
    }
    for key in ("protocol_evidence", "design_evidence", "candidate_evidence", "candidate_input_evidence"):
        row = manifest.get(key)
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise ResumeError("Missing frozen evidence pin: " + key)
        path = _resolve_pin(root, row["path"])
        actual = sha256_file(path) if path.is_file() else None
        if actual != row["sha256"]:
            raise ResumeError("Frozen evidence changed: " + key)
        evidence[key] = {"path": str(path), "sha256": actual}

    resource = manifest.get("resource_execution_revision", {}).get("evidence", {})
    resource_path = _resolve_pin(root, resource.get("path", ""))
    resource_sha = sha256_file(resource_path) if resource_path.is_file() else None
    if resource_sha != resource.get("sha256"):
        raise ResumeError("Frozen resource execution evidence changed")
    evidence["resource_execution_revision"] = {"path": str(resource_path), "sha256": resource_sha}

    checkpoint = manifest.get("checkpoints", {}).get("pi0", {})
    if not isinstance(checkpoint.get("sha256"), str) or len(checkpoint["sha256"]) != 64:
        raise ResumeError("Checkpoint SHA pin missing")
    evidence["checkpoint_identity_artifacts"] = _verify_pin_map(
        root, checkpoint.get("identity_artifacts", {}), "checkpoint identity artifacts")

    if sha256_file(BORROWER_PATH) != BORROWER_SHA256:
        raise ResumeError("Frozen borrowed_controller.py changed")
    if sha256_file(WATCH_SOURCE_PATH) != WATCH_SOURCE_SHA256:
        raise ResumeError("Pinned GPU watcher source changed")
    watch = read_json(WATCH_RECEIPT_PATH)
    if not all((watch.get("pid") == WATCH_PID,
                str(watch.get("proc_start_ticks")) == WATCH_START_TICKS,
                watch.get("run_id") == WATCH_RUN_ID,
                watch.get("source_path") == str(WATCH_SOURCE_PATH),
                watch.get("source_sha256") == WATCH_SOURCE_SHA256,
                watch.get("poll_seconds") == POLL_SECONDS,
                watch.get("mode") == "observe_only")):
        raise ResumeError("Cached watcher process receipt differs from the pinned daemon")
    evidence.update(manifest_hash=MANIFEST_HASH, manifest_file_sha256=raw_sha,
                    borrower_source_sha256=BORROWER_SHA256,
                    watcher_source_sha256=WATCH_SOURCE_SHA256)
    return {"manifest": manifest, "evidence": evidence}


def proc_record(pid: int, proc_root: Path = Path("/proc")) -> dict | None:
    try:
        text = (Path(proc_root) / str(int(pid)) / "stat").read_text()
    except FileNotFoundError:
        return None
    try:
        _, tail = text.rsplit(")", 1)
        fields = tail.split()
        return {
            "state": fields[0],
            "ppid": int(fields[1]),
            "pgrp": int(fields[2]),
            "session": int(fields[3]),
            "start_ticks": fields[19],
        }
    except (IndexError, ValueError) as error:
        raise ResumeError("Cannot parse /proc/%s/stat" % pid) from error


def proc_snapshot(proc_root: Path = Path("/proc")) -> dict[int, dict]:
    rows = {}
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        record = proc_record(int(entry.name), proc_root)
        if record is not None:
            rows[int(entry.name)] = record
    return rows


def assert_original_resources(proc_reader: Callable[[int], dict | None] = proc_record) -> None:
    pins = [(SCHEDULER_PID, SCHEDULER_START_TICKS, "original scheduler")]
    for gpu_uuid, item in GPU_PINS.items():
        pins.extend([
            (item["original_session_pid"], item["original_session_start_ticks"], "original session " + gpu_uuid),
            (item["original_keeper_pid"], item["original_keeper_start_ticks"], "original keeper " + gpu_uuid),
        ])
    for pid, expected_start, name in pins:
        actual = proc_reader(pid)
        if not actual or actual["state"] in ("Z", "X", "x") or actual["start_ticks"] != expected_start:
            raise ResumeError("Pinned %s process identity changed" % name)


def expected_argv(gpu_uuid: str, root: Path = ROOT, stage_root: Path = STAGE) -> list[str]:
    if gpu_uuid not in GPU_PINS:
        raise ResumeError("GPU UUID is outside the two fixed original devices")
    return [*EXPECTED_ARGV_PREFIX, gpu_uuid, "--launch"]


def registry_entry(registry: Mapping, gpu_uuid: str) -> dict:
    if registry.get("manifest_hash") != MANIFEST_HASH or registry.get("manifest_file_sha256") != MANIFEST_FILE_SHA256:
        raise ResumeError("Shared controller registry has a different manifest identity")
    matches = [row for row in registry.get("processes", []) if row.get("gpu_uuid") == gpu_uuid]
    if len(matches) != 1:
        raise ResumeError("Shared controller registry must contain exactly one row for the selected GPU")
    row = matches[0]
    argv = expected_argv(gpu_uuid)
    if row.get("argv") != argv or row.get("actual_argv") != argv:
        raise ResumeError("Shared registry no longer contains the frozen original borrower argv")
    if type(row.get("pid")) is not int or not str(row.get("start_ticks", "")):
        raise ResumeError("Shared registry predecessor PID/start identity is incomplete")
    return row


def assigned_registry_row_matches(row: Mapping, gpu_uuid: str, child: Mapping) -> bool:
    """Ignore peer-GPU edits; bind only the UUID row owned by this child."""
    argv = expected_argv(gpu_uuid)
    return all((
        row.get("gpu_uuid") == gpu_uuid,
        row.get("pid") == child.get("pid"),
        str(row.get("start_ticks")) == str(child.get("start_ticks")),
        row.get("argv") == argv,
        row.get("actual_argv") == argv,
        row.get("log") == child.get("log"),
        row.get("resource_resume_attempt") == child.get("attempt_id"),
    ))


def update_assigned_registry_after_exit(registry_path: Path, gpu_uuid: str,
                                       child: Mapping, returncode: int, ended_at: float,
                                       *, lock_path: Path) -> tuple[bool, str, dict | None, str | None]:
    """Update only our GPU row; concurrent peer-GPU registry changes are allowed."""
    with cpu_file_lock(lock_path):
        raw = Path(registry_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        try:
            registry = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False, digest, None, "shared_registry_invalid_json_at_exit"
        if not isinstance(registry, dict):
            return False, digest, None, "shared_registry_root_invalid_at_exit"
        if (registry.get("manifest_hash") != MANIFEST_HASH or
                registry.get("manifest_file_sha256") != MANIFEST_FILE_SHA256):
            return False, digest, None, "shared_registry_manifest_identity_changed_at_exit"
        rows = registry.get("processes")
        if not isinstance(rows, list):
            return False, digest, None, "shared_registry_process_rows_invalid_at_exit"
        if any(not isinstance(item, dict) for item in rows):
            return False, digest, None, "shared_registry_process_row_invalid_at_exit"
        matches = [item for item in rows if isinstance(item, dict) and item.get("gpu_uuid") == gpu_uuid]
        if len(matches) != 1:
            return False, digest, matches[0] if matches else None, "assigned_gpu_registry_row_missing_or_duplicated_at_exit"
        row = matches[0]
        if not assigned_registry_row_matches(row, gpu_uuid, child):
            return False, digest, row, "assigned_gpu_registry_identity_changed_at_exit"
        updated = dict(row, exit_code=returncode, ended_unix=ended_at,
                       state="exited", assignment_exit_recorded=True)
        registry["processes"] = [updated if item.get("gpu_uuid") == gpu_uuid else item
                                 for item in registry["processes"]]
        durable_replace_json(registry_path, registry)
        final_raw = Path(registry_path).read_bytes()
        return True, hashlib.sha256(final_raw).hexdigest(), updated, None


def assert_dead_predecessor(row: Mapping, expected_pid: int, expected_start: str,
                            proc_reader: Callable[[int], dict | None] = proc_record,
                            proc_root: Path = Path("/proc")) -> None:
    if row.get("pid") != expected_pid or str(row.get("start_ticks")) != str(expected_start):
        raise ResumeError("Shared registry predecessor PID/start differs from caller's explicit pin")
    actual = proc_reader(expected_pid)
    if actual is not None:
        if actual.get("start_ticks") != str(expected_start):
            raise ManualReconcileRequired("Predecessor PID was reused; manual registry reconciliation required")
        if actual.get("state") not in ("Z", "X", "x"):
            raise ResumeError("Predecessor borrower is still alive")
    # Frozen Science children inherit the borrower's session/process group.
    # A remaining group member means the previous model process has not been
    # reconciled, so never stack another model load on top of it.
    if any(p.get("session") == expected_pid or p.get("pgrp") == expected_pid
           for pid, p in proc_snapshot(proc_root).items() if pid != expected_pid):
        raise ManualReconcileRequired("Previous borrower's process group still has children")


def validate_attempt_journal_chain(*, attempt_dir: Path, gpu_uuid: str,
                                    intent: Mapping, child: Mapping,
                                    exit_receipt: Mapping, classification: Mapping,
                                    registry_receipt: Mapping, exit_registry: Mapping,
                                    previous_child: Mapping | None = None) -> None:
    """Cross-bind every durable receipt before a prior attempt can be resumed."""
    def check(value: bool, message: str) -> None:
        if not value:
            raise ManualReconcileRequired(message + ": " + str(attempt_dir))

    documents = (intent, child, exit_receipt, classification, registry_receipt, exit_registry)
    check(all(isinstance(item, Mapping) for item in documents), "Attempt journal document is not an object")
    attempt_id = Path(attempt_dir).name
    argv = expected_argv(gpu_uuid)
    expected_log = str(Path(attempt_dir) / "borrowed_controller.log")
    predecessor = intent.get("predecessor")
    child_key = {"pid": child.get("pid"), "start_ticks": str(child.get("start_ticks"))}

    check(intent.get("schema") == "zero-case-resource-resume-intent-v1", "Attempt intent schema mismatch")
    check(attempt_id.isdigit(), "Attempt directory ID is malformed")
    check(intent.get("attempt_id") == attempt_id, "Attempt intent ID differs from directory")
    check(child.get("schema") == "zero-case-resource-resume-child-v1", "Child receipt schema mismatch")
    for name, document in (("child", child), ("exit", exit_receipt),
                           ("classification", classification), ("registry", registry_receipt),
                           ("exit registry", exit_registry)):
        check(document.get("attempt_id") == attempt_id, "%s receipt attempt ID mismatch" % name)
    check(classification.get("gpu_uuid") == gpu_uuid and
          classification.get("manifest_hash") == MANIFEST_HASH and
          classification.get("borrower_argv") == argv,
          "Classification GPU, manifest, or fixed argv mismatch")
    check(isinstance(predecessor, Mapping) and classification.get("predecessor") == dict(predecessor),
          "Classification predecessor differs from intent")
    check(classification.get("child_pid") == child.get("pid") and
          str(classification.get("child_start_ticks")) == str(child.get("start_ticks")),
          "Classification child identity differs from child receipt")

    check(intent.get("gpu_uuid") == gpu_uuid and child.get("gpu_uuid") == gpu_uuid,
          "Attempt GPU UUID mismatch")
    check(intent.get("manifest_hash") == MANIFEST_HASH and
          intent.get("manifest_file_sha256") == MANIFEST_FILE_SHA256 and
          child.get("manifest_hash") == MANIFEST_HASH,
          "Attempt manifest identity mismatch")
    check(intent.get("argv") == argv and child.get("argv") == argv,
          "Attempt fixed borrower argv mismatch")
    check(child.get("log") == expected_log, "Child log path differs from attempt directory")
    check(type(child.get("pid")) is int and child.get("pid") > 0 and
          re.fullmatch(r"[0-9]+", str(child.get("start_ticks", ""))) is not None,
          "Child PID/start identity is incomplete")

    check(isinstance(predecessor, Mapping) and type(predecessor.get("pid")) is int and
          predecessor.get("pid") > 0 and
          re.fullmatch(r"[0-9]+", str(predecessor.get("start_ticks", ""))) is not None,
          "Intent predecessor identity is incomplete")
    check(registry_receipt.get("predecessor") == dict(predecessor),
          "Intent and launch registry predecessor disagree")
    check(registry_receipt.get("shared_registry_before_sha256") ==
          intent.get("shared_registry_sha256_before"),
          "Intent and launch registry preimage SHA disagree")
    for label, digest in (
            ("intent registry preimage", intent.get("shared_registry_sha256_before")),
            ("launch registry postimage", registry_receipt.get("shared_registry_after_sha256")),
            ("exit registry postimage", exit_registry.get("shared_registry_after_exit_sha256"))):
        check(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
              "%s SHA is malformed" % label)
    if previous_child is not None:
        expected_predecessor = {"pid": previous_child.get("pid"),
                                "start_ticks": str(previous_child.get("start_ticks"))}
        check(dict(predecessor) == expected_predecessor,
              "Attempt predecessor does not match the previous completed child")

    check(registry_receipt.get("registry_updated") is True and
          registry_receipt.get("child") == child_key,
          "Launch registry receipt is not bound to the child")
    check(exit_receipt.get("pid") == child.get("pid") and
          str(exit_receipt.get("start_ticks")) == str(child.get("start_ticks")),
          "Exit receipt PID/start differs from child receipt")
    check(type(exit_receipt.get("returncode")) is int,
          "Exit receipt return code is malformed")
    check(exit_registry.get("registry_updated") is True and
          exit_registry.get("child") == child_key,
          "Exit registry receipt is not bound to the child")
    assigned_row = exit_registry.get("assigned_row")
    check(isinstance(assigned_row, Mapping) and
          assigned_registry_row_matches(assigned_row, gpu_uuid, child),
          "Exit registry assigned row differs from child identity")
    check(assigned_row.get("state") == "exited" and
          assigned_row.get("assignment_exit_recorded") is True and
          assigned_row.get("exit_code") == exit_receipt.get("returncode"),
          "Exit registry row differs from exit receipt")
    check(classification.get("assigned_registry_row_after_exit") == assigned_row,
          "Classification assigned row differs from exit registry receipt")

    check(classification.get("shared_registry_sha256_after_launch") ==
          registry_receipt.get("shared_registry_after_sha256"),
          "Classification launch-registry SHA differs from launch receipt")
    check(classification.get("shared_registry_sha256_after_exit") ==
          exit_registry.get("shared_registry_after_exit_sha256"),
          "Classification exit-registry SHA differs from exit receipt")
    check(classification.get("child_receipt_sha256") == sha256_file(Path(attempt_dir) / "child_receipt.json"),
          "Classification child-receipt SHA mismatch")
    check(classification.get("exit_receipt_sha256") == sha256_file(Path(attempt_dir) / "exit.json"),
          "Classification exit-receipt SHA mismatch")
    check(classification.get("exit_registry_receipt_sha256") ==
          sha256_file(Path(attempt_dir) / "exit_registry_receipt.json"),
          "Classification exit-registry-receipt SHA mismatch")

    if classification.get("controller_pid") is not None:
        check(classification.get("controller_pid") == child.get("pid"),
              "Classification controller PID differs from child")
    if classification.get("controller_start_ticks") is not None:
        check(str(classification.get("controller_start_ticks")) == str(child.get("start_ticks")),
              "Classification controller start differs from child")


def build_cached_status_reader(status_path: Path = WATCH_STATUS_PATH) -> Callable[[], dict]:
    return lambda: read_json(status_path)


def cached_device(status: Mapping, gpu_uuid: str, now: float,
                  *, max_age: float = MAX_CACHE_AGE_SECONDS) -> tuple[bool, dict]:
    if gpu_uuid not in GPU_PINS:
        return False, {}
    query = status.get("query_finished_at")
    try:
        age = float(now) - float(query)
    except (TypeError, ValueError):
        return False, {}
    try:
        poll_seconds = float(status.get("poll_seconds", -1))
    except (TypeError, ValueError):
        return False, {}
    daemon_ok = all((status.get("pid") == WATCH_PID,
                     str(status.get("proc_start_ticks")) == WATCH_START_TICKS,
                     status.get("run_id") == WATCH_RUN_ID,
                     status.get("source_sha256") == WATCH_SOURCE_SHA256,
                     status.get("error") is None,
                     poll_seconds == POLL_SECONDS,
                     0.0 <= age <= max_age))
    if not daemon_ok:
        return False, {}
    device_rows = [row for row in status.get("devices", []) if row.get("gpu_uuid") == gpu_uuid]
    if len(device_rows) != 1:
        return False, {}
    return True, device_rows[0]


@dataclass
class AvailabilityWindow:
    gpu_uuid: str
    first_query: float | None = None
    previous_query: float | None = None
    samples: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.first_query = None
        self.previous_query = None
        self.samples.clear()

    def consume(self, status: Mapping, now: float) -> bool:
        """Consume one cached row; return whether the continuous window is ready."""
        fresh, device = cached_device(status, self.gpu_uuid, now)
        keeper = GPU_PINS.get(self.gpu_uuid, {}).get("host_keeper_pid")
        memory = device.get("memory_mib", 0) if fresh else 0
        try:
            query = float(status.get("query_finished_at"))
        except (TypeError, ValueError):
            query = None
        available = bool(fresh and keeper is not None and
            device.get("compute_process_count") == 1 and
            device.get("host_compute_pids") == [keeper] and
            isinstance(memory, (int, float)) and 0 < float(memory) <= KEEPER_MEMORY_LIMIT_MIB)

        gap_bad = (query is not None and self.previous_query is not None and
                   (query < self.previous_query or query - self.previous_query > MAX_QUERY_GAP_SECONDS))
        if not available or gap_bad:
            self.reset()
        if available and query is not None and query != self.previous_query:
            if self.first_query is None:
                self.first_query = query
            self.samples.append({
                "query_finished_at": query,
                "memory_mib": float(memory),
                "compute_process_count": device["compute_process_count"],
                "host_compute_pids": list(device["host_compute_pids"]),
            })
            self.previous_query = query
        return self.ready

    @property
    def elapsed(self) -> float:
        if self.first_query is None or self.previous_query is None:
            return 0.0
        return self.previous_query - self.first_query

    @property
    def ready(self) -> bool:
        return (len(self.samples) >= REQUIRED_DISTINCT_SAMPLES and
                self.elapsed >= REQUIRED_AVAILABLE_SECONDS)


def wait_for_owner_window(gpu_uuid: str, status_reader: Callable[[], Mapping], *,
                          clock: Callable[[], float] = time.time,
                          sleep: Callable[[float], None] = time.sleep,
                          on_sample: Callable[[dict], None] | None = None) -> list[dict]:
    if gpu_uuid not in GPU_PINS:
        raise ResumeError("GPU UUID is outside the two fixed original devices")
    window = AvailabilityWindow(gpu_uuid)
    while True:
        now = clock()
        status = status_reader()
        ready = window.consume(status, now)
        if on_sample is not None:
            on_sample({"time": now, "ready": ready, "samples": len(window.samples),
                       "available_seconds": window.elapsed})
        if ready:
            return list(window.samples)
        sleep(POLL_SECONDS)


def snapshot_complete_markers(stage_root: Path = STAGE) -> dict:
    """Read only the six condition-level discovery completion markers."""
    result = {}
    for condition in CONDITIONS:
        path = Path(stage_root) / condition / "discovery_shard00of01_complete.json"
        if not path.is_file():
            result[condition] = None
            continue
        marker = read_json(path)
        count = marker.get("completed_after", marker.get("already_completed"))
        if not all((marker.get("complete") is True,
                    marker.get("manifest_hash") == MANIFEST_HASH,
                    marker.get("condition") == condition,
                    count == 72)):
            raise ResumeError("Existing completion marker is not the frozen 72-case marker: " + str(path))
        result[condition] = {"path": str(path), "sha256": sha256_file(path), "cases": count}
    return result


def assert_complete_markers_unchanged(before: Mapping, after: Mapping) -> None:
    """Preserve every accepted marker while allowing valid peer additions."""
    for condition, marker in before.items():
        if marker is not None and after.get(condition) != marker:
            raise AttentionRequired("Existing accepted completion marker changed: " + condition)


def assert_prior_progress_preserved(before_markers: Mapping, after_markers: Mapping,
                                    before_runs: Mapping[str, set[str]],
                                    after_runs: Mapping[str, set[str]]) -> None:
    assert_complete_markers_unchanged(before_markers, after_markers)
    for condition, prior_names in before_runs.items():
        if not set(prior_names).issubset(after_runs.get(condition, set())):
            raise AttentionRequired("A preexisting service run directory was deleted: " + condition)


def snapshot_run_directories(stage_root: Path = STAGE) -> dict[str, set[str]]:
    """List only immediate service run directory names; never traverse episodes."""
    snapshot = {}
    for condition in CONDITIONS:
        runs = Path(stage_root) / condition / "runs"
        snapshot[condition] = {entry.name for entry in runs.iterdir() if entry.is_dir()} if runs.is_dir() else set()
    return snapshot


def _argv_value(argv: Sequence[str], option: str) -> str | None:
    try:
        index = list(argv).index(option)
        return str(argv[index + 1])
    except (ValueError, IndexError):
        return None


def run_identity_is_bound(identity: Mapping, *, stage_root: Path, run_dir: Path,
                          condition: str, gpu_uuid: str, controller_pid: int) -> bool:
    """Check the frozen service identity fields without relying on top-level stage."""
    if not isinstance(identity, Mapping):
        return False
    server_command = identity.get("server_command", [])
    worker_command = identity.get("worker_command", [])
    if not isinstance(server_command, list) or not isinstance(worker_command, list):
        return False
    technical_revision = identity.get("technical_runtime_revision")
    resource_revision = identity.get("resource_execution_revision")
    if not isinstance(technical_revision, Mapping) or not isinstance(resource_revision, Mapping):
        return False
    expected_server_module = "preparations.ease_insights_20261103.vlm_within_suite_v1.server"
    expected_worker_module = "preparations.ease_insights_20261103.vlm_within_suite_v1.worker"
    return all((
        identity.get("manifest_hash") == MANIFEST_HASH,
        identity.get("condition") == condition,
        identity.get("controller_pid") == controller_pid,
        identity.get("stateful_workers_per_server") == 1,
        technical_revision.get("scientific_manifest_hash") == MANIFEST_HASH,
        technical_revision.get("phase") == "within_suite_discovery_v1",
        resource_revision.get("scientific_manifest_hash") == MANIFEST_HASH,
        resource_revision.get("controller_pid") == controller_pid,
        resource_revision.get("controller_source_sha256") == BORROWER_SHA256,
        _argv_value(server_command, "--condition") == condition,
        _argv_value(server_command, "--gpu-uuid") == gpu_uuid,
        _argv_value(server_command, "--manifest") == str(Path(stage_root) / "manifest.json"),
        _argv_value(server_command, "--output") == str(run_dir),
        expected_server_module in server_command,
        _argv_value(worker_command, "--condition") == condition,
        _argv_value(worker_command, "--stage") == "discovery",
        _argv_value(worker_command, "--output") == str(stage_root),
        expected_worker_module in worker_command,
    ))


def validate_added_run_directories(stage_root: Path, *, before_runs: Mapping[str, set[str]],
                                   after_runs: Mapping[str, set[str]], gpu_uuid: str,
                                   own_controller_pid: int | None,
                                   peer_pids: Mapping[str, int]) -> list[tuple[str, str, Path]]:
    """Bind newly added runs to this child or the registry-pinned peer borrower."""
    own_runs = []
    for condition in CONDITIONS:
        added = set(after_runs.get(condition, set())) - set(before_runs.get(condition, set()))
        for name in sorted(added):
            run_dir = Path(stage_root) / condition / "runs" / name
            if not name.startswith("discovery_shard00of01_") or not run_dir.is_dir():
                raise AttentionRequired("New service run name or directory is not pinned: " + str(run_dir))
            identity_path = run_dir / "identity.json"
            if not identity_path.is_file():
                raise AttentionRequired("New service run has no identity.json: " + str(run_dir))
            try:
                identity = read_json(identity_path)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise AttentionRequired("New service run identity is unreadable: " + str(run_dir)) from error
            if not isinstance(identity, dict):
                raise AttentionRequired("New service run identity is not an object: " + str(run_dir))
            controller_pid = identity.get("controller_pid")
            server_command = identity.get("server_command", [])
            server_gpu_uuid = (_argv_value(server_command, "--gpu-uuid")
                               if isinstance(server_command, list) else None)
            if own_controller_pid is not None and controller_pid == own_controller_pid:
                if server_gpu_uuid != gpu_uuid or not run_identity_is_bound(
                        identity, stage_root=stage_root, run_dir=run_dir,
                        condition=condition, gpu_uuid=gpu_uuid,
                        controller_pid=own_controller_pid):
                    raise AttentionRequired("New run from our controller has a mismatched service identity: " + str(run_dir))
                own_runs.append((condition, name, run_dir))
                continue
            if server_gpu_uuid not in peer_pids or peer_pids.get(server_gpu_uuid) != controller_pid:
                raise AttentionRequired("New service run is not bound to the registry-pinned peer borrower: " + str(run_dir))
            if not run_identity_is_bound(
                    identity, stage_root=stage_root, run_dir=run_dir,
                    condition=condition, gpu_uuid=server_gpu_uuid,
                    controller_pid=int(controller_pid)):
                raise AttentionRequired("New peer run has a mismatched frozen service identity: " + str(run_dir))
    return own_runs


def peer_controller_pids(registry: Mapping, selected_gpu_uuid: str) -> dict[str, int]:
    """Read only resource identity rows needed to bind concurrent peer runs."""
    if (registry.get("manifest_hash") != MANIFEST_HASH or
            registry.get("manifest_file_sha256") != MANIFEST_FILE_SHA256):
        raise ResumeError("Shared controller registry has a different manifest identity")
    result = {}
    for gpu_uuid in GPU_PINS:
        if gpu_uuid == selected_gpu_uuid:
            continue
        row = registry_entry(registry, gpu_uuid)
        result[gpu_uuid] = row["pid"]
    return result


def read_peer_controller_pids(registry_path: Path, selected_gpu_uuid: str) -> dict[str, int]:
    return peer_controller_pids(read_json(registry_path), selected_gpu_uuid)


def extra_foreign_compute_after_exit(status: Mapping, gpu_uuid: str, exit_time: float,
                                    own_pids: set[int], now: float) -> dict | None:
    fresh, device = cached_device(status, gpu_uuid, now)
    if not fresh:
        return None
    try:
        query = float(status["query_finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    keeper = GPU_PINS[gpu_uuid]["host_keeper_pid"]
    host_pids = device.get("host_compute_pids")
    count = device.get("compute_process_count")
    if (query <= exit_time or not isinstance(host_pids, list) or keeper not in host_pids or
            not isinstance(count, int) or count <= 1):
        return None
    try:
        numeric_pids = {int(pid) for pid in host_pids}
    except (TypeError, ValueError):
        return None
    foreign = sorted(pid for pid in numeric_pids if pid != keeper and pid not in own_pids)
    if not foreign:
        return None
    return {"query_finished_at": query, "compute_process_count": count,
            "host_compute_pids": list(host_pids), "foreign_compute_pids_unattributed": foreign,
            "memory_mib": device.get("memory_mib"), "gpu_uuid": gpu_uuid}


def persistent_extra_foreign_compute(status_observations: Sequence[tuple[Mapping, float]],
                                     gpu_uuid: str, exit_time: float,
                                     own_pids: set[int]) -> dict | None:
    """Require three distinct consecutive post-exit cached polls sharing a PID."""
    samples = []
    common_foreign: set[int] | None = None
    previous_query = None
    for status, now in status_observations:
        row = extra_foreign_compute_after_exit(status, gpu_uuid, exit_time, own_pids, now)
        query = status.get("query_finished_at")
        try:
            query = float(query)
        except (TypeError, ValueError):
            row = None
        if row is None:
            samples.clear()
            common_foreign = None
            previous_query = None
            continue
        if previous_query is not None:
            gap = query - previous_query
            if gap == 0:
                continue
            if (gap < MIN_POST_EXIT_SAMPLE_GAP_SECONDS or
                    gap > MAX_QUERY_GAP_SECONDS):
                samples.clear()
                common_foreign = None
                previous_query = None
        pids = set(row["foreign_compute_pids_unattributed"])
        common_foreign = pids if common_foreign is None else common_foreign & pids
        if not common_foreign:
            samples.clear()
            common_foreign = pids
        samples.append(row)
        previous_query = query
        if len(samples) >= POST_EXIT_FOREIGN_SAMPLES and common_foreign:
            return {
                "observations": samples[-POST_EXIT_FOREIGN_SAMPLES:],
                "shared_unattributed_foreign_pids": sorted(common_foreign),
                "required_distinct_samples": POST_EXIT_FOREIGN_SAMPLES,
            }
    return None


def wait_for_post_exit_foreign(status_reader: Callable[[], Mapping], gpu_uuid: str,
                               exit_time: float, own_pids: set[int], *,
                               clock: Callable[[], float] = time.time,
                               sleep: Callable[[float], None] = time.sleep,
                               max_polls: int = POST_EXIT_MAX_POLLS) -> list[tuple[Mapping, float]]:
    """Read only cached status until 3 distinct post-exit extra-compute samples."""
    observations = []
    for _ in range(max_polls):
        now = clock()
        status = status_reader()
        observations.append((status, now))
        if persistent_extra_foreign_compute(observations, gpu_uuid, exit_time, own_pids):
            return observations
        if len(observations) < max_polls:
            sleep(POLL_SECONDS)
    return observations


def classify_zero_case_startup_oom(*, stage_root: Path, gpu_uuid: str, child_pid: int,
                                   child_start_ticks: str, child_returncode: int,
                                   exit_time: float, own_group_members: Sequence[Mapping],
                                   before_runs: Mapping[str, set[str]], after_runs: Mapping[str, set[str]],
                                   complete_before: Mapping, complete_after: Mapping,
                                   post_exit_statuses: Sequence[tuple[Mapping, float]], now: float,
                                   assigned_registry_identity_ok: bool = True,
                                   peer_pids: Mapping[str, int] | None = None) -> dict:
    """Accept only a newly-created, pre-health Pi0 OOM with no worker/cases."""
    def attention(reason: str, **details) -> dict:
        return {"classification": "attention", "retryable": False, "reason": reason, **details}

    if gpu_uuid not in GPU_PINS:
        return attention("unrecognized_gpu")
    if not assigned_registry_identity_ok:
        return attention("assigned_gpu_registry_identity_changed")
    try:
        assert_prior_progress_preserved(complete_before, complete_after, before_runs, after_runs)
        own_new_dirs = validate_added_run_directories(
            stage_root, before_runs=before_runs, after_runs=after_runs,
            gpu_uuid=gpu_uuid, own_controller_pid=child_pid,
            peer_pids=peer_pids or {})
    except AttentionRequired as error:
        return attention("peer_or_preexisting_progress_not_preserved_or_bound", detail=str(error))
    if child_returncode == 0:
        return {"classification": "assignment_complete", "retryable": False,
                "reason": "borrower_exit_zero"}
    if child_returncode is None:
        return attention("missing_child_exit_receipt")
    if own_group_members:
        return attention("own_process_group_has_remaining_children",
                         process_group_members=list(own_group_members))
    if len(own_new_dirs) != 1:
        return attention("expected_exactly_one_new_run_from_own_controller",
                         own_new_run_count=len(own_new_dirs))
    condition, run_name, run_dir = own_new_dirs[0]
    if condition not in CONDITIONS or not run_name.startswith("discovery_shard00of01_"):
        return attention("new_run_name_or_condition_not_pinned", run=str(run_dir))
    if complete_before.get(condition) is not None:
        return attention("borrower_attempted_already_completed_condition", condition=condition)
    if not run_dir.is_dir():
        return attention("new_run_directory_disappeared", run=str(run_dir))
    entries = {path.name for path in run_dir.iterdir()}
    if entries != {"identity.json", "server.log"}:
        return attention("startup_run_not_exact_zero_case_artifact_set", run=str(run_dir),
                         entries=sorted(entries))
    identity_path = run_dir / "identity.json"
    server_log = run_dir / "server.log"
    try:
        identity = read_json(identity_path)
    except (OSError, ValueError, json.JSONDecodeError):
        identity = {}
    if not run_identity_is_bound(identity, stage_root=stage_root, run_dir=run_dir,
                                 condition=condition, gpu_uuid=gpu_uuid,
                                 controller_pid=child_pid):
        return attention("new_run_identity_not_bound_to_own_controller", run=str(run_dir),
                         identity_sha256=sha256_file(identity_path))

    if server_log.stat().st_size > 4 * 1024 * 1024:
        return attention("server_log_exceeds_bounded_classifier_size", run=str(run_dir),
                         server_log_size_bytes=server_log.stat().st_size)
    text = server_log.read_bytes().decode("utf-8", errors="replace")
    lines = text.splitlines()
    trace_starts = [i for i, line in enumerate(lines) if line.strip() == "Traceback (most recent call last):"]
    if not trace_starts:
        return attention("log_has_no_pi0_initialization_traceback", run=str(run_dir),
                         server_log_sha256=sha256_file(server_log))
    trace = lines[trace_starts[-1]:]
    exception_line = "RuntimeError: CUDA error: out of memory"
    exception_indexes = [i for i, line in enumerate(trace) if line.strip() == exception_line]
    exception_headers = [line.strip() for line in trace
                         if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*:\s*", line.strip())]
    if not exception_indexes:
        return attention("last_traceback_not_exact_pre_health_pi0_inference_oom", run=str(run_dir),
                         server_log_sha256=sha256_file(server_log),
                         last_exception=next((line.strip() for line in reversed(trace) if line.strip()), ""),
                         pi0_constructor_call=False, pi0_infer_init_frame=False,
                         health_or_progression_log_token=False)
    oom_index = exception_indexes[-1]
    suffix = [line.strip() for line in trace[oom_index + 1:] if line.strip()]
    recognized_warning_tail = [
        "CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.",
        "For debugging consider passing CUDA_LAUNCH_BLOCKING=1",
        "Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.",
    ]
    warning_tail_ok = (not suffix or
                       (len(suffix) <= len(recognized_warning_tail) and
                        suffix == recognized_warning_tail[:len(suffix)]))
    final_exception = trace[oom_index].strip()
    only_exact_exception = exception_headers == [exception_line]
    trace_text = "\n".join(trace[:oom_index + 1])
    pi0_call = "Pi0Inference(" in trace_text or "Pi0Inference.__init__" in trace_text
    pi0_init_frame = re.search(r"File [\"'].*pi0_infer\.py[\"'], line \d+, in __init__", trace_text) is not None
    exact_oom = final_exception == exception_line and only_exact_exception and warning_tail_ok
    progression_tokens = any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in (
        r"\bREADY\b", r"\bhealth\b", r"\bprotocol\b", r"\bworker\b",
        r"\bepisodes?\b", r"episode[_ -]receipt", r"\bcomplete(?:\.json)?\b"))
    if not exact_oom or not pi0_call or not pi0_init_frame or progression_tokens:
        return attention("last_traceback_not_exact_pre_health_pi0_inference_oom", run=str(run_dir),
                         server_log_sha256=sha256_file(server_log),
                         last_exception=final_exception, pi0_constructor_call=pi0_call,
                         pi0_infer_init_frame=pi0_init_frame,
                         health_or_progression_log_token=progression_tokens,
                         recognized_warning_tail=warning_tail_ok,
                         traceback_exception_headers=exception_headers,
                         traceback_suffix=suffix)

    foreign = persistent_extra_foreign_compute(
        post_exit_statuses, gpu_uuid, exit_time,
        {child_pid, *(int(row["pid"]) for row in own_group_members)})
    if foreign is None:
        return attention("no_persistent_extra_foreign_compute_after_child_exit", run=str(run_dir),
                         server_log_sha256=sha256_file(server_log),
                         required_distinct_samples=POST_EXIT_FOREIGN_SAMPLES)

    return {
        "classification": "retryable_zero_case_pi0_init_oom_with_foreign_compute",
        "retryable": True,
        "condition": condition,
        "run_path": str(run_dir),
        "run_files": sorted(entries),
        "controller_pid": child_pid,
        "controller_start_ticks": child_start_ticks,
        "identity_sha256": sha256_file(identity_path),
        "server_log_sha256": sha256_file(server_log),
        "protocol_worker_or_case_artifacts_present": False,
        "remaining_own_model_children": 0,
        "post_exit_foreign_compute": foreign,
        "scientific_outcome_or_success_read": False,
    }


def process_group_members(child_pid: int, proc_root: Path = Path("/proc")) -> list[dict]:
    snapshot = proc_snapshot(proc_root)
    return [dict(pid=pid, **row) for pid, row in snapshot.items()
            if pid != child_pid and (row.get("session") == child_pid or row.get("pgrp") == child_pid)]


def _write_attempt_status(resource_dir: Path, value: Mapping) -> None:
    durable_replace_json(Path(resource_dir) / "status.json", value)


def validate_predecessor(gpu_uuid: str, registry_path: Path, expected_registry_sha256: str,
                         expected_old_pid: int, expected_old_start_ticks: str,
                         *, proc_reader: Callable[[int], dict | None] = proc_record,
                         proc_root: Path = Path("/proc")) -> tuple[dict, str]:
    raw = Path(registry_path).read_bytes()
    actual_registry_sha = hashlib.sha256(raw).hexdigest()
    if actual_registry_sha != expected_registry_sha256:
        raise ManualReconcileRequired("Shared controller registry SHA differs from explicit review pin")
    registry = json.loads(raw)
    entry = registry_entry(registry, gpu_uuid)
    assert_dead_predecessor(entry, expected_old_pid, expected_old_start_ticks, proc_reader, proc_root)
    return registry, actual_registry_sha


class ResourceResumeController:
    """Synchronous external supervisor; all dependencies are injectable for CPU tests."""
    def __init__(self, *, root: Path = ROOT, stage_root: Path = STAGE,
                 resource_dir: Path = RESOURCE_DIR, proc_root: Path = Path("/proc"),
                 status_reader: Callable[[], Mapping] | None = None,
                 process_reader: Callable[[int], dict | None] | None = None,
                 snapshot_reader: Callable[[], dict] | None = None,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep,
                 max_auto_resumes: int = MAX_AUTO_RESUMES):
        self.root = Path(root)
        self.stage_root = Path(stage_root)
        self.resource_dir = Path(resource_dir)
        self.proc_root = Path(proc_root)
        self.status_reader = status_reader or build_cached_status_reader(
            self.root / "results/ease_insights_20261103/gpu_release_watch/status.json")
        self.process_reader = process_reader or (lambda pid: proc_record(pid, self.proc_root))
        self.snapshot_reader = snapshot_reader or (lambda: proc_snapshot(self.proc_root))
        self.popen = popen
        self.clock = clock
        self.sleep = sleep
        self.max_auto_resumes = max_auto_resumes

    @property
    def shared_registry_path(self) -> Path:
        return self.stage_root / "controllers/controller_processes.json"

    def _verify_before_wait(self) -> dict:
        identity = verify_frozen_identity(self.root, self.stage_root)
        assert_original_resources(self.process_reader)
        watcher = self.process_reader(WATCH_PID)
        if (not watcher or watcher.get("state") in ("Z", "X", "x") or
                watcher.get("start_ticks") != WATCH_START_TICKS):
            raise ResumeError("Pinned GPU watcher process identity changed")
        return identity

    def _preflight_journal(self, gpu_uuid: str) -> tuple[list[dict], dict | None]:
        attempts_root = self.resource_dir / "attempts" / gpu_uuid
        if not attempts_root.exists():
            return [], None
        prior = []
        terminal = None
        attempt_dirs = sorted((p for p in attempts_root.iterdir() if p.is_dir()), key=lambda p: p.name)
        previous_child = None
        for index, attempt_dir in enumerate(attempt_dirs):
            intent = attempt_dir / "intent.json"
            if not intent.is_file():
                raise ManualReconcileRequired("Attempt directory has no durable intent: " + str(attempt_dir))
            child_receipt = attempt_dir / "child_receipt.json"
            exit_receipt = attempt_dir / "exit.json"
            classification = attempt_dir / "classification.json"
            registry_receipt = attempt_dir / "registry_receipt.json"
            exit_registry_receipt = attempt_dir / "exit_registry_receipt.json"
            if (not child_receipt.is_file() or not exit_receipt.is_file() or
                    not classification.is_file() or not registry_receipt.is_file() or
                    not exit_registry_receipt.is_file()):
                raise ManualReconcileRequired("Pending attempt intent requires manual reconciliation: " + str(attempt_dir))
            try:
                row = {"intent": read_json(intent), "child": read_json(child_receipt),
                       "exit": read_json(exit_receipt), "classification": read_json(classification),
                       "registry": read_json(registry_receipt),
                       "exit_registry": read_json(exit_registry_receipt),
                       "path": str(attempt_dir)}
            except (OSError, ValueError, TypeError) as error:
                raise ManualReconcileRequired("Attempt journal receipt is unreadable: " + str(attempt_dir)) from error
            validate_attempt_journal_chain(
                attempt_dir=attempt_dir, gpu_uuid=gpu_uuid,
                intent=row["intent"], child=row["child"], exit_receipt=row["exit"],
                classification=row["classification"], registry_receipt=row["registry"],
                exit_registry=row["exit_registry"], previous_child=previous_child)
            cls = row["classification"].get("classification")
            if cls == "retryable_zero_case_pi0_init_oom_with_foreign_compute":
                if (row["classification"].get("retryable") is not True or
                        row["exit"].get("returncode") in (None, 0)):
                    raise ManualReconcileRequired("Safe OOM classification lacks a nonzero child exit")
                self._verify_preserved_failure(row["classification"])
                prior.append(row)
                terminal = row
                previous_child = row["child"]
                continue
            if (cls == "assignment_complete" and
                    row["classification"].get("retryable") is False and
                    row["exit"].get("returncode") == 0):
                if index != len(attempt_dirs) - 1:
                    raise ManualReconcileRequired("Attempt directory exists after completed assignment")
                terminal = row
                break
            raise ManualReconcileRequired("Prior attempt is not an approved safe retry: " + str(attempt_dir))
        return prior, terminal

    @staticmethod
    def _verify_preserved_failure(classification: Mapping) -> None:
        run_dir = Path(classification["run_path"])
        if {path.name for path in run_dir.iterdir()} != {"identity.json", "server.log"}:
            raise ManualReconcileRequired("Preserved zero-case failure run changed")
        for name, key in (("identity.json", "identity_sha256"), ("server.log", "server_log_sha256")):
            if sha256_file(run_dir / name) != classification.get(key):
                raise ManualReconcileRequired("Preserved failure artifact changed: " + name)

    def _wait_window(self, gpu_uuid: str, attempt_id: str) -> list[dict]:
        window = AvailabilityWindow(gpu_uuid)
        while True:
            assert_original_resources(self.process_reader)
            status = self.status_reader()
            now = self.clock()
            ready = window.consume(status, now)
            _write_attempt_status(self.resource_dir / ("status_" + gpu_uuid + ".json"), {
                "phase": "waiting_for_original_keeper_window",
                "time": now,
                "attempt_id": attempt_id,
                "gpu_uuid": gpu_uuid,
                "samples": len(window.samples),
                "available_seconds": window.elapsed,
                "policy_seconds": [15, 2],
                "cache_age_limit_seconds": MAX_CACHE_AGE_SECONDS,
                "no_gpu_query": True,
            })
            if ready:
                return list(window.samples)
            self.sleep(POLL_SECONDS)

    def _start_attempt(self, *, gpu_uuid: str, registry: dict, registry_sha: str,
                       expected_old_pid: int, expected_old_start: str,
                       samples: list[dict], complete_markers: dict,
                       before_runs: dict[str, set[str]], attempt_dir: Path,
                       resource_registry_path: Path) -> tuple[subprocess.Popen, dict, str, object]:
        argv = expected_argv(gpu_uuid, self.root, self.stage_root)
        old_row = registry_entry(registry, gpu_uuid)
        assert_dead_predecessor(old_row, expected_old_pid, expected_old_start,
                                self.process_reader, self.proc_root)
        attempt_id = attempt_dir.name
        intent = {
            "schema": "zero-case-resource-resume-intent-v1",
            "attempt_id": attempt_id,
            "time": self.clock(),
            "gpu_uuid": gpu_uuid,
            "manifest_hash": MANIFEST_HASH,
            "manifest_file_sha256": MANIFEST_FILE_SHA256,
            "argv": argv,
            "predecessor": {"pid": expected_old_pid, "start_ticks": str(expected_old_start)},
            "shared_registry_sha256_before": registry_sha,
            "availability_samples": samples,
            "completed_condition_markers": complete_markers,
            "run_directories_before": {k: sorted(v) for k, v in before_runs.items()},
            "retry_eligibility": "only exact zero-case pre-health Pi0 initialization OOM with no own model children and post-exit extra foreign compute",
            "no_gpu_query": True,
            "no_signals": True,
        }
        durable_exclusive_json(attempt_dir / "intent.json", intent)
        log_path = attempt_dir / "borrowed_controller.log"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        handle = os.fdopen(fd, "ab", buffering=0)
        try:
            child = self.popen(argv, cwd=self.root, stdin=subprocess.DEVNULL, stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        except BaseException as error:
            handle.close()
            raise ManualReconcileRequired("Popen failed after durable intent; manual reconcile required") from error

        try:
            actual = self.process_reader(child.pid)
            if not actual or actual.get("state") in ("Z", "X", "x"):
                raise ManualReconcileRequired("Child receipt could not bind a live Popen PID; durable intent retained")
            receipt = {
                "schema": "zero-case-resource-resume-child-v1",
                "attempt_id": attempt_id,
                "time": self.clock(),
                "pid": child.pid,
                "start_ticks": actual["start_ticks"],
                "session": actual.get("session", child.pid),
                "pgrp": actual.get("pgrp", child.pid),
                "argv": argv,
                "log": str(log_path),
                "gpu_uuid": gpu_uuid,
                "manifest_hash": MANIFEST_HASH,
            }
            current_raw = self.shared_registry_path.read_bytes()
            current_sha = hashlib.sha256(current_raw).hexdigest()
            if current_sha != registry_sha:
                raise ManualReconcileRequired("Shared registry changed after intent; manual reconcile required")
            current_registry = json.loads(current_raw)
            current_row = registry_entry(current_registry, gpu_uuid)
            if (current_row.get("pid") != expected_old_pid or
                    str(current_row.get("start_ticks")) != str(expected_old_start)):
                raise ManualReconcileRequired("Shared registry predecessor changed after intent")
            # Create receipt before updating shared registry. A crash in either
            # direction leaves a pending journal that refuses duplicate launch.
            durable_exclusive_json(attempt_dir / "child_receipt.json", receipt)
            new_row = dict(current_row)
            for key in ("exit_code", "ended_unix", "assignment_exit_recorded"):
                new_row.pop(key, None)
            new_row.update(pid=child.pid, start_ticks=actual["start_ticks"],
                state=actual["state"], log=str(log_path), started_unix=self.clock(),
                recovery="foreign_oom_zero_case_resource_resume_v1",
                resource_resume_attempt=attempt_id,
                owner_available_seconds=samples[-1]["query_finished_at"] - samples[0]["query_finished_at"],
                available_query_start=samples[0]["query_finished_at"],
                available_query_end=samples[-1]["query_finished_at"])
            current_registry["processes"] = [new_row if row.get("gpu_uuid") == gpu_uuid else row
                                              for row in current_registry["processes"]]
            current_registry["time"] = self.clock()
            durable_replace_json(self.shared_registry_path, current_registry)
            shared_after = hashlib.sha256(self.shared_registry_path.read_bytes()).hexdigest()
            local = read_json(resource_registry_path)
            local.setdefault("gpu_states", {})[gpu_uuid] = {
                "attempt_id": attempt_id,
                "child": receipt,
                "shared_registry_sha256": shared_after,
                "phase": "borrower_running",
            }
            local["updated_at"] = self.clock()
            durable_replace_json(resource_registry_path, local)
            durable_exclusive_json(attempt_dir / "registry_receipt.json", {
                "attempt_id": attempt_id,
                "shared_registry_before_sha256": registry_sha,
                "shared_registry_after_sha256": shared_after,
                "predecessor": {"pid": expected_old_pid, "start_ticks": str(expected_old_start)},
                "child": {"pid": child.pid, "start_ticks": actual["start_ticks"]},
                "registry_updated": True,
            })
            return child, receipt, shared_after, handle
        except BaseException as error:
            handle.close()
            if isinstance(error, ManualReconcileRequired):
                raise
            raise ManualReconcileRequired("Popen child exists but durable child/registry receipt failed") from error

    def _run_attempt(self, *, gpu_uuid: str, samples: list[dict], markers_before: dict,
                     runs_before: dict[str, set[str]], expected_registry_sha: str | None,
                     expected_assigned_row: Mapping,
                     old_pid: int, old_start: str,
                     resource_registry_path: Path) -> tuple[int, dict, str, dict, dict[str, set[str]]]:
        identity = verify_frozen_identity(self.root, self.stage_root)
        if identity["manifest"]["manifest_hash"] != MANIFEST_HASH:
            raise ResumeError("Frozen science identity changed")
        assert_original_resources(self.process_reader)

        # One global CPU RMW lock serializes registry updates across both GPUs.
        # The exact predecessor is checked under that lock immediately before
        # O_EXCL intent creation and Popen; no GPU lease is opened or changed.
        with cpu_file_lock(self.resource_dir / "registry_rmw.lock"):
            current_raw = self.shared_registry_path.read_bytes()
            registry_sha = hashlib.sha256(current_raw).hexdigest()
            if expected_registry_sha is not None and registry_sha != expected_registry_sha:
                raise ManualReconcileRequired("Shared registry SHA changed before launch")
            registry = json.loads(current_raw)
            row = registry_entry(registry, gpu_uuid)
            if dict(row) != dict(expected_assigned_row):
                raise ManualReconcileRequired("Assigned GPU registry row changed before launch")
            assert_dead_predecessor(row, old_pid, old_start, self.process_reader, self.proc_root)
            current_markers = snapshot_complete_markers(self.stage_root)
            current_runs = snapshot_run_directories(self.stage_root)
            peer_pids = peer_controller_pids(registry, gpu_uuid)
            try:
                assert_prior_progress_preserved(markers_before, current_markers,
                                                runs_before, current_runs)
                validate_added_run_directories(
                    self.stage_root, before_runs=runs_before, after_runs=current_runs,
                    gpu_uuid=gpu_uuid, own_controller_pid=None, peer_pids=peer_pids)
            except AttentionRequired as error:
                raise ManualReconcileRequired("Peer or accepted progress changed at launch boundary: " + str(error)) from error
            markers_before, runs_before = current_markers, current_runs
            if all(marker is not None for marker in markers_before.values()):
                raise StageAlreadyComplete(markers_before)
            attempt_id = str(self.clock_ns())
            attempt_dir = self.resource_dir / "attempts" / gpu_uuid / attempt_id
            try:
                attempt_dir.mkdir(parents=True, exist_ok=False)
                fsync_directory(attempt_dir.parent)
            except FileExistsError as error:
                raise ManualReconcileRequired("Attempt ID already exists; refusing duplicate") from error
            child, child_receipt, registry_after_sha, log_handle = self._start_attempt(
                gpu_uuid=gpu_uuid, registry=registry, registry_sha=registry_sha,
                expected_old_pid=old_pid, expected_old_start=old_start,
                samples=samples, complete_markers=markers_before, before_runs=runs_before,
                attempt_dir=attempt_dir, resource_registry_path=resource_registry_path)
        try:
            returncode = child.wait()
        finally:
            log_handle.close()
        exit_time = self.clock()
        exit_row = {"attempt_id": attempt_id, "time": exit_time, "returncode": returncode,
                    "pid": child_receipt["pid"], "start_ticks": child_receipt["start_ticks"]}
        durable_exclusive_json(attempt_dir / "exit.json", exit_row)
        group_members = process_group_members(child_receipt["pid"], self.proc_root)
        after_runs = snapshot_run_directories(self.stage_root)
        markers_after = snapshot_complete_markers(self.stage_root)
        assigned_identity_ok, registry_exit_sha, row_after_exit, registry_exit_error = \
            update_assigned_registry_after_exit(
                self.shared_registry_path, gpu_uuid, child_receipt, returncode, exit_time,
                lock_path=self.resource_dir / "registry_rmw.lock")
        exit_registry_receipt = {
            "attempt_id": attempt_id,
            "time": self.clock(),
            "registry_updated": assigned_identity_ok,
            "reason": registry_exit_error,
            "shared_registry_after_exit_sha256": registry_exit_sha,
            "child": {"pid": child_receipt["pid"], "start_ticks": child_receipt["start_ticks"]},
            "assigned_row": row_after_exit,
            "peer_gpu_row_changes_allowed": True,
        }
        durable_exclusive_json(attempt_dir / "exit_registry_receipt.json", exit_registry_receipt)
        try:
            current_peer_pids = read_peer_controller_pids(self.shared_registry_path, gpu_uuid)
        except (OSError, ValueError, ResumeError):
            current_peer_pids = {}
        post_exit_statuses: list[tuple[Mapping, float]] = []
        classification = classify_zero_case_startup_oom(
            stage_root=self.stage_root, gpu_uuid=gpu_uuid, child_pid=child_receipt["pid"],
            child_start_ticks=child_receipt["start_ticks"], child_returncode=returncode,
            exit_time=exit_time, own_group_members=group_members,
            before_runs=runs_before, after_runs=after_runs,
            complete_before=markers_before, complete_after=markers_after,
            post_exit_statuses=post_exit_statuses, now=self.clock(),
            assigned_registry_identity_ok=assigned_identity_ok,
            peer_pids=current_peer_pids)
        if classification.get("reason") == "no_persistent_extra_foreign_compute_after_child_exit":
            post_exit_statuses = wait_for_post_exit_foreign(
                self.status_reader, gpu_uuid, exit_time,
                {child_receipt["pid"], *(int(row["pid"]) for row in group_members)},
                clock=self.clock, sleep=self.sleep)
            # A peer can finish a condition or start another service while we
            # collect the three post-exit polls; classify against that current
            # snapshot, while preserving launch-time accepted artifacts.
            after_runs = snapshot_run_directories(self.stage_root)
            markers_after = snapshot_complete_markers(self.stage_root)
            try:
                current_peer_pids = read_peer_controller_pids(self.shared_registry_path, gpu_uuid)
            except (OSError, ValueError, ResumeError):
                current_peer_pids = {}
            classification = classify_zero_case_startup_oom(
                stage_root=self.stage_root, gpu_uuid=gpu_uuid, child_pid=child_receipt["pid"],
                child_start_ticks=child_receipt["start_ticks"], child_returncode=returncode,
                exit_time=exit_time, own_group_members=group_members,
                before_runs=runs_before, after_runs=after_runs,
                complete_before=markers_before, complete_after=markers_after,
                post_exit_statuses=post_exit_statuses, now=self.clock(),
                assigned_registry_identity_ok=assigned_identity_ok,
                peer_pids=current_peer_pids)
        classification.update({"attempt_id": attempt_id, "gpu_uuid": gpu_uuid,
                               "manifest_hash": MANIFEST_HASH,
                               "borrower_argv": expected_argv(gpu_uuid),
                               "predecessor": {"pid": old_pid, "start_ticks": str(old_start)},
                               "child_pid": child_receipt["pid"],
                               "child_start_ticks": str(child_receipt["start_ticks"]),
                               "shared_registry_sha256_after_launch": registry_after_sha,
                               "shared_registry_sha256_after_exit": registry_exit_sha,
                               "assigned_registry_exit_update_error": registry_exit_error,
                               "assigned_registry_row_after_exit": row_after_exit,
                               "child_receipt_sha256": sha256_file(attempt_dir / "child_receipt.json"),
                               "exit_receipt_sha256": sha256_file(attempt_dir / "exit.json"),
                               "exit_registry_receipt_sha256": sha256_file(attempt_dir / "exit_registry_receipt.json")})
        durable_exclusive_json(attempt_dir / "classification.json", classification)
        with cpu_file_lock(self.resource_dir / "registry_rmw.lock"):
            local = read_json(resource_registry_path)
            local.setdefault("gpu_states", {})[gpu_uuid] = {
                "attempt_id": attempt_id,
                "child": child_receipt,
                "shared_registry_sha256": registry_exit_sha,
                "phase": classification["classification"],
            }
            local["updated_at"] = self.clock()
            durable_replace_json(resource_registry_path, local)
        return returncode, classification, registry_exit_sha, markers_after, after_runs

    def clock_ns(self) -> int:
        return time.time_ns()

    def run(self, *, gpu_uuid: str, expected_registry_sha256: str,
            expected_old_pid: int, expected_old_start_ticks: str,
            approve_launch: bool) -> dict:
        if not approve_launch:
            raise ResumeError("Explicit --approve-launch is required; this is a launch-capable tool")
        if gpu_uuid not in GPU_PINS:
            raise ResumeError("GPU UUID is outside the two fixed original devices")
        if not re.fullmatch(r"[0-9a-f]{64}", expected_registry_sha256):
            raise ResumeError("An exact controller registry SHA256 review pin is required")
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        singleton_path = self.resource_dir / ("singleton_" + gpu_uuid + ".lock")
        with cpu_file_lock(singleton_path, nonblocking=True):
            self._verify_before_wait()
            prior, terminal = self._preflight_journal(gpu_uuid)
            retry_count = len(prior)
            if retry_count > self.max_auto_resumes:
                raise AttentionRequired("Resource-only automatic resume cap is exhausted")
            resource_registry_path = self.resource_dir / "registry.json"
            with cpu_file_lock(self.resource_dir / "registry_rmw.lock"):
                if resource_registry_path.exists():
                    resource_registry = read_json(resource_registry_path)
                else:
                    resource_registry = {"schema": "zero-case-resource-resume-registry-v1", "gpu_states": {}}
                    durable_exclusive_json(resource_registry_path, resource_registry)
            if terminal:
                child = terminal["child"]
                classification = terminal["classification"]
                if (int(child.get("pid", -1)) != expected_old_pid or
                        str(child.get("start_ticks")) != str(expected_old_start_ticks) or
                        classification.get("shared_registry_sha256_after_exit") !=
                        terminal["exit_registry"].get("shared_registry_after_exit_sha256")):
                    raise ManualReconcileRequired("Caller pins do not match the last durable safe attempt")
                state = resource_registry.get("gpu_states", {}).get(gpu_uuid, {})
                if (state.get("attempt_id") != child.get("attempt_id") or
                        state.get("phase") != classification.get("classification") or
                        state.get("shared_registry_sha256") !=
                        classification.get("shared_registry_sha256_after_exit")):
                    raise ManualReconcileRequired("Local resource registry is not reconciled with safe attempt")
            initial_registry, actual_sha = validate_predecessor(
                gpu_uuid, self.shared_registry_path, expected_registry_sha256,
                expected_old_pid, expected_old_start_ticks, self.process_reader, self.proc_root)
            if actual_sha != expected_registry_sha256:
                raise ManualReconcileRequired("Review-pinned registry changed before wait")
            if terminal and terminal["classification"].get("classification") == "assignment_complete":
                return terminal["classification"]
            # The exact whole-file SHA is an invocation-time review pin. Once
            # accepted, launches validate this GPU's exact predecessor row
            # under the global RMW lock, allowing harmless peer-row updates.
            expected_registry_sha256 = None
            expected_assigned_row = registry_entry(initial_registry, gpu_uuid)
            if terminal and expected_assigned_row != terminal["exit_registry"].get("assigned_row"):
                raise ManualReconcileRequired("Caller-reviewed assigned GPU row differs from last durable receipt")
            markers_before = snapshot_complete_markers(self.stage_root)
            if all(marker is not None for marker in markers_before.values()):
                result = {"phase": "stage_already_complete", "no_launch": True,
                          "completed_markers": markers_before, "manifest_hash": MANIFEST_HASH}
                with cpu_file_lock(self.resource_dir / "registry_rmw.lock"):
                    durable_replace_json(self.resource_dir / ("status_" + gpu_uuid + ".json"), result)
                return result
            runs_before = snapshot_run_directories(self.stage_root)
            self._verify_before_wait()
            while True:
                samples = self._wait_window(gpu_uuid, str(self.clock_ns()))
                # After every wait window, reread source pins, original holders,
                # completion markers and run directories. The launch then
                # validates only the assigned GPU's registry row under RMW.
                self._verify_before_wait()
                current_markers = snapshot_complete_markers(self.stage_root)
                current_runs = snapshot_run_directories(self.stage_root)
                current_peer_pids = read_peer_controller_pids(self.shared_registry_path, gpu_uuid)
                try:
                    assert_prior_progress_preserved(markers_before, current_markers,
                                                    runs_before, current_runs)
                    validate_added_run_directories(
                        self.stage_root, before_runs=runs_before, after_runs=current_runs,
                        gpu_uuid=gpu_uuid, own_controller_pid=None,
                        peer_pids=current_peer_pids)
                except AttentionRequired as error:
                    raise ManualReconcileRequired("Peer or accepted progress changed during wait: " + str(error)) from error
                markers_before, runs_before = current_markers, current_runs
                if all(marker is not None for marker in markers_before.values()):
                    result = {"phase": "stage_already_complete", "no_launch": True,
                              "completed_markers": markers_before, "manifest_hash": MANIFEST_HASH}
                    with cpu_file_lock(self.resource_dir / "registry_rmw.lock"):
                        durable_replace_json(self.resource_dir / ("status_" + gpu_uuid + ".json"), result)
                    return result
                try:
                    returncode, classification, registry_after, accepted_markers_after, accepted_runs_after = self._run_attempt(
                        gpu_uuid=gpu_uuid, samples=samples, markers_before=markers_before,
                        runs_before=runs_before, expected_registry_sha=expected_registry_sha256,
                        expected_assigned_row=expected_assigned_row,
                        old_pid=expected_old_pid, old_start=expected_old_start_ticks,
                        resource_registry_path=resource_registry_path)
                except StageAlreadyComplete as complete:
                    result = {"phase": "stage_already_complete", "no_launch": True,
                              "completed_markers": complete.markers, "manifest_hash": MANIFEST_HASH}
                    with cpu_file_lock(self.resource_dir / "registry_rmw.lock"):
                        durable_replace_json(self.resource_dir / ("status_" + gpu_uuid + ".json"), result)
                    return result
                if classification.get("classification") == "assignment_complete" and returncode == 0:
                    return classification
                if (classification.get("classification") != "retryable_zero_case_pi0_init_oom_with_foreign_compute" or
                        not classification.get("retryable")):
                    raise AttentionRequired("Borrower failure requires attention: " + classification.get("reason", "unknown"))
                retry_count += 1
                if retry_count > self.max_auto_resumes:
                    raise AttentionRequired("Safe zero-case OOM resume cap reached; prior failed run preserved")
                self._verify_preserved_failure(classification)
                markers_before, runs_before = accepted_markers_after, accepted_runs_after
                expected_assigned_row = classification.get("assigned_registry_row_after_exit")
                if not isinstance(expected_assigned_row, dict):
                    raise ManualReconcileRequired("Safe failure receipt lacks the assigned GPU exit row")
                expected_old_pid = int(expected_assigned_row["pid"])
                expected_old_start_ticks = str(expected_assigned_row["start_ticks"])
                # Caller SHA is a one-time review pin. After this controller's
                # first durable assignment, peer-GPU row updates are allowed;
                # each later launch still validates our exact row under RMW.
                expected_registry_sha256 = None
                # The next loop starts a fresh cache window and can only launch
                # after another complete 15s/9-sample owner-only interval.
                runs_before = snapshot_run_directories(self.stage_root)
                markers_before = snapshot_complete_markers(self.stage_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="wait on cached watcher data and resume the pinned borrower")
    run.add_argument("--gpu-uuid", required=True, choices=tuple(GPU_PINS))
    run.add_argument("--expected-registry-sha256", required=True)
    run.add_argument("--expected-old-pid", required=True, type=int)
    run.add_argument("--expected-old-start-ticks", required=True)
    run.add_argument("--approve-launch", required=True, action="store_true",
                     help="Explicit launch authorization; never implied by --help or preflight")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        controller = ResourceResumeController()
        result = controller.run(gpu_uuid=args.gpu_uuid,
            expected_registry_sha256=args.expected_registry_sha256,
            expected_old_pid=args.expected_old_pid,
            expected_old_start_ticks=args.expected_old_start_ticks,
            approve_launch=args.approve_launch)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
