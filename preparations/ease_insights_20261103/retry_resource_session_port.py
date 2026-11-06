#!/usr/bin/env python3
"""One-shot, CPU-only retry preparation for the two pinned v3 sessions.

Default execution is a read-only dry run. ``--apply`` appends at most one
``__port_retry1`` task per exact port-attention outcome, commits the matching
activation, then atomically publishes identity-bound Session resume requests.
It never signals or starts a process, queries a GPU, writes scheduler state,
or changes scientific source, plan, case, marker, status, or outcome files.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from typing import Callable, Mapping


PROJECT = Path("/root/nfs/embodied/benchmark")
EXPERIMENT = PROJECT / "results/ease_insights_20261103"
CAMPAIGN = EXPERIMENT / "gpu_resource_campaign_v3"
PLAN_PATH = CAMPAIGN / "plan.json"
HISTORY_NAME = "technical_port_retry_1793856599463613971"
HISTORY = CAMPAIGN / "transactions" / HISTORY_NAME
OLD_REQUEST_ID = "port_reuse_ready_1793856599466824667"
REPORT_DIR = PROJECT / "reports/ease_insights_20261103/gpu_continuous_handoff_20261105/port_retry_helper"
WATCH_DIR = EXPERIMENT / "gpu_release_watch"
PLAN_SHA256 = "18f3609856c026342bd79064a59a8a97f9e5ebdb9d8eb06240b2f8ab05fb5e42"
HISTORY_SHA256 = {
    "preparation.json": "ba2a28b14101d486d97e9fab2eadc267d36b92c9f93c917e14f2c7266afc987a",
    "published.json": "6730a54ce02b2c82c80807ffbcca87b1de8b438400264a3bde34efd000cc8d90",
    "committed.json": "dd4a57b44fe39bda7323e7e70b8ef95f4dfb4331a7ebe5021f093cbab95de8b7",
    "resumes/slot00/" + OLD_REQUEST_ID + ".json": "06c847835a8dd9c3a511b746442e57bb8556c2254865977643e7089e4003d78f",
    "resumes/slot01/" + OLD_REQUEST_ID + ".json": "36ed421ee006e82618bfe35bc74be7b6e7a916f4aa33e43430ff6724ffebc3d3",
    "sessions/slot00/resume_receipts/" + OLD_REQUEST_ID + ".json": "f89b0767c83b76c7e3522aab00b5e1109843ae9aa9495603667a6d54ed1deac6",
    "sessions/slot01/resume_receipts/" + OLD_REQUEST_ID + ".json": "89f6c565043cfb0c6faf3907000b656937340286cd665d7f098c503f9862684f",
}
PINNED_SOURCE_NAMES = (
    "preparations/ease_insights_20261103/await_assigned_gpu.py",
    "preparations/ease_insights_20261103/gpu_owned_keeper.py",
    "preparations/ease_insights_20261103/gpu_resource_campaign.py",
    "preparations/ease_insights_20261103/gpu_resource_session.py",
    "preparations/ease_insights_20261103/reassign_waiting_gate.py",
    "preparations/ease_insights_20261103/reassign_waiting_gate_v2.py",
    "preparations/ease_insights_20261103/scheduler.py",
    "preparations/ease_insights_20261103/vlm_local_search_v1/runner.py",
)
PORT_ERROR = "OSError(98, 'Address already in use')"
PLAN_SCHEMA = "ease-gpu-resource-campaign-v1"
TASK_SCHEMA = "ease-gpu-resource-task-v1"
ACTIVATION_SCHEMA = "ease-gpu-resource-activation-v1"
RESUME_SCHEMA = "ease-gpu-resource-resume-v1"
RESUME_FIELDS = frozenset((
    "schema", "action", "plan_sha256", "session_pid", "proc_start_ticks",
    "gpu_uuid", "attention_sha256", "reason",
))
TASK_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,200}")
WATCH_FRESH_SECONDS = 6.0
MAX_KEEPER_MIB = 512.0


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value, *, indent=2) -> bytes:
    return (json.dumps(value, indent=indent, ensure_ascii=False, allow_nan=False) + "\n").encode()


def attention_sha256(value) -> str:
    # Session.validate_resume uses this exact serializer (including defaults).
    return sha256(json.dumps(value, sort_keys=True).encode())


class ValidationError(RuntimeError):
    def __init__(self, message: str, status: str = "refused"):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SlotSpec:
    name: str
    gpu_uuid: str
    port: int
    session_pid: int
    session_start: str
    session_parent: int
    session_argv: tuple[str, ...]
    keeper_pid: int
    keeper_start: str
    nonce: str
    keeper_argv: tuple[str, ...]
    scheduler_start: str = "2622761021"
    scheduler_argv: tuple[str, ...] = ()


def _session_argv(gpu_uuid: str, port: int, slot: str) -> tuple[str, ...]:
    return (
        "/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.venv/bin/python",
        "-B", "-u", "-m", "preparations.ease_insights_20261103.gpu_resource_session",
        "--gpu-uuid", gpu_uuid, "--port", str(port), "--plan", str(PLAN_PATH),
        "--output", str(CAMPAIGN / "sessions" / slot), "--launch",
    )


def _scheduler_argv() -> tuple[str, ...]:
    return (
        "/usr/bin/python3", "-u", "-m", "preparations.ease_insights_20261103.scheduler",
        "--queue", str(EXPERIMENT / "queue.json"), "--output", str(EXPERIMENT / "scheduler"),
        "--max-gpus", "2", "--scan-seconds", "2", "--idle-seconds", "0",
    )


DEFAULT_SLOTS = (
    SlotSpec(
        "slot00", "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1", 47757,
        631907, "2623408608", 600088,
        _session_argv("GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1", 47757, "slot00"),
        632159, "2623410211", "9faaecd8f9e440158abee6f9c95d4e62",
        ("/usr/bin/python3", "-m", "preparations.ease_insights_20261103.gpu_owned_keeper",
         "--gpu-uuid", "GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1", "--control-fd", "6",
         "--ready-fd", "9", "--nonce", "9faaecd8f9e440158abee6f9c95d4e62",
         "--parent-pid", "631907", "--parent-start-ticks", "2623408608", "--allocation-mib", "64"),
        scheduler_argv=_scheduler_argv(),
    ),
    SlotSpec(
        "slot01", "GPU-599477fb-0507-40dc-14df-d491f7a7b202", 44105,
        631915, "2623408624", 600088,
        _session_argv("GPU-599477fb-0507-40dc-14df-d491f7a7b202", 44105, "slot01"),
        632160, "2623410216", "55c894da4b734358ad245ced9b042d68",
        ("/usr/bin/python3", "-m", "preparations.ease_insights_20261103.gpu_owned_keeper",
         "--gpu-uuid", "GPU-599477fb-0507-40dc-14df-d491f7a7b202", "--control-fd", "5",
         "--ready-fd", "9", "--nonce", "55c894da4b734358ad245ced9b042d68",
         "--parent-pid", "631915", "--parent-start-ticks", "2623408624", "--allocation-mib", "64"),
        scheduler_argv=_scheduler_argv(),
    ),
)


@dataclass(frozen=True)
class Baseline:
    project: Path = PROJECT
    experiment: Path = EXPERIMENT
    campaign: Path = CAMPAIGN
    report_dir: Path = REPORT_DIR
    watch_dir: Path = WATCH_DIR
    proc_root: Path = Path("/proc")
    expected_plan_sha256: str = PLAN_SHA256
    history_hashes: Mapping[str, str] = None
    slots: tuple[SlotSpec, ...] = DEFAULT_SLOTS

    def __post_init__(self):
        if self.history_hashes is None:
            object.__setattr__(self, "history_hashes", HISTORY_SHA256)


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_bytes())
    except (OSError, ValueError, TypeError) as error:
        raise ValidationError("Cannot read canonical JSON %s: %s" % (path, error)) from error


def canonical(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path:
        raise ValidationError("Noncanonical absolute path: " + str(path))
    return path


def _uid(proc_dir: Path) -> int | None:
    try:
        for line in (proc_dir / "status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def proc_identity(pid: int, proc_root: Path = Path("/proc")) -> dict:
    """Read local /proc identity; never maps watcher host PIDs to local PIDs."""
    directory = Path(proc_root) / str(pid)
    try:
        before = (directory / "stat").read_text()
        argv = [part.decode(errors="replace") for part in (directory / "cmdline").read_bytes().split(b"\0") if part]
        after = (directory / "stat").read_text()
        first = before.rsplit(")", 1)[1].split()
        last = after.rsplit(")", 1)[1].split()
        if int(before.split(" ", 1)[0]) != pid or int(after.split(" ", 1)[0]) != pid or first[19] != last[19]:
            raise ValidationError("Process identity changed while reading PID %s" % pid)
        state, parent, start = last[0], int(last[1]), last[19]
        if state in ("Z", "X", "x") or not argv:
            raise ValidationError("Pinned PID %s is not a live process" % pid)
        return {"pid": pid, "start_ticks": str(start), "ppid": parent, "state": state,
                "argv": argv, "uid": _uid(directory)}
    except ValidationError:
        raise
    except (OSError, ValueError, IndexError) as error:
        raise ValidationError("Cannot prove live PID %s: %s" % (pid, error)) from error


def proc_children(pid: int, proc_root: Path = Path("/proc")) -> list[int]:
    path = Path(proc_root) / str(pid) / "task" / str(pid) / "children"
    try:
        raw = path.read_text().strip()
    except OSError as error:
        raise ValidationError("Cannot read direct children of PID %s: %s" % (pid, error)) from error
    if not raw:
        return []
    try:
        children = [int(value) for value in raw.split()]
    except ValueError as error:
        raise ValidationError("Invalid direct-child list for PID %s" % pid) from error
    if len(children) != len(set(children)) or any(value <= 0 for value in children):
        raise ValidationError("Duplicate or invalid direct child for PID %s" % pid)
    return sorted(children)


def plain_bind_ready(port: int) -> bool:
    """Match Session.reserve_port: an ordinary loopback bind, with no reuse flags."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", int(port)))
        return True
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            return False
        raise ValidationError("Plain bind probe failed unexpectedly on port %s: %r" % (port, error)) from error


def _identity_matches(actual: dict, *, pid: int, start: str, ppid: int, argv: tuple[str, ...], label: str):
    expected = {"pid": int(pid), "start_ticks": str(start), "ppid": int(ppid), "argv": list(argv), "uid": 0}
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValidationError("%s identity differs at %s: expected %r, found %r" % (label, key, value, actual.get(key)))


class PortRetryController:
    def __init__(
        self,
        baseline: Baseline = Baseline(),
        *,
        bind_probe: Callable[[int], bool] = plain_bind_ready,
        clock: Callable[[], float] = time.time,
        clock_ns: Callable[[], int] = time.time_ns,
    ):
        self.b = baseline
        self.bind_probe = bind_probe
        self.clock = clock
        self.clock_ns = clock_ns

    def _helper_sha(self) -> str:
        return sha256(Path(__file__).read_bytes())

    def _verify_plan(self) -> tuple[dict, str]:
        path = canonical(self.b.campaign / "plan.json")
        raw = path.read_bytes()
        digest = sha256(raw)
        if digest != self.b.expected_plan_sha256:
            raise ValidationError("Pinned v3 plan hash changed")
        plan = json.loads(raw)
        if plan.get("schema") != PLAN_SCHEMA or plan.get("campaign_root") != str(self.b.campaign):
            raise ValidationError("Campaign plan schema/root differs")
        hashes = plan.get("source_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(PINNED_SOURCE_NAMES):
            raise ValidationError("The exact eight source pins are required")
        for name, expected in hashes.items():
            source = canonical(self.b.project / name)
            if sha256(source.read_bytes()) != expected:
                raise ValidationError("Pinned campaign source changed: " + name)
        expected_outputs = [
            str(self.b.experiment / "vlm_local_search_v1" / "discovery"),
            str(self.b.experiment / "vlm_local_search_v1" / "confirmation"),
        ]
        if plan.get("allowed_outputs") != expected_outputs or plan.get("task_schema") != TASK_SCHEMA:
            raise ValidationError("The fixed scientific output/task scope differs")
        return plan, digest

    def _verify_history(self, plan_sha: str) -> dict:
        for relative, expected in self.b.history_hashes.items():
            path = canonical(self.b.campaign / relative) if relative.startswith(("resumes/", "sessions/")) else canonical(self.b.campaign / "transactions" / HISTORY_NAME / relative)
            if sha256(path.read_bytes()) != expected:
                raise ValidationError("Pinned successful port-retry evidence changed: " + relative)
        history_dir = self.b.campaign / "transactions" / HISTORY_NAME
        preparation = read_json(history_dir / "preparation.json")
        published = read_json(history_dir / "published.json")
        activation = read_json(history_dir / "committed.json")
        if preparation.get("schema") != "ease-explicit-technical-port-retry-v1" or preparation.get("phase") != "prepared":
            raise ValidationError("Historical preparation schema/phase differs")
        if preparation.get("plan_sha256") != plan_sha or preparation.get("source_or_plan_changed") is not False:
            raise ValidationError("Historical transaction plan/source proof differs")
        task_hashes = preparation.get("task_sha256")
        if not isinstance(task_hashes, dict) or published.get("task_sha256") != task_hashes or activation.get("task_sha256") != task_hashes:
            raise ValidationError("Historical task descriptor/activation hashes do not match")
        if (published.get("phase") != "awaiting_owned_session_resume_ack"
                or published.get("resource_release_files_written") is not False
                or activation.get("schema") != ACTIVATION_SCHEMA
                or activation.get("plan_sha256") != plan_sha):
            raise ValidationError("Historical transaction was not the verified append-only resume")
        records = preparation.get("records")
        if not isinstance(records, list) or len(records) != 2:
            raise ValidationError("Historical transaction must bind exactly two owned sessions")
        by_slot = {record.get("slot"): record for record in records}
        if set(by_slot) != {spec.name for spec in self.b.slots}:
            raise ValidationError("Historical session slots differ")
        for spec in self.b.slots:
            record = by_slot[spec.name]
            session = record.get("session", {})
            keeper = record.get("keeper", {})
            if (session.get("pid") != spec.session_pid or str(session.get("start_ticks")) != spec.session_start
                    or session.get("ppid") != spec.session_parent or tuple(session.get("argv", ())) != spec.session_argv):
                raise ValidationError("Historical session identity differs for " + spec.name)
            if (keeper.get("pid") != spec.keeper_pid or str(keeper.get("start_ticks")) != spec.keeper_start
                    or keeper.get("ppid") != spec.session_pid or tuple(keeper.get("argv", ())) != spec.keeper_argv):
                raise ValidationError("Historical keeper identity differs for " + spec.name)
            if record.get("port") != spec.port or record.get("keeper_ready", {}).get("nonce") != spec.nonce:
                raise ValidationError("Historical port/keeper nonce differs for " + spec.name)
            request_path = self.b.campaign / "resumes" / spec.name / (OLD_REQUEST_ID + ".json")
            request_raw = request_path.read_bytes()
            request = json.loads(request_raw)
            receipt = read_json(self.b.campaign / "sessions" / spec.name / "resume_receipts" / (OLD_REQUEST_ID + ".json"))
            proof = receipt.get("source_proof", {})
            if (receipt.get("phase") != "resumed" or receipt.get("request_sha256") != sha256(request_raw)
                    or receipt.get("plan_sha256") != plan_sha or receipt.get("session_pid") != spec.session_pid
                    or str(receipt.get("proc_start_ticks")) != spec.session_start
                    or receipt.get("gpu_uuid") != spec.gpu_uuid
                    or proof.get("verified") is not True or proof.get("plan_sha256") != plan_sha
                    or proof.get("source_hashes") != json.loads((self.b.campaign / "plan.json").read_bytes()).get("source_hashes")
                    or proof.get("session_source_sha256") != sha256((self.b.project / "preparations/ease_insights_20261103/gpu_resource_session.py").read_bytes())
                    or proof.get("keeper", {}).get("nonce") != spec.nonce):
                raise ValidationError("Historical Session ACK/source proof differs for " + spec.name)
            if set(request) != RESUME_FIELDS or request.get("schema") != RESUME_SCHEMA or request.get("action") != "resume_after_attention":
                raise ValidationError("Historical resume request schema differs for " + spec.name)
        return {"preparation": preparation, "published": published, "activation": activation, "records": by_slot}

    def _verify_binding_and_processes(self, spec: SlotSpec) -> dict:
        slot_dir = self.b.campaign / "sessions" / spec.name
        binding = read_json(slot_dir / "binding.json")
        current_session = binding.get("session", {})
        current_scheduler = binding.get("scheduler", {})
        if (binding.get("job_id") != "ease_gpu_resource_session_v1_" + spec.name[-2:]
                or tuple(current_session.get("argv", ())) != spec.session_argv
                or current_session.get("pid") != spec.session_pid
                or str(current_session.get("start_ticks")) != spec.session_start
                or current_session.get("ppid") != spec.session_parent):
            raise ValidationError("Session binding changed for " + spec.name)
        if (current_scheduler.get("pid") != spec.session_parent
                or str(current_scheduler.get("start_ticks")) != spec.scheduler_start
                or tuple(current_scheduler.get("argv", ())) != spec.scheduler_argv):
            raise ValidationError("Recorded scheduler parent changed for " + spec.name)
        if (binding.get("lease_path") != "/tmp/embodied-benchmark-gpu-%s.lock" % spec.gpu_uuid
                or not binding.get("lease_fds")):
            raise ValidationError("Recorded inherited lease identity is missing for " + spec.name)
        scheduler = proc_identity(spec.session_parent, self.b.proc_root)
        session = proc_identity(spec.session_pid, self.b.proc_root)
        keeper = proc_identity(spec.keeper_pid, self.b.proc_root)
        _identity_matches(scheduler, pid=spec.session_parent, start=spec.scheduler_start,
                          ppid=1, argv=spec.scheduler_argv, label=spec.name + " scheduler")
        _identity_matches(session, pid=spec.session_pid, start=spec.session_start,
                          ppid=spec.session_parent, argv=spec.session_argv, label=spec.name + " session")
        _identity_matches(keeper, pid=spec.keeper_pid, start=spec.keeper_start,
                          ppid=spec.session_pid, argv=spec.keeper_argv, label=spec.name + " keeper")
        ready = read_json(slot_dir / "keeper.json")
        if (ready.get("schema") != "ease-gpu-owned-keeper-v1" or ready.get("phase") != "ready"
                or ready.get("context_created") is not True or ready.get("allocated_bytes") != 64 * 1024 * 1024
                or ready.get("gpu_uuid") != spec.gpu_uuid or ready.get("pid") != spec.keeper_pid
                or str(ready.get("proc_start_ticks")) != spec.keeper_start or ready.get("nonce") != spec.nonce):
            raise ValidationError("Live keeper receipt/nonce differs for " + spec.name)
        if proc_children(spec.keeper_pid, self.b.proc_root):
            raise ValidationError("Keeper has an unexpected live child for " + spec.name)
        children = proc_children(spec.session_pid, self.b.proc_root)
        if spec.keeper_pid not in children:
            raise ValidationError("Pinned keeper is not the live session child for " + spec.name)
        child_identities = []
        for pid in children:
            child_identities.append(proc_identity(pid, self.b.proc_root))
        return {"binding": binding, "scheduler": scheduler, "session": session, "keeper": keeper,
                "keeper_ready": ready, "children": child_identities}

    def _watcher_device(self, spec: SlotSpec, phase: str) -> dict:
        receipt = read_json(self.b.watch_dir / "process_receipt.json")
        status = read_json(self.b.watch_dir / "status.json")
        source_path = canonical(self.b.project / "preparations/ease_insights_20261103/gpu_release_watch.py")
        if (receipt.get("mode") != "observe_only" or receipt.get("poll_seconds") != 2.0
                or status.get("mode") != "observe_only"
                or receipt.get("source_path") != str(source_path)
                or receipt.get("source_sha256") != sha256(source_path.read_bytes())):
            raise ValidationError("Live read-only GPU watcher source identity differs")
        pid = receipt.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            raise ValidationError("Watcher has no valid local PID")
        live = proc_identity(pid, self.b.proc_root)
        if (str(live.get("start_ticks")) != str(receipt.get("proc_start_ticks"))
                or " ".join(live.get("argv", [])) != receipt.get("command_line")
                or receipt.get("boot_id") != self._boot_id()):
            raise ValidationError("GPU watcher PID/start/argv/boot identity differs")
        for key in ("pid", "proc_start_ticks", "source_sha256", "run_id"):
            left = status.get(key)
            right = receipt.get(key)
            if str(left) != str(right):
                raise ValidationError("Fresh watcher status does not match its live receipt (%s)" % key)
        if status.get("error") is not None:
            raise ValidationError("Live watcher reports an error", "waiting")
        try:
            finished = float(status["query_finished_at"])
            age = self.clock() - finished
            heartbeat = float(status["heartbeat_at"])
            all_devices = status["devices"]
            memory = float(next(item["memory_mib"] for item in all_devices if item.get("gpu_uuid") == spec.gpu_uuid))
            devices = [item for item in all_devices if item.get("gpu_uuid") == spec.gpu_uuid]
            count = int(devices[0]["compute_process_count"]) if len(devices) == 1 else -1
            host_pids = devices[0].get("host_compute_pids") if len(devices) == 1 else None
        except (OSError, ValueError, KeyError, TypeError, StopIteration, IndexError) as error:
            raise ValidationError("Watcher has no unique assigned-GPU sample: %s" % error, "waiting") from error
        if (not math.isfinite(age) or age < 0 or age > WATCH_FRESH_SECONDS
                or not math.isfinite(heartbeat) or heartbeat < finished
                or self.clock() - heartbeat < 0 or self.clock() - heartbeat > WATCH_FRESH_SECONDS):
            raise ValidationError("Assigned-GPU watcher sample is stale", "waiting")
        if (not isinstance(all_devices, list) or len(all_devices) != 4
                or {item.get("index") for item in all_devices} != {0, 1, 2, 3}
                or len({item.get("gpu_uuid") for item in all_devices}) != 4
                or len(devices) != 1 or not math.isfinite(memory) or memory < 0 or count <= 0):
            raise ValidationError("Assigned-GPU watcher sample is invalid", "waiting")
        if not isinstance(host_pids, list) or len(host_pids) != count or any(not isinstance(value, int) or value <= 0 for value in host_pids):
            raise ValidationError("Watcher compute inventory is incomplete", "waiting")
        if phase in ("attention_holding", "idle_holding"):
            if count != 1 or memory > MAX_KEEPER_MIB:
                raise ValidationError("Eligible GPU is not the sole bounded keeper yet", "waiting")
            attribution = "sole_bounded_keeper"
        else:
            # A valid live service may add its server (and optional worker) to
            # the keeper. Host watcher PIDs remain opaque; we never equate them
            # to local /proc child PIDs.
            if count > 3 or (count == 1 and memory > MAX_KEEPER_MIB) or (count >= 2 and memory <= MAX_KEEPER_MIB):
                raise ValidationError("Running service GPU sample is outside the owned keeper/server envelope", "waiting")
            attribution = "owned_running_service_with_keeper"
        return {"run_id": receipt.get("run_id"), "pid": pid,
                "proc_start_ticks": str(receipt.get("proc_start_ticks")), "query_finished_at": finished,
                "age_seconds": age, "gpu_uuid": spec.gpu_uuid, "memory_mib": memory,
                "compute_process_count": count, "host_compute_pids_unmapped": host_pids,
                "attribution": attribution}

    def _boot_id(self) -> str:
        try:
            return (Path(self.b.proc_root) / "sys/kernel/random/boot_id").read_text().strip()
        except OSError as error:
            raise ValidationError("Cannot verify watcher boot identity: %s" % error) from error

    def _load_activated_task(self, task_id: str) -> tuple[dict, bytes, str]:
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise ValidationError("Invalid task ID in live session status")
        task_path = canonical(self.b.campaign / "tasks" / (task_id + ".json"))
        raw = task_path.read_bytes()
        task = json.loads(raw)
        if task.get("id") != task_id or task.get("schema") != TASK_SCHEMA:
            raise ValidationError("Live task ID/schema differs from its descriptor")
        digest = sha256(raw)
        activation = canonical(Path(task.get("activation_file", "")))
        if (activation.parent.parent != self.b.campaign / "transactions" or activation.name != "committed.json"):
            raise ValidationError("Task activation is outside the fixed campaign transactions")
        committed = read_json(activation)
        if (committed.get("schema") != ACTIVATION_SCHEMA or committed.get("plan_sha256") != self.b.expected_plan_sha256
                or committed.get("task_sha256", {}).get(task_id) != digest):
            raise ValidationError("Live task lacks its exact committed descriptor hash")
        return task, raw, digest

    def _validate_science_task(self, task: dict, plan: dict, *, target: bool) -> dict:
        required = {"schema", "id", "logical_job_id", "argv", "complete_file", "manifest_hash",
                    "output", "condition", "stage", "shard_index", "shards", "env", "activation_file"}
        if set(task) != required or task.get("schema") != TASK_SCHEMA:
            raise ValidationError("Scientific task schema/fields differ")
        if task.get("env") != {} or not isinstance(task.get("argv"), list) or not task["argv"]:
            raise ValidationError("Scientific task argv/env differs from the pinned empty-env runner")
        if not isinstance(task.get("logical_job_id"), str) or not task["logical_job_id"]:
            raise ValidationError("Scientific task logical job identity is missing")
        output = canonical(Path(task["output"]))
        if str(output) not in plan["allowed_outputs"] or task["argv"][0] != "/root/nfs/embodied/benchmark/policies/realtime-vla-flash/.venv/bin/python":
            raise ValidationError("Scientific task output or frozen Python differs")
        if not task.get("condition") or task.get("stage") not in ("discovery", "confirm"):
            raise ValidationError("Scientific task condition/stage is outside the frozen runner")
        if (not isinstance(task.get("shard_index"), int) or isinstance(task.get("shard_index"), bool)
                or not isinstance(task.get("shards"), int) or isinstance(task.get("shards"), bool)
                or task["shards"] <= 0 or not 0 <= task["shard_index"] < task["shards"]):
            raise ValidationError("Scientific task shard tuple is invalid")
        expected_argv = [task["argv"][0], "-m", "preparations.ease_insights_20261103.vlm_local_search_v1.runner",
                         "--output", str(output), "--condition", task["condition"], "--gpu-uuid", "{gpu_uuid}",
                         "--port", "{port}", "--launch"]
        if task["argv"] != expected_argv:
            raise ValidationError("Scientific argv is not the exact frozen runner invocation")
        manifest_path = canonical(output / "manifest.json")
        manifest = read_json(manifest_path)
        if (manifest.get("manifest_hash") != task.get("manifest_hash")
                or task["condition"] not in manifest.get("planned_conditions", [])):
            raise ValidationError("Scientific manifest/condition identity differs")
        manifest_identity = manifest.get("manifest_hash")
        manifest_payload = dict(manifest)
        manifest_payload.pop("manifest_hash", None)
        computed_manifest_hash = sha256(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode())
        if manifest_identity != computed_manifest_hash:
            raise ValidationError("Scientific manifest content differs from its frozen manifest hash")
        source_files = manifest.get("source_files")
        if not isinstance(source_files, dict) or not source_files:
            raise ValidationError("Scientific manifest has no frozen implementation source pins")
        for name, expected in source_files.items():
            source = canonical(self.b.project / name)
            if not source.is_relative_to(self.b.project.resolve()) or sha256(source.read_bytes()) != expected:
                raise ValidationError("Frozen scientific implementation source changed: " + name)
        expected_stage = "discovery" if manifest.get("phase") == "local_search_discovery_v1" else (
            "confirm" if manifest.get("phase") == "local_search_confirmation_v1" else None)
        if expected_stage != task["stage"]:
            raise ValidationError("Task stage differs from its immutable manifest")
        marker = canonical(output / task["condition"] / ("%s_shard%02dof%02d_complete.json" %
                           (task["stage"], task["shard_index"], task["shards"])))
        if canonical(Path(task["complete_file"])) != marker:
            raise ValidationError("Scientific completion marker path differs")
        if target:
            if "__port_retry" in task["id"]:
                raise ValidationError("Refusing retry of a retry task")
            if marker.exists():
                raise ValidationError("Failed task's complete marker already exists")
            episode_root = canonical(output / task["condition"] / "episodes")
            if episode_root.exists():
                if episode_root.is_symlink():
                    raise ValidationError("Target condition episode directory is a symlink")
                receipts = list(episode_root.rglob("episode_receipt.json"))
                if receipts:
                    raise ValidationError("Target condition already has committed episode receipts")
        return {"output": output, "manifest": manifest, "marker": marker,
                "target_episode_receipt_count": 0 if target else None}

    def _running_child_roles(self, spec: SlotSpec, task: dict, identities: list[dict]) -> dict:
        by_pid = {item["pid"]: item for item in identities}
        science = [item for pid, item in by_pid.items() if pid != spec.keeper_pid]
        if len(science) not in (1, 2):
            raise ValidationError("Running session must own exactly a server and optional worker")
        stage_label = "discovery" if task["stage"] == "discovery" else "confirm"
        index, shards = task["shard_index"], task["shards"]
        output = Path(task["output"])
        manifest = output / "manifest.json"
        server_candidates = []
        worker_candidates = []
        for child in science:
            if child.get("uid") != 0:
                raise ValidationError("Running science child UID differs from its owned session")
            argv = child["argv"]
            if len(argv) >= 3 and argv[1:3] == ["-m", "preparations.ease_insights_20261103.vlm_local_search_v1.server"]:
                server_candidates.append(child)
            elif len(argv) >= 3 and argv[1:3] == ["-m", "preparations.ease_insights_20261103.vlm_local_search_v1.worker"]:
                worker_candidates.append(child)
            else:
                raise ValidationError("Running session has an unknown direct science child")
        if len(server_candidates) != 1 or len(worker_candidates) > 1:
            raise ValidationError("Running direct server/worker child count differs")
        server = server_candidates[0]
        logs_arg = self._option(server["argv"], "--output")
        expected_server = [task["argv"][0], "-m", "preparations.ease_insights_20261103.vlm_local_search_v1.server",
                           "--manifest", str(manifest), "--condition", task["condition"], "--port", str(spec.port),
                           "--gpu-uuid", spec.gpu_uuid, "--output", logs_arg]
        logs = canonical(Path(logs_arg or ""))
        run_prefix = "%s_shard%02dof%02d_" % (stage_label, index, shards)
        if (server["ppid"] != spec.session_pid or server["argv"] != expected_server
                or not logs.is_relative_to(output / task["condition"] / "runs")
                or not logs.name.startswith(run_prefix) or not logs.name[len(run_prefix):].isdecimal()):
            raise ValidationError("Running server parent/argv/UUID/port/condition identity differs")
        worker = worker_candidates[0] if worker_candidates else None
        if worker is not None:
            expected_worker = ["/root/nfs/embodied/benchmark/policies/libero-eval-v21/.venv/bin/python",
                              "-m", "preparations.ease_insights_20261103.vlm_local_search_v1.worker",
                              "--output", str(output), "--condition", task["condition"], "--port", str(spec.port),
                              "--stage", stage_label, "--shard-index", str(index), "--shards", str(shards)]
            if worker["ppid"] != spec.session_pid or worker["argv"] != expected_worker:
                raise ValidationError("Running worker parent/argv/UUID/port/condition identity differs")
        return {"server": {"pid": server["pid"], "start_ticks": server["start_ticks"], "argv": server["argv"]},
                "worker": None if worker is None else {"pid": worker["pid"], "start_ticks": worker["start_ticks"], "argv": worker["argv"]}}

    @staticmethod
    def _option(argv: list[str], name: str) -> str | None:
        try:
            index = argv.index(name)
            return argv[index + 1]
        except (ValueError, IndexError):
            return None

    def _slot_state(self, spec: SlotSpec, plan: dict,
                    allowed_orphans: Mapping[str, dict] | None = None) -> dict:
        live = self._verify_binding_and_processes(spec)
        status = read_json(self.b.campaign / "sessions" / spec.name / "status.json")
        if (status.get("schema") != "ease-gpu-resource-session-v1" or status.get("gpu_uuid") != spec.gpu_uuid
                or status.get("session_pid") != spec.session_pid):
            raise ValidationError("Session live status identity differs for " + spec.name)
        phase = status.get("phase")
        if phase not in ("attention_holding", "idle_holding", "running"):
            raise ValidationError("Session phase is not safely recoverable: %s/%s" % (spec.name, phase), "waiting")
        watcher = self._watcher_device(spec, phase)
        child_pids = sorted(item["pid"] for item in live["children"])
        if phase in ("attention_holding", "idle_holding"):
            if child_pids != [spec.keeper_pid]:
                raise ValidationError("Paused/idle session still has a live science child: " + spec.name, "waiting")
            port_ready = self.bind_probe(spec.port)
            if not port_ready:
                raise ValidationError("Owned port remains busy; wait for a fresh bind-ready observation: " + spec.name, "waiting")
            slot = {"phase": phase, "status_time": status.get("time"), "task_id": status.get("task_id"),
                    "port": spec.port, "port_bind_ready": True, "children": live["children"], "watcher": watcher}
            if phase == "idle_holding":
                if status.get("attention") is not None:
                    raise ValidationError("Idle session unexpectedly retains attention")
                return {"spec": spec, "slot": slot, "target": None, "live": live}
            attention = status.get("attention")
            if not isinstance(attention, dict) or attention.get("error") != PORT_ERROR:
                raise ValidationError("Attention does not match the exact address-in-use outcome: " + spec.name)
            task_id = attention.get("task_id")
            if status.get("task_id") not in (None, task_id):
                raise ValidationError("Status/attention task IDs differ for " + spec.name)
            task, raw, task_sha = self._load_activated_task(task_id)
            task_paths = self._validate_science_task(task, plan, target=True)
            # Outcome filenames are SHA256(task_id UTF-8), independent from task bytes.
            outcome_path = canonical(self.b.campaign / "outcomes" / (sha256(task_id.encode()) + ".json"))
            outcome_raw = outcome_path.read_bytes()
            outcome = json.loads(outcome_raw)
            if outcome.get("error") != PORT_ERROR:
                raise ValidationError("Old attention outcome is not the exact address-in-use error")
            if (outcome.get("schema") != "ease-gpu-resource-session-v1" or outcome.get("phase") != "attention"
                    or outcome.get("task_id") != task_id or outcome.get("task_sha256") != task_sha
                    or outcome.get("gpu_uuid") != spec.gpu_uuid
                    or attention != {"task_id": task_id, "error": PORT_ERROR}):
                raise ValidationError("Attention status/outcome/task hash do not match for " + spec.name)
            new_id = task_id + "__port_retry1"
            if "__port_retry" in task_id or not TASK_ID_RE.fullmatch(new_id):
                raise ValidationError("Refusing retry-of-retry or invalid bounded task ID")
            new_task_path = self.b.campaign / "tasks" / (new_id + ".json")
            if self._activation_already_contains(new_id):
                raise ValidationError("Single retry task ID already exists; refusing replay: " + new_id)
            slot["attention"] = attention
            slot["attention_sha256"] = attention_sha256(attention)
            slot["task_sha256"] = task_sha
            slot["outcome_sha256"] = sha256(outcome_raw)
            slot["task_id"] = task_id
            slot["new_task_id"] = new_id
            slot["target_episode_receipt_count"] = task_paths["target_episode_receipt_count"]
            new_task = dict(task)
            new_task["id"] = new_id
            # This is deliberately the only other changed field.
            new_task["activation_file"] = "PENDING_ACTIVATION_PATH"
            if set(new_task) != set(task) or {key for key in task if task[key] != new_task[key]} != {"id", "activation_file"}:
                raise ValidationError("Retry descriptor changed fields outside id/activation_file")
            if new_task_path.exists():
                allowed = (allowed_orphans or {}).get(new_id)
                if allowed is None:
                    raise ValidationError("Single retry task ID already exists; refusing replay: " + new_id)
                actual_raw = new_task_path.read_bytes()
                actual = json.loads(actual_raw)
                expected = dict(task, id=new_id, activation_file=allowed["activation_file"])
                if actual != expected or sha256(actual_raw) != allowed["sha256"]:
                    raise ValidationError("Existing unactivated retry descriptor differs from this append attempt")
            return {"spec": spec, "slot": slot, "target": {"old_task": task, "old_task_raw": raw,
                    "old_task_sha256": task_sha, "outcome": outcome, "outcome_sha256": sha256(outcome_raw),
                    "attention": attention, "attention_sha256": attention_sha256(attention),
                    "new_task": new_task, "task_paths": task_paths}, "live": live}
        if status.get("attention") is not None or status.get("task_id") is None:
            raise ValidationError("Running session task identity is incomplete")
        task, raw, task_sha = self._load_activated_task(status["task_id"])
        self._validate_science_task(task, plan, target=False)
        if status.get("logical_job_id") != task.get("logical_job_id"):
            raise ValidationError("Running session logical task differs from its descriptor")
        roles = self._running_child_roles(spec, task, live["children"])
        slot = {"phase": phase, "status_time": status.get("time"), "task_id": task["id"],
                "task_sha256": task_sha, "logical_job_id": task["logical_job_id"],
                "port": spec.port, "port_bind_ready": None, "children": live["children"],
                "science_child_roles": roles, "watcher": watcher}
        return {"spec": spec, "slot": slot, "target": None, "live": live}

    def _activation_already_contains(self, task_id: str) -> bool:
        for path in (self.b.campaign / "transactions").glob("*/committed.json"):
            try:
                activation = json.loads(path.read_bytes())
            except (OSError, ValueError):
                continue
            if task_id in activation.get("task_sha256", {}):
                return True
        return False

    def inspect(self, *, allowed_orphans: Mapping[str, dict] | None = None,
                allow_existing_transaction: bool = False) -> dict:
        self_sha = self._helper_sha()
        plan, plan_sha = self._verify_plan()
        history = self._verify_history(plan_sha)
        slots = [self._slot_state(spec, plan, allowed_orphans) for spec in self.b.slots]
        targets = [item for item in slots if item["target"] is not None]
        if not targets:
            raise ValidationError("No exact address-in-use attention is ready for a one-shot retry", "waiting")
        stable_targets = []
        for item in slots:
            if item["target"] is None:
                continue
            spec, slot = item["spec"], item["slot"]
            stable_targets.append({"slot": spec.name, "task_id": slot["task_id"],
                "task_sha256": slot["task_sha256"], "outcome_sha256": slot["outcome_sha256"],
                "attention_sha256": slot["attention_sha256"], "new_task_id": slot["new_task_id"],
                "session": {"pid": spec.session_pid, "start_ticks": spec.session_start,
                            "ppid": spec.session_parent, "argv": list(spec.session_argv)},
                "keeper": {"pid": spec.keeper_pid, "start_ticks": spec.keeper_start,
                           "ppid": spec.session_pid, "nonce": spec.nonce, "argv": list(spec.keeper_argv)}})
        slot_identity = {}
        for item in slots:
            spec, slot = item["spec"], item["slot"]
            slot_identity[spec.name] = {"phase": slot["phase"], "task_id": slot.get("task_id"),
                "task_sha256": slot.get("task_sha256"), "logical_job_id": slot.get("logical_job_id"),
                "attention_sha256": slot.get("attention_sha256"),
                "session": {"pid": spec.session_pid, "start_ticks": spec.session_start,
                            "ppid": spec.session_parent, "argv": list(spec.session_argv)},
                "keeper": {"pid": spec.keeper_pid, "start_ticks": spec.keeper_start,
                           "ppid": spec.session_pid, "nonce": spec.nonce, "argv": list(spec.keeper_argv)},
                "children": [{key: child.get(key) for key in ("pid", "start_ticks", "ppid", "argv")}
                             for child in slot["children"]],
                "science_child_roles": slot.get("science_child_roles")}
        logical = {"schema": "ease-explicit-port-retry-input-v1", "plan_sha256": plan_sha,
                   "self_sha256": self_sha, "targets": stable_targets, "slot_identity": slot_identity}
        audit_id = sha256(json.dumps(logical, sort_keys=True).encode())[:32]
        request_id = "port_retry_ready_" + audit_id
        transaction = self.b.campaign / "transactions" / ("port_retry_helper_" + audit_id)
        for spec in self.b.slots:
            request_path = self.b.campaign / "resumes" / spec.name / (request_id + ".json")
            receipt_path = self.b.campaign / "sessions" / spec.name / "resume_receipts" / (request_id + ".json")
            if request_path.exists() or receipt_path.exists():
                raise ValidationError("This single-use resume request ID already exists")
        if transaction.exists() and not allow_existing_transaction:
            raise ValidationError("This audit already created an append-only transaction; inspect it before retrying")
        if transaction.exists() and allow_existing_transaction:
            existing_preparation = read_json(transaction / "preparation.json")
            if existing_preparation.get("audit_id") != audit_id:
                raise ValidationError("Existing append-only transaction does not belong to this audit")
        return {"schema": "ease-explicit-port-retry-audit-v1", "audit_id": audit_id,
                "request_id": request_id, "transaction_dir": transaction,
                "self_sha256": self_sha, "plan_sha256": plan_sha,
                "source_hashes": plan["source_hashes"], "input_tuple": logical,
                "slots": slots, "targets": targets, "history": history}

    def _verify_repeat(self, initial: dict, *, allowed_orphans: Mapping[str, dict] | None = None,
                       allow_existing_transaction: bool = False) -> dict:
        current = self.inspect(allowed_orphans=allowed_orphans,
                               allow_existing_transaction=allow_existing_transaction)
        if current["audit_id"] != initial["audit_id"]:
            # The inspect pass also re-probed every newly idle/attention port.
            raise ValidationError("Owned slot/task tuple changed during publication validation; inspect the fresh port evidence and restart")
        for old, new in zip(initial["slots"], current["slots"]):
            if old["spec"].name != new["spec"].name:
                raise ValidationError("Fixed slot order changed")
            if old["slot"]["phase"] == "running" and new["slot"]["phase"] != "running":
                # inspect() has re-evaluated its port when it became eligible.
                raise ValidationError("Running slot changed phase; fresh eligibility/port was rechecked, so fail closed and rerun dry run")
            if old["slot"]["phase"] != "running" and old["slot"]["phase"] != new["slot"]["phase"]:
                raise ValidationError("Paused/idle slot changed phase during publication validation")
        return current

    @staticmethod
    def _write_exclusive(path: Path, payload: bytes) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        PortRetryController._sync_dir(path.parent)

    @staticmethod
    def _publish_atomically_no_replace(path: Path, payload: bytes, nonce: str) -> None:
        """Atomic same-directory link publication; never replaces an existing inbox ID."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name("." + path.name + "." + nonce + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
            PortRetryController._sync_dir(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _sync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _request(self, target: dict, plan_sha: str, session_pid: int, start: str, gpu_uuid: str) -> dict:
        value = {"schema": RESUME_SCHEMA, "action": "resume_after_attention", "plan_sha256": plan_sha,
                 "session_pid": session_pid, "proc_start_ticks": str(start), "gpu_uuid": gpu_uuid,
                 "attention_sha256": target["attention_sha256"],
                 "reason": "The prior successor failed before server/cases with the exact address-in-use error; its original outcomes are preserved and the same keeper remains live."}
        if set(value) != RESUME_FIELDS:
            raise ValidationError("Internal resume request schema is not exact")
        return value

    def apply(self, initial: dict) -> dict:
        transaction = initial["transaction_dir"]
        audit_id = initial["audit_id"]
        request_id = initial["request_id"]
        # Recheck all pins, identities, children, live watcher evidence and
        # every eligible port immediately before creating append-only records.
        fresh = self._verify_repeat(initial)
        transaction.mkdir(parents=False, exist_ok=False)
        self._sync_dir(transaction.parent)
        preparation = {"schema": "ease-explicit-port-retry-helper-v1", "phase": "prepared",
            "audit_id": audit_id, "request_id": request_id, "time": self.clock(),
            "plan_sha256": fresh["plan_sha256"], "self_sha256": fresh["self_sha256"],
            "source_hashes": fresh["source_hashes"], "input_tuple": fresh["input_tuple"],
            "validation_passes": [self._pass_record(initial), self._pass_record(fresh)],
            "signals_sent": False, "gpu_queries_run": False, "jobs_state_written": False,
            "resource_release_files_written": False, "valid_outcomes_modified": False}
        prep_path = transaction / "preparation.json"
        self._write_exclusive(prep_path, json_bytes(preparation))
        task_hashes = {}
        for item in fresh["targets"]:
            target = item["target"]
            new_task = dict(target["new_task"])
            new_task["activation_file"] = str(transaction / "committed.json")
            old_task = target["old_task"]
            changed = {key for key in old_task if old_task[key] != new_task[key]}
            if changed != {"id", "activation_file"}:
                raise ValidationError("New descriptor did not change exactly id/activation_file")
            payload = json_bytes(new_task)
            task_hashes[new_task["id"]] = sha256(payload)
            descriptor = self.b.campaign / "tasks" / (new_task["id"] + ".json")
            if descriptor.exists():
                raise ValidationError("Append-only retry descriptor appeared during publication")
            self._publish_atomically_no_replace(descriptor, payload, audit_id + "_" + new_task["id"])
        # If a running slot becomes idle/attention during descriptor append,
        # inspect re-probes any newly eligible port; any changed tuple stops
        # before activation. Session.reserve_port remains the final race guard.
        activation_path = transaction / "committed.json"
        allowed_orphans = {task_id: {"sha256": digest, "activation_file": str(activation_path)}
                           for task_id, digest in task_hashes.items()}
        fresh_after_tasks = self._verify_repeat(fresh, allowed_orphans=allowed_orphans,
                                                allow_existing_transaction=True)
        activation = {"schema": ACTIVATION_SCHEMA, "plan_sha256": fresh_after_tasks["plan_sha256"],
                     "task_sha256": task_hashes}
        self._publish_atomically_no_replace(activation_path, json_bytes(activation), audit_id)
        request_records = {}
        for item in fresh_after_tasks["targets"]:
            spec = item["spec"]
            target = item["target"]
            request = self._request(target, fresh_after_tasks["plan_sha256"], spec.session_pid,
                                     spec.session_start, spec.gpu_uuid)
            payload = json_bytes(request)
            request_path = self.b.campaign / "resumes" / spec.name / (request_id + ".json")
            self._publish_atomically_no_replace(request_path, payload, audit_id + "_" + spec.name)
            request_records[spec.name] = {"path": str(request_path), "request_sha256": sha256(payload),
                                         "attention_sha256": target["attention_sha256"],
                                         "session_pid": spec.session_pid, "proc_start_ticks": spec.session_start,
                                         "gpu_uuid": spec.gpu_uuid}
        published = {"schema": "ease-explicit-port-retry-helper-publication-v1",
                     "phase": "published_wait_ack", "audit_id": audit_id, "request_id": request_id,
                     "time": self.clock(), "activation": str(activation_path),
                     "task_sha256": task_hashes, "resume_requests": request_records,
                     "all_scientific_fields_unchanged": True,
                     "old_completed_and_attention_outcomes_preserved": True,
                     "resource_release_files_written": False,
                     "acknowledgement": "Await fresh per-session resume_receipts with phase=resumed and source_proof.verified=true."}
        self._write_exclusive(transaction / "published.json", json_bytes(published))
        return {"phase": "published_wait_ack", "audit_id": audit_id, "request_id": request_id,
                "transaction_dir": str(transaction), "task_sha256": task_hashes,
                "resume_requests": request_records, "self_sha256": fresh_after_tasks["self_sha256"],
                "plan_sha256": fresh_after_tasks["plan_sha256"]}

    @staticmethod
    def _pass_record(value: dict) -> dict:
        return {"audit_id": value["audit_id"], "self_sha256": value["self_sha256"],
                "plan_sha256": value["plan_sha256"],
                "slots": {item["spec"].name: item["slot"] for item in value["slots"]},
                "target_task_ids": [item["target"]["new_task"]["id"] for item in value["slots"] if item["target"]]}

    @contextmanager
    def controller_lock(self):
        self.b.report_dir.mkdir(parents=True, exist_ok=True)
        path = self.b.report_dir / "controller.lock"
        with path.open("a") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValidationError("Another port-retry helper holds the controller lock", "waiting") from error
            yield

    def _write_report(self, report: dict) -> Path:
        self.b.report_dir.mkdir(parents=True, exist_ok=True)
        suffix = "%s_%d.json" % (report.get("audit_id", "unbound"), self.clock_ns())
        path = self.b.report_dir / ("run_" + suffix)
        self._write_exclusive(path, json_bytes(report))
        return path

    def run(self, *, apply: bool = False) -> dict:
        with self.controller_lock():
            report = {"schema": "ease-explicit-port-retry-helper-report-v1", "mode": "apply" if apply else "dry_run",
                      "time": self.clock(), "self_sha256": self._helper_sha(), "actions_taken": []}
            try:
                audited = self.inspect()
                report.update({"phase": "dry_run_ready", "audit_id": audited["audit_id"],
                               "plan_sha256": audited["plan_sha256"], "source_hashes": audited["source_hashes"],
                               "input_tuple": audited["input_tuple"],
                               "slots": self._pass_record(audited)["slots"],
                               "candidate_tasks": [item["target"]["new_task"]["id"] for item in audited["slots"] if item["target"]],
                               "resume_request_id": audited["request_id"]})
                if apply:
                    result = self.apply(audited)
                    report.update(result)
                    report["actions_taken"] = ["append task descriptors", "publish matching activation", "atomically publish resume requests"]
            except ValidationError as error:
                partial_tx = None
                if apply and "audited" in locals() and audited["transaction_dir"].exists():
                    partial_tx = audited["transaction_dir"]
                report["phase"] = "fail_closed_partial" if partial_tx else error.status
                report["reason"] = str(error)
                report["actions_taken"] = ([] if partial_tx is None else
                    ["append-only transaction may be partial; inspect before any further action"])
                if partial_tx is not None:
                    report["transaction_dir"] = str(partial_tx)
                    expected_task_ids = [item["target"]["new_task"]["id"]
                                         for item in audited["slots"] if item["target"]]
                    report["task_descriptors_present"] = sorted(
                        task_id for task_id in expected_task_ids
                        if (self.b.campaign / "tasks" / (task_id + ".json")).exists())
                    report["activation_committed"] = (partial_tx / "committed.json").exists()
                    report["resume_requests_published"] = sorted(
                        spec.name for spec in self.b.slots
                        if (self.b.campaign / "resumes" / spec.name / (audited["request_id"] + ".json")).exists())
            except (OSError, ValueError, TypeError, KeyError) as error:
                partial_tx = None
                if apply and "audited" in locals() and audited["transaction_dir"].exists():
                    partial_tx = audited["transaction_dir"]
                report["phase"] = "fail_closed_partial" if partial_tx else "refused"
                report["reason"] = "Fail-closed file/protocol error: %s: %s" % (type(error).__name__, error)
                report["actions_taken"] = ([] if partial_tx is None else
                    ["append-only transaction may be partial; inspect before any further action"])
                if partial_tx is not None:
                    report["transaction_dir"] = str(partial_tx)
                    report["activation_committed"] = (partial_tx / "committed.json").exists()
                    report["resume_requests_published"] = sorted(
                        spec.name for spec in self.b.slots
                        if (self.b.campaign / "resumes" / spec.name / (audited["request_id"] + ".json")).exists())
            report["report_path"] = str(self._write_report(report))
            return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Append the bounded retry transaction and publish resume requests")
    args = parser.parse_args(argv)
    controller = PortRetryController()
    report = controller.run(apply=args.apply)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if report.get("phase") in ("dry_run_ready", "published_wait_ack", "waiting") else 2


if __name__ == "__main__":
    sys.exit(main())
