"""Read-only, two-second all-GPU observer and incremental idle-event waiter.

Run the persistent observer with ``python -m preparations.ease_insights_20261103.gpu_release_watch``.
Wait separately with ``...gpu_release_watch --wait-for-idle-event --after-event-id N``.
The waiter exits on the first subsequent transition into idle; feed its event_id
back into the next waiter. No GPU lease, scheduler, gate, or model is touched.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


SOURCE = Path(__file__).resolve()
OUTPUT = SOURCE.parents[2] / "results/ease_insights_20261103/gpu_release_watch"
POLL_SECONDS = 2.0
QUERY_TIMEOUT_SECONDS = 0.85
MEMORY_LIMIT_MIB = 500.0
EXPECTED_INDICES = set(range(4))
UUID = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def process_identity():
    # /proc field 22, parsed after the closing ')' because comm can contain spaces.
    fields = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()
    return dict(pid=os.getpid(), proc_start_ticks=int(fields[19]),
                source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                source_path=str(SOURCE), python_executable=str(Path(sys.executable).resolve()),
                command_line=Path(f"/proc/{os.getpid()}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip(),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())


def parse_queries(gpu_raw, apps_raw):
    """Require exactly GPU 0..3 and fully valid rows before any idle verdict."""
    devices = {}
    seen_uuids = set()
    for row in gpu_raw.splitlines():
        if not row.strip():
            continue
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 3:
            raise ValueError("GPU inventory row needs index,uuid,memory.used")
        index_raw, gpu_uuid, memory_raw = parts
        if not index_raw.isdecimal() or not UUID.fullmatch(gpu_uuid):
            raise ValueError("Invalid GPU identity")
        index, memory = int(index_raw), float(memory_raw)
        if index in devices or gpu_uuid in seen_uuids or not math.isfinite(memory) or memory < 0:
            raise ValueError("Duplicate GPU identity or invalid/nonfinite memory")
        devices[index] = dict(index=index, gpu_uuid=gpu_uuid, memory_mib=memory,
                              compute_process_count=0, host_compute_pids=[])
        seen_uuids.add(gpu_uuid)
    if set(devices) != EXPECTED_INDICES:
        raise ValueError("Incomplete inventory: expected exactly GPU indices 0,1,2,3")
    by_uuid = {item["gpu_uuid"]: item for item in devices.values()}
    seen_apps = set()
    for row in apps_raw.splitlines():
        if not row.strip():
            continue
        parts = [part.strip() for part in row.split(",")]
        if len(parts) != 2:
            raise ValueError("Compute process row needs gpu_uuid,pid")
        gpu_uuid, pid_raw = parts
        if gpu_uuid not in by_uuid or not pid_raw.isdecimal() or int(pid_raw) <= 0:
            raise ValueError("Unknown GPU or invalid compute PID")
        key = (gpu_uuid, int(pid_raw))
        if key in seen_apps:
            raise ValueError("Duplicate compute process row")
        seen_apps.add(key)
        by_uuid[gpu_uuid]["host_compute_pids"].append(int(pid_raw))
        by_uuid[gpu_uuid]["compute_process_count"] += 1
    for item in devices.values():
        idle = item["compute_process_count"] == 0 and item["memory_mib"] < MEMORY_LIMIT_MIB
        item.update(state="idle" if idle else "busy", idle=idle)
    return [devices[index] for index in sorted(devices)]


def scan():
    """A failed or incomplete pair marks all four devices unknown, never idle."""
    started = time.time()
    query_records = []
    raw = {}
    error = None
    for kind, query in (("gpu", "--query-gpu=index,uuid,memory.used"),
                        ("apps", "--query-compute-apps=gpu_uuid,pid")):
        command = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
        record = dict(kind=kind, started_at=time.time(), command=command)
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=QUERY_TIMEOUT_SECONDS, check=True)
            record.update(stdout=result.stdout, stderr=result.stderr, returncode=result.returncode)
            raw[kind] = result.stdout
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, subprocess.CalledProcessError):
                record.update(stdout=exc.stdout, stderr=exc.stderr, returncode=exc.returncode)
            error = record["error"]
        record["finished_at"] = time.time()
        query_records.append(record)
    if error is None:
        try:
            devices = parse_queries(raw["gpu"], raw["apps"])
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if error is not None:
        devices = [dict(index=index, gpu_uuid=None, memory_mib=None,
                        compute_process_count=None, host_compute_pids=None,
                        state="unknown", idle=None) for index in sorted(EXPECTED_INDICES)]
    finished = time.time()
    return dict(query_started_at=started, query_started_at_utc=utc(started),
                query_finished_at=finished, query_finished_at_utc=utc(finished),
                query_duration_seconds=finished-started, error=error,
                queries=query_records, devices=devices)


def last_event_id(path, byte_limit=None):
    latest = 0
    if Path(path).exists():
        with Path(path).open("rb") as stream:
            data = stream.read() if byte_limit is None else stream.read(byte_limit)
        for line in data.splitlines(keepends=True):
            if not line.endswith(b"\n"):
                break
            latest = max(latest, int(json.loads(line)["event_id"]))
    return latest


def repair_partial_tail(path):
    """Only the locked writer repairs an interrupted final append on restart."""
    path = Path(path)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            complete_end = data.rfind(b"\n") + 1
            path.with_name(f"events.partial_tail.{time.time_ns()}").write_bytes(data[complete_end:])
            with path.open("r+b") as stream:
                stream.truncate(complete_end)


def transition_events(previous, observation, identity, next_id):
    """Emit initialization and state changes once; never one event per poll."""
    events = []
    for item in observation["devices"]:
        old = previous.get(item["index"])
        state = item["state"]
        if old == state:
            continue
        next_id += 1
        events.append(dict(event_id=next_id, event_type="gpu_state_changed",
                           previous_state=old, state=state,
                           idle_event=state == "idle", release_event=old == "busy" and state == "idle",
                           idle_event_kind=("initial_idle" if old is None else
                                            "released" if old == "busy" else "idle_after_unknown") if state == "idle" else None,
                           index=item["index"], gpu_uuid=item["gpu_uuid"],
                           memory_mib=item["memory_mib"], compute_process_count=item["compute_process_count"],
                           query_started_at=observation["query_started_at"],
                           query_started_at_utc=observation["query_started_at_utc"],
                           query_finished_at=observation["query_finished_at"],
                           query_finished_at_utc=observation["query_finished_at_utc"],
                           error=observation["error"], **identity))
    return events, next_id


def watch(output=OUTPUT, max_polls=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # CLI has one fixed output/lock path. Tests may use a temporary directory.
    with (output / "watcher.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(error="observer_already_running", receipt=str(output / "process_receipt.json"))), file=sys.stderr)
            return 4
        identity = dict(**process_identity(), run_id=str(uuid.uuid4()))
        started = time.time()
        receipt = dict(**identity, started_at=started, started_at_utc=utc(started),
                       output_dir=str(output), poll_seconds=POLL_SECONDS,
                       query_timeout_seconds=QUERY_TIMEOUT_SECONDS, memory_limit_mib=MEMORY_LIMIT_MIB,
                       mode="observe_only", expected_indices=sorted(EXPECTED_INDICES))
        atomic_json(output / "process_receipt.json", receipt)
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(receipt) + "\n")
        lock.flush()
        print(json.dumps(dict(event_type="observer_started", **receipt)), flush=True)
        repair_partial_tail(output / "events.jsonl")
        event_id = last_event_id(output / "events.jsonl")
        previous = {}
        polls = 0
        next_poll = time.monotonic()
        while max_polls is None or polls < max_polls:
            observation = scan()
            polls += 1
            events, event_id = transition_events(previous, observation, identity, event_id)
            if events:
                with (output / "events.jsonl").open("a") as stream:
                    for event in events:
                        line = json.dumps(event, allow_nan=False)
                        stream.write(line + "\n")
                        print(line, flush=True)
                    stream.flush()
            previous = {item["index"]: item["state"] for item in observation["devices"]}
            heartbeat = time.time()
            atomic_json(output / "status.json", dict(**identity, mode="observe_only", poll_seconds=POLL_SECONDS,
                        poll_count=polls, latest_event_id=event_id, heartbeat_at=heartbeat,
                        heartbeat_at_utc=utc(heartbeat), **observation))
            next_poll += POLL_SECONDS
            # Slow/error queries never create a burst of catch-up scans.
            now = time.monotonic()
            if next_poll < now:
                next_poll = now
            if max_polls is None or polls < max_polls:
                time.sleep(max(0.0, next_poll - now))
    return 0


def event_context(event, output, cursor_run_id=None):
    """Distinguish a real historical idle event from current observed availability."""
    now = time.time()
    context = dict(event_age_seconds=max(0.0, now-event["query_finished_at"]),
                   cursor_run_id=cursor_run_id,
                   event_matches_cursor_run=event.get("run_id") == cursor_run_id if cursor_run_id else None,
                   current_state="unknown", current_observation_fresh=False,
                   current_observation_includes_event=False,
                   current_observation_age_seconds=None, current_identity_matches_receipt=False,
                   watcher_alive=False, event_from_current_watcher=False)
    try:
        receipt = json.loads((Path(output) / "process_receipt.json").read_text())
        status = json.loads((Path(output) / "status.json").read_text())
        fields = Path(f"/proc/{receipt['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        command = Path(f"/proc/{receipt['pid']}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        alive = (int(fields[19]) == receipt["proc_start_ticks"] and fields[0] != "Z"
                 and command == receipt["command_line"])
        matches = all(status.get(key) == receipt.get(key) for key in
                      ("pid", "proc_start_ticks", "source_sha256", "run_id"))
        age = now - float(status["query_finished_at"])
        fresh = math.isfinite(age) and 0 <= age <= 3 * POLL_SECONDS
        includes_event = status["query_finished_at"] >= event["query_finished_at"]
        context.update(watcher_alive=alive, current_identity_matches_receipt=matches,
                       current_observation_age_seconds=age if math.isfinite(age) else None,
                       current_observation_fresh=fresh,
                       current_observation_includes_event=includes_event,
                       event_from_current_watcher=event.get("run_id") == receipt["run_id"],
                       active_run_id=receipt["run_id"],
                       current_query_started_at=status["query_started_at"],
                       current_query_finished_at=status["query_finished_at"])
        if alive and matches and fresh and includes_event:
            devices = [item for item in status["devices"] if item["index"] == event["index"]
                       and item["gpu_uuid"] == event["gpu_uuid"]]
            if len(devices) == 1:
                context["current_state"] = devices[0]["state"]
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        context["current_context_error"] = f"{type(exc).__name__}: {exc}"
    return dict(**event, **context)


def wait_for_idle_event(output=OUTPUT, after_event_id=None, timeout_seconds=0.0, after_run_id=None):
    """Tail only event records; never issue nvidia-smi or acquire the watcher lock."""
    path = Path(output) / "events.jsonl"
    if after_event_id is None:
        # Freeze the entry byte boundary: events appended while reading history
        # belong to this wait and must not be swallowed into its baseline.
        try:
            baseline_size = path.stat().st_size
        except FileNotFoundError:
            baseline_size = 0
        after_event_id = last_event_id(path, byte_limit=baseline_size)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None
    offset = 0
    while True:
        if path.exists():
            with path.open() as stream:
                stream.seek(offset)
                while True:
                    position = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith("\n"):
                        offset = position
                        break
                    offset = stream.tell()
                    event = json.loads(line)
                    if event["event_id"] > after_event_id and event.get("idle_event") is True:
                        print(json.dumps(event_context(event, output, after_run_id), allow_nan=False), flush=True)
                        return 0
        if deadline is not None and time.monotonic() >= deadline:
            return 3
        remaining = max(0.0, deadline-time.monotonic()) if deadline is not None else 0.1
        time.sleep(min(0.1, remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-idle-event", action="store_true")
    parser.add_argument("--after-event-id", type=int)
    parser.add_argument("--after-run-id", help="cursor run identity; output explicitly identifies cross-restart events")
    parser.add_argument("--timeout-seconds", type=float, default=0.0, help="wait timeout; 0 means indefinite")
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be finite and nonnegative")
    if args.after_event_id is not None and args.after_event_id < 0:
        parser.error("--after-event-id must be nonnegative")
    if not args.wait_for_idle_event and (args.after_event_id is not None or args.after_run_id is not None or args.timeout_seconds != 0):
        parser.error("event cursor and timeout apply only with --wait-for-idle-event")
    if args.wait_for_idle_event:
        return wait_for_idle_event(after_event_id=args.after_event_id, timeout_seconds=args.timeout_seconds,
                                   after_run_id=args.after_run_id)
    return watch()


if __name__ == "__main__":
    raise SystemExit(main())
