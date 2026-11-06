"""Keep real VRAM across frozen within-suite experiments and CPU gaps.

Only --launch starts a CUDA helper or scientific children. The original
scheduler sessions, GPU leases, and 64-MiB keepers remain untouched. The new
helper holds 16 GiB while idle; it retains 8 GiB during model loading and
execution so the 24-GiB card can also accommodate the model. Model memory is
additional; this is not a claim of a continuous 16-GiB floor during loading.
"""
import argparse
import json
import math
import os
from pathlib import Path
import select
import subprocess
import time
import uuid

from preparations.ease_insights_20261103.vlm_within_suite_v1 import borrowed_controller as bc
from preparations.ease_insights_20261103.reservoir_v1 import keeper as km

ROOT = bc.ROOT
RESOURCE_ROOT = bc.EXPERIMENT / "reservoir_v1"
MODULE = "preparations.ease_insights_20261103.reservoir_v1.keeper"
WATCH = bc.EXPERIMENT / "gpu_release_watch/status.json"
MANIFEST = "6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d"
FILE_SHA = "e9dc20f236a74da7e10493488ea644f0050501159f0c2e390e77c8f2ab653ddb"
WATCH_SHA = "4314b4b5d7eca4f88968b0d66da305a5cd2f6aff4829f363ae76b951b9495680"
HOST_KEEPERS = {"GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1": 1975204,
                "GPU-599477fb-0507-40dc-14df-d491f7a7b202": 1975205}
require, read, sha, write = bc.require, bc.read, bc.sha, bc.write


def cached_device(status, gpu_uuid, now):
    require(status.get("pid") == 385842 and str(status.get("proc_start_ticks")) == "2618445988"
            and status.get("run_id") == "060b0172-ba52-4c9b-9c39-03c7fb604d69"
            and status.get("source_sha256") == WATCH_SHA and status.get("poll_seconds") == 2
            and status.get("error") is None, "Cached GPU observer identity changed")
    query = status.get("query_finished_at")
    require(type(query) in (int, float) and math.isfinite(query) and 0 <= now - query <= 3,
            "Cached GPU observation is stale")
    devices = [r for r in status.get("devices", []) if r.get("gpu_uuid") == gpu_uuid]
    require(len(devices) == 1, "Assigned UUID is missing or duplicated in observer cache")
    return query, devices[0]


class IdleWindow:
    def __init__(self):
        self.samples = []

    def update(self, query, row, gpu_uuid):
        available = (row.get("host_compute_pids") == [HOST_KEEPERS[gpu_uuid]]
                     and row.get("compute_process_count") == 1
                     and type(row.get("memory_mib")) in (int, float)
                     and 0 < row["memory_mib"] <= 512)
        if (not available or (self.samples and
                (query < self.samples[-1]["query_finished_at"]
                 or query - self.samples[-1]["query_finished_at"] > 3))):
            self.samples.clear()
        if available and (not self.samples or query != self.samples[-1]["query_finished_at"]):
            self.samples.append(dict(query_finished_at=query, device=row))
        elapsed = query - self.samples[0]["query_finished_at"] if self.samples else 0
        return len(self.samples) >= 9 and elapsed >= 15, elapsed


def verify_amendment(path, expected):
    require(sha(path) == expected, "Resource amendment changed")
    value = read(path)
    require(value.get("schema") == "ease-reservoir-resource-amendment-v1"
            and value.get("manifest_hash") == MANIFEST and value.get("manifest_file_sha256") == FILE_SHA
            and value.get("idle_reservation_mib") == km.IDLE_MIB
            and value.get("running_reservation_mib") == km.RUNNING_MIB
            and value.get("policy_seconds") == [15, 2]
            and value.get("scientific_sources_changed") is False, "Resource amendment identity differs")
    require(value.get("source_hashes"), "Resource amendment has no source pins")
    for name, digest in value["source_hashes"].items():
        candidate = ROOT / name
        require(candidate.resolve().is_relative_to(ROOT) and sha(candidate) == digest,
                "Pinned resource source changed: " + name)
    discovery = bc.NEW_ROOT / "discovery/manifest.json"
    require(sha(discovery) == FILE_SHA and read(discovery).get("manifest_hash") == MANIFEST,
            "Frozen scientific manifest changed")
    return value


class Client:
    def __init__(self, gpu_uuid, output, runtime):
        self.gpu_uuid, self.output, self.runtime = gpu_uuid, output, runtime
        self.child = None
        self.writer = self.reader = self.log = None
        self.nonce = uuid.uuid4().hex
        self.sequence = 0
        self.pending = b""
        self.identity = self.ready = None

    def start(self):
        control_read, self.writer = os.pipe()
        self.reader, ready_write = os.pipe()
        own = self.runtime.self_identity()
        argv = ["/usr/bin/python3", "-B", "-u", "-m", MODULE,
                "--gpu-uuid", self.gpu_uuid, "--control-fd", str(control_read),
                "--ready-fd", str(ready_write), "--nonce", self.nonce,
                "--parent-pid", str(own["pid"]), "--parent-start-ticks", own["start_ticks"]]
        self.log = (self.output / "reservoir.log").open("a")
        try:
            self.child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT, close_fds=True,
                pass_fds=(control_read, ready_write))
        finally:
            os.close(control_read)
            os.close(ready_write)
        self.identity = self.runtime.identity(self.child.pid)
        self.ready = self.receive("ready", km.IDLE_MIB)
        write(self.output / "keeper_ready.json", self.ready)
        return self.ready

    def live(self):
        return (self.child is not None and self.child.poll() is None
                and bc.legacy.identity.same_process(self.runtime.identity(self.child.pid), self.identity))

    def receive(self, phase, mib):
        deadline = time.monotonic() + 30
        while b"\n" not in self.pending:
            require(self.live(), "Reservoir child died or changed identity")
            require(time.monotonic() < deadline, "Reservoir acknowledgement timed out")
            readable, _, _ = select.select([self.reader], [], [], 0.2)
            if readable:
                block = os.read(self.reader, 4096)
                require(block, "Reservoir acknowledgement pipe closed")
                self.pending += block
                require(len(self.pending) <= 65536, "Reservoir acknowledgement too large")
        line, self.pending = self.pending.split(b"\n", 1)
        value = json.loads(line)
        require(value.get("schema") == km.SCHEMA and value.get("gpu_uuid") == self.gpu_uuid
                and value.get("pid") == self.child.pid
                and str(value.get("proc_start_ticks")) == self.identity.get("start_ticks")
                and value.get("nonce") == self.nonce and value.get("sequence") == self.sequence,
                "Reservoir acknowledgement identity differs")
        write(self.output / "keeper_latest_ack.json", value)
        require(value.get("phase") == phase and value.get("context_created") is True
                and value.get("allocated_bytes") == mib * km.base.MIB,
                "Reservoir target not achieved: " + str(value))
        self.ready = value
        return value

    def resize(self, mib):
        km.target_bytes(mib)
        require(self.live(), "Cannot resize a lost reservoir")
        self.sequence += 1
        payload = (json.dumps(dict(schema=km.SCHEMA, op="resize", nonce=self.nonce,
                                  sequence=self.sequence, target_mib=mib)) + "\n").encode()
        require(os.write(self.writer, payload) == len(payload), "Incomplete reservoir control write")
        return self.receive("resized", mib)

    def close(self):
        # Only invoked on explicit guardian shutdown, never on task completion.
        if self.writer is not None:
            os.close(self.writer)
            self.writer = None
        if self.child is not None:
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.runtime.cleanup_child(self.child)
        if self.reader is not None:
            os.close(self.reader)
            self.reader = None
        if self.log is not None:
            self.log.close()


def owned_observation(gpu_uuid, rows, bulk_host_pid, retained_mib):
    expected = {HOST_KEEPERS[gpu_uuid], bulk_host_pid}
    require(len(rows) == 2 and {r[0] for r in rows} == expected,
            "Assigned GPU has additional or missing contexts")
    memory = dict(rows)
    require(all(math.isfinite(v) and v > 0 for v in memory.values())
            and memory[HOST_KEEPERS[gpu_uuid]] <= 512
            and retained_mib <= memory[bulk_host_pid] <= retained_mib + 512,
            "Assigned reservoir memory/sole-context consistency failed")
    return dict(gpu_uuid=gpu_uuid, host_process_memory_mib=memory,
                original_keeper_host_pid=HOST_KEEPERS[gpu_uuid], reservoir_host_pid=bulk_host_pid,
                retained_allocation_mib=retained_mib,
                attribution="live_nonce_bound_helper_plus_unique_additional_sized_context_consistency",
                pid_namespace_mapping="unique_additional_context_inference_not_direct_host_PID_mapping")


class Runtime(bc.BorrowedRuntime):
    def __init__(self, guard, gpu_uuid, client, bulk_host_pid):
        super().__init__(guard, gpu_uuid)
        self.client, self.bulk_host_pid = client, bulk_host_pid

    def scan_owned(self, gpu_uuid, keeper, maximum):
        require(self.keeper_live(keeper) and self.client.live(), "An owned keeper is no longer live")
        raw = subprocess.check_output(["nvidia-smi", "--id=" + gpu_uuid,
            "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        rows = []
        for line in raw.splitlines():
            if line.strip():
                actual_uuid, pid, used = [part.strip() for part in line.split(",")]
                require(actual_uuid == gpu_uuid, "Assigned process query returned another UUID")
                rows.append((int(pid), float(used)))
        retained_mib = self.client.ready["allocated_bytes"] // km.base.MIB
        return owned_observation(gpu_uuid, rows, self.bulk_host_pid, retained_mib)


class Session(bc.BorrowedSession):
    def __init__(self, guard, gpu_uuid, port, manifest, client, bulk_host_pid, amendment):
        super().__init__(guard, gpu_uuid, port, manifest)
        self.runtime = Runtime(guard, gpu_uuid, client, bulk_host_pid)
        self.amendment, self.client = amendment, client

    def check_keeper(self):
        super().check_keeper()
        require(self.client.live(), "Reservoir child no longer matches its owned identity")

    def execution_revision(self, observation):
        original = super().execution_revision(observation)
        original.update(schema="ease-reservoir-science-resource-v1",
            controller_source_sha256=sha(__file__), reused_borrowed_controller_sha256=sha(bc.__file__),
            resource_amendment=self.amendment, reservoir_keeper=self.client.ready,
            scientific_runtime_unchanged=True,
            scope="Frozen scientific commands with separate persistent resizable VRAM reservoir")
        return original


def marker_valid(path, manifest):
    if not path.exists():
        return False
    value = read(path)
    require(value.get("complete") is True and value.get("manifest_hash") == manifest["manifest_hash"]
            and value.get("condition") == path.parent.name
            and value.get("completed_after", value.get("already_completed")) == manifest["cases_per_condition"],
            "Existing scientific completion marker differs")
    return True


def execute_condition(prepared, session, science, client):
    """Retain memory through all model and worker cleanup, then restore idle size."""
    client.resize(km.RUNNING_MIB)
    try:
        return science.execute(prepared, session)
    finally:
        session.runtime.reap_scientific_children()
        client.resize(km.IDLE_MIB)


def serve_request(path, args, guard, client, bulk_host_pid, amendment, output):
    from preparations.ease_insights_20261103.vlm_within_suite_v1 import common, runner, worker
    value = read(path)
    stage_root = Path(value["output"]).resolve()
    require(value.get("schema") == "ease-reservoir-science-request-v1"
            and stage_root.parent == bc.NEW_ROOT and stage_root.name in ("discovery", "confirmation")
            and sha(stage_root / "manifest.json") == value.get("manifest_file_sha256"),
            "Scientific request destination or raw manifest differs")
    manifest = common.read_manifest(stage_root / "manifest.json", verify_sources=True)
    require(manifest["manifest_hash"] == value.get("manifest_hash"), "Scientific request manifest differs")
    stage = common.stage_for(manifest)
    results = []
    for cid in manifest["planned_conditions"]:
        with bc.lock(stage_root / "claims" / (cid + ".lock"), optional=True) as acquired:
            if not acquired:
                continue
            marker = stage_root / cid / (stage + "_shard00of01_complete.json")
            if marker_valid(marker, manifest):
                continue
            verify_amendment(args.amendment, args.amendment_sha256)
            guard.full()
            port = bc.fresh_port()
            session = Session(guard, args.gpu_uuid, port, manifest, client, bulk_host_pid, amendment)
            # Before shrinking, reject any unrecognized context. No model or
            # scientific run directory exists at this resource preflight.
            session.runtime.scan_owned(args.gpu_uuid, session.keeper, 512)
            opts = argparse.Namespace(output=str(stage_root), condition=cid, stage=stage,
                shard_index=0, shards=1, gpu_uuid=args.gpu_uuid, port=port,
                launch=False, preflight_only=False)
            prepared = runner.run(opts)
            write(output / "status.json", dict(phase="running_condition", condition=cid,
                time=time.time(), gpu_uuid=args.gpu_uuid, controller_pid=os.getpid(),
                manifest_hash=manifest["manifest_hash"], idle_reservation_mib=km.IDLE_MIB,
                running_reservation_mib=km.RUNNING_MIB))
            result = execute_condition(prepared, session, bc.legacy.Science(runner=runner, worker=worker), client)
            results.append(dict(condition=cid, result=result))
    return dict(phase="assignment_complete", results=results, time=time.time(),
                manifest_hash=manifest["manifest_hash"], holder_resources_retained=True)


def run(args):
    amendment_value = verify_amendment(args.amendment, args.amendment_sha256)
    require(args.gpu_uuid in HOST_KEEPERS, "Only the two original held UUIDs are assigned")
    guard = bc.HolderGuard()
    if not args.launch:
        return dict(mode="dry_run", amendment=amendment_value, holder_proof=guard.full(), no_cuda_started=True)
    output = RESOURCE_ROOT / "sessions" / args.gpu_uuid
    output.mkdir(parents=True, exist_ok=True)
    client = Client(args.gpu_uuid, output, bc.legacy.Runtime())
    with bc.lock(bc.CAMPAIGN / "broker.lock", shared=True), bc.lock(
            bc.NEW_ROOT / "resources" / (args.gpu_uuid + ".borrower.lock")):
        guard.full()
        own = bc.legacy.identity.LocalRuntime().identity(os.getpid())
        write(output / "guardian.json", dict(own, schema="ease-reservoir-guardian-v1",
            amendment_path=str(args.amendment), amendment_sha256=args.amendment_sha256,
            source_sha256=sha(__file__), created=time.time(), gpu_uuid=args.gpu_uuid))
        window = IdleWindow()
        while True:
            if (output / "release").exists():
                return dict(phase="explicit_release_before_allocation", gpu_uuid=args.gpu_uuid)
            guard.quick(args.gpu_uuid)
            try:
                query, device = cached_device(read(WATCH), args.gpu_uuid, time.time())
                ready, elapsed = window.update(query, device, args.gpu_uuid)
            except (OSError, ValueError, RuntimeError):
                window.samples.clear()
                ready, elapsed, device = False, 0, {}
            write(output / "status.json", dict(phase="waiting_original_owner_available", time=time.time(),
                available_seconds=elapsed, device=device, policy_seconds=[15, 2], target_mib=km.IDLE_MIB))
            if ready:
                verify_amendment(args.amendment, args.amendment_sha256)
                attempt = str(time.time_ns())
                write(output / "allocation_attempts" / (attempt + "_intent.json"),
                    dict(time=time.time(), samples=window.samples, target_mib=km.IDLE_MIB,
                         guardian_pid=os.getpid(), scientific_children_started=False))
                try:
                    client.start()
                    break
                except Exception as error:
                    # Another allocator may win the race after the cached
                    # window. This is not a scientific attempt: close only
                    # this helper, and require a new complete idle window.
                    client.close()
                    write(output / "allocation_attempts" / (attempt + "_failed.json"),
                        dict(time=time.time(), error=repr(error), scientific_children_started=False))
                    client = Client(args.gpu_uuid, output, bc.legacy.Runtime())
                    window.samples.clear()
            time.sleep(2)
        try:
            # Bind the unique new sized context using fresh cached observation.
            # This is an explicit consistency inference across PID namespaces.
            bulk_host_pid = None
            while bulk_host_pid is None:
                guard.quick(args.gpu_uuid)
                require(client.live(), "Reservoir lost before context binding")
                try:
                    query, device = cached_device(read(WATCH), args.gpu_uuid, time.time())
                except (OSError, ValueError, RuntimeError):
                    time.sleep(2)
                    continue
                extra = set(device.get("host_compute_pids", [])) - {HOST_KEEPERS[args.gpu_uuid]}
                if device.get("compute_process_count") == 2 and len(extra) == 1:
                    candidate = next(iter(extra))
                    probe = Session(guard, args.gpu_uuid, 1, {"manifest_hash": MANIFEST}, client,
                                    candidate, dict(path=str(args.amendment), sha256=args.amendment_sha256))
                    try:
                        observation = probe.runtime.scan_owned(args.gpu_uuid, probe.keeper, 512)
                        bulk_host_pid = candidate
                        write(output / "context_binding.json", dict(observation, keeper=client.ready,
                            time=time.time(), guardian=own))
                    except RuntimeError:
                        pass
                if bulk_host_pid is None:
                    write(output / "status.json", dict(phase="holding_16GiB_waiting_context_consistency",
                        time=time.time(), reservoir=client.ready, device=device))
                    time.sleep(1)
            attention = False
            while not (output / "release").exists():
                guard.quick(args.gpu_uuid)
                require(client.live(), "Persistent reservoir died")
                if not attention:
                    for request in sorted((RESOURCE_ROOT / "requests").glob("*.json")):
                        outcome = output / "outcomes" / request.name
                        if outcome.exists():
                            continue
                        amendment = dict(path=str(args.amendment), sha256=args.amendment_sha256)
                        try:
                            result = serve_request(request, args, guard, client, bulk_host_pid, amendment, output)
                        except Exception as error:
                            result = dict(phase="attention", error=repr(error), time=time.time(),
                                automatic_scientific_retry=False, reservoir_retained=client.live())
                            attention = True
                        write(outcome, dict(result, request_path=str(request), request_sha256=sha(request)))
                        if attention:
                            break
                write(output / "status.json", dict(phase="attention_holding" if attention else "idle_holding_16GiB",
                    time=time.time(), guardian_pid=os.getpid(), gpu_uuid=args.gpu_uuid,
                    reservoir=client.ready, original_holders_retained=True))
                time.sleep(2)
            return dict(phase="explicit_release", gpu_uuid=args.gpu_uuid)
        finally:
            client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--amendment-sha256", required=True)
    parser.add_argument("--launch", action="store_true")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
