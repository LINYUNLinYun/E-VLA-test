"""Matched old-KV AB/BA experiments and separate continuous-request replay.

AB/BA bootstraps each arm from the same observation outside timing. Graphs
survive reset, so only the two existing KV slots are ever retained. Continuous
replay deliberately does not wait for background completion between requests.
The optional policy freshness gate can wait for the previous Prefix. A final
owned-stream drain is timed separately from response wall and Event statistics.
"""
import argparse
import contextlib
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shlex
import sys
import time
import traceback

import numpy as np
import torch

from . import policy as policy_module
from .experiments import event, load_backend, native_raw
from .graph_bench import cache_digest
from .openpi_backend import cache_tensors
from .policy import PipelinePolicy
from .scheduler import Observation
from .scheduling_bench import span
from .submission import submit
from .layer_bench import gpu_snapshot, require_idle_gpu, require_same_processes


MODES = {
    "eager_serial": dict(mode="serial_reuse"),
    "eager_dual_ae_first": dict(submission_order="ae_first"),
    "eager_dual_prefill_first": dict(submission_order="prefill_first"),
    "eager_fresh_serial": dict(mode="fresh_serial"),
    "graph_serial": dict(cuda_graphs=True, mode="serial_reuse"),
    "graph_serial_high": dict(cuda_graphs=True, mode="serial_reuse", ae_priority=-1),
    "graph_ae_first": dict(cuda_graphs=True, submission_order="ae_first"),
    "graph_encoder_overlap": dict(cuda_graphs=True, submission_order="ae_first"),
    "graph_prefill_first": dict(cuda_graphs=True, submission_order="prefill_first"),
    "graph_ae_high": dict(cuda_graphs=True, submission_order="ae_first", ae_priority=-1),
    "graph_pre_high": dict(cuda_graphs=True, submission_order="prefill_first", ae_priority=-1),
    "graph_fresh_serial": dict(cuda_graphs=True, mode="fresh_serial"),
}


def settings(name):
    return dict(mode="async_reuse", interval=1, submission_order="ae_first",
                cuda_graphs=False, ae_priority=0, **{}) | MODES[name]


def statistics(values):
    values = [x for x in values if x is not None]
    if not values:
        return None
    return dict(n=len(values), mean=float(np.mean(values)),
                p50=float(np.percentile(values, 50)), p95=float(np.percentile(values, 95)))


def differences(reference, actual):
    delta = np.asarray(reference, dtype=np.float64)-np.asarray(actual, dtype=np.float64)
    return dict(mae=float(np.abs(delta).mean()), max_abs=float(np.abs(delta).max()))


def arm_order(names, pair_index, reverse=False):
    return names if (pair_index + int(reverse)) % 2 == 0 else names[::-1]


def selective_kind(name, configured_kind):
    if name == "graph_encoder_overlap":
        if configured_kind != "all":
            raise ValueError("graph_encoder_overlap requires --selective all")
        return "vision_only"
    return configured_kind


def switch_selective_after_reset(policy, kind):
    """Change only replay waits, after the existing Pool drain/reset finished."""
    if kind not in ("all", "vision_only"):
        raise ValueError("Same-Graph comparison supports all/vision_only only")
    pool, graphs = policy.pool, policy.graphs
    if (policy.request != 0 or pool.active is not None or pool.pending is not None
            or any(slot.state != "FREE" or slot.leases or slot.readers or slot.ready is not None
                   for slot in pool.slots)):
        raise RuntimeError("Selective replay kind may change only after Pool drain/reset")
    if graphs is not None:
        if (getattr(graphs, "kind", None) not in ("all", "vision_only")
                or graphs.ae_pending or graphs.vision_pending):
            raise RuntimeError("Selective Graph events must be reset before changing replay kind")
        graphs.kind = kind
    policy.backend.selective_overlap = kind


def configure_arm(policy, name, configured_kind, *, mixed_selective=False):
    # configure already drains the Pool and clears selective pending Events.
    # Keep this call unchanged for historical (single --selective) modes.
    previous_graphs = policy.graphs
    policy.configure(**settings(name))
    if mixed_selective:
        if policy.graphs is not previous_graphs:
            raise RuntimeError("Selective comparison must preserve the same Graph owner")
        switch_selective_after_reset(policy, selective_kind(name, configured_kind))


def freshness_gate(rows, enabled):
    """The optional old-KV contract applies to measured requests, not modules."""
    if not enabled:
        return dict(enabled=False, applicable=False, passed=None)
    selected = [row for row in rows if row["phase"] in ("paired", "continuous")]
    violations = []
    for index, row in enumerate(selected):
        failures = []
        if row.get("prefill_submissions") != 1:
            failures.append("prefill_submissions_must_equal_one")
        if row.get("observation_age") != 1:
            failures.append("old_snapshot_age_must_equal_one")
        if row.get("steady_recapture") != 0:
            failures.append("steady_graph_recapture_must_equal_zero")
        if failures:
            violations.append(dict(row=index, request=row.get("request"), mode=row.get("mode"),
                                   failures=failures))
    return dict(enabled=True, applicable=bool(selected), passed=not violations,
                measured_requests=len(selected), violations=violations,
                scope="Measured paired/continuous requests only; warmup, bootstrap and isolated modules excluded.")


def drain_continuous(policy, rows, *, now=time.perf_counter):
    """Freeze all-work wall before Event-statistics collection or Pool polling."""
    selected = [row for row in rows if row["phase"] == "continuous"]
    if not selected:
        return None
    response_start, response_end = selected[0]["host_begin"], selected[-1]["host_end"]
    drain_begin = now()
    # Every response has already completed its AE/output. These owned stream
    # fences also cover the final background Prefill; no whole-device fence.
    policy.background.synchronize()
    policy.ae_stream.synchronize()
    all_work_end = now()
    response_wall, all_work_wall = response_end - response_start, all_work_end - response_start
    prefills = sum(row["prefill_submissions"] for row in selected)
    return dict(response_wall_seconds=response_wall, response_hz=len(selected) / response_wall,
                all_work_wall_seconds=all_work_wall, all_work_hz=len(selected) / all_work_wall,
                drain_wall_seconds=all_work_end - drain_begin,
                response_to_drain_begin_seconds=drain_begin - response_end,
                all_work_end_monotonic_s=all_work_end,
                completed_measured_prefills=prefills,
                completed_measured_prefills_per_second=prefills / all_work_wall,
                scope="First measured request through final owned-stream drain. Intervening reset/bootstrap included; their Prefills excluded from measured numerator. Event-statistics collection and cleanup excluded. Drain is Host fence wall, not kernel busy time.")


class Recorder:
    """Retain Events and CPU outputs, never extra GPU KV snapshots."""
    def __init__(self, backend):
        self.backend = backend
        self.rows = []
        self.pending = []
        self.history = {}
        self.addresses = None
        self.engine_id = None
        self.capture_engines = 0

    @contextlib.contextmanager
    def intercept(self):
        original_submit = policy_module.submit
        original_output = self.backend.output
        self.jobs = []
        self.normalized = None
        self.output_begin = None
        def observed_submit(*args, **kwargs):
            value, job = original_submit(*args, **kwargs)
            self.jobs.append((args[1], job))
            return value, job
        def observed_output(prepared, raw):
            self.output_begin = time.perf_counter()
            # Preserve the original two GPU->CPU transfers and transformation.
            state = prepared.state[0].detach().cpu().numpy()
            actions = raw[0].detach().cpu().numpy()
            self.normalized = actions.copy()
            return self.backend.policy._output_transform(dict(state=state, actions=actions))["actions"]
        policy_module.submit = observed_submit
        self.backend.output = observed_output
        try:
            yield
        finally:
            policy_module.submit = original_submit
            self.backend.output = original_output

    def predict(self, policy, records, index, seed, *, mode, phase, pair=None,
                arm_order=None, measured=True):
        record = records[index % len(records)]
        origin = event()
        origin.record()
        before = time.perf_counter()
        with self.intercept(), span(f"cewp/{phase}/{mode}/{pair}/{index}"):
            response = policy.predict(record["obs"], record["instruction"], "libero_spatial", seed)
        after = time.perf_counter()
        assert response["pool"]["live_snapshots"] <= 2
        epoch = response["snapshot"]["epoch"]
        self.history[epoch, response["request"]] = index % len(records)
        source = self.history[epoch, response["prefix_observation_id"]]
        if policy.graphs is not None:
            addresses = [tuple(x.data_ptr() for x in cache_tensors(s.prefix.cache)) for s in policy.graphs.slots]
            assert not set(addresses[0]).intersection(addresses[1])
            if self.engine_id != id(policy.graphs):
                self.engine_id = id(policy.graphs)
                self.capture_engines += 1
                self.addresses = addresses
            assert addresses == self.addresses
        if not measured:
            return response
        row = dict(mode=mode, phase=phase, pair=pair, arm_order=arm_order,
                   selective_kind=getattr(policy.graphs, "kind", None),
                   index=index % len(records), source_index=source, seed=seed,
                   request=response["request"], epoch=epoch,
                   observation_age=response["observation_request_age"],
                   snapshot=response["snapshot"], cpu_cpu_ms=(after-before)*1000,
                   host_begin=before,host_end=after,
                   ae_output_ms=response["timing"]["action_ready_host_ms"],
                   policy_response_ms=response["timing"]["response_host_ms"],
                   previous_prefix_wait_ms=response["timing"].get("previous_prefix_wait_ms",0),
                   graph_setup_ms=response["graph_setup_ms"],
                   steady_recapture=int(response["graph_setup_ms"] > 0),
                   pool=response["pool"], diagnostics=response["diagnostics"],
                   peak_allocated=response["peak_allocated"], peak_reserved=response["peak_reserved"],
                   source_age_at_response_ms=response["source_receipt_age_at_response_ms"],
                   normalized=self.normalized, actions=response["actions"],
                   kv_addresses=[list(x) for x in self.addresses] if policy.graphs else None)
        submit_end = max((j.cpu_end for _, j in self.jobs if j.cpu_end < self.output_begin), default=self.output_begin)
        row["ae_wait_after_submit_ms"] = (self.output_begin-submit_end)*1000
        row["prefill_submissions"] = sum(label == "prefill" for label, _ in self.jobs)
        row["prefill_done_at_response"] = all(j.done.query() for label, j in self.jobs if label == "prefill")
        self.rows.append(row)
        self.pending.append((row, origin, self.jobs))
        return response

    def finish_events(self):
        # Called only after the measured trajectory; does not serialize replay.
        for row, origin, jobs in self.pending:
            ends = []
            for label, job in jobs:
                job.done.synchronize()
                completion = origin.elapsed_time(job.done)
                ends.append(completion)
                row[label+"_completion_event_ms"] = completion
                row[label+"_gpu_event_ms"] = job.start.elapsed_time(job.done)
                row[label+"_cpu_submit_ms"] = (job.cpu_end-job.cpu_begin)*1000
                row[label+"_done_at_submit"] = job.done_at_submit
            row["both_completion_event_ms"] = max(ends, default=0)
            row["completed_prefill_count"] = sum(label == "prefill" for label, _ in jobs)
        self.pending.clear()

    def isolated(self, policy, records, index, seed, mode, module):
        record=records[index % len(records)]
        origin=event()
        origin.record()
        begin=time.perf_counter()
        prepared=self.backend.prepare(native_raw(record))
        noise=self.backend.noise(seed)
        ready=event()
        ready.record()
        phase="isolated_"+module
        with span(f"cewp/{phase}/{mode}/{index}"):
            if module=="ae":
                lease=policy.pool.acquire()
                source=self.history[lease.identity.epoch,lease.observation.observation_id]
                snapshot=dict(epoch=lease.identity.epoch,slot=lease.identity.slot,generation=lease.identity.generation)
                function,args=((policy.graphs.generate,(policy.pool,lease,prepared,noise)) if policy.graphs else
                               (self.backend.generate,(lease.value,prepared,noise)))
                raw,job=submit(policy.ae_stream,"ae",function,*args,dependency=ready)
                wait_begin=time.perf_counter()
                job.done.synchronize()
                wait_ms=(time.perf_counter()-wait_begin)*1000
                with self.intercept():
                    actions=self.backend.output(prepared,raw)
                policy.pool.release(lease,job.done)
                del lease,raw
            else:
                observation=Observation(policy.request,policy.pool.epoch,policy.pool.task,prepared)
                self.history[policy.pool.epoch,policy.request]=index % len(records)
                policy.pool.offer(observation)
                refresh=policy._refresh(ready)
                if refresh is None:
                    raise RuntimeError("Isolated Prefill had no free slot")
                job=refresh["submission"]
                wait_begin=time.perf_counter()
                job.done.synchronize()
                wait_ms=(time.perf_counter()-wait_begin)*1000
                policy.pool.poll()
                source=index % len(records)
                identity=policy.pool.active
                snapshot=dict(epoch=identity.epoch,slot=identity.slot,generation=identity.generation)
        end=time.perf_counter()
        row=dict(mode=mode,phase=phase,pair=None,arm_order=None,index=index % len(records),
                 selective_kind=getattr(policy.graphs, "kind", None),
                 source_index=source,seed=seed,request=policy.request,epoch=policy.pool.epoch,
                 observation_age=0 if module=="prefill" else policy.request-policy.pool._slot(policy.pool.active).observation.observation_id,
                 snapshot=snapshot,cpu_cpu_ms=(end-begin)*1000,ae_output_ms=(end-begin)*1000 if module=="ae" else None,
                 graph_setup_ms=0,steady_recapture=0,pool=dict(live_snapshots=policy.pool.live_snapshots),
                 peak_allocated=torch.cuda.max_memory_allocated(self.backend.device),
                 peak_reserved=torch.cuda.max_memory_reserved(self.backend.device),
                 ae_wait_after_submit_ms=wait_ms if module=="ae" else None,
                 source_age_at_response_ms=None,prefill_done_at_response=module=="prefill")
        row["prefill_submissions"] = int(module == "prefill")
        if module=="ae":
            row.update(normalized=self.normalized,actions=actions)
        else:
            row["prefix_digest"]=cache_digest(policy.pool._slot(policy.pool.active).value)
        self.rows.append(row)
        self.pending.append((row,origin,[(module,job)]))
        policy.request+=1

    def verify(self, backend, records, max_error, mae_error):
        """Caller has closed both graph slots before allocating reference KV."""
        from preparations.idea1_v2_1.runtime_client import input_digest
        prefix, old_index = None, None
        # Reuse only CPU oracle outputs across independent scheduling arms.
        # Exact observation digests, D and noise seed key every entry. No GPU
        # Tensor, KV or Graph survives through this cache, and the immutable
        # weights belong to this one backend instance throughout the experiment.
        if not hasattr(backend, "_cewp_cpu_oracles"):
            backend._cewp_cpu_oracles = {}
        reference_cache = backend._cewp_cpu_oracles
        digests = [input_digest(r["obs"], r["instruction"]) for r in records]
        for row in self.rows:
            if "prefix_digest" in row:
                del prefix
                prefix=backend.build_prefix(backend.prepare(native_raw(records[row["index"]])))
                row["correct"]=cache_digest(prefix)==row["prefix_digest"]
                old_index=None
                continue
            key = (backend.num_steps, digests[row["source_index"]], digests[row["index"]], row["seed"])
            if key not in reference_cache:
                if old_index != row["source_index"]:
                    del prefix
                    prefix = backend.build_prefix(backend.prepare(native_raw(records[row["source_index"]])))
                    old_index = row["source_index"]
                    before = cache_digest(prefix)
                prepared = backend.prepare(native_raw(records[row["index"]]))
                raw = backend.generate(prefix, prepared, backend.noise(row["seed"]))
                reference_cache[key] = (raw[0].cpu().numpy().copy(), backend.output(prepared, raw))
                assert cache_digest(prefix) == before, "Reference AE mutated old KV"
                del raw
            normalized, decoded = reference_cache[key]
            row["normalized_error"] = differences(normalized, row.pop("normalized"))
            row["decoded_error"] = differences(decoded, row.pop("actions"))
            row["correct"] = all(x["max_abs"] <= max_error and x["mae"] <= mae_error
                                 for x in (row["normalized_error"], row["decoded_error"]))
        del prefix
        return all(r["correct"] for r in self.rows)


def summarize(rows):
    result = {}
    for phase, mode in sorted({(r["phase"], r["mode"]) for r in rows}):
        selected = [r for r in rows if (r["phase"], r["mode"]) == (phase, mode)]
        metrics = ["cpu_cpu_ms", "ae_output_ms", "prefill_completion_event_ms",
                   "both_completion_event_ms", "ae_wait_after_submit_ms", "ae_gpu_event_ms",
                   "prefill_gpu_event_ms", "ae_cpu_submit_ms", "prefill_cpu_submit_ms",
                   "source_age_at_response_ms"]
        result[phase+"/"+mode] = {k: statistics([r.get(k) for r in selected]) for k in metrics}
        result[phase+"/"+mode].update(
            selective_kinds=sorted({r["selective_kind"] for r in selected if r.get("selective_kind") is not None}),
            observation_ages=statistics([r["observation_age"] for r in selected]),
            consumed_snapshots=len({(r["epoch"],r["snapshot"]["slot"],r["snapshot"]["generation"]) for r in selected}),
            prefill_submissions=sum(r.get("prefill_submissions", int("prefill_gpu_event_ms" in r)) for r in selected),
            prefill_done_at_response=sum(r["prefill_done_at_response"] for r in selected),
            peak_allocated=max(r["peak_allocated"] for r in selected),
            peak_reserved=max(r["peak_reserved"] for r in selected),
            steady_recaptures=sum(r["steady_recapture"] for r in selected),
            correct=all(r.get("correct",False) for r in selected))
        if phase=="continuous":
            wall_seconds=selected[-1]["host_end"]-selected[0]["host_begin"]
            result[phase+"/"+mode]["trajectory_wall_seconds"]=wall_seconds
            result[phase+"/"+mode]["prefill_submissions_per_second"]=sum("prefill_gpu_event_ms" in r for r in selected)/wall_seconds
            result[phase+"/"+mode]["consumed_snapshot_fraction"]=result[phase+"/"+mode]["consumed_snapshots"]/len(selected)
            result[phase+"/"+mode]["frequency_scope"]="Measured requests only; trajectory wall includes intervening reset/bootstrap, whose Prefills are excluded from numerator"
    return result


def write_results(dest, result):
    result["summary"] = summarize(result["rows"])
    for group in result["groups"]:
        if group["phase"] == "continuous" and group.get("trajectory") is not None:
            key = "continuous/" + group["modes"][0]
            if key in result["summary"]:
                result["summary"][key].update(group["trajectory"])
    result["paired_speedups"]={}
    for group_index,group in enumerate(result["groups"]):
        if group["phase"]!="paired":
            continue
        first,second=group["modes"]
        paired={}
        for row in result["rows"]:
            if row.get("group")==group_index:
                paired.setdefault(row["pair"],{})[row["mode"]]=row
        ratios=[arms[first]["cpu_cpu_ms"]/arms[second]["cpu_cpu_ms"] for arms in paired.values()
                if first in arms and second in arms]
        result["paired_speedups"][f"{group_index}:{first}/{second}"]=statistics(ratios)
    temporary = dest/"result.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, default=lambda x: x.tolist() if hasattr(x,"tolist") else str(x)))
    temporary.replace(dest/"result.json")
    flat=[]
    for row in result["rows"]:
        item={k:v for k,v in row.items() if not isinstance(v,(dict,list,np.ndarray))}
        for key in ("normalized_error","decoded_error"):
            item.update({key+"_"+k:v for k,v in row.get(key,{}).items()})
        flat.append(item)
    if flat:
        with (dest/"measurements.csv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=sorted(set().union(*(r.keys() for r in flat))))
            writer.writeheader()
            writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu-uuid", help="Optional strict exclusive GPU guard, including graphics processes")
    parser.add_argument("--model-config", default="pi05_libero")
    parser.add_argument("--steps", type=int, choices=[1,2,5,10], default=10)
    parser.add_argument("--noise-period", type=int, default=0,
                        help="Optional benchmark seed cycle; 0 preserves historical per-request seeds")
    parser.add_argument("--kind", choices=["requests","isolated"], default="requests")
    parser.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--pairs", nargs="*", default=["graph_serial:graph_ae_first"])
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--reverse-order", action="store_true", help="Reverse the first AB/BA arm for an independent repeat")
    parser.add_argument("--continuous", nargs="*", choices=list(MODES), default=[])
    parser.add_argument("--continuous-count", type=int, default=100)
    parser.add_argument("--reset-every", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--hoist-schedule", action="store_true")
    parser.add_argument("--hoist-modulation", action="store_true")
    parser.add_argument("--await-previous-prefix", action="store_true")
    parser.add_argument("--selective", choices=["all","vision_only","vlm_only"])
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--max-error", type=float, default=1e-5)
    parser.add_argument("--mae-error", type=float, default=1e-6)
    args = parser.parse_args()
    if args.noise_period < 0:
        parser.error("--noise-period must be nonnegative")
    seed_for = lambda index: 1701 + (index % args.noise_period if args.noise_period else index)
    pairs = [tuple(p.split(":")) for p in args.pairs]
    for pair in pairs:
        if len(pair)!=2 or any(name not in MODES for name in pair):
            parser.error("--pairs requires mode_a:mode_b")
        a,b = map(settings,pair)
        if (a["cuda_graphs"],a["ae_priority"]) != (b["cuda_graphs"],b["ae_priority"]):
            parser.error("Each pair must share Graph enablement and priority; compare other settings in separate groups")
        if any(settings(n)["mode"]=="fresh_serial" for n in pair):
            parser.error("Fresh-KV results have different semantics; use --continuous for those baselines")
        if "graph_encoder_overlap" in pair and any(name not in ("graph_serial", "graph_ae_first", "graph_encoder_overlap") for name in pair):
            parser.error("graph_encoder_overlap pairs support graph_serial/graph_ae_first only")
    requested_modes=[name for pair in pairs for name in pair]+args.continuous
    if "graph_encoder_overlap" in requested_modes and args.selective != "all":
        parser.error("graph_encoder_overlap requires explicit --selective all; historical modes retain the configured selective kind")
    if args.selective in ("vision_only","vlm_only"):
        expected="ae_first" if args.selective=="vision_only" else "prefill_first"
        for name in [n for pair in pairs for n in pair]+args.continuous:
            config=settings(name)
            if config["cuda_graphs"] and config["mode"]=="async_reuse" and config["submission_order"]!=expected:
                parser.error(f"{args.selective} requires {expected} for concurrent modes")
    dest=Path(args.output)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"source").mkdir()
    hashes={}
    for path in Path(__file__).parent.glob("*.py"):
        data=path.read_bytes()
        (dest/"source"/path.name).write_bytes(data)
        hashes[path.name]=hashlib.sha256(data).hexdigest()
    result=dict(status="preparing", completed=False, performance_eligible=False,
                args=vars(args),rows=[],groups=[],source_sha256=hashes,
                command=shlex.join([sys.executable,"-m","preparations.idea2.cewp_bench",*sys.argv[1:]]),
                environment={key:os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS",
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "JAX_PLATFORMS")},
                bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                tolerance=dict(max_abs=args.max_error,mae=args.mae_error),
                scope="AB/BA same-old-KV bootstrapped pairs; independent continuous replay. Event completion spans include host dispatch gaps. GPU busy overlap requires separate nsys.")
    write_results(dest,result)
    try:
        if args.gpu_uuid:
            result["gpu_before_init"]=require_idle_gpu(args.gpu_uuid)
            result["environment"]["CUDA_VISIBLE_DEVICES"]=os.environ.get("CUDA_VISIBLE_DEVICES")
        result["status"]="running"
        _run(args,dest,result,pairs,seed_for)
        result["completed"]=bool(result["groups"]) and all(group.get("status")=="passed" for group in result["groups"])
        result["status"]="passed" if result["completed"] else "incomplete"
        result["performance_eligible"]=(result["completed"] and not args.profile and
            all((group["phase"]=="paired" and args.count>=100) or
                (group["phase"]=="continuous" and args.continuous_count>=100) for group in result["groups"]))
    except BaseException as error:
        result.update(status="failed",completed=False,performance_eligible=False,
                      error=repr(error),traceback=traceback.format_exc())
        for group in result["groups"]:
            if group.get("status")=="running":
                group["status"]="failed"
        raise
    finally:
        write_results(dest,result)


def _run(args,dest,result,pairs,seed_for):
    backend=load_backend(args.checkpoint,model_config=args.model_config,num_steps=args.steps)
    backend.await_previous_prefix=args.await_previous_prefix
    if args.hoist_schedule:
        from .cewp_preparation import PreparedScheduleCore
        backend.graph_core_class=PreparedScheduleCore
    if args.hoist_modulation:
        from .cewp_preparation import PreparedModulationCore
        backend.graph_core_class=PreparedModulationCore
    if args.selective:
        from .cewp_segmented import SelectiveGraphs
        backend.graph_engine_class=SelectiveGraphs
        backend.selective_overlap=args.selective
    original_records=pickle.loads(Path(args.bank).read_bytes())
    instruction=original_records[args.instruction_index]["instruction"]
    selected_indices=[i for i,r in enumerate(original_records) if r["instruction"]==instruction]
    records=[original_records[i] for i in selected_indices]
    if len(records)<2:
        raise ValueError("Need at least two observations of the selected instruction")
    result["selected_bank_indices"]=selected_indices
    result["instruction"]=instruction
    result["identity"]=backend.identity
    result["gpu_uuid"]=str(torch.cuda.get_device_properties(backend.device).uuid)
    own_pids=None
    if args.gpu_uuid:
        if torch.cuda.device_count()!=1 or result["gpu_uuid"].lower().removeprefix("gpu-")!=args.gpu_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError("Visible CUDA device does not match the requested exclusive GPU")
        occupancy=gpu_snapshot(args.gpu_uuid)
        own_pids={row["pid"] for row in occupancy["processes"]}
        if len(own_pids)!=1 or occupancy["graphics_processes"]:
            raise RuntimeError("Expected one CUDA process and no graphics processes after model load")
        result["own_gpu_processes"]=sorted(own_pids)
        result["gpu_after_model_load"]=occupancy
    groups=[("paired",pair) for pair in pairs]+[("continuous",(name,)) for name in args.continuous]
    if args.kind=="isolated":
        groups=[("isolated",(name,)) for name in (args.continuous or ["eager_serial","graph_serial"])]
    for group_number,(phase,names) in enumerate(groups):
        print("Group",group_number,phase,names,flush=True)
        mixed_selective=phase=="paired" and "graph_encoder_overlap" in names
        group=dict(phase=phase,modes=names,status="running",correct=None,trajectory=None,
                   selective_kinds={name:selective_kind(name,args.selective) for name in names},
                   same_graph_selective_comparison=mixed_selective)
        result["groups"].append(group)
        if own_pids is not None:
            group["gpu_before"]=require_same_processes(args.gpu_uuid,own_pids)
        if args.compile:
            from .cewp_compile import install
            result["compiler"]=install(backend)
        recorder=Recorder(backend)
        if args.selective:
            # No previous policy/Graph survives its group cleanup. Set the
            # initial kind before this group's first capture; all captures use
            # identical kernels, while the alias changes only replay waits.
            backend.selective_overlap=selective_kind(names[0],args.selective)
        policy=PipelinePolicy(backend,diagnostics=True,**settings(names[0]))
        cold=[]
        try:
            for i in range(max(2,args.warmup)):
                response=recorder.predict(policy,records,i,900+i,mode=names[0],phase="warmup",measured=False)
                if response["graph_setup_ms"]:
                    cold.append(response["graph_setup_ms"])
            torch.cuda.reset_peak_memory_stats(backend.device)
            if args.profile:
                torch.cuda.profiler.start()
            try:
                if phase=="paired":
                    for pair_index in range(args.count):
                        order=arm_order(names,pair_index,args.reverse_order)
                        for ordinal,name in enumerate(order):
                            configure_arm(policy,name,args.selective,mixed_selective=mixed_selective)
                            recorder.predict(policy,records,2*pair_index,7000+pair_index,mode=name,phase="bootstrap",measured=False)
                            recorder.predict(policy,records,2*pair_index+1,seed_for(pair_index),mode=name,
                                             phase=phase,pair=pair_index,arm_order=ordinal)
                        if pair_index%10==9:
                            recorder.finish_events()
                            print("Pairs",pair_index+1,flush=True)
                elif phase=="continuous":
                    policy.reset()
                    for index in range(args.continuous_count):
                        if index and args.reset_every and index%args.reset_every==0:
                            policy.reset()
                        if policy.request==0:
                            recorder.predict(policy,records,index-1,7000+index,mode=names[0],phase="bootstrap",measured=False)
                        recorder.predict(policy,records,index,seed_for(index),mode=names[0],phase=phase)
                    group["trajectory"]=drain_continuous(policy,recorder.rows)
                else:
                    policy.reset()
                    recorder.predict(policy,records,0,7000,mode=names[0],phase="bootstrap",measured=False)
                    for module in ("ae","prefill"):
                        for index in range(args.count):
                            recorder.isolated(policy,records,index,seed_for(index),names[0],module)
            finally:
                if args.profile:
                    torch.cuda.profiler.stop()
            recorder.finish_events()
            if own_pids is not None:
                group["gpu_after_timing"]=require_same_processes(args.gpu_uuid,own_pids)
        finally:
            for row in recorder.rows:
                row["group"]=group_number
            result["rows"].extend(recorder.rows)
            group.update(capture_engines=recorder.capture_engines,cold_capture_ms=cold,
                         freshness_gate=freshness_gate(recorder.rows,args.await_previous_prefix))
            try:
                policy.close()
                group["policy_closed"]=True
            finally:
                del policy
                gc.collect()
                if args.compile:
                    from .cewp_compile import uninstall
                    uninstall(backend)
                write_results(dest,result)
        print("Verify",group_number,flush=True)
        passed=recorder.verify(backend,records,args.max_error,args.mae_error)
        group["correct"]=passed
        expected_count=(2*args.count if phase in ("paired","isolated") else args.continuous_count)
        group["expected_measurements"]=expected_count
        group["completed_measurements"]=len(recorder.rows)
        complete=expected_count>=0 and len(recorder.rows)==expected_count
        fresh_ok=not args.await_previous_prefix or group["freshness_gate"]["passed"]
        same_graph_ok=not mixed_selective or (recorder.capture_engines==1
            and all(row["steady_recapture"]==0 for row in recorder.rows))
        group["same_graph_selective_passed"]=same_graph_ok if mixed_selective else None
        group["status"]="passed" if passed and fresh_ok and complete and same_graph_ok else "failed"
        if own_pids is not None:
            group["gpu_after_verification"]=require_same_processes(args.gpu_uuid,own_pids)
        write_results(dest,result)
        if not passed:
            raise AssertionError("Predeclared numerical tolerance failed; evidence retained in result.json")
        if not fresh_ok:
            raise AssertionError("Old-KV freshness/Prefill/recapture contract failed; evidence retained in result.json")
        if not complete:
            raise AssertionError("Measured request count is incomplete; evidence retained in result.json")
        if not same_graph_ok:
            raise AssertionError("Selective comparison recaptured or replaced Graph ownership; evidence retained in result.json")
    print(json.dumps(result["summary"],indent=2),flush=True)


if __name__=="__main__":
    main()
