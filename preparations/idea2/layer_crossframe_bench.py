"""Fresh-KV two-frame pipeline with an explicit no-lookahead baseline.

Capacity mode has continuously available recorded inputs and reports admission
latency, not camera age. --arrival-period-ms instead replays an independent
periodic camera clock: latest available unadmitted frame wins, at most one frame
is admitted ahead, and an admitted frame is never replaced. This is an input
arrival simulation, not a robot closed-loop experiment.
"""
import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shlex
import sys
import time
import traceback

import numpy as np

from .layer_bench import (LayerRunner, ae_completion_stream, audit_graph_schedules, capture_sources, eager_oracle, gpu_snapshot,
                          input_schedule, iteration_timing_fields, require_idle_gpu,
                          record_reader_tail, require_same_processes, statistics, validate_tail_policy, write_result)


MODES = ("fresh_serial", "lookahead_serial", "encoder_overlap", "combined")


def poll_lookahead(clock, ae_done, attempt, window_ms, *, delay_ms=0, now=time.perf_counter, sleep=time.sleep):
    """Bound only arrival polling, not a subsequently admitted preprocessing job.

    With delay_ms=0, capacity and zero-window modes retain the original single
    immediate attempt. Positive delay defers frame selection, with AE completion
    taking priority; the polling window starts afterward.
    With a positive periodic-arrival window, AE completion wins even if the
    camera arrival is simultaneous. Sleep granularity is at most 0.5 ms.
    """
    if not math.isfinite(window_ms) or window_ms < 0:
        raise ValueError("Lookahead window must be finite and nonnegative")
    if not math.isfinite(delay_ms) or delay_ms < 0:
        raise ValueError("Lookahead delay must be finite and nonnegative")
    delay_begin = now()
    delay_elapsed_ms = 0.0
    if delay_ms > 0:
        delay_deadline = delay_begin + delay_ms / 1000
        while True:
            if ae_done.query():
                return None, dict(poll_ms=0.0, delay_ms=(now() - delay_begin) * 1000,
                                  attempts=0, exit_reason="ae_completed_during_delay", attempt_ms=0.0)
            remaining = delay_deadline - now()
            if remaining <= 0:
                break
            sleep(min(0.0005, remaining))
        delay_elapsed_ms = (now() - delay_begin) * 1000
    # The existing polling budget begins only after the optional delay.
    begin = now()
    attempt_seconds = 0.0
    attempts = 0

    def try_admit():
        nonlocal attempt_seconds, attempts
        before = now()
        candidate = attempt()
        attempt_seconds += now() - before
        attempts += 1
        return candidate

    def finish(candidate, reason):
        return candidate, dict(poll_ms=max(0.0, now() - begin - attempt_seconds) * 1000,
                               delay_ms=delay_elapsed_ms, attempts=attempts, exit_reason=reason,
                               attempt_ms=attempt_seconds * 1000)

    if not clock.period or window_ms == 0:
        candidate = try_admit()
        return finish(candidate, "single_attempt")
    deadline = begin + window_ms / 1000
    while True:
        if ae_done.query():
            return finish(None, "ae_completed")
        checked = now()
        if checked >= deadline:
            return finish(None, "window_expired")
        if clock.available(checked) is not None:
            if ae_done.query():
                return finish(None, "ae_completed")
            if now() >= deadline:
                return finish(None, "window_expired")
            candidate = try_admit()
            if candidate is not None:
                return finish(candidate, "admitted")
        checked = now()
        if checked >= deadline:
            return finish(None, "window_expired")
        arrival_delay = clock.next_delay(checked)
        sleep(min(0.0005, deadline - checked, arrival_delay if arrival_delay > 0 else 0.0005))


class ArrivalClock:
    """One waiting camera frame; in-flight current/lookahead frames are pinned."""
    def __init__(self, period_ms, start):
        if period_ms < 0:
            raise ValueError("Arrival period cannot be negative")
        self.period = period_ms / 1000
        self.start = start
        self.last = -1
        self.dropped = 0

    def available(self, now):
        latest = self.last + 1 if not self.period else math.floor((now - self.start) / self.period)
        return latest if latest > self.last else None

    def admit(self, now):
        selected = self.available(now)
        if selected is None:
            return None
        skipped = selected - self.last - 1
        self.dropped += skipped
        self.last = selected
        capture = self.start + selected * self.period if self.period else None
        return selected, capture, skipped

    def next_delay(self, now):
        return max(0.0, self.start + (self.last + 1) * self.period - now) if self.period else 0.0

    def final_counts(self, now):
        if not self.period:
            return dict(admitted=self.last + 1, dropped=0, buffered_unprocessed=0,
                        received=None, arrival_mode="saturated_capacity")
        received = max(0, math.floor((now - self.start) / self.period) + 1)
        unprocessed = max(0, received - self.last - 1)
        return dict(admitted=self.last + 1 - self.dropped,
                    dropped=self.dropped + max(0, unprocessed - 1),
                    dropped_before_admission=self.dropped,
                    buffered_unprocessed=min(1, unprocessed), received=received,
                    arrival_mode="periodic_camera_replay")


@dataclass
class Frame:
    source_id: int
    frame: int
    seed: int
    capture: float | None
    dropped_before: int
    admitted: float
    prepared: object
    noise: object
    input_ready: object
    origin: object
    identity: object
    metadata: object
    prefix_begin: object
    vision_done: object
    prepare_ms: float
    noise_ms: float
    encode_submit_ms: float


class CrossframeRunner(LayerRunner):
    def __init__(self, backend, graphs, records, *, iteration_timing=False, lookahead_window_ms=0, lookahead_delay_ms=0):
        super().__init__(backend, graphs, records, iteration_timing=iteration_timing)
        self.retiring = []
        self.lookahead_window_ms = lookahead_window_ms
        self.lookahead_delay_ms = lookahead_delay_ms

    def retire(self):
        self.pool.poll()
        self.retiring[:] = [identity for identity in self.retiring
                           if not self.pool.retire_complete(identity)]

    def reset(self, *, phase, mode, index):
        row = super().reset(phase=phase, mode=mode, index=index)
        self.retiring.clear()
        return row

    def admit_encode(self, clock, *, boundary=None):
        """Never waits for a future arrival or an unavailable lookahead slot."""
        from .experiments import native_raw
        from .scheduler import Observation

        self.retire()
        if not any(slot.state == "FREE" for slot in self.pool.slots):
            return None
        admitted = time.perf_counter()
        selected = clock.admit(admitted)
        if selected is None:
            return None
        source_id, capture, skipped = selected
        frame, seed = input_schedule(source_id, len(self.records))
        torch = self.torch
        origin = torch.cuda.Event(enable_timing=True)
        origin.record()
        prepare_start = time.perf_counter()
        prepared = self.backend.prepare(native_raw(self.records[frame]))
        prepare_end = time.perf_counter()
        noise = self.backend.noise(seed)
        noise_end = time.perf_counter()
        ready = torch.cuda.Event()
        ready.record()
        observation = Observation(source_id, self.pool.epoch, "layer_bench", prepared,
                                  host_receipt_time=admitted)
        self.pool.offer(observation)
        reservation = self.pool.reserve()
        if reservation is None:
            raise RuntimeError("The sole Host lost a free lookahead slot during admission")
        identity = reservation[0]
        encode_begin = time.perf_counter()
        prefix_begin, vision_done = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with torch.cuda.stream(self.graphs.vision_stream):
            self.graphs.vision_stream.wait_event(ready)
            if boundary is not None:
                self.graphs.vision_stream.wait_event(boundary)
            prefix_begin.record(self.graphs.vision_stream)
        metadata = self.graphs.encode(self.pool, identity, prepared)
        vision_done.record(self.graphs.vision_stream)
        self.pool.track_preparation(identity, metadata)
        encode_end = time.perf_counter()
        return Frame(source_id, frame, seed, capture, skipped, admitted, prepared, noise,
                     ready, origin, identity, metadata, prefix_begin, vision_done,
                     (prepare_end - prepare_start) * 1000, (noise_end - prepare_end) * 1000,
                     (encode_end - encode_begin) * 1000)

    def acquire_current(self, clock):
        """Wait for input/slot only when no current action is awaiting return."""
        begin = time.perf_counter()
        while True:
            selected = self.admit_encode(clock)
            if selected is not None:
                return selected, (time.perf_counter() - begin) * 1000
            if clock.available(time.perf_counter()) is None:
                time.sleep(min(0.002, clock.next_delay(time.perf_counter())))
                continue
            # Arrival and Event completion can change after admit_encode's
            # earlier checks. Recheck retirement/free slots before diagnosing
            # a stuck pool; a newly arrived frame simply needs another attempt.
            self.retire()
            if any(slot.state == "FREE" for slot in self.pool.slots):
                continue
            events = [event for slot in self.pool.slots for event in
                      ([slot.ready] if slot.ready is not None else []) + slot.readers if not event.query()]
            if not events:
                self.retire()
                if any(slot.state == "FREE" for slot in self.pool.slots):
                    continue
                raise RuntimeError("No free frame slot and no pending owned work can retire one")
            events[0].synchronize()

    def step(self, current, clock, mode, *, phase, index, previous_ae=None,
             allow_lookahead=True, verify=False):
        from .graph_bench import cache_digest
        from .openpi_backend import cache_tensors

        torch = self.torch
        if mode not in MODES:
            raise ValueError("Unknown crossframe schedule")
        execute_begin = time.perf_counter()
        identity = current.identity
        with torch.cuda.nvtx.range(f"crossframe/{phase}/{mode}/frame_{current.source_id}/slot_{identity.slot}/gen_{identity.generation}"):
            prefill_begin = time.perf_counter()
            vlm_begin = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.graphs.prefix_stream):
                if previous_ae is not None:
                    self.graphs.prefix_stream.wait_event(previous_ae)
                if self.graphs.vision_stream is not self.graphs.prefix_stream:
                    self.graphs.prefix_stream.wait_event(current.vision_done)
                vlm_begin.record(self.graphs.prefix_stream)
            prefix, prefix_done, layers, metadata = self.graphs.prefill(self.pool, identity)
            self.pool.submitted_layers(identity, prefix, prefix_done, layers, metadata_ready=metadata)
            prefill_end = time.perf_counter()
            prefill_done_at_submit = prefix_done.query()
            lease = self.pool.acquire_layers(identity)
            ae_begin = time.perf_counter()
            ae_begin_event = torch.cuda.Event(enable_timing=True)
            ae_begin_event.record(self.graphs.ae_stream)
            try:
                raw, ae_done, marks = self.graphs.generate(self.pool, lease, current.prepared, current.noise,
                                                         input_ready=current.input_ready, serial=mode != "combined")
                ae_submit_end = time.perf_counter()
                ae_done_at_submit = ae_done.query()
                # The three lookahead arms have identical Host work/polling.
                # fresh_serial is an explicit no-Host-lookahead baseline.
                lookahead_begin = time.perf_counter()
                boundary = ae_done if mode == "lookahead_serial" else marks[1]
                if allow_lookahead and mode != "fresh_serial":
                    next_frame, poll = poll_lookahead(clock, ae_done,
                        lambda: self.admit_encode(clock, boundary=boundary), self.lookahead_window_ms,
                        delay_ms=self.lookahead_delay_ms)
                else:
                    next_frame = None
                    poll = dict(poll_ms=0.0, delay_ms=0.0, attempts=0, attempt_ms=0.0, exit_reason="disabled")
                lookahead_end = time.perf_counter()
                ae_done_after_lookahead = ae_done.query()
                ae_done.synchronize()
                output_begin = time.perf_counter()
                state = current.prepared.state[0].detach().cpu().numpy()
                normalized = raw[0].detach().cpu().numpy()
                normalized_saved = normalized.copy()
                decoded = self.backend.policy._output_transform(dict(state=state, actions=normalized))["actions"]
                output_end = time.perf_counter()
                self.pool.release(lease, ae_done)
            except BaseException:
                if not lease.released:
                    tail = record_reader_tail(torch, self.graphs)
                    self.pool.release(lease, tail)
                raise
        self.retiring.append(identity)
        self.retire()
        response_end = time.perf_counter()
        row = dict(phase=phase, mode=mode, index=index, frame=current.frame, seed=current.seed,
                   source_id=current.source_id, epoch=identity.epoch, slot=identity.slot,
                   generation=identity.generation, observation_age=0, kv_conditioning_offset_frames=0,
                   camera_age_frames_at_output=(max(0, math.floor((output_end - clock.start) / clock.period)
                                                    - current.source_id) if clock.period else None),
                   host_begin=current.admitted, host_end=response_end, capture_host=current.capture,
                   cpu_cpu_ms=(response_end - current.admitted) * 1000,
                   admission_to_action_ms=(output_end - current.admitted) * 1000,
                   capture_to_action_ms=(output_end - current.capture) * 1000 if current.capture is not None else None,
                   execute_to_action_ms=(output_end - execute_begin) * 1000,
                   prepare_cpu_ms=current.prepare_ms, noise_cpu_ms=current.noise_ms,
                   encode_cpu_submit_ms=current.encode_submit_ms,
                   vlm_cpu_submit_ms=(prefill_end - prefill_begin) * 1000,
                   prefill_cpu_submit_ms=current.encode_submit_ms + (prefill_end - prefill_begin) * 1000,
                   ae_cpu_submit_ms=(ae_submit_end - ae_begin) * 1000,
                   lookahead_host_ms=(lookahead_end - lookahead_begin) * 1000,
                   lookahead_poll_ms=poll["poll_ms"], lookahead_delay_ms=poll["delay_ms"],
                   lookahead_attempts=poll["attempts"],
                   lookahead_exit_reason=poll["exit_reason"],
                   lookahead_admitted=next_frame is not None,
                   ae_wait_after_lookahead_ms=(output_begin - lookahead_end) * 1000,
                   output_cpu_ms=(output_end - output_begin) * 1000,
                   dropped_before=current.dropped_before,
                   prefill_done_at_submit=prefill_done_at_submit,
                   ae_done_at_submit=ae_done_at_submit, ae_done_after_lookahead=ae_done_after_lookahead,
                   prefill_done_at_response=prefix_done.query(), live_snapshots=self.pool.live_snapshots,
                   graph_recaptures=0, num_steps=self.backend.num_steps,
                   **iteration_timing_fields(self.backend.num_steps))
        if self.iteration_timing:
            row.update(iteration_timing_fields(self.backend.num_steps, marks))
        self.pending.append((row, current.origin, current.prefix_begin, current.vision_done,
                             vlm_begin, prefix_done, ae_begin_event, ae_done))
        self.outputs.append((row, normalized_saved, np.asarray(decoded).copy()))
        if verify:
            prefix_done.synchronize()
            slot = self.graphs.slots[identity.slot]
            row["first_actions"] = slot.first_actions[0].detach().cpu().numpy().tolist()
            row["kv_digest"] = cache_digest(prefix)
            row["kv_addresses"] = tuple(tensor.data_ptr() for tensor in cache_tensors(prefix.cache))
        return row, next_frame, ae_done

    def finish_events(self):
        for row, origin, prefix_begin, vision_done, vlm_begin, prefix_done, ae_begin, ae_done in self.pending:
            prefix_done.synchronize()
            ae_done.synchronize()
            row.update(prefill_gpu_event_ms=prefix_begin.elapsed_time(prefix_done),
                       vision_embedding_event_ms=prefix_begin.elapsed_time(vision_done),
                       vision_to_vlm_gap_event_ms=vision_done.elapsed_time(vlm_begin),
                       vlm_event_ms=vlm_begin.elapsed_time(prefix_done),
                       ae_gpu_event_ms=ae_begin.elapsed_time(ae_done),
                       prefill_completion_event_ms=origin.elapsed_time(prefix_done),
                       ae_completion_event_ms=origin.elapsed_time(ae_done))
            row["both_completion_event_ms"] = max(row["prefill_completion_event_ms"], row["ae_completion_event_ms"])
            row["model_completion_event_ms"] = (row["both_completion_event_ms"]
                                                 - row["prefill_completion_event_ms"]
                                                 + row["prefill_gpu_event_ms"])
        self.pending.clear()

    def sequence(self, mode, count, *, phase, period_ms=0, reset_every=0, verify=False, profile=False):
        self.reset(phase=phase, mode=mode, index=0)
        torch = self.torch
        current = previous_ae = None
        rows = []
        if profile:
            torch.cuda.profiler.start()
        begin = time.perf_counter()
        clock = ArrivalClock(period_ms, begin)
        try:
            for index in range(count):
                if index and reset_every and index % reset_every == 0:
                    if current is not None:
                        raise RuntimeError("Lookahead must stop before an episode reset")
                    self.reset(phase=phase, mode=mode, index=index)
                    previous_ae = None
                acquisition_ms = 0.0
                if current is None:
                    current, acquisition_ms = self.acquire_current(clock)
                allow = index + 1 < count and not (reset_every and (index + 1) % reset_every == 0)
                row, current, previous_ae = self.step(current, clock, mode, phase=phase, index=index,
                                                     previous_ae=previous_ae, allow_lookahead=allow, verify=verify)
                row["acquire_current_host_ms"] = acquisition_ms
                rows.append(row)
            responses_end = time.perf_counter()
            self.graphs.prefix_stream.synchronize()
            ae_completion_stream(self.graphs).synchronize()
            self.retire()
            end = time.perf_counter()
        finally:
            if profile:
                torch.cuda.profiler.stop()
        self.finish_events()
        counts = clock.final_counts(responses_end)
        if counts["admitted"] != count:
            raise RuntimeError("Unexpected unused prefetched frame at trajectory end")
        group = dict(phase=phase, mode=mode, count=count, arrival_period_ms=period_ms,
                     reset_every=reset_every,
                     response_trajectory_ms=(responses_end - begin) * 1000,
                     all_work_trajectory_ms=(end - begin) * 1000,
                     response_hz=count / (responses_end - begin), all_work_hz=count / (end - begin),
                     completed_prefills=count, consumed_snapshots=count,
                     completed_prefill_hz=count / (end - begin),
                     output_interval_ms=statistics([(b["host_end"] - a["host_end"]) * 1000
                                                    for a, b in zip(rows, rows[1:])]),
                     arrival_counts=counts, trimmed_interior=trajectory_interior(rows))
        return rows, group


def trajectory_interior(rows):
    """Trim two fill frames and one drain frame; this does not prove steadiness."""
    if len(rows) < 5:
        return None
    middle = rows[2:-1]
    return dict(scope="Trimmed short-trajectory interior, not a claim of pure steady-state execution.",
                excluded_first=2, excluded_last=1, count=len(middle),
                output_interval_ms=statistics([(right["host_end"] - left["host_end"]) * 1000
                                               for left, right in zip(middle, middle[1:])]),
                latency={key: statistics([row[key] for row in middle])
                         for key in ("cpu_cpu_ms", "admission_to_action_ms", "capture_to_action_ms")})


def trajectory_pair_summary(groups, modes):
    paired = [group for group in groups if group["phase"] == "paired"]
    result = dict(scope="Matched complete capacity trajectories; time includes fill and drain. Interior excludes first two and last one frames.",
                  by_mode={}, comparisons={})
    if not paired:
        return result
    for mode in modes:
        selected = [group for group in paired if group["mode"] == mode]
        result["by_mode"][mode] = {
            key: statistics([group[key] for group in selected])
            for key in ("all_work_trajectory_ms", "response_trajectory_ms", "all_work_hz")}
        result["by_mode"][mode]["interior_interval_mean_ms"] = statistics([
            group["trimmed_interior"]["output_interval_ms"]["mean"] for group in selected])
        result["by_mode"][mode]["interior_admission_latency_p95_ms"] = statistics([
            group["trimmed_interior"]["latency"]["admission_to_action_ms"]["p95"] for group in selected])
    pairs = {}
    for group in paired:
        current = pairs.setdefault(group["pair"], {})
        if group["mode"] in current:
            raise RuntimeError("Duplicate arm in a paired trajectory")
        current[group["mode"]] = group
    if any(set(pair) != set(modes) for pair in pairs.values()):
        raise RuntimeError("Paired trajectory summary requires every requested arm")
    for baseline in ("fresh_serial", "lookahead_serial"):
        if baseline not in modes:
            continue
        result["comparisons"][baseline] = {}
        for mode in modes:
            if mode == baseline:
                continue
            result["comparisons"][baseline][mode] = dict(
                full_trajectory_speedup=statistics([
                    pair[baseline]["all_work_trajectory_ms"] / pair[mode]["all_work_trajectory_ms"]
                    for pair in pairs.values()]),
                interior_interval_speedup=statistics([
                    pair[baseline]["trimmed_interior"]["output_interval_ms"]["mean"]
                    / pair[mode]["trimmed_interior"]["output_interval_ms"]["mean"]
                    for pair in pairs.values()]))
    return result


def baseline_comparisons(summary, groups):
    """Distribution comparisons, not matched per-frame cross-arm errors."""
    by_mode = {group["mode"]: group for group in groups if group["phase"] == "continuous"}
    result = {}
    for baseline in ("fresh_serial", "lookahead_serial"):
        if baseline not in summary or baseline not in by_mode:
            continue
        candidates = {}
        for mode, values in summary.items():
            if mode not in by_mode:
                continue
            entry = dict(all_work_hz_ratio=by_mode[mode]["all_work_hz"] / by_mode[baseline]["all_work_hz"],
                         capture_p95_ratio=None, capture_p95_delta_ms=None,
                         capture_p95_budget_checks=None)
            for metric, label in (("capture_to_action_ms", "capture"), ("admission_to_action_ms", "admission")):
                reference, current = summary[baseline][metric], values[metric]
                if reference is None or current is None:
                    continue
                entry[label + "_p95_delta_ms"] = current["p95"] - reference["p95"]
                entry[label + "_p95_ratio"] = current["p95"] / reference["p95"] if reference["p95"] else None
                if label == "capture":
                    entry["capture_p95_budget_checks"] = {
                        str(budget): current["p95"] <= reference["p95"] * (1 + budget)
                        for budget in (0, 0.02, 0.05, 0.1)}
            counts = by_mode[mode]["arrival_counts"]
            entry["drop_fraction"] = counts["dropped"] / counts["received"] if counts["received"] else None
            candidates[mode] = entry
        result[baseline] = candidates
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--output", required=True)
    result.add_argument("--gpu-uuid", required=True)
    result.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    result.add_argument("--num-steps", type=int, choices=(2, 10), default=2)
    result.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    result.add_argument("--boundary", choices=("kv_ready", "block_ready"), default="kv_ready")
    result.add_argument("--group-size", type=int, choices=(1, 2, 4), default=1)
    result.add_argument("--ae-layout", choices=("whole", "per_step"), default="whole",
                        help="per_step captures AE1 and the remaining D-1 steps as two graphs")
    result.add_argument("--tail-policy", choices=("native", "wait_prefix", "full"), default="native",
                        help="wait_prefix/full requires per_step; full additionally requires --ae-sm-min")
    result.add_argument("--lifecycle-audit", action="store_true")
    result.add_argument("--warmup", type=int, default=10)
    result.add_argument("--continuous-count", type=int, default=100)
    result.add_argument("--pairs", type=int, default=0,
                        help="Matched capacity trajectory groups, alternating whole-arm order; 0 disables")
    result.add_argument("--pair-length", type=int, default=16,
                        help="Frames per paired arm, including fill/drain; require >=5")
    result.add_argument("--reset-every", type=int, default=50)
    result.add_argument("--arrival-period-ms", type=float, default=0)
    result.add_argument("--lookahead-window-ms", type=float, default=0,
                        help="Periodic arrival polling budget; AE done has priority. 0 checks once, capacity mode is unchanged.")
    result.add_argument("--lookahead-delay-ms", type=float, default=0,
                        help="Delay next-frame admission, stopping when AE completes; polling window starts afterward. 0 preserves immediate admission.")
    result.add_argument("--hoist-modulation", action="store_true")
    result.add_argument("--ae-priority", type=int, choices=(0, -1), default=0)
    result.add_argument("--ae-sm-min", type=int, default=0)
    result.add_argument("--vlm-full", action="store_true",
                        help="Partition AE/Encoder only and run VLM on a normal full-device stream; requires --ae-sm-min and excludes combined")
    result.add_argument("--iteration-timing", action="store_true")
    result.add_argument("--profile", action="store_true")
    result.add_argument("--profile-mode", choices=MODES, default="combined")
    result.add_argument("--max-error", type=float, default=1e-5)
    result.add_argument("--mae-error", type=float, default=1e-6)
    return result


def validate_crossframe_tail_policy(args):
    validate_tail_policy(args)
    selected = (args.profile_mode,) if args.profile else args.modes
    if args.tail_policy == "full" and any(mode in ("encoder_overlap", "combined") for mode in selected):
        raise ValueError("full tail currently supports fresh_serial/lookahead_serial only; concurrent next-frame Encoder would share its SMs")
    if args.vlm_full:
        if args.ae_sm_min <= 0:
            raise ValueError("--vlm-full requires --ae-sm-min > 0")
        if "combined" in selected:
            raise ValueError("--vlm-full supports fresh_serial/lookahead_serial/encoder_overlap only; combined would overlap full-device VLM with partitioned AE")


def resource_layout(args, green_info, device_sm_count):
    """Describe eligibility, not an exclusive reservation for normal streams."""
    counts = green_info["sm_counts"] if green_info is not None else (device_sm_count, device_sm_count)
    return dict(ae=dict(stream="green_partition" if green_info else "normal", sm_count=counts[0]),
                vision=dict(stream="green_partition" if green_info else "normal", sm_count=counts[1]),
                vlm=dict(stream="normal" if args.vlm_full or not green_info else "green_partition",
                         sm_count=device_sm_count if args.vlm_full else counts[1]),
                vision_shares_vlm_stream=not args.vlm_full,
                ae_vision_disjoint_sm_partitions=green_info is not None,
                vlm_full=args.vlm_full,
                scope="Normal streams may use all device SMs; this does not reserve exclusive resources. Separate full-device VLM is scheduled after prior AE and current Encoder completion.")


def main():
    args = parser().parse_args()
    validate_crossframe_tail_policy(args)
    if (args.warmup < 2 or args.continuous_count < 1 or args.reset_every < 0 or args.ae_sm_min < 0
            or args.pairs < 0 or args.pair_length < 5
            or not math.isfinite(args.arrival_period_ms) or args.arrival_period_ms < 0
            or not math.isfinite(args.lookahead_window_ms) or args.lookahead_window_ms < 0
            or not math.isfinite(args.lookahead_delay_ms) or args.lookahead_delay_ms < 0):
        raise ValueError("Invalid warmup, count, reset, quota, or arrival period")
    if len(set(args.modes)) != len(args.modes) or (args.pairs and len(args.modes) < 2):
        raise ValueError("Modes must be distinct; paired trajectories need at least two arms")
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    result = dict(status="preparing", args=vars(args), steps=args.num_steps, rows=[], smoke=[], groups=[],
                  environment={key: os.environ.get(key) for key in
                               ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_MAX_CONNECTIONS", "OMP_NUM_THREADS",
                                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                  performance_eligible=False,
                  command=shlex.join([sys.executable, "-m", "preparations.idea2.layer_crossframe_bench", *sys.argv[1:]]),
                  source_sha256=capture_sources(dest), bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                  tolerance=dict(max_abs=args.max_error, mae=args.mae_error),
                  semantics="Fresh same-frame KV for every AE iteration. Encoder lookahead waits for AE1; next-frame VLM waits for all D iterations. fresh_serial disables Host lookahead; the other three arms share identical one-frame CPU lookahead and polling policy.",
                  ae_timing_semantics={"ae_iteration1_event_ms": "First AE iteration including KV waits.",
                      **({"ae_iteration2_event_ms": "Second and final AE iteration."} if args.num_steps == 2 else
                         {"ae_tail2to10_event_ms": "Combined AE iterations 2 through 10, not a single iteration."})},
                  source_matching="Periodic schedules may drop/admit different source IDs. Each output is checked against its actual source frame and seed, not a different arm's selected frame.",
                  timing_semantics="Event spans include waits/gaps/contention, not kernel busy time. both_completion includes input preparation gaps; model_completion starts before Encoder input staging, not at its first kernel. Prefill span includes the separately reported Vision-to-VLM gap. observation_age/kv_conditioning_offset_frames denote KV relative to the selected conditioning frame, not the latest camera frame at output. Capacity mode has no camera age; periodic mode reports capture-to-action and camera_age_frames_at_output including admission/queueing.",
                  lookahead_window_semantics="Optional delay defers frame selection and exits when AE completes; window bounds arrival polling after that delay only. Delay and preparation remain in Host latency. Once admitted, preprocessing can outlast it and delay the current action; all such delay remains in measured latency.",
                  paired_scope="Optional full capacity trajectories with no mid-trajectory reset and matching source/noise sequence. Periodic arrivals apply only to continuous groups.",
                  independent_repeat="Run again in a new process/output directory; no independent repeat is claimed by this run.")
    write_result(dest, result)
    runner = graphs = green_pair = ae_stream = prefix_stream = vision_stream = tail_stream = None
    failure = None
    try:
        result["gpu_before_init"] = require_idle_gpu(args.gpu_uuid)
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
        import torch
        from .experiments import load_backend, native_raw
        from .external_events import cross_graph_self_test
        from .layer_graphs import LayerGraphs
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one CUDA device must be visible")
        actual = str(torch.cuda.get_device_properties(0).uuid)
        if actual.lower().removeprefix("gpu-") != args.gpu_uuid.lower().removeprefix("gpu-"):
            raise RuntimeError("Visible CUDA device UUID differs from requested exclusive GPU")
        result["actual_cuda_uuid"] = actual
        result["external_events"] = cross_graph_self_test()
        occupancy = gpu_snapshot(args.gpu_uuid)
        own_pids = {row["pid"] for row in occupancy["processes"]}
        if len(own_pids) != 1 or occupancy["graphics_processes"]:
            raise RuntimeError("Expected one CUDA process and no graphics processes after self-test")
        result["own_gpu_processes"] = sorted(own_pids)
        backend = load_backend(args.checkpoint, model_config="pi05_libero", num_steps=args.num_steps)
        result["identity"] = backend.identity
        bank = pickle.loads(Path(args.bank).read_bytes())
        indices = [index for index, row in enumerate(bank) if row["instruction"] == bank[0]["instruction"]]
        records = [bank[index] for index in indices]
        if len(records) < 2:
            raise RuntimeError("At least two changing observations are required")
        result["selected_bank_indices"] = indices
        oracle = eager_oracle(backend, records)
        result["oracle"] = [dict(index=row["index"], seed=row["seed"], kv_digest=row["kv_digest"],
                                 times=row["times"], actual_nfe=row["actual_nfe"],
                                 schedule_reference=row["schedule_reference"])
                            for row in oracle]
        if args.hoist_modulation:
            from .cewp_preparation import PreparedModulationCore
            backend.graph_core_class = PreparedModulationCore
        if args.ae_sm_min:
            from .green_context import GreenContextPair, build_library
            library = build_library(dest / "green_build")
            green_pair = GreenContextPair(library, args.ae_sm_min, device=backend.device,
                                          priorities=(args.ae_priority, 0))
            if args.vlm_full:
                ae_stream, vision_stream = green_pair.streams
                prefix_stream = torch.cuda.Stream(device=backend.device)
            else:
                ae_stream, prefix_stream = green_pair.streams
            result["green_context"] = dict(green_pair.info, branch_order=["ae", "vision" if args.vlm_full else "prefix"])
        else:
            prefix_stream = torch.cuda.Stream(device=backend.device)
            ae_stream = torch.cuda.Stream(device=backend.device, priority=args.ae_priority)
            result["green_context"] = None
        result["resource_layout"] = resource_layout(args, result["green_context"],
                                                   torch.cuda.get_device_properties(backend.device).multi_processor_count)
        prepared = backend.prepare(native_raw(records[0]))
        noise = backend.noise(input_schedule(0, len(records))[1])
        if args.tail_policy == "full":
            tail_stream = torch.cuda.Stream(device=backend.device, priority=args.ae_priority)
        graphs = LayerGraphs(backend, prepared, noise, prefix_stream, ae_stream,
                             boundary=args.boundary, group_size=args.group_size, profile_labels=args.profile,
                             ae_layout=args.ae_layout, tail_stream=tail_stream,
                             tail_gate=args.tail_policy != "native", vision_stream=vision_stream)
        del prepared, noise
        result["graph_setup_ms"] = graphs.setup_ms
        result["kv_slot_addresses"] = graphs.addresses
        result["graph_schedule_audit"] = audit_graph_schedules(graphs, oracle)
        runner = CrossframeRunner(backend, graphs, records, iteration_timing=args.iteration_timing,
                                  lookahead_window_ms=args.lookahead_window_ms, lookahead_delay_ms=args.lookahead_delay_ms)
        if args.lifecycle_audit:
            from .layer_lifecycle import audit
            result["lifecycle_audit"] = audit(backend, graphs, runner.pool, records, oracle,
                                               max_error=args.max_error, mae_error=args.mae_error)
        modes = [args.profile_mode] if args.profile else args.modes
        for mode in modes:
            rows, _ = runner.sequence(mode, 2 * len(records), phase="smoke", verify=True, reset_every=len(records))
            result["smoke"].extend(rows)
            if not runner.verify_outputs(oracle, args.max_error, args.mae_error):
                raise RuntimeError("Crossframe first/final action or KV smoke failed")
            if any(tuple(row["kv_addresses"]) != tuple(graphs.addresses[row["slot"]]) for row in rows):
                raise RuntimeError("KV addresses changed across frame contents or reset")
            write_result(dest, result)
        result["smoke_status"] = "passed"
        for mode in modes:
            runner.sequence(mode, args.warmup, phase="warmup")
            if not runner.verify_outputs(oracle, args.max_error, args.mae_error):
                raise RuntimeError("Crossframe warmup numerical gate failed")
        result["status"] = "measuring"
        write_result(dest, result)
        if args.pairs and not args.profile:
            for pair_index in range(args.pairs):
                ordered = modes if pair_index % 2 == 0 else list(reversed(modes))
                input_signature = None
                for ordinal, mode in enumerate(ordered):
                    gpu_before = require_same_processes(args.gpu_uuid, own_pids)
                    torch.cuda.reset_peak_memory_stats()
                    rows, group = runner.sequence(mode, args.pair_length, phase="paired", period_ms=0, reset_every=0)
                    signature = [(row["source_id"], row["frame"], row["seed"]) for row in rows]
                    if input_signature is not None and signature != input_signature:
                        raise RuntimeError("Paired trajectory arms consumed different observations or noise seeds")
                    input_signature = signature
                    for row in rows:
                        row.update(pair=pair_index, arm_order=ordinal,
                                   trimmed_interior=2 <= row["index"] < args.pair_length - 1)
                    result["rows"].extend(rows)
                    group.update(pair=pair_index, arm_order=ordinal,
                                 correct=runner.verify_outputs(oracle, args.max_error, args.mae_error),
                                 input_signature_sha256=hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
                                 gpu_before=gpu_before, gpu_after=require_same_processes(args.gpu_uuid, own_pids),
                                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    result["groups"].append(group)
                    if not group["correct"]:
                        write_result(dest, result)
                        raise RuntimeError("Paired crossframe trajectory numerical gate failed")
                write_result(dest, result)
                if pair_index % 10 == 9:
                    print(f"Crossframe pairs {pair_index + 1}/{args.pairs}", flush=True)
        for mode in modes:
            gpu_before = require_same_processes(args.gpu_uuid, own_pids)
            torch.cuda.reset_peak_memory_stats()
            rows, group = runner.sequence(mode, args.continuous_count, phase="continuous",
                                          period_ms=args.arrival_period_ms, reset_every=args.reset_every,
                                          profile=args.profile)
            result["rows"].extend(rows)
            group.update(correct=runner.verify_outputs(oracle, args.max_error, args.mae_error),
                         gpu_before=gpu_before, gpu_after=require_same_processes(args.gpu_uuid, own_pids),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved())
            result["groups"].append(group)
            write_result(dest, result)
            if not group["correct"]:
                raise RuntimeError("Crossframe measured output numerical gate failed")
        keys = ("admission_to_action_ms", "capture_to_action_ms", "execute_to_action_ms", "cpu_cpu_ms",
                "prepare_cpu_ms", "noise_cpu_ms", "output_cpu_ms", "lookahead_host_ms", "lookahead_poll_ms",
                "prefill_cpu_submit_ms", "ae_cpu_submit_ms", "vision_embedding_event_ms", "vlm_event_ms",
                "vision_to_vlm_gap_event_ms", "prefill_gpu_event_ms", "ae_gpu_event_ms",
                "both_completion_event_ms", "model_completion_event_ms", *iteration_timing_fields(args.num_steps))
        result["summary"] = {mode: {key: statistics([row[key] for row in result["rows"]
                                                    if row["mode"] == mode and row["phase"] == "continuous"])
                                    for key in keys} for mode in modes}
        result["paired_summary"] = trajectory_pair_summary(result["groups"], modes)
        result["baseline_comparisons"] = baseline_comparisons(result["summary"], result["groups"])
        result["resets"] = runner.reset_rows
        result["status"] = "passed"
        result["performance_eligible"] = not args.profile and not args.iteration_timing and args.continuous_count >= 100
        result["paired_performance_eligible"] = not args.profile and not args.iteration_timing and args.pairs >= 100
    except BaseException as error:
        failure = repr(error)
        result.update(status="failed", error=failure, traceback=traceback.format_exc())
    finally:
        drained = True
        if graphs is not None:
            try:
                graphs.close()
            except BaseException:
                drained = False
                result.update(status="failed", cleanup_error=traceback.format_exc())
        runner = graphs = ae_stream = prefix_stream = vision_stream = tail_stream = None
        gc.collect()
        if green_pair is not None and drained:
            try:
                green_pair.close()
                result["green_context_closed"] = True
            except BaseException:
                result.update(status="failed", green_cleanup_error=traceback.format_exc())
        if result["status"] != "passed":
            result["performance_eligible"] = False
            result["paired_performance_eligible"] = False
        write_result(dest, result)
    if result["status"] != "passed":
        raise RuntimeError(failure or "Crossframe cleanup failed; see result.json")


if __name__ == "__main__":
    main()
