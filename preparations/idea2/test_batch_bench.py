"""CPU checks for batching sample identity, admission and timing boundaries."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from .batch_bench import (_cleanup_before_lease_release, arm_order, attach_checks, bank_schedule,
                          compare_lanes, parser, release_green_resources,
                          prepare_in_stream, seed_for, sequence, summaries, validate_args)


class BatchScheduleTests(unittest.TestCase):
    def bank(self, count=13):
        return [dict(instruction="same" if i < count else "another") for i in range(count + 2)]

    def test_all_batch_sizes_keep_the_same_records_and_per_record_noise(self):
        mappings = []
        for size in (1, 2, 4):
            schedule = bank_schedule(self.bank(), size)
            self.assertEqual(schedule["selected_bank_indices"], list(range(12)))
            self.assertEqual(schedule["excluded_same_task_remainder"], [12])
            rows = {}
            for frame in schedule["frames"]:
                self.assertEqual(len(set(frame["bank_indices"])), size)
                self.assertEqual(frame["lane_ids"], [f"lane{lane}" for lane in range(size)])
                rows.update(zip(frame["bank_indices"], frame["seeds"], strict=True))
            mappings.append(rows)
        self.assertEqual(mappings[0], mappings[1])
        self.assertEqual(mappings[1], mappings[2])
        self.assertEqual(seed_for(3, 0), seed_for(3, 2))
        self.assertNotEqual(seed_for(3, 0), seed_for(4, 0))

    def test_changing_frames_are_required_and_task_never_mixed(self):
        with self.assertRaisesRegex(ValueError, "two changing"):
            bank_schedule(self.bank(7), 4)
        with self.assertRaisesRegex(ValueError, "empty"):
            bank_schedule([], 1)
        schedule = bank_schedule(self.bank(), 4)
        self.assertEqual(len(schedule["frames"]), 3)
        self.assertNotIn(13, schedule["selected_bank_indices"])

    def test_paired_arms_alternate_and_independent_repeat_reverses(self):
        self.assertEqual([arm_order("paired", i) for i in range(4)],
                         [["serial", "pipeline"], ["pipeline", "serial"],
                          ["serial", "pipeline"], ["pipeline", "serial"]])
        self.assertEqual(arm_order("paired", 0, True), ["pipeline", "serial"])
        self.assertEqual(arm_order("green", 1), ["green"])

    def test_a_single_bad_lane_fails_even_if_aggregate_mae_is_small(self):
        reference = np.zeros((4, 10, 32))
        actual = reference.copy()
        actual[3, 0, 0] = 2e-5
        checks = compare_lanes(reference, actual, ["a", "b", "c", "d"])
        self.assertLess(checks["aggregate"]["mae"], 1e-6)
        self.assertFalse(checks["passed"])
        self.assertFalse(checks["per_lane"][3]["passed"])
        actual[3, 0, 0] = np.nan
        self.assertFalse(compare_lanes(reference, actual, ["a", "b", "c", "d"])["passed"])


class BatchCliTests(unittest.TestCase):
    def test_prefix_preparation_is_explicit_and_online_only(self):
        args = parser().parse_args(["--dry-run"])
        self.assertEqual(args.prepare_stream, "caller")
        args.prepare_stream = "prefix"
        with self.assertRaisesRegex(ValueError, "requires online"):
            validate_args(args)
        args.input_mode = "online"
        validate_args(args)
        args.prepare_stream = "dedicated"
        validate_args(args)

    def test_launch_is_explicit_and_profile_cannot_be_paired(self):
        args = parser().parse_args([])
        with self.assertRaisesRegex(ValueError, "explicit physical"):
            validate_args(args)
        args.dry_run = True
        validate_args(args)
        args.profile = True
        with self.assertRaisesRegex(ValueError, "one explicit mode"):
            validate_args(args)
        args.mode = "green"
        validate_args(args)

    def test_help_and_dryrun_never_import_torch_or_query_gpus(self):
        root = Path(__file__).resolve().parents[2]
        script = """
import contextlib, io, json, subprocess, sys
def forbidden(*a, **kw): raise AssertionError('GPU query/subprocess is forbidden')
subprocess.run = forbidden
from preparations.idea2.batch_bench import main
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    main(['--dry-run', '--batch-size', '4'])
assert 'torch' not in sys.modules
assert 'jax' not in sys.modules
value = json.loads(buffer.getvalue())
assert value['gpu_work_started'] is False
assert value['timed_requests'] == 1600
with contextlib.redirect_stdout(io.StringIO()):
    main(['--dry-run', '--input-mode', 'online', '--prepare-stream', 'dedicated'])
assert 'torch' not in sys.modules
with contextlib.redirect_stdout(io.StringIO()):
    try:
        main(['--help'])
    except SystemExit as exc:
        assert exc.code == 0
assert 'torch' not in sys.modules
"""
        process = subprocess.run([sys.executable, "-c", script], cwd=root,
                                 capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_cleanup_holds_lease_through_runtime_failure(self):
        operations = []
        @contextmanager
        def lease():
            operations.append("claim")
            try:
                yield
            finally:
                operations.append("release")
        with self.assertRaisesRegex(RuntimeError, "request failed"):
            with lease(), _cleanup_before_lease_release(lambda: operations.append("close_graphs")):
                raise RuntimeError("request failed")
        self.assertEqual(operations, ["claim", "close_graphs", "release"])

    def test_cross_batch_diagnostic_keeps_failed_gate_and_checks_same_batch(self):
        original = [dict(normalized=np.zeros((2, 10, 32)), decoded=np.zeros((2, 10, 7)))]
        same_batch = [dict(normalized=np.ones((2, 10, 32)), decoded=np.ones((2, 10, 7)))]
        outputs = [(same_batch[0]["normalized"].copy(), same_batch[0]["decoded"].copy())]
        rows = [dict(mode="serial", frame=0)]
        self.assertFalse(attach_checks(rows, outputs, original, ["a", "b"]))
        self.assertTrue(attach_checks(rows, outputs, original, ["a", "b"],
            same_batch_oracle=same_batch, diagnostic_batch_drift=True))
        self.assertFalse(rows[0]["checks"]["b1_equivalence_passed"])
        self.assertFalse(rows[0]["checks"]["fresh_normalized"]["passed"])
        outputs[0][0][1, 0, 0] += 2e-5
        self.assertFalse(attach_checks(rows, outputs, original, ["a", "b"],
            same_batch_oracle=same_batch, diagnostic_batch_drift=True))
        with self.assertRaisesRegex(ValueError, "independent same-B oracle"):
            attach_checks(rows, outputs, original, ["a", "b"], diagnostic_batch_drift=True)

    def test_diagnostic_does_not_excuse_mixed_schedule_error(self):
        original = [dict(normalized=np.zeros((2, 10, 32)), decoded=np.zeros((2, 10, 7)))]
        rows = [dict(mode="pipeline", frame=0, numerical=dict(matched_schedule=dict(passed=False)))]
        self.assertFalse(attach_checks(rows,
            [(original[0]["normalized"], original[0]["decoded"])], original, ["a", "b"],
            same_batch_oracle=original, diagnostic_batch_drift=True))

    def test_green_workspaces_and_cache_are_released_before_contexts(self):
        from unittest.mock import patch
        operations = ["caller_tensors_released"]
        pair = SimpleNamespace(streams=[SimpleNamespace(synchronize=lambda: operations.append("stream_wait"))],
                               close=lambda: operations.append("contexts"))
        fake = SimpleNamespace(_C=SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: operations.append("workspaces")),
            cuda=SimpleNamespace(synchronize=lambda: operations.append("device_wait"),
                                 empty_cache=lambda: operations.append("allocator_cache")))
        resources = [pair]
        with patch("preparations.idea2.batch_bench.gc.collect", lambda: operations.append("gc")):
            release_green_resources(resources, fake)
        self.assertLess(operations.index("caller_tensors_released"), operations.index("contexts"))
        self.assertLess(operations.index("workspaces"), operations.index("contexts"))
        self.assertLess(operations.index("allocator_cache"), operations.index("contexts"))
        self.assertEqual(operations[-1], "contexts")
        self.assertEqual(resources, [])


class SequenceBoundaryTests(unittest.TestCase):
    def test_prepare_event_is_recorded_on_producer_then_waited_by_caller(self):
        operations = []
        prefix = object()
        ready = SimpleNamespace(record=lambda stream: operations.append(("record", stream)))
        caller = SimpleNamespace(wait_event=lambda event: operations.append(("wait", event)))
        @contextmanager
        def context(stream):
            operations.append(("enter", stream))
            try:
                yield
            finally:
                operations.append(("exit", stream))
        prepared = SimpleNamespace()
        backend = SimpleNamespace(device="cuda:0", prepare_batch=lambda raws: (
            operations.append(("prepare", raws)) or prepared))
        cuda = SimpleNamespace(stream=context, Event=lambda: ready, current_stream=lambda device: caller)
        self.assertIs(prepare_in_stream(backend, [7], stream=prefix, cuda=cuda), prepared)
        self.assertIs(prepared._batch_prepare_ready, ready)
        self.assertEqual(operations, [("enter", prefix), ("prepare", [7]), ("record", prefix),
                                      ("exit", prefix), ("wait", ready)])

    def test_original_prepare_path_does_not_use_cuda(self):
        prepared = SimpleNamespace()
        backend = SimpleNamespace(prepare_batch=lambda raws: prepared)
        self.assertIs(prepare_in_stream(backend, [3], cuda=object()), prepared)
        self.assertFalse(hasattr(prepared, "_batch_prepare_ready"))

    def test_prepare_failure_restores_context_without_publishing_ready(self):
        operations = []
        @contextmanager
        def context(stream):
            operations.append("enter")
            try:
                yield
            finally:
                operations.append("exit")
        def prepare(raws):
            operations.append("prepare")
            raise RuntimeError("bad input")
        backend = SimpleNamespace(prepare_batch=prepare)
        cuda = SimpleNamespace(stream=context)
        with self.assertRaisesRegex(RuntimeError, "bad input"):
            prepare_in_stream(backend, [1], stream=object(), cuda=cuda)
        self.assertEqual(operations, ["enter", "prepare", "exit"])

    def fixture(self):
        operations = []
        contract = SimpleNamespace(epoch=0)
        def reset():
            operations.append("contract_reset")
            contract.epoch += 1
        contract.reset = reset
        def prepare(raw):
            operations.append(("prepare", raw[0]))
            return SimpleNamespace(frame=raw[0], epoch=contract.epoch)
        backend = SimpleNamespace(batch_size=2, contract=contract,
                                  prepare_batch=prepare,
                                  output_batch=lambda prepared, value: value + 1)
        engine = SimpleNamespace()
        engine.drain = lambda: operations.append("drain")
        engine.reset = lambda prepared: operations.append(("bootstrap", prepared.frame, prepared.epoch))
        def request(prepared, noise, **kwargs):
            operations.append(("request", prepared.frame, kwargs["schedule"], kwargs["collect_events"]))
            self.assertEqual(prepared.epoch, contract.epoch)
            if kwargs["schedule"] == "serial":
                self.assertFalse(kwargs["next_encoder"])
            future = kwargs["lookahead"]() if kwargs["lookahead"] is not None else None
            return np.full((2, 10, 3), noise), dict(cpu_cpu_ms=2.0, frontiers=[3] * 10), future
        engine.request = request
        engine.finish_record = lambda row: operations.append(("finish_record", row["index"]))
        frames = [[0, 1], [2, 3], [4, 5]]
        schedule = dict(frames=[dict(seeds=[i, i + 1], bank_indices=frame)
                                for i, frame in enumerate(frames)])
        return engine, backend, frames, [0, 1, 2], schedule, operations

    def test_prepared_setup_outside_group_and_event_reads_follow_final_drain(self):
        engine, backend, frames, noises, schedule, operations = self.fixture()
        rows, group, outputs = sequence(engine, backend, frames, noises, schedule,
            mode="serial", input_mode="prepared", start=0, length=4,
            phase="paired", pair=0, lead=3,
            profile_control=lambda active: operations.append(("profiler", active)))
        self.assertEqual(operations[:5], ["drain", "contract_reset", ("prepare", 0),
                                         ("prepare", 2), ("prepare", 4)])
        self.assertEqual(operations[5:8], [("bootstrap", 4, 1), "drain", ("profiler", True)])
        self.assertEqual(operations[-6:], ["drain", ("profiler", False),
                                         ("finish_record", 0), ("finish_record", 1),
                                         ("finish_record", 2), ("finish_record", 3)])
        self.assertEqual([row["frame"] for row in rows], [0, 1, 2, 0])
        self.assertEqual(len(outputs), 4)
        self.assertAlmostEqual(group["sample_hz"], group["request_hz"] * 2)
        self.assertFalse(group["bootstrap_and_prepare_setup_in_timing"])

    def test_online_preparation_is_inside_live_capture_and_next_encoder_requested(self):
        engine, backend, frames, noises, schedule, operations = self.fixture()
        rows, _, _ = sequence(engine, backend, frames, noises, schedule,
            mode="pipeline", input_mode="online", start=1, length=2,
            phase="paired", pair=0, lead=3,
            profile_control=lambda active: operations.append(("profiler", active)))
        self.assertEqual(operations[:5], ["drain", "contract_reset", ("prepare", 0),
                                         ("bootstrap", 0, 1), "drain"])
        live = operations.index(("profiler", True))
        self.assertGreater(operations.index(("prepare", 2)), live)
        self.assertGreater(operations.index(("prepare", 4)), live)
        self.assertTrue(all(row["camera_age_applicable"] is False for row in rows))

    def test_disabled_lookahead_prepares_every_current_input_without_future_encoder(self):
        engine, backend, frames, noises, schedule, operations = self.fixture()
        request = engine.request
        def current_only(prepared, noise, **kwargs):
            self.assertIsNone(kwargs["lookahead"])
            self.assertFalse(kwargs["next_encoder"])
            return request(prepared, noise, **kwargs)
        engine.request = current_only
        rows, group, _ = sequence(engine, backend, frames, noises, schedule,
            mode="pipeline", input_mode="online", start=0, length=4,
            phase="paired", pair=0, lead=3, lookahead_enabled=False)
        self.assertEqual([entry[1] for entry in operations if isinstance(entry, tuple) and entry[0]=="prepare"],
                         [4, 0, 2, 4, 0])
        self.assertFalse(group["lookahead_enabled"])
        for row in rows:
            self.assertFalse(row["lookahead_enabled"])
            self.assertEqual(row["input_prepare_age_at_admission_ms"], 0)
            self.assertGreater(row["input_prepare_to_cpu_ms"], 0)
            self.assertLessEqual(row["input_prepare_to_cpu_ms"], row["cpu_cpu_ms"])

    def test_input_lifetime_keeps_preparation_in_the_previous_request(self):
        engine, backend, frames, noises, schedule, _ = self.fixture()
        rows, _, _ = sequence(engine, backend, frames, noises, schedule,
            mode="pipeline", input_mode="online", start=0, length=4,
            phase="paired", pair=0, lead=3)
        self.assertEqual(rows[0]["input_prepare_age_at_admission_ms"], 0)
        for row in rows[1:]:
            self.assertGreater(row["input_prepare_age_at_admission_ms"], 0)
            self.assertGreater(row["input_prepare_to_cpu_ms"], row["cpu_cpu_ms"])

    def test_throughput_uses_all_work_time_and_sums_samples_not_stage_spans(self):
        rows = [dict(mode="serial", phase="paired", cpu_cpu_ms=20, ae_ms=1000),
                dict(mode="pipeline", phase="paired", cpu_cpu_ms=15, ae_ms=2000)]
        groups = [dict(mode="serial", phase="paired", count=8, all_work_ms=160, sample_hz=100),
                  dict(mode="pipeline", phase="paired", count=8, all_work_ms=120, sample_hz=400 / 3)]
        result = summaries(rows, groups, 2)
        self.assertEqual(result["serial"]["sample_hz"], 100)
        self.assertAlmostEqual(result["paired_comparison"]["throughput_speedup"], 4 / 3)
        self.assertNotIn("prefill_completion_host_ms", result["serial"])

    def test_profile_stops_even_if_request_submission_fails(self):
        engine, backend, frames, noises, schedule, operations = self.fixture()
        def request(*args, **kwargs):
            raise RuntimeError("submission failed")
        engine.request = request
        with self.assertRaisesRegex(RuntimeError, "submission failed"):
            sequence(engine, backend, frames, noises, schedule,
                mode="serial", input_mode="prepared", start=0, length=2,
                phase="profile", pair=0, lead=3,
                profile_control=lambda active: operations.append(("profiler", active)))
        self.assertEqual(operations[-1], ("profiler", False))


if __name__ == "__main__":
    unittest.main()
