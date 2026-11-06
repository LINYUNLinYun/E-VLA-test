"""CPU regressions for the independent camera clock and admission accounting."""
import contextlib
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .layer_crossframe_bench import (ArrivalClock, CrossframeRunner, baseline_comparisons,
                                    parser, poll_lookahead, resource_layout, trajectory_interior,
                                    trajectory_pair_summary, validate_crossframe_tail_policy)


_BENCH_PACKAGE = sys.modules[CrossframeRunner.__module__].__package__


class ArrivalTests(unittest.TestCase):
    def test_arrival_between_admission_and_availability_checks_retries_free_slot(self):
        runner = object.__new__(CrossframeRunner)
        frame = object()
        runner.admit_encode = Mock(side_effect=[None, frame])
        runner.retire = Mock()
        runner.pool = SimpleNamespace(slots=[SimpleNamespace(state="FREE")])
        clock = SimpleNamespace(available=lambda now: 1)
        result, _ = runner.acquire_current(clock)
        self.assertIs(result, frame)
        self.assertEqual(runner.admit_encode.call_count, 2)

    def test_event_finishing_during_pending_scan_is_retired_before_error(self):
        runner = object.__new__(CrossframeRunner)
        frame = object()
        runner.admit_encode = Mock(side_effect=[None, frame])
        slot = SimpleNamespace(state="READY", ready=SimpleNamespace(query=lambda: True), readers=[])
        runner.pool = SimpleNamespace(slots=[slot])
        calls = []
        def retire():
            calls.append(1)
            if len(calls) == 2:
                slot.state = "FREE"
        runner.retire = retire
        result, _ = runner.acquire_current(SimpleNamespace(available=lambda now: 1))
        self.assertIs(result, frame)
        self.assertEqual(len(calls), 2)

    def test_truly_unretirable_pool_still_raises(self):
        runner = object.__new__(CrossframeRunner)
        runner.admit_encode = Mock(return_value=None)
        runner.retire = Mock()
        runner.pool = SimpleNamespace(slots=[SimpleNamespace(state="READY", ready=None, readers=[])])
        with self.assertRaisesRegex(RuntimeError, "No free frame slot"):
            runner.acquire_current(SimpleNamespace(available=lambda now: 1))

    def test_capacity_mode_does_not_invent_camera_timestamps_or_drop(self):
        clock = ArrivalClock(0, 10)
        self.assertEqual(clock.admit(10), (0, None, 0))
        self.assertEqual(clock.admit(1000), (1, None, 0))
        counts = clock.final_counts(10000)
        self.assertEqual(counts["admitted"], 2)
        self.assertEqual(counts["dropped"], 0)
        self.assertIsNone(counts["received"])

    def test_future_frame_is_not_admitted_or_waited_for(self):
        clock = ArrivalClock(100, 10)
        self.assertEqual(clock.admit(10), (0, 10, 0))
        self.assertIsNone(clock.admit(10.05))
        self.assertEqual(clock.last, 0)
        self.assertAlmostEqual(clock.next_delay(10.05), 0.05)

    def test_latest_frame_wins_only_before_admission(self):
        clock = ArrivalClock(100, 10)
        current = clock.admit(10)
        self.assertEqual(clock.available(10.35), 3)
        self.assertEqual(clock.last, 0)
        next_frame = clock.admit(10.35)
        self.assertEqual(current, (0, 10, 0))
        self.assertEqual(next_frame[0], 3)
        self.assertAlmostEqual(next_frame[1], 10.3)
        self.assertEqual(next_frame[2], 2)

    def test_received_frames_equal_admitted_dropped_and_one_waiting_frame(self):
        clock = ArrivalClock(100, 10)
        clock.admit(10)
        clock.admit(10.25)
        clock.admit(10.65)
        counts = clock.final_counts(10.99)
        self.assertEqual(counts["received"], 10)
        self.assertEqual(counts["admitted"], 3)
        self.assertEqual(counts["dropped_before_admission"], 4)
        self.assertEqual(counts["dropped"], 6)
        self.assertEqual(counts["buffered_unprocessed"], 1)
        self.assertEqual(counts["received"], counts["admitted"] + counts["dropped"] + counts["buffered_unprocessed"])

    def test_no_duplicate_admission_at_identical_timestamp(self):
        clock = ArrivalClock(100, 10)
        clock.admit(10.35)
        self.assertIsNone(clock.admit(10.35))

    def test_negative_period_rejected(self):
        with self.assertRaises(ValueError):
            ArrivalClock(-1, 0)

    def test_repeat_can_reverse_whole_trajectory_order(self):
        args = parser().parse_args(["--checkpoint", "unused", "--output", "unused", "--gpu-uuid", "GPU-test",
                                   "--modes", "combined", "encoder_overlap", "lookahead_serial",
                                   "--arrival-period-ms", "50"])
        self.assertEqual(args.modes, ["combined", "encoder_overlap", "lookahead_serial"])
        self.assertEqual(args.arrival_period_ms, 50)

    def test_default_modes_include_no_host_lookahead_baseline(self):
        args = parser().parse_args(["--checkpoint", "unused", "--output", "unused", "--gpu-uuid", "GPU-test"])
        self.assertEqual(args.modes, ["fresh_serial", "lookahead_serial", "encoder_overlap", "combined"])
        self.assertEqual(args.lookahead_window_ms, 0)
        self.assertEqual(args.lookahead_delay_ms, 0)
        self.assertEqual(args.pairs, 0)
        self.assertEqual(args.pair_length, 16)
        self.assertFalse(args.vlm_full)

    def test_import_does_not_initialize_or_import_torch(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import preparations.idea2.layer_crossframe_bench; assert 'torch' not in sys.modules"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class VirtualTime:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class VlmFullTests(unittest.TestCase):
    @staticmethod
    def args(*extra):
        return parser().parse_args(["--checkpoint", "unused", "--output", "unused",
                                    "--gpu-uuid", "GPU-test", *extra])

    def test_vlm_full_requires_partition_and_excludes_combined(self):
        with self.assertRaisesRegex(ValueError, "requires --ae-sm-min"):
            validate_crossframe_tail_policy(self.args("--vlm-full", "--modes", "encoder_overlap"))
        with self.assertRaisesRegex(ValueError, "combined would overlap"):
            validate_crossframe_tail_policy(self.args("--vlm-full", "--ae-sm-min", "64"))
        validate_crossframe_tail_policy(self.args("--vlm-full", "--ae-sm-min", "64", "--modes",
                                                  "fresh_serial", "lookahead_serial", "encoder_overlap"))

    def test_profile_mode_uses_same_restrictions_and_full_tail_still_excludes_encoder_overlap(self):
        validate_crossframe_tail_policy(self.args("--vlm-full", "--ae-sm-min", "64", "--profile",
                                                  "--profile-mode", "encoder_overlap"))
        with self.assertRaisesRegex(ValueError, "combined would overlap"):
            validate_crossframe_tail_policy(self.args("--vlm-full", "--ae-sm-min", "64", "--profile"))
        with self.assertRaisesRegex(ValueError, "full tail currently supports"):
            validate_crossframe_tail_policy(self.args("--vlm-full", "--ae-sm-min", "64", "--ae-layout",
                "per_step", "--tail-policy", "full", "--modes", "encoder_overlap"))

    def test_resource_layout_records_actual_counts_and_full_device_is_not_reserved(self):
        args = self.args("--vlm-full", "--ae-sm-min", "60", "--modes", "encoder_overlap")
        layout = resource_layout(args, dict(sm_counts=[64, 64]), 128)
        self.assertEqual(layout["ae"]["sm_count"], 64)
        self.assertEqual(layout["vision"]["sm_count"], 64)
        self.assertEqual(layout["vlm"], dict(stream="normal", sm_count=128))
        self.assertFalse(layout["vision_shares_vlm_stream"])
        self.assertTrue(layout["ae_vision_disjoint_sm_partitions"])
        args.vlm_full = False
        self.assertEqual(resource_layout(args, dict(sm_counts=[64, 64]), 128)["vlm"]["sm_count"], 64)
        self.assertFalse(resource_layout(args, None, 128)["ae_vision_disjoint_sm_partitions"])

    @staticmethod
    def streams(calls, separate):
        class Stream:
            def __init__(self, name):
                self.name = name
            def wait_event(self, event):
                calls.append(("wait", self.name, event))
        prefix = Stream("prefix")
        return prefix, Stream("vision") if separate else prefix

    @staticmethod
    def fake_torch(calls):
        class Event:
            def __init__(self, **kwargs):
                pass
            def record(self, stream=None):
                calls.append(("record", stream.name if stream else None, self))
        return SimpleNamespace(cuda=SimpleNamespace(Event=Event,
            stream=lambda stream: contextlib.nullcontext(),
            nvtx=SimpleNamespace(range=lambda label: contextlib.nullcontext())))

    def test_admission_dependencies_and_timing_follow_encoder_stream_without_extra_default_operations(self):
        for separate in (False, True):
            calls = []
            prefix, vision = self.streams(calls, separate)
            runner = CrossframeRunner.__new__(CrossframeRunner)
            runner.torch = self.fake_torch(calls)
            runner.retire = Mock()
            runner.records = [dict(instruction="task")]
            prepared, noise, identity, metadata, boundary = (object() for _ in range(5))
            runner.backend = SimpleNamespace(prepare=lambda raw: prepared, noise=lambda seed: noise)
            runner.pool = SimpleNamespace(slots=[SimpleNamespace(state="FREE")], epoch=0,
                offer=Mock(), reserve=lambda: (identity,), track_preparation=Mock())
            def encode(*args):
                calls.append(("encode", vision.name, None))
                return metadata
            runner.graphs = SimpleNamespace(prefix_stream=prefix, vision_stream=vision, encode=encode)
            with patch.dict(sys.modules, {_BENCH_PACKAGE + ".experiments": SimpleNamespace(native_raw=lambda record: record)}):
                frame = runner.admit_encode(ArrivalClock(0, 0), boundary=boundary)
            self.assertEqual([call[:2] for call in calls], [
                ("record", None), ("record", None), ("wait", vision.name),
                ("wait", vision.name), ("record", vision.name), ("encode", vision.name), ("record", vision.name)])
            self.assertIs(calls[2][2], frame.input_ready)
            self.assertIs(calls[3][2], boundary)
            self.assertIs(calls[4][2], frame.prefix_begin)
            self.assertIs(calls[6][2], frame.vision_done)
            runner.pool.track_preparation.assert_called_once_with(identity, metadata)

    def test_vlm_timer_starts_after_distinct_encoder_stream_and_previous_ae(self):
        for separate in (False, True):
            calls = []
            prefix, vision = self.streams(calls, separate)
            runner = CrossframeRunner.__new__(CrossframeRunner)
            runner.torch = self.fake_torch(calls)
            runner.pool = object()
            identity = SimpleNamespace(slot=0, generation=1, epoch=0)
            previous, encoded = object(), object()
            current = SimpleNamespace(identity=identity, source_id=0, vision_done=encoded)
            runner.graphs = SimpleNamespace(prefix_stream=prefix, vision_stream=vision,
                                            prefill=Mock(side_effect=RuntimeError("stop before replay")))
            modules = {_BENCH_PACKAGE + ".graph_bench": SimpleNamespace(cache_digest=Mock()),
                       _BENCH_PACKAGE + ".openpi_backend": SimpleNamespace(cache_tensors=Mock())}
            with patch.dict(sys.modules, modules), self.assertRaisesRegex(RuntimeError, "stop before replay"):
                runner.step(current, None, "encoder_overlap", phase="test", index=0, previous_ae=previous)
            expected = [("wait", "prefix", previous)]
            if separate:
                expected.append(("wait", "prefix", encoded))
            self.assertEqual(calls[:-1], expected)
            self.assertEqual(calls[-1][:2], ("record", "prefix"))


class DoneAt:
    def __init__(self, timer, when):
        self.timer, self.when = timer, when

    def query(self):
        return self.timer.value >= self.when


class LookaheadWindowTests(unittest.TestCase):
    def test_delay_selects_latest_frame_not_frame_available_at_start(self):
        timer = VirtualTime()
        clock = ArrivalClock(2, -0.002)
        clock.admit(-0.002)
        candidate, diagnostic = poll_lookahead(clock, DoneAt(timer, 1),
            lambda: clock.admit(timer.now()), 0, delay_ms=8, now=timer.now, sleep=timer.sleep)
        self.assertEqual(candidate[0], 5)
        self.assertEqual(candidate[2], 4)
        self.assertAlmostEqual(diagnostic["delay_ms"], 8)
        self.assertAlmostEqual(diagnostic["poll_ms"], 0)
        self.assertEqual(diagnostic["attempts"], 1)
        self.assertTrue(all(value <= .0005 for value in timer.sleeps))

    def test_ae_completion_interrupts_delay_without_admission(self):
        for done_at in (0, .002, .008):
            timer = VirtualTime()
            calls = []
            candidate, diagnostic = poll_lookahead(ArrivalClock(0, 0), DoneAt(timer, done_at),
                lambda: calls.append(True), 0, delay_ms=8, now=timer.now, sleep=timer.sleep)
            self.assertIsNone(candidate)
            self.assertEqual(calls, [])
            self.assertEqual(diagnostic["exit_reason"], "ae_completed_during_delay")
            self.assertLessEqual(timer.now(), done_at + .0005)

    def test_window_budget_starts_after_delay(self):
        timer = VirtualTime()
        clock = ArrivalClock(11, 0)
        clock.admit(0)
        candidate, diagnostic = poll_lookahead(clock, DoneAt(timer, 1),
            lambda: clock.admit(timer.now()), 5, delay_ms=8, now=timer.now, sleep=timer.sleep)
        self.assertEqual(candidate[0], 1)
        self.assertAlmostEqual(diagnostic["delay_ms"], 8)
        self.assertAlmostEqual(diagnostic["poll_ms"], 3)
        self.assertAlmostEqual(timer.now(), .011)

    def test_invalid_delay_rejected(self):
        for delay in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                poll_lookahead(ArrivalClock(0, 0), None, None, 0, delay_ms=delay)

    def run_poll(self, period_ms, done_at, window_ms):
        timer = VirtualTime()
        clock = ArrivalClock(period_ms, 0)
        clock.admit(0)
        calls = []
        def admit():
            calls.append(timer.value)
            return clock.admit(timer.value)
        candidate, diagnostic = poll_lookahead(clock, DoneAt(timer, done_at), admit, window_ms,
                                               now=timer.now, sleep=timer.sleep)
        return candidate, diagnostic, timer, calls

    def test_ae_completion_wins_before_future_arrival(self):
        candidate, diagnostic, timer, calls = self.run_poll(10, 0.002, 20)
        self.assertIsNone(candidate)
        self.assertEqual(diagnostic["exit_reason"], "ae_completed")
        self.assertEqual(calls, [])
        self.assertLessEqual(timer.value, 0.0025)
        self.assertTrue(all(duration <= 0.0005 for duration in timer.sleeps))

    def test_completion_wins_over_simultaneous_camera_arrival(self):
        candidate, diagnostic, _, calls = self.run_poll(2, 0.002, 20)
        self.assertIsNone(candidate)
        self.assertEqual(diagnostic["exit_reason"], "ae_completed")
        self.assertEqual(calls, [])

    def test_window_stops_before_future_frame_without_admission(self):
        candidate, diagnostic, timer, calls = self.run_poll(10, 1, 5)
        self.assertIsNone(candidate)
        self.assertEqual(diagnostic["exit_reason"], "window_expired")
        self.assertLessEqual(timer.value, 0.005 + 1e-12)
        self.assertEqual(calls, [])

    def test_arrival_inside_window_is_admitted(self):
        candidate, diagnostic, timer, calls = self.run_poll(2, 1, 5)
        self.assertEqual(candidate, (1, 0.002, 0))
        self.assertEqual(diagnostic["exit_reason"], "admitted")
        self.assertEqual(calls, [0.002])
        self.assertAlmostEqual(timer.value, 0.002)

    def test_zero_window_checks_once_without_sleep(self):
        candidate, diagnostic, timer, calls = self.run_poll(10, 1, 0)
        self.assertIsNone(candidate)
        self.assertEqual(diagnostic["attempts"], 1)
        self.assertEqual(calls, [0])
        self.assertEqual(timer.sleeps, [])

    def test_capacity_mode_still_admits_immediately_even_if_ae_already_done(self):
        candidate, diagnostic, timer, calls = self.run_poll(0, 0, 20)
        self.assertEqual(candidate, (1, None, 0))
        self.assertEqual(diagnostic["exit_reason"], "single_attempt")
        self.assertEqual(calls, [0])
        self.assertEqual(timer.sleeps, [])

    def test_poll_budget_does_not_hide_time_spent_preparing_admitted_input(self):
        timer = VirtualTime()
        clock = ArrivalClock(2, 0)
        clock.admit(0)
        def slow_admission():
            result = clock.admit(timer.value)
            timer.value += 0.03
            return result
        candidate, diagnostic = poll_lookahead(clock, DoneAt(timer, 1), slow_admission, 5,
                                               now=timer.now, sleep=timer.sleep)
        self.assertIsNotNone(candidate)
        self.assertAlmostEqual(diagnostic["poll_ms"], 2)
        self.assertAlmostEqual(diagnostic["attempt_ms"], 30)
        self.assertAlmostEqual(timer.value, 0.032)


class ComparisonTests(unittest.TestCase):
    def test_camera_slo_not_substituted_with_admission_latency(self):
        summary = {mode: dict(capture_to_action_ms=None, admission_to_action_ms={"p95": value})
                   for mode, value in (("fresh_serial", 80), ("lookahead_serial", 70), ("combined", 60))}
        groups = [dict(phase="continuous", mode=mode, all_work_hz=10,
                       arrival_counts=dict(received=None, dropped=0)) for mode in summary]
        comparison = baseline_comparisons(summary, groups)
        self.assertEqual(set(comparison), {"fresh_serial", "lookahead_serial"})
        for baseline in comparison.values():
            self.assertIsNone(baseline["combined"]["capture_p95_budget_checks"])
            self.assertIsNone(baseline["combined"]["capture_p95_ratio"])

    def test_camera_latency_budget_and_drop_are_both_reported(self):
        summary = {mode: dict(capture_to_action_ms={"p95": value}, admission_to_action_ms={"p95": value})
                   for mode, value in (("fresh_serial", 100), ("combined", 103))}
        groups = [dict(phase="continuous", mode=mode, all_work_hz=10,
                       arrival_counts=dict(received=100, dropped=20)) for mode in summary]
        row = baseline_comparisons(summary, groups)["fresh_serial"]["combined"]
        self.assertFalse(row["capture_p95_budget_checks"]["0.02"])
        self.assertTrue(row["capture_p95_budget_checks"]["0.05"])
        self.assertEqual(row["capture_p95_delta_ms"], 3)
        self.assertEqual(row["drop_fraction"], 0.2)


class TrajectoryPairTests(unittest.TestCase):
    @staticmethod
    def rows():
        return [dict(host_end=value / 1000, cpu_cpu_ms=index, admission_to_action_ms=index,
                     capture_to_action_ms=None)
                for index, value in enumerate((0, 1, 3, 7, 15, 31))]

    def test_interior_excludes_first_two_and_last_frame_and_boundary_intervals(self):
        interior = trajectory_interior(self.rows())
        self.assertEqual(interior["count"], 3)
        self.assertEqual(interior["output_interval_ms"]["n"], 2)
        self.assertAlmostEqual(interior["output_interval_ms"]["mean"], 6)
        self.assertEqual(interior["latency"]["admission_to_action_ms"]["mean"], 3)
        self.assertIsNone(interior["latency"]["capture_to_action_ms"])
        self.assertIsNone(trajectory_interior(self.rows()[:4]))

    def groups(self):
        return [dict(phase="paired", pair=pair, mode=mode, all_work_trajectory_ms=duration,
                     response_trajectory_ms=duration, all_work_hz=6000 / duration,
                     trimmed_interior=trajectory_interior(self.rows()))
                for pair, values in enumerate(((100, 50), (50, 100)))
                for mode, duration in zip(("fresh_serial", "combined"), values)]

    def test_speedup_uses_matched_trajectory_ratios_not_ratio_of_aggregates(self):
        result = trajectory_pair_summary(self.groups(), ("fresh_serial", "combined"))
        speedup = result["comparisons"]["fresh_serial"]["combined"]["full_trajectory_speedup"]
        self.assertEqual(speedup["n"], 2)
        self.assertEqual(speedup["mean"], 1.25)

    def test_incomplete_or_duplicate_pair_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "every requested arm"):
            trajectory_pair_summary(self.groups()[:-1], ("fresh_serial", "combined"))
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            trajectory_pair_summary(self.groups() + self.groups()[:1], ("fresh_serial", "combined"))

    def test_continuous_measurements_are_not_mixed_into_short_trajectory_pairs(self):
        groups = self.groups()
        groups.append(dict(phase="continuous", mode="combined", all_work_trajectory_ms=1))
        result = trajectory_pair_summary(groups, ("fresh_serial", "combined"))
        self.assertEqual(result["by_mode"]["combined"]["all_work_trajectory_ms"]["n"], 2)

    def test_model_span_excludes_prepare_gap_but_both_span_keeps_it(self):
        class Event:
            def __init__(self, stamp):
                self.stamp = stamp
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return other.stamp - self.stamp
        runner = CrossframeRunner.__new__(CrossframeRunner)
        row = {}
        runner.pending = [(row, *(Event(stamp) for stamp in (0, 30, 40, 42, 70, 31, 75)))]
        runner.finish_events()
        self.assertEqual(row["both_completion_event_ms"], 75)
        self.assertEqual(row["model_completion_event_ms"], 45)


if __name__ == "__main__":
    unittest.main()
