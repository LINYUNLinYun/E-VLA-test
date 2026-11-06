"""CPU checks for D2/D10 schedule gates and truthful iteration diagnostics."""
from contextlib import redirect_stderr
import io
import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .layer_bench import (ae_completion_stream, audit_graph_schedules, iteration_timing_fields,
                          native_schedule_reference, parser, record_reader_tail,
                          require_idle_gpu, require_same_processes, gpu_snapshot,
                          validate_schedule, validate_tail_policy)
from .layer_crossframe_bench import parser as crossframe_parser, validate_crossframe_tail_policy


class StepConfigurationTests(unittest.TestCase):
    required = ["--checkpoint", "unused", "--output", "unused", "--gpu-uuid", "GPU-test"]

    def test_d2_defaults_and_d10_two_segment_layout_are_explicit(self):
        for make_parser in (parser, crossframe_parser):
            with self.subTest(parser=make_parser.__module__):
                self.assertEqual(make_parser().parse_args(self.required).num_steps, 2)
                args = make_parser().parse_args(self.required +
                    ["--num-steps", "10", "--ae-layout", "per_step", "--lifecycle-audit"])
                self.assertEqual(args.num_steps, 10)
                self.assertEqual(args.ae_layout, "per_step")
                self.assertTrue(args.lifecycle_audit)

    def test_tail_policy_requires_split_graphs_and_full_requires_partition(self):
        for make_parser in (parser, crossframe_parser):
            for policy in ("wait_prefix", "full"):
                args = make_parser().parse_args(self.required + ["--tail-policy", policy])
                with self.assertRaisesRegex(ValueError, "per_step"):
                    validate_tail_policy(args)
            args = make_parser().parse_args(self.required +
                ["--tail-policy", "full", "--ae-layout", "per_step"])
            with self.assertRaisesRegex(ValueError, "ae-sm-min"):
                validate_tail_policy(args)
            args.ae_sm_min = 32
            validate_tail_policy(args)
            args.tail_policy, args.ae_sm_min = "wait_prefix", 0
            validate_tail_policy(args)

    def test_unsupported_step_count_cannot_silently_change_experiment(self):
        for make_parser in (parser, crossframe_parser):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                make_parser().parse_args(self.required + ["--num-steps", "5"])

    def test_crossframe_full_tail_does_not_claim_exclusive_sms_with_encoder_overlap(self):
        args = crossframe_parser().parse_args(self.required +
            ["--tail-policy", "full", "--ae-layout", "per_step", "--ae-sm-min", "32"])
        with self.assertRaisesRegex(ValueError, "concurrent next-frame Encoder"):
            validate_crossframe_tail_policy(args)
        args.modes = ["fresh_serial", "lookahead_serial"]
        validate_crossframe_tail_policy(args)
        args.profile = True
        args.profile_mode = "combined"
        with self.assertRaises(ValueError):
            validate_crossframe_tail_policy(args)


class GraphicsIsolationTests(unittest.TestCase):
    def test_snapshot_keeps_compute_and_graphics_evidence(self):
        guard = SimpleNamespace(graphics_processes=Mock(return_value=[dict(pid=12, type="G", memory_mib=1)]))
        module = sys.modules[gpu_snapshot.__module__]
        with patch.dict(sys.modules, {module.__package__ + ".gpu_guard": guard}), patch.object(
                module, "smi_rows", side_effect=[
                    [["GPU-test", "test", "2", "0"]], [["11", "GPU-test", "1"]]]):
            result = gpu_snapshot("GPU-test")
        self.assertEqual(result["processes"][0]["pid"], 11)
        self.assertEqual(result["graphics_processes"][0]["pid"], 12)
        guard.graphics_processes.assert_called_once_with("GPU-test")

    def test_small_idle_graphics_allocation_still_blocks_initial_and_continuing_runs(self):
        snapshot = dict(processes=[], graphics_processes=[dict(pid=12, type="G", memory_mib=1)],
                        memory_used_mib=2, utilization_percent=0)
        with patch.object(sys.modules[require_idle_gpu.__module__], "gpu_snapshot", return_value=snapshot), patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": "unchanged"}):
            with self.assertRaisesRegex(RuntimeError, "not exclusive/idle"):
                require_idle_gpu("GPU-test")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "unchanged")
            snapshot["processes"] = [dict(pid=11)]
            with self.assertRaisesRegex(RuntimeError, "occupancy changed"):
                require_same_processes("GPU-test", {11})


class ReaderCompletionTests(unittest.TestCase):
    def fixture(self, separate):
        operations = []
        events = []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                operations.append(("wait", self.name, event.index))

        class Event:
            def __init__(self):
                self.index = len(events)
                events.append(self)

            def record(self, stream):
                operations.append(("record", stream.name, self.index))

        graphs = SimpleNamespace(ae_stream=Stream("first"))
        if separate:
            graphs.completion_stream = Stream("tail")
        return SimpleNamespace(cuda=SimpleNamespace(Event=Event)), graphs, operations, events

    def test_legacy_stream_uses_one_reader_fence(self):
        torch, graphs, operations, events = self.fixture(False)
        self.assertIs(ae_completion_stream(graphs), graphs.ae_stream)
        self.assertIs(record_reader_tail(torch, graphs), events[0])
        self.assertEqual(operations, [("record", "first", 0)])

    def test_failure_before_handoff_still_fences_submitted_first_segment(self):
        torch, graphs, operations, events = self.fixture(True)
        tail = record_reader_tail(torch, graphs)
        self.assertIs(tail, events[1])
        self.assertEqual(operations, [("record", "first", 0),
                                      ("wait", "tail", 0), ("record", "tail", 1)])

class NativeScheduleTests(unittest.TestCase):
    def test_d2_schedule_retains_exact_old_nodes(self):
        reference = native_schedule_reference(2, "cpu")
        self.assertEqual(reference["times"], [1.0, 0.5])
        self.assertEqual(reference["dt"], -0.5)
        self.assertEqual(reference["final_time"], 0.0)
        validate_schedule(reference["times"], 2, [1.0, 0.5], label="D2")

    def test_d10_decimal_or_recomputed_nodes_are_rejected(self):
        reference = native_schedule_reference(10, "cpu")
        self.assertEqual(reference["actual_nfe"], 10)
        self.assertLess(reference["final_time"], reference["stop_threshold"])
        self.assertNotEqual(reference["times"][-1], 0.1)
        validate_schedule(reference["times"], 10, reference["times"], label="D10")
        for changed in ([1.0 - i / 10 for i in range(10)],
                        [1.0, 0.5] * 5,
                        reference["times"][:-1]):
            with self.assertRaisesRegex(RuntimeError, "schedule/NFE"):
                validate_schedule(changed, 10, reference["times"], label="changed")

    def test_interior_node_nfe_and_nonfinite_fail_independently(self):
        reference = native_schedule_reference(10, "cpu")["times"]
        changed = reference.copy()
        changed[5] = changed[4]
        with self.assertRaises(RuntimeError):
            validate_schedule(changed, 10, reference, label="wrong middle")
        with self.assertRaises(RuntimeError):
            validate_schedule(reference, 9, reference, label="wrong NFE")
        changed[5] = float("nan")
        with self.assertRaises(RuntimeError):
            validate_schedule(changed, 10, reference, label="nonfinite")

    def test_every_slot_and_prepared_condition_is_checked(self):
        times = native_schedule_reference(10, "cpu")["times"]
        slots = [SimpleNamespace(schedule_times=times.copy(), actual_nfe=10,
                                 condition_times=None),
                 SimpleNamespace(schedule_times=times.copy(), actual_nfe=10,
                                 condition_times=times.copy())]
        oracle = [dict(times=times, actual_nfe=10)]
        graph = SimpleNamespace(slots=slots)
        result = audit_graph_schedules(graph, oracle)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["slots"]), 2)
        slots[1].condition_times[8] = slots[1].condition_times[7]
        with self.assertRaisesRegex(RuntimeError, "Prepared conditions slot 1"):
            audit_graph_schedules(graph, oracle)
        slots[1].condition_times = None
        slots[1].actual_nfe = 2
        with self.assertRaisesRegex(RuntimeError, "Graph slot 1"):
            audit_graph_schedules(graph, oracle)


class IterationTimingTests(unittest.TestCase):
    @staticmethod
    def marks():
        class Event:
            def __init__(self, time):
                self.time = time

            def elapsed_time(self, other):
                return other.time - self.time

        return [Event(value) for value in (0, 3, 30)]

    def test_d10_tail_is_not_labeled_as_one_iteration(self):
        row = iteration_timing_fields(10, self.marks())
        self.assertEqual(row["ae_iteration1_event_ms"], 3)
        self.assertEqual(row["ae_tail2to10_event_ms"], 27)
        self.assertNotIn("ae_iteration2_event_ms", row)
        self.assertEqual(set(row), set(iteration_timing_fields(10)))

    def test_d2_preserves_legacy_iteration_names(self):
        row = iteration_timing_fields(2, self.marks())
        self.assertEqual(row, dict(ae_iteration1_event_ms=3, ae_iteration2_event_ms=27))
        self.assertNotIn("ae_tail2to10_event_ms", row)

    def test_wrong_marker_count_is_not_silently_interpreted(self):
        with self.assertRaisesRegex(RuntimeError, "three|start"):
            iteration_timing_fields(10, self.marks()[:2])


if __name__ == "__main__":
    unittest.main()
