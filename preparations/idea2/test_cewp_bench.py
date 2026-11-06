"""CPU regressions for old-KV fairness and response/all-work timing boundaries."""
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from . import cewp_bench as bench


class SelectiveSwitchTests(unittest.TestCase):
    def fixture(self):
        calls = []
        pool = bench.policy_module.Pool()
        class Graphs:
            _kind = "all"
            ae_pending = vision_pending = True
            def __init__(self):
                self.slots = [SimpleNamespace(prefix=SimpleNamespace(cache=[SimpleNamespace(data_ptr=lambda address=address: address)]))
                              for address in (100, 200)]
            @property
            def kind(self):
                return self._kind
            @kind.setter
            def kind(self, value):
                if (any(slot.ready is not None or slot.readers or slot.state != "FREE" for slot in pool.slots)
                        or self.ae_pending or self.vision_pending):
                    raise AssertionError("Changed kind before all producer/reader work drained")
                calls.append(("kind", value))
                self._kind = value
            def reset_events(self):
                calls.append("reset_events")
                self.ae_pending = self.vision_pending = False
        policy = bench.PipelinePolicy.__new__(bench.PipelinePolicy)
        policy.pool = pool
        policy.backend = SimpleNamespace(identity={}, selective_overlap="all",
            model=SimpleNamespace(config=SimpleNamespace(action_horizon=50)),
            graph_engine_class=Mock(side_effect=AssertionError("No recapture allowed")))
        policy.graphs = Graphs()
        policy.request = 1
        policy.submission_order = "ae_first"
        policy.cuda_graphs = True
        policy.ae_priority = 0
        policy.ae_stream = object()
        policy.workers = None
        policy.diagnostics = False
        policy._prefix_timings = []
        pool.slots[0].state = "ACTIVE"
        pool.slots[0].ready = SimpleNamespace(synchronize=lambda: calls.append("drain_producer"))
        pool.slots[0].readers = [SimpleNamespace(synchronize=lambda: calls.append("drain_reader"))]
        return policy, calls

    def test_alias_is_explicit_and_existing_selective_modes_keep_their_meaning(self):
        self.assertEqual(bench.settings("graph_encoder_overlap"), bench.settings("graph_ae_first"))
        for configured in (None, "all", "vision_only", "vlm_only"):
            self.assertEqual(bench.selective_kind("graph_ae_first", configured), configured)
        self.assertEqual(bench.selective_kind("graph_encoder_overlap", "all"), "vision_only")
        for configured in (None, "vision_only", "vlm_only"):
            with self.assertRaisesRegex(ValueError, "requires --selective all"):
                bench.selective_kind("graph_encoder_overlap", configured)

    def test_switch_follows_real_policy_drain_reset_and_preserves_graphs_slots_and_stream(self):
        policy, calls = self.fixture()
        graphs, slots, stream = policy.graphs, tuple(policy.graphs.slots), policy.ae_stream
        bench.configure_arm(policy, "graph_encoder_overlap", "all", mixed_selective=True)
        self.assertEqual(calls, ["drain_producer", "drain_reader", "reset_events", ("kind", "vision_only")])
        self.assertEqual(policy.backend.selective_overlap, "vision_only")
        bench.configure_arm(policy, "graph_ae_first", "all", mixed_selective=True)
        self.assertEqual(calls[-2:], ["reset_events", ("kind", "all")])
        self.assertIs(policy.graphs, graphs)
        self.assertEqual(tuple(policy.graphs.slots), slots)
        self.assertIs(policy.ae_stream, stream)
        policy.backend.graph_engine_class.assert_not_called()

    def test_in_flight_mutation_and_unreleased_reader_are_rejected_without_kind_change(self):
        policy, calls = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "after Pool drain/reset"):
            bench.switch_selective_after_reset(policy, "vision_only")
        policy.pool.slots[0].leases = 1
        with self.assertRaisesRegex(RuntimeError, "Finish and release"):
            bench.configure_arm(policy, "graph_encoder_overlap", "all", mixed_selective=True)
        self.assertEqual(policy.graphs.kind, "all")
        self.assertEqual(policy.backend.selective_overlap, "all")
        self.assertEqual(calls, [])

    def test_pending_graph_events_or_replaced_graph_owner_are_rejected(self):
        policy, _ = self.fixture()
        policy.reset()
        policy.graphs.ae_pending = True
        with self.assertRaisesRegex(RuntimeError, "events must be reset"):
            bench.switch_selective_after_reset(policy, "vision_only")
        policy.configure = lambda **kwargs: setattr(policy, "graphs", object())
        with self.assertRaisesRegex(RuntimeError, "same Graph owner"):
            bench.configure_arm(policy, "graph_encoder_overlap", "all", mixed_selective=True)

    def test_historical_configure_does_not_add_a_selector_mutation(self):
        policy = SimpleNamespace(graphs=SimpleNamespace(kind="vision_only"), configure=Mock())
        bench.configure_arm(policy, "graph_ae_first", "vision_only")
        policy.configure.assert_called_once_with(**bench.settings("graph_ae_first"))
        self.assertEqual(policy.graphs.kind, "vision_only")

    def test_recorder_reports_actual_kind_and_one_capture_engine_across_both_arms(self):
        policy, _ = self.fixture()
        recorder = bench.Recorder(policy.backend)
        recorder.intercept = contextlib.nullcontext
        recorder.jobs = []
        recorder.normalized = bench.np.zeros(1)
        recorder.output_begin = 0
        def predict(*args):
            policy.request += 1
            return dict(pool=dict(live_snapshots=0), snapshot=dict(epoch=policy.pool.epoch, slot=0, generation=1),
                request=policy.request, prefix_observation_id=policy.request, observation_request_age=1,
                timing=dict(action_ready_host_ms=0, response_host_ms=0), graph_setup_ms=0,
                diagnostics={}, peak_allocated=0, peak_reserved=0, source_receipt_age_at_response_ms=0,
                actions=bench.np.zeros(1))
        policy.predict = predict
        records = [dict(obs={}, instruction="task")]
        with patch.object(bench, "event", return_value=Mock()), patch.object(bench, "span", side_effect=lambda label: contextlib.nullcontext()), patch.object(bench, "cache_tensors", side_effect=lambda cache: cache):
            recorder.predict(policy, records, 0, 1, mode="graph_ae_first", phase="warmup", measured=False)
            for name in ("graph_encoder_overlap", "graph_ae_first"):
                bench.configure_arm(policy, name, "all", mixed_selective=True)
                recorder.predict(policy, records, 0, 1, mode=name, phase="paired")
        self.assertEqual([row["selective_kind"] for row in recorder.rows], ["vision_only", "all"])
        self.assertEqual(recorder.capture_engines, 1)
        self.assertEqual([row["steady_recapture"] for row in recorder.rows], [0, 0])
        self.assertEqual(recorder.rows[0]["kv_addresses"], recorder.rows[1]["kv_addresses"])
        policy.backend.graph_engine_class.assert_not_called()


class FreshnessTests(unittest.TestCase):
    def row(self, **changes):
        return dict(phase="continuous", mode="graph_ae_first", request=1,
                    prefill_submissions=1, observation_age=1, steady_recapture=0) | changes

    def test_contract_is_opt_in_and_does_not_apply_to_bootstrap_or_modules(self):
        bad = self.row(observation_age=3, prefill_submissions=0)
        self.assertIsNone(bench.freshness_gate([bad], False)["passed"])
        result = bench.freshness_gate([self.row(), self.row(phase="bootstrap", observation_age=0),
                                     self.row(phase="isolated_ae", prefill_submissions=0)], True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["measured_requests"], 1)

    def test_missing_duplicate_prefill_old_or_fresh_kv_and_recapture_each_fail(self):
        for change in (dict(prefill_submissions=0), dict(prefill_submissions=2),
                       dict(observation_age=0), dict(observation_age=2), dict(steady_recapture=1)):
            with self.subTest(change=change):
                result = bench.freshness_gate([self.row(**change)], True)
                self.assertFalse(result["passed"])
                self.assertEqual(len(result["violations"]), 1)

    def test_reverse_changes_order_without_changing_arm_names_or_pair_index(self):
        names = ("graph_serial", "graph_ae_first")
        self.assertEqual(bench.arm_order(names, 0), names)
        self.assertEqual(bench.arm_order(names, 1), names[::-1])
        self.assertEqual(bench.arm_order(names, 0, True), names[::-1])
        self.assertEqual(bench.arm_order(names, 1, True), names)


class DrainBoundaryTests(unittest.TestCase):
    def test_gpu_drain_ends_before_statistics_and_cleanup(self):
        clock = SimpleNamespace(value=.012)
        calls = []
        def finish_prefix():
            calls.append("prefix")
            clock.value += .004
        policy = SimpleNamespace(background=SimpleNamespace(synchronize=finish_prefix),
                                 ae_stream=SimpleNamespace(synchronize=lambda: calls.append("ae")),
                                 pool=SimpleNamespace(poll=Mock(side_effect=AssertionError("Not part of drain"))))
        rows = [dict(phase="continuous", host_begin=0., host_end=.010, prefill_submissions=1)]
        result = bench.drain_continuous(policy, rows, now=lambda: clock.value)
        self.assertEqual(calls, ["prefix", "ae"])
        self.assertAlmostEqual(result["response_wall_seconds"], .010)
        self.assertAlmostEqual(result["all_work_wall_seconds"], .016)
        self.assertAlmostEqual(result["drain_wall_seconds"], .004)
        self.assertAlmostEqual(result["response_to_drain_begin_seconds"], .002)
        # An intentionally slow CPU Event-statistics pass must not move the
        # already frozen GPU drain boundary or masquerade as GPU waiting.
        clock.value += .100
        self.assertAlmostEqual(result["all_work_wall_seconds"], .016)
        self.assertEqual(result["completed_measured_prefills"], 1)
        self.assertAlmostEqual(result["response_hz"], 100.)
        self.assertAlmostEqual(result["all_work_hz"], 62.5)

    def test_empty_or_noncontinuous_rows_do_not_trigger_stream_fences(self):
        policy = SimpleNamespace(background=Mock(), ae_stream=Mock())
        self.assertIsNone(bench.drain_continuous(policy, []))
        self.assertIsNone(bench.drain_continuous(policy, [dict(phase="paired")]))
        policy.background.synchronize.assert_not_called()
        policy.ae_stream.synchronize.assert_not_called()

    def test_event_statistics_count_completed_prefills_without_owning_kv(self):
        class Event:
            def __init__(self, stamp):
                self.stamp = stamp
                self.calls = 0
            def synchronize(self):
                self.calls += 1
            def elapsed_time(self, other):
                return other.stamp - self.stamp
        jobs = [(name, SimpleNamespace(start=Event(start), done=Event(end),
                    cpu_begin=.001, cpu_end=.002, done_at_submit=False))
                for name, start, end in (("ae", 1., 6.), ("prefill", 2., 9.))]
        row = {}
        recorder = bench.Recorder(None)
        recorder.pending = [(row, Event(0.), jobs)]
        recorder.finish_events()
        self.assertEqual(row["both_completion_event_ms"], 9.)
        self.assertEqual(row["completed_prefill_count"], 1)
        self.assertEqual(recorder.pending, [])
        self.assertTrue(all(job.done.calls == 1 for _, job in jobs))

    def test_summary_preserves_response_wall_and_adds_all_work_wall(self):
        row = dict(phase="continuous", mode="graph_ae_first", request=1, epoch=1,
                   observation_age=1, snapshot=dict(slot=0, generation=1),
                   host_begin=0., host_end=.01, cpu_cpu_ms=10., prefill_submissions=1,
                   prefill_done_at_response=False, peak_allocated=0, peak_reserved=0,
                   steady_recapture=0, correct=True)
        result = dict(rows=[row], groups=[dict(phase="continuous", modes=["graph_ae_first"],
                      trajectory=dict(response_wall_seconds=.01, all_work_wall_seconds=.015,
                                      all_work_hz=1/.015, drain_wall_seconds=.005))])
        with tempfile.TemporaryDirectory() as directory:
            bench.write_results(Path(directory), result)
            saved = json.loads((Path(directory)/"result.json").read_text())
        summary = saved["summary"]["continuous/graph_ae_first"]
        self.assertEqual(summary["trajectory_wall_seconds"], .01)
        self.assertEqual(summary["all_work_wall_seconds"], .015)


class MainCompletionTests(unittest.TestCase):
    def fixture(self, directory):
        bank = Path(directory)/"bank.pkl"
        bank.write_bytes(pickle.dumps([]))
        output = Path(directory)/"out"
        argv = ["cewp_bench", "--checkpoint", "unused", "--bank", str(bank), "--output", str(output)]
        return output, argv

    def test_alias_cli_requires_all_and_rejects_unsupported_prefill_first_pair(self):
        for extra in (["--pairs", "graph_ae_first:graph_encoder_overlap"],
                      ["--selective", "vision_only", "--pairs", "graph_ae_first:graph_encoder_overlap"],
                      ["--selective", "all", "--pairs", "graph_prefill_first:graph_encoder_overlap"]):
            with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
                output, argv = self.fixture(directory)
                with patch.object(bench.sys, "argv", argv + extra), patch.object(bench, "_run") as run:
                    with self.assertRaises(SystemExit):
                        bench.main()
                    run.assert_not_called()
                self.assertFalse(output.exists())

    def test_optional_guard_precedes_model_run_and_reverse_flag_is_recorded(self):
        order = []
        def guard(uuid):
            order.append(("guard", uuid))
            os.environ["CUDA_VISIBLE_DEVICES"] = uuid
            return dict(uuid=uuid, graphics_processes=[])
        def run(args, dest, result, pairs, seed_for):
            order.append(("run", os.environ["CUDA_VISIBLE_DEVICES"]))
            self.assertTrue(args.reverse_order)
            result["groups"] = [dict(phase="continuous", modes=["graph_ae_first"], status="passed")]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=False):
            output, argv = self.fixture(directory)
            argv += ["--gpu-uuid", "GPU-test", "--reverse-order"]
            with patch.object(bench.sys, "argv", argv), patch.object(bench, "require_idle_gpu", side_effect=guard), patch.object(bench, "_run", side_effect=run):
                bench.main()
            saved = json.loads((output/"result.json").read_text())
        self.assertEqual(order, [("guard", "GPU-test"), ("run", "GPU-test")])
        self.assertEqual(saved["status"], "passed")
        self.assertTrue(saved["completed"])
        self.assertIn("--reverse-order", saved["command"])
        self.assertIn("CUDA_DEVICE_MAX_CONNECTIONS", saved["environment"])
        self.assertEqual(saved["environment"]["CUDA_VISIBLE_DEVICES"], "GPU-test")

    def test_historical_unguarded_mode_does_not_query_gpu_and_failure_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            output, argv = self.fixture(directory)
            with patch.object(bench.sys, "argv", argv), patch.object(bench, "require_idle_gpu") as guard, patch.object(bench, "_run", side_effect=RuntimeError("test failure")):
                with self.assertRaisesRegex(RuntimeError, "test failure"):
                    bench.main()
                guard.assert_not_called()
            saved = json.loads((output/"result.json").read_text())
        self.assertEqual(saved["status"], "failed")
        self.assertFalse(saved["completed"])
        self.assertFalse(saved["performance_eligible"])

    def test_group_drains_before_statistics_and_closes_graphs_before_oracle(self):
        calls = []
        backend = SimpleNamespace(device="cpu", identity={})
        class Policy:
            def __init__(self, backend, **kwargs):
                self.request = 0
                self.background = SimpleNamespace(synchronize=lambda: calls.append("drain_prefix"))
                self.ae_stream = SimpleNamespace(synchronize=lambda: calls.append("drain_ae"))
            def reset(self):
                self.request = 0
            def close(self):
                calls.append("close_graphs")
        class Recorder:
            def __init__(self, backend):
                self.rows = []
                self.capture_engines = 1
            def predict(self, policy, records, index, seed, *, mode, phase, measured=True):
                policy.request += 1
                if measured:
                    begin = bench.time.perf_counter()
                    self.rows.append(dict(phase=phase, mode=mode, request=policy.request, epoch=1,
                        observation_age=1, snapshot=dict(slot=0, generation=1), host_begin=begin,
                        host_end=bench.time.perf_counter(), cpu_cpu_ms=0., prefill_submissions=1,
                        prefill_done_at_response=False, peak_allocated=0, peak_reserved=0,
                        steady_recapture=0))
                return dict(graph_setup_ms=0)
            def finish_events(self):
                calls.append("event_statistics")
            def verify(self, *args):
                calls.append("oracle")
                if "close_graphs" not in calls:
                    raise AssertionError("Reference KV allocated while Graph KV owners remain")
                for row in self.rows:
                    row["correct"] = True
                return True
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            bank = dest/"bank.pkl"
            bank.write_bytes(pickle.dumps([dict(instruction="task"), dict(instruction="task")]))
            args = SimpleNamespace(checkpoint="unused", model_config="pi05_libero", steps=10,
                await_previous_prefix=True, hoist_schedule=False, hoist_modulation=False,
                selective=None, bank=str(bank), instruction_index=0, gpu_uuid=None,
                kind="requests", continuous=["graph_ae_first"], compile=False, warmup=2,
                profile=False, continuous_count=1, reset_every=0, count=1,
                max_error=1e-5, mae_error=1e-6, reverse_order=False)
            result = dict(rows=[], groups=[])
            with patch.object(bench, "load_backend", return_value=backend), patch.object(bench, "PipelinePolicy", Policy), patch.object(bench, "Recorder", Recorder), patch.object(bench.torch.cuda, "get_device_properties", return_value=SimpleNamespace(uuid="GPU-test")), patch.object(bench.torch.cuda, "reset_peak_memory_stats"):
                bench._run(args, dest, result, [], lambda index: 1701 + index)
        self.assertEqual(calls, ["drain_prefix", "drain_ae", "event_statistics", "close_graphs", "oracle"])
        self.assertEqual(result["groups"][0]["status"], "passed")
        self.assertTrue(result["groups"][0]["freshness_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
