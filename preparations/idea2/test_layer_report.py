"""CPU evidence-gating regressions; no model or CUDA initialization."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .layer_report import (add_slo, aggregate, base_row, collect_crossframe, collect_layer, collect_legacy, collect_native,
                          completion, model_span, numerical_evidence, ranking_reasons)


def valid_row(**updates):
    return dict(correct=True, normalized_error=dict(max_abs=0.0, mae=0.0),
                decoded_error=dict(max_abs=0.0, mae=0.0), **updates)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.case = self.root / "case"
        (self.case / "source").mkdir(parents=True)
        source = b"# frozen benchmark source\n"
        (self.case / "source" / "runner.py").write_bytes(source)
        self.hashes = {"runner.py": hashlib.sha256(source).hexdigest()}
        self.path = self.case / "result.json"

    def tearDown(self):
        self.temp.cleanup()

    def layer(self):
        rows = [valid_row(phase=phase, mode=mode, pair=index,
                         cpu_cpu_ms=100.0, both_completion_event_ms=75.,
                         prefill_completion_event_ms=70., prefill_gpu_event_ms=40.)
                for phase in ("paired", "continuous") for mode in ("serial", "overlap") for index in range(100)]
        smoke = [valid_row(kv_equal=True, first_actions_error=dict(max_abs=0.0, mae=0.0))]
        return dict(status="passed", args=dict(boundary="kv_ready", count=100, continuous_count=100,
                                               warmup=10, profile=False, iteration_timing=False),
                    rows=rows, smoke=smoke, smoke_status="passed", external_events=dict(passed=True),
                    groups=[dict(phase="paired", correct=True), dict(phase="continuous", mode="serial", correct=True),
                            dict(phase="continuous", mode="overlap", correct=True)], source_sha256=self.hashes)

    def legacy(self):
        return dict(args=dict(steps=2, kind="requests", pairs=[], continuous=["graph_ae_first"],
                              continuous_count=100, count=100, warmup=10),
                    rows=[valid_row(phase="continuous", mode="graph_ae_first", observation_age=1) for _ in range(100)],
                    groups=[dict(phase="continuous", modes=["graph_ae_first"], correct=True)], source_sha256=self.hashes)

    def test_modern_passed_requires_full_source_samples_and_smoke(self):
        data = self.layer()
        self.assertTrue(completion(self.path, data)[0])
        partial = copy.deepcopy(data)
        partial["rows"].pop()
        self.assertFalse(completion(self.path, partial)[0])
        partial = copy.deepcopy(data)
        partial["smoke"] = []
        self.assertFalse(completion(self.path, partial)[0])
        (self.case / "source" / "runner.py").write_text("changed")
        self.assertFalse(completion(self.path, data)[0])

    def test_status_measuring_never_promoted_even_when_observed_rows_are_complete(self):
        data = self.layer()
        data["status"] = "measuring"
        self.assertFalse(completion(self.path, data)[0])

    def test_counts_cannot_be_satisfied_by_duplicate_arm(self):
        data = self.layer()
        for row in data["rows"]:
            row["mode"] = "serial"
        self.assertEqual(completion(self.path, data), (False, "arm_or_phase_sample_counts_incomplete"))

    def test_legacy_no_status_uses_receipt_or_complete_frozen_artifact(self):
        data = self.legacy()
        succeeded = dict(state="process_succeeded", returncode=0)
        self.assertTrue(completion(self.path, data, succeeded)[0])
        self.assertTrue(completion(self.path, data)[0])
        self.assertFalse(completion(self.path, data, dict(state="running"))[0])
        self.assertFalse(completion(self.path, data, dict(state="process_failed", returncode=1))[0])
        data["rows"][-1].pop("correct")
        self.assertFalse(completion(self.path, data, succeeded)[0])

    def test_unknown_error_stays_unknown_not_zero(self):
        evidence = numerical_evidence([dict(correct=True)])
        self.assertIsNone(evidence["numerical_max_abs"])
        self.assertIsNone(evidence["numerical_max_mae"])
        self.assertFalse(evidence["numerical_complete"])

    def test_resource_counts_follow_named_branches(self):
        data = self.layer()
        data["green_context"] = dict(branch_order=["ae", "prefix"], sm_counts=[16, 112], device_sm_count=128)
        row = base_row(self.path, data, self.root, True, "test")
        self.assertEqual((row["actual_ae_sms"], row["actual_prefix_sms"]), (16, 112))
        data["green_context"]["branch_order"] = ["prefix", "ae"]
        row = base_row(self.path, data, self.root, True, "test")
        self.assertEqual((row["actual_ae_sms"], row["actual_prefix_sms"]), (112, 16))
        del data["green_context"]["branch_order"]
        row = base_row(self.path, data, self.root, True, "test")
        self.assertIsNone(row["actual_ae_sms"])
        self.assertIsNone(row["actual_prefix_sms"])

    def test_prefix_cap_does_not_assign_unused_sms_to_ae(self):
        data = self.layer()
        data["args"].update(prefix_sm_cap=120, count=20, continuous_count=30)
        data["green_context"] = dict(branch_order=["prefix", "unused"], sm_counts=[120, 8],
            ae_resource_policy="unrestricted_primary_context", prefix_actual_sm_count=120, device_sm_count=128)
        row = base_row(self.path, data, self.root, True, "test")
        self.assertEqual(row["actual_prefix_sms"], 120)
        self.assertIsNone(row["actual_ae_sms"])
        self.assertEqual(row["requested_prefix_sm_cap"], 120)
        self.assertEqual(row["device_sm_count"], 128)
        self.assertEqual(row["ae_resource_policy"], "unrestricted_primary_context")
        self.assertTrue(ranking_reasons(data, True, data["rows"][:20]))

    def test_worst_smoke_error_not_hidden_by_zero_measured_errors(self):
        data = self.layer()
        data["smoke"][0]["first_actions_error"]["max_abs"] = 0.1
        self.assertFalse(completion(self.path, data)[0])

    def test_unknown_connection_environment_not_filled_with_assumed_default(self):
        data = self.layer()
        row = base_row(self.path, data, self.root, True, "test")
        self.assertIsNone(row["cuda_device_max_connections"])
        data["environment"] = {"CUDA_DEVICE_MAX_CONNECTIONS": "32"}
        self.assertEqual(base_row(self.path, data, self.root, True, "test")["cuda_device_max_connections"], "32")

    def test_diagnostic_profile_and_short_cases_are_not_formal(self):
        data = self.layer()
        data["args"].update(iteration_timing=True, profile=True)
        reasons = ranking_reasons(data, True, data["rows"][:30], paired_count=20)
        for reason in ("profile_only", "iteration_timing_diagnostic", "fewer_than_100_samples", "fewer_than_100_matched_pairs"):
            self.assertIn(reason, reasons)

    def test_model_span_excludes_initial_prepare_gap(self):
        self.assertEqual(model_span(dict(both_completion_event_ms=75, prefill_completion_event_ms=70,
                                        prefill_gpu_event_ms=40)), 45)
        self.assertIsNone(model_span(dict(both_completion_event_ms=75)))

    def test_layer_paired_ratios_are_not_ratio_of_two_medians(self):
        data = self.layer()
        data["rows"] = [valid_row(phase="paired", mode=mode, pair=index, cpu_cpu_ms=value)
                        for index, values in enumerate(((100, 50), (50, 100)))
                        for mode, value in zip(("serial", "overlap"), values)]
        tables = dict(layer=[], paired_ratios=[])
        base = base_row(self.path, data, self.root, True, "test")
        collect_layer(base, data, tables)
        row = next(row for row in tables["paired_ratios"] if row["metric"] == "cpu_cpu_ms")
        self.assertEqual(row["mean"], 1.25)
        self.assertEqual(row["n"], 2)
        self.assertFalse(row["ranking_eligible"])

    def test_old_kv_low_latency_cannot_override_missing_prefills_and_stale_age(self):
        data = self.legacy()
        for index, row in enumerate(data["rows"]):
            row["cpu_cpu_ms"] = 40
            row["observation_age"] = 2 if index < 64 else 3
            if index < 53:
                row["prefill_gpu_event_ms"] = 44
        tables = dict(legacy=[], paired_ratios=[])
        collect_legacy(base_row(self.path, data, self.root, True, "test"), data, tables)
        row = tables["legacy"][0]
        self.assertEqual(row["prefill_submissions"], 53)
        self.assertAlmostEqual(row["observation_age_mean"], 2.36)
        self.assertFalse(row["freshness_ok"])
        self.assertFalse(row["ranking_eligible"])

    def test_layer_completion_and_wait_metrics_preserve_missing_values(self):
        data = self.layer()
        data["rows"] = [valid_row(phase="continuous", mode="serial", prefill_completion_event_ms=30,
            ae_completion_event_ms=40, both_completion_event_ms=40, prefill_gpu_event_ms=20,
            ae_host_wait_ms=12, graph_recaptures=0),
            valid_row(phase="continuous", mode="serial", graph_recaptures=2)]
        tables = dict(layer=[], paired_ratios=[])
        collect_layer(base_row(self.path, data, self.root, True, "test"), data, tables)
        row = tables["layer"][0]
        self.assertEqual(row["ae_completion_event_ms_p50"], 40)
        self.assertEqual(row["prefill_completion_event_ms_n"], 1)
        self.assertEqual(row["ae_host_wait_ms_mean"], 12)
        self.assertEqual(row["graph_recaptures_max"], 2)
        self.assertEqual(row["graph_recaptures_n"], 2)
        self.assertIsNone(row["slot_wait_ms_mean"])
        self.assertEqual(row["slot_wait_ms_n"], 0)

    def test_crossframe_counts_are_observed_not_assumed(self):
        data = self.layer()
        data["args"].update(arrival_period_ms=0, lookahead_delay_ms=8)
        data["groups"] = [dict(phase="continuous", mode="combined", completed_prefill_hz=21.5)]
        data["rows"] = [valid_row(phase="continuous", mode="combined", lookahead_admitted=True,
            ae_done_after_lookahead=False, kv_conditioning_offset_frames=0, encode_cpu_submit_ms=1.5,
            prefill_gpu_event_ms=20, both_completion_event_ms=50, lookahead_delay_ms=7.5),
            valid_row(phase="continuous", mode="combined", lookahead_admitted=False)]
        tables = dict(capacity=[], arrival=[], paired_ratios=[], capacity_trajectories=[])
        collect_crossframe(base_row(self.path, data, self.root, True, "test"), data, tables)
        row = tables["capacity"][0]
        self.assertEqual(row["completed_prefill_hz"], 21.5)
        self.assertEqual(row["requested_lookahead_delay_ms"], 8)
        self.assertEqual(row["lookahead_delay_ms_mean"], 7.5)
        self.assertEqual(row["lookahead_delay_ms_n"], 1)
        self.assertEqual(row["lookahead_admitted_count"], 1)
        self.assertEqual(row["lookahead_admitted_n"], 2)
        self.assertEqual(row["ae_done_after_lookahead_count"], 0)
        self.assertEqual(row["ae_done_after_lookahead_n"], 1)
        self.assertEqual(row["kv_conditioning_offset_frames_mean"], 0)
        self.assertEqual(row["kv_conditioning_offset_frames_n"], 1)
        self.assertEqual(row["encode_cpu_submit_ms_p50"], 1.5)
        self.assertEqual(row["prefill_gpu_event_ms_p50"], 20)
        self.assertEqual(row["both_completion_event_ms_p50"], 50)
        self.assertIsNone(row["graph_recaptures_max"])
        self.assertEqual(row["graph_recaptures_n"], 0)
        data["rows"] = [valid_row(phase="continuous", mode="combined")]
        collect_crossframe(base_row(self.path, data, self.root, True, "test"), data, tables)
        self.assertIsNone(tables["capacity"][-1]["lookahead_admitted_count"])

    def test_periodic_pareto_keeps_drop_as_a_cost_and_separates_arrival_rates(self):
        rows = []
        for case, latency, hz, drop, period in (("a", 100, 20, .1, 50), ("b", 110, 19, .2, 50),
                                               ("c", 120, 18, 0, 50), ("d", 90, 25, 0, 100)):
            rows.append(dict(case=case, mode="combined", ranking_eligible=True, capture_to_action_ms_p95=latency,
                             all_work_hz=hz, drop_fraction=drop, gpu_uuid="gpu", arrival_period_ms=period,
                             bank_sha256="bank", checkpoint="model"))
        tables = dict(arrival=rows, slo_budgets=[])
        add_slo(tables)
        self.assertEqual([row["pareto_latency_throughput_drop"] for row in rows], [True, False, True, True])

    def test_new_run_is_discovered_on_next_snapshot_and_partial_stays_excluded(self):
        self.path.write_text(json.dumps(self.layer()))
        before = aggregate(self.root)
        self.assertEqual(len(before["inventory"]), 1)
        other = self.root / "still_running"
        other.mkdir()
        (other / "result.json").write_text(json.dumps(dict(status="measuring", args=dict(boundary="kv_ready"), rows=[])))
        after = aggregate(self.root)
        self.assertEqual(len(after["inventory"]), 2)
        observed = next(row for row in after["inventory"] if row["case"] == "still_running")
        self.assertFalse(observed["complete"])

    def test_native_repeatability_is_not_claimed_as_crossframework_numeric_error(self):
        source = b"# native frozen source\n"
        (self.case / "source.py").write_bytes(source)
        np.save(self.case / "actions.npy", np.zeros((100, 10, 7)))
        data = dict(backend="original OpenPI JAX", args=dict(count=100, warmup=10, steps=2),
                    rows=[dict(cpu_cpu_ms=80) for _ in range(100)],
                    source_sha256=hashlib.sha256(source).hexdigest(), repeatability=dict(passed=True, max_abs=0.0))
        self.assertTrue(completion(self.path, data)[0])
        tables = dict(native=[])
        collect_native(base_row(self.path, data, self.root, True, "test"), data, tables)
        self.assertIsNone(tables["native"][0]["numerical_max_abs"])
        self.assertEqual(tables["native"][0]["repeatability_max_abs"], 0)
        np.save(self.case / "actions.npy", np.zeros((10, 10, 7)))
        self.assertFalse(completion(self.path, data)[0])


if __name__ == "__main__":
    unittest.main()
