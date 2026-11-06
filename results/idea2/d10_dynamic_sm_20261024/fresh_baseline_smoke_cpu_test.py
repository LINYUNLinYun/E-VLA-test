"""CPU-only fresh-KV protocol and unchanged-base provenance checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).with_name("fresh_baseline_smoke.py")
SPEC = importlib.util.spec_from_file_location("fresh_smoke", PATH)
fresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fresh)
BASE_SHA = hashlib.sha256(fresh.BASE_PATH.read_bytes()).hexdigest()
fresh.install()


class FreshTests(unittest.TestCase):
    def test_exact_eight_fresh_cases_and_independent_gpu_defaults(self):
        cases = fresh.episode_cases()
        self.assertEqual(len(cases), 8)
        self.assertEqual({row["mode"] for row in cases}, {"fresh_serial"})
        self.assertEqual({(row["suite"], row["task"], row["init"]) for row in cases},
                         {(suite, task, 0) for suite in fresh.base.SUITES for task in (0, 5)})
        args = fresh.parser().parse_args([])
        self.assertFalse(args.execute)
        self.assertEqual((args.gpu_uuid, args.egl_device, args.port), (fresh.GPU_UUID, 3, 19083))
        self.assertTrue(args.output.endswith("closedloop_d10_fresh_gpu0"))
        server, client = fresh.base.environments(args)
        self.assertEqual(server["CUDA_VISIBLE_DEVICES"], fresh.GPU_UUID)
        self.assertEqual(client["CUDA_VISIBLE_DEVICES"], "")

    def artifacts(self, directory):
        result = dict(success=False, actions=6, predictions=2)
        protocol = dict(fresh.base.expected_protocol("fresh_serial"), config="pi05=True")
        rows = [dict(seed=11+i, mode="fresh_serial", observation_request_age=0,
                     prefix_observation_id=i, current_observation_id=i, graph_kv_buffers=2,
                     pool=dict(live_snapshots=2), chunk_actions=[[0.]*7]*10,
                     graph_setup_ms=1 if i == 0 else 0, input_hash=f"input{i}",
                     client_roundtrip_ms=100+i, timing=dict(action_ready_host_ms=90+i)) for i in range(2)]
        (directory/"result.json").write_text(json.dumps(result))
        (directory/"native_protocol.json").write_text(json.dumps(protocol))
        (directory/"infer.jsonl").write_text("\n".join(map(json.dumps, rows)))
        (directory/"telemetry.jsonl").write_text("\n".join(json.dumps(dict(eef_pos=[0,0,0],eef_quat=[0,0,0,1])) for _ in range(7)))
        return rows

    def test_age_zero_all_requests_and_true_task_failure_remain_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.artifacts(directory)
            result = fresh.validate_artifacts(directory, SimpleNamespace(model_seed=lambda *args:11+args[-1]), fresh.episode_cases()[0])
        self.assertTrue(result["valid"])
        self.assertFalse(result["success"])
        self.assertEqual(result["expected_observation_age"], 0)
        self.assertEqual(result["steady_fresh_roundtrip_ms"]["n"], 1)
        self.assertFalse(any("oldkv" in key for key in result))
        self.assertFalse(result["cross_gpu_latency_comparable"])

    def test_old_kv_or_wrong_identity_is_rejected(self):
        for change in (dict(observation_request_age=1), dict(prefix_observation_id=0), dict(mode="async_reuse")):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                rows = self.artifacts(directory)
                rows[1].update(change)
                (directory/"infer.jsonl").write_text("\n".join(map(json.dumps, rows)))
                with self.assertRaisesRegex(RuntimeError, "Fresh current-frame"):
                    fresh.validate_artifacts(directory, SimpleNamespace(model_seed=lambda *args:11+args[-1]), fresh.episode_cases()[0])

    def test_plan_and_worker_dispatch_keep_fresh_semantics_and_both_sources(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(fresh.base, "dependency_probe", return_value=dict(passed=True)):
            args = fresh.parser().parse_args(["--output", temporary])
            plan = fresh.make_plan(args)
            self.assertEqual(plan["total_episodes"], 8)
            self.assertEqual(plan["expected_observation_age"], 0)
            self.assertEqual(plan["protocol"]["mode"], "fresh_serial")
            self.assertEqual(Path(fresh.base.__file__), fresh.WRAPPER)
            self.assertTrue((Path(temporary)/"source"/fresh.WRAPPER.name).is_file())
            self.assertIn(str(fresh.BASE_PATH.relative_to(fresh.ROOT)), plan["source_sha256"])
            self.assertIn(str(fresh.WRAPPER.relative_to(fresh.ROOT)), plan["source_sha256"])
            self.assertIn("no cross-GPU", plan["comparison_scope"])
        self.assertEqual(hashlib.sha256(fresh.BASE_PATH.read_bytes()).hexdigest(), BASE_SHA)

    def test_existing_namespace_gpu_guard_is_reused(self):
        ownership = fresh.base.client_gpu_ownership(dict(processes=[dict(pid=9000)], graphics_processes=[dict(pid=9001)]), {9000}, 111)
        self.assertEqual(ownership["observed_graphics_host_pid"], 9001)
        self.assertEqual(ownership["client_container_pid"], 111)


if __name__ == "__main__":
    unittest.main()
