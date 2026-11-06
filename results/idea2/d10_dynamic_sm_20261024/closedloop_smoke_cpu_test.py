"""CPU-only orchestration checks; no server, simulator or GPU is started."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("closedloop_smoke", Path(__file__).with_name("closedloop_smoke.py"))
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


class PlanTests(unittest.TestCase):
    def test_exact_eight_cases_per_mode_with_counterbalanced_order(self):
        cases = smoke.episode_cases()
        self.assertEqual(len(cases), 16)
        for label in smoke.MODES:
            selected = [row for row in cases if row["label"] == label]
            self.assertEqual(len(selected), 8)
            self.assertEqual({(row["suite"], row["task"], row["init"]) for row in selected},
                             {(suite, task, 0) for suite in smoke.SUITES for task in (0, 5)})
        for index in range(8):
            self.assertEqual([row["label"] for row in cases[2*index:2*index+2]],
                             ["O0", "O1"] if index % 2 == 0 else ["O1", "O0"])
        self.assertEqual(smoke.episode_cases(True)[0]["label"], "O1")

    def test_default_cli_only_writes_a_plan(self):
        plan = dict(cases=smoke.episode_cases(), dependency_checks={})
        with tempfile.TemporaryDirectory() as temporary, patch.object(smoke.sys, "argv", ["smoke", "--output", temporary]), patch.object(smoke, "make_plan", return_value=plan), patch.object(smoke, "execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            smoke.main()
            execute.assert_not_called()
            self.assertEqual(json.loads((Path(temporary)/"plan.json").read_text()), plan)

    def test_client_has_no_cuda_and_explicit_egl_mapping(self):
        args = smoke.parser().parse_args([])
        server, client = smoke.environments(args)
        self.assertEqual(server["CUDA_VISIBLE_DEVICES"], smoke.GPU_UUID)
        self.assertEqual(client["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(client["MUJOCO_EGL_DEVICE_ID"], "2")

    def test_health_rejects_an_accidental_model_or_sampling_change(self):
        protocol = dict(smoke.expected_protocol("serial_reuse"), config="pi05=True")
        smoke.check_protocol(protocol, "serial_reuse")
        for key, value in (("H", 50), ("num_steps", 2), ("R", 10), ("ae_priority", -1),
                           ("selective_overlap", "all"), ("preparation_core", "Other")):
            with self.assertRaises(RuntimeError):
                smoke.check_protocol(protocol | {key:value}, "serial_reuse")


class WorkerTests(unittest.TestCase):
    def case(self, temporary):
        case = dict(smoke.episode_cases()[0], directory=str(Path(temporary)/"episode"), port=19082)
        path = Path(temporary)/"request.json"
        path.write_text(json.dumps(case))
        return path, case

    def test_native_episode_arguments_and_final_drain_before_complete(self):
        calls = []
        protocol = dict(smoke.expected_protocol("serial_reuse"), config="pi05=True")
        def rpc(port, request):
            calls.append(request["op"])
            return dict(protocol=protocol) if request["op"] == "health" else dict(ok=True, epoch=2)
        def episode(directory, port, model, suite, task, init, mask, **kwargs):
            self.assertEqual((model, suite, task, init, mask), ("pi05", "libero_object", 0, 0, []))
            self.assertEqual(kwargs, dict(kind="idea2_pi05_d10_oldkv_smoke", reference=False))
            calls.append("native_episode")
            directory.mkdir()
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(directory / "video.mp4"))
            writer.append_data(object())
            writer.close()
            return dict(success=False, video="video.mp4")
        def validate(*args):
            self.assertEqual(calls[-1], "drain")
            return dict(valid=True, success=False, steps=280, predictions=56)
        client = SimpleNamespace(rpc=rpc, run_episode=episode)
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""), patch.dict(sys.modules, {"preparations.idea1_v2_1.runtime_client":client}), patch.object(smoke, "validate_artifacts", side_effect=validate):
            path, case = self.case(temporary)
            receipt = smoke.run_worker(path)
            complete = json.loads((Path(case["directory"])/"complete.json").read_text())
            result = json.loads((Path(case["directory"])/"result.json").read_text())
            self.assertFalse((Path(case["directory"])/"video.mp4").exists())
        self.assertEqual(calls, ["health", "native_episode", "drain"])
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(complete["status"], "valid")
        self.assertFalse(complete["success"])
        self.assertIsNone(result["video"])
        self.assertFalse(result["video_recorded"])
        self.assertEqual(result["video_frames_discarded"], 1)
        self.assertGreaterEqual(receipt["all_work_wall_s"], receipt["response_episode_wall_s"])

    def test_episode_exception_still_drains_and_never_marks_complete(self):
        calls = []
        protocol = dict(smoke.expected_protocol("serial_reuse"), config="pi05=True")
        def rpc(port, request):
            calls.append(request["op"])
            return dict(protocol=protocol) if request["op"] == "health" else dict(ok=True)
        client = SimpleNamespace(rpc=rpc, run_episode=Mock(side_effect=RuntimeError("simulator failed")))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, CUDA_VISIBLE_DEVICES=""), patch.dict(sys.modules, {"preparations.idea1_v2_1.runtime_client":client}):
            path, case = self.case(temporary)
            with self.assertRaisesRegex(RuntimeError, "simulator failed"):
                smoke.run_worker(path)
            self.assertFalse((Path(case["directory"])/"complete.json").exists())
            receipt = json.loads((Path(temporary)/"episode.worker_receipt.json").read_text())
        self.assertEqual(calls, ["health", "drain"])
        self.assertEqual(receipt["status"], "technical_failure")

    def test_client_refuses_visible_cuda(self):
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="0"), self.assertRaisesRegex(RuntimeError, "must have CUDA_VISIBLE_DEVICES empty"):
            smoke.run_worker("unused")


class GpuAttributionTests(unittest.TestCase):
    def test_host_and_container_pid_namespaces_may_differ(self):
        snapshot = dict(processes=[dict(pid=2967437)], graphics_processes=[dict(pid=2971000)])
        result = smoke.client_gpu_ownership(snapshot, {2967437}, 107729)
        self.assertEqual(result["observed_graphics_host_pid"], 2971000)
        self.assertEqual(result["client_container_pid"], 107729)
        self.assertIn("need not match", result["namespace_caveat"])

    def test_wrong_egl_card_has_no_successful_attribution(self):
        snapshot = dict(processes=[dict(pid=2967437)], graphics_processes=[])
        self.assertIsNone(smoke.client_gpu_ownership(snapshot, {2967437}, 107729))

    def test_extra_cuda_or_graphics_processes_are_rejected(self):
        for snapshot in (dict(processes=[dict(pid=2967437), dict(pid=99)], graphics_processes=[dict(pid=2971000)]),
                         dict(processes=[dict(pid=2967437)], graphics_processes=[dict(pid=2971000), dict(pid=99)])):
            with self.assertRaises(RuntimeError):
                smoke.client_gpu_ownership(snapshot, {2967437}, 107729)


if __name__ == "__main__":
    unittest.main()
