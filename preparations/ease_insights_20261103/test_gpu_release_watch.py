"""CPU-only checks: no tests query a real GPU or signal a process."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import gpu_release_watch as watch


UUIDS = [f"GPU-{index:08x}-0000-0000-0000-000000000000" for index in range(4)]


def inventory(memories=(0, 0, 0, 0)):
    return "".join(f"{index}, {UUIDS[index]}, {memory}\n" for index, memory in enumerate(memories))


def observation(memories=(0, 0, 0, 0), apps=""):
    now = time.time()
    return dict(query_started_at=now-0.01, query_started_at_utc=watch.utc(now-0.01),
                query_finished_at=now, query_finished_at_utc=watch.utc(now), error=None,
                devices=watch.parse_queries(inventory(memories), apps))


def idle_event(event_id=1, run_id="run-a"):
    events, _ = watch.transition_events({}, observation(), {"run_id": run_id}, event_id-1)
    return events[0]


class ParseTests(unittest.TestCase):
    def test_idle_requires_zero_processes_and_memory_strictly_below_500(self):
        devices = watch.parse_queries(inventory((499, 500, 0, 501)), f"{UUIDS[2]}, 999999\n")
        self.assertEqual([item["state"] for item in devices], ["idle", "busy", "busy", "busy"])
        self.assertEqual(devices[2]["host_compute_pids"], [999999])

    def test_invalid_or_incomplete_data_never_becomes_idle(self):
        bad = [inventory().splitlines()[0], "", inventory()+inventory().splitlines()[0],
               inventory(("nan", 0, 0, 0)), inventory(("inf", 0, 0, 0)),
               inventory((-1, 0, 0, 0)), inventory().replace(UUIDS[1], UUIDS[0])]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                watch.parse_queries(raw, "")
        for apps in (f"{UUIDS[0]}, 0", f"{UUIDS[0]}, nan", "garbage", f"{UUIDS[0]}, 3, 1",
                     "GPU-deadbeef-0000-0000-0000-000000000000, 1",
                     f"{UUIDS[0]}, 1\n{UUIDS[0]}, 1\n"):
            with self.subTest(apps=apps), self.assertRaises(ValueError):
                watch.parse_queries(inventory(), apps)

    def test_query_failure_and_partial_query_are_unknown(self):
        for failed in (subprocess.TimeoutExpired("nvidia-smi", 0.85),
                       subprocess.CalledProcessError(1, "nvidia-smi", output="partial", stderr="fail"),
                       FileNotFoundError("no binary")):
            with self.subTest(error=failed), patch.object(watch.subprocess, "run", side_effect=[failed, subprocess.CompletedProcess([], 0, "", "")]):
                result = watch.scan()
                self.assertEqual([item["state"] for item in result["devices"]], ["unknown"]*4)
                self.assertTrue(all(item["idle"] is None for item in result["devices"]))
        with patch.object(watch.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0, "0, invalid, nan", ""), subprocess.CompletedProcess([], 0, "", "")]):
            self.assertTrue(all(item["idle"] is None for item in watch.scan()["devices"]))

    def test_success_queries_all_gpus_with_bounded_timeouts(self):
        with patch.object(watch.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0, inventory(), ""), subprocess.CompletedProcess([], 0, "", "")]) as run:
            self.assertEqual(len(watch.scan()["devices"]), 4)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertFalse(any(arg.startswith("--id") for arg in call.args[0]))
            self.assertEqual(call.kwargs["timeout"], 0.85)


class EventTests(unittest.TestCase):
    def test_initial_release_unknown_and_no_repeated_poll_events(self):
        observed = observation()
        events, seq = watch.transition_events({}, observed, {}, 0)
        self.assertEqual(seq, 4)
        self.assertTrue(all(event["idle_event_kind"] == "initial_idle" for event in events))
        previous = {index: "idle" for index in range(4)}
        self.assertEqual(watch.transition_events(previous, observed, {}, seq), ([], 4))
        previous[0] = "busy"
        previous[1] = "unknown"
        events, seq = watch.transition_events(previous, observed, {}, seq)
        self.assertEqual([event["idle_event_kind"] for event in events], ["released", "idle_after_unknown"])
        self.assertTrue(events[0]["release_event"])
        self.assertFalse(events[1]["release_event"])

    def test_writer_partial_tail_recovery_keeps_sequence_monotonic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"events.jsonl"
            line = json.dumps(idle_event(7))+"\n"
            path.write_text(line+'{"event_id": 8')
            watch.repair_partial_tail(path)
            self.assertEqual(path.read_text(), line)
            self.assertEqual(watch.last_event_id(path), 7)
            self.assertEqual(len(list(Path(folder).glob("events.partial_tail.*"))), 1)

    def test_wait_uses_cursor_and_returns_current_state_for_historical_idle(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            identity = dict(**watch.process_identity(), run_id="run-b")
            current = observation((1000, 0, 0, 0))
            watch.atomic_json(output/"process_receipt.json", identity)
            watch.atomic_json(output/"status.json", dict(**identity, **current))
            event = idle_event(5, "run-a")
            event["query_finished_at"] = current["query_finished_at"]-0.01
            (output/"events.jsonl").write_text(json.dumps(event)+"\n")
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(watch.wait_for_idle_event(output, 4, 0.1, "run-a"), 0)
            result = json.loads(captured.getvalue())
            self.assertEqual(result["current_state"], "busy")
            self.assertTrue(result["watcher_alive"])
            self.assertTrue(result["event_matches_cursor_run"])
            self.assertFalse(result["event_from_current_watcher"])
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(watch.wait_for_idle_event(output, 5, 0.01), 3)
                self.assertEqual(captured.getvalue(), "")

    def test_stale_or_dead_current_observer_cannot_claim_current_idle(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            identity = dict(**watch.process_identity(), run_id="run-a")
            current = observation()
            current["query_finished_at"] = time.time()-100
            watch.atomic_json(output/"process_receipt.json", identity)
            watch.atomic_json(output/"status.json", dict(**identity, **current))
            result = watch.event_context(idle_event(), output)
            self.assertEqual(result["current_state"], "unknown")
            self.assertFalse(result["current_observation_fresh"])
            identity["proc_start_ticks"] += 1
            watch.atomic_json(output/"process_receipt.json", identity)
            self.assertFalse(watch.event_context(idle_event(), output)["watcher_alive"])

    def test_default_wait_does_not_swallow_event_appended_during_baseline_read(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            path = output/"events.jsonl"
            path.write_text(json.dumps(idle_event(1))+"\n")
            original = watch.last_event_id
            def append_during_read(path, byte_limit=None):
                with Path(path).open("a") as stream:
                    stream.write(json.dumps(idle_event(2))+"\n")
                return original(path, byte_limit)
            with patch.object(watch, "last_event_id", side_effect=append_during_read), contextlib.redirect_stdout(io.StringIO()) as captured:
                self.assertEqual(watch.wait_for_idle_event(output, timeout_seconds=0.1), 0)
            self.assertEqual(json.loads(captured.getvalue())["event_id"], 2)

    def test_singleton_denies_second_writer_before_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            with (Path(folder)/"watcher.lock").open("a+") as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                with patch.object(watch, "scan") as scan, contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(watch.watch(folder, max_polls=1), 4)
                    scan.assert_not_called()

    def test_monitor_heartbeat_advances_without_duplicate_events(self):
        with tempfile.TemporaryDirectory() as folder:
            observed = observation((1000, 1000, 1000, 1000))
            with patch.object(watch, "scan", return_value=observed), patch.object(watch.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(watch.watch(folder, max_polls=2), 0)
            self.assertEqual(sleep.call_count, 1)
            self.assertLessEqual(sleep.call_args.args[0], 2)
            output = Path(folder)
            self.assertEqual(json.loads((output/"status.json").read_text())["poll_count"], 2)
            self.assertEqual(len((output/"events.jsonl").read_text().splitlines()), 4)
            receipt = json.loads((output/"process_receipt.json").read_text())
            self.assertEqual(receipt["mode"], "observe_only")
            self.assertEqual(receipt["proc_start_ticks"], watch.process_identity()["proc_start_ticks"])


if __name__ == "__main__":
    unittest.main()
