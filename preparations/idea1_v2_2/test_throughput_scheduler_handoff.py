"""CPU-only scheduler transaction tests. All process signals are mocked."""
from contextlib import contextmanager, ExitStack
import copy
import fcntl
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

from common import atomic_json
import throughput_scheduler_handoff as helper


def identity(pid, ppid):
    return dict(pid=pid, ppid=ppid, start_ticks=str(pid*10), state='R', argv=['known'], cwd='/known')


class SchedulerHandoffTests(unittest.TestCase):
    def test_other_stage_and_extra_scheduler_worker_rejected_before_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            scheduler, worker, server = identity(1, 9), identity(2, 1), identity(3, 2)
            scheduler.update(cwd=str(helper.HERE), argv=['python', str(helper.HERE/'runtime_runner.py'),
                'run', '--model', 'pi0', '--attempt', str(attempt), '--launch'])
            worker.update(cwd=str(helper.HERE), argv=['python', str(helper.HERE/'runtime_runner.py'),
                'worker', '--model', 'pi0', '--attempt', str(attempt), '--gpu', '0', '--shard', '0', '--launch'])
            server.update(cwd=str(helper.HERE), argv=['python', str(helper.HERE/'runtime_server.py'), '--model', 'pi0',
                '--output-dir', str(attempt/'services/pi0_gpu0'), '--port', '123', '--launch'])
            atomic_json(attempt/'services/pi0_gpu0/service_lease.json', dict(pid=3, start_ticks='30', gpu='0', port=123))
            values = {1: scheduler, 2: worker, 3: server}
            with patch.object(helper, 'verify_revision', return_value=({'source_attempt': str(attempt)}, {})), \
                 patch.object(helper, 'process_identity', side_effect=lambda pid: values[pid]), \
                 patch.object(helper, 'lock_fds', return_value={}), \
                 patch.object(helper, 'compiler_tree', return_value=[]), \
                 patch.object(helper, 'live_children', return_value=[worker, identity(5, 1)]):
                with self.assertRaises(ValueError): helper.inspect_chain('/revision', 1, 2, 3)
                scheduler['argv'][2] = 'benchmark'
                with self.assertRaises(ValueError): helper.inspect_chain('/revision', 1, 2, 3)

    def run_fixture(self, root, reserved=False, worker_wait_error=False, extra_worker_child=False, pending=False, interrupt_after_stop=False):
        destination, attempt = root/'handoff', root/'attempt'
        destination.mkdir()
        atomic_json(destination/'manifest.json', {'fixture': True})
        atomic_json(attempt/'episode_ledger.json', dict(entries=[dict(status='reserved' if reserved else 'valid', pid=2, start_ticks='20')]))
        chain = dict(scheduler=identity(1, 99), worker=identity(2, 1), server=identity(3, 2),
                     compiler_tree=[identity(4, 3)], service_lease_sha256='lease')
        plan = dict(throughput_revision=str(root/'revision'), chain=copy.deepcopy(chain))
        signals, dead = [], set()
        def send(fd, number):
            pid = fd-100
            signals.append((pid, number))
            if pid == 2 and number == signal.SIGTERM:
                with (attempt/'episode_ledger.lock').open('a') as stream:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                dead.add(2)
            elif pid == 3 and number == signal.SIGTERM:
                dead.update((3, 4))
            elif number == signal.SIGSTOP:
                chain['scheduler']['state'] = 'T'
                if interrupt_after_stop:
                    signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            elif number == signal.SIGCONT:
                # Busy-ledger failure must have released the original transaction lock.
                with (attempt/'episode_ledger.lock').open('a') as stream:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                chain['scheduler']['state'] = 'R'
            elif pid == 1 and number == signal.SIGKILL:
                self.assertTrue({2, 3, 4} <= dead)
                dead.add(1)
        def inspect(*args, **kwargs):
            return copy.deepcopy(chain)
        def children(pid, mapping=None):
            if pid == 2:
                return [chain['server']] + ([identity(5, 2)] if extra_worker_child else [])
            return []
        def wait(fd, timeout=10):
            if fd == 102 and worker_wait_error:
                raise TimeoutError('mocked exit failure after worker signal')
        @contextmanager
        def available(path):
            yield
        with ExitStack() as stack:
            for target, value in [
                ('verify_plan', lambda path: plan),
                ('verify_revision', lambda path: ({'source_attempt': str(attempt)}, {})),
                ('inspect_chain', inspect), ('live_children', children),
                ('process_identity', lambda pid: copy.deepcopy(chain['scheduler'])),
                ('start_ticks', lambda pid: None if pid in dead else str(pid*10)),
                ('pending_stop_handler', lambda pid: pending),
                ('wait_pidfd', wait), ('exclusive', available)]:
                stack.enter_context(patch.object(helper, target, side_effect=value))
            stack.enter_context(patch.object(helper.os, 'pidfd_open', side_effect=lambda pid: pid+100, create=True))
            stack.enter_context(patch.object(helper.os, 'close'))
            stack.enter_context(patch.object(helper.signal, 'pidfd_send_signal', side_effect=send, create=True))
            try:
                result = helper.execute(destination, timeout=.03 if reserved else 2)
            except (TimeoutError, ValueError, KeyboardInterrupt) as error:
                result = error
        return signals, result, destination

    def test_success_stop_boundary_server_subtree_kill_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            signals, result, destination = self.run_fixture(Path(directory))
            self.assertEqual(signals, [(1, signal.SIGSTOP), (2, signal.SIGTERM), (3, signal.SIGTERM), (1, signal.SIGKILL)])
            self.assertTrue(result['complete'])
            self.assertTrue((destination/'boundary_ledger.json').exists())
            self.assertFalse((destination.parent/'attempt'/'PAUSE').exists())

    def test_active_reservation_releases_lock_and_only_resumes_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            signals, result, destination = self.run_fixture(Path(directory), reserved=True)
            self.assertIsInstance(result, TimeoutError)
            self.assertEqual(signals, [(1, signal.SIGSTOP), (1, signal.SIGCONT)])
            self.assertFalse((destination/'complete.json').exists())

    def test_operator_interrupt_after_stop_is_logged_and_safely_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            signals, result, destination = self.run_fixture(Path(directory), interrupt_after_stop=True)
            self.assertIsInstance(result, KeyboardInterrupt)
            self.assertEqual(signals, [(1, signal.SIGSTOP), (1, signal.SIGCONT)])
            self.assertTrue(helper.read(destination/'state.json')['events'][-1]['scheduler_resumed'])

    def test_pending_handler_prohibits_automatic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            signals, _, destination = self.run_fixture(Path(directory), reserved=True, pending=True)
            self.assertEqual(signals, [(1, signal.SIGSTOP)])
            self.assertTrue(helper.read(destination/'state.json')['events'][-1]['manual_recovery_required'])

    def test_any_worker_signal_permanently_prohibits_automatic_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            signals, result, destination = self.run_fixture(Path(directory), worker_wait_error=True)
            self.assertIsInstance(result, TimeoutError)
            self.assertEqual(signals, [(1, signal.SIGSTOP), (2, signal.SIGTERM)])
            self.assertTrue(helper.read(destination/'state.json')['events'][-1]['manual_recovery_required'])

    def test_unreviewed_worker_child_rejects_without_worker_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            signals, result, _ = self.run_fixture(Path(directory), extra_worker_child=True)
            self.assertIsInstance(result, ValueError)
            self.assertEqual(signals, [(1, signal.SIGSTOP), (1, signal.SIGCONT)])

    def test_pid_or_compiler_identity_change_rejected(self):
        expected = dict(scheduler=identity(1, 9), worker=identity(2, 1), server=identity(3, 2),
                        compiler_tree=[identity(4, 3)], service_lease_sha256='lease')
        for target in ('scheduler', 'worker', 'server', 'compiler_tree'):
            actual = copy.deepcopy(expected)
            (actual[target][0] if target == 'compiler_tree' else actual[target])['start_ticks'] = 'reused'
            with self.subTest(target=target), self.assertRaises(ValueError):
                helper.same_chain(actual, expected)


if __name__ == '__main__':
    unittest.main()
