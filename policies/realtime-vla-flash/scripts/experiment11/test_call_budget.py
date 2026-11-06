import tempfile
import unittest
from pathlib import Path
from scripts.experiment11.call_budget import CallBudget
from scripts.experiment11 import launch_shadow_supplement as supplement


class BudgetTest(unittest.TestCase):
    def test_failed_calls_remain_charged_and_limit_is_shared(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'budget.sqlite'
            a, b = CallBudget(path, 2), CallBudget(path, 2)
            call = a.reserve({'sample': 'a'})
            a.finish(call, 'failed', {})
            b.reserve({'sample': 'b'})
            with self.assertRaises(RuntimeError):
                a.reserve({'sample': 'c'})
            self.assertEqual(b.summary()['charged_calls'], 2)
            with self.assertRaises(RuntimeError):
                CallBudget(path, 3)

    def test_attempt_isolation_and_worker_propagation(self):
        try:
            supplement.set_attempt('test_round2')
            self.assertIn('test_round2', str(supplement.STATUS))
            self.assertIn('test_round2', str(supplement.OUTPUT))
            command = supplement.child_command({'suite': 'libero_object'}, 0)
            self.assertEqual(command[-2:], ['--attempt-id', 'test_round2'])
            with self.assertRaises(ValueError):
                supplement.set_attempt('../bad')
        finally:
            supplement.set_attempt('attempt1_20261011')


if __name__ == '__main__':
    unittest.main()
