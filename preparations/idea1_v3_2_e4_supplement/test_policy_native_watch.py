import unittest
from unittest.mock import patch
from .policy_native_watch import sample


class IdleContract(unittest.TestCase):
    def result(self,processes="",memory=4,utilization=0):
        def query(command,**kwargs):
            if any("query-compute-apps" in part for part in command):return processes
            return "1, GPU-test, %d, %d"%(memory,utilization)
        with patch("subprocess.check_output",query):return sample(1)

    def test_foreign_pid_is_busy_even_zero_utilization(self):
        self.assertFalse(self.result("123, GPU-test, 1 MiB")["idle"])

    def test_memory_or_utilization_cannot_be_ignored(self):
        self.assertFalse(self.result(memory=100)["idle"])
        self.assertFalse(self.result(utilization=1)["idle"])
        self.assertTrue(self.result()["idle"])

    def test_owned_gpu3_never_queried(self):
        with patch("subprocess.check_output") as query:
            with self.assertRaises(ValueError):sample(3)
            query.assert_not_called()


if __name__=="__main__":unittest.main()
