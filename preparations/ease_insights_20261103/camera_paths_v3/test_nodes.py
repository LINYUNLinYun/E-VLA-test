"""Catch the D2/D10 legacy validator mismatch before any new closed-loop run."""
import unittest
from .nodes import scheduled_nfe
from . import worker,cost_records


class NativeNodesContracts(unittest.TestCase):
    def test_D5_D10_exact_native_nodes_accepted_by_both_consumers(self):
        for n in (5,10):
            nodes=[1-i/n for i in range(n+1)]
            self.assertEqual(scheduled_nfe(nodes),n);self.assertEqual(worker.scheduled_nfe(nodes),n)
            self.assertEqual(cost_records.scheduled_nfe(nodes),n)

    def test_D2_and_malformed_schedules_rejected(self):
        for nodes in ([1.,.5,0.],tuple([1-i/5 for i in range(6)]),[1.,.8,.6,.4,.2,float('nan')],[1.,.8,.6,.4,.2,0.],[True,.8,.6,.4,1-4/5,0.]):
            with self.assertRaises(ValueError):scheduled_nfe(nodes)


if __name__=='__main__':unittest.main()
