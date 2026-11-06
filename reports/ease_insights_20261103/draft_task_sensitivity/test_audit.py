"""Combinatorial task-sign and Holm boundaries for four initial states per task."""
import unittest
from .audit import signflip,holm


class TaskSensitivityContracts(unittest.TestCase):
    def test_all_positive_tasks_require_all_positive_signs(self):
        self.assertEqual(signflip([4]*10)['p_one_sided'],1/1024)
        self.assertEqual(signflip([4]*5+[3]*5)['p_one_sided'],1/1024)

    def test_zero_tasks_remain_in_the_exact_assignment_count(self):
        r=signflip([4]*9+[0]);self.assertEqual(r['extreme_assignments'],2);self.assertEqual(r['p_one_sided'],1/512)
        self.assertEqual(signflip([0]*10)['p_one_sided'],1.)

    def test_negative_direction_is_not_positive_evidence(self):
        self.assertEqual(signflip([-4]*10)['p_one_sided'],1.)

    def test_two_fixed_hypotheses_use_Holm(self):
        self.assertEqual(holm([1/1024,1/1024]),[1/512,1/512])
        self.assertEqual(holm([1.,1/1024]),[1.,1/512])


if __name__=='__main__':unittest.main()
