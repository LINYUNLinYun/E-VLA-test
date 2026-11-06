"""Independent arithmetic must match fixed known task and eligibility boundaries."""
import unittest
from reports.ease_insights_20261103.camera_paths_v7.confirmation_statistics_audit import direction,adjusted,verdict


class IndependentConfirmationTest(unittest.TestCase):
    def values(self,tasks):return [1 if t<tasks and i==0 else 0 for t in range(10) for i in range(5)]
    def test_ties_and_six_task_exact_probability(self):
        self.assertEqual(direction([0]*50)['p_one_sided'],1.)
        d=direction(self.values(6));self.assertEqual(d['p_one_sided'],1/64);self.assertEqual(d['task_difference_counts'],[1]*6+[0]*4)
        self.assertGreater(d['task_interval']['low'],0)
    def test_whole_task_zero_boundary_survives_state_heterogeneity(self):
        d=direction([1,-1,1,-1,0]*10);self.assertEqual(d['task_interval'],dict(low=0.,high=0.))
        self.assertLess(d['case_interval']['low'],0)
    def test_holm_preserves_original_order_and_monotonicity(self):
        self.assertEqual(adjusted([.04,.01]),[.04,.02]);self.assertEqual(adjusted([1/64]*2),[1/32]*2)
    def test_positive_pair_still_requires_quality_and_matching_cost(self):
        def run(count=184,ratio=1.5):
            return verdict([direction(self.values(6)),direction(self.values(6))],dict(full_d10=190,dense_d2=185,attn6_d2=count,ffn6_d2=184),dict(full_d10=1.,dense_d2=1.5,attn6_d2=ratio,ffn6_d2=1.5))[0]
        self.assertEqual(run(),'complementarity_reproduced');self.assertEqual(run(150),'not_reproduced');self.assertEqual(run(ratio=1.9),'not_reproduced')
    def test_positive_means_with_four_tasks_are_not_reproduction(self):
        v=verdict([direction(self.values(4)),direction(self.values(4))],dict(full_d10=190,dense_d2=185,attn6_d2=184,ffn6_d2=184),dict(full_d10=1.,dense_d2=1.5,attn6_d2=1.5,ffn6_d2=1.5))
        self.assertEqual(v[0],'not_reproduced')
    def test_noise_repeats_cannot_create_extra_case_units(self):
        with self.assertRaises(ValueError):direction([0]*100)


if __name__=='__main__':unittest.main()
