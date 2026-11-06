import unittest
import numpy as np
from .common import SUITES,select_q,tag,phase2_qs,metrics


class Selection(unittest.TestCase):
    def rows(self,fresh_grip=.05):
        rows=[]
        for suite in SUITES:
            for noise in ('e0','e1'):
                for name,t,g in [('fresh2',4.,fresh_grip),(tag(.3),6.,.02),(tag(.9),3.,.06),(tag(1.),4.,fresh_grip)]:
                    rows.append(dict(split='calibration',suite=suite,task=0,noise=noise,reference=noise,horizon=5,candidate=name,translation_mm=t,gripper_sign=g))
        return rows

    def test_gripper_constraint_and_fresh_endpoint_not_called_reuse(self):
        choice=select_q(self.rows(),[.3,.9,1.],'fresh2')
        self.assertEqual(choice['q'],.3);self.assertTrue(choice['has_gripper_not_higher_candidate'])

    def test_no_feasible_q_is_recorded(self):
        choice=select_q(self.rows(0.),[.3,.9,1.],'fresh2')
        self.assertEqual(choice['q'],.3);self.assertFalse(choice['has_gripper_not_higher_candidate'])

    def test_heldout_cannot_enter_parameter_selection(self):
        rows=self.rows();rows[0]['split']='heldout'
        with self.assertRaises(AssertionError):select_q(rows,[.3,.9,1.],'fresh2')

    def test_stage2_qs_stay_below_one_except_explicit_endpoint(self):
        self.assertEqual(phase2_qs(.9),[.8,.9,.95,1.])
        self.assertEqual(phase2_qs(.1),[.05,.1,.2,1.])

    def test_translation_clipping_and_gripper_are_separate(self):
        a=np.zeros((50,7));ref=np.zeros((50,7));raw=np.zeros((50,32))
        a[:5,0]=2.;ref[:5,0]=1.;a[:5,6]=-1.
        m=metrics(a,raw,ref,raw,5)
        self.assertEqual(m['translation_mm'],0.);self.assertEqual(m['translation_l2'],1.);self.assertEqual(m['gripper_sign'],1.)


if __name__=='__main__':unittest.main()
