"""Prospective synthetic confirmation verifies the fixed two-direction decision."""
import unittest
from preparations.ease_insights_20261103.camera_paths_v4.common import SUITES,CONDITIONS
from preparations.ease_insights_20261103.vlm.analyze import key
from reports.ease_insights_20261103.camera_paths_v4.confirmation_analysis import direction_analysis,apply_verdict
A,B='agent_deep_d2','wrist_deep_d2'


class ConfirmationVerdictTest(unittest.TestCase):
    def fixture(self,n):
        m=dict(suites=list(SUITES),tasks=list(range(10)),confirmation_inits=[10,11,12,13,14])
        scenes=[dict(suite=s,task=t,init=i) for s in SUITES for t in range(10) for i in m['confirmation_inits']]
        r={(cid,key(scene)):dict(success=True) for cid in CONDITIONS for scene in scenes}
        directions=[dict(suite='libero_goal',better=A,worse=B),dict(suite='libero_10',better=B,worse=A)]
        for d in directions:
            for t in range(n):r[d['worse'],key(dict(suite=d['suite'],task=t,init=10))]['success']=False
        return [direction_analysis(d,m,r,scenes,True) for d in directions]
    def test_six_positive_tasks_both_directions_pass_primary_family(self):
        d=self.fixture(6);v=apply_verdict(d,{A:dict(usable=True),B:dict(usable=True)})
        self.assertEqual(v,'complementarity_reproduced');self.assertEqual([x['inference']['primary_holm2_p'] for x in d],[.03125,.03125])
    def test_four_positive_tasks_do_not_pass_despite_positive_mean(self):
        self.assertEqual(apply_verdict(self.fixture(4),{A:dict(usable=True),B:dict(usable=True)}),'not_reproduced')
    def test_quality_or_cost_eligibility_cannot_be_ignored(self):
        self.assertEqual(apply_verdict(self.fixture(6),{A:dict(usable=False),B:dict(usable=True)}),'not_reproduced')
    def test_zero_direction_is_negative(self):
        self.assertEqual(apply_verdict(self.fixture(0),{A:dict(usable=True),B:dict(usable=True)}),'not_reproduced')


if __name__=='__main__':unittest.main()
