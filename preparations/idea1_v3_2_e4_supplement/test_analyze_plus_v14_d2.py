import unittest
import numpy as np
from .analyze_plus_v14_d2 import summarize,actual_distance,LABELS


class Analysis(unittest.TestCase):
    def test_partial_common_denominator_and_separate_originals(self):
        jobs=[dict(group=group,source_index=i,case=dict(mother_suite='goal',category='Light',difficulty_level=1))
              for i,group in enumerate(('perturbed350','perturbed350','original40'))]
        results={label:dict(success=label!='v14_d2') for label in LABELS}
        rows,pairs=summarize(jobs,[(jobs[0],results)])
        perturb=[r for r in rows if r['group']=='perturbed350' and r['dimension']=='overall']
        self.assertEqual({r['common_complete_cases'] for r in perturb},{1});self.assertEqual({r['expected_cases'] for r in perturb},{2})
        originals=[r for r in rows if r['group']=='original40' and r['dimension']=='overall']
        self.assertTrue(all(r['success_rate'] is None for r in originals))
        pair=next(r for r in pairs if r['group']=='perturbed350' and r['dimension']=='overall' and r['reference']=='v14_d10')
        self.assertEqual((pair['wins'],pair['losses'],pair['success_difference']),(0,1,-1.))

    def test_actual_steps_only_no_padding_or_synthetic_zero(self):
        first={0:np.array([9.,0,0]),1:np.array([0.,0,0]),2:np.array([.01,0,0]),3:np.array([8.,0,0])}
        second={1:np.array([0.,0,0]),2:np.array([0.,0,0])}
        d=actual_distance(first,second)
        self.assertEqual((d['matched_steps'],d['first_common_step'],d['last_common_step']),(2,1,2))
        self.assertAlmostEqual(d['mean_distance_mm'],5.);self.assertAlmostEqual(d['max_distance_mm'],10.)


if __name__=='__main__':unittest.main()
