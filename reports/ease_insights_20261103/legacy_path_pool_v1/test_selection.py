"""Known profile crossings; no historical quality is opened by these tests."""
import copy
import unittest
from .analyze import SUITES,select,direction


def fixture():
    result={c:{s:[10]*10 for s in SUITES} for c in ('Full18','A','B')}
    result['A']['libero_object']=[7]*10;result['B']['libero_spatial']=[7]*10
    return result


class Selection(unittest.TestCase):
    def test_broad_opposite_profiles_selected(self):
        value=select(fixture());self.assertEqual(value['status'],'selected')
        self.assertEqual(value['selected']['paths'],['A','B']);self.assertEqual(value['selected']['direction_difference_counts'],[30,30])
        self.assertEqual([d['suite'] for d in value['selected']['directions']],['libero_spatial','libero_object'])
    def test_sparse_two_task_effect_is_not_broad_candidate(self):
        data=fixture();data['B']['libero_spatial']=[0,0]+[10]*8;self.assertEqual(select(data)['status'],'no_candidate')
    def test_poor_global_quality_is_not_rescued_by_crossing(self):
        data=fixture();data['A']['libero_goal']=[0]*10;self.assertEqual(select(data)['status'],'no_candidate')
    def test_lexical_tie_and_full_reference_exclusion(self):
        data=fixture();data['C']=copy.deepcopy(data['A']);value=select(data)
        self.assertEqual(value['selected']['paths'],['A','B']);self.assertTrue(all('Full18' not in c['paths'] for c in value['ranked_candidates']))
    def test_selection_does_not_mutate_inputs(self):
        data=fixture();before=copy.deepcopy(data);select(data);self.assertEqual(before,data)
    def test_all_ties_retain_all1024_assignments(self):
        data=fixture();data['B']=copy.deepcopy(data['A']);r=direction(data,dict(suite='libero_spatial',better='A',worse='B'))
        self.assertEqual(r['exploratory_task_signflip_p'],1.);self.assertEqual(r['exploratory_task_bootstrap']['low'],0.)
    def test_consistent_task_effect_has_known_exact_p(self):
        r=direction(fixture(),dict(suite='libero_spatial',better='A',worse='B'))
        self.assertEqual(r['exploratory_task_signflip_p'],1/1024);self.assertAlmostEqual(r['exploratory_task_bootstrap']['low'],.3)


if __name__=='__main__':unittest.main()
