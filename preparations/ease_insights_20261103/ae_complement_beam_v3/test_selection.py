"""Direct success discovery can select any qualified AE14 neighbor/anchor pair."""
import unittest
from .common import CONDITIONS,SUITES
from .selection import select_pair,key,usable
from reports.ease_insights_20261103.ae_complement_beam_v3.statistics_audit import independent_decision


class SelectionTest(unittest.TestCase):
    def fixture(self):
        m=dict(suites=list(SUITES),tasks=list(range(10)),discovery_inits=[45,46],analytical_conditions=list(CONDITIONS))
        r={(c,key(dict(suite=s,task=t,init=i))):dict(success=True) for c in CONDITIONS for s in SUITES for t in range(10) for i in [45,46]}
        costs=dict(valid_services=8,median_full_over_policy_cycle={c:1.5 if c!='full_d10' else 1. for c in CONDITIONS})
        return m,r,costs

    def fail(self,r,cid,suite,tasks):
        for t in tasks:
            for i in [45,46]:r[cid,key(dict(suite=suite,task=t,init=i))]['success']=False

    def test_qualified_crossing_is_selected_by_actual_success(self):
        m,r,c=self.fixture();self.fail(r,'path_b','libero_goal',[0,1]);self.fail(r,'path_a','libero_10',[0,1])
        v=select_pair(m,r,c,True);self.assertEqual(v['status'],'selected');self.assertEqual(v['selected']['paths'],['path_a','path_b'])
        self.assertEqual([d['suite'] for d in v['selected']['directions']],['libero_goal','libero_10'])

    def test_original_anchor_may_be_one_of_the_two_fixed_budget_paths(self):
        m,r,c=self.fixture();self.fail(r,'path_a','libero_goal',[0,1]);self.fail(r,'ae14_base','libero_10',[0,1])
        self.assertEqual(select_pair(m,r,c,True)['selected']['paths'],['ae14_base','path_a'])

    def test_uniform_dominance_or_single_positive_task_does_not_select(self):
        m,r,c=self.fixture();self.fail(r,'path_a','libero_goal',[0,1]);self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')
        m,r,c=self.fixture();self.fail(r,'path_b','libero_goal',[0]);self.fail(r,'path_a','libero_10',[0,1])
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_quality_and_pair_cost_gates_still_apply(self):
        self.assertFalse(usable(67,80,80));self.assertTrue(usable(68,80,80))
        m,r,c=self.fixture();self.fail(r,'path_b','libero_goal',[0,1]);self.fail(r,'path_a','libero_10',[0,1]);c['median_full_over_policy_cycle']['path_a']=1.9
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_incomplete_matrix_cannot_select(self):
        m,r,c=self.fixture();self.assertEqual(select_pair(m,r,c,False)['status'],'incomplete_discovery')

    def test_independent_integer_decision_matches_full_six_condition_matrix(self):
        m,r,c=self.fixture();self.fail(r,'path_b','libero_goal',[0,1]);self.fail(r,'path_a','libero_10',[0,1])
        counts={cid:{s:[sum(int(r[cid,key(dict(suite=s,task=t,init=i))]['success']) for i in [45,46]) for t in range(10)] for s in SUITES} for cid in CONDITIONS}
        actual=independent_decision(counts,c['median_full_over_policy_cycle'],list(SUITES));expected=select_pair(m,r,c,True)
        self.assertEqual(actual['status'],expected['status']);self.assertEqual(actual['eligible_crossings'],len(expected['eligible_crossings']))
        for k in ['paths','directions','ranking_counts','lexical_tie_key']:self.assertEqual(actual['selected'][k],expected['selected'][k])


if __name__=='__main__':unittest.main()
