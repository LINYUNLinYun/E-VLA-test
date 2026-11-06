"""Fixed-direction discovery tests; no proxy/suite/path reselection is allowed."""
import copy
import unittest
from .common import CONDITIONS, DIRECTIONS, SUITES
from .selection import select_pair, key
from reports.ease_insights_20261103.ae_complement_beam_v2.statistics_audit import independent_decision


class SelectionTest(unittest.TestCase):
    def fixture(self):
        manifest=dict(suites=list(SUITES),tasks=list(range(10)),discovery_inits=[45,46],analytical_conditions=list(CONDITIONS))
        scenes=[dict(suite=s,task=t,init=i) for s in SUITES for t in range(10) for i in [45,46]]
        records={(cid,key(scene)):dict(success=True) for cid in CONDITIONS for scene in scenes}
        costs=dict(valid_services=4,median_full_over_policy_cycle={c:1.5 if c!='full_d10' else 1. for c in CONDITIONS})
        return manifest,records,costs

    def fail(self, records, cid, suite, tasks):
        for task in tasks:
            for init in [45,46]:records[cid,key(dict(suite=suite,task=task,init=init))]['success']=False

    def crossing(self):
        m,r,c=self.fixture();self.fail(r,'path_b','libero_object',[0,1]);self.fail(r,'path_a','libero_spatial',[0,1]);return m,r,c

    def test_exact_proxy_pair_and_directions_are_selected(self):
        m,r,c=self.crossing();v=select_pair(m,r,c,True)
        self.assertEqual(v['status'],'selected');self.assertEqual(v['selected']['paths'],['path_a','path_b'])
        self.assertEqual([{k:d[k] for k in ['suite','better','worse']} for d in v['selected']['directions']],DIRECTIONS)
        self.assertEqual(len(v['eligible_crossings']),1)

    def test_reversed_directions_do_not_get_reselected(self):
        m,r,c=self.fixture();self.fail(r,'path_a','libero_object',[0,1]);self.fail(r,'path_b','libero_spatial',[0,1])
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_crossing_in_other_suites_does_not_get_reselected(self):
        m,r,c=self.fixture();self.fail(r,'path_b','libero_goal',[0,1]);self.fail(r,'path_a','libero_10',[0,1])
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_one_positive_task_cannot_satisfy_direction(self):
        m,r,c=self.fixture();self.fail(r,'path_b','libero_object',[0]);self.fail(r,'path_a','libero_spatial',[0,1])
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_cost_and_quality_gates_cannot_be_replaced_by_crossing(self):
        m,r,c=self.crossing();c['median_full_over_policy_cycle']['path_a']=1.14
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')
        m,r,c=self.crossing();c['median_full_over_policy_cycle']['path_a']=1.8
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')
        m,r,c=self.crossing();self.fail(r,'path_a','libero_goal',list(range(10)))
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_incomplete_discovery_cannot_select(self):
        m,r,c=self.crossing();self.assertEqual(select_pair(m,r,c,False)['status'],'incomplete_discovery')

    def test_independent_integer_decision_matches_fixed_cases(self):
        for fixture in [self.fixture,self.crossing]:
            m,r,c=fixture();counts={cid:{s:[sum(int(r[cid,key(dict(suite=s,task=t,init=i))]['success']) for i in [45,46]) for t in range(10)] for s in SUITES} for cid in CONDITIONS}
            actual=independent_decision(counts,c['median_full_over_policy_cycle'],list(SUITES));expected=select_pair(m,r,c,True)
            self.assertEqual(actual['status'],expected['status']);self.assertEqual(actual['eligible_crossings'],len(expected['eligible_crossings']))
            self.assertEqual(actual['selected'] is None,expected['selected'] is None)
            if actual['selected']:
                for k in ['paths','directions','ranking_counts','lexical_tie_key']:self.assertEqual(actual['selected'][k],expected['selected'][k])


if __name__=='__main__':unittest.main()
