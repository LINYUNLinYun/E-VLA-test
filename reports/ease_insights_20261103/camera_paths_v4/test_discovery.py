"""Scientific decision tests use synthetic counts,never real partial outcomes."""
import unittest
import numpy as np
from preparations.ease_insights_20261103.camera_paths_v4.common import CONDITIONS,SUITES
from reports.ease_insights_20261103.camera_paths_v4.discovery_analysis import select_pair,key,usable
from reports.ease_insights_20261103.camera_paths_v4.statistics_audit import independent_decision
A,B='agent_deep_d2','wrist_deep_d2'


class DiscoveryTest(unittest.TestCase):
    def fixture(self):
        m=dict(suites=list(SUITES),tasks=list(range(10)),discovery_inits=[45,46],analytical_conditions=list(CONDITIONS))
        rows={(c,key(dict(suite=s,task=t,init=i))):dict(success=True) for c in CONDITIONS for s in SUITES for t in range(10) for i in (45,46)}
        costs=dict(valid_services=4,median_full_over_policy_cycle={c:1. if c=='full_d10' else 1.5 for c in CONDITIONS})
        return m,rows,costs
    def failures(self,rows,c,s,tasks,inits=(45,)):
        for t in tasks:
            for i in inits:rows[c,key(dict(suite=s,task=t,init=i))]['success']=False
    def crossing(self):
        m,r,c=self.fixture();self.failures(r,B,'libero_goal',range(4));self.failures(r,A,'libero_10',range(4));return m,r,c
    def test_four_task_opposite_preferences_select_fixed_pair(self):
        m,r,c=self.crossing();v=select_pair(m,r,c,True)
        self.assertEqual(v['status'],'selected');self.assertEqual(v['selected']['paths'],[A,B])
        self.assertEqual([d['suite'] for d in v['selected']['directions']],['libero_goal','libero_10'])
    def test_two_positive_tasks_are_insufficient_even_with_twenty_point_gaps(self):
        m,r,c=self.fixture();self.failures(r,B,'libero_goal',range(2),(45,46));self.failures(r,A,'libero_10',range(2),(45,46))
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')
    def test_uniform_dominance_cannot_select(self):
        m,r,c=self.fixture();self.failures(r,B,'libero_goal',range(4));self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')
    def test_dense_control_is_never_a_candidate(self):
        m,r,c=self.fixture();self.failures(r,'dense_d2','libero_goal',range(4));self.failures(r,A,'libero_10',range(4))
        v=select_pair(m,r,c,True);self.assertEqual(v['status'],'no_candidate');self.assertEqual(v['candidate_order'],[A,B])
    def test_quality_and_native_pair_cost_gates(self):
        self.assertFalse(usable(67,80,80));self.assertTrue(usable(68,80,80))
        for ratio in (1.14,1.9):
            m,r,c=self.crossing();c['median_full_over_policy_cycle'][A]=ratio
            self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')
    def test_incomplete_or_missing_cost_cannot_select(self):
        m,r,c=self.crossing();self.assertEqual(select_pair(m,r,c,False)['status'],'incomplete_discovery')
        c['valid_services']=0;self.assertEqual(select_pair(m,r,c,True)['status'],'missing_valid_cost_probe')
    def test_independent_integer_implementation_matches_synthetic_matrices(self):
        for seed in range(30):
            m,r,c=self.crossing();rng=np.random.default_rng(seed)
            for cid in (A,B):
                for _ in range(4):
                    s=SUITES[int(rng.integers(4))];t=int(rng.integers(10));i=45+int(rng.integers(2))
                    r[cid,key(dict(suite=s,task=t,init=i))]['success']=False
            counts={cid:{s:[sum(int(r[cid,key(dict(suite=s,task=t,init=i))]['success']) for i in (45,46)) for t in range(10)] for s in SUITES} for cid in CONDITIONS}
            actual=independent_decision(counts,c['median_full_over_policy_cycle'],list(SUITES));expected=select_pair(m,r,c,True)
            self.assertEqual(actual['status'],expected['status']);self.assertEqual(actual['eligible_crossings'],len(expected['eligible_crossings']))
            if actual['selected']:
                for field in ('paths','directions','ranking_counts','lexical_tie_key'):self.assertEqual(actual['selected'][field],expected['selected'][field])


if __name__=='__main__':unittest.main()
