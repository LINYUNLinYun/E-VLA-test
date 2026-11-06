"""The separate integer-count auditor agrees on qualified and dominant synthetic pairs."""
import unittest
from .test_selection import synthetic
from .selection import select_pair
from reports.ease_insights_20261103.camera_paths_v3.statistics_audit import independent_decision


class IndependentSelectionContracts(unittest.TestCase):
    def check(self,m,records,costs):
        counts={cid:{s:[sum(int(records[cid,(s,t,i)]['success']) for i in (45,46)) for t in range(10)] for s in m['suites']} for cid in m['analytical_conditions']}
        a=independent_decision(counts,costs['median_full_over_policy_cycle'],m['suites']);b=select_pair(m,records,costs,True)
        self.assertEqual(a['status'],b['status']);self.assertEqual(a['eligible_crossings'],len(b['eligible_crossings']))
        if a['selected']:
            for key in ('paths','directions','ranking_counts','lexical_tie_key'):self.assertEqual(a['selected'][key],b['selected'][key])

    def test_crossing_and_mismatched_cost(self):
        m,r,c=synthetic();self.check(m,r,c)
        c['median_full_over_policy_cycle']['wrist_fresh_d5']=1.25;self.check(m,r,c)

    def test_uniform_dominance(self):
        m,r,c=synthetic()
        for (cid,k),row in r.items():
            if cid=='agent_fresh_d5':row['success']=True
        self.check(m,r,c)


if __name__=='__main__':unittest.main()
