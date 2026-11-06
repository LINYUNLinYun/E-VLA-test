"""Depth/time pair selection requires coverage,cost matching and opposite preferences."""
import tempfile
import unittest
from .common import CONDITIONS,SUITES
from .selection import select_pair,freeze
from preparations.ease_insights_20261103.vlm.analyze import key


def synthetic():
    m=dict(suites=list(SUITES),tasks=list(range(10)),discovery_inits=[45,46],analytical_conditions=list(CONDITIONS));records={}
    for cid in CONDITIONS:
        for s in SUITES:
            for t in range(10):
                for i in (45,46):
                    good=cid in ('full_d10','dense_d2','ae14_d3')
                    if cid=='dense_d2' and s=='libero_object' and t<2:good=False
                    if cid=='ae14_d3' and s=='libero_spatial' and t<2:good=False
                    records[cid,key(dict(suite=s,task=t,init=i))]=dict(success=good)
    ratios={c:1.6 for c in CONDITIONS};ratios['full_d10']=1.
    return m,records,dict(valid_services=8,median_full_over_policy_cycle=ratios)


class SelectionContracts(unittest.TestCase):
    def test_candidate_pair_must_match_actual_cost(self):
        m,r,c=synthetic();choice=select_pair(m,r,c,True)
        self.assertEqual(choice['status'],'selected');self.assertEqual(choice['selected']['paths'],['ae14_d3','dense_d2'])
        c['median_full_over_policy_cycle']['ae14_d3']=1.25;self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_uniform_dominance_does_not_support_complementarity(self):
        m,r,c=synthetic()
        for (cid,k),row in r.items():
            if cid=='dense_d2':row['success']=True
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_incomplete_data_cannot_select(self):
        m,r,c=synthetic();self.assertEqual(select_pair(m,r,c,False)['status'],'incomplete_discovery')

    def test_preserved_controls_are_not_new_cases(self):
        report=dict(complete=True,valid_episodes=480,new_discovery_episodes=480,preserved_reference_episodes=0,selection=dict(status='no_candidate'))
        with tempfile.TemporaryDirectory() as path,self.assertRaises(ValueError):freeze(path,report)


if __name__=='__main__':unittest.main()
