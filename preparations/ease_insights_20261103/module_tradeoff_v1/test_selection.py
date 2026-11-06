"""The new module cost gate and preserved/new case counts have separate roles."""
import tempfile
import unittest
from .common import CONDITIONS,SUITES
from .selection import select_pair,freeze
from preparations.ease_insights_20261103.vlm.analyze import key


def synthetic():
    m=dict(suites=list(SUITES),tasks=list(range(10)),discovery_inits=[45,46],analytical_conditions=list(CONDITIONS))
    records={}
    for c in CONDITIONS:
        for s in SUITES:
            for t in range(10):
                for i in (45,46):
                    good=c in ('full_d10','vlm_c','ae_16')
                    if c=='vlm_c' and s=='libero_spatial' and t<2:good=False
                    if c=='ae_16' and s=='libero_object' and t<2:good=False
                    records[c,key(dict(suite=s,task=t,init=i))]=dict(success=good)
    ratios={c:1.6 for c in CONDITIONS};ratios['full_d10']=1.
    return m,records,dict(valid_services=8,median_full_over_policy_cycle=ratios)


class SelectionContracts(unittest.TestCase):
    def test_complement_pair_must_match_measured_cost(self):
        m,r,cost=synthetic();cost['median_full_over_policy_cycle']['ae_16']=1.25
        self.assertEqual(select_pair(m,r,cost,True)['status'],'no_candidate')
        cost['median_full_over_policy_cycle']['ae_16']=1.55
        choice=select_pair(m,r,cost,True)
        self.assertEqual(choice['status'],'selected')
        self.assertEqual(choice['selected']['paths'],['ae_16','vlm_c'])
        self.assertTrue(choice['selected']['cost_matched'])

    def test_incomplete_new_cases_never_select(self):
        m,r,cost=synthetic()
        self.assertEqual(select_pair(m,r,cost,False)['status'],'incomplete_discovery')

    def test_preserved_controls_cannot_be_counted_as_new_replicates(self):
        report=dict(complete=True,valid_episodes=480,new_discovery_episodes=480,preserved_reference_episodes=0,selection=dict(status='no_candidate'))
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):freeze(tmp,report)


if __name__=='__main__':unittest.main()
