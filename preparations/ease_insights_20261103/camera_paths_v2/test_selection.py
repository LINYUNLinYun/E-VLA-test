"""Only the prespecified camera pair can qualify, with full coverage and matched cost."""
import tempfile
import unittest
from .common import CONDITIONS,PAIR_PATHS,SUITES
from .selection import select_pair,freeze
from preparations.ease_insights_20261103.vlm.analyze import key


def synthetic():
    m=dict(suites=list(SUITES),tasks=list(range(10)),discovery_inits=[45,46],analytical_conditions=list(CONDITIONS))
    records={}
    for cid in CONDITIONS:
        for suite in SUITES:
            for task in range(10):
                for init in (45,46):
                    good=not ((cid=='agent_fresh_d5' and suite=='libero_object' or cid=='wrist_fresh_d5' and suite=='libero_spatial') and task<2)
                    records[cid,key(dict(suite=suite,task=task,init=init))]=dict(success=good)
    ratios=dict(full_d10=1.,both_fresh_d5=1.4,agent_fresh_d5=1.6,wrist_fresh_d5=1.55)
    return m,records,dict(valid_services=6,median_full_over_policy_cycle=ratios)


class CameraSelectionContracts(unittest.TestCase):
    def test_complementary_pair_needs_measured_cost_match(self):
        m,r,c=synthetic();chosen=select_pair(m,r,c,True)
        self.assertEqual(chosen['status'],'selected');self.assertEqual(chosen['selected']['paths'],list(PAIR_PATHS))
        c['median_full_over_policy_cycle']['wrist_fresh_d5']=1.25
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_incomplete_data_cannot_select(self):
        m,r,c=synthetic();self.assertEqual(select_pair(m,r,c,False)['status'],'incomplete_discovery')

    def test_uniform_dominance_does_not_qualify_as_complementarity(self):
        m,r,c=synthetic()
        for (cid,k),row in r.items():
            if cid=='agent_fresh_d5':row['success']=True
        self.assertEqual(select_pair(m,r,c,True)['status'],'no_candidate')

    def test_both_fresh_control_never_substitutes_for_a_bad_camera_path(self):
        m,r,c=synthetic()
        for (cid,k),row in r.items():
            if cid=='wrist_fresh_d5':row['success']=False
        chosen=select_pair(m,r,c,True)
        self.assertEqual(chosen['status'],'no_candidate');self.assertEqual(chosen['candidate_order'],list(PAIR_PATHS))

    def test_old_Full_cells_are_not_new_cases(self):
        report=dict(complete=True,valid_episodes=320,new_discovery_episodes=320,preserved_reference_episodes=0,selection=dict(status='no_candidate'))
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):freeze(tmp,report)


if __name__=='__main__':unittest.main()
