"""Meaningful finite checks of fixed budgets, independent cases and selection/inference."""
import copy
import unittest
from . import common as c
from .selection import select_pair,usable
from .analyze import task_block_signflip,apply_verdict
from .statistics import bootstrap
from preparations.ease_insights_20261103.vlm.analyze import key


def manifest(stage='discovery'):
    m=dict(phase=c.DISCOVERY_PHASE if stage=='discovery' else c.CONFIRMATION_PHASE,
        conditions=c.conditions(),suites=list(c.SUITES),tasks=list(range(10)),
        case_table=[dict(suite=s,task=t) for s in c.SUITES for t in range(10)],
        science=dict(policy_action_limits=dict(c.LIMITS),settling_actions=10))
    if stage=='discovery':m.update(discovery_inits=[45,46],planned_conditions=list(c.CONDITIONS),cases_per_condition=80,total_discovery_episodes=400)
    else:m.update(confirmation_inits=[47,48,49],planned_conditions=['full_d10','spread_a','spread_b'],cases_per_condition=120,total_confirmation_episodes=360)
    return m


def synthetic(crossing):
    m=manifest();records={}
    for cfg in c.conditions():
        cid=cfg['id']
        for scene in c.cases(m):
            good=cid in ('full_d10','spread_a','spread_b')
            if cid=='spread_b' and scene['suite']=='libero_spatial' and scene['task']<2:good=False
            if crossing and cid=='spread_a' and scene['suite']=='libero_object' and scene['task']<2:good=False
            records[cid,key(scene)]=dict(success=good)
    costs=dict(valid_services=1,median_full_over_policy_cycle={cid:1. if cid=='full_d10' else 1.3 for cid in c.CONDITIONS})
    return m,records,costs


class ScientificContracts(unittest.TestCase):
    def test_geometric_masks_have_equal_budgets_and_large_distance(self):
        masks=list(c.MASKS.values())
        self.assertTrue(all(len(set(m))==6 and all(1<=l<=17 for l in m) for m in masks))
        self.assertTrue(all(len(set(a)^set(b))>=6 for i,a in enumerate(masks) for b in masks[i+1:]))
        for config in c.conditions()[1:]:
            self.assertEqual((config['D'],config['refresh_D'],config['R'],config['period']),(2,10,5,2))

    def test_case_shards_and_disjoint_holdout(self):
        d=manifest();f=manifest('confirm')
        for m,n in ((d,80),(f,120)):
            all_cases=c.cases(m);parts=[c.cases(m,None,i,2) for i in range(2)]
            self.assertEqual(len(all_cases),n)
            self.assertEqual({key(s) for s in all_cases},{key(s) for p in parts for s in p})
            self.assertFalse({key(s) for s in parts[0]} & {key(s) for s in parts[1]})
        self.assertFalse({key(s) for s in c.cases(d)} & {key(s) for s in c.cases(f)})

    def test_true_opposite_directions_select_pair(self):
        m,r,cost=synthetic(True);v=select_pair(m,r,cost,True)
        self.assertEqual(v['status'],'selected')
        self.assertEqual(v['selected']['paths'],['spread_a','spread_b'])
        self.assertEqual([(x['suite'],x['better']) for x in v['selected']['directions']],
                         [('libero_spatial','spread_a'),('libero_object','spread_b')])

    def test_global_dominance_never_means_complementarity(self):
        m,r,cost=synthetic(False)
        self.assertEqual(select_pair(m,r,cost,True)['status'],'no_candidate')
        self.assertEqual(select_pair(m,r,cost,False)['status'],'incomplete_discovery')

    def test_unusable_or_slow_path_not_promoted(self):
        m,r,cost=synthetic(True)
        cost['median_full_over_policy_cycle']['spread_a']=1.14
        self.assertEqual(select_pair(m,r,cost,True)['status'],'no_candidate')
        self.assertTrue(usable(60,72,80))
        self.assertFalse(usable(59,72,80))
        self.assertFalse(usable(60,73,80))

    def test_task_signflip_keeps_ties_and_three_init_bound(self):
        self.assertEqual(task_block_signflip([3]*10)['p_one_sided'],1/1024)
        self.assertEqual(task_block_signflip([3]*5+[0]*5)['p_one_sided'],1/32)
        self.assertEqual(task_block_signflip([0]*10)['p_one_sided'],1.)
        with self.assertRaises(ValueError):task_block_signflip([4]*10)

    def test_confirmation_requires_both_adjusted_directions(self):
        def direction(p):
            return dict(inference=dict(primary_task_block=dict(p_one_sided=p),sensitivity_case_mcnemar_p=.001,
                positive_direction=True,task_cluster_lower_bound_positive=True,every_leave_one_task_out_positive=True))
        eligibility=dict(a=dict(usable=True),b=dict(usable=True))
        self.assertEqual(apply_verdict([direction(1/64),direction(1/64)],eligibility),'complementarity_reproduced')
        self.assertEqual(apply_verdict([direction(1/32),direction(1/32)],eligibility),'not_reproduced')
        self.assertEqual(apply_verdict([direction(1/1024),direction(.9)],eligibility),'not_reproduced')

    def test_bootstrap_uses_ten_task_clusters(self):
        values=[v for v in (1,1,1,1,1,0,0,0,0,0) for _ in range(3)];ids=[t for t in range(10) for _ in range(3)]
        result=bootstrap(values,ids,7,11,1000)
        self.assertEqual(result['task_cluster']['clusters'],10)
        self.assertEqual(result,bootstrap(values,ids,7,11,1000))
        self.assertLess(result['task_cluster']['low'],.5)
        self.assertGreater(result['task_cluster']['high'],.5)
        with self.assertRaises(ValueError):bootstrap(values[:20],ids[:20],7,11,1000)


if __name__=='__main__':unittest.main()
