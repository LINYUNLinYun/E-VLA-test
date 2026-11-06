"""Exact AE14/D3 neighbor budget and unchanged source/stage contracts."""
import copy
import unittest
from . import common


def manifest(stage):
    tasks=[0] if stage=='calibration' else list(range(10))
    m=dict(phase='AE14_neighbor_'+stage+'_v3',conditions=common.conditions(),suites=list(common.SUITES),tasks=tasks,
           case_table=[dict(suite=s,task=t) for s in common.SUITES for t in tasks],science=dict(policy_action_limits=dict(common.LIMITS),settling_actions=10))
    if stage=='calibration':m.update(calibration_inits=[45],planned_conditions=['full_d10','ae14_base'],cases_per_condition=4,total_calibration_episodes=8)
    elif stage=='discovery':m.update(discovery_inits=[45,46],planned_conditions=list(common.NEW_CONDITIONS),cases_per_condition=80,total_new_discovery_episodes=320,reference_conditions=common.REFERENCE_CONDITIONS,analytical_cells=480)
    else:m.update(confirmation_inits=[47,48,49],planned_conditions=['full_d10','ae14_base','path_a'],cases_per_condition=120,total_confirmation_episodes=360)
    return m


class Contracts(unittest.TestCase):
    def test_all_fast_configs_keep14_layers_at_D3(self):
        for config in common.conditions():
            if config['id']=='full_d10':
                self.assertEqual(config['D'],10);self.assertEqual(config['ae_skip'],[])
            else:
                self.assertEqual(config['D'],3);self.assertEqual(len(config['ae_skip']),4)
                self.assertEqual(config['skip'],[]);self.assertEqual(config['period'],2)

    def test_each_new_mask_is_exactly_one_swap_from_anchor(self):
        anchor=set(common.PATHS['ae14_base'][1])
        for cid in common.NEW_CONDITIONS:
            mask=set(common.PATHS[cid][1]);self.assertEqual(len(anchor-mask),1);self.assertEqual(len(mask-anchor),1)
            self.assertTrue(mask-anchor <= {4,9})

    def test_every_world_stage_has_fixed_cases_and_shards(self):
        for stage,n in [('calibration',4),('discovery',80),('confirm',120)]:
            m=manifest(stage);self.assertEqual(len(common.cases(m)),n)
            shards=[common.cases(m,stage,i,2) for i in range(2)]
            self.assertEqual(sum(map(len,shards)),n)
            self.assertFalse({(s['suite'],s['task']) for s in shards[0]} & {(s['suite'],s['task']) for s in shards[1]})

    def test_wrong_budget_or_initials_are_rejected(self):
        m=manifest('discovery');m['conditions'][2]['D']=2
        with self.assertRaises(ValueError):common.validate_manifest(m)
        m=manifest('discovery');m['discovery_inits']=[47,48]
        with self.assertRaises(ValueError):common.validate_manifest(m)

    def test_calibration_only_runs_known_controls(self):
        m=manifest('calibration');m['planned_conditions']=['full_d10','path_a']
        with self.assertRaises(ValueError):common.validate_manifest(m)
        m=manifest('discovery')
        with self.assertRaises(ValueError):common.planned_config(m,'ae14_base')

    def test_both_reference_algorithms_match_actual_source_conditions(self):
        for alias,cid in [('full_d10','full_d10'),('ae14_base','ae14_d3')]:
            root,m,cm,w,source=common.reference(alias,verify_sources=True)
            self.assertEqual(source,cid)
            old=copy.deepcopy(next(c for c in m['conditions'] if c['id']==source));old['id']=alias;old.setdefault('ae_skip',[])
            self.assertEqual(old,next(c for c in common.conditions() if c['id']==alias))


if __name__=='__main__':unittest.main()
