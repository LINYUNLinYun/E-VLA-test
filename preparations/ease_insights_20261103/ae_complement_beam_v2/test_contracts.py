"""CPU boundaries for fixed masks, stages and control-only calibration."""
import copy
import unittest
from . import common


def manifest(stage):
    tasks = [0] if stage == 'calibration' else list(range(10))
    m = dict(phase='AE_joint_mask_' + stage + '_v2', conditions=common.conditions(), suites=list(common.SUITES), tasks=tasks,
             case_table=[dict(suite=s,task=t) for s in common.SUITES for t in tasks],
             science=dict(policy_action_limits=dict(common.LIMITS), settling_actions=10), proxy_preregistered_directions=copy.deepcopy(common.DIRECTIONS))
    if stage == 'calibration':
        m.update(calibration_inits=[45], planned_conditions=['full_d10','ae16_d2'], cases_per_condition=4, total_calibration_episodes=8)
    elif stage == 'discovery':
        m.update(discovery_inits=[45,46], planned_conditions=['path_a','path_b'], cases_per_condition=80, total_new_discovery_episodes=160,
                 reference_conditions=common.REFERENCE_CONDITIONS, analytical_conditions=list(common.CONDITIONS))
    else:
        m.update(confirmation_inits=[47,48,49], planned_conditions=['full_d10','path_a','path_b'], cases_per_condition=120,total_confirmation_episodes=360)
    return m


class Contracts(unittest.TestCase):
    def test_pair_has_equal_depth_and_nfe_and_frozen_masks(self):
        configs = {r['id']:r for r in common.conditions()}
        self.assertEqual(configs['path_a']['ae_skip'], [2,13])
        self.assertEqual(configs['path_b']['ae_skip'], [7,14])
        for cid in ['path_a','path_b']:
            self.assertEqual(configs[cid]['D'],2)
            self.assertEqual(len(configs[cid]['ae_skip']),2)
            self.assertEqual(configs[cid]['skip'],[])
        self.assertEqual(configs['full_d10']['ae_skip'],[])

    def test_all_stage_budgets_and_fixed_task_shards(self):
        for stage,n in [('calibration',4),('discovery',80),('confirm',120)]:
            m=manifest(stage)
            self.assertEqual(len(common.cases(m)),n)
            a,b=[common.cases(m,stage,i,2) for i in range(2)]
            self.assertEqual(len(a)+len(b),n)
            self.assertFalse({(v['suite'],v['task']) for v in a}&{(v['suite'],v['task']) for v in b})

    def test_directions_cannot_flip_or_change_suite(self):
        for direction in [dict(suite='libero_goal',better='path_a',worse='path_b'),dict(suite='libero_object',better='path_b',worse='path_a')]:
            m=manifest('discovery');m['proxy_preregistered_directions'][0]=direction
            with self.assertRaises(ValueError):common.validate_manifest(m)

    def test_wrong_initials_and_new_paths_in_calibration_rejected(self):
        m=manifest('discovery');m['discovery_inits']=[47,48]
        with self.assertRaises(ValueError):common.validate_manifest(m)
        m=manifest('calibration');m['planned_conditions']=['full_d10','path_a']
        with self.assertRaises(ValueError):common.validate_manifest(m)

    def test_old_ae16_is_reference_not_new_discovery_condition(self):
        m=manifest('discovery')
        with self.assertRaises(ValueError):common.planned_config(m,'ae16_d2')
        self.assertEqual(common.planned_config(m,'path_a')['ae_skip'],[2,13])

    def test_complete_proxy_pin_resolves_without_model_loading(self):
        v=common.proxy_evidence()
        self.assertEqual(v['sha256'],common.SCREEN_SELECTION_SHA)


if __name__=='__main__':unittest.main()
