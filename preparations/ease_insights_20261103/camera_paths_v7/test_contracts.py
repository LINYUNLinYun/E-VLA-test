"""Stage sizes and physical task sharding for fixed VLM component paths."""
import copy
import unittest
from . import common as c


def manifest(stage):
    tasks=[0] if stage=='calibration' else list(range(10))
    m=dict(phase='component_paths_'+stage+'_v7',conditions=c.conditions(),noise_repeat=0,suites=list(c.SUITES),tasks=tasks,
        case_table=[dict(suite=s,task=t) for s in c.SUITES for t in tasks],science=dict(settling_actions=10,policy_action_limits=c.LIMITS))
    if stage=='calibration':m.update(calibration_inits=[45],planned_conditions=['full_d10','dense_d2'],cases_per_condition=4,total_calibration_episodes=8)
    elif stage=='discovery':m.update(discovery_inits=[45,46],planned_conditions=list(c.NEW_CONDITIONS),cases_per_condition=80,total_new_discovery_episodes=160,analytical_cells=320,reference_conditions={'full_d10':'full_d10','dense_d2':'dense_d2'})
    else:m.update(confirmation_inits=[10,11,12,13,14],planned_conditions=list(c.CONDITIONS),cases_per_condition=200,total_confirmation_episodes=800)
    return m


class ContractTest(unittest.TestCase):
    def test_exact_counts_and_disjoint_task_shards(self):
        for stage,total in [('calibration',4),('discovery',80),('confirm',200)]:
            m=manifest(stage);a,b=c.cases(m,stage,0,2),c.cases(m,stage,1,2)
            keys=lambda rows:{(r['suite'],r['task'],r['init']) for r in rows}
            self.assertEqual(len(a)+len(b),total);self.assertFalse(keys(a)&keys(b))
    def test_protocol_deviations_rejected(self):
        for key,value in [('noise_repeat',1),('confirmation_inits',[15,16,17,18,19]),('planned_conditions',list(c.NEW_CONDITIONS))]:
            m=manifest('confirm');m[key]=value
            with self.assertRaises(ValueError):c.validate_manifest(m)


if __name__=='__main__':unittest.main()
