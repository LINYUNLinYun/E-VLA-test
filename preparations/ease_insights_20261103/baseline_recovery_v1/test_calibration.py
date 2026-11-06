"""CPU contracts for the changed case universe and suite-specific rollout limits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from . import common as c
from .analyze import decide
from .worker import validate_action_artifacts,verify_physical_identity
from .rollout import episode


def manifest():
    return dict(phase=c.PHASE,calibration_inits=[44],conditions=c.conditions(),planned_conditions=['full_d10'],
        suites=list(c.SUITES),tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in c.SUITES for t in range(10)],
        cases_per_condition=40,total_calibration_episodes=40,health_rule=copy.deepcopy(c.HEALTH_RULE),
        science=dict(policy_action_limits=dict(c.LIMITS),settling_actions=10))


class Contracts(unittest.TestCase):
    def test_two_shards_exhaust_all_tasks_once(self):
        m=manifest();parts=[c.cases(m,'calibration',i,2) for i in range(2)]
        self.assertEqual([20,20],list(map(len,parts)))
        self.assertEqual({json.dumps(s,sort_keys=True) for s in c.cases(m)},
                         {json.dumps(s,sort_keys=True) for p in parts for s in p})
        self.assertFalse({json.dumps(s,sort_keys=True) for s in parts[0]} & {json.dumps(s,sort_keys=True) for s in parts[1]})

    def test_old_uniform_budget_is_rejected(self):
        m=manifest();m['science']['policy_action_limits']['libero_10']=400
        with self.assertRaises(ValueError):c.validate_manifest(m)
        with self.assertRaises(ValueError):episode('/unused',1,dict(suite='libero_spatial',task=0,init=44),c.conditions()[0],None,'x',max_policy_actions=400)

    def test_full_only_and_reserved_init(self):
        with self.assertRaises(ValueError):c.validate_condition(dict(c.conditions()[0],D=2))
        m=manifest();m['calibration_inits']=[45]
        with self.assertRaises(ValueError):c.cases(m)

    def test_health_gate_requires_overall_and_every_suite(self):
        self.assertTrue(decide(dict(zip(c.SUITES,[8,8,8,8]))))
        self.assertTrue(decide(dict(zip(c.SUITES,[6,8,8,10]))))
        self.assertFalse(decide(dict(zip(c.SUITES,[5,9,9,9]))))
        self.assertFalse(decide(dict(zip(c.SUITES,[7,8,8,8]))))

    def test_physical_identity_binds_per_suite_limit(self):
        scene=dict(suite='libero_10',task=0,init=44)
        identity=dict(case=scene,manifest_hash='m',init_vector_sha256='i',bddl_sha256='b',prompt='p',env_seed=7,settling=10,policy_action_limit=520,control_hz=20.)
        m=dict(manifest_hash='m',official_init_availability=dict(cases=[dict(suite='libero_10',task=0,init_sha256={'44':'i'},bddl_sha256='b',prompt='p')]))
        self.assertEqual(verify_physical_identity(m,scene,identity)['policy_action_limit'],520)
        for key,value in [('policy_action_limit',400),('prompt','other'),('init_vector_sha256','other')]:
            with self.assertRaises(ValueError):verify_physical_identity(m,scene,dict(identity,**{key:value}))

    def test_actual_action_prefix_and_timeout_count(self):
        for suite,limit in c.LIMITS.items():
            with self.subTest(suite=suite),tempfile.TemporaryDirectory() as tmp:
                directory=Path(tmp);arrays={};rows=[];telemetry=[dict(step=0,action=None)]
                for index in range(limit//5):
                    actions=np.full((50,7),index,dtype=np.float32);arrays[f'actions_{index}']=actions
                    arrays[f'raw_{index}']=np.zeros((50,32),np.float32)
                    rows.append(dict(executed_length=5,execute_length=5))
                    first_step=len(telemetry)
                    telemetry.extend(dict(step=first_step+i,action=actions[i].tolist()) for i in range(5))
                np.savez(directory/'arrays.npz',**arrays)
                (directory/'telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in telemetry))
                result=dict(case=dict(suite=suite),actions=limit,policy_action_limit=limit,settling_actions=10,termination='timeout',success=False)
                validate_action_artifacts(directory,rows,result)
                with self.assertRaises(ValueError):validate_action_artifacts(directory,rows,dict(result,actions=limit-5))
                # A modified executed command must fail despite valid shapes/counts.
                telemetry[1]['action'][0]+=1
                (directory/'telemetry.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in telemetry))
                with self.assertRaises(ValueError):validate_action_artifacts(directory,rows,result)


if __name__=='__main__':unittest.main()
