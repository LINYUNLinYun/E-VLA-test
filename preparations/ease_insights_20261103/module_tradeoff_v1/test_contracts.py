"""CPU contracts for module calibration and truthful exported AE execution."""
import copy
import unittest
from types import SimpleNamespace
from . import common as c
from .ae_dispatch import ScopedLayers
from .worker import validate_ae
from .policy import Policy
from .test_policy import FakeBackend


class ModuleContracts(unittest.TestCase):
    def test_four_known_calibration_controls_are_balanced(self):
        m=dict(phase='module_tradeoff_calibration_v1',calibration_inits=[45],conditions=c.conditions(),
            planned_conditions=['full_d10'],suites=list(c.SUITES),tasks=[0],case_table=[dict(suite=s,task=0) for s in c.SUITES],
            cases_per_condition=4,total_calibration_episodes=4,science=dict(policy_action_limits=dict(c.LIMITS),settling_actions=10))
        scenes=c.cases(m);parts=[c.cases(m,'calibration',i,2) for i in range(2)]
        self.assertEqual([len(p) for p in parts],[2,2])
        self.assertEqual(len(scenes),4)
        self.assertTrue(all(s['init']==45 and s['task']==0 for s in scenes))
        self.assertEqual({s['suite'] for p in parts for s in p},set(c.SUITES))

    def test_default_policy_uses_real_fixed_module_validator(self):
        backend=FakeBackend();p=Policy(backend);cfg=next(c0 for c0 in c.conditions() if c0['id']=='ae_16');p.configure(cfg)
        for _ in range(2):
            row=p.predict({},'prompt','suite',7);p.acknowledge(row['actions'][:5])
        self.assertEqual(backend.samples[-1]['ae_skip'],(2,8))
        self.assertFalse(row['route']['refresh'])
        with self.assertRaises(ValueError):p.configure(dict(cfg,ae_skip=[1,2]))

    def test_ae_receipt_cannot_fake_actual_layer_visits(self):
        namespace=SimpleNamespace(_decoder_layer=lambda *a:None);scope=ScopedLayers(namespace)
        with scope.use((2,8)) as visited:
            for _ in range(2):
                for i in range(18):namespace._decoder_layer(None,None,512,i)
        receipt=visited.receipt(2)
        row=dict(route=dict(ae_execution_audit=receipt,ae_bypassed_layers=[2,8]),timing=dict(ae_execution_audit=receipt,actual_NFE=2))
        validate_ae(row,[2,8],2)
        for changed in ('actual_layer_calls','attempted_layer_calls','actual_NFE'):
            broken=copy.deepcopy(row);broken['route']['ae_execution_audit'][changed]+=1
            with self.assertRaises(ValueError):validate_ae(broken,[2,8],2)

    def test_fixed_definitions_keep_full_ae18_and_preserve_axes(self):
        configs={v['id']:v for v in c.conditions()}
        self.assertEqual(c.validate_condition(configs['full_d10']),((),()))
        self.assertEqual(c.validate_condition(configs['vlm_c']),((4,7,10,13,15,17),()))
        self.assertEqual([len(configs[k]['ae_skip']) for k in ('ae_16','ae_14','ae_12')],[2,4,6])
        self.assertTrue(all(configs[k]['skip']==[] for k in ('ae_16','ae_14','ae_12')))
        with self.assertRaises(ValueError):c.validate_condition(dict(configs['full_d10'],ae_skip=[2]))


if __name__=='__main__':unittest.main()
