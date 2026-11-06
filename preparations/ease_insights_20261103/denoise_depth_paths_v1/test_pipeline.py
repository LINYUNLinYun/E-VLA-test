"""Scientific stage/count contracts and D3/D4 consumers, without model loading."""
import copy
import unittest
from preparations.ease_insights_20261103.module_tradeoff_v1.test_policy import FakeBackend
from . import common,worker,cost_records,nodes
from .policy import Policy


class PipelineContracts(unittest.TestCase):
    def test_empty_VLM_mask_uses_each_configured_fast_schedule(self):
        for cfg in common.conditions():
            b=FakeBackend();p=Policy(b);p.configure(cfg)
            for i in range(6):
                r=p.predict({},'prompt','suite',7+i);p.acknowledge(r['actions'][:5])
            expected=[10]*6 if cfg['id']=='full_d10' else [10,cfg['D'],cfg['D'],10,cfg['D'],cfg['D']]
            self.assertEqual([s['steps'] for s in b.samples],expected)
            self.assertTrue(all(not q.get('skip') for q in b.prepared))
            if cfg['id']!='full_d10':self.assertEqual(r['route']['history_steps'],[0,15])

    def test_all_consumers_use_exact_D2_D3_D4_D10_nodes(self):
        for n in (2,3,4,10):
            value=[1-i/n for i in range(n+1)]
            self.assertEqual(worker.scheduled_nfe(value),n);self.assertEqual(cost_records.scheduled_nfe(value),n)
        self.assertIs(worker.scheduled_nfe,nodes.scheduled_nfe);self.assertIs(cost_records.scheduled_nfe,nodes.scheduled_nfe)

    def test_calibration_has_eight_repeats_and_four_scenes(self):
        m=dict(phase='denoise_depth_calibration_v1',calibration_inits=[45],conditions=common.conditions(),suites=list(common.SUITES),tasks=[0],
            case_table=[dict(suite=s,task=0) for s in common.SUITES],science=dict(policy_action_limits=common.LIMITS,settling_actions=10),
            planned_conditions=['full_d10','ae16_d2'],cases_per_condition=4,total_calibration_episodes=8)
        self.assertEqual(len(common.cases(m))*len(m['planned_conditions']),8)
        self.assertEqual(len(common.cases(m,shard_index=0,shards=2)),2)
        bad=copy.deepcopy(m);bad['total_calibration_episodes']=4
        with self.assertRaises(ValueError):common.cases(bad)

    def test_reference_origins_and_aliases_remain_distinct(self):
        a=common.reference('full_d10',verify_sources=True);b=common.reference('ae16_d2',verify_sources=True)
        self.assertNotEqual(a[0],b[0]);self.assertNotEqual(a[1]['manifest_hash'],b[1]['manifest_hash'])
        self.assertEqual((a[-1],b[-1]),('full_d10','ae_16'))

    def test_native_AE_depth_receipts_cover_D3_and_D4(self):
        for n,skip in [(3,[2,6,7,8]),(4,[2,4,6,7,8,9])]:
            audit=dict(actual_NFE=n,attempted_layers_per_step=list(range(1,19)),executed_layers_per_step=[l for l in range(1,19) if l not in skip],
                bypassed_layers_per_step=skip,actual_layer_calls=n*(18-len(skip)),attempted_layer_calls=n*18,bypassed_layer_calls=n*len(skip),original_helper_restored=True,
                scope='Observed eager helper dispatch, not FLOPs, latency or quality')
            row=dict(route=dict(ae_execution_audit=audit,ae_bypassed_layers=skip),timing=dict(ae_execution_audit=copy.deepcopy(audit),actual_NFE=n))
            worker.validate_ae(row,skip,n);bad=copy.deepcopy(row);bad['timing']['actual_NFE']=2
            with self.assertRaises(ValueError):worker.validate_ae(bad,skip,n)


if __name__=='__main__':unittest.main()
