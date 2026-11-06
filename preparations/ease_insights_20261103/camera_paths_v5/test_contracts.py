"""Prospective stage and production prefix audit contracts;CPU only."""
import copy
from types import SimpleNamespace
import unittest
from . import common as c,worker as w
from reports.ease_insights_20261103.prefix_kv_preparation_v2.test_packing import fixture
from reports.ease_insights_20261103.prefix_kv_preparation_v2.packing import PrefixPacker
from reports.ease_insights_20261103.prefix_kv_preparation_v3.test_sampling import FakeAdapter
from reports.ease_insights_20261103.prefix_kv_preparation_v3.sampling import PrefixVisits,sample_with_prefix


def manifest(stage):
    tasks=[0] if stage=='calibration' else list(range(10))
    m=dict(phase='prefix_attention_'+stage+'_v5',conditions=c.conditions(),noise_repeat=0,suites=list(c.SUITES),tasks=tasks,
        case_table=[dict(suite=s,task=t) for s in c.SUITES for t in tasks],science=dict(settling_actions=10,policy_action_limits=c.LIMITS))
    if stage=='calibration':m.update(calibration_inits=[45],planned_conditions=['full_d10','dense_d2'],cases_per_condition=4,total_calibration_episodes=8)
    elif stage=='discovery':m.update(discovery_inits=[45,46],planned_conditions=list(c.NEW_CONDITIONS),cases_per_condition=80,total_new_discovery_episodes=160,analytical_cells=320,reference_conditions={'full_d10':'full_d10','dense_d2':'dense_d2'})
    else:m.update(confirmation_inits=[10,11,12,13,14],planned_conditions=list(c.CONDITIONS),cases_per_condition=200,total_confirmation_episodes=800)
    return m


def row(cfg,full):
    context=fixture(True);namespace=SimpleNamespace(_decoder_layer=lambda *args:None)
    pattern='all' if full else cfg['prefix_pattern'];nfe=10 if full else 2
    result,ae,prefix=sample_with_prefix(FakeAdapter(namespace),PrefixVisits(namespace),PrefixPacker(),context,77,nfe,pattern)
    if pattern!='all':prefix['packing_cuda_ms']=.1
    layers=[list(c.FULL),list(c.FULL)]
    camera=dict(kind='native',updated_views=[0,1],cached_views=[],vision_pipeline_invocations=1,actual_kernel_input_batches=[2],
        actual_vision_layers_by_camera=layers,effective_vision_layers_by_camera=layers,image_layer_evaluations=54,fresh_images_consumed=True,
        vision_cache='none',numerical_reference_only=False,single_view_projection=None,camera_order=['agentview','wrist'],
        scope='Observed logical image-layer visits and tensor batches,not GPU kernel-call counts,FLOPs,latency or quality',original_vision_helper_restored=True)
    timing=dict(actual_NFE=nfe,ae_execution_audit=ae,camera_execution_audit=camera,prefix_execution_audit=prefix,
        prefix_pack_wall_ms=prefix['packing_wall_ms'],prefix_pack_cuda_ms=prefix['packing_cuda_ms'],prefix_restore_wall_ms=prefix['restore_wall_ms'])
    return dict(timing=timing,route=dict(ae_execution_audit=ae,ae_bypassed_layers=[],camera_execution_audit=camera,prefix_pattern=pattern,prefix_execution_audit=prefix))


class ContractTest(unittest.TestCase):
    def test_exact_stage_counts_and_disjoint_task_shards(self):
        for stage,total in [('calibration',4),('discovery',80),('confirm',200)]:
            m=manifest(stage);a,b=c.cases(m,stage,0,2),c.cases(m,stage,1,2);keys=lambda r:{(x['suite'],x['task'],x['init']) for x in r}
            self.assertEqual(len(a)+len(b),total);self.assertFalse(keys(a)&keys(b))
    def test_unknown_prefix_noiserepeat_and_missing_dense_control_rejected(self):
        m=manifest('discovery');m['conditions'][2]['prefix_pattern']='arbitrary'
        with self.assertRaises(ValueError):c.validate_manifest(m)
        m=manifest('confirm');m['noise_repeat']=1
        with self.assertRaises(ValueError):c.validate_manifest(m)
        m=manifest('confirm');m['planned_conditions'].remove('dense_d2')
        with self.assertRaises(ValueError):c.validate_manifest(m)
    def test_actual_cpu_sampler_audits_accepted_with_explicit_gpu_timing_fixture(self):
        for cfg in c.conditions():
            for full in (True,False):
                if cfg['id']=='full_d10' and not full:continue
                r=row(cfg,full);w.validate_camera(r,cfg,full);w.validate_ae(r,[],10 if full else 2)
    def test_original_positions_indices_and_consumed_length_cannot_be_forged(self):
        cfg=c.conditions()[2]
        for field,value in [('actual_prefix_length',264),('kept_index_sha256','wrong'),('original_buffers_restored',False),('decoder_absolute_positions',list(range(263,314)))]:
            r=copy.deepcopy(row(cfg,False));r['route']['prefix_execution_audit'][field]=value
            with self.assertRaises(ValueError):w.validate_camera(r,cfg,False)
    def test_masked_reference_or_partial_vision_cannot_pass_as_quality(self):
        cfg=c.conditions()[2];r=row(cfg,False);r['route']['prefix_execution_audit']['reference_only']=True
        with self.assertRaises(ValueError):w.validate_camera(r,cfg,False)
        r=row(cfg,False);r['route']['camera_execution_audit']['kind']='allocated'
        with self.assertRaises(ValueError):w.validate_camera(r,cfg,False)
    def test_packing_cost_and_full_storage_are_enforced(self):
        cfg=c.conditions()[2];r=row(cfg,False);r['route']['prefix_execution_audit']['packing_cuda_ms']=float('nan')
        with self.assertRaises(ValueError):w.validate_camera(r,cfg,False)
        with self.assertRaises(ValueError):w.validate_camera(row(cfg,False),cfg,True)


if __name__=='__main__':unittest.main()
