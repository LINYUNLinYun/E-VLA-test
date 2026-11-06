"""Frozen scientific budget and rejection of misleading execution receipts."""
import copy
import unittest
from .common import conditions,validate_condition
from .worker import validate_camera,validate_ae


class ScientificContracts(unittest.TestCase):
    def test_camera_pair_differs_only_in_identity_and_selected_camera(self):
        configs=conditions();a=copy.deepcopy(configs[2]);b=copy.deepcopy(configs[3])
        self.assertEqual((a.pop('id'),b.pop('id')),('agent_fresh_d5','wrist_fresh_d5'))
        self.assertEqual((a.pop('fresh_camera'),b.pop('fresh_camera')),(0,1));self.assertEqual(a,b)
        self.assertEqual((a['D'],a['period'],a['R'],a['refresh_D']),(5,4,5,10))
        for cfg in configs:
            self.assertEqual(validate_condition(cfg),((),()))
            bad=dict(cfg,D=2)
            with self.assertRaises(ValueError):validate_condition(bad)

    def test_camera_validator_rejects_oracle_wrong_batch_age_or_helper(self):
        cfg=conditions()[2]
        audit=dict(kind='single',updated_views=[0],cached_views=[1],vision_pipeline_invocations=1,
            single_view_arithmetic='native_two_view_projection_order',
            actual_kernel_input_batch=1,image_layer_evaluations=27,cache_source_step=25,cache_age_steps=20,
            numerical_reference_only=False,camera_order=['agentview','wrist'],
            scope='Observed native input batch and feature cache; image-layer evaluations are not kernel-call counts or FLOPs',original_vision_helper_restored=True)
        row=dict(route=dict(camera_execution_audit=audit),timing=dict(camera_execution_audit=copy.deepcopy(audit)))
        validate_camera(row,cfg,False,25,20)
        for field,value in [('numerical_reference_only',True),('actual_kernel_input_batch',2),('cache_source_step',0),('cache_age_steps',5),('original_vision_helper_restored',False)]:
            bad=copy.deepcopy(row);bad['route']['camera_execution_audit'][field]=value
            with self.assertRaises(ValueError):validate_camera(bad,cfg,False,25,20)

    def test_native_AE18_D5_audit_rejects_fewer_visits(self):
        audit=dict(actual_NFE=5,attempted_layers_per_step=list(range(1,19)),executed_layers_per_step=list(range(1,19)),bypassed_layers_per_step=[],
            actual_layer_calls=90,attempted_layer_calls=90,bypassed_layer_calls=0,original_helper_restored=True,
            scope='Observed eager helper dispatch, not FLOPs, latency or quality')
        row=dict(route=dict(ae_execution_audit=audit,ae_bypassed_layers=[]),timing=dict(ae_execution_audit=copy.deepcopy(audit),actual_NFE=5))
        validate_ae(row,[],5)
        for field,value in [('actual_NFE',2),('actual_layer_calls',89),('bypassed_layers_per_step',[2]),('original_helper_restored',False)]:
            bad=copy.deepcopy(row);bad['route']['ae_execution_audit'][field]=value
            with self.assertRaises(ValueError):validate_ae(bad,[],5)


if __name__=='__main__':unittest.main()
