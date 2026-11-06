"""CPU state-dependent operator and episode-isolation tests; no GPU claims."""
import copy
import unittest
import numpy as np
import torch
from preparations.idea1_v3_4_E2_module_skip.backend import encoder,module_identity
from preparations.idea1_v3_4_E2_module_skip.test_backend import MockKernels,buffers
from .oracle import encoder as oracle
from .common import conditions,FIXED_LAYERS,validate_condition
from .component_audit import expected,validate_components
from .policy import Policy


class FakeBackend:
    def reset_vision(self):pass
    def request_vision(self,value):
        if value!={'kind':'native'}:raise ValueError('Wrong vision')
    def prepare_components(self,obs,instruction,**kwargs):
        weights,b=buffers();identity=module_identity(instruction,2,b['encoder_x'].shape)
        residuals,_,meta=encoder(MockKernels(),weights,b,2,identity=identity,**kwargs)
        meta.pop('_profile_events',None)
        meta.update(original_encoder_helper_restored=True,grad_enabled=False,selector_calls=0,reference_only=False)
        return dict(module_identity=identity,residuals=residuals,module_meta=meta,camera_execution_audit={}),{}
    def sample(self,context,seed,steps):
        return np.zeros((50,7),np.float32),torch.zeros((50,32)),dict(component_execution_audit=context['module_meta'],
            component_reference_only=False,ae_execution_audit={},camera_execution_audit={})


class ComponentTest(unittest.TestCase):
    def test_full_compute_restore_oracle_matches_physically_skipped_routes(self):
        w,b=buffers();residuals,_,_=encoder(MockKernels(),w,b,2,capture=True)
        previous={k:{l:r*.98 for l,r in bank.items()} for k,bank in residuals.items()}
        for kind in ('attn','ffn'):
            for old,ratio in [(None,0.),(previous,1/3),(previous,2/3)]:
                kwargs={'skip_'+kind:FIXED_LAYERS,'history':residuals,'older':old,'ratio':ratio}
                w,prod=buffers();native=MockKernels();encoder(native,w,prod,2,**kwargs)
                w,ref=buffers();all_native=MockKernels();_,_,meta=oracle(all_native,w,ref,2,**kwargs)
                for key in ('encoder_x','encoder_K','encoder_V'):self.assertTrue(torch.equal(prod[key],ref[key]),(kind,ratio,key))
                for layer in FIXED_LAYERS:
                    if kind=='attn':
                        self.assertNotIn(('attn',layer),native.calls);self.assertNotIn(('qkv',layer),native.calls)
                        self.assertIn(('kv',layer),native.calls);self.assertIn(('attn',layer),all_native.calls)
                    else:
                        self.assertNotIn(('ffn_gate',layer),native.calls);self.assertNotIn(('ffn_down',layer),native.calls)
                        self.assertIn(('ffn_gate',layer),all_native.calls)
                self.assertTrue(meta['reference_only']);self.assertEqual(meta['full_compute_calls']['qkv'],list(range(1,19)))
    def test_policy_cadence_component_capture_history_and_ratio(self):
        for cfg in conditions():
            p=Policy(FakeBackend());p.configure(cfg)
            for i in range(9):
                r=p.predict({},'prompt','libero_goal',i);full=cfg['id']=='full_d10' or i%3==0
                self.assertEqual(r['route']['refresh'],full);validate_components(r,cfg,full)
                self.assertEqual(len(p.history),min(i//3+1,2) if cfg['component'] else 0)
                if p.history:self.assertEqual(set(p.history[-1]['residuals'][cfg['component']]),set(FIXED_LAYERS))
                p.acknowledge(r['actions'][:5])
    def test_partial_ack_uses_actual_executed_step_gap(self):
        cfg=conditions()[2];p=Policy(FakeBackend());p.configure(cfg)
        for n in [3,5,5,5]:
            r=p.predict({},'p','libero_goal',7);p.acknowledge(r['actions'][:n])
        r=p.predict({},'p','libero_goal',7);validate_components(r,cfg,False)
        self.assertEqual(r['route']['history_steps'],[0,13]);self.assertEqual(r['route']['extrapolation_ratio'],5/13)
    def test_reset_clears_prompt_residuals_and_pending_proposal(self):
        cfg=conditions()[2];p=Policy(FakeBackend());p.configure(cfg)
        r=p.predict({},'first','libero_goal',7)
        with self.assertRaises(RuntimeError):p.predict({},'first','libero_goal',8)
        p.acknowledge(r['actions'][:5])
        with self.assertRaises(ValueError):p.predict({},'changed','libero_goal',7)
        p.configure(cfg);r=p.predict({},'changed','libero_goal',7)
        self.assertEqual(r['step'],0);self.assertTrue(r['route']['refresh']);self.assertEqual(len(p.history),1)
    def test_audit_rejects_forged_route_reference_and_ratio(self):
        cfg=conditions()[2];p=Policy(FakeBackend());p.configure(cfg)
        r=p.predict({},'p','libero_goal',7);p.acknowledge(r['actions'][:5]);r=p.predict({},'p','libero_goal',7)
        validate_components(r,cfg,False)
        for key,value in [('reference_only',True),('skip_attn',[]),('ratio',.5),('current_observation_kv',False),('grad_enabled',True)]:
            forged=copy.deepcopy(r);forged['route']['component_execution_audit'][key]=value
            with self.assertRaises(ValueError):validate_components(forged,cfg,False)
    def test_candidate_layer_positions_and_component_kind_are_frozen(self):
        for key,value in [('component','other'),('component_layers',[1,2,3,4,5,6]),('period',4),('compensation',None)]:
            cfg=copy.deepcopy(conditions()[2]);cfg[key]=value
            with self.assertRaises(ValueError):validate_condition(cfg)


if __name__=='__main__':unittest.main()
