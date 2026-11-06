"""CPU checks of schedule semantics and real upstream DDIM construction."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np

from preparations.idea1_v2_2.schedules import normalize_schedule,flow_schedule,ddim_indices
from preparations.idea1_v2_2.runtime_models import NativePolicy

class FakePolicy(NativePolicy):
    """Checks acceptance orchestration only, never stands in for GPU parity."""
    def __init__(self,model_id='pi0'):
        self.model_id=model_id; self.calls=0; self.output_dir=None; self.code_hashes={}; self.visited=[]
        self.pi0=SimpleNamespace(startup_events=[])
    def prepare_condition(self,obs,instruction,skip=(),*args,**kwargs):
        return dict(input_hash='real-point-placeholder',skip=skip)
    def sample_condition(self,prepared,seed,schedule,category='parity',**kwargs):
        self.calls+=1; self.visited.append((schedule['kind'],schedule['num_steps'],schedule['m'],category))
        return dict(actions=np.ones((50,7),dtype=np.float32),noise_hash='fixed',audit=dict(actual_NFE=schedule['num_steps']))
    def predict(self,obs,instruction,seed,skip,schedule,category='parity',**kwargs):
        return self.sample_condition(self.prepare_condition(obs,instruction,skip),seed,schedule,category)
    def native_reference(self,obs,instruction,seed,skip,n,return_details=False):
        self.calls+=1
        return dict(actions=np.ones((50,7),dtype=np.float32),noise_hash='fixed',audit=dict(actual_NFE=n))

class SchedulesTest(unittest.TestCase):
    def test_pi0_native_fused_d10_and_canonical_uniform2(self):
        import torch
        from preparations.idea1_v2_2.pi0_adapter import Pi0Adapter
        from preparations.idea1_v2_2.schedules import flow_schedule
        torch.manual_seed(71)
        w={
            'decoder_action_fused_time_biases':torch.randn(10,1024,dtype=torch.bfloat16),
            'decoder_action_fused_out_proj_w':torch.randn(1024,32,dtype=torch.bfloat16),
            'decoder_action_fused_out_proj_b':torch.randn(32,dtype=torch.bfloat16),
            'decoder_action_time_mlp_in_w':torch.randn(2048,1024,dtype=torch.bfloat16),
            'decoder_action_in_proj_b':torch.randn(1024,dtype=torch.bfloat16),
            'decoder_action_time_mlp_in_b':torch.randn(1024,dtype=torch.bfloat16)}
        adapter=Pi0Adapter.__new__(Pi0Adapter); adapter.torch=torch; adapter.schedules={}
        runtime=SimpleNamespace(weights=w)
        bias,outs,_,_=adapter.schedule(runtime,flow_schedule('pi0',num_steps=10))
        self.assertTrue(torch.equal(bias,w['decoder_action_fused_time_biases']))
        self.assertTrue(all(torch.equal(v,w['decoder_action_fused_out_proj_w']) for v in outs))
        bias,_,_,_=adapter.schedule(runtime,flow_schedule('pi0',.5))
        mlp=w['decoder_action_time_mlp_in_w']; times=1.-torch.arange(2,dtype=torch.float32)/2.
        fraction=torch.arange(0,1024,2,dtype=torch.float32)/1024
        period=4e-3*(4./4e-3)**fraction
        pos=(1.0/period*(2.*torch.pi))[None,:]*times[:,None]
        emb=torch.cat([torch.sin(pos),torch.cos(pos)],1).to(mlp.dtype)
        expected=((mlp[:1024].T@w['decoder_action_in_proj_b'])[None,:]+emb@mlp[1024:]+w['decoder_action_time_mlp_in_b'][None,:]).to(torch.bfloat16)
        self.assertTrue(torch.equal(bias,expected))
    def test_parity_orchestrates_all_nodes_native_and_roundtrip_in_budget(self):
        for model in ('pi0','xiaomi-robotic0','db-cogact'):
            policy=FakePolicy(model)
            schedules=[dict(kind='ddim_custom' if model=='db-cogact' else 'flow',m=m,num_steps=2) for m in (.5,.3,.7,.1,.25,.9,.75,.6)]
            result=policy.parity({},'instruction',schedules=schedules)
            self.assertTrue(result['passed'])
            self.assertEqual(result['actual_predictions'],policy.calls)
            self.assertLessEqual(policy.calls,4*len(schedules)+8)
            labels=[r['comparison'] for r in result['comparisons']]
            self.assertEqual(labels.count('cache_vs_independent'),8)
            self.assertEqual(labels.count('roundtrip_custom_vs_initial'),8)
            self.assertEqual(labels.count('minimal_vs_instrumented'),8)
            self.assertIn('native_roundtrip',labels)
            self.assertIn('upstream_native_D2',labels)

    def test_parity_rejects_wrong_actual_nfe(self):
        policy=FakePolicy()
        original=policy.sample_condition
        def wrong(*args,**kwargs):
            result=original(*args,**kwargs); result['audit']['actual_NFE']=1; return result
        policy.sample_condition=wrong
        self.assertFalse(policy.parity({},'instruction')['passed'])
    def test_flow_direction_and_complete_interval(self):
        for model in ('pi0','xiaomi-robotic0'):
            for m in (.5,.3,.7,.1,.25,.9,.75,.6):
                s=flow_schedule(model,m)
                self.assertEqual(s['NFE'],2)
                self.assertAlmostEqual(sum(s['dt']),-1 if model=='pi0' else 1)
                # A varying velocity exposes using stale initial x at step two.
                x=2.
                for t,dt in zip(s['timesteps'],s['dt']): x=x+dt*(x+t)
                first=2+s['dt'][0]*(2+s['timesteps'][0])
                self.assertAlmostEqual(x,first+s['dt'][1]*(first+s['timesteps'][1]))

    def test_reject_cross_family_and_truncation(self):
        for model,s in [('pi0',dict(kind='ddim_custom',m=.5,num_steps=2)),
                        ('db-cogact',dict(kind='flow',m=.5,num_steps=2)),
                        ('xiaomi-robotic0',dict(kind='flow',m=1,num_steps=2)),
                        ('pi0',dict(kind='flow',m=.5,num_steps=3)),
                        ('pi0',dict(kind='native',m=None,num_steps=True))]:
            with self.assertRaises(ValueError): normalize_schedule(model,s)

    def test_native_defaults(self):
        self.assertEqual(normalize_schedule('xiaomi-robotic0')['num_steps'],5)
        self.assertEqual(normalize_schedule('db-cogact')['num_steps'],10)

    def test_actual_db_respacing_and_custom_clean_endpoint(self):
        path=Path(__file__).resolve().parents[2]/'policies/db-cogact/dexbotic/model/cogact/action_model/diffusion.py'
        spec=importlib.util.spec_from_file_location('native_diffusion_v22_test',path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        kwargs=dict(noise_schedule='squaredcos_cap_v2',diffusion_steps=100,sigma_small=True,learn_sigma=False)
        native10=module.create_diffusion('ddim10',**kwargs)
        native2=module.create_diffusion('ddim2',**kwargs)
        self.assertEqual(native10.timestep_map,list(range(0,100,10)))
        self.assertEqual(native2.timestep_map,[0,50])
        base=module.create_diffusion('',**kwargs)
        indices=ddim_indices(max(native10.timestep_map),.5)
        custom=module.SpacedDiffusion(use_timesteps=indices,betas=base.betas,
            model_mean_type=native10.model_mean_type,model_var_type=native10.model_var_type,loss_type=native10.loss_type)
        self.assertEqual(custom.timestep_map,[45,90])
        self.assertEqual(custom.alphas_cumprod_prev[0],1.)
        np.testing.assert_allclose(custom.alphas_cumprod,base.alphas_cumprod[[45,90]])
        self.assertNotEqual(custom.timestep_map,native2.timestep_map)

if __name__=='__main__': unittest.main()
