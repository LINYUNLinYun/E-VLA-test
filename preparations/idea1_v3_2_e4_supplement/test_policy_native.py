import json
from pathlib import Path
import unittest
import importlib.util
import numpy as np
from scipy.spatial.transform import Rotation
from .policy_native import (native_libero_axisangle, native_libero_state, ContinuousQuaternionLift,
    decode_actions, normalize, calvin_state7, calvin_relative_target, PolicyContract, require_matched_control)


class Contracts(unittest.TestCase):
    def test_preserved_native_quaternion_branch(self):
        root = Path(__file__).resolve().parents[2]
        source = json.loads((root/"results/idea1_v3_2/e4_cpu_adapters/source_reference.json").read_text())
        q = np.array(source["actual_eef_world_quaternion_xyzw"])
        native = native_libero_axisangle(q)
        canonical = Rotation.from_quat(q).as_rotvec()
        self.assertGreater(native[0], 3.1)
        self.assertLess(canonical[0], -3.1)
        np.testing.assert_allclose(Rotation.from_rotvec(native).as_matrix(), Rotation.from_quat(q).as_matrix(),atol=1e-12)
        np.testing.assert_allclose(canonical,source["reference_state8"][3:6],atol=1e-12)
        den=np.sqrt(1-q[3]**2)
        np.testing.assert_array_equal(native,q[:3]*2*np.arccos(q[3])/den)

    def test_lift_continuity_and_episode_reset(self):
        q = np.array([1.,0,0,0])
        lift = ContinuousQuaternionLift(q)
        np.testing.assert_allclose(lift.encode(-q),[np.pi,0,0])
        near = Rotation.from_rotvec([np.pi+.01,0,0]).as_quat()
        self.assertGreater(lift.encode(-near)[0],np.pi)
        lift.reset()
        np.testing.assert_array_equal(lift.previous,q)

    def test_quaternion_validation(self):
        with self.assertRaises(ValueError): native_libero_axisangle([2,0,0,0])
        np.testing.assert_array_equal(native_libero_axisangle([0,0,0,1]),np.zeros(3))

    def test_independent_decode_and_extra_delta(self):
        state=np.arange(8,dtype=float)
        mean=np.arange(7,dtype=float)/10
        std=np.arange(7,dtype=float)+1
        raw=np.ones((2,32))
        expected=np.tile(mean+std+1e-6,(2,1)); expected[:,:6]+=state[:6]
        np.testing.assert_allclose(decode_actions(raw,mean,std,state,extra_delta=True),expected)
        np.testing.assert_array_equal(normalize(mean,mean,std),np.zeros(7))
        np.testing.assert_array_equal(decode_actions(np.zeros((1,32)),mean,std,state,extra_delta=False)[0],mean)

    def test_calvin_native_state_and_euler_controller(self):
        state=calvin_state7([1,2,3],[.2,.3,.4],-1,gripper_semantics="action_sign")
        np.testing.assert_array_equal(state,[1,2,3,.2,.3,.4,-1])
        with self.assertRaises(ValueError): calvin_state7([1,2,3],[0,0,0],.04,gripper_semantics="action_sign")
        self.assertEqual(calvin_state7([1,2,3],[0,0,0],.04,gripper_semantics="opening_width_m")[-1],.04)
        xyz,euler,grip=calvin_relative_target(np.ones(7),state[:3],state[3:6],max_rel_pos=.02,
            max_rel_orn=.05,magic_scaling_pos=1,magic_scaling_orn=1)
        np.testing.assert_allclose(xyz,[1.02,2.02,3.02]); np.testing.assert_allclose(euler,[.25,.35,.45])
        self.assertEqual(grip,1)

    def test_control_families_not_silently_matched(self):
        a=PolicyContract("libero","norm1","delta7","libero",8,7,50,5)
        b=PolicyContract("calvin","norm2","relative7","calvin",7,7,50,5)
        require_matched_control(a,a)
        with self.assertRaises(ValueError): require_matched_control(a,b)
        p=PolicyContract("pusht","norm3","absolute_pixel2","pusht",2,2,50,50)
        self.assertEqual(p.execute,50)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"optional project Torch CPU test")
    def test_real_tensor_kv_shell_identity_cache_and_restore(self):
        import torch
        from types import SimpleNamespace
        from .policy_native import install_native_transformers_overlay,native_vlm_kv_shell
        install_native_transformers_overlay(Path(__file__).resolve().parents[2])
        class Cache:
            def update(self,k,v,index,kwargs): self.value=(k,v,index); return k,v
        class Layer:
            def __init__(self):
                self.self_attn=SimpleNamespace(head_dim=2,layer_idx=0,
                    k_proj=torch.nn.Linear(4,2,bias=False),v_proj=torch.nn.Linear(4,2,bias=False))
            def input_layernorm(self,x,cond): return x*2,None
            def forward(self,*args,**kwargs): raise AssertionError("full attention/MLP must not run")
        layer=Layer(); original=layer.forward
        model=SimpleNamespace(paligemma_with_expert=SimpleNamespace(paligemma=SimpleNamespace(
            language_model=SimpleNamespace(layers=[layer]))))
        x=torch.arange(12,dtype=torch.float32).view(1,3,4); cache=Cache()
        with native_vlm_kv_shell(model,(1,)) as counts:
            output=layer.forward(x,past_key_value=cache,use_cache=True,
                position_embeddings=(torch.ones(1,3,2),torch.zeros(1,3,2)))
            self.assertIs(output[0],x); self.assertEqual(counts["kv_shell_calls"],1)
            torch.testing.assert_close(cache.value[0],layer.self_attn.k_proj(x*2).view(1,3,1,2).transpose(1,2))
            torch.testing.assert_close(cache.value[1],layer.self_attn.v_proj(x*2).view(1,3,1,2).transpose(1,2))
        self.assertEqual(layer.forward,original)


if __name__ == "__main__": unittest.main()
