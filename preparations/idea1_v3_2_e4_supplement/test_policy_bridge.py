import importlib.util
import json
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .policy_bridge import (bridge_proprio,bound_normalize,decode_bridge_actions,official_adapter,
    NativeBridgePolicy,DEFAULT_ROT,STATS,cpu_contract)


class BridgeContracts(unittest.TestCase):
    def test_topdown_rotation_is_not_identity_tcp_assumption(self):
        q=Rotation.from_matrix(DEFAULT_ROT).as_quat()
        pose=np.r_[[.3,.02,.1],q[[3,0,1,2]],.7]
        np.testing.assert_allclose(bridge_proprio(pose),[.3,.02,.1,0,0,0,.7],atol=1e-14)

    def test_external_state_bound_and_clip(self):
        stats=json.loads(STATS.read_text())
        low=np.asarray(stats["proprio"]["p01"]);high=np.asarray(stats["proprio"]["p99"])
        np.testing.assert_allclose(bound_normalize(low,stats),-np.ones(7))
        np.testing.assert_allclose(bound_normalize(high+1,stats),np.ones(7))
        self.assertTrue(np.isfinite(bound_normalize((low+high)/2,stats)).all())

    def test_native_decode_no_double_normalizer_or_libero_scale(self):
        stats=json.loads(STATS.read_text());raw=np.zeros((4,7));raw[:,6]=[0,.5,.50001,1]
        decoded=decode_bridge_actions(raw,stats)
        expected=(np.array(stats["action"]["p01"])+np.array(stats["action"]["p99"]))/2
        np.testing.assert_allclose(decoded[:,:3],np.tile(expected[:3],(4,1)))
        np.testing.assert_array_equal(decoded[:,6],[-1,-1,1,1])
        # Native denormalizer deliberately extrapolates rather than clipping.
        raw[:,0]=2
        self.assertGreater(decode_bridge_actions(raw,stats)[0,0],stats["action"]["p99"][0])

    def test_permission_gate_precedes_any_model_constructor(self):
        with self.assertRaises(PermissionError): NativeBridgePolicy()
        with self.assertRaises(PermissionError): NativeBridgePolicy(launch=True)

    def test_pinned_checkpoint_header_and_runtime_blocker(self):
        result=cpu_contract()
        self.assertEqual((result["H"],result["R"],result["D"]),(4,4,10))
        self.assertEqual(result["checkpoint_header"]["tensor_count"],776)
        self.assertFalse(result["model_constructed"])

    @unittest.skipUnless(importlib.util.find_spec("torch"),"project Torch CPU adapter parity")
    def test_independent_state_and_decoder_match_actual_official_adapter(self):
        adapter=official_adapter(use_bf16=False)
        q=Rotation.from_euler("xyz",[.2,-.3,.4]).as_quat()
        pose=np.r_[[.31,.03,.07],q[[3,0,1,2]],.8]
        observation={"agent":{"eef_pos":pose}}
        expected=adapter.preprocess_proprio(observation)
        np.testing.assert_allclose(bridge_proprio(pose),expected,atol=1e-12)
        stats=json.loads(STATS.read_text())
        raw=np.random.default_rng(7).normal(size=(4,7))
        np.testing.assert_allclose(decode_bridge_actions(raw,stats),adapter.postprocess(raw),atol=1e-12)
        image=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3)
        processed=adapter.preprocess({"observation.images.top":image,"observation.state":observation,"task":"put carrot on plate"})
        np.testing.assert_allclose(processed["observation.state"].numpy()[0],bound_normalize(expected,stats),atol=1e-7)
        self.assertEqual(tuple(processed["observation.images.top"].shape),(1,3,224,224))
        self.assertEqual(processed["task"],["put carrot on plate"])


if __name__=="__main__":unittest.main()
