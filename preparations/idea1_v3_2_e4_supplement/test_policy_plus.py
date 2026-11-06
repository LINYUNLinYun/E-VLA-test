import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from .policy_plus import NativeLiberoPlusPolicy,HEADS,freeze
from .policy_server import native_libero_raw
from preparations.idea1_v3_2.transfer_prompts import instruction_embedding,SparseSafeTensorRows


class PlusContracts(unittest.TestCase):
    def test_native_original_observation_not_transfer_bridge(self):
        image=np.arange(4*5*3,dtype=np.uint8).reshape(4,5,3)
        q=np.array([.9996096147738871,-.0009063364830718679,-.027923584546988157,-.00026463527139470797])
        obs=dict(agentview_image=image,robot0_eye_in_hand_image=image+1,robot0_eef_pos=np.array([.1,.2,.3]),
            robot0_eef_quat=q,robot0_gripper_qpos=np.array([.04,-.04]))
        raw=native_libero_raw(obs)
        np.testing.assert_array_equal(raw["base_rgb"],image[::-1,::-1])
        np.testing.assert_array_equal(raw["wrist_rgb"],(image+1)[::-1,::-1])
        expected=np.r_[obs["robot0_eef_pos"],q[:3]*2*np.arccos(q[3])/np.sqrt(1-q[3]**2),obs["robot0_gripper_qpos"]].astype(np.float32)
        np.testing.assert_array_equal(raw["state8"],expected)
        self.assertGreater(raw["state8"][3],3.1)

    def test_native_metadata_and_matching_heads(self):
        policy=NativeLiberoPlusPolicy({},"hash",backend=SimpleNamespace(protocol={}))
        self.assertNotIn("continuous",policy.protocol["state_convention"])
        self.assertEqual(HEADS["libero_spatial"],"draft_spatial")
        self.assertEqual((policy.protocol["H"],policy.protocol["R"],policy.protocol["D"]),(50,5,10))

    def test_actual_new_language_tokens_and_all_real_rows(self):
        lookup=SparseSafeTensorRows()
        source=instruction_embedding("put the bowl on the plate",row_lookup=lookup)
        perturbed=instruction_embedding("gently place the bowl onto the plate please",row_lookup=lookup)
        self.assertNotEqual(source["token_ids"],perturbed["token_ids"])
        self.assertEqual(perturbed["embeddings"].shape[0],len(perturbed["token_ids"]))
        self.assertTrue(torch.isfinite(perturbed["embeddings"].float()).all())
        self.assertEqual(perturbed["instruction"],"gently place the bowl onto the plate please")

    def test_import_only_receipt_cannot_authorize_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);assets=directory/"assets.json";env=directory/"env.json"
            assets.write_text(json.dumps(dict(status="complete",gpu_used=False,source_sha256={})))
            env.write_text(json.dumps(dict(status="complete",renderer_constructed=False)))
            with self.assertRaisesRegex(ValueError,"actual native renderer"):
                freeze(assets,env,directory/"missing_selection.json",2e10)


if __name__=="__main__": unittest.main()
