import unittest
import numpy as np
from preparations.idea1_v2_1.runtime_models import NativePolicy, digest, native_observation, quat_axisangle


class RuntimeTests(unittest.TestCase):
    def test_hash(self):
        a=np.arange(6,dtype=np.uint8).reshape(2,3)
        self.assertEqual(digest({'a':a,'b':2}),digest({'b':2,'a':a.copy()}))
        self.assertNotEqual(digest(a),digest(a.astype(np.int16)))

    def test_native_observation(self):
        image=np.arange(18,dtype=np.uint8).reshape(2,3,3)
        images,state=native_observation(dict(agentview_image=image,robot0_eye_in_hand_image=image,
            robot0_eef_pos=[1,2,3],robot0_eef_quat=[0,0,0,1],robot0_gripper_qpos=[.1,.2]))
        np.testing.assert_array_equal(np.asarray(images[0]),image[::-1,::-1])
        np.testing.assert_allclose(state,[1,2,3,0,0,0,.1,.2])

    def test_quaternion(self):
        np.testing.assert_allclose(quat_axisangle([0,0,1,0]),[0,0,np.pi])

    def test_redundant_forward_cleanup_requires_exact_identity(self):
        class Layer:
            def forward(self): pass
        policy=object.__new__(NativePolicy)
        layer=Layer(); policy.layers=[layer]
        layer.forward=layer.forward
        self.assertEqual(policy._clear_redundant_class_forwards(),[1])
        self.assertNotIn('forward',vars(layer))
        layer.forward=lambda: None
        with self.assertRaisesRegex(ValueError,'Unknown instance'):
            policy._clear_redundant_class_forwards()
        self.assertIn('forward',vars(layer))


if __name__=='__main__': unittest.main()
