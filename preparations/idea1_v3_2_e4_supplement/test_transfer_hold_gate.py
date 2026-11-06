import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .transfer_hold_gate import rot


class HoldGateTest(unittest.TestCase):
    def test_actual_rotation_conventions(self):
        expected=Rotation.from_euler('xyz',[.1,-.2,.3])
        q=expected.as_quat()[[3,0,1,2]]
        for actual in [dict(eef_euler_world=[.1,-.2,.3]),dict(eef_quaternion_wxyz=q)]:
            self.assertLess(np.linalg.norm((rot(actual)*expected.inv()).as_rotvec()),1e-12)

    def test_world_axis_projection_not_raw_world_sign(self):
        base=Rotation.from_euler('z',np.pi).as_matrix()
        for sign in [1,-1]:
            expected=base.dot(np.array([sign,0,0]));delta=expected*.05
            self.assertGreater(delta.dot(expected),0)
            self.assertEqual(np.sign(delta[0]),-sign)


if __name__=='__main__':unittest.main()
