import unittest
import numpy as np
from .rules_c1 import gate,pose_angle


class Boundary(unittest.TestCase):
    def test_strict_equal_boundary_and_sign(self):
        a=np.tile([.1,0,0,0,0,0,-1.],(5,1)); q=np.tile([0.,0,0,1],(5,1)); q[-1]=[0,0,np.sin(.1),np.cos(.1)]
        threshold=pose_angle(q)
        self.assertFalse(gate(a,q,pose=threshold)[0])
        self.assertTrue(gate(a,q,pose=np.nextafter(threshold,np.inf))[0])
        self.assertFalse(gate(a,q,pose=np.nextafter(threshold,-np.inf))[0])
        q[-1]*=-1; self.assertEqual(pose_angle(q),threshold)
        self.assertTrue(gate(a,q,pose=None)[0])


if __name__=='__main__': unittest.main()
