import unittest
import numpy as np
from .rules import gate,acceptance


class Rules(unittest.TestCase):
    def test_quaternion_sign_and_wrap(self):
        commands=np.tile([.1,0,0,0,0,0,-1.],(5,1))
        q=np.array([[0,0,np.sin(np.radians(a)/2),np.cos(np.radians(a)/2)] for a in (179,179.5,180,180.5,181)])
        q[1]*=-1
        self.assertTrue(gate(commands,q,pose=2.01)[0])
        self.assertFalse(gate(commands,q,pose=1.9)[0])
        q[0]=0; self.assertIn('c_invalid',gate(commands,q)[1])

    def test_accumulated_pose(self):
        a=np.tile([.1,0,0,0,0,0,-1.],(5,1))
        q=np.array([[0,0,np.sin(np.radians(x)/2),np.cos(np.radians(x)/2)] for x in (0,2,4,6,8)])
        self.assertFalse(gate(a,q,pose=5)[0])
        self.assertTrue(gate(a,q,pose=None)[0])

    def test_causal_history_and_zero_command(self):
        a=np.zeros((5,7)); q=np.tile([0,0,0,1.],(5,1))
        self.assertTrue(gate(a,q)[0]); a[-1,0]=.1
        self.assertIn('a_turn',gate(a,q)[1]); self.assertFalse(gate(a[:4],q[:4])[0])

    def test_soft_budget(self):
        raw=np.zeros((5,7)); hats=np.zeros((2,5,7)); hats[:,:,:6]=np.array([.04,.11,.04,.04,.04])[None,:,None]
        self.assertEqual(acceptance(raw,hats,.1)[0],1)
        self.assertEqual(acceptance(raw,hats,.1,alpha=1.25)[0],5)
        hats[:,2,:6]=.11
        self.assertEqual(acceptance(raw,hats,.1,alpha=1.25)[0],2)
        hats[:,0,:6]=.2
        self.assertEqual(acceptance(raw,hats,.1,alpha=1.5)[0],0)

    def test_xiaomi_no_gripper_protection(self):
        raw=np.zeros((10,32)); hats=np.zeros((2,10,32)); hats[:,:,6]=1
        self.assertEqual(acceptance(raw,hats,.1,10,last_gripper=-1,protect=False)[0],10)
        self.assertEqual(acceptance(raw,hats,.1,10,last_gripper=-1,protect=True)[0],0)


if __name__=='__main__': unittest.main()
