import unittest
import torch
from .backend import static_patch
from .common import MASK

class Formulas(unittest.TestCase):
    def test_three_static_maps(self):
        torch.manual_seed(3412);x=torch.randn(3,2048).to(torch.bfloat16);z=x.float()
        a=dict(scale=torch.full((2048,),1.25),bias=torch.full((2048,),.5))
        self.assertTrue(torch.equal(static_patch(x,'channel',a),z*1.25+.5))
        identity=dict(scale=torch.ones(2048),bias=torch.zeros(2048))
        self.assertTrue(torch.allclose(static_patch(x,'hadamard',identity),z,atol=1e-5,rtol=1e-5))
        a=dict(input_mean=torch.zeros(2048),residual_mean=torch.ones(2048),left=torch.zeros(2048,32),right=torch.zeros(32,2048))
        self.assertTrue(torch.equal(static_patch(x,'linear',a,8),z+1))
        self.assertEqual(MASK,(2,3,5,14,16,17))

if __name__=='__main__':unittest.main()
