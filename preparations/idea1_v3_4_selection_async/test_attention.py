import unittest
import torch
from .core import split_attend,ORIGINAL_ATTEND


class MaskSemantics(unittest.TestCase):
    def test_split_preserves_values_and_three_input_derivatives(self):
        torch.manual_seed(18);torch.set_num_threads(4)
        q=torch.randn(51,2048,dtype=torch.float64,requires_grad=True)
        k=torch.randn(55,256,dtype=torch.float64,requires_grad=True)
        v=torch.randn(55,256,dtype=torch.float64,requires_grad=True)
        mask=torch.ones(51,55,dtype=torch.bool);mask[0,5:]=False
        expected=ORIGINAL_ATTEND(q,k,v,mask);got=split_attend(q,k,v,mask)
        torch.testing.assert_close(got,expected,atol=1e-12,rtol=1e-12)
        cotangent=torch.randn_like(got)
        a=torch.autograd.grad(expected,(q,k,v),cotangent)
        b=torch.autograd.grad(got,(q,k,v),cotangent)
        for x,y in zip(a,b):torch.testing.assert_close(x,y,atol=1e-10,rtol=1e-10)


if __name__=='__main__':unittest.main()
