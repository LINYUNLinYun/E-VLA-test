"""Verify the single-view bridge dispatches original two-part kernels and rejects batch2."""
from types import SimpleNamespace
import unittest
import numpy as np
from .arithmetic import project


class Kernel:
    def __init__(self):self.calls=[]
    def __getitem__(self,grid):
        def call(**kwargs):self.calls.append((grid,kwargs))
        return call


class ArithmeticContracts(unittest.TestCase):
    def test_original_two_part_kernels_receive_true_single_view_sequence(self):
        mm=Kernel();merge=Kernel();native=SimpleNamespace(matmul_512x1152x1152_twopart_bias_res=mm,combine_1536_1152_twopart=merge)
        x=np.zeros((1,256,1152));w,b,r,o,s=object(),object(),object(),object(),object()
        project(native,x,w,b,r,o,s)
        self.assertEqual((len(mm.calls),len(merge.calls)),(1,1))
        grid,args=mm.calls[0];self.assertEqual(grid,(128,));self.assertEqual((args['seq_len'],args['features'],args['hidden']),(256,1152,1152))
        self.assertIs(args['inp_ptr'],x);self.assertIs(args['old_ptr'],r);self.assertIs(args['out2_ptr'],s)
        self.assertEqual(merge.calls[0][1]['seq_len'],256)

    def test_two_view_input_is_not_silently_run_as_one(self):
        with self.assertRaises(ValueError):project(None,np.zeros((2,256,1152)),None,None,None,None,None)


if __name__=='__main__':unittest.main()
