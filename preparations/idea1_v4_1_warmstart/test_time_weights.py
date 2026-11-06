"""A real frequency-endpoint regression: catches i/512 versus i/511."""
import math
from types import SimpleNamespace
import unittest
import torch
from .sampling import matching_time_weights
from .schedule import nodes


class TimeWeightsTest(unittest.TestCase):
    def test_time_projection_preserves_openpi_max_period(self):
        torch.set_num_threads(2)
        mlp=torch.zeros((2048,1024),dtype=torch.bfloat16)
        mlp[1024:]=torch.eye(1024,dtype=torch.bfloat16)
        runtime=SimpleNamespace(weights={
            'decoder_action_time_mlp_in_w':mlp,
            'decoder_action_in_proj_b':torch.zeros(1024,dtype=torch.bfloat16),
            'decoder_action_time_mlp_in_b':torch.zeros(1024,dtype=torch.bfloat16),
            'decoder_action_fused_out_proj_w':torch.zeros((1024,32),dtype=torch.bfloat16),
            'decoder_action_fused_out_proj_b':torch.zeros(32,dtype=torch.bfloat16)})
        adapter=SimpleNamespace(torch=torch,schedules={})
        biases,_,_,_=matching_time_weights(adapter,runtime,nodes(.2,2))
        for i,t in enumerate((.2,.1)):
            # Last frequency must have period exactly4, as in OpenPI.
            expected=torch.tensor([math.sin(2*math.pi*t/4),math.cos(2*math.pi*t/4)],dtype=torch.bfloat16)
            torch.testing.assert_close(biases[i,[511,1023]],expected,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()
