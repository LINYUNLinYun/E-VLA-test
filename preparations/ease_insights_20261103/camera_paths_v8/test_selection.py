"""CPU query-role/layout/quota checks; no real-GPU or closed-loop claim."""
import unittest
from types import SimpleNamespace
import torch
from .selection import choose,language_query,state_query

class SelectionTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);self.prefix=516
        self.keys=torch.zeros((512,256),dtype=torch.bfloat16)
        for view in range(2):
            self.keys[view*256:view*256+128,0]=4
            self.keys[view*256+128:(view+1)*256,1]=4
        self.expected_language=list(range(128))+list(range(256,384))
        self.expected_state=list(range(128,256))+list(range(384,512))
    def test_ties_quota_and_original_positions(self):
        result=choose(torch.zeros((8,256)),self.keys,'state_query',1)
        self.assertEqual(result['kept_visual_indices'],self.expected_language)
        self.assertEqual(int(result['mask'][:256].sum()),128);self.assertEqual(int(result['mask'][256:].sum()),128)
        self.assertFalse(result['prefix_compute_reduction_claimed'])
    def test_language_role_excludes_image_queries_and_uses_L18(self):
        q=torch.zeros((self.prefix*8,256),dtype=torch.bfloat16);q[:512*8,1]=100;q[512*8:,0]=2
        k=torch.zeros((18,self.prefix,256),dtype=torch.bfloat16);k[17,:512]=self.keys
        context=dict(seq=self.prefix,K=k,runtime=SimpleNamespace(buffers={'encoder_Q':q}))
        result=language_query(context)
        self.assertEqual(result['kept_visual_indices'],self.expected_language);self.assertEqual(result['query_rows'],32)
        self.assertEqual(result['source_layer'],18)
    def test_state_role_excludes_noisy_action_queries(self):
        q=torch.zeros((408,256),dtype=torch.bfloat16);q[:8,1]=2;q[8:,0]=100
        k=torch.zeros((self.prefix+51,256),dtype=torch.bfloat16);k[:512]=self.keys
        result=state_query(q,k,self.prefix)
        self.assertEqual(result['kept_visual_indices'],self.expected_state);self.assertEqual(result['query_rows'],8)
        self.assertEqual(result['source_layer'],1)
    def test_invalid_layout_role_and_nonfinite_rejected(self):
        for q,k,mode in [(torch.zeros((7,255)),self.keys,'state_query'),(torch.zeros((8,256)),self.keys,'other'),(torch.full((8,256),float('nan')),self.keys,'state_query')]:
            with self.assertRaises(ValueError):choose(q,k,mode,1)
        with self.assertRaises(ValueError):state_query(torch.zeros((400,256)),torch.zeros((self.prefix+51,256)),self.prefix)

if __name__=='__main__':unittest.main()
