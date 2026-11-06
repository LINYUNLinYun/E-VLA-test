"""CPU tests for camera ownership and cache semantics; GPU arithmetic is separate."""
import unittest
import torch
from .vision_views import SHAPES,SPLIT_KEY,SCRATCH_PER_VIEW
from .vision_dispatch import run_vision


class VisionDispatchContracts(unittest.TestCase):
    def setUp(self):
        self.b={k:torch.zeros(v) for k,v in SHAPES.items()}
        self.b[SPLIT_KEY]=torch.zeros(2*SCRATCH_PER_VIEW);self.calls=[]

    def native(self,w,b,views):
        self.calls.append((views,b['vision_x'].shape[0]))
        for i in range(views):b['vision_x'][i].fill_(float(b['observation_images_normalized'][i,0,0,0])+3)

    def test_refresh_captures_independent_raw_history(self):
        cache,a=run_vision(self.native,None,self.b,dict(kind='refresh'))
        self.b['vision_x'].zero_()
        self.assertTrue(torch.all(cache==3));self.assertEqual(self.calls,[(2,2)])
        self.assertEqual((a['actual_kernel_input_batch'],a['image_layer_evaluations']),(2,54))

    def test_selected_camera_is_computed_other_is_own_refresh(self):
        for camera in (0,1):
            old=torch.stack([torch.full((256,1152),11.),torch.full((256,1152),17.)])
            self.b['observation_images_normalized'].fill_(7.)
            cache,a=run_vision(self.native,None,self.b,dict(kind='single',camera=camera,cache=old,source_step=25,age_steps=20),single_native=self.native)
            self.assertIsNone(cache);self.assertTrue(torch.all(self.b['vision_x'][camera]==10))
            self.assertTrue(torch.equal(self.b['vision_x'][1-camera],old[1-camera]))
            self.assertEqual(a['updated_views'],[camera]);self.assertEqual(a['cached_views'],[1-camera])
            self.assertEqual((a['actual_kernel_input_batch'],a['cache_source_step'],a['cache_age_steps']),(1,25,20))
        self.assertEqual(self.calls,[(1,1),(1,1)])

    def test_oracle_is_explicit_and_does_not_call_vision(self):
        raw=torch.full((2,256,1152),9.)
        _,a=run_vision(self.native,None,self.b,dict(kind='reference',raw=raw))
        self.assertTrue(a['numerical_reference_only']);self.assertEqual(self.calls,[])
        self.assertTrue(torch.equal(self.b['vision_x'],raw))

    def test_bad_cache_rejected_before_any_native_call(self):
        for cache,source in [(None,0),(torch.zeros((1,256,1152)),0),(torch.zeros((2,256,1152),dtype=torch.float64),0),(torch.zeros((2,256,1152)),-1)]:
            with self.assertRaises(ValueError):run_vision(self.native,None,self.b,dict(kind='single',camera=0,cache=cache,source_step=source))
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
