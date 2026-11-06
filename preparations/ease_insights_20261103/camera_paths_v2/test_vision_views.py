"""CPU alias/shape checks; these do not prove GPU numerical equivalence."""
import unittest
import numpy as np
from .vision_views import BATCH_KEYS,SHAPES,SPLIT_KEY,SCRATCH_PER_VIEW,camera_buffers


class CameraBufferContracts(unittest.TestCase):
    def make(self):
        values={k:np.zeros(shape,np.float32) for k,shape in SHAPES.items()}
        values[SPLIT_KEY]=np.zeros(2*SCRATCH_PER_VIEW,np.float32);values['encoder_K']=object()
        return values

    def test_one_view_shapes_are_real_aliases(self):
        buffers=self.make()
        for camera in (0,1):
            selected=camera_buffers(buffers,camera)
            for key in BATCH_KEYS:
                self.assertEqual(selected[key].shape,(1,*SHAPES[key][1:]))
                self.assertTrue(np.shares_memory(selected[key],buffers[key]))
            self.assertEqual(selected[SPLIT_KEY].shape,(SCRATCH_PER_VIEW,))
            self.assertIs(selected['encoder_K'],buffers['encoder_K'])

    def test_selected_view_write_cannot_change_other_view(self):
        for camera in (0,1):
            buffers=self.make();selected=camera_buffers(buffers,camera)
            for key in BATCH_KEYS:
                selected[key].fill(7)
                self.assertTrue(np.all(buffers[key][camera]==7))
                self.assertTrue(np.all(buffers[key][1-camera]==0))

    def test_wrong_camera_or_batch_shape_is_rejected(self):
        buffers=self.make()
        for camera in (-1,2,True):
            with self.assertRaises(ValueError):camera_buffers(buffers,camera)
        buffers['vision_QKV']=np.zeros((1,256,3456),np.float32)
        with self.assertRaises(ValueError):camera_buffers(buffers,0)

    def test_short_scratch_is_rejected(self):
        buffers=self.make();buffers[SPLIT_KEY]=buffers[SPLIT_KEY][:-1]
        with self.assertRaises(ValueError):camera_buffers(buffers,1)


if __name__=='__main__':unittest.main()
