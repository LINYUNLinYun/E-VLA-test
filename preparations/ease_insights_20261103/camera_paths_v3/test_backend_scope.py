"""CPU exception restoration of scoped vision dispatch without loading a model."""
import sys
import threading
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch
import torch
from .backend import Backend,DenseBackend
from .vision_views import SHAPES,SPLIT_KEY,SCRATCH_PER_VIEW


class VisionScopeContracts(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def native(w,b,n):self.calls.append(n);b['vision_x'].fill_(3)
        self.native=SimpleNamespace(vision_encoder=native)
        self.b=Backend.__new__(Backend);self.b.native_vision=native;self.b.vision_lock=threading.Lock()
        self.b.vision_request=None;self.b.last_full_vision=None;self.b.last_camera_audit=None
        package=ModuleType('scripts.spec.triton');package.pi0_infer=self.native
        self.modules={'scripts':ModuleType('scripts'),'scripts.spec':ModuleType('scripts.spec'),'scripts.spec.triton':package}
        self.buffers={k:torch.zeros(shape) for k,shape in SHAPES.items()}
        self.buffers[SPLIT_KEY]=torch.zeros(2*SCRATCH_PER_VIEW)

    def fake_prepare(self,backend,obs,prompt):
        self.native.vision_encoder(None,self.buffers,2);return {},{}

    def test_restore_after_success_and_capture_own_raw(self):
        self.b.request_vision(dict(kind='refresh'))
        with patch.dict(sys.modules,self.modules),patch.object(DenseBackend,'prepare_residual',lambda backend,obs,prompt:self.fake_prepare(backend,obs,prompt)):
            context,_=self.b.prepare_residual({},'prompt')
        self.assertIs(self.native.vision_encoder,self.b.native_vision)
        self.assertTrue(context['camera_execution_audit']['original_vision_helper_restored'])
        self.assertEqual(self.calls,[2]);self.assertFalse(self.b.vision_lock.locked())
        self.buffers['vision_x'].zero_();self.assertTrue(torch.all(self.b.last_full_vision==3))

    def test_restore_after_downstream_exception(self):
        def failed(*args):
            self.fake_prepare(*args);raise RuntimeError('intentional downstream failure')
        with patch.dict(sys.modules,self.modules),patch.object(DenseBackend,'prepare_residual',failed):
            with self.assertRaises(RuntimeError):self.b.prepare_residual({},'prompt')
        self.assertIs(self.native.vision_encoder,self.b.native_vision);self.assertFalse(self.b.vision_lock.locked())

    def test_unknown_entry_is_rejected_and_not_overwritten(self):
        other=lambda *args:None;self.native.vision_encoder=other
        with patch.dict(sys.modules,self.modules):
            with self.assertRaises(RuntimeError):self.b.prepare_residual({},'prompt')
        self.assertIs(self.native.vision_encoder,other);self.assertFalse(self.b.vision_lock.locked())

    def test_non_camera_residual_options_rejected(self):
        for kwargs in (dict(skip=(2,)),dict(ratio=.5),dict(reference=True)):
            with self.assertRaises(ValueError):self.b.prepare_residual({},'prompt',**kwargs)


if __name__=='__main__':unittest.main()
