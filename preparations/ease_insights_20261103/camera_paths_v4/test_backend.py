"""CPU-only helper ownership contracts;constructs no real backend/model."""
import threading
import types
import unittest
from unittest.mock import patch
from .backend import Backend,DenseBackend
from . import backend as module


class BackendOwnership(unittest.TestCase):
    def fixture(self):
        original=lambda *args:None;native=types.SimpleNamespace(vision_encoder=original)
        package=types.ModuleType('scripts');package.__path__=[];spec=types.ModuleType('scripts.spec');spec.__path__=[]
        triton=types.ModuleType('scripts.spec.triton');triton.__path__=[];triton.pi0_infer=native
        modules={'scripts':package,'scripts.spec':spec,'scripts.spec.triton':triton}
        b=object.__new__(Backend);b.native_vision=original;b.vision_lock=threading.Lock();b.vision_request=None;b.last_camera_audit=None
        return b,native,original,modules
    def test_success_consumes_request_and_restores_original_helper(self):
        b,n,original,modules=self.fixture();b.request_vision(dict(kind='native'))
        def parent(self,*args,**kwargs):n.vision_encoder({}, {},2);return {},{}
        with patch.dict('sys.modules',modules),patch.object(DenseBackend,'prepare_residual',new=parent),patch.object(module,'run_vision',return_value=dict(fresh_images_consumed=True)) as run:
            context,timing=b.prepare_residual({},'test')
        self.assertIs(n.vision_encoder,original);self.assertIsNone(b.vision_request);self.assertFalse(b.vision_lock.locked())
        self.assertTrue(context['camera_execution_audit']['original_vision_helper_restored']);run.assert_called_once()
    def test_parent_failure_restores_helper_and_unlocks(self):
        b,n,original,modules=self.fixture()
        def parent(self,*args,**kwargs):n.vision_encoder({}, {},2);raise RuntimeError('parent failed')
        with patch.dict('sys.modules',modules),patch.object(DenseBackend,'prepare_residual',new=parent),patch.object(module,'run_vision',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'parent failed'):b.prepare_residual({},'test')
        self.assertIs(n.vision_encoder,original);self.assertFalse(b.vision_lock.locked());self.assertIsNone(b.last_camera_audit)
    def test_prepatched_helper_is_not_overwritten(self):
        b,n,original,modules=self.fixture();external=lambda *args:None;n.vision_encoder=external
        with patch.dict('sys.modules',modules):
            with self.assertRaisesRegex(RuntimeError,'already changed'):b.prepare_residual({},'test')
        self.assertIs(n.vision_encoder,external);self.assertFalse(b.vision_lock.locked())
    def test_consumer_helper_ownership_violation_restores_then_raises(self):
        b,n,original,modules=self.fixture()
        def parent(self,*args,**kwargs):n.vision_encoder({}, {},2);n.vision_encoder=lambda *args:None;return {},{}
        with patch.dict('sys.modules',modules),patch.object(DenseBackend,'prepare_residual',new=parent),patch.object(module,'run_vision',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'ownership changed'):b.prepare_residual({},'test')
        self.assertIs(n.vision_encoder,original);self.assertFalse(b.vision_lock.locked());self.assertIsNone(b.last_camera_audit)
    def test_request_overwrite_or_vlm_bypass_is_rejected(self):
        b,n,original,modules=self.fixture();b.request_vision(dict(kind='native'))
        with self.assertRaises(RuntimeError):b.request_vision(dict(kind='native'))
        with self.assertRaises(ValueError):b.prepare_residual({},'test',skip=[3])
        self.assertFalse(b.vision_lock.locked());self.assertIs(n.vision_encoder,original)


if __name__=='__main__':unittest.main()
