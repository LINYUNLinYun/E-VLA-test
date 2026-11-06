"""Fresh per-camera depth allocation using the original native arithmetic."""
import threading
import torch
from preparations.ease_insights_20261103.module_tradeoff_v1.backend import Backend as DenseBackend
from .vision import run_vision


class Backend(DenseBackend):
    def __init__(self):
        super().__init__()
        from scripts.spec.triton import pi0_infer as native
        self.native_vision=native.vision_encoder;self.vision_lock=threading.Lock()
        self.vision_request=None;self.last_camera_audit=None
        self.protocol.update(camera_numerical_gate_passed=False,AE_numerical_gate_passed=False,
            skip_semantics='Bothcameraimagesfresh;VLM18/AE18;FullD10vision27+27,FastD2vision27+18or18+27,originalvisionlayerindicesandtwo-viewBF16projectionorder.')

    def reset_vision(self):
        self.vision_request=None;self.last_camera_audit=None

    def request_vision(self,value):
        if self.vision_request is not None:raise RuntimeError('Previous fresh-camera request was not consumed')
        self.vision_request=value

    @torch.no_grad()
    def prepare_residual(self,obs,instruction,skip=(),capture_layers=(),history=None,older=None,ratio=0.,**kwargs):
        if skip or capture_layers or history or older or ratio!=0 or kwargs:
            raise ValueError('Fresh vision allocation keeps VLM18 and has no residual/cache compensation')
        from scripts.spec.triton import pi0_infer as native
        if not self.vision_lock.acquire(blocking=False):raise RuntimeError('Vision allocation is single-request')
        request=self.vision_request or dict(kind='native');self.vision_request=None;self.last_camera_audit=None;calls=0;audit=None;installed=False
        def wrapped(weights,buffers,num_views):
            nonlocal calls,audit
            if num_views!=2 or calls:raise ValueError('Unexpected native vision invocation')
            calls+=1;audit=run_vision(self.native_vision,native,weights,buffers,request)
        try:
            if native.vision_encoder is not self.native_vision:raise RuntimeError('Native vision entry already changed')
            native.vision_encoder=wrapped;installed=True
            try:context,timing=super().prepare_residual(obs,instruction)
            finally:
                if installed:
                    lost=native.vision_encoder is not wrapped;native.vision_encoder=self.native_vision
                    if lost:raise RuntimeError('Vision helper ownership changed;original restored')
            if calls!=1 or audit is None:raise RuntimeError('Missing actual fresh vision dispatch')
            audit['original_vision_helper_restored']=True;self.last_camera_audit=audit;context['camera_execution_audit']=audit
            return context,timing
        finally:self.vision_lock.release()

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=()):
        if ae_skip:raise ValueError('Fresh vision allocation retains all18 AE layers')
        actions,raw,cost=super().sample(context,seed,steps,ae_skip=())
        if 'camera_execution_audit' not in context:raise ValueError('Missing vision allocation context identity')
        return actions,raw,dict(cost,camera_execution_audit=context['camera_execution_audit'])
