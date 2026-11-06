"""Preparation: native D5/D10 AE18 with one fresh camera and one cached view."""
import threading
import torch
from preparations.ease_insights_20261103.module_tradeoff_v1.backend import Backend as DenseBackend
from .vision_dispatch import run_vision
from .arithmetic import single_view_vision


class Backend(DenseBackend):
    def __init__(self):
        super().__init__()
        from scripts.spec.triton import pi0_infer as native
        self.native_vision=native.vision_encoder;self.vision_lock=threading.Lock()
        self.vision_request=None;self.last_full_vision=None;self.last_camera_audit=None
        self.protocol.update(skip_semantics='Full VLM18/AE18; FastD5 updates one or both native camera branches; other raw vision features held from own Full refresh',
            camera_numerical_gate_passed=False,AE_numerical_gate_passed=False)

    def request_vision(self,value):
        if self.vision_request is not None:raise RuntimeError('Previous camera request was not consumed')
        self.vision_request=value

    def reset_vision(self):
        self.vision_request=None;self.last_full_vision=None;self.last_camera_audit=None

    @torch.no_grad()
    def prepare_residual(self,obs,instruction,skip=(),capture_layers=(),history=None,older=None,ratio=0.,**kwargs):
        if skip or capture_layers or history or older or ratio!=0 or kwargs:raise ValueError('Camera paths keep all VLM18 and use no VLM residual compensation')
        if not self.vision_lock.acquire(blocking=False):raise RuntimeError('Camera dispatch is single-request')
        from scripts.spec.triton import pi0_infer as native
        request=self.vision_request or dict(kind='both');self.vision_request=None
        cache=None;audit=None;calls=0
        def wrapped(weights,buffers,num_views):
            nonlocal cache,audit,calls
            if num_views!=2 or calls:raise ValueError('Unexpected native vision invocation')
            calls+=1;cache,audit=run_vision(self.native_vision,weights,buffers,request,
                single_native=lambda w,b,n:single_view_vision(native,w,b,n))
        try:
            if native.vision_encoder is not self.native_vision:raise RuntimeError('Native vision entry changed')
            native.vision_encoder=wrapped
            try:context,timing=super().prepare_residual(obs,instruction)
            finally:
                if native.vision_encoder is not wrapped:raise RuntimeError('Camera entry changed inside scoped dispatch')
                native.vision_encoder=self.native_vision
            if calls!=1 or audit is None:raise RuntimeError('Missing actual native vision dispatch')
            audit['original_vision_helper_restored']=True
            if cache is not None:self.last_full_vision=cache
            self.last_camera_audit=audit;context['camera_execution_audit']=audit
            return context,timing
        finally:self.vision_lock.release()

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=()):
        if ae_skip or type(steps) is not int or steps not in (5,10):raise ValueError('Camera paths use native AE18 at D5 or D10')
        with self.ae_dispatch.use(()) as observed:
            result=self.adapter.sample(context,seed,dict(kind='native',m=None,num_steps=steps),False)
        actions,raw,_,audit,ms,decode=result
        expected=list(range(1,19))*steps
        if audit['actual_NFE']!=steps or observed.attempted!=expected or observed.executed!=expected or observed.bypassed or not observed.original_restored:
            raise ValueError('Native D5/D10 actual AE18 visits differ')
        visits=dict(actual_NFE=steps,attempted_layers_per_step=list(range(1,19)),executed_layers_per_step=list(range(1,19)),
            bypassed_layers_per_step=[],actual_layer_calls=18*steps,attempted_layer_calls=18*steps,bypassed_layer_calls=0,
            original_helper_restored=True,scope='Observed eager helper dispatch, not FLOPs, latency or quality')
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=audit['nodes'],actual_NFE=steps,
            ae_execution_audit=visits,camera_execution_audit=context.get('camera_execution_audit'))
