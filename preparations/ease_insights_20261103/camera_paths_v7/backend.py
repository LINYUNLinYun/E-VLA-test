"""Fixed VLM component bypass; current-observation KV and fresh native vision."""
import importlib
import threading
import torch
from preparations.idea1_v3_4_E2_module_skip import backend as modules
from preparations.ease_insights_20261103.module_tradeoff_v1.ae_dispatch import ScopedLayers,sample_with_dispatch
from preparations.ease_insights_20261103.module_tradeoff_v1.backend import verify_native_sources,HELPER
from preparations.ease_insights_20261103.camera_paths_v4.vision import run_vision
from .oracle import encoder as oracle_encoder


class Backend(modules.Backend):
    def __init__(self):
        verify_native_sources();super().__init__()
        from scripts.spec.triton import pi0_infer as native
        from pathlib import Path
        helper=importlib.import_module('scripts.experiment1.run_intermediate_action_probe')
        if Path(helper._decoder_layer.__code__.co_filename).resolve()!=HELPER:
            raise ValueError('Unexpected native decoder helper')
        self.ae_dispatch=ScopedLayers(helper)
        self.native_vision=native.vision_encoder;self.vision_lock=threading.Lock()
        self.reset_vision()
        self.protocol.update(component_numerical_gate_passed=False,AE_numerical_gate_passed=False,
            camera_numerical_gate_passed=False,
            skip_semantics='Fixed VLM attention-only or FFN-only skips; current-observation KV at all18 original layers; Full residual capture and Fast extrapolation; both cameras fresh27+27, AE18.')

    def reset_vision(self):self.vision_request=None;self.last_camera_audit=None
    def request_vision(self,value):
        if value!={'kind':'native'}:raise ValueError('Only full fresh native vision is allowed')
        if self.vision_request is not None:raise RuntimeError('Unconsumed fresh-vision request')
        self.vision_request=value

    @torch.no_grad()
    def prepare_components(self,obs,instruction,oracle=False,**kwargs):
        from scripts.spec.triton import pi0_infer as native
        if not self.vision_lock.acquire(blocking=False):raise RuntimeError('Single-request backend')
        request=self.vision_request or dict(kind='native');self.vision_request=None
        calls=0;audit=None
        def wrapped(weights,buffers,num_views):
            nonlocal calls,audit
            if num_views!=2 or calls:raise ValueError('Unexpected vision invocation')
            calls+=1;audit=run_vision(self.native_vision,native,weights,buffers,request)
        try:
            with modules._ENCODER_LOCK:
                original_encoder=modules.encoder
                native_encoder=native.transformer_encoder
                if native.vision_encoder is not self.native_vision:raise RuntimeError('Vision entry changed')
                native.vision_encoder=wrapped
                if oracle:modules.encoder=oracle_encoder
                try:context,timing=super().prepare_modules(obs,instruction,**kwargs)
                finally:
                    lost=native.vision_encoder is not wrapped
                    native.vision_encoder=self.native_vision;modules.encoder=original_encoder
                    if lost:raise RuntimeError('Vision ownership changed; original restored')
            if calls!=1 or audit is None:raise RuntimeError('Missing fresh native vision')
            if native.transformer_encoder is not native_encoder:raise RuntimeError('Native encoder helper not restored')
            context['module_meta'].update(original_encoder_helper_restored=True,
                grad_enabled=bool(torch.is_grad_enabled()),selector_calls=0,reference_only=bool(oracle))
            audit['original_vision_helper_restored']=True
            self.last_camera_audit=audit;context['camera_execution_audit']=audit
            context['component_reference_only']=bool(oracle)
            return context,timing
        finally:self.vision_lock.release()

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=()):
        if ae_skip:raise ValueError('All18 AE layers must execute')
        result,dispatch=sample_with_dispatch(self.adapter,self.ae_dispatch,context,seed,steps,())
        actions,raw,_,audit,ms,decode=result
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=audit['nodes'],actual_NFE=audit['actual_NFE'],
            ae_execution_audit=dispatch,camera_execution_audit=context['camera_execution_audit'],
            component_execution_audit=context['module_meta'],component_reference_only=context['component_reference_only'])

    @torch.no_grad()
    def sample_native_reference(self,context,seed,steps):
        if self.ae_dispatch.namespace._decoder_layer is not self.ae_dispatch.original:
            raise RuntimeError('Native reference requested inside AE scope')
        return modules.TritonBackend.sample(self,context,seed,steps)
