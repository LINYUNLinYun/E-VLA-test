"""Preparation only: native D10/D2 arithmetic with observed AE helper bypass.

No model is constructed on import. This backend has not passed real-GPU
numerical gates and is not connected to the running broad-mask experiment.
"""
import hashlib
import importlib
from pathlib import Path
import torch
from preparations.ease_insights_20261103.vlm.backend import Backend as VLMBackend
from .ae_dispatch import ScopedLayers,sample_with_dispatch

ROOT=Path(__file__).resolve().parents[3]
SAMPLER=ROOT/'preparations/idea1_v2_2/pi0_adapter.py'
HELPER=ROOT/'policies/realtime-vla-flash/scripts/experiment1/run_intermediate_action_probe.py'
SOURCE_PINS={SAMPLER:'5048a70340b0c6cda65c9d665011b0c8f92cd7d997df1102df534992b4bda623',
             HELPER:'e3ebf445ebb5144f2a2615b63b5efd229a75b6bfe78550267768ac397ec8ef8e'}


def verify_native_sources():
    for path,expected in SOURCE_PINS.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Native AE arithmetic source changed: '+str(path))


class Backend(VLMBackend):
    def __init__(self):
        verify_native_sources()
        super().__init__()
        helper=importlib.import_module('scripts.experiment1.run_intermediate_action_probe')
        if Path(helper._decoder_layer.__code__.co_filename).resolve()!=HELPER:
            raise ValueError('Unexpected native decoder helper origin')
        self.ae_dispatch=ScopedLayers(helper)
        self.protocol.update(ae_dispatch_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            native_ae_sources={str(p):v for p,v in SOURCE_PINS.items()},
            skip_semantics='VLM original-index KV shells and residuals as before; per-request AE hidden updates bypassed only on FastD2; FullD10 retains18',
            AE_numerical_gate_passed=False)

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=()):
        result,dispatch=sample_with_dispatch(self.adapter,self.ae_dispatch,context,seed,steps,ae_skip)
        actions,raw,_,audit,ms,decode=result
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=audit['nodes'],
            actual_NFE=audit['actual_NFE'],ae_execution_audit=dispatch)

    @torch.no_grad()
    def sample_native_reference(self,context,seed,steps):
        if self.ae_dispatch.namespace._decoder_layer is not self.ae_dispatch.original:
            raise RuntimeError('Native reference requested inside an active AE scope')
        return super().sample(context,seed,steps)
