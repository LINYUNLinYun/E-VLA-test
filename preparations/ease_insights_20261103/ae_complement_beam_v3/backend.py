"""Unchanged original adapter arithmetic with observed AE depth/time schedules."""
import torch
from preparations.ease_insights_20261103.module_tradeoff_v1.backend import Backend as DenseBackend
from .dispatch import sample_with_dispatch


class Backend(DenseBackend):
    def __init__(self):
        super().__init__()
        self.protocol.update(skip_semantics='VLM18 and both fresh vision27 branches; AE bypass only on Fast uniform-EulerD2/D3/D4; FullD10 retains AE18',
            adapter_euler_step_options=[2,3,4,10],AE_numerical_gate_passed=False)

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=()):
        result,observed=sample_with_dispatch(self.adapter,self.ae_dispatch,context,seed,steps,ae_skip)
        actions,raw,_,audit,ms,decode=result
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=audit['nodes'],actual_NFE=audit['actual_NFE'],ae_execution_audit=observed)
