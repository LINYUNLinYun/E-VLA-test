"""CPU preparation: original dense-prefill VLM18/AE18 with optional KV edge removal.

This backend is unlaunched, not connected to a service and not GPU-validated.
"""
import hashlib
import torch
from pathlib import Path
from preparations.ease_insights_20261103.module_tradeoff_v1.backend import Backend as DenseBackend
from reports.ease_insights_20261103.prefix_kv_preparation_v2.packing import PrefixPacker
from .sampling import PrefixVisits,sample_with_prefix


class Backend(DenseBackend):
    def __init__(self):
        super().__init__()
        self.prefix_packer=PrefixPacker();self.prefix_visits=PrefixVisits(self.ae_dispatch.namespace)
        self.protocol.update(prefix_attention_numerical_gate_passed=False,
            prefix_attention_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            prefix_scope='Both fresh full vision27 and VLM18 prefill first;optional removal of half per-camera K/V entries for FastD2 AE18 only. No prefill saving is claimed.')

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=(),prefix_pattern='all'):
        if ae_skip:raise ValueError('This preparation always retains all18 action-expert layers')
        result,ae,prefix=sample_with_prefix(self.adapter,self.prefix_visits,self.prefix_packer,context,seed,steps,prefix_pattern)
        actions,raw,_,nodes,ms,decode=result
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=nodes['nodes'],actual_NFE=nodes['actual_NFE'],
            ae_execution_audit=ae,prefix_execution_audit=prefix,
            prefix_pack_wall_ms=prefix['packing_wall_ms'],prefix_pack_cuda_ms=prefix['packing_cuda_ms'],
            prefix_restore_wall_ms=prefix['restore_wall_ms'])
