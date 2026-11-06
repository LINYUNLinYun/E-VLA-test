"""Unlaunched full-fresh-vision backend plus compact-prefix AE and masked oracle."""
import torch
from preparations.ease_insights_20261103.camera_paths_v4.backend import Backend as FreshBackend
from reports.ease_insights_20261103.prefix_kv_preparation_v2.packing import PrefixPacker
from reports.ease_insights_20261103.prefix_kv_preparation_v3.sampling import PrefixVisits,sample_with_prefix
from reports.ease_insights_20261103.prefix_kv_preparation_v4.oracle import MaskedPrefixOracle


class Backend(FreshBackend):
    def __init__(self):
        super().__init__()
        from scripts.spec.triton import pi0_infer as native
        from preparations.idea1_v3.safe_qk import bounded_matmul_abT_scale
        self.prefix_packer=PrefixPacker();self.prefix_visits=PrefixVisits(self.ae_dispatch.namespace)
        self.masked_prefix_oracle=MaskedPrefixOracle(native,bounded_matmul_abT_scale)
        self.protocol.update(prefix_attention_numerical_gate_passed=False,
            prefix_scope='Fullfreshvision27+27andVLM18prefill;FastAE18D2retains128imagekeyspercameraandalltext.OriginaldecoderabsoluteRoPE.Packingispartofpolicycost.')

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=(),prefix_pattern='all'):
        if ae_skip:raise ValueError('Prefix preparation retains AE18')
        if context.get('camera_execution_audit',{}).get('kind')!='native':raise ValueError('Need observed fresh full-vision context')
        result,ae,prefix=sample_with_prefix(self.adapter,self.prefix_visits,self.prefix_packer,context,seed,steps,prefix_pattern)
        actions,raw,_,nodes,ms,decode=result
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=nodes['nodes'],actual_NFE=nodes['actual_NFE'],
            ae_execution_audit=ae,camera_execution_audit=context['camera_execution_audit'],prefix_execution_audit=prefix,
            prefix_pack_wall_ms=prefix['packing_wall_ms'],prefix_pack_cuda_ms=prefix['packing_cuda_ms'],prefix_restore_wall_ms=prefix['restore_wall_ms'])

    @torch.no_grad()
    def sample_masked_reference(self,context,seed,pattern):
        if pattern not in ('identity','center_columns','checkerboard'):raise ValueError('Unplanned numerical reference')
        with self.masked_prefix_oracle.use(context['seq'],pattern) as audit:
            result=super().sample_native_reference(context,seed,2)
        if not audit['original_helper_restored'] or len(audit['calls'])!=36:raise ValueError('Masked reference lacks all AE18D2 attention calls')
        return (*result,audit)
