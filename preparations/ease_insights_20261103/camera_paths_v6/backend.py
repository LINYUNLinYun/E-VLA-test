"""Full fresh native vision/VLM;FastD2 masks attention edges without compaction."""
import torch
from preparations.ease_insights_20261103.camera_paths_v4.backend import Backend as FreshBackend
from reports.ease_insights_20261103.prefix_kv_preparation_v4.oracle import MaskedPrefixOracle
from .attention import MaskedAttention


class Backend(FreshBackend):
    def __init__(self):
        super().__init__()
        from scripts.spec.triton import pi0_infer as native
        from preparations.idea1_v3.safe_qk import bounded_matmul_abT_scale
        self.attention_mask=MaskedAttention(native,bounded_matmul_abT_scale)
        self.mask_reference=MaskedPrefixOracle(native,bounded_matmul_abT_scale)
        self.protocol.update(prefix_attention_numerical_gate_passed=False,prefix_compute_reduction_claimed=False,
            prefix_scope='OriginalfullQK/Vstorageandwork,bothfreshvision27+27,VLM18/AE18.FastD2masksimageedgesonly;alltextandoriginalRoPEretained.')

    @torch.no_grad()
    def sample(self,context,seed,steps,ae_skip=(),prefix_pattern='all'):
        if ae_skip or steps not in (2,10) or (steps==10 and prefix_pattern!='all'):raise ValueError('Unsupported mask/AE/native schedule')
        if context.get('camera_execution_audit',{}).get('kind')!='native':raise ValueError('Need full fresh native vision')
        with self.attention_mask.use(context,prefix_pattern) as audit:
            actions,raw,cost=super().sample(context,seed,steps,ae_skip=())
        if audit['actual_QK_calls']!=18*steps or not audit['original_helper_restored'] or not audit['original_buffer_references_preserved']:
            raise ValueError('Actual full-storage attention calls or restoration differ')
        return actions,raw,dict(cost,prefix_execution_audit=audit)

    @torch.no_grad()
    def sample_masked_reference(self,context,seed,pattern):
        with self.mask_reference.use(context['seq'],pattern) as audit:
            result=super().sample_native_reference(context,seed,2)
        if not audit['original_helper_restored'] or len(audit['calls'])!=36:raise ValueError('Incomplete independent mask reference')
        return (*result,audit)
