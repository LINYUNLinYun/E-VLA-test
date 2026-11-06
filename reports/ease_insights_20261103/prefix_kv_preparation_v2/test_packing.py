"""CPU-only contracts for token identity and exception-safe runtime ownership."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from .packing import PrefixPacker

torch.set_num_threads(1)


def fixture(logits=False):
    prefix=519
    k=torch.arange(18*prefix*256,dtype=torch.float32).reshape(18,prefix,256)
    v=-k
    buffers=dict(encoder_K=torch.empty(18,prefix+51,256),encoder_V=torch.empty(18,prefix+51,256),
        decoder_attn_buf=torch.empty(408,prefix+51),decoder_rope_weights=torch.arange(51*256).reshape(51,256),
        encoder_x=torch.empty(prefix,2048))
    if logits:buffers['decoder_attn_logits_buf']=torch.empty(408,prefix+51)
    return dict(runtime=SimpleNamespace(buffers=buffers),K=k,V=v,seq=prefix,state=torch.zeros(1,32))


class Packing(unittest.TestCase):
    def check(self,pattern,logits=False):
        context=fixture(logits);original=dict(context['runtime'].buffers);packer=PrefixPacker()
        with packer.packed(context,pattern) as (packed,audit):
            self.assertIs(packed['runtime'],context['runtime']);self.assertIs(packed['state'],context['state'])
            self.assertIs(packed['runtime'].buffers['decoder_rope_weights'],original['decoder_rope_weights'])
            self.assertIs(packed['runtime'].buffers['encoder_x'],original['encoder_x'])
            indices=audit['kept_prefix_indices'];self.assertEqual(indices,sorted(set(indices)))
            self.assertTrue(torch.equal(packed['K'],context['K'][:,indices]));self.assertTrue(torch.equal(packed['V'],context['V'][:,indices]))
            self.assertEqual(indices[-7:],list(range(512,519)));self.assertEqual(audit['decoder_absolute_positions'],list(range(519,570)))
            self.assertEqual(audit['decoder_storage_positions'],list(range(packed['seq'],packed['seq']+51)))
            self.assertTrue(packed['runtime'].buffers['encoder_K'].is_contiguous())
            self.assertEqual(tuple(packed['runtime'].buffers['encoder_K'].shape),(18,packed['seq']+51,256))
            if logits:self.assertEqual(tuple(packed['runtime'].buffers['decoder_attn_logits_buf'].shape),(408,packed['seq']+51))
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()));self.assertFalse(packer.active)
        return audit

    def test_identity_packing_preserves_all_values_and_rope(self):
        audit=self.check('identity');self.assertEqual(audit['visual_tokens_per_camera'],[256,256])

    def test_center_columns_exact_camera_and_text_identity(self):
        audit=self.check('center_columns');self.assertEqual(audit['visual_tokens_per_camera'],[128,128])
        for i in audit['kept_prefix_indices'][:256]:self.assertTrue(4<=i%16<12)

    def test_checkerboard_with_optional_logits_storage(self):
        audit=self.check('checkerboard',logits=True);self.assertEqual(audit['visual_tokens_per_camera'],[128,128])
        for i in audit['kept_prefix_indices'][:256]:self.assertEqual((i//16+i%16)%2,0)

    def test_consumer_error_restores_original_buffers(self):
        context=fixture(True);original=dict(context['runtime'].buffers);packer=PrefixPacker()
        with self.assertRaisesRegex(RuntimeError,'consumer failed'):
            with packer.packed(context,'center_columns'):
                raise RuntimeError('consumer failed')
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()));self.assertFalse(packer.active)

    def test_nested_packing_rejected_and_outer_transaction_restored(self):
        context=fixture();original=dict(context['runtime'].buffers);packer=PrefixPacker()
        with self.assertRaises(ValueError):
            with packer.packed(context,'center_columns'):
                with packer.packed(context,'checkerboard'):pass
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()))

    def test_ownership_violation_restores_then_raises(self):
        context=fixture();original=dict(context['runtime'].buffers);packer=PrefixPacker()
        with self.assertRaisesRegex(RuntimeError,'ownership'):
            with packer.packed(context,'checkerboard'):
                context['runtime'].buffers['encoder_K']=torch.empty(1)
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()));self.assertFalse(packer.active)

    def test_runtime_dictionary_replacement_restored(self):
        context=fixture();original=context['runtime'].buffers;packer=PrefixPacker()
        with self.assertRaisesRegex(RuntimeError,'ownership'):
            with packer.packed(context,'checkerboard'):context['runtime'].buffers={}
        self.assertIs(context['runtime'].buffers,original)

    def test_invalid_full_prefix_shape_fails_before_mutation(self):
        context=fixture();context['K']=context['K'][:,:-1];original=dict(context['runtime'].buffers)
        with self.assertRaises(ValueError):
            with PrefixPacker().packed(context,'checkerboard'):pass
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()))

    def test_no_cache_clear_while_borrowed_and_scratch_reused_after_release(self):
        context=fixture();packer=PrefixPacker()
        with packer.packed(context,'center_columns'):
            first=context['runtime'].buffers['encoder_K']
            with self.assertRaises(ValueError):packer.clear()
        with packer.packed(context,'center_columns'):self.assertIs(first,context['runtime'].buffers['encoder_K'])
        packer.clear();self.assertFalse(packer.cache)


    def test_partial_allocation_failure_never_commits_cache_or_buffers(self):
        context=fixture(True);original=dict(context['runtime'].buffers);packer=PrefixPacker();allocate=torch.empty;calls=[]
        def limited(*args,**kwargs):
            calls.append(1)
            if len(calls)==4:raise RuntimeError('allocation failed')
            return allocate(*args,**kwargs)
        with patch('reports.ease_insights_20261103.prefix_kv_preparation_v2.packing.torch.empty',side_effect=limited):
            with self.assertRaisesRegex(RuntimeError,'allocation failed'):
                with packer.packed(context,'checkerboard'):pass
        self.assertFalse(packer.cache);self.assertFalse(packer.active)
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()))

    def test_stale_cache_dtype_rejected_before_mutation(self):
        context=fixture();packer=PrefixPacker()
        with packer.packed(context,'center_columns'):pass
        for key in ('encoder_K','encoder_V','decoder_attn_buf'):
            context['runtime'].buffers[key]=context['runtime'].buffers[key].double()
        context['K']=context['K'].double();context['V']=context['V'].double();original=dict(context['runtime'].buffers)
        with self.assertRaisesRegex(ValueError,'device/dtype'):
            with packer.packed(context,'center_columns'):pass
        self.assertTrue(all(context['runtime'].buffers[k] is v for k,v in original.items()))


if __name__=='__main__':unittest.main()
