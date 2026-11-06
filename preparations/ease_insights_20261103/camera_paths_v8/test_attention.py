"""CPU fake kernels test edge consumption and restoration, not Triton parity."""
from types import SimpleNamespace
import unittest
import torch
from reports.ease_insights_20261103.prefix_kv_preparation_v4.test_oracle import fake,Kernel
from .attention import RoleAttention
from .oracle import RoleReference


def setup():
    torch.set_num_threads(1);native=fake();mask=RoleAttention(native,native.matmul_abT_scale);g=torch.Generator().manual_seed(86)
    def soft(inp,m,n,prefix,out,keep,**kw):
        ids=torch.arange(n);retained=torch.ones(n,dtype=torch.bool);retained[:512]=keep
        x=inp.float().clone();x[:,~retained]=float('-inf');x[:8,prefix+1:]=float('-inf');out.copy_(torch.softmax(x,-1))
    mask.kernel=Kernel(soft);prefix=519
    buffers={'decoder_rope_weights':torch.empty((51,256)),'encoder_Q':torch.randn((prefix*8,256),generator=g)}
    context=dict(seq=prefix,runtime=SimpleNamespace(buffers=buffers),K=torch.randn((18,prefix,256),generator=g))
    q=torch.randn((408,256),generator=g);k=torch.randn((prefix+51,256),generator=g);out=torch.empty((408,prefix+51))
    return native,mask,context,q,k,out

class RoleAttentionTest(unittest.TestCase):
    def test_all_identity_is_bitwise_native(self):
        n,m,c,q,k,out=setup();ref=torch.empty_like(out);n.matmul_k8_256_n_softmax_mask0(q,k,ref,519)
        with m.use(c,'all') as audit:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
        self.assertTrue(torch.equal(out,ref));self.assertEqual(audit['selector_calls'],0)
        self.assertTrue(audit['original_helper_restored']);self.assertTrue(audit['original_buffer_references_preserved'])
    def test_both_roles_equal_independent_original_index_reference(self):
        for mode in ('language_query','state_query'):
            n,m,c,q,k,out=setup();ref=torch.empty_like(out)
            with m.use(c,mode) as audit:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
            with RoleReference(n,n.matmul_abT_scale).use(519,audit['selection']['kept_visual_indices']) as proof:
                n.matmul_k8_256_n_softmax_mask0(q,k,ref,519)
            self.assertTrue(torch.equal(out,ref));self.assertEqual(audit['selector_calls'],1)
            self.assertEqual(audit['actual_QK_shapes'],[[408,570,256]]);self.assertEqual(audit['original_decoder_absolute_positions'],list(range(519,570)))
            self.assertTrue(proof['reference_only']);self.assertEqual(len(audit['kept_prefix_indices']),263)
    def test_state_selection_once_and_new_request_reselects(self):
        n,m,c,q,k,out=setup()
        with m.use(c,'state_query') as first:
            n.matmul_k8_256_n_softmax_mask0(q,k,out,519);kept=list(first['selection']['kept_visual_indices'])
            q.neg_();n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
            self.assertEqual(first['selection']['kept_visual_indices'],kept)
        self.assertEqual(first['selector_calls'],1);self.assertEqual(first['actual_QK_calls'],2)
        with m.use(c,'state_query') as second:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
        self.assertEqual(second['selector_calls'],1);self.assertNotEqual(second['selection']['kept_visual_indices'],kept)
    def test_selector_failure_and_consumer_exception_unlock_without_overwrite(self):
        n,m,c,*_=setup();m.language_selector=lambda c:(_ for _ in ()).throw(RuntimeError('selector'))
        with self.assertRaises(RuntimeError):
            with m.use(c,'language_query'):pass
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,m.original);self.assertFalse(m.lock.locked())
        with self.assertRaises(RuntimeError):
            with m.use(c,'state_query'):raise RuntimeError('consumer')
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,m.original);self.assertFalse(m.lock.locked())
    def test_nested_prepatched_and_compact_helpers_rejected(self):
        n,m,c,q,k,out=setup()
        with m.use(c,'all'):
            with self.assertRaises(ValueError):
                with m.use(c,'state_query'):pass
        with self.assertRaises(ValueError):
            with m.use(c,'state_query'):n.matmul_k8_256_n_softmax_mask0(q,k[:314],out[:,:314].contiguous(),263)
        replacement=object();n.matmul_k8_256_n_softmax_mask0=replacement
        with self.assertRaises(ValueError):
            with m.use(c,'all'):pass
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,replacement)

if __name__=='__main__':unittest.main()
