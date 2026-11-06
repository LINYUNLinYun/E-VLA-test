"""CPU fake-native checks;real fused Triton numerical parity remains mandatory."""
from types import SimpleNamespace
import unittest
import torch
from reports.ease_insights_20261103.prefix_kv_preparation_v4.test_oracle import fake,Kernel
from reports.ease_insights_20261103.prefix_kv_preparation_v4.oracle import MaskedPrefixOracle
from .attention import MaskedAttention


def setup():
    native=fake();mask=MaskedAttention(native,native.matmul_abT_scale)
    def soft(inp,m,n,prefix,out,pattern,**kw):
        ids=torch.arange(n);column=ids%16;row=ids//16%16
        keep=(ids>=512)|(((column>=4)&(column<12)) if pattern==0 else (row+column)%2==0)
        values=inp.float().clone();values[:,~keep]=float('-inf');values[:8,prefix+1:]=float('-inf');out.copy_(torch.softmax(values,-1))
    mask.kernel=Kernel(soft);buffers={'decoder_rope_weights':torch.empty((51,256))}
    ctx=dict(seq=519,runtime=SimpleNamespace(buffers=buffers));g=torch.Generator().manual_seed(84)
    q=torch.randn((408,256),generator=g);k=torch.randn((570,256),generator=g);out=torch.empty((408,570))
    return native,mask,ctx,q,k,out


class AttentionTest(unittest.TestCase):
    def test_all_edge_identity_is_bitwise_original(self):
        n,m,c,q,k,out=setup();ref=torch.empty_like(out);n.matmul_k8_256_n_softmax_mask0(q,k,ref,519)
        with m.use(c,'all') as audit:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
        self.assertTrue(torch.equal(ref,out));self.assertEqual(audit['actual_QK_shapes'],[[408,570,256]])
        self.assertFalse(audit['prefix_compute_reduction_claimed']);self.assertTrue(audit['original_buffer_references_preserved'])
    def test_two_full_storage_masks_are_bitwise_independent_index_mask_reference(self):
        for pattern in ('center_columns','checkerboard'):
            n,m,c,q,k,out=setup();oracle=MaskedPrefixOracle(n,n.matmul_abT_scale);ref=torch.empty_like(out)
            with m.use(c,pattern) as audit:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
            with oracle.use(519,pattern):n.matmul_k8_256_n_softmax_mask0(q,k,ref,519)
            self.assertTrue(torch.equal(out,ref));self.assertEqual(audit['visual_keys_per_camera'],[128,128])
            self.assertEqual(audit['storage_prefix'],519);self.assertEqual(audit['effective_visible_prefix_keys'],263)
    def test_consumer_exception_restores_and_unlocks(self):
        n,m,c,*_=setup()
        with self.assertRaises(RuntimeError):
            with m.use(c,'center_columns'):raise RuntimeError('consumer')
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,m.original);self.assertFalse(m.lock.locked())
    def test_prepatched_and_nested_helpers_are_preserved(self):
        n,m,c,*_=setup()
        with m.use(c,'all'):
            wrapper=n.matmul_k8_256_n_softmax_mask0
            with self.assertRaises(ValueError):
                with m.use(c,'checkerboard'):pass
            self.assertIs(n.matmul_k8_256_n_softmax_mask0,wrapper)
        replacement=object();n.matmul_k8_256_n_softmax_mask0=replacement
        with self.assertRaises(ValueError):
            with m.use(c,'all'):pass
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,replacement)
    def test_compact_storage_cannot_be_mislabeled_as_full_work(self):
        n,m,c,q,k,out=setup()
        with self.assertRaises(ValueError):
            with m.use(c,'center_columns'):n.matmul_k8_256_n_softmax_mask0(q,k[:314],out[:,:314].contiguous(),263)
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,m.original)
    def test_buffer_or_helper_ownership_violation_is_reported(self):
        for change in ('buffer','helper'):
            n,m,c,*_=setup()
            with self.assertRaises((ValueError,RuntimeError)):
                with m.use(c,'all'):
                    if change=='buffer':c['runtime'].buffers['decoder_rope_weights']=torch.empty(1)
                    else:n.matmul_k8_256_n_softmax_mask0=object()
            self.assertIs(n.matmul_k8_256_n_softmax_mask0,m.original);self.assertFalse(m.lock.locked())


if __name__=='__main__':unittest.main()
