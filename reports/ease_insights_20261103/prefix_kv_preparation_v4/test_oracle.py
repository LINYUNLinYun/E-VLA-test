"""CPU mathematical/mutation checks;does not establish GPU numerical parity."""
from types import SimpleNamespace
import unittest
import torch
from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan
from reports.ease_insights_20261103.prefix_kv_preparation_v4.oracle import MaskedPrefixOracle
torch.set_num_threads(1)


class Kernel:
    def __init__(self,fn):self.fn=fn
    def __getitem__(self,grid):return self.fn


def fake():
    def qk(q,k,out,m,n,d,scale,**kw):out.copy_(((q.float()@k.float().T)*scale).to(out.dtype))
    def soft(inp,m,n,heads,prefix,out,**kw):
        x=inp.float().clone();x[:heads,prefix+1:]=float('-inf');out.copy_(torch.softmax(x,-1))
    native=SimpleNamespace(matmul_abT_scale=Kernel(qk),softmax_kernel_mask0=Kernel(soft))
    def attention(q,k,out,prefix,logits_out=None):
        logits=out if logits_out is None else logits_out
        qk(q,k,logits,q.shape[0],k.shape[0],256,256**-.5);soft(logits,q.shape[0],k.shape[0],8,prefix,out)
    native.matmul_k8_256_n_softmax_mask0=attention;return native


class OracleTest(unittest.TestCase):
    def tensors(self,prefix=519):
        g=torch.Generator().manual_seed(113)
        return torch.randn((408,256),generator=g),torch.randn((prefix+51,256),generator=g),torch.empty((408,prefix+51))
    def test_identity_is_bitwise_native_and_restores_helper(self):
        n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale);q,k,out=self.tensors();ref=torch.empty_like(out);n.matmul_k8_256_n_softmax_mask0(q,k,ref,519)
        with o.use(519,'identity') as audit:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
        self.assertTrue(torch.equal(out,ref));self.assertIs(n.matmul_k8_256_n_softmax_mask0,o.original);self.assertTrue(audit['original_helper_restored'])
    def test_both_masks_equal_independent_compact_attention(self):
        for pattern in ('center_columns','checkerboard'):
            n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale);q,k,out=self.tensors();layout=plan(pattern,519)
            positions=list(layout.keep_indices)+list(range(519,570));index=torch.tensor(positions)
            logits=(q.double()@k[index].double().T)*(256**-.5);logits[:8,layout.packed_prefix+1:]=float('-inf');ref=torch.softmax(logits,-1)
            with o.use(519,pattern) as audit:n.matmul_k8_256_n_softmax_mask0(q,k,out,519)
            self.assertTrue(torch.allclose(out[:,index].double(),ref,atol=1e-6,rtol=1e-5));self.assertEqual(int(torch.count_nonzero(out[:,audit['dropped_prefix_indices']])),0)
            self.assertEqual(audit['calls'][0]['actual_QK_key_tokens'],570);self.assertEqual(audit['calls'][0]['masked_original_prefix_tokens'],256)
            self.assertEqual(int(torch.count_nonzero(out[:8,520:])),0);self.assertGreater(float(out[8:,520:].sum()),0)
    def test_bfloat16_and_explicit_float_logits_are_supported(self):
        for explicit in (False,True):
            n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale);q,k,out=self.tensors();out=out.to(torch.bfloat16);logits=torch.empty_like(out,dtype=torch.float32) if explicit else None
            with o.use(519,'center_columns'):n.matmul_k8_256_n_softmax_mask0(q,k,out,519,logits)
            self.assertTrue(torch.isfinite(out).all())
    def test_sampling_exception_restores_helper_and_unlocks(self):
        n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale)
        with self.assertRaisesRegex(RuntimeError,'consumer'):
            with o.use(519,'checkerboard'):raise RuntimeError('consumer')
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,o.original)
        with o.use(519,'identity'):pass
    def test_prepatched_or_unsafe_kernel_is_preserved(self):
        for field in ('matmul_k8_256_n_softmax_mask0','matmul_abT_scale'):
            n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale);replacement=object();setattr(n,field,replacement)
            with self.assertRaises(ValueError):
                with o.use(519,'identity'):pass
            self.assertIs(getattr(n,field),replacement)
    def test_nested_use_is_rejected_and_outer_helper_survives(self):
        n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale)
        with o.use(519,'identity'):
            wrapper=n.matmul_k8_256_n_softmax_mask0
            with self.assertRaises(ValueError):
                with o.use(519,'center_columns'):pass
            self.assertIs(n.matmul_k8_256_n_softmax_mask0,wrapper)
    def test_ownership_violation_restores_original(self):
        n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale)
        with self.assertRaises(RuntimeError):
            with o.use(519,'identity'):n.matmul_k8_256_n_softmax_mask0=object()
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,o.original)
    def test_compact_storage_cannot_masquerade_as_dense_reference(self):
        n=fake();o=MaskedPrefixOracle(n,n.matmul_abT_scale);q,k,out=self.tensors()
        with self.assertRaises(ValueError):
            with o.use(519,'center_columns'):n.matmul_k8_256_n_softmax_mask0(q,k[:314],out[:,:314].contiguous(),263)
        self.assertIs(n.matmul_k8_256_n_softmax_mask0,o.original)


if __name__=='__main__':unittest.main()
