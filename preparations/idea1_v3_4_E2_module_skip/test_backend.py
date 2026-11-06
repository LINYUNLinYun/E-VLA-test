"""CPU route checks; numerical Triton checks are run by the GPU harness."""
import unittest

import torch

from .backend import encoder, module_identity


class MockKernels:
    """Small state-dependent operators expose dispatch and injection order."""
    def __init__(self):
        self.calls = []

    def layer_norm_matmul_n256_1152_2048_bias(self, vision, *args):
        x = args[-2]
        x.copy_(vision)

    def rms_matmul_n_2048_2560_qkv_rope(self, x, w, rope, q, key, value, norm):
        self.calls.append(("qkv", int(w) + 1))
        q.copy_(x)
        key.copy_(x.float() + float(w) / 32)
        value.copy_(x.float() - float(w) / 32)

    def rms_matmul_n_2048_2560_kv_rope(self, x, w, rope, key, value, norm):
        self.calls.append(("kv", int(w) + 1))
        key.copy_(x.float() + float(w) / 32)
        value.copy_(x.float() - float(w) / 32)

    def AttnSingleKey(self, q, key, value, scale):
        return (q.float() + key.float()) / 32

    def matmul_n_2048_2048_res(self, attn, w, x):
        self.calls.append(("attn", int(w) + 1))
        x.copy_(x.float() + attn)

    def rms_matmul_n_2048_16384_gate(self, x, gate, up, hidden, norm):
        self.calls.append(("ffn_gate", int(gate) + 1))
        hidden.copy_(x.float() / 16 + float(gate) / 64)

    def matmul_n_16384_2048_res(self, hidden, w, x):
        self.calls.append(("ffn_down", int(w) + 1))
        x.copy_(x.float() + hidden.float())


def buffers():
    initial = torch.tensor([[.25, -.5, .75, 1.], [1., .5, -.5, -.75]], dtype=torch.bfloat16)
    scalar = torch.arange(18)
    w = dict(vision_final_norm_w=None, vision_final_norm_b=None,
             encoder_multi_modal_projector_w=None, encoder_multi_modal_projector_b=None,
             encoder_attn_qkv_w=scalar, encoder_attn_o_w=scalar,
             encoder_ffn_gate_w=scalar, encoder_ffn_up_w=scalar, encoder_ffn_down_w=scalar)
    b = dict(vision_x=initial.clone(), encoder_x=torch.zeros_like(initial),
             vision_x_norm=None, encoder_rope_weights=None,
             encoder_x_norm=torch.empty_like(initial), encoder_Q=torch.empty_like(initial),
             encoder_K=torch.empty((18, *initial.shape), dtype=torch.bfloat16),
             encoder_V=torch.empty((18, *initial.shape), dtype=torch.bfloat16),
             encoder_hidden=torch.empty_like(initial))
    return w, b


def run(**kwargs):
    k = MockKernels()
    w, b = buffers()
    residuals, traces, meta = encoder(k, w, b, 2, **kwargs)
    return k, b, residuals, traces, meta


class NativeModuleRoutingTests(unittest.TestCase):
    def test_capture_leaves_full_unchanged(self):
        _, plain, _, _, _ = run()
        _, captured, residuals, traces, meta = run(capture=True, traced=(1, 8, 17))
        for key in ("encoder_x", "encoder_K", "encoder_V"):
            self.assertTrue(torch.equal(plain[key], captured[key]), key)
        self.assertEqual(meta["captured"], {kind: list(range(1, 18)) for kind in ("attn", "ffn")})
        for kind in residuals:
            self.assertEqual(len(residuals[kind]), 17)
            self.assertTrue(all(value.dtype == torch.float32 for value in residuals[kind].values()))
        for layer, trace in traces.items():
            self.assertTrue(torch.equal(residuals["attn"][layer], trace["after_attn"].float() - trace["input"].float()))
            self.assertTrue(torch.equal(residuals["ffn"][layer], trace["output"].float() - trace["after_attn"].float()))

    def test_native_dispatch_bypasses_each_skipped_operator(self):
        k, _, _, _, meta = run(skip_attn=(2, 5), skip_ffn=(3, 5))
        self.assertNotIn(("attn", 2), k.calls)
        self.assertNotIn(("qkv", 2), k.calls)
        self.assertIn(("kv", 2), k.calls)
        self.assertNotIn(("ffn_gate", 3), k.calls)
        self.assertNotIn(("ffn_down", 3), k.calls)
        self.assertIn(("attn", 3), k.calls)
        self.assertNotIn(("attn", 5), k.calls)
        self.assertNotIn(("ffn_gate", 5), k.calls)
        self.assertIn(("qkv", 18), k.calls)
        self.assertEqual(len(meta["routes"]), 18)

    def test_true_module_residual_injection_restores_full_and_current_kv(self):
        _, full, residuals, _, _ = run(capture=True)
        for attn, ffn in (((2, 5), ()), ((), (2, 5)), ((2, 5), (2, 5)), (tuple(range(1, 18)), tuple(range(1, 18)))):
            _, replaced, _, _, _ = run(skip_attn=attn, skip_ffn=ffn, history=residuals, reference=True)
            for key in ("encoder_x", "encoder_K", "encoder_V"):
                self.assertTrue(torch.equal(full[key], replaced[key]), (attn, ffn, key))

    def test_attention_compensation_precedes_executed_ffn(self):
        _, _, residuals, full_traces, _ = run(capture=True, traced=(1,))
        _, _, _, patched_traces, _ = run(skip_attn=(1,), history=residuals, traced=(1,))
        self.assertTrue(torch.equal(full_traces[1]["after_attn"], patched_traces[1]["after_attn"]))
        self.assertTrue(torch.equal(full_traces[1]["output"], patched_traces[1]["output"]))

    def test_first_bank_hold_and_extrapolation(self):
        _, _, residuals, _, _ = run(capture="ffn")
        zero = {"ffn": {layer: torch.zeros_like(value) for layer, value in residuals["ffn"].items()}}
        _, _, _, held, _ = run(skip_ffn=(1,), history=residuals, ratio=.8, traced=(1,))
        _, _, _, first, _ = run(skip_ffn=(1,), history=residuals, ratio=0., traced=(1,))
        self.assertTrue(torch.equal(held[1]["output"], first[1]["output"]))
        _, _, _, extra, _ = run(skip_ffn=(1,), history=residuals, older=zero, ratio=.4, traced=(1,))
        estimate = residuals["ffn"][1] + .4 * residuals["ffn"][1]
        expected = (extra[1]["after_attn"].float() + estimate).to(torch.bfloat16)
        self.assertTrue(torch.equal(expected, extra[1]["output"]))

    def test_partial_capture_does_not_materialize_skipped_module(self):
        k, _, residuals, _, meta = run(skip_ffn=(1,), capture="ffn")
        self.assertEqual(set(residuals), {"ffn"})
        self.assertNotIn(1, residuals["ffn"])
        self.assertNotIn(("ffn_gate", 1), k.calls)
        self.assertEqual(len(meta["captured"]["ffn"]), 16)

    def test_invalid_history_and_l18_rejected(self):
        with self.assertRaises(ValueError):
            run(skip_ffn=(18,))
        with self.assertRaises(ValueError):
            run(skip_attn=(1,), history={"ffn": {}})
        with self.assertRaises(ValueError):
            run(skip_ffn=(1,), history={"ffn": {1: torch.zeros(1)}})
        with self.assertRaises(ValueError):
            run(older={"ffn": {}})
        identity = module_identity("prompt-a", 2, (2, 4))
        with self.assertRaises(ValueError):
            run(history={"residuals": {"ffn": {}}, "identity": identity},
                identity=module_identity("prompt-b", 2, (2, 4)))


if __name__ == "__main__":
    unittest.main()
