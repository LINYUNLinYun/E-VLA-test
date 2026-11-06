"""CPU checks of protocol invariants that could invalidate paired experiments."""
import copy
import unittest
import numpy as np
from .common import MASKS, conditions, cases
from .policy import Policy


class Raw:
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.zeros((50, 32), np.float32)


class FakeBackend:
    def __init__(self): self.calls = []; self.steps = []
    def prepare_residual(self, obs, instruction, **kwargs):
        self.calls.append(dict(obs=obs, **kwargs))
        return dict(residuals={i: obs for i in kwargs.get("capture_layers", ())}, execution_audit={}), {}
    def sample(self, context, seed, steps):
        self.steps.append(steps)
        return np.zeros((50, 7), np.float32), Raw(), {}


class ProtocolTests(unittest.TestCase):
    def test_cost_probe_excludes_warmup_and_keeps_paired_rounds(self):
        from .benchmark import measure_condition, summarize
        record = dict(obs=0, instruction="prompt", suite="suite", seed=123)
        blocks = [measure_condition(FakeBackend(), cfg, record) for cfg in conditions()]
        for block in blocks:
            self.assertEqual(len(block["warmup"]), 6)
            self.assertEqual([r["round"] for r in block["measured"]], list(range(6, 18)))
            self.assertEqual(block["summary"]["overall"]["n"], 12)
            self.assertEqual(block["summary"]["three_request_cycle"]["n"], 4)
        self.assertEqual(blocks[0]["summary"]["full"]["n"], 12)
        for block in blocks[1:]:
            self.assertEqual(block["summary"]["full"]["n"], 4)
            self.assertEqual(block["summary"]["fast"]["n"], 8)
            self.assertEqual([r["model_noise_sha256"] for r in block["measured"]],
                             [r["model_noise_sha256"] for r in blocks[0]["measured"]])
            self.assertEqual(block["measured"][0]["history_steps"], [15, 30])
        synthetic = [dict(policy_wall_ms=x, refresh=(i % 3 == 0)) for i,x in enumerate([9,3,3]*4)]
        summary = summarize(synthetic)
        self.assertEqual(summary["overall"]["mean_ms"], 5)
        self.assertEqual(summary["full"]["mean_ms"], 9)
        self.assertEqual(summary["fast"]["mean_ms"], 3)
        self.assertEqual(summary["three_request_cycle"]["mean_ms"], 15)

    def test_each_mask_owns_capture_execution_and_history(self):
        hashes = []
        for cfg in conditions()[1:]:
            backend = FakeBackend(); policy = Policy(backend); policy.configure(cfg)
            rows = []
            for i in range(6):
                row = policy.predict(i, "prompt", "suite", 100+i); rows.append(row)
                policy.acknowledge(row["actions"][:5])
            self.assertEqual(backend.steps, [10, 2, 2, 10, 2, 2])
            mask = tuple(cfg["skip"])
            self.assertEqual(backend.calls[0]["capture_layers"], mask)
            self.assertEqual(backend.calls[3]["capture_layers"], mask)
            self.assertEqual([backend.calls[i]["skip"] for i in (1, 2, 4, 5)], [mask]*4)
            self.assertEqual(backend.calls[4]["history"], dict.fromkeys(mask, 3))
            self.assertEqual(backend.calls[4]["older"], dict.fromkeys(mask, 0))
            self.assertEqual([backend.calls[i]["ratio"] for i in (4,5)], [1/3, 2/3])
            self.assertEqual([r["step"] for r in rows], [0,5,10,15,20,25])
            hashes.append([r["model_noise_sha256"] for r in rows])
            policy.configure(cfg)
            self.assertEqual(policy.history, []); self.assertEqual(policy.step, 0)
        self.assertEqual(hashes, [hashes[0]]*3)

    def test_full_and_execution_ack(self):
        backend = FakeBackend(); policy = Policy(backend); policy.configure(conditions()[0])
        for index in range(4):
            row = policy.predict(index, "p", "s", 0)
            with self.assertRaises(RuntimeError): policy.predict(index, "p", "s", 0)
            with self.assertRaises(ValueError): policy.acknowledge(np.ones((5,7), np.float32))
            policy.acknowledge(row["actions"][:2])
        self.assertEqual(backend.steps, [10]*4); self.assertEqual(policy.step, 8)
        self.assertEqual(policy.history, [])

    def test_no_mutable_scientific_condition_or_validation(self):
        for key, value in (("skip", [1,2,3,4,5,6]), ("D", 3), ("period", 3), ("R", 10)):
            cfg = copy.deepcopy(conditions()[1]); cfg[key] = value
            with self.assertRaises(ValueError): Policy(FakeBackend()).configure(cfg)
        manifest = dict(suites=["a", "b", "c", "d"], tasks=list(range(10)), discovery_inits=[30])
        self.assertEqual(len(cases(manifest, "discovery")), 40)
        shards = [cases(manifest, "discovery", i, 3) for i in range(3)]
        self.assertEqual(sorted(str(x) for shard in shards for x in shard), sorted(str(x) for x in cases(manifest, "discovery")))
        with self.assertRaises(ValueError): cases(manifest, "confirm")

    def test_encoder_dispatch_uses_each_mask(self):
        import torch
        from .backend import encoder
        class Kernels:
            def __init__(self): self.qkv=[]; self.kv=[]; self.ffn=[]
            def layer_norm_matmul_n256_1152_2048_bias(self, *args): pass
            def rms_matmul_n_2048_2560_qkv_rope(self, x, w, *args): self.qkv.append(w)
            def rms_matmul_n_2048_2560_kv_rope(self, x, w, *args): self.kv.append(w)
            def AttnSingleKey(self, *args): return None
            def matmul_n_2048_2048_res(self, *args): pass
            def rms_matmul_n_2048_16384_gate(self, x, w, *args): self.ffn.append(w)
            def matmul_n_16384_2048_res(self, *args): pass
        weights = dict.fromkeys(("vision_final_norm_w", "vision_final_norm_b", "encoder_multi_modal_projector_w", "encoder_multi_modal_projector_b"))
        for name in ("encoder_attn_qkv_w", "encoder_attn_o_w", "encoder_ffn_gate_w", "encoder_ffn_up_w", "encoder_ffn_down_w"):
            weights[name] = list(range(1, 19))
        buffers = dict.fromkeys(("vision_x", "encoder_x", "vision_x_norm", "encoder_rope_weights", "encoder_Q", "encoder_x_norm", "encoder_hidden"))
        buffers.update(encoder_K=np.zeros((18,1)), encoder_V=np.zeros((18,1)))
        with torch.no_grad():
            for mask in MASKS.values():
                kernels = Kernels()
                _, _, audit = encoder(kernels, weights, buffers, 1, skip=mask)
                self.assertEqual(kernels.kv, list(mask))
                self.assertEqual(kernels.qkv, [i for i in range(1,19) if i not in mask])
                self.assertEqual(kernels.ffn, [i for i in range(1,18) if i not in mask])
                self.assertEqual(audit["hidden_update_layers"], kernels.ffn)
                self.assertFalse(audit["grad_enabled"])


if __name__ == "__main__": unittest.main()
