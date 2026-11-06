"""CPU-only preparation checks; no checkpoint, rollout, or real-action parity.

The Xiaomi arithmetic test extracts definitions from the pinned source AST,
avoiding transformers/flash-attn imports. It is a toy with the actual decoder
arithmetic, not a loaded model. Run: python3 -m unittest discover -s
preparations/idea1_v2_1 -p test_adapters.py
"""
import ast
import __future__
import hashlib
import functools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import adapters

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if torch is not None:
    class ToyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, hidden_states, *, use_cache=False, past_key_values=None,
                    output_attentions=False):
            self.calls += 1
            return hidden_states + 1


class MaskTests(unittest.TestCase):
    def test_strict_mask(self):
        for values in ([True], [0], [4], [2, 2], [1]):
            with self.assertRaises(ValueError):
                adapters._mask(values, 3, (1,))
        self.assertEqual(adapters._mask([3, 2], 3), (2, 3))

    def test_lazy_import(self):
        import subprocess
        import sys
        result = subprocess.run([sys.executable, "-S", "-c",
                                 "import adapters, sys; assert 'torch' not in sys.modules"],
                                cwd=Path(__file__).parent, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipIf(torch is None, "torch unavailable: no package installation attempted")
class GenericTests(unittest.TestCase):
    def setUp(self):
        self.layers = nn.ModuleList([ToyLayer(), ToyLayer(), ToyLayer()]).eval()
        self.contract = adapters.HiddenLayerContract("ToyLayer", adapters.source_sha256(self.layers[0]),
                                                     3, "tensor", (1,))
        self.x = torch.zeros(1, 2, 4)

    def run_layers(self, **kwargs):
        value = self.x
        for layer in self.layers:
            value = layer(value, **kwargs)
        return value

    def test_empty_fast_reference_and_calls(self):
        native = self.run_layers()
        with adapters.hidden_update_bypass(self.layers, [], self.contract):
            torch.testing.assert_close(self.run_layers(), native, rtol=0, atol=0)
        before = self.layers[1].calls
        with adapters.hidden_update_bypass(self.layers, [2], self.contract) as audit:
            fast = self.run_layers()
        self.assertEqual(self.layers[1].calls, before)
        self.assertEqual(audit["fully_eliminated_layers"], [2])
        with adapters.hidden_update_bypass(self.layers, [2], self.contract, reference=True):
            slow = self.run_layers()
        self.assertEqual(self.layers[1].calls, before + 1)
        torch.testing.assert_close(fast, slow, rtol=0, atol=0)
        self.assertTrue(all("forward" not in vars(layer) for layer in self.layers))

    def test_fail_closed_and_restoration(self):
        for kwargs in ({"use_cache": True}, {"past_key_values": object()}, {"output_attentions": True}):
            with self.assertRaises(ValueError):
                with adapters.hidden_update_bypass(self.layers, [2], self.contract):
                    self.run_layers(**kwargs)
            self.assertNotIn("forward", vars(self.layers[1]))
        bad = adapters.HiddenLayerContract("ToyLayer", "0" * 64, 3, "tensor", ())
        with self.assertRaises(ValueError):
            with adapters.hidden_update_bypass(self.layers, [2], bad):
                pass
        with self.assertRaises(ValueError):
            with adapters.xiaomi_hidden_bypass(self.layers, []):
                pass

    def test_probe_groups_and_cleanup(self):
        records = []
        with adapters.layer_probes(self.layers, {"visual": torch.tensor([[True, False]]),
                                                "text": torch.tensor([[False, True]])}, records):
            self.run_layers()
        self.assertEqual(len(records), 6)
        self.assertEqual(records[0]["count"], 1)
        self.assertEqual(records[0]["update_rms"], 1.0)
        self.run_layers()
        self.assertEqual(len(records), 6)
        self.assertTrue(all(not layer._forward_hooks for layer in self.layers))


class FakeCache:
    def __init__(self):
        self.values = {}
        self.updates = []

    def update(self, key, value, index, kwargs):
        self.updates.append(index)
        if index in self.values:
            old_key, old_value = self.values[index]
            key = torch.cat((old_key, key), dim=-2)
            value = torch.cat((old_value, value), dim=-2)
        self.values[index] = (key.clone(), value.clone())
        return key, value


@unittest.skipIf(torch is None, "torch unavailable: no package installation attempted")
class XiaomiArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).parent / "metadata/xiaomi-robotic0/modeling_mibot.py"
        if not source.is_file():
            raise unittest.SkipTest("pinned Xiaomi metadata source not present")
        if hashlib.sha256(source.read_bytes()).hexdigest() != adapters.XIAOMI_SOURCE_SHA256:
            raise AssertionError("Xiaomi metadata source changed; re-audit required")
        names = {"rotate_half", "apply_rotary_pos_emb", "repeat_kv", "eager_attention_forward",
                 "Qwen3VLTextRMSNorm", "Qwen3VLTextAttention", "Qwen3VLTextMLP", "Qwen3VLTextDecoderLayer"}
        tree = ast.parse(source.read_text())
        definitions = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
        namespace = {"torch": torch, "nn": nn, "GradientCheckpointingLayer": nn.Module,
                     "use_kernel_forward_from_hub": lambda name: lambda cls: cls,
                     "ACT2FN": {"silu": torch.nn.functional.silu}}
        exec(compile(definitions, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
        cls.namespace = namespace

    def make_layer(self, index):
        cfg = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1,
                              attention_dropout=0.0, attention_bias=True, rms_norm_eps=1e-6,
                              intermediate_size=12, hidden_act="silu", _attn_implementation="eager")
        return self.namespace["Qwen3VLTextDecoderLayer"](cfg, index).eval()

    def test_actual_decoder_kv_shell_prefill_and_append(self):
        torch.manual_seed(7)
        layer = self.make_layer(20)  # original 1-based L21
        fast_cache, slow_cache = FakeCache(), FakeCache()
        calls = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 0, "mlp": 0}
        handles = []
        for name in calls:
            module = layer.mlp if name == "mlp" else getattr(layer.self_attn, name)
            def hook(module, args, output, name=name):
                calls[name] += 1
            handles.append(module.register_forward_hook(hook))
        try:
            for length in (3, 1):
                hidden = torch.randn(1, length, 8)
                # Nonidentity RoPE ensures rotation is actually compared.
                angle = torch.randn(1, length, 2).repeat(1, 1, 2)
                kwargs = dict(position_embeddings=(angle.cos(), angle.sin()), attention_mask=None,
                              cache_position=torch.arange(length))
                original = layer(hidden, past_key_values=slow_cache, **kwargs)
                self.assertFalse(torch.equal(original, hidden))
                before = dict(calls)
                result = adapters._xiaomi_kv_shell(layer, hidden, past_key_values=fast_cache, **kwargs)
                self.assertIs(result, hidden)
                for name in ("q_proj", "o_proj", "mlp"):
                    self.assertEqual(calls[name], before[name], name)
                for name in ("k_proj", "v_proj"):
                    self.assertEqual(calls[name], before[name] + 1, name)
                for fast, slow in zip(fast_cache.values[20], slow_cache.values[20]):
                    torch.testing.assert_close(fast, slow, rtol=0, atol=0)
            self.assertEqual(fast_cache.updates, [20, 20])
            self.assertEqual(fast_cache.values[20][0].shape[-2], 4)
        finally:
            for handle in handles:
                handle.remove()

    def test_shell_requires_cache(self):
        layer = self.make_layer(35)
        with self.assertRaises(ValueError):
            adapters._xiaomi_kv_shell(layer, torch.zeros(1, 1, 8), (torch.ones(1, 1, 4), torch.zeros(1, 1, 4)))

    def test_36_original_indices_and_fusion_toy(self):
        torch.manual_seed(11)
        layers = [self.make_layer(i) for i in range(36)]
        selected = {4, 20, 21, 36}
        hidden = torch.randn(1, 2, 8)
        fusion = [torch.randn_like(hidden) for _ in range(3)]
        rotary = (torch.ones(1, 2, 4), torch.zeros(1, 2, 4))
        def run(reference):
            cache, value = FakeCache(), hidden.clone()
            for number, layer in enumerate(layers, 1):
                if number not in selected:
                    value = layer(value, rotary, past_key_values=cache)
                elif reference:
                    layer(value, rotary, past_key_values=cache)  # slow, discard update
                else:
                    value = adapters._xiaomi_kv_shell(layer, value, rotary, past_key_values=cache)
                if number <= 3:
                    value = value + fusion[number - 1]
            return torch.nn.functional.layer_norm(value, (8,)), cache
        with torch.no_grad():
            slow, slow_cache = run(True)
            fast, fast_cache = run(False)
        torch.testing.assert_close(slow, fast, rtol=0, atol=0)
        self.assertEqual(fast_cache.updates, list(range(36)))
        self.assertEqual(sorted(fast_cache.values)[-16:], list(range(20, 36)))
        for index in range(36):
            for a, b in zip(slow_cache.values[index], fast_cache.values[index]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@unittest.skipIf(torch is None, "torch unavailable: no package installation attempted")
class DBQwen2ArithmeticTests(unittest.TestCase):
    """Official Qwen2 arithmetic with toy config; no transformers import.

    Uses fixed local metadata or an installed exact-version source. Tests never
    access the network. Hash must match before AST definitions execute.
    """
    @classmethod
    def setUpClass(cls):
        pinned = Path(__file__).parent / "metadata/db-cogact/modeling_qwen2_v4_57_6.py"
        if pinned.is_file() and hashlib.sha256(pinned.read_bytes()).hexdigest() != adapters.DB_QWEN2_SOURCE_SHA256:
            raise AssertionError("pinned Qwen2 metadata changed; re-audit required")
        candidates = [pinned] + [Path(p) / "transformers/models/qwen2/modeling_qwen2.py" for p in sys.path]
        data = next((p.read_bytes() for p in candidates if p.is_file()
                     and hashlib.sha256(p.read_bytes()).hexdigest() == adapters.DB_QWEN2_SOURCE_SHA256), None)
        if data is None:
            raise unittest.SkipTest("pinned local Qwen2 4.57.6 source absent; no network access attempted")
        if hashlib.sha256(data).hexdigest() != adapters.DB_QWEN2_SOURCE_SHA256:
            raise AssertionError("Qwen2 source changed; re-audit required")
        names = {"rotate_half", "apply_rotary_pos_emb", "repeat_kv", "eager_attention_forward",
                 "Qwen2RMSNorm", "Qwen2Attention", "Qwen2MLP", "Qwen2DecoderLayer"}
        tree = ast.parse(data.decode())
        definitions = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names], type_ignores=[])
        def deprecated(*args, **kwargs):
            def decorate(fn):
                @functools.wraps(fn)
                def wrapper(*args, **kwargs):
                    return fn(*args, **kwargs)
                return wrapper
            return decorate
        namespace = {"torch": torch, "nn": nn, "GradientCheckpointingLayer": nn.Module,
                     "deprecate_kwarg": deprecated,
                     "use_kernel_forward_from_hub": lambda name: lambda cls: cls,
                     "ACT2FN": {"silu": torch.nn.functional.silu}}
        exec(compile(definitions, adapters.DB_QWEN2_SOURCE_URL, "exec", flags=__future__.annotations.compiler_flag), namespace)
        cls.namespace = namespace

    def make_layers(self):
        cfg = SimpleNamespace(hidden_size=8, num_attention_heads=2, num_key_value_heads=1,
                              attention_dropout=0.0, rms_norm_eps=1e-6,
                              intermediate_size=12, hidden_act="silu", _attn_implementation="eager",
                              sliding_window=None, layer_types=["full_attention"] * 28,
                              num_hidden_layers=28)
        return nn.ModuleList([self.namespace["Qwen2DecoderLayer"](cfg, i) for i in range(28)]).eval(), cfg

    def test_official_decoder_shell_and_28_indices(self):
        torch.manual_seed(17)
        layers, cfg = self.make_layers()
        skipped = {3, 4, 16, 28}
        # Qwen2Model/native readout scaffolding only; all decoder arithmetic is
        # from the audited official source. Mocked hashes below test override
        # wiring, and are never used as evidence of a loaded model's source.
        Qwen2Model = type("Qwen2Model", (nn.Module,), {})
        CogACTForCausalLM = type("CogACTForCausalLM", (nn.Module,), {})
        CogActModel = type("CogActModel", (nn.Module,), {})
        text, outer, container = Qwen2Model(), CogACTForCausalLM(), CogActModel()
        text.layers, text.config = layers, cfg
        container.llm, outer.model = text, container
        outer.eval()
        def toy_source(obj):
            return adapters.DB_COGACT_SOURCE_SHA256 if obj in (outer, container) else adapters.DB_QWEN2_SOURCE_SHA256
        x = torch.randn(1, 3, 8)
        angle = torch.randn(1, 3, 2).repeat(1, 1, 2)
        rotary = (angle.cos(), angle.sin())
        def run(cache, value, embeddings):
            for layer in layers:
                value = layer(value, use_cache=True, past_key_values=cache,
                              position_embeddings=embeddings, cache_position=torch.arange(value.shape[1]))
            return torch.nn.functional.layer_norm(value, (8,))
        calls = {"q": 0, "o": 0, "ffn": 0, "k": 0, "v": 0}
        handles = []
        for number in skipped:
            layer = layers[number - 1]
            for name, module in (("q", layer.self_attn.q_proj), ("o", layer.self_attn.o_proj),
                                 ("ffn", layer.mlp), ("k", layer.self_attn.k_proj), ("v", layer.self_attn.v_proj)):
                def count(module, args, output, name=name):
                    calls[name] += 1
                handles.append(module.register_forward_hook(count))
        try:
            with torch.no_grad(), patch.object(adapters, "source_sha256", side_effect=toy_source):
                native = run(FakeCache(), x, rotary)
                with adapters.db_cogact_hidden_bypass(outer, []) as audit:
                    torch.testing.assert_close(run(FakeCache(), x, rotary), native, rtol=0, atol=0)
                fast_cache, slow_cache = FakeCache(), FakeCache()
                with adapters.db_cogact_hidden_bypass(outer, skipped, reference=True) as audit:
                    slow = run(slow_cache, x, rotary)
                self.assertEqual(audit["reference_discard_layers"], sorted(skipped))
                before = dict(calls)
                with adapters.db_cogact_hidden_bypass(outer, skipped) as audit:
                    fast = run(fast_cache, x, rotary)
                self.assertEqual(audit["kv_shell_layers"], sorted(skipped))
                self.assertEqual(audit["fully_eliminated_layers"], [])
                for name in ("q", "o", "ffn"):
                    self.assertEqual(calls[name], before[name])
                for name in ("k", "v"):
                    self.assertEqual(calls[name], before[name] + len(skipped))
                torch.testing.assert_close(fast, slow, rtol=0, atol=0)
                # Append one token; this is cache arithmetic coverage, not an
                # assertion that native action inference uses language decode.
                extra = torch.randn(1, 1, 8)
                extra_rope = (rotary[0][:, :1], rotary[1][:, :1])
                with adapters.db_cogact_hidden_bypass(outer, skipped, reference=True):
                    slow_append = run(slow_cache, extra, extra_rope)
                with adapters.db_cogact_hidden_bypass(outer, skipped):
                    fast_append = run(fast_cache, extra, extra_rope)
                torch.testing.assert_close(fast_append, slow_append, rtol=0, atol=0)
                self.assertEqual(fast_cache.updates, list(range(28)) * 2)
                for index in range(28):
                    for a, b in zip(fast_cache.values[index], slow_cache.values[index]):
                        self.assertEqual(a.shape[-2], 4)
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                with self.assertRaises(ValueError):
                    with adapters.db_cogact_hidden_bypass(outer, [1]):
                        pass
            self.assertTrue(all("forward" not in vars(layer) for layer in layers))
            # Real source guard remains fail-closed outside the scaffolding.
            with self.assertRaises((ValueError, TypeError)):
                with adapters.db_cogact_hidden_bypass(outer, []):
                    pass
        finally:
            for handle in handles:
                handle.remove()

    def test_db_shell_rejects_missing_native_cache(self):
        layers, cfg = self.make_layers()
        for kwargs in ({}, {"use_cache": False, "past_key_values": FakeCache()},
                       {"use_cache": True, "past_key_values": FakeCache()}):
            with self.assertRaises(ValueError):
                adapters._db_qwen2_kv_shell(layers[2], torch.zeros(1, 1, 8), **kwargs)


if __name__ == "__main__":
    unittest.main()
