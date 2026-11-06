"""CPU-only contract tests; no checkpoint loading, downloads, or GPU allocation."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import pathlib
import sys
from types import SimpleNamespace
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "policies/openpi/src"))
sys.path.insert(0, str(ROOT / "policies/openpi/packages/openpi-client/src"))

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from flax.nnx import bridge
    from openpi.models import gemma
    HAVE_OPENPI = True
except ImportError:
    HAVE_OPENPI = False

from experiments.pi05_hidden_probe.backend import Backend, Patch, Prefix, attention_layout, select_positions


class CharacterTokenizer:
    """Deterministic offset-aware test tokenizer with upstream Gemma vocab size."""

    def vocab_size(self):
        return 257152

    def bos_id(self):
        return 2

    def eos_id(self):
        return 1

    def encode(self, text, *, out_type=None, add_bos=False, add_eos=False):
        if out_type == "immutable_proto":
            return SimpleNamespace(pieces=[SimpleNamespace(id=ord(c) + 10, begin=i, end=i + 1)
                                           for i, c in enumerate(text)])
        return ([2] if add_bos else []) + [ord(c) + 10 for c in text] + ([1] if add_eos else [])

    def decode(self, ids):
        return " ".join(map(str, ids))


@unittest.skipUnless(HAVE_OPENPI, "Run with policies/openpi/.venv/bin/python for OpenPI dependencies")
class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4)
        llm = bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="float32", adarms=True))
        llm.lazy_init(rngs=nnx.Rngs(17), method="init", use_adarms=[False, True])
        cls.backend = Backend.__new__(Backend)
        cls.backend._llm = llm
        cls.backend._tokenizer = CharacterTokenizer()
        cls.backend._configure_runtime()

    def prefix(self, text="X?\n", seed=5):
        backend = self.backend
        ids, offsets = backend._token_map(text)
        images = jnp.asarray(np.random.default_rng(seed).normal(size=(6, backend.width)), dtype=jnp.float32)
        language = backend._llm(jnp.asarray(ids, dtype=jnp.int32)[None], method="embed")[0]
        embedded = jnp.concatenate((images, language))
        return Prefix(embedded=embedded, input_mask=np.array([1, 1, 1, 1, 0, 0] + [1] * len(ids), dtype=bool),
                      image_spans={"base_0_rgb": [0, 1], "left_wrist_0_rgb": [2, 3], "right_wrist_0_rgb": [4, 5]},
                      text_token_ids=ids, text_positions=list(range(6, 6 + len(ids))), prompt_text=text,
                      text_offsets=offsets, task_text=text.rstrip(), task_span=(0, len(text.rstrip())))

    def assert_close(self, a, b, atol=2e-6):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol, rtol=2e-6)

    def test_original_scan_and_all_residual_boundaries(self):
        prefix = self.prefix()
        indices = np.flatnonzero(prefix.input_mask).tolist()
        full = self.backend.forward(prefix, capture_layers=(0,))
        self.assertEqual(full.hidden_states[0].shape, prefix.embedded.shape)
        self.assert_close(full.hidden_states[0], prefix.embedded)
        out = self.backend.forward(prefix, capture_layers=(0, 1, 2), capture_positions=indices)
        reference = self.backend.reference_forward(prefix)
        self.assert_close(out.normalized, reference.normalized)
        self.assert_close(out.logits, reference.logits)
        self.assert_close(out.hidden_states[0], np.asarray(prefix.embedded)[indices])

        def manual_layers(module, embedded, positions, mask):
            block = gemma.Block(configs=module.configs, parent=None)
            xs = [embedded, None]
            residuals = []
            for layer in range(2):
                params = jax.tree.map(lambda value: value[layer], module.variables["params"]["layers"])
                xs, _ = block.apply({"params": params}, xs, None, positions, mask[:, None], [None, None], True)
                residuals.append(xs[0])
            return residuals

        positions, mask = attention_layout(prefix.input_mask, len(prefix.input_mask))
        residuals = self.backend._llm(prefix.embedded[None], positions, mask, method=manual_layers)
        for layer, expected in enumerate(residuals, start=1):
            self.assert_close(out.hidden_states[layer], expected[0, indices])

    def test_self_patch_at_every_boundary(self):
        prefix = self.prefix()
        indices = select_positions(prefix, "images")
        original = self.backend.forward(prefix, capture_layers=(0, 1, 2), capture_positions=indices)
        for layer in range(3):
            result = self.backend.forward(prefix, patch=Patch(layer, indices, original.hidden_states[layer]))
            self.assert_close(result.normalized, original.normalized)
            self.assert_close(result.logits, original.logits)

    def test_bfloat16_matches_upstream_scan(self):
        config = gemma.Config(width=8, depth=2, mlp_dim=16, num_heads=2, num_kv_heads=1, head_dim=4)
        llm = bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="bfloat16", adarms=True))
        llm.lazy_init(rngs=nnx.Rngs(31), method="init", use_adarms=[False, True])
        backend = Backend.__new__(Backend)
        backend._llm = llm
        backend._tokenizer = CharacterTokenizer()
        backend._configure_runtime()
        prefix = self.prefix()
        out = backend.forward(prefix)
        expected = backend.reference_forward(prefix)
        self.assert_close(out.normalized, expected.normalized, atol=0)
        self.assert_close(out.logits, expected.logits, atol=0)

    def test_full_active_image_replacement_at_layer_zero(self):
        target = self.prefix()
        indices = select_positions(target, "images")
        real = dataclass_replace(target, embedded=target.embedded.at[jnp.asarray(indices)].set(
            jnp.asarray(np.random.default_rng(83).normal(size=(len(indices), self.backend.width)), dtype=jnp.float32)))
        expected = self.backend.reference_forward(real)
        donor = self.backend.forward(real, capture_layers=(0,), capture_positions=indices).hidden_states[0]
        actual = self.backend.forward(target, patch=Patch(0, indices, donor))
        self.assert_close(actual.normalized, expected.normalized)
        self.assert_close(actual.logits, expected.logits)
        self.assertGreater(np.max(np.abs(self.backend.forward(target).logits - expected.logits)), 1e-4)

    def test_fixed_prefix_answer_causality_and_padding_positions(self):
        valid = np.array([1, 1, 0, 1, 1, 1], dtype=bool)
        positions, mask = attention_layout(valid, prefix_length=4)
        np.testing.assert_array_equal(positions, [[0, 1, 1, 2, 3, 4]])
        mask = np.asarray(mask)[0]
        self.assertTrue(mask[0, 3])
        self.assertFalse(mask[:4, 4:].any())
        self.assertTrue(mask[5, 4])
        self.assertFalse(mask[4, 5])
        self.assertFalse(mask[:, 2].any())

        prefix = self.prefix()
        original = self.backend.forward(prefix)
        first = self.backend._forward(prefix, answer_ids=[80])
        second = self.backend._forward(prefix, answer_ids=[80, 81])
        active = np.flatnonzero(prefix.input_mask)
        self.assert_close(original.normalized[active], second.normalized[active])
        self.assert_close(first.normalized[-1], second.normalized[len(prefix.input_mask)])

    def test_generate_reapplies_patch_on_every_step(self):
        prefix = self.prefix()
        indices = select_positions(prefix, "text:placeholder")
        patch = Patch(1, indices, np.ones((1, self.backend.width), dtype=np.float32))
        calls = []
        original = self.backend._forward

        def observe(*args, **kwargs):
            calls.append((kwargs.get("patch"), list(kwargs.get("answer_ids", []))))
            return original(*args, **kwargs)

        self.backend._forward = observe
        try:
            generated = self.backend.generate(prefix, patch=patch, max_new_tokens=3)
        finally:
            self.backend._forward = original
        self.assertEqual(len(calls), len(generated["token_ids"]))
        self.assertGreater(len(calls), 1)
        for index, (actual_patch, previous_ids) in enumerate(calls):
            self.assertIs(actual_patch, patch)
            self.assertEqual(previous_ids, generated["token_ids"][:index])
            expected = original(prefix, patch=patch, answer_ids=previous_ids)
            self.assertEqual(int(expected.logits.argmax()), generated["token_ids"][index])

    def test_candidate_likelihood_is_causal_sum_and_optional_mean(self):
        prefix = self.prefix()
        scores = self.backend.score_answers(prefix, ["ab", "a"])
        means = self.backend.score_answers(prefix, ["ab"], length_normalize=True)
        ids = self.backend._tokenizer.encode("ab")
        expected = 0.0
        for index, token in enumerate(ids):
            logits = self.backend._forward(prefix, answer_ids=ids[:index]).logits.astype(np.float64)
            expected += logits[token] - np.logaddexp.reduce(logits)
        self.assertAlmostEqual(scores["ab"], expected, places=7)
        self.assertAlmostEqual(means["ab"], expected / 2, places=7)

    def test_selectors_use_verified_offsets_and_exclude_masked_camera(self):
        prefix = self.prefix("pick mug X\n")
        self.assertEqual(select_positions(prefix, "images"), [0, 1, 2, 3])
        self.assertEqual(select_positions(prefix, "image:left_wrist_0_rgb"), [2, 3])
        self.assertEqual(select_positions(prefix, "text:object", "mug"), [6 + 1 + len("pick mug") - 1])
        self.assertNotEqual(select_positions(prefix, "text:last"), select_positions(prefix, "text:last_task"))
        self.assertEqual(select_positions(prefix, "text:placeholder"), select_positions(prefix, "text:last_task"))
        with self.assertRaisesRegex(ValueError, "single-token"):
            select_positions(self.prefix("XYZ?"), "text:placeholder", "XYZ")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            select_positions(self.prefix("X and X"), "text:placeholder")
        with self.assertRaisesRegex(ValueError, "no valid"):
            select_positions(prefix, "image:right_wrist_0_rgb")

    def test_invalid_patch_rejected_before_execution(self):
        prefix = self.prefix()
        for patch in [Patch(3, [0], np.zeros((1, 8))), Patch(0, [4], np.zeros((1, 8))),
                      Patch(0, [0, 0], np.zeros((2, 8))), Patch(0, [0], np.full((1, 8), np.nan)),
                      Patch(0, [len(prefix.input_mask)], np.zeros((1, 8)))]:
            with self.assertRaises(ValueError):
                self.backend.forward(prefix, patch=patch)

    def test_prepare_source_preserves_policy_and_target_preserves_exact_text(self):
        from openpi import transforms
        from openpi.models.model import ModelType
        from openpi.models.tokenizer import PaligemmaTokenizer
        from openpi.policies.libero_policy import LiberoInputs

        backend = Backend.__new__(Backend)
        backend._llm = self.backend._llm
        backend._tokenizer = self.backend._tokenizer
        backend._config = SimpleNamespace(model=SimpleNamespace(max_token_len=48))
        source_tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
        source_tokenizer._tokenizer = backend._tokenizer
        source_tokenizer._max_len = 48
        backend._tokenize_transform = transforms.TokenizePrompt(source_tokenizer, discrete_state_input=False)
        backend._transforms = (LiberoInputs(ModelType.PI05), backend._tokenize_transform)

        def embed_prefix(obs):
            images = [jnp.broadcast_to(jnp.mean(value, axis=(1, 2, 3))[:, None, None], (1, 2, 8))
                      for value in obs.images.values()]
            masks = [jnp.repeat(value[:, None], 2, axis=1) for value in obs.image_masks.values()]
            text = backend._llm(obs.tokenized_prompt, method="embed")
            embedded = jnp.concatenate([*images, text], axis=1)
            return embedded, jnp.concatenate([*masks, obs.tokenized_prompt_mask], axis=1), jnp.zeros(embedded.shape[1], dtype=bool)

        backend._model = SimpleNamespace(embed_prefix=embed_prefix)
        raw = {"observation/image": np.full((224, 224, 3), 128, np.uint8),
               "observation/wrist_image": np.full((224, 224, 3), 64, np.uint8),
               "observation/state": np.arange(8, dtype=np.float32), "prompt": "  pick_mug\nnow  "}
        source = backend.prepare_source(raw)
        prompt = "answer en What_color is X?\n"
        target = backend.prepare_target(raw, prompt)
        self.assertEqual(source.prompt_text, "pick mug now\n")
        self.assertEqual(target.prompt_text, prompt)
        self.assertEqual(source.text_token_ids[0], backend._tokenizer.bos_id())
        self.assertEqual(target.text_token_ids, backend._tokenizer.encode(prompt, add_bos=True))
        self.assertEqual(source.image_spans, target.image_spans)
        self.assertEqual(source.metadata["camera_masks"], target.metadata["camera_masks"])
        self.assertEqual(select_positions(source, "text:last_task"), [6 + len("pick mug now")])
        self.assertEqual(raw["prompt"], "  pick_mug\nnow  ")
        np.testing.assert_array_equal(source.state, raw["observation/state"])


def dataclass_replace(obj, **kwargs):
    import dataclasses
    return dataclasses.replace(obj, **kwargs)


if __name__ == "__main__":
    unittest.main()
