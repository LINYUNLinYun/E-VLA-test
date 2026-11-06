"""CPU tiny-model evidence for the unmodified HF Target path."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.pi05_hidden_probe.backend import Patch, Prefix, select_positions
from experiments.pi05_hidden_probe.hf_backend import DEFAULT_RUNTIME, HFBackend, load_official_runtime

if not DEFAULT_RUNTIME.exists():
    pytest.skip("Independent official Transformers runtime is not installed", allow_module_level=True)

torch, _, Model = load_official_runtime()
from transformers import PaliGemmaConfig


@pytest.fixture(scope="module")
def backend():
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = PaliGemmaConfig(image_token_index=31, projection_dim=16, hidden_size=16,
        text_config={"model_type": "gemma", "vocab_size": 64, "hidden_size": 16,
                     "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 8, "pad_token_id": 0},
        vision_config={"model_type": "siglip_vision_model", "hidden_size": 16,
                       "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                       "image_size": 8, "patch_size": 4, "projection_dim": 16})
    config._attn_implementation = "eager"
    model = Model(config).float().eval().requires_grad_(False)
    obj = HFBackend.__new__(HFBackend)
    obj.torch, obj._model, obj._lm = torch, model, model.model.language_model
    obj.device, obj.dtype, obj.width, obj.depth = torch.device("cpu"), torch.float32, 16, 2
    obj._configure_helpers()
    obj._tokenizer = SimpleNamespace(eos_token_id=1,
        encode=lambda text, add_special_tokens=False: {"red": [10], "blue": [11]}[text],
        decode=lambda ids, skip_special_tokens=True: " ".join(map(str, ids)))
    return obj


def prefix(backend, seed=3):
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randn((3, 3, 8, 8), generator=generator)
    with torch.inference_mode():
        images = backend._model.get_image_features(pixels).reshape(-1, 16)
        words = backend._model.get_input_embeddings()(torch.tensor([2, 4, 5]))
        raw = torch.cat([images, words])
    return Prefix((raw * backend._scale).numpy(), np.array([1]*8 + [0]*4 + [1]*3, bool),
                  {"base_0_rgb": list(range(4)), "left_wrist_0_rgb": list(range(4, 8)), "right_wrist_0_rgb": list(range(8, 12))},
                  [2, 4, 5], [12, 13, 14], prompt_text="X?", text_offsets=[(0,0), (0,1), (1,2)],
                  metadata={"raw_embeddings": raw.numpy(), "pixel_values": pixels.numpy(), "input_ids": [31]*12 + [2, 4, 5]})


def test_unhooked_official_forward_and_each_self_patch(backend):
    p = prefix(backend)
    reference = backend.reference_forward(p)
    result = backend.forward(p, capture_layers=[0, 1, 2])
    np.testing.assert_array_equal(result.normalized, reference.normalized)
    np.testing.assert_array_equal(result.logits, reference.logits)
    np.testing.assert_array_equal(result.hidden_states[0], p.embedded)
    positions = select_positions(p, "images")
    for layer in [0, 1, 2]:
        patched = backend.forward(p, patch=Patch(layer, positions, result.hidden_states[layer][positions]))
        np.testing.assert_array_equal(patched.logits, result.logits)
        np.testing.assert_array_equal(patched.normalized, result.normalized)


def test_layer0_image_replacement_is_equivalent(backend):
    real, blank = prefix(backend, 3), prefix(backend, 9)
    positions = select_positions(real, "images")
    result = backend.forward(blank, patch=Patch(0, positions, real.embedded[positions]))
    # Masked camera values cannot influence any valid token.
    expected = backend.forward(real)
    np.testing.assert_array_equal(result.normalized[real.input_mask], expected.normalized[real.input_mask])
    np.testing.assert_array_equal(result.logits, expected.logits)


def test_prefix_cannot_read_the_answer_and_scoring_uses_only_past(backend):
    p = prefix(backend)
    a = backend._forward(p, answer_ids=[6])
    b = backend._forward(p, answer_ids=[7])
    # Fully masked padding-query rows are not model inputs or attention keys.
    # Every valid prefix row must remain independent of the answer.
    np.testing.assert_array_equal(a.normalized[:len(p.input_mask)][p.input_mask],
                                  b.normalized[:len(p.input_mask)][p.input_mask])
    initial = backend.forward(p).logits.astype(np.float64)
    log_z = initial.max() + np.log(np.exp(initial - initial.max()).sum())
    scores = backend.score_answers(p, ["red", "blue"])
    assert scores["red"] == pytest.approx(initial[10] - log_z)
    assert scores["blue"] == pytest.approx(initial[11] - log_z)


def test_patch_is_reapplied_on_every_generated_token_and_hooks_are_removed(backend, monkeypatch):
    p = prefix(backend)
    intervention = Patch(1, [0], np.zeros((1, 16), np.float32))
    calls = []
    original = backend._forward
    def observed(*args, **kwargs):
        calls.append((kwargs.get("patch"), list(kwargs.get("answer_ids", []))))
        result = original(*args, **kwargs)
        result.logits[1] = -1e9
        return result
    monkeypatch.setattr(backend, "_forward", observed)
    backend.generate(p, patch=intervention, max_new_tokens=3)
    assert [len(answer) for _, answer in calls] == [0, 1, 2]
    assert all(patch is intervention for patch, _ in calls)
    assert all(not layer._forward_hooks and not layer._forward_pre_hooks for layer in backend._lm.layers)


def test_invalid_or_masked_positions_are_rejected(backend):
    p = prefix(backend)
    with pytest.raises(ValueError, match="valid prefix"):
        backend.forward(p, patch=Patch(1, [8], np.zeros((1, 16))))
    with pytest.raises(ValueError, match="dimensions"):
        backend.forward(p, patch=Patch(1, [0], np.zeros((1, 8))))
