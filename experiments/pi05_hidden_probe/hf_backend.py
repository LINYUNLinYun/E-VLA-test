"""Frozen official PaliGemma readout of pi0.5 residuals; no learned mapping.

This module uses a separately installed, unmodified Transformers runtime. The
OpenPI environment patches its own Torch Gemma files, which are unsuitable for
using an ordinary HF PaliGemma checkpoint as a text decoder.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
from pathlib import Path
import sys

import numpy as np

from .backend import ForwardResult, Patch, Prefix, select_positions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUNTIME = ROOT / "shared/caches/pi05_hidden_probe_hf_runtime"


def load_official_runtime(path=DEFAULT_RUNTIME):
    path = Path(path).resolve()
    if not (path / "transformers/__init__.py").is_file():
        raise ValueError(f"Missing independent official HF runtime: {path}")
    if "transformers" in sys.modules:
        imported = Path(sys.modules["transformers"].__file__).resolve()
        if not imported.is_relative_to(path):
            raise RuntimeError("The patched OpenPI Transformers was already imported; load the HF Target before the JAX Source")
    sys.path.insert(0, str(path))
    import torch
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
    from transformers.models.gemma import modeling_gemma
    from transformers.models.paligemma import modeling_paligemma
    for module in [modeling_gemma, modeling_paligemma]:
        if not Path(module.__file__).resolve().is_relative_to(path):
            raise RuntimeError("Mixed patched/unmodified Transformers modules")
    return torch, AutoProcessor, PaliGemmaForConditionalGeneration


class HFBackend:
    """Three camera slots (two active), with scaled block-boundary residuals."""

    def __init__(self, checkpoint, device="cuda:0", dtype="bfloat16", hf_runtime=None, **unused):
        if unused:
            raise ValueError(f"Unknown HF Target settings: {sorted(unused)}")
        self.torch, processor_cls, model_cls = load_official_runtime(hf_runtime or DEFAULT_RUNTIME)
        self.device = self.torch.device(device)
        self.dtype = getattr(self.torch, dtype)
        self._checkpoint = Path(checkpoint).resolve()
        self._runtime = Path(hf_runtime or DEFAULT_RUNTIME).resolve()
        self._processor = processor_cls.from_pretrained(self._checkpoint, local_files_only=True)
        self._model, loading = model_cls.from_pretrained(
            self._checkpoint, local_files_only=True, torch_dtype=self.dtype,
            attn_implementation="eager", output_loading_info=True)
        if loading["missing_keys"] or loading["mismatched_keys"] or loading["unexpected_keys"]:
            raise ValueError(f"HF checkpoint loading is not exact: {loading}")
        self._loading = loading
        self._model.to(self.device).eval().requires_grad_(False)
        self._lm = self._model.model.language_model
        self._tokenizer = self._processor.tokenizer
        self.depth = len(self._lm.layers)
        self.width = int(self._model.config.text_config.hidden_size)
        self.vocab_size = int(self._model.config.text_config.vocab_size)
        self._per_image = int(self._model.config.text_config.num_image_tokens)
        self._image_id = int(self._model.config.image_token_id)
        self._configure_helpers()

    def _configure_helpers(self):
        self._scale = self.torch.tensor(self.width ** 0.5, dtype=self.dtype, device=self.device)

    def _array(self, value):
        return value.detach().float().cpu().numpy()

    def _tokens(self, text):
        encoded = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [int(self._tokenizer.bos_token_id), *map(int, encoded["input_ids"])], [(0, 0), *map(tuple, encoded["offset_mapping"])]

    def prepare_source(self, raw_obs):
        raise ValueError("PaliGemma is a readout Target only; Source must remain the actual pi0.5 policy")

    def prepare_target(self, raw_obs, prompt):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("Target prompt must be nonempty text")
        from PIL import Image
        t = self.torch
        images = [Image.fromarray(np.asarray(raw_obs["observation/image"], dtype=np.uint8)),
                  Image.fromarray(np.asarray(raw_obs["observation/wrist_image"], dtype=np.uint8)),
                  Image.fromarray(np.zeros_like(raw_obs["observation/image"], dtype=np.uint8))]
        pixels = self._processor.image_processor(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        ids, offsets = self._tokens(prompt)
        with t.inference_mode():
            image_raw = self._model.get_image_features(pixels).reshape(-1, self.width)
            words = self._model.get_input_embeddings()(t.tensor(ids, device=self.device))
            raw = t.cat([image_raw, words], dim=0)
            scaled = raw * self._scale
        per = self._per_image
        names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        mask = np.array([True] * (per * 2) + [False] * per + [True] * len(ids))
        return Prefix(embedded=self._array(scaled), input_mask=mask,
            image_spans={name: list(range(i * per, (i + 1) * per)) for i, name in enumerate(names)},
            text_token_ids=ids, text_positions=list(range(per * 3, per * 3 + len(ids))),
            prompt_text=prompt, text_offsets=offsets,
            metadata={"raw_embeddings": self._array(raw), "pixel_values": self._array(pixels),
                      "input_ids": [self._image_id] * (per * 3) + ids,
                      "camera_masks": dict(zip(names, [True, True, False])), "tokens_per_camera": per})

    def _layout(self, prefix, answer_ids):
        t = self.torch
        valid = t.as_tensor(np.r_[prefix.input_mask, np.ones(len(answer_ids), bool)], device=self.device, dtype=t.bool)
        n = len(valid)
        prefix_n = len(prefix.input_mask)
        i = t.arange(n, device=self.device)
        segment = t.clamp(i - prefix_n + 1, min=0)
        allowed = (segment[:, None] >= segment[None, :]) & valid[:, None] & valid[None, :]
        mask = t.where(allowed, t.zeros((), dtype=self.dtype, device=self.device),
                       t.full((), t.finfo(self.dtype).min, dtype=self.dtype, device=self.device))[None, None]
        # Native PaliGemma uses 1-indexed positions; padded slots do not advance them.
        positions = t.cumsum(valid, dim=0)[None]
        return mask, positions, int(t.where(valid)[0][-1])

    def _validate_patch(self, prefix, patch):
        if patch is None:
            return
        if type(patch.layer) is not int or not 0 <= patch.layer <= self.depth:
            raise ValueError("Invalid Target patch layer")
        p = np.asarray(patch.positions)
        if p.ndim != 1 or not len(p) or not np.issubdtype(p.dtype, np.integer) or len(set(p.tolist())) != len(p):
            raise ValueError("Patch positions must be distinct integers")
        if np.any(p < 0) or np.any(p >= len(prefix.input_mask)) or not prefix.input_mask[p].all():
            raise ValueError("Cannot patch positions outside the valid prefix")
        if np.shape(patch.values) != (len(p), self.width) or not np.isfinite(patch.values).all():
            raise ValueError("Source/Target dimensions do not match, or patch contains nonfinite values")

    def _forward(self, prefix, capture_layers=(), capture_positions=None, patch=None, answer_ids=()):
        t = self.torch
        self._validate_patch(prefix, patch)
        if any(type(layer) is not int or not 0 <= layer <= self.depth for layer in capture_layers):
            raise ValueError("Capture layers must be integers within 0..L")
        selected = list(range(len(prefix.input_mask))) if capture_positions is None else list(capture_positions)
        raw = t.tensor(prefix.metadata["raw_embeddings"], dtype=self.dtype, device=self.device)[None]
        scaled = t.tensor(prefix.embedded, dtype=self.dtype, device=self.device)[None]
        if answer_ids:
            answer = self._model.get_input_embeddings()(t.tensor([list(answer_ids)], device=self.device))
            raw = t.cat([raw, answer], dim=1)
            scaled = t.cat([scaled, answer * self._scale], dim=1)
        mask, positions, readout = self._layout(prefix, answer_ids)
        captures, handles = {}, []
        values = None if patch is None else t.tensor(patch.values, dtype=self.dtype, device=self.device)

        def first_input(module, args):
            # Define H^0 after scaling, so injected pi0.5 vectors are not scaled a second time.
            hidden = scaled.clone()
            if patch is not None and patch.layer == 0:
                hidden[:, patch.positions] = values[None]
            if 0 in capture_layers:
                captures[0] = self._array(hidden[0, selected])
            return (hidden, *args[1:])

        def after_layer(layer):
            def hook(module, args, output):
                hidden = output[0]
                if patch is not None and patch.layer == layer:
                    hidden = hidden.clone()
                    hidden[:, patch.positions] = values[None]
                if layer in capture_layers:
                    captures[layer] = self._array(hidden[0, selected])
                return (hidden, *output[1:])
            return hook

        handles.append(self._lm.layers[0].register_forward_pre_hook(first_input))
        for index, layer in enumerate(self._lm.layers, start=1):
            if index in capture_layers or (patch is not None and patch.layer == index):
                handles.append(layer.register_forward_hook(after_layer(index)))
        try:
            with t.inference_mode():
                output = self._lm(inputs_embeds=raw, attention_mask=mask, position_ids=positions,
                                  use_cache=False, return_dict=True)
                normalized = output.last_hidden_state
                logits = self._model.lm_head(normalized[:, readout])
        finally:
            for handle in handles:
                handle.remove()
        return ForwardResult(captures, self._array(logits[0]), self._array(normalized[0]))

    def forward(self, prefix, capture_layers=(), capture_positions=None, patch=None):
        return self._forward(prefix, capture_layers, capture_positions, patch)

    def reference_forward(self, prefix):
        """Unhooked official multimodal forward on precisely the same Target inputs."""
        t = self.torch
        mask, positions, _ = self._layout(prefix, [])
        with t.inference_mode():
            output = self._model(input_ids=t.tensor([prefix.metadata["input_ids"]], device=self.device),
                pixel_values=t.tensor(prefix.metadata["pixel_values"], device=self.device, dtype=self.dtype),
                attention_mask=mask, position_ids=positions, output_hidden_states=True,
                use_cache=False, return_dict=True, logits_to_keep=1)
        return ForwardResult({}, self._array(output.logits[0, -1]), self._array(output.hidden_states[-1][0]))

    def generate(self, prefix, patch=None, max_new_tokens=16):
        answer = []
        for _ in range(max_new_tokens):
            token = int(np.argmax(self._forward(prefix, patch=patch, answer_ids=answer).logits))
            answer.append(token)
            if token == self._tokenizer.eos_token_id:
                break
        return {"text": self._tokenizer.decode(answer, skip_special_tokens=True), "token_ids": answer,
                "eos": bool(answer and answer[-1] == self._tokenizer.eos_token_id)}

    def score_answers(self, prefix, answers, patch=None, length_normalize=False):
        scores, first_logits = {}, None
        for answer in answers:
            tokens = self._tokenizer.encode(answer, add_special_tokens=False)
            if not tokens:
                raise ValueError("Empty candidate answer")
            total = 0.
            for i, token in enumerate(tokens):
                if i == 0 and first_logits is not None:
                    logits = first_logits
                else:
                    logits = self._forward(prefix, patch=patch, answer_ids=tokens[:i]).logits.astype(np.float64)
                    if i == 0:
                        first_logits = logits
                maximum = logits.max()
                total += float(logits[token] - maximum - np.log(np.exp(logits - maximum).sum()))
            scores[answer] = total / len(tokens) if length_normalize else total
        return scores

    def audit(self, prefix=None):
        import transformers
        from transformers.models.gemma import modeling_gemma
        from transformers.models.paligemma import modeling_paligemma
        sources = {str(Path(m.__file__).resolve()): hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                   for m in [modeling_gemma, modeling_paligemma]}
        report = {"backend": "hf-paligemma", "checkpoint": str(self._checkpoint), "depth": self.depth,
                  "width": self.width, "vocab_size": self.vocab_size, "device": str(self.device), "dtype": str(self.dtype),
                  "transformers": transformers.__version__, "torch": self.torch.__version__,
                  "runtime": str(self._runtime), "code_sha256": sources, "weight_loading": self._loading,
                  "tokenizer_sha256": hashlib.sha256(self._tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                  "layer_convention": "0=Gemma-scaled embeddings; k=full block residual before final norm",
                  "input_contract": "agent, wrist, masked zero camera; native HF preprocessing; exact prompt/BOS",
                  "position_ids": "1-indexed cumulative valid positions (native PaliGemma convention)",
                  "attention": "bidirectional fixed prefix; causal answers; full recomputation each step",
                  "mapping": "identity; no pooling, normalization, or trained alignment"}
        if prefix is not None:
            report["prefix"] = {"shape": list(prefix.embedded.shape), "mask": prefix.input_mask.tolist(),
                                "image_spans": prefix.image_spans, "text_token_ids": prefix.text_token_ids}
        return report
