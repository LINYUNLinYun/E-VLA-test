"""Keep invariant Pi0 weights when changing prompt-shaped scratch buffers.

This is not wired into any experiment. CPU ownership contracts do not establish
GPU numerical parity, peak memory, or the requested reservation floor. Those
must pass a separate gate before this helper can serve scientific episodes.
"""
from __future__ import annotations

import gc
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
KERNEL_PATH = ROOT / 'policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py'
KERNEL_SHA256 = 'c554399a9adf25b1ee512a73325cbe05f56943b1a4cbc1748934d2f413cd56b8'
VALIDATION_STATUS = 'preparation_only_GPU_parity_and_memory_gate_pending'


def verify_kernel_source():
    actual = hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest()
    if actual != KERNEL_SHA256:
        raise RuntimeError('Pi0 kernel source changed; review a new resident revision')


def prompt_buffer_shapes(prompt_length, *, attention_logits_fp32=False):
    if type(prompt_length) is not int or prompt_length <= 0:
        raise ValueError('Prompt length must be a positive integer')
    encoder = 2 * 256 + prompt_length
    decoder = 50 + 1
    shapes = {
        'encoder_rope_weights': (encoder, 256),
        'encoder_x': (encoder, 2048),
        'encoder_x_norm': (encoder, 2048),
        'encoder_K': (18, encoder + decoder, 256),
        'encoder_V': (18, encoder + decoder, 256),
        'encoder_Q': (encoder * 8, 256),
        'encoder_hidden': (encoder, 16384),
        'decoder_attn_buf': (decoder * 8, encoder + decoder),
    }
    if attention_logits_fp32:
        shapes['decoder_attn_logits_buf'] = shapes['decoder_attn_buf']
    return shapes


class TorchOperations:
    """Imported only for an already constructed CUDA runtime, never at import."""

    def __init__(self):
        import torch
        self.torch = torch

    def synchronize(self, device):
        self.torch.cuda.synchronize(device=device)

    def allocate(self, shape, template):
        return self.torch.empty(shape, dtype=template.dtype, device=template.device)

    def copy_embedding(self, destination, embedding):
        destination.copy_(embedding.to(destination.device))

    def initialize_rope(self, runtime):
        # Same arithmetic, order and BF16 conversion as the pinned constructor.
        torch = self.torch
        device = runtime.weights['language_embeds'].device
        encoder = runtime.num_views * 256 + runtime.prompt_len
        decoder = runtime.chunk_size + 1
        for key, count, offset in (
            ('encoder_rope_weights', encoder, 0),
            ('decoder_rope_weights', decoder, encoder),
        ):
            position_ids = torch.arange(count, device=device)
            if offset:
                position_ids = position_ids + offset
            inv_freq = 1.0 / (10000 ** (
                torch.arange(0, 256, 2, dtype=torch.float32, device=device) / 256))
            phase = inv_freq[None, :] * position_ids[:, None]
            cosine = torch.cos(phase).to(torch.bfloat16)
            sine = torch.sin(phase).to(torch.bfloat16)
            runtime.buffers[key].copy_(
                torch.cat([cosine[:, :, None], sine[:, :, None]], 2).view(-1, 256))

    def capture(self, runtime):
        runtime.infer_graph = self.torch.cuda.CUDAGraph()
        runtime.record_infer_graph()


class PromptRuntimeOwner:
    """Own one initialized runtime and its immutable weight tensor references.

    The future caller must serialize inference, acknowledge the previous policy
    proposal, reset episode/policy residual state, and pass every cache holding
    captured graphs. This class neither resets policy state nor implements a
    multi-condition server. It never calls empty_cache or a model constructor.
    """

    def __init__(self, runtime, *, operations=None):
        verify_kernel_source()
        if runtime.num_views != 2 or runtime.chunk_size != 50:
            raise ValueError('Only the pinned two-view H50 runtime is supported')
        self.runtime = runtime
        self.operations = operations if operations is not None else TorchOperations()
        self._weights = runtime.weights
        self._anchors = {k: v for k, v in runtime.weights.items() if k != 'language_embeds'}
        self._ready = True
        self.rebind_events = []
        self._validate_shapes(runtime.prompt_len)

    @property
    def ready(self):
        return self._ready

    def invariant_tensors(self):
        self.assert_ready()
        return tuple(self._anchors.values())

    def assert_ready(self):
        if not self._ready:
            raise RuntimeError('Prompt transition failed; runtime must not serve inference')
        if self.runtime.weights is not self._weights:
            self._ready = False
            raise RuntimeError('Invariant weight dictionary was replaced')
        if set(self.runtime.weights) != {*self._anchors, 'language_embeds'}:
            self._ready = False
            raise RuntimeError('Invariant weight key set changed')
        if any(self.runtime.weights[k] is not v for k, v in self._anchors.items()):
            self._ready = False
            raise RuntimeError('Invariant model weight allocation was replaced')
        try:
            if self.runtime.num_views != 2 or self.runtime.chunk_size != 50:
                raise ValueError('Runtime view/horizon contract changed')
            self._validate_shapes(self.runtime.prompt_len)
        except Exception:
            self._ready = False
            raise

    def _validate_shapes(self, prompt_length):
        expected = prompt_buffer_shapes(prompt_length,
            attention_logits_fp32=bool(getattr(self.runtime, 'attention_logits_fp32', False)))
        if tuple(self.runtime.weights['language_embeds'].shape) != (prompt_length, 2048):
            raise ValueError('Runtime language embedding shape differs')
        for key, shape in expected.items():
            if tuple(self.runtime.buffers[key].shape) != shape:
                raise ValueError('Runtime scratch shape differs: ' + key)
        if tuple(self.runtime.buffers['decoder_rope_weights'].shape) != (51, 256):
            raise ValueError('Decoder RoPE shape differs')

    def rebind(self, embedding, *, episode_boundary, graph_caches=()):
        self.assert_ready()
        shape = tuple(embedding.shape)
        if len(shape) != 2 or shape[1] != 2048 or type(shape[0]) is not int or shape[0] <= 0:
            raise ValueError('Expected a nonempty [prompt_length,2048] embedding')
        previous = self.runtime.prompt_len
        changed = shape[0] != previous
        if not episode_boundary:
            raise ValueError('Prompt changes require an acknowledged episode boundary')
        runtime, ops = self.runtime, self.operations
        try:
            ops.synchronize(runtime.weights['language_embeds'].device)
        except Exception:
            self._ready = False
            raise
        if not changed:
            try:
                ops.copy_embedding(runtime.weights['language_embeds'], embedding)
            except Exception:
                self._ready = False
                raise
            return dict(changed=False, prompt_length=previous, invariant_weights_retained=True)

        specs = prompt_buffer_shapes(shape[0],
            attention_logits_fp32=bool(getattr(runtime, 'attention_logits_fp32', False)))
        # Keep only allocation metadata. Holding old tensors here would retain
        # every old graph/scratch allocation while building the replacement.
        templates = {key: (runtime.buffers[key].dtype, runtime.buffers[key].device) for key in specs}
        old_embedding = runtime.weights['language_embeds']
        embedding_template = (old_embedding.dtype, old_embedding.device)
        self._ready = False
        try:
            for cache in graph_caches:
                cache.clear()
            runtime.infer_graph = None
            for key in specs:
                del runtime.buffers[key]
            del runtime.weights['language_embeds']
            del old_embedding
            gc.collect()
            runtime.prompt_len = shape[0]
            for key, buffer_shape in specs.items():
                runtime.buffers[key] = ops.allocate(buffer_shape, _AllocationTemplate(*templates[key]))
            runtime.weights['language_embeds'] = ops.allocate(shape, _AllocationTemplate(*embedding_template))
            ops.copy_embedding(runtime.weights['language_embeds'], embedding)
            ops.initialize_rope(runtime)
            ops.capture(runtime)
            ops.synchronize(runtime.weights['language_embeds'].device)
            self._ready = True
            self.assert_ready()
            self._validate_shapes(shape[0])
        except Exception:
            self._ready = False
            raise
        event = dict(changed=True, previous_prompt_length=previous,
            prompt_length=shape[0], invariant_weights_retained=True,
            weight_initializations=0, scratch_graph_initializations=1,
            validation_status=VALIDATION_STATUS)
        self.rebind_events.append(event)
        return event


class _AllocationTemplate:
    def __init__(self, dtype, device):
        self.dtype, self.device = dtype, device
