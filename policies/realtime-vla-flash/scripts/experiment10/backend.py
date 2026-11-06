"""Experiment-only arbitrary VLM-mask and full-interval D1..D10 inference.

One runtime per prompt length; KV shell semantics match the production kernels.
Schedule tensors are cloned per D so CUDA graphs cannot alias another schedule.
"""
from __future__ import annotations

import gc
from collections import OrderedDict
import pickle
import time

import numpy as np

from scripts.experiment10.common import CONVERTED, sha256


def validate_mask(skip):
    values = tuple(int(i) for i in skip)
    if len(set(values)) != len(values) or any(i < 1 or i > 18 for i in values):
        raise ValueError('Expected unique one-based VLM layers 1..18')
    return tuple(sorted(values))


def decoder(runtime, schedule):
    from scripts.experiment1.run_intermediate_action_probe import _decoder_layer
    from scripts.spec.triton import pi0_infer as kernels
    time_biases, out_w, out_b = schedule
    w, b = runtime.weights, runtime.buffers
    seq = b['encoder_x'].shape[0]
    kernels.matmul_1_32_1024_bias(b['observation_state_normalized'], w['decoder_state_in_proj_w'],
                                w['decoder_state_in_proj_b'], b['decoder_state_buf'])
    for step in range(len(time_biases)):
        b['decoder_x'][:1].copy_(b['decoder_state_buf'])
        kernels.matmul_k_32_1024_bias_silu(b['diffusion_noise'], w['decoder_action_fused_in_proj_w'],
                                          time_biases[step], b['decoder_x_buf'])
        kernels.matmul_k_1024_1024_bias(b['decoder_x_buf'], w['decoder_action_mlp_w'],
                                       w['decoder_action_mlp_b'], b['decoder_x'][1:])
        for layer in range(18):
            _decoder_layer(w, b, seq, layer)
        kernels.rms_matmul_k_1024_32_bias_res(b['decoder_x'][1:], out_w, out_b,
                                           b['diffusion_noise'], b['decoder_norm_factor_buf'])


class Backend:
    def __init__(self, metadata):
        import torch
        from scripts.experiment1.run_intermediate_action_probe import _load_prompt_embeddings
        from scripts.spec.spec_serve_policy import _build_policy_components
        torch.set_grad_enabled(False)
        self.embeddings = _load_prompt_embeddings(CONVERTED, metadata)
        self.weight_hash = sha256(CONVERTED / 'base/base_weights.pkl')
        with (CONVERTED / 'base/base_weights.pkl').open('rb') as stream:
            self.checkpoint = pickle.load(stream)
        _, self.transform, self.output_transform = _build_policy_components(
            config_name='pi0_libero', checkpoint_dir=CONVERTED / 'base')
        self.runtime, self.length = None, None
        self.graphs, self.schedules, self.parity_lengths = OrderedDict(), {}, set()

    def prepare(self, observation, meta, noise_array):
        import torch
        from scripts.spec.triton import pi0_infer
        from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
        embedding = self.embeddings[meta['suite'], meta['task_description']]
        if self.length != len(embedding):
            self.graphs.clear()
            self.schedules.clear()
            self.runtime = None
            gc.collect()
            torch.cuda.empty_cache()
            checkpoint = dict(self.checkpoint, language_embeds=embedding)
            self.runtime = pi0_infer.Pi0Inference(checkpoint, num_views=2, chunk_size=50, denoise_steps=10)
            self.length = len(embedding)
        self.runtime.weights['language_embeds'].copy_(embedding.to('cuda'))
        self.transformed = self.transform(observation)
        self.inputs = prepare_triton_inputs_from_transformed(transformed=self.transformed, device='cuda',
                              action_horizon=50, action_dim=32, noise=noise_array)

    def schedule(self, steps):
        if steps not in (1, 2, 3, 4, 5, 7, 10):
            raise ValueError('Supported complete-interval schedules: D1/2/3/4/5/7/10')
        if steps not in self.schedules:
            w = self.runtime.weights
            suffix = ''
            if steps == 5:
                self.runtime._build_five_step_weights()
                suffix = '_5'
            elif steps != 10:
                self.runtime._build_custom_step_weights(steps)
                suffix = '_custom'
            self.schedules[steps] = tuple(t.clone() for t in (
                w['decoder_action_fused_time_biases' + suffix][:steps],
                w['decoder_action_fused_out_proj_w' + suffix], w['decoder_action_fused_out_proj_b' + suffix]))
        return self.schedules[steps]

    def record(self, skip, steps):
        from scripts.spec.triton import pi0_infer as kernels
        w, b = self.runtime.weights, self.runtime.buffers
        schedule = self.schedule(steps)
        b['encoder_x'][512:].copy_(w['language_embeds'])
        kernels.vision_encoder(w, b, 2)
        kernels.transformer_encoder(w, b, b['encoder_x'].shape[0],
                                    vlm_layer_indices=tuple(i for i in range(18) if i + 1 not in skip))
        decoder(self.runtime, schedule)

    def infer(self, skip=(), steps=10, *, timed=False):
        import torch
        skip = validate_mask(skip)
        self.schedule(steps)
        key = (skip, steps)
        images, state, noise = self.inputs
        b = self.runtime.buffers
        def reset():
            b['observation_images_normalized'].copy_(images)
            b['observation_state_normalized'].copy_(state)
            b['diffusion_noise'].copy_(noise)
        if self.length not in self.parity_lengths:
            ordinary = self.runtime.forward(images, state, noise).clone()
            reset()
            self.record((), 10)
            torch.cuda.synchronize()
            if not torch.equal(ordinary, b['diffusion_noise']):
                raise RuntimeError('Custom decoder D10 parity failed')
            self.parity_lengths.add(self.length)
        if key not in self.graphs:
            # Each CUDA graph owns its allocator pool. Do not retain one pool
            # for every mask in a thousand-candidate search.
            if len(self.graphs) >= 10:
                self.graphs.popitem(last=False)
            for _ in range(3):
                reset()
                self.record(skip, steps)
            reset()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.record(skip, steps)
            self.graphs[key] = graph
        self.graphs.move_to_end(key)
        reset()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started = time.monotonic()
        start.record()
        self.graphs[key].replay()
        end.record()
        end.synchronize()
        cuda_ms = start.elapsed_time(end)
        raw = b['diffusion_noise'].float().cpu().numpy().copy()
        commands = np.asarray(self.output_transform({'state': np.asarray(self.transformed['state']).copy(),
                                  'actions': raw.copy()})['actions'], dtype=np.float32)
        timing = {'cuda_policy_ms': float(cuda_ms), 'warm_call_ms': (time.monotonic() - started) * 1000,
                  'timing_is_dedicated_benchmark': False, 'steps': steps, 'skip_layers': skip}
        return raw, commands, timing
