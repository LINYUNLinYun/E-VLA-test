"""Original OpenPI eager prefix split, with the original CUDA Euler recurrence."""
import copy
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Prepared:
    observation: object
    images: list
    image_masks: list
    tokens: torch.Tensor
    token_masks: torch.Tensor
    state: torch.Tensor


@dataclass
class Prefix:
    cache: object
    pad_masks: torch.Tensor
    position_ids: torch.Tensor


def cache_tensors(cache):
    if hasattr(cache, "key_cache"):
        return tuple(x for pair in zip(cache.key_cache, cache.value_cache, strict=True) for x in pair)
    if hasattr(cache, "layers"):
        return tuple(x for layer in cache.layers for x in (layer.keys, layer.values))
    return tuple(x for pair in cache for x in pair)


def error(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    delta = a - b
    return dict(max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta ** 2))),
                reference_abs_max=float(np.abs(a).max()))


class OpenPIBackend:
    def __init__(self, policy, num_steps=10):
        if not policy._is_pytorch_model:
            raise ValueError("A native full OpenPI PyTorch policy is required")
        self.policy, self.model = policy, policy._model
        self.device = torch.device(policy._pytorch_device)
        self.num_steps = num_steps
        if num_steps != 10:
            raise ValueError("This first experiment freezes original N=10")
        if self.model.config.pytorch_compile_mode is not None:
            raise ValueError("First experiment requires eager model config")
        self.model.eval()
        # Both original call sites assign this same immutable value. Set it once
        # before either stream starts; original denoise_step is reused unchanged.
        self.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        self.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

    @torch.no_grad()
    def prepare(self, raw):
        import jax
        from openpi.models.model import Observation
        data = self.policy._input_transform(copy.deepcopy(raw))
        data = jax.tree.map(lambda x: torch.from_numpy(np.array(x, copy=True)).to(self.device)[None], data)
        observation = Observation.from_dict(data)
        values = self.model._preprocess_observation(observation, train=False)
        return Prepared(observation, *values)

    @torch.no_grad()
    def build_prefix(self, prepared):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        stream = torch.cuda.current_stream(self.device)
        for tensor in [*prepared.images, *prepared.image_masks, prepared.tokens, prepared.token_masks]:
            tensor.record_stream(stream)
        embs, masks, att = self.model.embed_prefix(prepared.images, prepared.image_masks,
                                                  prepared.tokens, prepared.token_masks)
        positions = torch.cumsum(masks, dim=1) - 1
        attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(masks, att))
        _, cache = self.model.paligemma_with_expert.forward(
            attention_mask=attention, position_ids=positions, past_key_values=None,
            inputs_embeds=[embs, None], use_cache=True)
        return Prefix(cache, masks, positions)

    @torch.no_grad()
    def velocity(self, prefix, prepared, noise, timestep):
        return self.model.denoise_step(prepared.state, prefix.pad_masks, prefix.cache, noise, timestep)

    @torch.no_grad()
    def generate(self, prefix, prepared, noise):
        # Keep the CUDA scalar dtype, in-place time recurrence, dt and x update
        # from sample_actions. E0 audits exactly N iterations against its while.
        if self.device.type == "cuda":
            stream = torch.cuda.current_stream(self.device)
            for tensor in (*cache_tensors(prefix.cache), prefix.pad_masks, prepared.state, noise):
                tensor.record_stream(stream)
        dt = torch.tensor(-1.0 / self.num_steps, dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        x = noise
        for _ in range(self.num_steps):
            v = self.velocity(prefix, prepared, x, time.expand(noise.shape[0]))
            x = x + dt * v
            time += dt
        return x.clone()  # Stable controller-owned output, included in timing.

    def original(self, prepared, noise):
        return self.model.sample_actions(self.device, prepared.observation, noise=noise, num_steps=self.num_steps)

    def noise(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn((1, self.model.config.action_horizon, self.model.config.action_dim),
                           generator=generator, device=self.device, dtype=torch.float32)

    def output(self, prepared, actions):
        return self.policy._output_transform(dict(state=prepared.state[0].detach().cpu().numpy(),
                                                 actions=actions[0].detach().cpu().numpy()))["actions"]

    @property
    def identity(self):
        import transformers
        return dict(torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
                    device=str(self.device), gpu=torch.cuda.get_device_name(self.device),
                    config=repr(self.model.config), attention="eager", execution="eager",
                    num_steps=self.num_steps, state_path="prefix" if getattr(self.model.config, "discrete_state_input", False)
                    else ("none" if self.model.pi05 else "suffix"))
