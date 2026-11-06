"""Fixed-input pi0/pi05 wrapper hoisting metadata before CUDA capture.

No original model method is changed. Every neural image/language/state/action/
time operation and the configured CUDA FP32 Euler updates are preserved.
Each configured step count is an independent matched-computation experiment.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F

from .openpi_backend import Prefix,cache_tensors,error


@dataclass
class PrefixMetadata:
    masks: torch.Tensor
    positions: torch.Tensor
    attention: torch.Tensor


def digest_cache(prefix):
    digest = hashlib.sha256()
    for tensor in cache_tensors(prefix.cache):
        digest.update(str((tuple(tensor.shape),tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class GraphSafeBackend:
    def __init__(self,backend,current,next_observation):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        self.original,self.model,self.device = backend,backend.model,backend.device
        self.num_steps = backend.num_steps
        if self.num_steps not in (1, 2, 5, 10):
            raise ValueError("Supported matched denoising strata are N=1, 2, 5, or 10")
        self.current,self.next = current,next_observation
        self.image_tokens = int(self.model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.num_patches)
        self.prefix_metadata = {}
        # These scalar/list constructions run once, before any capture starts.
        self.dt = torch.tensor(-1.0/self.num_steps,dtype=torch.float32,device=self.device)
        self.initial_time = torch.tensor(1.0,dtype=torch.float32,device=self.device)
        for prepared in (current,next_observation):
            masks = torch.cat([*(mask[:,None].expand(mask.shape[0],self.image_tokens) for mask in prepared.image_masks),prepared.token_masks],dim=1)
            att = torch.zeros_like(masks,dtype=torch.bool)
            attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(masks,att))
            self.prefix_metadata[id(prepared)] = PrefixMetadata(masks,torch.cumsum(masks,dim=1)-1,attention)
        masks = self.prefix_metadata[id(current)].masks
        batch = masks.shape[0]
        horizon = self.model.config.action_horizon
        suffix_len = horizon + (not self.model.pi05)
        suffix_masks = torch.ones(batch,suffix_len,dtype=torch.bool,device=self.device)
        pattern = ([1] if not self.model.pi05 else [])+[1]+[0]*(horizon-1)
        suffix_att = torch.tensor(pattern,dtype=self.model.action_in_proj.weight.dtype,device=self.device)[None].expand(batch,-1)
        prefix_part = masks[:,None,:].expand(batch,suffix_len,masks.shape[1])
        suffix_part = make_att_2d_masks(suffix_masks,suffix_att)
        self.suffix_attention = self.model._prepare_attention_masks_4d(torch.cat([prefix_part,suffix_part],dim=2))
        self.suffix_positions = torch.sum(masks,dim=-1)[:,None]+torch.cumsum(suffix_masks,dim=1)-1

    @property
    def identity(self):
        return dict(self.original.identity,static_metadata="dt/time templates, full prefix/suffix masks and positions hoisted before capture; NN and configured Euler recurrence unchanged")

    def noise(self,seed):
        return self.original.noise(seed)

    def output(self,prepared,actions):
        return self.original.output(prepared,actions)

    @torch.no_grad()
    def build_prefix(self,prepared):
        metadata = self.prefix_metadata[id(prepared)]
        embeddings = []
        for image in prepared.images:
            value = self.model.paligemma_with_expert.embed_image(image)
            if value.shape[1]!=self.image_tokens:
                raise ValueError("Vision token layout differs from frozen metadata")
            embeddings.append(value)
        language = self.model.paligemma_with_expert.embed_language_tokens(prepared.tokens)
        embeddings.append(language*math.sqrt(language.shape[-1]))
        embs = torch.cat(embeddings,dim=1)
        _,cache = self.model.paligemma_with_expert.forward(attention_mask=metadata.attention,
            position_ids=metadata.positions,past_key_values=None,inputs_embeds=[embs,None],use_cache=True)
        return Prefix(cache,metadata.masks,metadata.positions)

    def _suffix(self,prepared,actions,timestep):
        from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
        if self.model.pi05:
            # Original PI0Pytorch.embed_suffix pi05 neural path, without its
            # capture-time host construction of the fixed attention pattern.
            time_emb = create_sinusoidal_pos_embedding(timestep,self.model.action_in_proj.out_features,
                                                       min_period=4e-3,max_period=4.0,device=timestep.device)
            time_emb = time_emb.type(dtype=timestep.dtype)
            action_emb = self.model.action_in_proj(actions)
            time_emb = F.silu(self.model.time_mlp_in(time_emb))
            time_emb = F.silu(self.model.time_mlp_out(time_emb))
            return action_emb, time_emb
        state = prepared.state.to(torch.float32) if self.model.state_proj.weight.dtype==torch.float32 else prepared.state
        state_emb = self.model.state_proj(state)
        time_emb = create_sinusoidal_pos_embedding(timestep,self.model.action_in_proj.out_features,
                                                   min_period=4e-3,max_period=4.0,device=timestep.device)
        time_emb = time_emb.type(dtype=timestep.dtype)
        action_emb = self.model.action_in_proj(actions)
        time_emb = time_emb[:,None,:].expand_as(action_emb)
        fused = torch.cat([action_emb,time_emb],dim=2)
        fused = self.model.action_time_mlp_out(F.silu(self.model.action_time_mlp_in(fused)))
        return torch.cat([state_emb[:,None,:],fused],dim=1)

    @torch.no_grad()
    def velocity(self,prefix,prepared,actions,timestep):
        if prepared is not self.current:
            raise ValueError("Only the fixed current suffix input belongs to this wrapper")
        suffix = self._suffix(prepared,actions,timestep)
        adarms_cond = None
        if self.model.pi05:
            suffix, adarms_cond = suffix
        outputs,_ = self.model.paligemma_with_expert.forward(attention_mask=self.suffix_attention,
            position_ids=self.suffix_positions,past_key_values=prefix.cache,inputs_embeds=[None,suffix],
            use_cache=False,adarms_cond=[None,adarms_cond])
        result = outputs[1][:,-self.model.config.action_horizon:].to(dtype=torch.float32)
        return self.model.action_out_proj(result)

    @torch.no_grad()
    def generate(self,prefix,prepared,noise,*,time_nodes=None,velocities=None):
        time = self.initial_time.clone()  # D2D reset, never a capture-time H2D scalar.
        x = noise
        for _ in range(self.num_steps):
            if time_nodes is not None:
                time_nodes.append(time.clone())
            velocity = self.velocity(prefix,prepared,x,time.expand(noise.shape[0]))
            if velocities is not None:
                velocities.append(velocity.clone())
            x = x+self.dt*velocity
            time += self.dt
        return x.clone()


def validate(backend,safe,current,next_observation,noise):
    """Original vs hoisted metadata: <=two complete cache snapshots at any time."""
    original = backend.build_prefix(current)
    selected = safe.build_prefix(current)
    original_hash,selected_hash = digest_cache(original),digest_cache(selected)
    original_times,original_velocities = [],[]
    saved_step = backend.model.denoise_step
    def audited_step(*args,**kwargs):
        original_times.append(args[-1].clone())
        value = saved_step(*args,**kwargs)
        original_velocities.append(value.clone())
        return value
    backend.model.denoise_step = audited_step
    try:
        reference = backend.generate(original,current,noise)
    finally:
        backend.model.denoise_step = saved_step
    times,velocities = [],[]
    actual = safe.generate(selected,current,noise,time_nodes=times,velocities=velocities)
    raw_error = error(reference.cpu(),actual.cpu())
    decoded_error = error(backend.output(current,reference),safe.output(current,actual))
    velocity_error = error(torch.stack(original_velocities).cpu(),torch.stack(velocities).cpu())
    time_equal = len(times)==len(original_times)==backend.num_steps and all(torch.equal(left.reshape(-1),right.reshape(-1)) for left,right in zip(times,original_times,strict=True))
    result = dict(normalized_error=raw_error,decoded_error=decoded_error,velocity_error=velocity_error,
                  original_nfe=len(original_times),graph_safe_nfe=len(times),
                  original_time_nodes=[float(value.reshape(-1)[0].cpu()) for value in original_times],
                  safe_time_nodes=[float(value.cpu()) for value in times],time_nodes_exact=time_equal,
                  prefix_kv_equal=original_hash==selected_hash,
                  old_original_kv_unchanged=digest_cache(original)==original_hash,
                  old_safe_kv_unchanged=digest_cache(selected)==selected_hash,
                  prefix_masks_equal=bool(torch.equal(original.pad_masks,selected.pad_masks)),
                  prefix_positions_equal=bool(torch.equal(original.position_ids,selected.position_ids)),atol=1e-5)
    del original,selected,reference,actual,original_velocities,velocities,original_times,times
    # Current two caches are released before the next pair is built.
    original = backend.build_prefix(next_observation)
    selected = safe.build_prefix(next_observation)
    result["next_prefix_kv_equal"] = digest_cache(original)==digest_cache(selected)
    del original,selected
    result["passed"] = (all(value["max_abs"]<=1e-5 for value in (raw_error,decoded_error,velocity_error))
                         and all(result[name] for name in ("time_nodes_exact","prefix_kv_equal","next_prefix_kv_equal",
                                 "old_original_kv_unchanged","old_safe_kv_unchanged","prefix_masks_equal","prefix_positions_equal")))
    return result


def cpu_self_test(*, pi05=False, num_steps=10):
    """Checkpoint-free exact comparison using original OpenPI embedding/step methods."""
    from types import SimpleNamespace,MethodType
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch,make_att_2d_masks
    from .openpi_backend import OpenPIBackend,Prepared
    class TinyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.words = torch.nn.Embedding(16,4)
            self.paligemma = SimpleNamespace(model=SimpleNamespace(vision_tower=SimpleNamespace(
                vision_model=SimpleNamespace(embeddings=SimpleNamespace(num_patches=2)))),
                language_model=SimpleNamespace(config=SimpleNamespace()))
            self.gemma_expert = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
        def embed_image(self,image):
            return image
        def embed_language_tokens(self,tokens):
            return self.words(tokens)
        def forward(self,*,inputs_embeds,past_key_values=None,adarms_cond=None,**unused):
            prefix,suffix = inputs_embeds
            if prefix is not None:
                return [prefix,None],[(prefix.clone(),prefix.clone())]
            value = suffix+past_key_values[0][0].mean(dim=1)[:,None,:]
            if adarms_cond is not None and adarms_cond[1] is not None:
                value = value+adarms_cond[1][:,None,:]
            return [None,value],None
    class TinyModel(PI0Pytorch):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(action_horizon=3,action_dim=2)
            self.pi05 = pi05
            self.gradient_checkpointing_enabled = False
            self.paligemma_with_expert = TinyBackbone()
            self.action_in_proj = torch.nn.Linear(2,4)
            self.action_out_proj = torch.nn.Linear(4,2)
            if pi05:
                self.time_mlp_in = torch.nn.Linear(4,4)
                self.time_mlp_out = torch.nn.Linear(4,4)
            else:
                self.state_proj = torch.nn.Linear(2,4)
                self.action_time_mlp_in = torch.nn.Linear(8,4)
                self.action_time_mlp_out = torch.nn.Linear(4,4)
    model = TinyModel().eval()
    original = SimpleNamespace(model=model,device=torch.device("cpu"),num_steps=num_steps)
    original.velocity = MethodType(OpenPIBackend.velocity,original)
    original.generate = MethodType(OpenPIBackend.generate,original)
    def build(prepared):
        with torch.no_grad():
            emb,pad,att = model.embed_prefix(prepared.images,prepared.image_masks,prepared.tokens,prepared.token_masks)
            positions = torch.cumsum(pad,dim=1)-1
            _,cache = model.paligemma_with_expert.forward(inputs_embeds=[emb,None],past_key_values=None,
                position_ids=positions,attention_mask=model._prepare_attention_masks_4d(make_att_2d_masks(pad,att)),use_cache=True)
            return Prefix(cache,pad,positions)
    original.build_prefix = build
    original.output = lambda prepared,actions:actions[0].detach().cpu().numpy()
    current = Prepared(None,[torch.randn(1,2,4)],[torch.ones(1,dtype=torch.bool)],
                       torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.bool),torch.randn(1,2))
    next_observation = Prepared(None,[torch.randn(1,2,4)],[torch.ones(1,dtype=torch.bool)],
                               torch.tensor([[3,4]]),torch.ones(1,2,dtype=torch.bool),torch.randn(1,2))
    safe = GraphSafeBackend(original,current,next_observation)
    result = validate(original,safe,current,next_observation,torch.randn(1,3,2))
    if not result["passed"]:
        raise AssertionError(result)
    return result


if __name__=="__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-self-test",action="store_true",required=True)
    parser.parse_args()
    print(json.dumps(cpu_self_test(),indent=2))
