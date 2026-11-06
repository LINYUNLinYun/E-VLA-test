"""Two-slot dynamic-input Graph dispatch; Pool remains the ownership authority.

Each slot owns one captured prefix (the only full KV allocation for that slot),
an independent prefill graph pool and an independent AE graph pool. AE reads
that slot's KV directly. Inputs are staged on the replay stream after the
producer event; only Pool-authorized free slots may be overwritten.
"""
from dataclasses import dataclass
import gc
import time

import torch

from .graph_sampler import GraphSafeBackend, PrefixMetadata
from .openpi_backend import Prepared, cache_tensors


def input_tensors(prepared):
    return [*prepared.images, *prepared.image_masks, prepared.tokens,
            prepared.token_masks, prepared.state]


def layout(prepared, noise):
    return (len(prepared.images), len(prepared.image_masks),
            tuple((tuple(x.shape), x.dtype, x.device) for x in [*input_tensors(prepared), noise]))


def stage_tensor(target, source):
    if target.shape != source.shape or target.dtype != source.dtype or target.device != source.device:
        raise ValueError("Graph input layout changed without rebuilding captures")
    source.record_stream(torch.cuda.current_stream(source.device))
    target.copy_(source, non_blocking=True)


class DynamicCore(GraphSafeBackend):
    """Same validated neural/FP32 Euler path, with masks recomputed on GPU.

Only the shapes and suffix block-attention pattern are capture constants.
Images, tokens, pad masks, state, noise and the selected old KV are dynamic.
"""
    def __init__(self, backend, prepared):
        super().__init__(backend, prepared, prepared)
        batch = prepared.state.shape[0]
        horizon = self.model.config.action_horizon
        suffix_len = horizon + (not self.model.pi05)
        self.suffix_masks = torch.ones(batch, suffix_len, dtype=torch.bool, device=self.device)
        pattern = ([1] if not self.model.pi05 else [])+[1]+[0]*(horizon-1)
        self.suffix_pattern = torch.tensor(pattern,
            dtype=self.model.action_in_proj.weight.dtype, device=self.device)[None].expand(batch, -1)

    @torch.no_grad()
    def build_prefix(self, prepared):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        masks = torch.cat([*(mask[:, None].expand(mask.shape[0], self.image_tokens)
                            for mask in prepared.image_masks), prepared.token_masks], dim=1)
        attention = self.model._prepare_attention_masks_4d(make_att_2d_masks(masks, torch.zeros_like(masks)))
        self.prefix_metadata[id(prepared)] = PrefixMetadata(masks, torch.cumsum(masks, dim=1)-1, attention)
        return super().build_prefix(prepared)

    @torch.no_grad()
    def generate(self, prefix, prepared, noise, *, time_nodes=None, velocities=None):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        masks = prefix.pad_masks
        suffix = make_att_2d_masks(self.suffix_masks, self.suffix_pattern)
        prefix_part = masks[:, None, :].expand(masks.shape[0], self.suffix_masks.shape[1], masks.shape[1])
        self.suffix_attention = self.model._prepare_attention_masks_4d(torch.cat([prefix_part, suffix], dim=2))
        self.suffix_positions = torch.sum(masks, dim=-1)[:, None]+torch.cumsum(self.suffix_masks, dim=1)-1
        return super().generate(prefix, prepared, noise, time_nodes=time_nodes, velocities=velocities)


@dataclass
class GraphSlot:
    prepared: Prepared
    noise: torch.Tensor
    core: DynamicCore
    prefix_graph: object
    prefix: object
    ae_graph: object = None
    actions: object = None


class RuntimeGraphs:
    def __init__(self, backend, prepared, noise, prefix_stream, ae_stream):
        self.backend = backend
        self.signature = layout(prepared, noise)
        self.prefix_stream, self.ae_stream = prefix_stream, ae_stream
        self.slots = []
        begin = time.perf_counter()
        # Cold setup only, before any Pool reservation or lease exists.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(backend.device))
        prefix_stream.wait_event(ready)
        ae_stream.wait_event(ready)
        try:
            for _ in range(2):
                with torch.cuda.stream(prefix_stream), torch.no_grad():
                    static = Prepared(None, [x.clone() for x in prepared.images],
                        [x.clone() for x in prepared.image_masks], prepared.tokens.clone(),
                        prepared.token_masks.clone(), prepared.state.clone())
                    static_noise = noise.clone()
                    core = getattr(backend, "graph_core_class", DynamicCore)(backend, static)
                    # Release temporary KV before capture; never a third snapshot.
                    for _ in range(2):
                        temporary = core.build_prefix(static)
                        prefix_stream.synchronize()
                        del temporary
                    gc.collect()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=prefix_stream):
                        prefix = core.build_prefix(static)
                    graph.replay()
                prefix_stream.synchronize()
                slot = GraphSlot(static, static_noise, core, graph, prefix)
                self.slots.append(slot)
                with torch.cuda.stream(ae_stream), torch.no_grad():
                    for _ in range(2):
                        temporary = core.generate(prefix, static, static_noise)
                        ae_stream.synchronize()
                        del temporary
                    slot.ae_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(slot.ae_graph, stream=ae_stream):
                        slot.actions = core.generate(prefix, static, static_noise)
                    slot.ae_graph.replay()
                ae_stream.synchronize()
            addresses = [{x.data_ptr() for x in cache_tensors(s.prefix.cache)} for s in self.slots]
            if addresses[0] & addresses[1]:
                raise RuntimeError("Graph KV slots alias")
            pools = [g.pool() for s in self.slots for g in (s.prefix_graph, s.ae_graph)]
            if len(set(pools)) != 4:
                raise RuntimeError("Concurrent captures must use independent graph pools")
            self.addresses = [tuple(x.data_ptr() for x in cache_tensors(s.prefix.cache)) for s in self.slots]
            self.setup_ms = (time.perf_counter()-begin)*1000
        except BaseException:
            self.close()
            raise

    def compatible(self, prepared, noise):
        return self.signature == layout(prepared, noise)

    @torch.no_grad()
    def build_prefix(self, pool, identity, prepared):
        owner = pool._slot(identity)
        if owner.state != "BUILDING" or owner.ready is not None or owner.leases or owner.readers:
            raise RuntimeError("Graph prefix overwrite requires an exclusive Pool reservation")
        slot = self.slots[identity.slot]
        if torch.cuda.current_stream(self.backend.device) != self.prefix_stream:
            raise RuntimeError("Prefill graph must use its dedicated stream")
        for target, source in zip(input_tensors(slot.prepared), input_tensors(prepared), strict=True):
            stage_tensor(target, source)
        slot.prefix_graph.replay()
        return slot.prefix

    @torch.no_grad()
    def generate(self, pool, lease, prepared, noise):
        owner = pool._slot(lease.identity)
        slot = self.slots[lease.identity.slot]
        if lease.released or owner.leases < 1 or lease.value is not slot.prefix or owner.value is not slot.prefix:
            raise RuntimeError("Graph AE requires the live lease for its captured KV slot")
        if torch.cuda.current_stream(self.backend.device) != self.ae_stream:
            raise RuntimeError("AE graph must use its dedicated stream")
        stage_tensor(slot.prepared.state, prepared.state)
        stage_tensor(slot.noise, noise)
        slot.ae_graph.replay()
        # Graph output is reusable workspace; controller output must own storage.
        return slot.actions.clone()

    def close(self):
        self.prefix_stream.synchronize()
        self.ae_stream.synchronize()
        self.slots.clear()
