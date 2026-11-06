"""Experimental two-stage Prefill graphs with stream Event selective overlap.

No AE dependency on new KV: vision_only delays VLM until OLD-KV AE ends;
vlm_only delays AE until vision ends. The all variant measures split overhead.
All waits are enqueued on GPU streams. Pool publication still waits for the
complete prefix. No full-cache staging copy is allocated.
"""
import gc
import math
import time

import torch

from .openpi_backend import Prefix, Prepared, cache_tensors
from .runtime_graph import DynamicCore, GraphSlot, RuntimeGraphs, input_tensors, layout, stage_tensor


@torch.no_grad()
def embeddings(core, prepared):
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    model = core.model
    masks = torch.cat([*(m[:, None].expand(m.shape[0], core.image_tokens)
                        for m in prepared.image_masks), prepared.token_masks], dim=1)
    positions = torch.cumsum(masks, dim=1)-1
    attention = model._prepare_attention_masks_4d(make_att_2d_masks(masks, torch.zeros_like(masks)))
    images = [model.paligemma_with_expert.embed_image(x) for x in prepared.images]
    language = model.paligemma_with_expert.embed_language_tokens(prepared.tokens)
    return torch.cat([*images, language*math.sqrt(language.shape[-1])], dim=1), masks, positions, attention


@torch.no_grad()
def language_prefix(core, values):
    embs, masks, positions, attention = values
    _, cache = core.model.paligemma_with_expert.forward(attention_mask=attention,
        position_ids=positions, past_key_values=None, inputs_embeds=[embs, None], use_cache=True)
    return Prefix(cache, masks, positions)


class SelectiveGraphs(RuntimeGraphs):
    def reset_events(self):
        # Called only after Pool has drained all recorded reader/producer events.
        self.vision_pending = self.ae_pending = False

    def __init__(self, backend, prepared, noise, prefix_stream, ae_stream):
        self.backend = backend
        self.signature = layout(prepared, noise)
        self.prefix_stream, self.ae_stream = prefix_stream, ae_stream
        self.kind = backend.selective_overlap
        if self.kind not in ("all", "vision_only", "vlm_only"):
            raise ValueError("Unknown selective overlap")
        self.slots = []
        self.vision_done, self.ae_done = torch.cuda.Event(), torch.cuda.Event()
        self.vision_pending = self.ae_pending = False
        begin = time.perf_counter()
        ready = torch.cuda.Event()
        ready.record()
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
                    for _ in range(2):
                        temporary = embeddings(core, static)
                        prefix_stream.synchronize()
                        del temporary
                    vision_graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(vision_graph, stream=prefix_stream):
                        values = embeddings(core, static)
                    vision_graph.replay()
                    for _ in range(2):
                        temporary = language_prefix(core, values)
                        prefix_stream.synchronize()
                        del temporary
                    gc.collect()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=prefix_stream):
                        prefix = language_prefix(core, values)
                    graph.replay()
                prefix_stream.synchronize()
                slot = GraphSlot(static, static_noise, core, graph, prefix)
                slot.vision_graph, slot.embedding_values = vision_graph, values
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
            assert not addresses[0] & addresses[1]
            pools = [g.pool() for s in self.slots for g in (s.vision_graph, s.prefix_graph, s.ae_graph)]
            assert len(set(pools)) == 6
            self.addresses = [tuple(x.data_ptr() for x in cache_tensors(s.prefix.cache)) for s in self.slots]
            self.setup_ms = (time.perf_counter()-begin)*1000
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def build_prefix(self, pool, identity, prepared):
        owner = pool._slot(identity)
        if owner.state != "BUILDING" or owner.ready is not None or owner.leases or owner.readers:
            raise RuntimeError("Selective prefix requires exclusive Pool reservation")
        if torch.cuda.current_stream(self.backend.device) != self.prefix_stream:
            raise RuntimeError("Selective prefix requires dedicated stream")
        slot = self.slots[identity.slot]
        for target, source in zip(input_tensors(slot.prepared), input_tensors(prepared), strict=True):
            stage_tensor(target, source)
        with torch.cuda.nvtx.range("prefill/vision"):
            slot.vision_graph.replay()
        self.vision_done.record(self.prefix_stream)
        self.vision_pending = True
        if self.kind == "vision_only" and self.ae_pending:
            self.prefix_stream.wait_event(self.ae_done)
            self.ae_pending = False
        with torch.cuda.nvtx.range("prefill/vlm"):
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
        if self.kind == "vlm_only" and self.vision_pending:
            self.ae_stream.wait_event(self.vision_done)
            self.vision_pending = False
        actions = super().generate(pool, lease, prepared, noise)
        self.ae_done.record(self.ae_stream)
        self.ae_pending = True
        return actions
