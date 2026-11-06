"""Opt-in D10 mixed-layer KV experiment; full model math and two KV slots.

Each Euler step freezes an acquire-published new-KV frontier on GPU. Completed
layers use new KV, remaining layers use the previous complete snapshot. ONE
layer scratch is reused by successive attentions, never a third full cache.
The sole Host owns all Pool operations. Old leases explicitly permit only KV
reads, allowing the independent old-slot Vision workspace to prepare next input.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import gc
import time

import torch

from .cewp_preparation import PreparedModulationCore
from .cewp_segmented import embeddings, language_prefix
from .external_events import ExternalEvents, initialized_events
from .layer_graphs import capture_labels
from .openpi_backend import Prefix, Prepared, cache_tensors
from .progressive_kv_kernels import publish, snapshot, select_layer
from .runtime_graph import input_tensors, stage_tensor
from .scheduler import Observation


class MixedCache:
    def __init__(self, old, new, scratch, frontiers):
        self.old, self.new, self.scratch, self.frontiers = old, new, scratch, frontiers
        self.step, self.last_layer = 0, -1

    def __getattr__(self, name):
        return getattr(self.new, name)

    def __len__(self):
        return len(self.new)

    def __getitem__(self, layer):
        # Original Gemma reads [layer][0] and [layer][1] separately.
        if layer != self.last_layer:
            select_layer(self.old[layer], self.new[layer], self.scratch,
                         self.frontiers, self.step, layer)
            self.last_layer = layer
        return self.scratch

    def update(self, *args, **kwargs):
        raise RuntimeError("AE is a read-only KV consumer")


@contextmanager
def publishing_layers(slot, external, stream):
    from transformers.cache_utils import DynamicCache
    original = DynamicCache.update
    calls = []

    def update(cache, key, value, layer_idx, cache_kwargs=None):
        result = original(cache, key, value, layer_idx, cache_kwargs)
        publish(slot.frontier, layer_idx + 1)
        external.mark(slot.layer_ready[layer_idx], stream)
        calls.append(layer_idx)
        return result

    try:
        DynamicCache.update = update
        yield
        if calls != list(range(len(slot.layer_ready))):
            raise RuntimeError("Producer did not publish each layer in order")
    finally:
        DynamicCache.update = original


@dataclass
class ProgressiveSlot:
    prepared: object
    ae_prepared: object
    noise: object
    core: object
    vision_graph: object
    values: object
    metadata_ready: object
    layer_ready: tuple
    frontier: object
    frontiers: object
    override: object
    prefix_graph: object = None
    prefix: object = None
    ae_graph: object = None
    actions: object = None
    scratch: object = None
    iteration_marks: tuple = ()


class ProgressiveGraphs:
    def __init__(self, backend, prepared, noise, *, profile=False, streams=None):
        self.backend, self.profile = backend, profile
        if streams is None:
            streams = tuple(torch.cuda.Stream(device=backend.device) for _ in range(3))
        if len(streams) != 3:
            raise ValueError("Expected Vision, Prefix and AE streams")
        self.vision_stream, self.prefix_stream, self.ae_stream = streams
        self.external = ExternalEvents()
        self.slots = []
        self.steps = backend.num_steps
        self.num_layers = len(backend.model.paligemma_with_expert.paligemma.language_model.layers)
        self.policy_constants = {v: torch.tensor(v, device=backend.device, dtype=torch.int32) for v in (-1, 0)}
        begin = time.perf_counter()
        initial = torch.cuda.Event()
        initial.record()
        for stream in self.streams:
            stream.wait_event(initial)
        try:
            for _ in range(2):
                self.slots.append(self._capture_prefix(prepared, noise))
            self.kv_addresses = [tuple(x.data_ptr() for x in cache_tensors(s.prefix.cache)) for s in self.slots]
            if set(self.kv_addresses[0]) & set(self.kv_addresses[1]):
                raise RuntimeError("The two KV slots alias")
            self.shared_scratch = None
            for index in range(2):
                self._capture_ae(index)
            pools = [g.pool() for s in self.slots for g in (s.vision_graph, s.prefix_graph, s.ae_graph)]
            if len(set(pools)) != len(pools):
                raise RuntimeError("Concurrent graphs require private pools")
            self.setup_ms = (time.perf_counter() - begin) * 1000
        except BaseException:
            self.close()
            raise

    @property
    def streams(self):
        return (self.vision_stream, self.prefix_stream, self.ae_stream)

    @torch.no_grad()
    def _capture_prefix(self, prepared, noise):
        vs, ps = self.vision_stream, self.prefix_stream
        with torch.cuda.stream(vs):
            static = Prepared(None, [x.clone() for x in prepared.images],
                              [x.clone() for x in prepared.image_masks], prepared.tokens.clone(),
                              prepared.token_masks.clone(), prepared.state.clone())
            ae = Prepared(None, static.images, static.image_masks, static.tokens,
                          static.token_masks, prepared.state.clone())
            core = PreparedModulationCore(self.backend, ae)
            static_noise = noise.clone()
            ready = tuple(initialized_events(vs, self.num_layers))
            metadata, = initialized_events(vs, 1)
            frontier = torch.zeros((), device=self.backend.device, dtype=torch.int32)
            frontiers = torch.zeros(self.steps, device=self.backend.device, dtype=torch.int32)
            override = torch.full((), -1, device=self.backend.device, dtype=torch.int32)
            for _ in range(2):
                temporary = embeddings(core, static)
                vs.synchronize()
                del temporary
            vision = torch.cuda.CUDAGraph()
            labels = capture_labels(core.model, "vision") if self.profile else nullcontext()
            with labels, torch.cuda.graph(vision, stream=vs):
                with torch.cuda.nvtx.range("Prefill/VisionEncoder+Projector+Metadata"):
                    values = embeddings(core, static)
            vision.replay()
            metadata.record(vs)
        ps.wait_event(metadata)
        with torch.cuda.stream(ps):
            for _ in range(2):
                temporary = language_prefix(core, values)
                ps.synchronize()
                del temporary
            for layer in range(self.num_layers):
                publish(frontier, layer + 1)
            ps.synchronize()
            gc.collect()
            slot = ProgressiveSlot(static, ae, static_noise, core, vision, values,
                                   metadata, ready, frontier, frontiers, override)
            slot.prefix_graph = torch.cuda.CUDAGraph()
            labels = capture_labels(core.model, "vlm") if self.profile else nullcontext()
            with labels, publishing_layers(slot, self.external, ps), torch.cuda.graph(slot.prefix_graph, stream=ps):
                with torch.cuda.nvtx.range("Prefill/VLM[release_KV_frontier]"):
                    slot.prefix = language_prefix(core, values)
            slot.prefix_graph.replay()
        ps.synchronize()
        return slot

    @torch.no_grad()
    def _capture_ae(self, index):
        slot, old = self.slots[index], self.slots[1-index]
        stream = self.ae_stream
        with torch.cuda.stream(stream):
            shapes = {tuple(x.shape) for s in self.slots for x in cache_tensors(s.prefix.cache)}
            if len(shapes) != 1 or not all(x.is_contiguous() for s in self.slots for x in cache_tensors(s.prefix.cache)):
                raise RuntimeError("Progressive selection requires uniform contiguous layer KV shapes")
            if self.shared_scratch is None:
                self.shared_scratch = tuple(torch.empty_like(x) for x in slot.prefix.cache[0])
            # Both AE graphs execute on ONE stream, so one layer scratch is
            # safe across their generations and remains outside private pools.
            slot.scratch = self.shared_scratch
            slot.iteration_marks = tuple(initialized_events(stream, self.steps + 1))
            mixed = MixedCache(old.prefix.cache, slot.prefix.cache, slot.scratch, slot.frontiers)
            selected = Prefix(mixed, slot.prefix.pad_masks, slot.prefix.position_ids)
            original_velocity = slot.core.velocity

            def run(capturing=False):
                count = []
                def mark(event):
                    if capturing:
                        self.external.mark(event, stream)
                    else:
                        event.record(stream)
                def velocity(prefix, prepared, actions, timestep):
                    step = len(count)
                    count.append(step)
                    mark(slot.iteration_marks[step])
                    snapshot(slot.frontier, slot.frontiers, step, override=slot.override)
                    mixed.step, mixed.last_layer = step, -1
                    with torch.cuda.nvtx.range(f"AE/iteration_{step+1}/mixed_KV"):
                        return original_velocity(selected, prepared, actions, timestep)
                slot.core.velocity = velocity
                try:
                    actions = slot.core.generate(selected, slot.ae_prepared, slot.noise)
                    mark(slot.iteration_marks[-1])
                finally:
                    slot.core.velocity = original_velocity
                if count != list(range(self.steps)):
                    raise RuntimeError("AE did not execute original D steps")
                return actions
            for _ in range(2):
                temporary = run()
                stream.synchronize()
                del temporary
            slot.ae_graph = torch.cuda.CUDAGraph()
            labels = capture_labels(slot.core.model, "ae") if self.profile else nullcontext()
            with labels, torch.cuda.graph(slot.ae_graph, stream=stream):
                with torch.cuda.nvtx.range("AE/D10[per_round_frozen_frontier]"):
                    slot.actions = run(capturing=True)
            slot.ae_graph.replay()
        stream.synchronize()

    @torch.no_grad()
    def encode(self, index, prepared, *, dependency=None, metadata_event=None):
        slot = self.slots[index]
        input_ready = torch.cuda.Event()
        input_ready.record()
        with torch.cuda.stream(self.vision_stream):
            self.vision_stream.wait_event(input_ready)
            if dependency is not None:
                for event in dependency if isinstance(dependency, (tuple, list)) else (dependency,):
                    self.vision_stream.wait_event(event)
            slot.vision_begin = torch.cuda.Event(enable_timing=True)
            slot.vision_begin.record(self.vision_stream)
            for target, source in zip(input_tensors(slot.prepared), input_tensors(prepared), strict=True):
                stage_tensor(target, source)
            slot.vision_graph.replay()
            # A live old generation retains its original metadata Event.
            slot.metadata_ready = metadata_event or torch.cuda.Event(enable_timing=True)
            slot.metadata_ready.record(self.vision_stream)
        return slot.metadata_ready

    @torch.no_grad()
    def prefill(self, index):
        slot = self.slots[index]
        with torch.cuda.stream(self.prefix_stream):
            self.prefix_stream.wait_event(slot.metadata_ready)
            slot.frontier.zero_()
            reset = torch.cuda.Event(enable_timing=True)
            reset.record(self.prefix_stream)
            slot.prefix_graph.replay()
            done = torch.cuda.Event(enable_timing=True)
            done.record(self.prefix_stream)
        return reset, done

    @torch.no_grad()
    def generate(self, index, prepared, noise, *, reset, prefix_done, lead, force_old=False):
        slot = self.slots[index]
        input_ready = torch.cuda.Event()
        input_ready.record()
        with torch.cuda.stream(self.ae_stream):
            self.ae_stream.wait_event(input_ready)
            self.ae_stream.wait_event(reset)
            if lead:
                self.ae_stream.wait_event(prefix_done if lead == self.num_layers else slot.layer_ready[lead-1])
            stage_tensor(slot.ae_prepared.state, prepared.state)
            stage_tensor(slot.noise, noise)
            slot.override.copy_(self.policy_constants[0 if force_old else -1])
            begin = torch.cuda.Event(enable_timing=True)
            begin.record(self.ae_stream)
            slot.ae_graph.replay()
            actions = slot.actions.clone()
            # Different Event handles are required for two different live leases.
            old_read, new_read = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            old_read.record(self.ae_stream)
            new_read.record(self.ae_stream)
        return actions, begin, old_read, new_read

    def close(self):
        for stream in self.streams:
            stream.synchronize()
        self.slots.clear()


class ProgressiveRuntime:
    """One scheduler Host; complete old snapshot + current partial layer lease.

Only same-task fixed mask/layout episodes are supported. The old lease is KV
only: all AE masks, positions, state and noise belong to the current new slot.
Thus next Vision can write old-slot embedding/metadata without touching AE.
"""
    def __init__(self, backend, prepared, noise, *, profile=False, graphs_factory=None):
        from .progressive_kv_pool import ProgressiveKVPool
        self.backend = backend
        factory = graphs_factory or ProgressiveGraphs
        self.graphs = factory(backend, prepared, noise, profile=profile)
        self.pool = ProgressiveKVPool(self.graphs.num_layers, epoch=1, task="progressive")
        self.next_id = 0
        self.last_producer = None
        self.pending_prepared = None
        self.records = []
        self.requires_reset = False

    def observation(self, prepared, observation_id):
        return Observation(observation_id, self.pool.epoch, self.pool.task, prepared)

    def _reserve(self, prepared, observation_id):
        ahead = self.pool.vision_ahead
        if ahead is not None:
            if ahead.observation.observation_id != observation_id or ahead.observation.payload is not prepared:
                raise RuntimeError("Prepared input differs from the accepted ahead ticket")
            observation = ahead.observation
        else:
            observation = self.observation(prepared, observation_id)
        self.pool.offer(observation)
        result = self.pool.reserve()
        if result is None:
            raise RuntimeError("No KV slot free after the prior read/producer completed")
        identity, _ = result
        ticket = self.pool.claim_vision_ahead(identity)
        slot = self.graphs.slots[identity.slot]
        if ticket is None:
            with torch.cuda.nvtx.range(f"submit/vision/frame={observation_id}/slot={identity.slot}/gen={identity.generation}"):
                metadata = self.graphs.encode(identity.slot, prepared)
            self.pool.track_preparation(identity, metadata)
        elif ticket.observation.observation_id != observation_id:
            raise RuntimeError("Lookahead vision does not match the admitted observation")
        return identity, slot

    def _submit(self, identity, slot):
        frame = self.pool._slot(identity).observation.observation_id
        with torch.cuda.nvtx.range(f"submit/vlm/frame={frame}/slot={identity.slot}/gen={identity.generation}"):
            reset, done = self.graphs.prefill(identity.slot)
        self.pool.submitted_layers(identity, slot.prefix, done, slot.layer_ready,
                                   metadata_ready=slot.metadata_ready)
        return reset, done

    def bootstrap(self, prepared):
        identity, slot = self._reserve(prepared, self.next_id)
        _, done = self._submit(identity, slot)
        done.synchronize()
        self.pool.poll()
        self.last_producer = done
        self.next_id += 1

    def reset(self, prepared):
        self.drain()
        self.pool.drain_reset(self.pool.epoch + 1, self.pool.task)
        self.next_id = 0
        self.pending_prepared = None
        self.bootstrap(prepared)
        self.requires_reset = False

    @torch.no_grad()
    def request(self, prepared, noise, *, lead=3, force_old=False, lookahead=None,
                next_encoder=False, verify=False, collect_events=True):
        if lead not in (0, 1, 3, 6, self.graphs.num_layers):
            raise ValueError("Supported leads are 0/1/3/6 and full-Prefix control")
        if self.requires_reset:
            raise RuntimeError("A prior failed submission requires explicit reset before reuse")
        begin_host = time.perf_counter()
        if self.last_producer is not None:
            self.last_producer.synchronize()
        self.pool.poll()
        old = self.pool.acquire_old_kv()
        if old is None:
            raise RuntimeError("Bootstrap a complete previous KV first")
        current = None
        try:
            identity, slot = self._reserve(prepared, self.next_id)
            origin = torch.cuda.Event(enable_timing=True)
            origin.record()
            reset, prefix_done = self._submit(identity, slot)
            current = self.pool.acquire_layers(identity)
            self.pool.validate_replay(current, layer_events=slot.layer_ready,
                                      metadata_event=slot.metadata_ready)
            submit_start = time.perf_counter()
            with torch.cuda.nvtx.range(f"progressive/request_{self.next_id}/lead_{lead}/slot_{identity.slot}"):
                actions, ae_begin, old_read, new_read = self.graphs.generate(
                    identity.slot, prepared, noise, reset=reset, prefix_done=prefix_done,
                    lead=lead, force_old=force_old)
            submit_ms = (time.perf_counter() - submit_start) * 1000
            next_prepared = None
            ahead_ms = 0.0
            if lookahead is not None:
                ahead_start = time.perf_counter()
                next_prepared = lookahead()
                ahead_ms = (time.perf_counter() - ahead_start) * 1000
                if next_prepared is not None and next_encoder:
                    # The old slot's KV stays pinned; only its independent
                    # Vision workspace is written, after current VLM completes.
                    observation = self.observation(next_prepared, self.next_id + 1)
                    metadata = torch.cuda.Event(enable_timing=True)
                    old_producer = self.pool.validate_vision_ahead(old.identity, observation, metadata,
                                                                 contract="old_kv_only")
                    dependency = (old_producer, prefix_done, slot.iteration_marks[1])
                    with torch.cuda.nvtx.range(f"submit/vision/frame={self.next_id+1}/slot={old.identity.slot}/gen={old.identity.generation+1}/ahead"):
                        self.graphs.encode(old.identity.slot, next_prepared, dependency=dependency,
                                           metadata_event=metadata)
                    self.pool.register_vision_ahead(old.identity,
                        observation, metadata,
                        contract="old_kv_only")
            new_read.synchronize()
            output = self.download_actions(actions)
            end_host = time.perf_counter()
            record = dict(request=self.next_id, epoch=identity.epoch, slot=identity.slot,
                          generation=identity.generation, old_slot=old.identity.slot,
                          old_generation=old.identity.generation, lead=lead, force_old=force_old,
                          next_encoder=bool(next_prepared is not None and next_encoder),
                          cpu_cpu_ms=(end_host-begin_host)*1000, ae_submit_ms=submit_ms,
                          lookahead_prepare_ms=ahead_ms, prefix_done_at_response=prefix_done.query(),
                          graph_recaptures=0, full_kv_slots=2)
            record.update(frontiers=slot.frontiers.detach().cpu().tolist(),
                          iteration_ms=[slot.iteration_marks[i].elapsed_time(slot.iteration_marks[i+1])
                                        for i in range(self.backend.num_steps)])
            record['_events'] = dict(vision_begin=slot.vision_begin, vision_done=slot.metadata_ready,
                reset=reset, prefix_done=prefix_done, ae_begin=ae_begin, ae_done=new_read)
            if next_prepared is not None and next_encoder:
                ahead_slot = self.graphs.slots[old.identity.slot]
                record['_events'].update(next_vision_begin=ahead_slot.vision_begin,
                                         next_vision_done=ahead_slot.metadata_ready)
            if collect_events or verify:
                prefix_done.synchronize()
                self.finish_record(record)
            if verify:
                record['numerical'] = self.verify(slot, self.graphs.slots[old.identity.slot], prepared, noise, output, record['frontiers'])
            self.pool.release(old, old_read)
            self.pool.release(current, new_read)
            current = old = None
            self.pool.poll()
            self.last_producer = prefix_done
            self.next_id += 1
            self.records.append(record)
            return output, record, next_prepared
        except BaseException:
            self.requires_reset = True
            for stream in self.graphs.streams:
                stream.synchronize()
            for lease in (old, current):
                if lease is not None and not lease.released:
                    event = torch.cuda.Event()
                    event.record(self.graphs.ae_stream)
                    self.pool.release(lease, event)
            raise

    def download_actions(self, actions):
        """Single-lane public shape; batch experiments override this boundary."""
        return actions[0].detach().cpu().numpy().copy()

    def verification_summary(self, prepared, references, output, actual_steps):
        from .layer_bench import difference
        return {**{name: difference(reference, output) for name, reference in references.items()},
                "actual_steps": actual_steps}

    @torch.no_grad()
    def verify(self, slot, old, prepared, noise, output, frontiers):
        """Replay the OBSERVED selection schedule serially; no extra full KV."""
        class StaticCache:
            def __getattr__(self, name): return getattr(slot.prefix.cache, name)
            def __len__(self): return len(slot.prefix.cache)
            def __getitem__(self, layer):
                return slot.prefix.cache[layer] if layer < frontiers[self.step] else old.prefix.cache[layer]
        cache = StaticCache()
        selected = Prefix(cache, slot.prefix.pad_masks, slot.prefix.position_ids)
        original = self.backend.velocity
        steps = []
        def velocity(prefix, p, x, t):
            cache.step = len(steps)
            steps.append(cache.step)
            return original(selected, p, x, t)
        try:
            self.backend.velocity = velocity
            serial = self.download_actions(self.backend.generate(selected, prepared, noise))
        finally:
            self.backend.velocity = original
        fresh = self.download_actions(self.backend.generate(slot.prefix, prepared, noise))
        old_prefix = Prefix(old.prefix.cache, slot.prefix.pad_masks, slot.prefix.position_ids)
        stale = self.download_actions(self.backend.generate(old_prefix, prepared, noise))
        return self.verification_summary(prepared,
            dict(matched_schedule=serial, versus_fresh=fresh, versus_old=stale), output, len(steps))

    def drain(self):
        for stream in self.graphs.streams:
            stream.synchronize()
        self.pool.poll()

    def finish_record(self, record):
        events = record.pop('_events', None)
        if events is None:
            return record
        end = events['ae_done']
        pd = events['prefix_done']
        pd.synchronize()
        end.synchronize()
        origin = events['reset']
        record.update(vision_ms=events['vision_begin'].elapsed_time(events['vision_done']),
                      vlm_ms=origin.elapsed_time(pd),
                      ae_ms=events['ae_begin'].elapsed_time(end),
                      ae_start_after_vlm_start_ms=origin.elapsed_time(events['ae_begin']),
                      ae_finish_after_vlm_ms=pd.elapsed_time(end),
                      exposed_ae_after_vlm_ms=max(0.0, pd.elapsed_time(end)),
                      model_after_encoder_ms=max(origin.elapsed_time(pd), origin.elapsed_time(end)))
        if 'next_vision_done' in events:
            events['next_vision_done'].synchronize()
            record.update(next_vision_ms=events['next_vision_begin'].elapsed_time(events['next_vision_done']),
                          ae_finish_minus_next_encoder_ms=events['next_vision_done'].elapsed_time(end),
                          next_encoder_start_after_vlm_ms=pd.elapsed_time(events['next_vision_begin']))
        return record

    def close(self):
        self.drain()
        self.pool.drain_reset(self.pool.epoch + 1, self.pool.task)
        self.graphs.close()
