"""Controller-facing synchronous API with independently progressing prefix work."""
import time

import torch

from .experiments import event, native_raw
from .scheduler import Observation, Pool


class PipelinePolicy:
    def __init__(self, backend, mode="async_reuse", interval=1):
        self.backend = backend
        self.background = torch.cuda.Stream(device=backend.device)
        self.pool = Pool()
        self.request = 0
        self.configure(mode, interval)

    def configure(self, mode, interval):
        if mode not in ("fresh_serial", "serial_reuse", "async_reuse") or interval not in (1, 2, 4):
            raise ValueError("Unknown frozen mode or interval")
        self.reset()
        self.mode, self.interval = mode, interval
        return self.protocol

    @property
    def protocol(self):
        return dict(self.backend.identity, mode=getattr(self, "mode", None), interval=getattr(self, "interval", None),
                    H=self.backend.model.config.action_horizon, R=5, cache_slots=2,
                    real_time=False, clock="LIBERO pauses during inference", full_model=True)

    def reset(self):
        self.pool.drain_reset(self.pool.epoch + 1, "")
        self.request = 0
        return dict(epoch=self.pool.epoch)

    def _refresh(self, input_ready, synchronous=False):
        reservation = self.pool.reserve()
        if reservation is None:
            return None
        identity, observation = reservation
        self.background.wait_event(input_ready)
        start, ready = event(), event()
        with torch.cuda.stream(self.background):
            start.record()
            value = self.backend.build_prefix(observation.payload)
            ready.record()
        self.pool.submitted(identity, value, ready)
        if synchronous:
            ready.synchronize()
            self.pool.poll()
        return dict(identity=identity, start=start, ready=ready)

    def predict(self, obs, instruction, suite, seed, **unused):
        from preparations.idea1_v2_1.runtime_client import input_digest
        begin = time.perf_counter()
        receipt_unix = time.time()
        if self.pool.task != instruction:
            self.pool.drain_reset(self.pool.epoch + 1, instruction)
            self.request = 0
        self.pool.poll()
        prepared = self.backend.prepare(native_raw(dict(obs=obs, instruction=instruction)))
        noise = self.backend.noise(seed)
        input_ready = event()
        input_ready.record()
        current = Observation(self.request, self.pool.epoch, instruction, prepared,
                              state_id=self.request, state_time=begin,
                              host_receipt_time=begin, host_receipt_unix=receipt_unix)
        bootstrapped = self.pool.active is None
        if self.mode == "fresh_serial" or bootstrapped:
            self.pool.offer(current)
            self._refresh(input_ready, synchronous=True)
        lease = self.pool.acquire()
        if lease is None:
            raise RuntimeError("No completed compatible prefix after initialization")
        # Select once, before submitting either task; a new publication can only
        # affect a later request. The lease pins all N steps and output copying.
        source, identity = lease.observation.observation_id, lease.identity
        source_receipt = lease.observation.host_receipt_time
        source_unix = lease.observation.host_receipt_unix
        slot = self.pool.slots[identity.slot]
        prefix_ready_observed, prefix_published = slot.ready_observed_at, slot.published_at
        generation_start = time.perf_counter()
        due = self.mode != "fresh_serial" and not bootstrapped and self.request % self.interval == 0
        if due:
            self.pool.offer(current)
        refresh = None
        if self.mode == "async_reuse" and self.pool.pending is not None:
            refresh = self._refresh(input_ready)
        ae_start, ae_done = event(), event()
        ae_start.record()
        raw = self.backend.generate(lease.value, prepared, noise)
        ae_done.record()
        ae_done.synchronize()
        actions = self.backend.output(prepared, raw)
        available = time.perf_counter()
        self.pool.release(lease, ae_done)
        del lease
        if self.mode == "serial_reuse" and self.pool.pending is not None:
            refresh = self._refresh(input_ready, synchronous=True)
        self.pool.poll()
        result = dict(actions=actions, execute_length=5, seed=seed,
                      input_hash=input_digest(obs, instruction),
                      mode=self.mode, interval=self.interval, request=self.request,
                      snapshot=dict(epoch=identity.epoch, slot=identity.slot, generation=identity.generation),
                      current_observation_id=self.request, prefix_observation_id=source,
                      suffix_state_id=self.request, reference_state_id=self.request,
                      observation_request_age=self.request-source,
                      source_host_receipt_monotonic_s=source_receipt,
                      source_host_receipt_unix_s=source_unix,
                      current_host_receipt_unix_s=receipt_unix,
                      prefix_ready_observed_monotonic_s=prefix_ready_observed,
                      prefix_published_monotonic_s=prefix_published,
                      generation_start_monotonic_s=generation_start,
                      source_receipt_age_at_generation_ms=(generation_start-source_receipt)*1000,
                      source_receipt_age_at_internal_output_ms=(available-source_receipt)*1000,
                      source_receipt_age_at_response_ms=(time.perf_counter()-source_receipt)*1000,
                      receipt_age_definition="Host observation receipt age, not image capture age",
                      camera_capture_age_ms=None, camera_clock="not recorded; host receipt is not capture",
                      generated_length=int(actions.shape[0]), submitted_length=5,
                      timing=dict(action_ready_host_ms=(available-begin)*1000,
                                  response_host_ms=(time.perf_counter()-begin)*1000,
                                  ae_gpu_ms=ae_start.elapsed_time(ae_done),
                                  availability_definition="action_ready_host_ms is internal; actual controller receives RPC response after response_host_ms plus transport"),
                      pool=dict(live_snapshots=self.pool.live_snapshots, publications=self.pool.publications,
                                pending_replacements=self.pool.pending_replacements, no_free_slot=self.pool.no_free_slot),
                      peak_allocated=torch.cuda.max_memory_allocated(self.backend.device),
                      peak_reserved=torch.cuda.max_memory_reserved(self.backend.device))
        self.request += 1
        return result
