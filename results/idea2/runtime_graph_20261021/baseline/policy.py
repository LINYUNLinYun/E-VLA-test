"""Controller-facing synchronous API with independently progressing prefix work."""
import time

import torch

from .experiments import event, native_raw
from .scheduler import Observation, Pool
from .submission import HostWorkers, submit


class PipelinePolicy:
    def __init__(self, backend, mode="async_reuse", interval=1,
                 submission_order="prefill_first", diagnostics=False, host_workers=False):
        self.backend = backend
        self.background = torch.cuda.Stream(device=backend.device)
        self.pool = Pool()
        self.request = 0
        self.submission_order = submission_order
        self.diagnostics = diagnostics
        self._prefix_timings = []
        self.workers = None
        self.ae_stream = torch.cuda.Stream(device=backend.device)
        self.configure(mode, interval, host_workers=host_workers)

    def configure(self, mode, interval, submission_order=None, diagnostics=None, host_workers=None):
        if mode not in ("fresh_serial", "serial_reuse", "async_reuse") or interval not in (1, 2, 4):
            raise ValueError("Unknown frozen mode or interval")
        order = self.submission_order if submission_order is None else submission_order
        if order not in ("ae_first", "prefill_first"):
            raise ValueError("Unknown submission order")
        self.reset()
        self.mode, self.interval = mode, interval
        self.submission_order = order
        if diagnostics is not None:
            self.diagnostics = diagnostics
        if host_workers is True and self.workers is None:
            self.workers = HostWorkers()
        elif host_workers is False and self.workers is not None:
            self.workers.close()
            self.workers = None
        return self.protocol

    @property
    def protocol(self):
        return dict(self.backend.identity, mode=getattr(self, "mode", None), interval=getattr(self, "interval", None),
                    submission_order=self.submission_order, diagnostics=self.diagnostics,
                    host_workers=2 if self.workers is not None else 0,
                    H=self.backend.model.config.action_horizon, R=5, cache_slots=2,
                    real_time=False, clock="LIBERO pauses during inference", full_model=True)

    def reset(self):
        self.pool.drain_reset(self.pool.epoch + 1, "")
        completed = self._completed_prefix_timings() if self.diagnostics else []
        self._prefix_timings.clear()
        self.request = 0
        return dict(epoch=self.pool.epoch, completed_prefills=completed)

    def _refresh(self, input_ready, synchronous=False):
        reservation = self.pool.reserve()
        if reservation is None:
            return None
        identity, observation = reservation
        value, job = submit(self.background, "prefill", self.backend.build_prefix,
                            observation.payload, dependency=input_ready)
        start, ready = job.start, job.done
        self.pool.submitted(identity, value, ready)
        if self.diagnostics:
            self._prefix_timings.append((identity, observation.observation_id, job))
        if synchronous:
            ready.synchronize()
            self.pool.poll()
        return dict(identity=identity, start=start, ready=ready, submission=job)

    def _completed_prefix_timings(self):
        completed, pending = [], []
        for identity, source, job in self._prefix_timings:
            if job.done.query():
                completed.append(dict(slot=identity.slot, generation=identity.generation,
                                      epoch=identity.epoch, observation_id=source, **job.metrics()))
            else:
                pending.append((identity, source, job))
        self._prefix_timings = pending
        return completed

    def close(self):
        self.reset()
        if self.workers is not None:
            self.workers.close()
            self.workers = None

    def _abort_generation(self, lease):
        done = event()
        done.record()
        self.pool.release(lease, done)
        self.pool.drain_reset(self.pool.epoch + 1, self.pool.task)
        self._prefix_timings.clear()
        self.request = 0

    def _worker_generation(self, lease, prepared, noise, input_ready):
        # All reservations/publications remain on this owner thread. Workers
        # only submit NN work, holding a fixed old KV lease or a reserved input.
        reservation = self.pool.reserve() if self.pool.pending is not None else None
        names = ("ae", "prefill") if self.submission_order == "ae_first" else ("prefill", "ae")
        futures = {}
        for name in names:
            if name == "ae":
                futures[name] = self.workers.enqueue(name, self.ae_stream, self.backend.generate,
                                                     lease.value, prepared, noise, dependency=input_ready)
            elif reservation is not None:
                futures[name] = self.workers.enqueue(name, self.background, self.backend.build_prefix,
                                                     reservation[1].payload, dependency=input_ready)
        try:
            results = self.workers.collect(futures, (self.ae_stream, self.background))
        except BaseException:
            # collect() joined both host jobs and drained both owned streams.
            # Release the pinned reader before clearing a failed reservation.
            self._abort_generation(lease)
            raise
        refresh = None
        if reservation is not None:
            identity, observation = reservation
            value, job = results.pop("prefill")
            self.pool.submitted(identity, value, job.done)
            if self.diagnostics:
                self._prefix_timings.append((identity, observation.observation_id, job))
            refresh = dict(identity=identity, start=job.start, ready=job.done, submission=job)
        raw, ae_job = results["ae"]
        return raw, ae_job, refresh

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
        if self.mode == "async_reuse" and self.workers is not None:
            raw, ae_job, refresh = self._worker_generation(lease, prepared, noise, input_ready)
        else:
            foreground = torch.cuda.current_stream(self.backend.device)
            try:
                if (self.mode == "async_reuse" and self.pool.pending is not None
                        and self.submission_order == "prefill_first"):
                    refresh = self._refresh(input_ready)
                raw, ae_job = submit(foreground, "ae", self.backend.generate, lease.value, prepared, noise)
                if (self.mode == "async_reuse" and self.pool.pending is not None
                        and self.submission_order == "ae_first"):
                    refresh = self._refresh(input_ready)
            except BaseException:
                foreground.synchronize()
                self.background.synchronize()
                self._abort_generation(lease)
                raise
        ae_start, ae_done = ae_job.start, ae_job.done
        submit_status = (dict(ae=ae_done.query(),
                             prefill=refresh["ready"].query() if refresh is not None else None)
                         if self.diagnostics else None)
        ae_done.synchronize()
        actions = self.backend.output(prepared, raw)
        available = time.perf_counter()
        self.pool.release(lease, ae_done)
        del lease
        if self.mode == "serial_reuse" and self.pool.pending is not None:
            refresh = self._refresh(input_ready, synchronous=True)
        self.pool.poll()
        diagnostic = (dict(ae=ae_job.metrics(),
                           prefill=refresh["submission"].metrics() if refresh is not None else None,
                           completed_prefills=self._completed_prefix_timings(),
                           done_after_both_submissions=submit_status)
                      if self.diagnostics else None)
        result = dict(actions=actions, execute_length=5, seed=seed,
                      input_hash=input_digest(obs, instruction),
                      mode=self.mode, interval=self.interval, request=self.request,
                      submission_order=self.submission_order, diagnostics=diagnostic,
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
