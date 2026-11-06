"""Module submission diagnostics. Event spans include dispatch gaps and waiting.

No device-wide fences: callers own dependencies and decide when results are
needed. Jobs retain only events/metrics, never an extra KV snapshot.
"""
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import torch

from .experiments import event


@dataclass
class Submission:
    start: object
    done: object
    cpu_begin: float
    cpu_end: float
    done_at_submit: bool
    stream: int

    def metrics(self):
        complete = self.done.query()
        return dict(cpu_submit_ms=(self.cpu_end-self.cpu_begin)*1000,
                    cpu_begin_monotonic_s=self.cpu_begin,
                    cpu_end_monotonic_s=self.cpu_end,
                    done_at_submit=self.done_at_submit, done_now=complete,
                    gpu_event_span_ms=self.start.elapsed_time(self.done) if complete else None,
                    stream=self.stream,
                    gpu_time_definition="CUDA event span includes dispatch gaps/resource waits; kernel busy time requires nsys")


def submit(stream, label, function, *args, dependency=None):
    start, done = event(), event()
    begin = time.perf_counter()
    with torch.cuda.stream(stream), torch.no_grad():
        if dependency is not None:
            stream.wait_event(dependency)
        torch.cuda.nvtx.range_push(label)
        try:
            start.record(stream)
            value = function(*args)
            done.record(stream)
        except BaseException:
            # Protect inputs/leases if a model call fails after partial dispatch.
            stream.synchronize()
            raise
        finally:
            torch.cuda.nvtx.range_pop()
    end = time.perf_counter()
    return value, Submission(start, done, begin, end, done.query(), stream.cuda_stream)


class HostWorkers:
    """One persistent submitter per module; neither worker can mutate Pool.

    result() waits for host submission only. CUDA completion remains controlled
    by the returned per-job event. Always join both host jobs, including errors,
    before the owner may reset a pool or release a lease.
    """
    def __init__(self):
        self.executors = {name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"idea2-{name}")
                          for name in ("ae", "prefill")}

    def enqueue(self, name, stream, function, *args, dependency=None):
        return self.executors[name].submit(submit, stream, name, function, *args, dependency=dependency)

    def collect(self, futures, streams):
        results, errors = {}, []
        for name, future in futures.items():
            try:
                results[name] = future.result()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            # Successful siblings may still have GPU work. Drain owned streams
            # before allowing exception unwinding to drop inputs or leases.
            for stream in streams:
                stream.synchronize()
            raise errors[0]
        return results

    def close(self):
        for executor in self.executors.values():
            executor.shutdown(wait=True)
