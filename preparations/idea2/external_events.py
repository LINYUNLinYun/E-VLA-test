"""Explicit cross-graph event nodes for the installed PyTorch 2.7 runtime.

PyTorch Event in this environment has no external=True argument. Use the
matching bundled cudart, with flags declared by the local CUDA headers.
Events must be initialized outside capture and live as long as their graphs.
"""
import ctypes
from pathlib import Path

import torch


class ExternalEvents:
    def __init__(self):
        runtime = Path(torch.__file__).parent.parent / "nvidia/cuda_runtime/lib/libcudart.so.12"
        self.library_path = str(runtime.resolve(strict=True))
        self.lib = ctypes.CDLL(self.library_path)
        self._record = self.lib.cudaEventRecordWithFlags
        self._wait = self.lib.cudaStreamWaitEvent
        for function in (self._record, self._wait):
            function.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
            function.restype = ctypes.c_int
        self._error = self.lib.cudaGetErrorString
        self._error.argtypes = (ctypes.c_int,)
        self._error.restype = ctypes.c_char_p

    def _check(self, status, operation):
        if status:
            raise RuntimeError(f"{operation}: CUDA {status}: {self._error(status).decode()}")

    @staticmethod
    def _handle(event):
        handle = event.cuda_event
        if not handle:
            raise RuntimeError("Initialize event on its producer stream before capture")
        return handle

    def mark(self, event, stream):
        self._check(self._record(self._handle(event), stream.cuda_stream, 1), "external record")

    def wait(self, event, stream):
        self._check(self._wait(stream.cuda_stream, self._handle(event), 1), "external wait")


def initialized_events(stream, count, *, timing=True):
    events = [torch.cuda.Event(enable_timing=timing) for _ in range(count)]
    for event in events:
        event.record(stream)
    return events


@torch.no_grad()
def cross_graph_self_test(repetitions=20):
    """Small diagnostic only: delayed producer and changing data catch stale waits.

The sleep kernel is deliberately NOT used anywhere in model experiments.
No CUDA resources are initialized merely by importing this module.
"""
    external = ExternalEvents()
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        source = torch.zeros(1024, device="cuda")
        target = torch.zeros_like(source)
        ready, = initialized_events(producer, 1)
    producer.synchronize()
    with torch.cuda.stream(consumer):
        result = torch.empty_like(source)
    consumer.synchronize()
    pgraph, cgraph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(pgraph, stream=producer):
        torch.cuda._sleep(500000)
        target.copy_(source)
        external.mark(ready, producer)
    with torch.cuda.graph(cgraph, stream=consumer):
        external.wait(ready, consumer)
        result.copy_(target)
    for index in range(repetitions):
        with torch.cuda.stream(producer):
            source.fill_(index + 1)
            pgraph.replay()
        with torch.cuda.stream(consumer):
            cgraph.replay()
        consumer.synchronize()
        if not torch.equal(result, torch.full_like(result, index + 1)):
            raise RuntimeError(f"Cross-graph wait consumed stale data at replay {index}")
    producer.synchronize()
    return dict(passed=True, repetitions=repetitions, runtime=external.library_path,
                producer_pool=str(pgraph.pool()), consumer_pool=str(cgraph.pool()))
