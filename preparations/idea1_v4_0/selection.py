"""Same fused four-direction graph; optional forecast-error gate direction."""
import time
import torch
from preparations.idea1_v3_4_selection_async.core import Graph as Base, calculate
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow, current_item
from preparations.idea1_v3_4_E2_comp_error_cost.model import ErrorModel
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import projection_weights


class Graph(Base):
    def __init__(self, backend, context, seed, kind='original'):
        from reports import idea1_v3_4_selection_async_batched_attention as direct
        from reports import idea1_v3_4_selection_async_rms_backward as rms
        from reports import idea1_v3_4_selection_async_elementwise as element
        direct.install(False)
        rms.install()
        element.install(glu=True)
        self.backend, self.kind, self.q = backend, kind, 4
        self.mode, self.batch = 'original', True
        tick = time.perf_counter()
        model = borrow(backend, context)
        self.model = ErrorModel(model) if kind == 'error' else model
        self.item = {k: v.clone() for k, v in self.stage(context, seed).items()}
        self.W = torch.as_tensor(projection_weights(4), device='cuda')
        self.g = torch.ones(17, device='cuda', requires_grad=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                calculate(self.model, self.item, self.W, self.g, True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.raw, self.gradient = calculate(self.model, self.item, self.W, self.g, True)
        torch.cuda.synchronize()
        self.capture_s = time.perf_counter()-tick
        self.begin = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.pending = False

    def stage(self, context, seed):
        item = current_item(context, seed)
        if self.kind == 'error':
            item['gate_error'] = context['selection_error']
        return item
