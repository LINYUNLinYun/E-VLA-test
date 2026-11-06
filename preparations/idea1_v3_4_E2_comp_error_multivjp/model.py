"""Nested output projections: one D10 forward, q reverse passes.

Projection zero is exactly the published positive weighted scalar. Additional
fixed Rademacher signs span the 5 x 7 outputs while preserving time magnitudes.
No projection is drawn or selected using an observation or an outcome.
"""
import time
import numpy as np
import torch
from preparations.idea1_v3_4_E2_comp_error_cost.model import ErrorModel, forecast_error
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow, current_item
from preparations.idea1_v3_4_E2_online_time_weights.selector import Selector as Base
from preparations.idea1_v3_4_E2_cycle3.common import coefficients, choose

PROJECTION_SEED = 340610

def projection_weights(q):
    if q not in (1, 2, 3, 4):
        raise ValueError(q)
    signs = np.ones((4, 5, 7), dtype=np.float32)
    signs[1:] = np.random.default_rng(PROJECTION_SEED).choice(
        [-1., 1.], size=(3, 5, 7)).astype(np.float32)
    return (signs[:q] * coefficients()[None]).copy()

def calculate(model, item, W, gates):
    with torch.enable_grad():
        raw = model.forward(item, gates=gates)
        outputs = raw[:5, :7]
        if outputs.dtype != torch.float64:
            outputs = outputs.float()
        values = (outputs[None] * W).sum((1, 2))
        grads = torch.stack([
            torch.autograd.grad(values[i], gates, retain_graph=i + 1 < len(W))[0]
            for i in range(len(W))])
    return raw.detach(), grads.detach()

class Selector(Base):
    def capture(self, c, item, W):
        tick = time.perf_counter()
        self.model = ErrorModel(borrow(self.backend, c))
        self.runtime_id = id(c['runtime'])
        self.item = {k: v.clone() for k, v in item.items()}
        self.W = W.clone()
        self.g = torch.ones(17, device='cuda', requires_grad=True)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                calculate(self.model, self.item, self.W, self.g)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.raw, self.grad = calculate(self.model, self.item, self.W, self.g)
        torch.cuda.synchronize()
        return (time.perf_counter() - tick) * 1000

    def run(self, c, seed, coeff, error, k=6):
        tick = time.perf_counter()
        item = current_item(c, seed)
        item['gate_error'] = error.detach()
        W = torch.as_tensor(coeff, device='cuda')
        cold = 0.
        if (self.graph is None or self.runtime_id != id(c['runtime']) or
                self.W.shape != W.shape):
            if self.graph is not None:
                self.clear()
            cold = self.capture(c, item, W)
        for name, value in item.items():
            self.item[name].copy_(value)
        self.W.copy_(W)
        self.graph.replay()
        grads = self.grad.float().cpu().numpy().copy()
        if not np.isfinite(grads).all():
            raise ValueError('Nonfinite multi-projection gradient')
        score = np.sqrt(np.mean(np.square(grads), axis=0))
        skip = choose(score, k)
        torch.cuda.synchronize()
        total = (time.perf_counter() - tick) * 1000
        return skip, grads, score, dict(selector_total_ms=total,
              selector_hot_ms=total-cold, selector_capture_ms=cold)
