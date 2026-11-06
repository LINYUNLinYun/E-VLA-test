"""One-VJP score for an observed residual forecast error; no future inputs.

This is a cost/functional probe, not a claim about closed-loop success.
The original Full-D10 model, selectors, and published experiments are untouched.
"""
import time

import numpy as np
import torch

from preparations.idea1_v3_4_E2_cycle3.common import choose
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import (
    Full10Model, Selector as OriginalSelector, borrow, calculate, current_item,
)


def forecast_error(current, last, older, ratio=1.0, out=None):
    """R_current - [R_last + ratio * (R_last - R_older)], FP32.

    All three dictionaries must describe the same prompt and layer coordinates.
    ratio uses actual elapsed Full steps, not the age of a future Fast request.
    Four ordinary elementwise operations/layer; no new model execution.
    """
    expected = set(range(1, 18))
    if any(set(bank) != expected for bank in (current, last, older)):
        raise ValueError('All seventeen layer histories are required')
    shape = current[1].shape
    if any(v.shape != shape or v.dtype != torch.float32
           for bank in (current, last, older) for v in bank.values()):
        raise ValueError('Expected matching FP32 residual histories')
    if out is None:
        out = torch.empty((17, *shape), dtype=torch.float32,
                          device=current[1].device)
    if out.shape != (17, *shape) or out.dtype != torch.float32:
        raise ValueError('Wrong forecast error buffer')
    with torch.no_grad():
        for i in range(17):
            layer = i + 1
            torch.sub(last[layer], older[layer], out=out[i])
            out[i].mul_(float(ratio)).add_(last[layer])
            torch.sub(current[layer], out[i], out=out[i])
    return out


class ErrorModel(Full10Model):
    def __init__(self, full_model):
        # Frozen native weights and D10 schedule are shared, not duplicated.
        self.__dict__.update(full_model.__dict__)

    def encode(self, item, gates=None, skip=(), patch=None, trace=False,
               start=0, prefix=None):
        if skip or patch is not None or start or prefix is not None:
            raise ValueError('Error scoring is defined on a complete Full path')
        x = item['x']
        error = item.get('gate_error')
        if gates is not None and (error is None or error.shape != (17, *x.shape)):
            raise ValueError('Seventeen matching forecast error directions required')
        ks, vs, tr = [], [], []
        for i in range(18):
            inp = x
            y, k, v = self.block(x, i, item['encoder_rope'], update=i < 17)
            if gates is not None and i < 17:
                # g=1 is exactly Full; g=0 injects -forecast_error.
                # Detached error keeps the derivative of the Full path intact.
                arithmetic = y if y.dtype == torch.float64 else y.float()
                x = (arithmetic - (1 - gates[i]) * error[i].detach()).to(y.dtype)
            else:
                x = y
            ks.append(k)
            vs.append(v)
            if trace:
                tr.append((inp, x))
        result = dict(K=ks, V=vs)
        if trace:
            result.update(inputs=[a for a, b in tr], outputs=[b for a, b in tr])
        return result


class Selector(OriginalSelector):
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

    def run(self, c, seed, coeff, error, k=4):
        tick = time.perf_counter()
        item = current_item(c, seed)
        item['gate_error'] = error.detach()
        W = torch.as_tensor(coeff, device='cuda')
        cold = 0.0
        if self.graph is None or self.runtime_id != id(c['runtime']):
            if self.graph is not None:
                self.clear()
            cold = self.capture(c, item, W)
        for name, value in item.items():
            self.item[name].copy_(value)
        self.W.copy_(W)
        self.graph.replay()
        grad = self.grad.float().cpu().numpy().copy()
        if not np.isfinite(grad).all():
            raise ValueError('Nonfinite error-direction gradient')
        score = abs(grad)
        skip = choose(score, k)
        torch.cuda.synchronize()
        total = (time.perf_counter() - tick) * 1000
        return skip, grad, score, dict(selector_total_ms=total,
                                      selector_hot_ms=total-cold,
                                      selector_capture_ms=cold)
