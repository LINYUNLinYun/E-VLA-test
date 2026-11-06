"""One eager differentiable Full-D10 forward, serial / 2+2 / four VJPs.

No compilation, CUDA Graph, changed dtype, new projection, or new gate.
The native inference still precedes this differentiable replay, as before.
"""
import time
import numpy as np
import torch
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow, current_item
from preparations.idea1_v3_4_E2_comp_error_multivjp.model import projection_weights
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from .scoring import score_and_choose


def calculate(model, item, weights, gates, batch_size=1, events=False):
    if batch_size not in (1, 2, 4) or weights.shape != (4, 5, 7):
        raise ValueError((batch_size, weights.shape))
    stamps = [torch.cuda.Event(enable_timing=True) for _ in range(3)] if events else None
    if stamps:
        stamps[0].record()
    with torch.enable_grad():
        raw = model.forward(item, gates=gates)
        outputs = raw[:5, :7]
        if outputs.dtype != torch.float64:
            outputs = outputs.float()
        values = (outputs[None] * weights).sum((1, 2))
        if stamps:
            stamps[1].record()
        if batch_size == 1:
            grads = torch.stack([torch.autograd.grad(values[i], gates,
                retain_graph=i < 3)[0] for i in range(4)])
        else:
            identity = torch.eye(4, device=values.device, dtype=values.dtype)
            grads = torch.cat([torch.autograd.grad(values, gates,
                grad_outputs=identity[i:i+batch_size], is_grads_batched=True,
                retain_graph=i+batch_size < 4)[0] for i in range(0, 4, batch_size)])
        if stamps:
            stamps[2].record()
    return raw.detach(), grads.detach(), stamps


class Selector:
    q = 4
    rule = 'spaced'

    def __init__(self, backend):
        self.backend = backend
        self.batch_size = 1
        self.aggregation = 'layer_rms_sum'
        self.clear()

    def clear(self):
        self.model = self.item = self.raw = self.grad = self.W = self.g = None
        self.runtime_id = None
        self.last_combination_scores = None

    def run(self, c, seed, coeff, k=6):
        if self.q != 4 or not np.array_equal(coeff, coefficients()):
            raise ValueError('Frozen four projections and time weights required')
        torch.cuda.synchronize()
        tick = time.perf_counter()
        setup_ms = 0.
        if self.runtime_id != id(c['runtime']):
            begin = time.perf_counter()
            self.model = borrow(self.backend, c)
            self.W = torch.as_tensor(projection_weights(4), device='cuda')
            self.g = torch.ones(17, device='cuda', requires_grad=True)
            self.runtime_id = id(c['runtime'])
            torch.cuda.synchronize()
            setup_ms = (time.perf_counter()-begin)*1000
        self.item = current_item(c, seed)
        self.raw, self.grad, stamps = calculate(self.model, self.item, self.W,
            self.g, self.batch_size, events=True)
        grads = self.grad.float().cpu().numpy().copy()
        begin = time.perf_counter()
        skip, layer_score, combination_score = score_and_choose(grads, k, self.aggregation)
        scoring_ms = (time.perf_counter()-begin)*1000
        self.last_combination_scores = combination_score
        total = (time.perf_counter()-tick)*1000
        return skip, grads, layer_score, dict(selector_total_ms=total,
            selector_hot_ms=total-setup_ms, selector_capture_ms=0.,
            selector_setup_ms=setup_ms, selector_scoring_ms=scoring_ms,
            selector_forward_ms=float(stamps[0].elapsed_time(stamps[1])),
            selector_backward_ms=float(stamps[1].elapsed_time(stamps[2])))
