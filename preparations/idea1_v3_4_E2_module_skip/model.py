"""Full-D10 replay with one VJP over FFN, A/F, or whole-block gates.

Only the first seventeen encoder layers are gated. The eighteenth layer still
produces current-observation K/V. Frozen weights and the ten-step action expert
are borrowed from the existing native service. Scores are returned for every
gate; budget-constrained mask selection belongs to the caller.
"""
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from preparations.idea1_v3_4_E2_E4_single import torch_model as tm
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import (
    Full10Model, borrow, current_item,
)


STRUCTURES = ('ffn', 'mixed', 'block')
MODULE_ORDER = ('attn', 'ffn')
LAYERS = tuple(range(1, 18))


def gate_shape(structure):
    if structure not in STRUCTURES:
        raise ValueError(f'Unknown gate structure: {structure}')
    return (17, 2) if structure == 'mixed' else (17,)


def _banks(residuals, structure):
    """Use native module histories; never reconstruct a block by adding A/F."""
    gate_shape(structure)
    if structure == 'mixed':
        return tuple(residuals[name] for name in MODULE_ORDER)
    name = 'ffn' if structure == 'ffn' else 'block'
    return (residuals[name] if name in residuals else residuals,)


def residual_tensor(residuals, structure='ffn'):
    """Convert native 1-based dictionaries to the documented gate coordinates."""
    banks = _banks(residuals, structure)
    if any(set(bank) != set(LAYERS) for bank in banks):
        raise ValueError('All seventeen matching residual layers are required')
    shape = banks[0][1].shape
    device = banks[0][1].device
    if any(v.shape != shape or v.dtype != torch.float32 or v.device != device
           for bank in banks for v in bank.values()):
        raise ValueError('Expected matching FP32 residuals on one device')
    if structure == 'mixed':
        return torch.stack([torch.stack([bank[layer] for bank in banks])
                            for layer in LAYERS])
    return torch.stack([banks[0][layer] for layer in LAYERS])


def forecast_error(current, last, older, ratio=1.0, out=None, structure='ffn'):
    """Current minus a forecast from the two *previous* Full observations.

    Input histories follow backend.py: {'attn': {1: tensor, ...}, 'ffn': ...}.
    The block control instead accepts the old {1: tensor, ...} block capture,
    or {'block': ...}; summing rounded A/F residuals is deliberately unsupported.
    Output is FP32, (17,*x.shape) or (17,2,*x.shape), A before F. The caller must
    compute this before appending the current Full to its history.
    """
    if not np.isfinite(ratio) or float(ratio) < 0:
        raise ValueError('Forecast ratio must be finite and nonnegative')
    histories = tuple(_banks(bank, structure) for bank in (current, last, older))
    expected = set(LAYERS)
    if any(set(bank) != expected for banks in histories for bank in banks):
        raise ValueError('All seventeen matching residual layers are required')
    reference = histories[0][0][1]
    if any(v.shape != reference.shape or v.dtype != torch.float32
           or v.device != reference.device
           for banks in histories for bank in banks for v in bank.values()):
        raise ValueError('Expected matching FP32 residual histories')
    shape = (*gate_shape(structure), *reference.shape)
    if out is None:
        out = torch.empty(shape, dtype=torch.float32, device=reference.device)
    if out.shape != shape or out.dtype != torch.float32 or out.device != reference.device:
        raise ValueError('Wrong forecast error buffer')
    with torch.no_grad():
        for layer_index, layer in enumerate(LAYERS):
            for module_index in range(len(histories[0])):
                dest = out[layer_index, module_index] if structure == 'mixed' else out[layer_index]
                now, recent, previous = (banks[module_index][layer] for banks in histories)
                torch.sub(recent, previous, out=dest)
                dest.mul_(float(ratio)).add_(recent)
                torch.sub(now, dest, out=dest)
    return out


def _arithmetic(x):
    return x if x.dtype == torch.float64 else x.float()


def _original_gate(origin, full, delta, gate):
    # Do the scalar arithmetic in FP32 for BF16 replay. At g=1, this is the
    # same rounded add as the ungated module, and g=0 retains its exact input.
    return (_arithmetic(origin) + gate * _arithmetic(delta)).to(full.dtype)


def _error_gate(full, error, gate):
    # Detached native error changes the perturbation direction, not the Full
    # path derivative. Attention's gradient subsequently passes through FFN.
    return (_arithmetic(full) - (1 - gate) * error.detach()).to(full.dtype)


class ModuleModel(Full10Model):
    def __init__(self, full_model, structure='ffn'):
        self.__dict__.update(full_model.__dict__)
        gate_shape(structure)
        self.structure = structure

    def attention(self, x, i, rope, update=True):
        w = self.w
        q, k, v = (tm.rms(x) @ w['encoder_attn_qkv_w'][i]).split((2048, 256, 256), -1)
        k = tm.rope(k, rope)
        if not update:
            return x, k, v, None
        delta = tm.attend(tm.rope(q, rope), k, v) @ w['encoder_attn_o_w'][i]
        return x + delta, k, v, delta

    def feed_forward(self, x, i):
        w = self.w
        n = tm.rms(x)
        h = F.gelu(n @ w['encoder_ffn_gate_w'][i], approximate='tanh') * (n @ w['encoder_ffn_up_w'][i])
        delta = h @ w['encoder_ffn_down_w'][i]
        return x + delta, delta

    def encode(self, item, gates=None, skip=(), patch=None, trace=False,
               start=0, prefix=None):
        if skip or patch is not None or start or prefix is not None:
            raise ValueError('Scoring is defined on the complete Full path')
        x = item['x']
        error = item.get('gate_error')
        shape = gate_shape(self.structure)
        if gates is not None and gates.shape != shape:
            raise ValueError(f'Expected gates shaped {shape}')
        if error is not None and error.shape != (*shape, *x.shape):
            raise ValueError('Forecast error does not match gate/input coordinates')
        ks, vs, inputs, after_attn, outputs = [], [], [], [], []
        for i in range(18):
            inp = x
            a, k, v, ra = self.attention(inp, i, item['encoder_rope'], update=i < 17)
            if i < 17:
                if gates is not None and self.structure == 'mixed':
                    a = (_error_gate(a, error[i, 0], gates[i, 0]) if error is not None
                         else _original_gate(inp, a, ra, gates[i, 0]))
                y, rf = self.feed_forward(a, i)
                if gates is None:
                    x = y
                elif self.structure == 'block':
                    x = (_error_gate(y, error[i], gates[i]) if error is not None
                         else gates[i] * y + (1 - gates[i]) * inp)
                else:
                    g = gates[i, 1] if self.structure == 'mixed' else gates[i]
                    e = error[i, 1] if error is not None and self.structure == 'mixed' else (error[i] if error is not None else None)
                    x = (_error_gate(y, e, g) if e is not None else _original_gate(a, y, rf, g))
            else:
                x = a
            ks.append(k)
            vs.append(v)
            if trace:
                inputs.append(inp)
                after_attn.append(a)
                outputs.append(x)
        result = dict(K=ks, V=vs)
        if trace:
            result.update(inputs=inputs, after_attn=after_attn, outputs=outputs)
        return result


def projected_loss(raw, coeff):
    # A double toy model retains double precision for finite-difference checks.
    return (_arithmetic(raw[:5, :7]) * coeff).sum()


def calculate(model, item, coeff, gates):
    with torch.enable_grad():
        raw = model.forward(item, gates=gates)
        grad = torch.autograd.grad(projected_loss(raw, coeff), gates)[0]
    return raw.detach(), grad.detach()


def finite_difference(model, item, coeff, indices=None, epsilon=1e-4):
    """Central differences on selected gate coordinates; does not alter weights."""
    shape = gate_shape(model.structure)
    dtype = torch.float64 if item['x'].dtype == torch.float64 else torch.float32
    gates = torch.ones(shape, device=item['x'].device, dtype=dtype,
                       requires_grad=True)
    coeff = torch.as_tensor(coeff, device=gates.device, dtype=gates.dtype)
    _, grad = calculate(model, item, coeff, gates)
    coordinates = list(np.ndindex(shape)) if indices is None else [
        (int(i),) if isinstance(i, (int, np.integer)) else tuple(i) for i in indices]
    rows = []
    with torch.no_grad():
        for coordinate in coordinates:
            plus, minus = gates.detach().clone(), gates.detach().clone()
            plus[coordinate] += epsilon
            minus[coordinate] -= epsilon
            estimate = (projected_loss(model.forward(item, gates=plus), coeff) -
                        projected_loss(model.forward(item, gates=minus), coeff)) / (2 * epsilon)
            rows.append(dict(index=list(coordinate), autograd=float(grad[coordinate]),
                             finite_difference=float(estimate),
                             absolute_error=float((grad[coordinate] - estimate).abs())))
    return rows


class Selector:
    """One shared Full-D10 forward and one positive weighted reverse pass.

    run returns (signed_gradient, absolute_scores, timing), never a mask.
    FFN and block coordinates are (17,), mixed coordinates are (17,2), A/F.
    error=None selects raw contributions; FP32 error selects neutral E gates.
    """
    def __init__(self, backend, structure='ffn', use_graph=True):
        gate_shape(structure)
        self.backend = backend
        self.structure = structure
        self.use_graph = use_graph
        self.clear()

    def clear(self):
        self.graph = self.model = self.item = self.raw = self.grad = None
        self.last_raw = None
        self.W = self.g = self.signature = self.runtime_id = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def capture(self, c, item, coeff, structure):
        tick = time.perf_counter()
        self.model = ModuleModel(borrow(self.backend, c), structure)
        self.runtime_id = id(c['runtime'])
        self.item = {name: value.clone() for name, value in item.items()}
        self.W = coeff.clone()
        self.g = torch.ones(gate_shape(structure), device=item['x'].device,
                            dtype=torch.float32, requires_grad=True)
        stream = torch.cuda.Stream(device=item['x'].device)
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

    def run(self, c, seed, coeff, error=None, structure=None, use_graph=None):
        tick = time.perf_counter()
        structure = self.structure if structure is None else structure
        gate_shape(structure)
        item = current_item(c, seed)
        if error is not None:
            error = residual_tensor(error, structure) if isinstance(error, dict) else error
            if error.dtype != torch.float32 or error.shape != (*gate_shape(structure), *item['x'].shape):
                raise ValueError('Expected matching FP32 forecast-error tensor')
            if error.device != item['x'].device:
                raise ValueError('Forecast error must be on the replay input device')
            item['gate_error'] = error.detach()
        coeff = torch.as_tensor(coeff, device=item['x'].device, dtype=torch.float32)
        if coeff.shape not in ((5, 1), (5, 7)) or not bool(torch.isfinite(coeff).all()):
            raise ValueError('Expected finite action projection shaped (5,1) or (5,7)')
        graph = self.use_graph if use_graph is None else use_graph
        cold = 0.0
        if graph:
            signature = (id(c['runtime']), structure,
                         tuple((name, tuple(value.shape), value.dtype, value.device)
                               for name, value in item.items()), tuple(coeff.shape))
            if self.graph is None or self.signature != signature:
                if self.graph is not None:
                    self.clear()
                cold = self.capture(c, item, coeff, structure)
                self.signature = signature
            for name, value in item.items():
                self.item[name].copy_(value)
            self.W.copy_(coeff)
            self.graph.replay()
            self.last_raw = self.raw
            gradient = self.grad.float().cpu().numpy().copy()
        else:
            model = ModuleModel(borrow(self.backend, c), structure)
            gates = torch.ones(gate_shape(structure), device=item['x'].device,
                               dtype=torch.float32, requires_grad=True)
            raw, grad = calculate(model, item, coeff, gates)
            self.last_raw = raw
            gradient = grad.float().cpu().numpy().copy()
        if not np.isfinite(gradient).all():
            raise ValueError('Nonfinite module-direction gradient')
        torch.cuda.synchronize()
        total = (time.perf_counter() - tick) * 1000
        return gradient, np.abs(gradient), dict(selector_total_ms=total,
            selector_hot_ms=total-cold, selector_capture_ms=cold,
            selector_forward_count=1, steady_vjp_count=1)


def verify_eager_graph(selector, c, seed, coeff, error=None, structure=None):
    """Actual GPU replay check with the experiment's predeclared tolerances."""
    graph_grad, graph_score, graph_timing = selector.run(
        c, seed, coeff, error=error, structure=structure, use_graph=True)
    graph_raw = selector.last_raw.float().cpu().numpy().copy()
    eager_grad, eager_score, eager_timing = selector.run(
        c, seed, coeff, error=error, structure=structure, use_graph=False)
    eager_raw = selector.last_raw.float().cpu().numpy().copy()
    checks = dict(raw_equal=bool(np.allclose(graph_raw, eager_raw, atol=1e-5, rtol=1e-5)),
                  gradient_equal=bool(np.allclose(graph_grad, eager_grad, atol=1e-4, rtol=1e-3)),
                  score_equal=bool(np.allclose(graph_score, eager_score, atol=1e-4, rtol=1e-3)))
    if not all(checks.values()):
        raise AssertionError(checks)
    return dict(**checks, raw_max_abs=float(np.max(np.abs(graph_raw-eager_raw))),
                gradient_max_abs=float(np.max(np.abs(graph_grad-eager_grad))),
                graph_timing=graph_timing, eager_timing=eager_timing)
