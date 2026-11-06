"""CPU algebra, neutral-gate, and finite-difference checks; not GPU timing."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from preparations.idea1_v3_4_E2_E4_single.torch_model import Model
from .model import ErrorModel, forecast_error


class TinyFull(Model):
    def __init__(self):
        g = torch.Generator().manual_seed(9011)
        self.a = torch.randn(18, 8, 8, generator=g, dtype=torch.float64) * .08
        self.b = torch.randn(18, 8, generator=g, dtype=torch.float64) * .03
        self.out = torch.randn(8, 50*32, generator=g, dtype=torch.float64) * .1

    def block(self, x, i, r, update=True):
        return (x + .1 * torch.tanh(x @ self.a[i] + self.b[i])
                if update else x), x * .3, x * .7

    def sample(self, item, c):
        z = torch.stack([v.mean(0) for v in c['V']]).mean(0)
        return torch.tanh(z @ self.out).reshape(50, 32)


class TinyError(ErrorModel):
    block = TinyFull.block
    sample = TinyFull.sample


def check():
    torch.manual_seed(312)
    base = TinyFull()
    new = TinyError(base)
    item = dict(x=torch.randn(4, 8, dtype=torch.float64), encoder_rope=None,
                gate_error=torch.randn(17, 4, 8, dtype=torch.float64) * .08)
    W = torch.tensor([1., 2., 4., 8., 16.], dtype=torch.float64)[:, None] / 31
    g = torch.ones(17, dtype=torch.float64, requires_grad=True)
    full = base.forward(item, gates=g)
    raw = new.forward(item, gates=g)
    assert torch.equal(full, raw), 'Neutral gate changed Full'
    loss = (raw[:5, :7] * W).sum()
    grad = torch.autograd.grad(loss, g)[0]
    fd = []
    eps = 1e-4
    with torch.no_grad():
        for i in range(17):
            plus, minus = g.detach().clone(), g.detach().clone()
            plus[i] += eps
            minus[i] -= eps
            fd.append(float(((new.forward(item, gates=plus)[:5, :7] -
                              new.forward(item, gates=minus)[:5, :7]) * W).sum()
                            / (2 * eps)))
    assert np.allclose(grad.numpy(), fd, atol=1e-8, rtol=1e-5)
    zero = dict(item, gate_error=torch.zeros_like(item['gate_error']))
    zero_raw = new.forward(zero, gates=g)
    zero_grad = torch.autograd.grad((zero_raw[:5, :7]*W).sum(), g)[0]
    assert torch.equal(zero_grad, torch.zeros_like(zero_grad))
    banks = [{i: torch.randn(7, 8) for i in range(1, 18)} for _ in range(3)]
    for ratio in [1/3, 2/3, 1., 1.5]:
        actual = forecast_error(*banks, ratio)
        expected = torch.stack([banks[0][i] - (banks[1][i] + ratio *
                              (banks[1][i]-banks[2][i])) for i in range(1, 18)])
        assert torch.equal(actual, expected)
    return dict(passed=True,neutral_gate_bitwise_equal=True,
                finite_difference_layers=17,
                finite_difference_max_abs=float(abs(grad.numpy()-fd).max()),
                zero_error_zero_gradient=True,forecast_formula_bitwise_equal=True,
                scope='CPU tiny nonlinear model; not actual VLA or GPU timing')


def cpu_math(repeats=40):
    # Full-size upper observed shape from the previous 40-case dynamic probe.
    # This is reported strictly as CPU elementwise timing, never a GPU estimate.
    torch.set_num_threads(4)
    shape = (530, 2048)
    banks = [{i: torch.randn(shape) for i in range(1, 18)} for _ in range(3)]
    out = torch.empty((17, *shape))
    for _ in range(10):
        forecast_error(*banks, out=out)
    times = []
    for _ in range(repeats):
        tick = time.perf_counter()
        forecast_error(*banks, out=out)
        times.append((time.perf_counter()-tick)*1000)
    return dict(device='cpu',threads=4,dtype='float32',shape=list(out.shape),
                repeats=repeats,median_ms=float(np.median(times)),
                mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),
                allocated_output_mib=out.numel()*out.element_size()/2**20,
                scope='Only forecast error arithmetic; not GPU/VLA latency')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--cpu-math', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    result = dict(functional=check())
    if a.cpu_math:
        result['cpu_math'] = cpu_math()
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'cpu_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
