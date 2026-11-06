"""Small nonlinear CPU check of shared-forward reverse passes and prefixes."""
import json
import numpy as np
import torch
from preparations.idea1_v3_4_E2_comp_error_cost.verify import TinyFull, TinyError
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3.io import atomic
from .model import calculate, projection_weights
from .common import OUT

def main():
    torch.set_num_threads(4)
    torch.manual_seed(312)
    model = TinyError(TinyFull())
    item = dict(x=torch.randn(4, 8, dtype=torch.float64), encoder_rope=None,
                gate_error=torch.randn(17, 4, 8, dtype=torch.float64) * .08)
    weights = projection_weights(4)
    assert np.array_equal(weights[0], coefficients())
    assert np.linalg.matrix_rank(weights.reshape(4, -1)) == 4
    checks = []
    largest = 0.
    for q in (1, 2, 3, 4):
        assert np.array_equal(projection_weights(q), weights[:q])
        W = torch.as_tensor(weights[:q], dtype=torch.float64)
        gates = torch.ones(17, dtype=torch.float64, requires_grad=True)
        raw, grads = calculate(model, item, W, gates)
        for j in range(q):
            independent, independent_grad = calculate(model, item, W[j:j+1], gates)
            assert torch.equal(raw, independent) and torch.equal(grads[j], independent_grad[0])
        if q == 4:
            eps = 1e-4
            fd = np.empty((4, 17))
            with torch.no_grad():
                for layer in range(17):
                    plus, minus = gates.detach().clone(), gates.detach().clone()
                    plus[layer] += eps
                    minus[layer] -= eps
                    delta = (model.forward(item, gates=plus)[:5, :7] -
                             model.forward(item, gates=minus)[:5, :7]) / (2*eps)
                    fd[:, layer] = (delta[None] * W).sum((1, 2)).numpy()
            largest = float(abs(fd - grads.numpy()).max())
            assert np.allclose(fd, grads.numpy(), atol=1e-8, rtol=1e-5)
        zero = dict(item, gate_error=torch.zeros_like(item['gate_error']))
        _, zg = calculate(model, zero, W, gates)
        assert torch.equal(zg, torch.zeros_like(zg))
        checks.append(dict(q=q,shared_independent_exact=True,zero_error_zero_gradient=True))
    result = dict(passed=True,checks=checks,finite_difference_max_abs=largest,
                  projection_weights=weights.tolist(),projection_rank=4,
                  scope='CPU nonlinear toy only; GPU/VLA checks are separate')
    OUT.mkdir(parents=True,exist_ok=True)
    atomic(OUT/'cpu_verification.json',result)
    print(json.dumps(result),flush=True)

if __name__ == '__main__':main()
