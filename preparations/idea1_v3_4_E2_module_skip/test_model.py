"""CPU checks of module gates, neutral errors, and central differences.

These use a nonlinear tiny chain, not the actual BF16 policy. Actual model
eager/CUDA-graph validation is exposed by model.verify_eager_graph.
"""
import json
import unittest

import numpy as np
import torch

from .model import ModuleModel, calculate, finite_difference, forecast_error, gate_shape


class TinyModule(ModuleModel):
    def __init__(self, structure='ffn'):
        self.structure = structure
        generator = torch.Generator().manual_seed(341101)
        self.aw = torch.randn(18, 8, 8, generator=generator, dtype=torch.float64) * .12
        self.fw = torch.randn(17, 8, 8, generator=generator, dtype=torch.float64) * .18
        self.out = torch.randn(8, 50*32, generator=generator, dtype=torch.float64) * .08

    def attention(self, x, i, rope, update=True):
        # K/V are formed from the current input, before this layer's A/F gates.
        k, v = torch.tanh(x * .3), torch.sin(x * .7)
        if not update:
            return x, k, v, None
        delta = .3 * torch.tanh(x @ self.aw[i])
        return x + delta, k, v, delta

    def feed_forward(self, x, i):
        delta = .4 * torch.tanh(x @ self.fw[i] + .1 * x.square())
        return x + delta, delta

    def sample(self, item, c):
        z = torch.stack([k.mean(0) + v.mean(0) for k, v in zip(c['K'], c['V'])]).mean(0)
        return torch.tanh(z @ self.out).reshape(50, 32)


def tiny_item():
    generator = torch.Generator().manual_seed(50101)
    return dict(x=torch.randn(4, 8, generator=generator, dtype=torch.float64),
                encoder_rope=None)


def cpu_verification():
    item = tiny_item()
    coeff = torch.tensor([1., 2., 4., 8., 16.], dtype=torch.float64)[:, None] / 31
    rows = []
    for structure in ('ffn', 'mixed', 'block'):
        model = TinyModule(structure)
        full = model.forward(item)
        shape = gate_shape(structure)
        gates = torch.ones(shape, dtype=torch.float64, requires_grad=True)
        raw, grad = calculate(model, item, coeff, gates)
        if not torch.equal(raw, full):
            raise AssertionError(f'{structure}: R all-one gate changed Full')
        error = torch.linspace(-.08, .09, int(np.prod(shape))*item['x'].numel(),
                               dtype=torch.float64).reshape(*shape, *item['x'].shape)
        error_item = dict(item, gate_error=error)
        error_raw, error_grad = calculate(model, error_item, coeff, gates)
        if not torch.equal(error_raw, full):
            raise AssertionError(f'{structure}: neutral E gate changed Full')
        differences = []
        for mode, sample in (('R', item), ('E', error_item)):
            checked = finite_difference(model, sample, coeff)
            autograd = [r['autograd'] for r in checked]
            numeric = [r['finite_difference'] for r in checked]
            if not np.allclose(autograd, numeric, atol=1e-8, rtol=1e-5):
                raise AssertionError((structure, mode, checked))
            differences.append(dict(mode=mode, gates_checked=len(checked),
                                    max_abs=max(r['absolute_error'] for r in checked)))
        zero_item = dict(item, gate_error=torch.zeros_like(error))
        _, zero_grad = calculate(model, zero_item, coeff, gates)
        if not torch.equal(zero_grad, torch.zeros_like(zero_grad)):
            raise AssertionError(f'{structure}: zero E direction has nonzero gradient')
        rows.append(dict(structure=structure, neutral_R_bitwise=True,
                         neutral_E_bitwise=True, zero_error_zero_gradient=True,
                         finite_difference=differences))
    return dict(passed=True, conditions=rows,
                scope='CPU float64 nonlinear tiny chain, no GPU/VLA equality claim')


class GateTests(unittest.TestCase):
    def test_neutral_and_finite_difference(self):
        self.assertTrue(cpu_verification()['passed'])

    def test_forecast_formula_and_buffer(self):
        generator = torch.Generator().manual_seed(71001)
        histories = [{name: {layer: torch.randn(4, 8, generator=generator)
                              for layer in range(1, 18)}
                      for name in ('attn', 'ffn')} for _ in range(3)]
        for structure in ('ffn', 'mixed'):
            for ratio in (0., .2, .8, 1., 1.5):
                actual = forecast_error(*histories, ratio, structure=structure)
                modules = ('attn', 'ffn') if structure == 'mixed' else ('ffn',)
                expected = torch.stack([torch.stack([
                    histories[0][name][layer] - (histories[1][name][layer] + ratio *
                    (histories[1][name][layer] - histories[2][name][layer]))
                    for name in modules]) for layer in range(1, 18)])
                if structure == 'ffn':
                    expected = expected[:, 0]
                self.assertTrue(torch.equal(actual, expected))
                same = forecast_error(*histories, ratio, out=actual, structure=structure)
                self.assertIs(same, actual)
                self.assertTrue(torch.equal(same, expected))
        blocks = [history['ffn'] for history in histories]
        actual = forecast_error(*blocks, .8, structure='block')
        self.assertTrue(torch.equal(actual, forecast_error(*histories, .8, structure='ffn')))
        with self.assertRaises(ValueError):
            forecast_error(*histories, structure='block')

    def test_shape_and_history_validation(self):
        histories = [{name: {layer: torch.zeros(4, 8) for layer in range(1, 18)}
                      for name in ('attn', 'ffn')} for _ in range(3)]
        for ratio in (-1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                forecast_error(*histories, ratio)
        histories[1]['ffn'][1] = torch.zeros(3, 8)
        with self.assertRaises(ValueError):
            forecast_error(*histories)
        model = TinyModule('mixed')
        with self.assertRaises(ValueError):
            model.forward(tiny_item(), gates=torch.ones(17, dtype=torch.float64))

    def test_current_kv_and_attention_before_ffn(self):
        model = TinyModule('mixed')
        item = tiny_item()
        gates = torch.ones(17, 2, dtype=torch.float64)
        trace = model.encode(item, gates=gates, trace=True)
        gates[0, 0] = 0
        changed = model.encode(item, gates=gates, trace=True)
        # A skip keeps this layer's real K/V and changes the input of this FFN.
        self.assertTrue(torch.equal(trace['K'][0], changed['K'][0]))
        self.assertTrue(torch.equal(trace['V'][0], changed['V'][0]))
        self.assertTrue(torch.equal(changed['after_attn'][0], item['x']))
        expected, _ = model.feed_forward(item['x'], 0)
        self.assertTrue(torch.equal(changed['outputs'][0], expected))
        self.assertFalse(torch.equal(trace['K'][1], changed['K'][1]))
        # The final KV-only layer also reflects the complete gated prefix.
        self.assertFalse(torch.equal(trace['K'][17], changed['K'][17]))


if __name__ == '__main__':
    print(json.dumps(cpu_verification(), indent=2), flush=True)
    unittest.main()
