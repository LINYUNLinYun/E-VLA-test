import unittest

import torch

from .openpi_backend import OpenPIBackend, Prefix


class FakeModel:
    def denoise_step(self, state, masks, cache, noise, timestep):
        self.times.append(timestep.clone())
        return noise * 0.2 + timestep[:, None, None]


class SamplingTests(unittest.TestCase):
    def test_fixed_loop_preserves_original_scalar_recursion_and_output_ownership(self):
        backend = object.__new__(OpenPIBackend)
        backend.model = FakeModel()
        backend.model.times = []
        backend.device = torch.device("cpu")
        backend.num_steps = 10
        prepared = type("Prepared", (), {"state": torch.zeros(1, 2)})()
        prefix = Prefix(cache=None, pad_masks=torch.ones(1, 2), position_ids=None)
        noise = torch.ones(1, 5, 2)
        dt = torch.tensor(-0.1, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32)
        reference = noise
        expected_times = []
        while time >= -dt / 2:
            expected_times.append(time.clone())
            reference = reference + dt * (reference * 0.2 + time)
            time += dt
        result = backend.generate(prefix, prepared, noise)
        self.assertEqual(len(expected_times), 10)
        self.assertTrue(torch.equal(result, reference))
        self.assertTrue(all(torch.equal(a[0], b) for a, b in zip(backend.model.times, expected_times, strict=True)))
        self.assertTrue(torch.equal(noise, torch.ones_like(noise)))
        second = backend.generate(prefix, prepared, noise)
        self.assertNotEqual(result.data_ptr(), second.data_ptr())
        self.assertFalse(result.requires_grad)


if __name__ == "__main__":
    unittest.main()
