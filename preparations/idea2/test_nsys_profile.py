"""CPU-only instrumentation proofs; no torch, model, CUDA or nsys required."""
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import patch

from preparations.idea2.nsys_profile import NVTX, compare, main


class Layer:
    def forward(self, value):
        return value + 1

    def __call__(self, value):
        return self.forward(value)


class Stack:
    def __init__(self, depth):
        self.layers = [Layer() for _ in range(depth)]

    def forward(self, value):
        # Both call forms must remain instrumented.
        for i, layer in enumerate(self.layers):
            value = layer(value) if i % 2 else layer.forward(value)
        return value


class Bundle:
    def __init__(self):
        self.paligemma = NS(vision_tower=NS(vision_model=NS(encoder=Stack(27))),
                           language_model=Stack(18), multi_modal_projector=Layer())
        self.gemma_expert = NS(model=Stack(18))

    def embed_image(self, value):
        return self.paligemma.multi_modal_projector(
            self.paligemma.vision_tower.vision_model.encoder.forward(value))


class Model:
    def __init__(self):
        self.paligemma_with_expert = Bundle()

    def embed_prefix(self, value):
        return sum(self.paligemma_with_expert.embed_image(value) for _ in range(3))

    def denoise_step(self, value):
        return self.paligemma_with_expert.gemma_expert.model.forward(value)


class Tensor:
    _version = 0
    shape = (2, 3)
    dtype = "float32"

    def data_ptr(self):
        return id(self)


class Backend:
    num_steps = 10

    def __init__(self):
        self.model = Model()

    def build_prefix(self, value):
        value = self.model.embed_prefix(value)
        self.model.paligemma_with_expert.paligemma.language_model.forward(value)
        return NS(cache=[Tensor()])

    def generate(self, prefix, value, noise):
        for _ in range(self.num_steps):
            value = self.model.denoise_step(value)
        return value + noise

    def prepare(self, value):
        return value

    def noise(self, value):
        return value

    def output(self, prepared, value):
        return value


class Sink:
    def __init__(self):
        self.stack, self.labels = [], []

    def range_push(self, label):
        self.stack.append(label)
        self.labels.append(label)

    def range_pop(self):
        self.stack.pop()


class ProfileTests(unittest.TestCase):
    def make(self, coarse=False):
        backend, sink = Backend(), Sink()
        trace = NVTX(backend, sink, tuple, coarse)
        trace.install(["base", "left", "right"])
        trace.enabled = True
        return backend, sink, trace

    def test_detailed_direct_forward_and_counts_restored(self):
        backend, sink, trace = self.make()
        for _ in range(2):
            prefix = backend.build_prefix(1)
            backend.output(1, backend.generate(prefix, 1, 2))
        proof = trace.validate(2)
        self.assertEqual(proof["counts"]["vision/L27"], 6)
        self.assertEqual(proof["counts"]["vlm/L18"], 2)
        self.assertEqual(proof["counts"]["action_expert/denoise_step10/layer18"], 2)
        self.assertFalse(sink.stack)
        self.assertTrue(trace.restore())
        self.assertNotIn("forward", vars(backend.model.paligemma_with_expert.gemma_expert.model.layers[0]))
        self.assertNotIn("generate", vars(backend))
        n = len(sink.labels)
        backend.generate(prefix, 1, 2)
        self.assertEqual(len(sink.labels), n)

    def test_coarse_keeps_numerics_and_no_layer_labels(self):
        backend, sink, trace = self.make(True)
        prefix = backend.build_prefix(0)
        expected = Backend().generate(prefix, 3, 4)
        self.assertEqual(backend.generate(prefix, 3, 4), expected)
        backend.output(0, expected)
        proof = trace.validate(1)
        self.assertFalse(proof["layer_counts_required"])
        self.assertFalse(any("/layer" in x or "/L" in x for x in sink.labels))
        self.assertIn("action_expert/denoise_step10", sink.labels)
        trace.restore()

    def test_exception_balances_ranges_and_restores_instance_override(self):
        backend = Backend()
        def fail(value):
            raise RuntimeError("deliberate")
        target = backend.model.paligemma_with_expert.gemma_expert.model.layers[2]
        target.forward = fail
        sink = Sink()
        trace = NVTX(backend, sink, tuple).install(["base", "left", "right"])
        trace.enabled = True
        prefix = backend.build_prefix(0)
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            backend.generate(prefix, 0, 0)
        self.assertEqual(sink.stack, [])
        self.assertEqual(trace.step, 0)
        trace.restore()
        self.assertIs(target.forward, fail)
        self.assertTrue(trace.restore())

    def test_version_mutation_and_missing_layer_rejected(self):
        backend, _, trace = self.make()
        prefix = backend.build_prefix(0)
        original = backend.model.denoise_step
        def mutate(value):
            prefix.cache[0]._version += 1
            return original(value)
        backend.model.denoise_step = mutate
        with self.assertRaisesRegex(AssertionError, "Prefix storage/version"):
            backend.generate(prefix, 0, 0)
        trace.restore()
        backend, _, trace = self.make()
        prefix = backend.build_prefix(0)
        backend.output(0, backend.generate(prefix, 0, 0))
        trace.counts["vlm/L01"] -= 1
        with self.assertRaises(AssertionError):
            trace.validate(1)
        trace.restore()

    def test_policy_reset_and_drain_labels(self):
        backend, sink, trace = self.make()
        class Pool:
            def drain_reset(self):
                return 7
        class Policy:
            pool = Pool()
            def reset(self):
                return self.pool.drain_reset()
        policy = Policy()
        trace.attach_policy(policy)
        self.assertEqual(policy.reset(), 7)
        self.assertEqual(sink.labels, ["runtime/reset", "runtime/drain_owned_events"])
        trace.restore()
        self.assertNotIn("reset", vars(policy))
        self.assertNotIn("drain_reset", vars(policy.pool))

    def test_cli_dry_run_no_run_no_files_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"new"
            argv = ["profile", "--bank", "not-read.pkl", "--output", str(output), "--coarse-only"]
            with patch("sys.argv", argv), patch("preparations.idea2.nsys_profile.run") as run, contextlib.redirect_stdout(io.StringIO()) as stdout:
                main()
            self.assertFalse(run.called)
            self.assertFalse(output.exists())
            self.assertTrue(json.loads(stdout.getvalue())["dry_run"])
            output.mkdir()
            with patch("sys.argv", argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main()

    def test_depth_and_absolute_tolerance(self):
        backend = Backend()
        backend.model.paligemma_with_expert.gemma_expert.model.layers.pop()
        with self.assertRaises(AssertionError):
            NVTX(backend, Sink(), tuple)
        self.assertTrue(compare([1], [1.000001])["passed"])
        self.assertFalse(compare([100], [100.0001])["passed"])


if __name__ == "__main__":
    unittest.main()
