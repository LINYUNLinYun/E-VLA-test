"""CPU checks for capture-only hooks; actual CUDA ordering needs GPU tests."""
from types import SimpleNamespace
from contextlib import nullcontext
import unittest
from unittest.mock import Mock, patch

import torch
from transformers.cache_utils import DynamicCache

from .layer_graphs import WaitingCache, record_layer_ready, TwoIterationGraphs, LayerGraphs


class PhaseHandoffTests(unittest.TestCase):
    def test_tail_waits_for_early_graph_and_full_prefix(self):
        operations = []
        early = object()
        tail = SimpleNamespace(wait_event=lambda event: operations.append(("wait", event)))
        handoff = SimpleNamespace(record=lambda stream: operations.append(("record", stream)))
        graph = object.__new__(TwoIterationGraphs)
        graph.stream, graph.tail_stream, graph.handoff = early, tail, handoff
        graph.graphs = [SimpleNamespace(replay=lambda: operations.append(("early",))),
                        SimpleNamespace(replay=lambda: operations.append(("tail",)))]
        prefix_done = object()
        with patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()):
            graph.replay(prefix_done=prefix_done)
        self.assertEqual(operations, [("early",), ("record", early),
            ("wait", handoff), ("wait", prefix_done), ("tail",)])

    def test_native_split_does_not_add_full_prefix_dependency(self):
        operations = []
        stream = SimpleNamespace(wait_event=lambda event: operations.append("handoff_wait"))
        graph = object.__new__(TwoIterationGraphs)
        graph.stream = graph.tail_stream = stream
        graph.handoff = SimpleNamespace(record=lambda stream: operations.append("record"))
        graph.graphs = [SimpleNamespace(replay=lambda: operations.append("early")),
                        SimpleNamespace(replay=lambda: operations.append("tail"))]
        with patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()):
            graph.replay()
        self.assertEqual(operations, ["early", "record", "handoff_wait", "tail"])

    def test_resource_switch_requires_split_and_producer_gate(self):
        backend = SimpleNamespace(num_steps=10)
        with self.assertRaisesRegex(ValueError, "split AE"):
            LayerGraphs(backend, None, None, None, object(), tail_stream=object())
        with self.assertRaisesRegex(ValueError, "full Prefix"):
            LayerGraphs(backend, None, None, None, object(), ae_layout="per_step", tail_stream=object())


class CaptureHookTests(unittest.TestCase):
    def model(self, count):
        layers = [torch.nn.Identity() for _ in range(count)]
        return SimpleNamespace(paligemma_with_expert=SimpleNamespace(
            paligemma=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))), layers

    def test_wait_is_before_first_read_and_only_once_per_layer(self):
        operations = []

        class Cache:
            def __getitem__(self, index):
                operations.append(("read", index))
                return (index, index + 1)

            def get_seq_length(self):
                return 123

        external = SimpleNamespace(wait=lambda event, stream: operations.append(("wait", event)))
        proxy = WaitingCache(Cache(), ("zero", "one"), external, object())
        self.assertEqual(proxy.get_seq_length(), 123)
        self.assertEqual(proxy[1][0], 1)
        self.assertEqual(proxy[1][1], 2)
        self.assertEqual(operations, [("wait", "one"), ("read", 1), ("read", 1)])
        with self.assertRaisesRegex(RuntimeError, "must not mutate"):
            proxy.update(None)

    def test_kv_mark_follows_original_update_and_group_tail(self):
        model, _ = self.model(5)
        operations = []
        external = SimpleNamespace(mark=lambda event, stream: operations.append(("ready", event)))

        def original(cache, key, value, index, kwargs):
            operations.append(("update", index))
            return key, value

        with patch.object(DynamicCache, "update", original):
            with record_layer_ready(model, tuple(range(5)), external, None, "kv_ready", 4):
                for index in range(5):
                    self.assertEqual(DynamicCache.update(None, "key", "value", index), ("key", "value"))
            self.assertIs(DynamicCache.update, original)
        self.assertEqual(operations, [*(('update', i) for i in range(4)),
            *(('ready', i) for i in range(4)), ('update', 4), ('ready', 4)])

    def test_update_restored_after_capture_failure(self):
        model, _ = self.model(2)
        original = DynamicCache.update
        with self.assertRaisesRegex(ValueError, "capture failed"):
            with record_layer_ready(model, (object(), object()), Mock(), None, "kv_ready", 1):
                raise ValueError("capture failed")
        self.assertIs(DynamicCache.update, original)

    def test_block_mark_follows_output_and_hooks_are_removed(self):
        model, layers = self.model(2)
        operations = []
        external = SimpleNamespace(mark=lambda event, stream: operations.append(event))
        with record_layer_ready(model, (0, 1), external, None, "block_ready", 1):
            for layer in layers:
                layer(torch.zeros(1))
        self.assertEqual(operations, [0, 1])
        for layer in layers:
            layer(torch.zeros(1))
            self.assertFalse(layer._forward_hooks)
        self.assertEqual(operations, [0, 1])


class VisionHandoffTests(unittest.TestCase):
    def test_vlm_waits_for_separate_encoder_but_default_adds_no_wait(self):
        for separate in (False, True):
            operations = []
            prefix_stream = SimpleNamespace(wait_event=lambda event: operations.append(("wait", event)))
            graph = object.__new__(LayerGraphs)
            graph.prefix_stream = prefix_stream
            graph.vision_stream = object() if separate else prefix_stream
            identity, metadata, previous = object(), object(), object()
            slot = SimpleNamespace(encoded_identity=identity, producer_identity=None,
                metadata_ready=metadata, layer_ready=(), prefix=object(),
                prefix_graph=SimpleNamespace(replay=lambda: operations.append(("replay",))))
            graph._exclusive = lambda pool, selected: slot
            graph._submit_label = lambda *args: "test"
            done = SimpleNamespace(record=lambda stream: operations.append(("record", stream)))
            with patch("torch.cuda.stream", side_effect=lambda stream: nullcontext()), \
                    patch("torch.cuda.nvtx.range", side_effect=lambda name: nullcontext()), \
                    patch("torch.cuda.Event", return_value=done):
                result = graph.prefill(None, identity, dependency=previous)
            expected = ([("wait", metadata)] if separate else [])
            self.assertEqual(operations, expected + [("wait", previous), ("replay",), ("record", prefix_stream)])
            self.assertIs(slot.producer_identity, identity)
            self.assertIs(result[1], done)


if __name__ == "__main__":
    unittest.main()
