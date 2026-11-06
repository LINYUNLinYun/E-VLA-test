"""CPU dependency checks for the fixed-input, two-slot pairing diagnostic."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

from .sm_sensitivity_bench import measure_encoder_pair


class EncoderPairTests(unittest.TestCase):
    def run_pair(self, mode, vision_first):
        operations, events = [], []

        class Stream:
            def __init__(self, name):
                self.name = name

            def wait_event(self, event):
                # Ordinary CUDA waits on not-yet-recorded Events can be no-ops.
                if not event.recorded:
                    raise AssertionError("wait submitted before producer record")
                operations.append(("wait", self.name, event.index))

        class Event:
            def __init__(self, **kwargs):
                self.index, self.recorded = len(events), False
                events.append(self)

            def record(self, stream):
                self.recorded = True
                operations.append(("record", stream.name, self.index))

            def synchronize(self):
                operations.append(("host_wait", self.index))

            def elapsed_time(self, other):
                return 1.0

        primary, ae_stream, vision_stream = map(Stream, ("primary", "ae", "vision"))
        torch = SimpleNamespace(cuda=SimpleNamespace(Event=Event,
            current_stream=lambda: primary, stream=lambda stream: nullcontext()))
        slot0 = SimpleNamespace(ae_graph=SimpleNamespace(replay=lambda: operations.append(("ae_slot", 0))))
        slot1 = SimpleNamespace(vision_graph=SimpleNamespace(replay=lambda: operations.append(("vision_slot", 1))))
        graphs = SimpleNamespace(slots=[slot0, slot1], ae_stream=ae_stream, prefix_stream=vision_stream)
        row = measure_encoder_pair(torch, graphs, mode, vision_first)
        self.assertEqual([item for item in operations if item[0] == "host_wait"], [("host_wait", 1)])
        self.assertIn(("wait", "primary", 3), operations)
        self.assertIn(("wait", "primary", 5), operations)
        return operations, row

    def test_serial_waits_for_ae_before_other_slot_vision(self):
        operations, _ = self.run_pair("serial", True)
        self.assertLess(operations.index(("record", "ae", 3)), operations.index(("wait", "vision", 3)))
        self.assertLess(operations.index(("wait", "vision", 3)), operations.index(("vision_slot", 1)))

    def test_overlap_has_no_interbranch_wait_in_either_order(self):
        for vision_first in (False, True):
            operations, row = self.run_pair("overlap", vision_first)
            self.assertEqual([(kind, stream, event) for kind, stream, event in
                              (op for op in operations if len(op) == 3 and op[0] == "wait")
                              if stream in ("ae", "vision")],
                             [("wait", "ae", 0), ("wait", "vision", 0)])
            self.assertEqual(operations.index(("vision_slot", 1)) < operations.index(("ae_slot", 0)), vision_first)
            self.assertEqual(row["submission_order"], "vision_first" if vision_first else "ae_first")


if __name__ == "__main__":
    unittest.main()
