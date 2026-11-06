"""Ownership rejection and dynamic-mask parity without requiring a GPU."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from .runtime_graph import DynamicCore, RuntimeGraphs
from .scheduler import Observation, Pool
from .test_submission import Event


class RuntimeGraphTests(unittest.TestCase):
    def setUp(self):
        self.pool = Pool(task="task")
        self.pool.offer(Observation(0,0,"task",None))
        self.identity,_ = self.pool.reserve()
        self.prefix = object()
        self.graphs = RuntimeGraphs.__new__(RuntimeGraphs)
        self.graphs.slots = [SimpleNamespace(prefix=self.prefix),SimpleNamespace(prefix=object())]

    def test_cannot_overwrite_published_or_leased_slot(self):
        self.pool.submitted(self.identity,self.prefix,Event(True))
        self.pool.poll()
        lease = self.pool.acquire()
        with self.assertRaisesRegex(RuntimeError,"exclusive Pool reservation"):
            self.graphs.build_prefix(self.pool,self.identity,None)
        self.pool.release(lease,Event(True))
        with self.assertRaisesRegex(RuntimeError,"exclusive Pool reservation"):
            self.graphs.build_prefix(self.pool,self.identity,None)

    def test_reservation_with_reader_is_rejected(self):
        self.pool.slots[0].readers.append(Event(False))
        with self.assertRaisesRegex(RuntimeError,"exclusive Pool reservation"):
            self.graphs.build_prefix(self.pool,self.identity,None)

    def test_stale_generation_and_released_lease_are_rejected(self):
        self.pool.submitted(self.identity,self.prefix,Event(True))
        lease = self.pool.acquire()
        self.pool.release(lease,Event(True))
        with self.assertRaisesRegex(RuntimeError,"live lease"):
            self.graphs.generate(self.pool,lease,None,None)
        self.pool.drain_reset(1,"task")
        with self.assertRaisesRegex(RuntimeError,"Stale slot"):
            self.graphs.build_prefix(self.pool,self.identity,None)

    def test_lease_cannot_read_another_slots_capture(self):
        self.pool.submitted(self.identity,object(),Event(True))
        lease = self.pool.acquire()
        with self.assertRaisesRegex(RuntimeError,"live lease"):
            self.graphs.generate(self.pool,lease,None,None)

    def test_dynamic_masks_preserve_original_d10_math(self):
        from . import graph_sampler,openpi_backend
        original_prepared = openpi_backend.Prepared
        counter = []
        def prepared(*args,**kwargs):
            value = original_prepared(*args,**kwargs)
            value.token_masks[0,len(counter)%2] = False
            if counter:
                value.image_masks[0].fill_(False)
            counter.append(True)
            return value
        with patch.object(graph_sampler,"GraphSafeBackend",lambda backend,current,nxt:DynamicCore(backend,current)), \
             patch.object(openpi_backend,"Prepared",prepared):
            self.assertTrue(graph_sampler.cpu_self_test()["passed"])


if __name__ == "__main__":
    unittest.main()
