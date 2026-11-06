import unittest

from .scheduler import Identity, Observation, Pool


class Event:
    def __init__(self, done=False):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.pool = Pool(3, "task")

    def obs(self, index, payload=None):
        return Observation(index, 3, "task", payload)

    def build(self, index, done=True):
        self.pool.offer(self.obs(index))
        identity, obs = self.pool.reserve()
        ready = Event(done)
        self.pool.submitted(identity, {"owned": index}, ready)
        self.pool.poll()
        return identity, ready

    def test_not_published_until_complete(self):
        identity, ready = self.build(0, False)
        self.assertIsNone(self.pool.acquire())
        ready.done = True
        self.assertEqual(self.pool.acquire().identity, identity)

    def test_slow_reader_backpressure_and_latest_pending(self):
        first, _ = self.build(0)
        lease = self.pool.acquire()
        reader = Event()
        self.pool.release(lease, reader)
        second, _ = self.build(1)
        self.assertEqual(self.pool.active, second)
        self.pool.offer(self.obs(2))
        self.pool.offer(self.obs(3))
        self.assertEqual(self.pool.pending_replacements, 1)
        self.assertIsNone(self.pool.reserve())
        self.assertEqual(self.pool.live_snapshots, 2)
        reader.done = True
        identity, observation = self.pool.reserve()
        self.assertEqual(observation.observation_id, 3)
        self.assertEqual(identity.slot, first.slot)
        self.assertGreater(identity.generation, first.generation)
        with self.assertRaises(RuntimeError):
            self.pool.submitted(first, {}, Event(True))

    def test_lease_pins_version_during_publication(self):
        first, _ = self.build(0)
        lease = self.pool.acquire()
        self.build(1)
        self.assertEqual(lease.identity, first)
        self.assertEqual(lease.value, {"owned": 0})
        self.assertEqual(self.pool.slots[first.slot].state, "RETIRED")
        self.pool.release(lease, Event(True))
        self.assertEqual(self.pool.slots[first.slot].state, "FREE")
        self.assertIsNone(lease.value)
        self.assertIsNone(lease.observation)

    def test_reset_drains_producer_and_rejects_old_callback(self):
        identity, ready = self.build(0, False)
        self.pool.drain_reset(4, "other")
        self.assertTrue(ready.done)
        self.assertEqual(self.pool.live_snapshots, 0)
        with self.assertRaises(RuntimeError):
            self.pool.submitted(identity, {}, Event(True))
        self.assertFalse(self.pool.offer(self.obs(1)))

    def test_reset_requires_generation_finished(self):
        self.build(0)
        lease = self.pool.acquire()
        with self.assertRaises(RuntimeError):
            self.pool.drain_reset(4, "other")
        reader = Event()
        self.pool.release(lease, reader)
        self.pool.drain_reset(4, "other")
        self.assertTrue(reader.done)

    def test_pending_newer_does_not_starve_completed_prefix(self):
        identity, ready = self.build(0, False)
        self.pool.offer(self.obs(10))
        ready.done = True
        self.pool.poll()
        self.assertEqual(self.pool.active, identity)
        self.assertEqual(self.pool.pending.observation_id, 10)

    def test_duplicate_release_is_error(self):
        self.build(0)
        lease = self.pool.acquire()
        self.pool.release(lease, Event(True))
        with self.assertRaises(RuntimeError):
            self.pool.release(lease, Event(True))


if __name__ == "__main__":
    unittest.main()
