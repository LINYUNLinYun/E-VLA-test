"""CPU ownership/race regressions for opt-in same-frame layer dispatch."""
import threading
import unittest

from .layer_scheduler import LayerLease, LayerPool
from .scheduler import Identity, Lease, Observation, Pool


class Event:
    def __init__(self, done=False, predecessors=()):
        self.done = done
        self.predecessors = tuple(predecessors)
        self.syncs = 0

    def query(self):
        return self.done

    def synchronize(self):
        for event in self.predecessors:
            event.synchronize()
        self.syncs += 1
        self.done = True


class Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)


class LayerOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.pool = LayerPool(3, epoch=7, task="task")

    def reserve(self, index):
        self.assertTrue(self.pool.offer(Observation(index, self.pool.epoch, "task", index)))
        reservation = self.pool.reserve()
        self.assertIsNotNone(reservation)
        return reservation[0]

    def submit(self, identity, *, done=False, events=None, metadata=None):
        events = tuple(events) if events is not None else tuple(Event() for _ in range(3))
        metadata = metadata if metadata is not None else Event()
        full = Event(done, (metadata, *events))
        value = {"generation": identity}
        submission = self.pool.submitted_layers(identity, value, full, events, metadata_ready=metadata)
        return submission, value

    def build(self, index, *, done=False):
        identity = self.reserve(index)
        submission, value = self.submit(identity, done=done)
        return identity, submission, value

    def test_partial_lease_does_not_publish_and_enqueues_gpu_waits_only(self):
        identity, submission, value = self.build(0)
        self.assertIsNone(self.pool.acquire())
        lease = self.pool.acquire_layers(identity)
        self.assertIsInstance(lease, LayerLease)
        self.assertIs(lease.value, value)
        stream = Stream()
        self.pool.wait_metadata(lease, stream)
        self.pool.wait_layer(lease, 1, stream)
        self.assertEqual(stream.waits, [submission.metadata_ready, submission.layer_ready[1]])
        self.assertEqual(sum(event.syncs for event in submission.events), 0)
        self.assertEqual(self.pool.publications, 0)

    def test_old_pool_still_rejects_second_building_slot(self):
        old = Pool(7, "task")
        old.offer(Observation(0, 7, "task", None))
        old.reserve()
        old.offer(Observation(1, 7, "task", None))
        self.assertIsNone(old.reserve())

    def test_two_building_slots_but_never_third(self):
        first, _, _ = self.build(0)
        second, _, _ = self.build(1)
        self.assertNotEqual(first.slot, second.slot)
        self.pool.offer(Observation(2, 7, "task", None))
        self.assertIsNone(self.pool.reserve())
        self.assertEqual(self.pool.live_snapshots, 2)

    def test_reader_completion_cannot_retire_running_producer(self):
        first, submission, _ = self.build(0)
        lease = self.pool.acquire_layers(first)
        self.pool.release(lease, Event(True))
        second, _, _ = self.build(1, done=True)
        self.pool.poll()
        self.assertEqual(self.pool.active, second)
        self.assertEqual(self.pool._slot(first).state, "BUILDING")
        self.pool.offer(Observation(2, 7, "task", None))
        self.assertIsNone(self.pool.reserve())
        submission.producer_done.done = True
        renewed, _ = self.pool.reserve()
        self.assertEqual(renewed.slot, first.slot)
        self.assertGreater(renewed.generation, first.generation)
        self.assertEqual(self.pool.active, second)

    def test_finished_producer_cannot_overwrite_pending_reader(self):
        first, _, _ = self.build(0, done=True)
        lease = self.pool.acquire_layers(first)
        reader = Event()
        self.pool.release(lease, reader)
        self.build(1, done=True)
        self.pool.offer(Observation(2, 7, "task", None))
        self.assertIsNone(self.pool.reserve())
        self.assertEqual(self.pool._slot(first).state, "RETIRED")
        reader.done = True
        self.assertEqual(self.pool.reserve()[0].slot, first.slot)

    def test_live_lease_pins_generation_after_full_publication(self):
        first, submission, value = self.build(0)
        lease = self.pool.acquire_layers(first)
        submission.producer_done.done = True
        self.pool.poll()
        self.build(1, done=True)
        self.pool.poll()
        self.assertIs(self.pool.validate_replay(lease), submission)
        self.assertIs(lease.value, value)
        self.assertEqual(self.pool._slot(first).state, "RETIRED")
        self.pool.release(lease, Event(True))
        self.assertEqual(self.pool._slot(first).state, "FREE")
        self.assertIsNone(lease.value)
        self.assertIsNone(lease.observation)

    def test_full_acquire_requires_done_and_cannot_impersonate_layer_lease(self):
        identity, submission, _ = self.build(0)
        submission.producer_done.done = True
        lease = self.pool.acquire()
        self.assertEqual(lease.identity, identity)
        with self.assertRaisesRegex(RuntimeError, "LayerLease"):
            self.pool.validate_replay(lease)
        self.pool.release(lease, Event(True))

    def test_graph_validation_checks_captured_addresses_without_wait(self):
        identity, submission, _ = self.build(0)
        lease = self.pool.acquire_layers(identity)
        self.assertIs(self.pool.validate_replay(lease, layer_events=submission.layer_ready,
                                               metadata_event=submission.metadata_ready), submission)
        with self.assertRaisesRegex(RuntimeError, "metadata event"):
            self.pool.validate_replay(lease, metadata_event=Event())
        with self.assertRaisesRegex(RuntimeError, "layer events"):
            self.pool.validate_replay(lease, layer_events=tuple(Event() for _ in range(3)))
        self.assertEqual(submission.producer_done.syncs, 0)

    def test_lease_cannot_be_forged_released_twice_or_replayed_after_release(self):
        identity, _, value = self.build(0)
        lease = self.pool.acquire_layers(identity)
        forged = LayerLease(identity, value, lease.observation)
        with self.assertRaisesRegex(RuntimeError, "live lease"):
            self.pool.release(forged, Event(True))
        self.pool.release(lease, Event(True))
        with self.assertRaisesRegex(RuntimeError, "live lease"):
            self.pool.validate_replay(lease)
        with self.assertRaisesRegex(RuntimeError, "live lease"):
            self.pool.release(lease, Event(True))

    def test_recorded_events_can_be_rebound_only_after_old_generation_is_free(self):
        first, old, _ = self.build(0)
        old_lease = self.pool.acquire_layers(first)
        reader = Event()
        self.pool.release(old_lease, reader)
        second = self.reserve(1)
        with self.assertRaisesRegex(RuntimeError, "another live slot"):
            self.submit(second, events=old.layer_ready)
        self.submit(second, done=True)
        self.pool.poll()
        old.producer_done.done = reader.done = True
        renewed = self.reserve(2)
        # Caller enqueues the new records before submitted_layers. Simulate
        # re-record resetting completion, rather than relying on old query().
        for event in (*old.layer_ready, old.metadata_ready):
            event.done = False
        current, _ = self.submit(renewed, events=old.layer_ready, metadata=old.metadata_ready)
        new_lease = self.pool.acquire_layers(renewed)
        self.assertIs(self.pool.validate_replay(new_lease), current)
        self.assertNotEqual(old.identity, current.identity)
        with self.assertRaises(RuntimeError):
            self.pool.acquire_layers(first)
        with self.assertRaises(RuntimeError):
            self.pool.validate_replay(old_lease)

    def test_full_done_and_reader_done_cannot_alias_kv_ready(self):
        identity = self.reserve(0)
        layer = Event()
        with self.assertRaisesRegex(ValueError, "own event"):
            self.pool.submitted_layers(identity, object(), layer, (layer,) * 3, metadata_ready=Event())
        submission, _ = self.submit(identity)
        lease = self.pool.acquire_layers(identity)
        for invalid in submission.events:
            with self.assertRaisesRegex(ValueError, "own event"):
                self.pool.release(lease, invalid)
        self.pool.release(lease, Event(True))

    def test_grouped_layer_fences_are_valid(self):
        event = Event()
        identity = self.reserve(0)
        submission, _ = self.submit(identity, events=(event, event, Event()))
        lease = self.pool.acquire_layers(identity)
        self.assertIs(self.pool.validate_replay(lease).layer_ready[0], submission.layer_ready[1])

    def test_invalid_identity_index_and_unsubmitted_reads_are_rejected(self):
        identity = self.reserve(0)
        for invalid in (Identity(7, -1, 1), Identity(7, 2, 1), Identity(7, 0.5, 1), Identity(6, 0, 1)):
            with self.assertRaises(RuntimeError):
                self.pool.acquire_layers(invalid)
        with self.assertRaisesRegex(RuntimeError, "No valid layer"):
            self.pool.acquire_layers(identity)
        self.submit(identity)
        lease = self.pool.acquire_layers(identity)
        for layer in (-1, 3, True):
            with self.assertRaises(ValueError):
                self.pool.wait_layer(lease, layer, Stream())

    def test_preparation_only_never_publishes_and_can_be_upgraded(self):
        identity = self.reserve(0)
        metadata = Event(True)
        self.pool.track_preparation(identity, metadata)
        self.pool.poll()
        self.assertIsNone(self.pool.acquire())
        with self.assertRaisesRegex(RuntimeError, "tracked metadata"):
            self.submit(identity, metadata=Event())
        submission, _ = self.submit(identity, metadata=metadata)
        self.assertIs(self.pool.validate_replay(self.pool.acquire_layers(identity)), submission)

    def test_reset_drains_vision_only_and_full_producer_and_reader(self):
        first, submission, _ = self.build(0)
        lease = self.pool.acquire_layers(first)
        reader = Event()
        self.pool.release(lease, reader)
        second = self.reserve(1)
        vision = Event()
        self.pool.track_preparation(second, vision)
        self.pool.drain_reset(8, "task")
        self.assertTrue(vision.done)
        self.assertTrue(reader.done)
        self.assertTrue(all(event.done for event in submission.events))
        self.assertEqual(self.pool.live_snapshots, 0)
        with self.assertRaises(RuntimeError):
            self.pool.acquire_layers(first)

    def test_reset_rejects_live_lease_and_unknown_partial_submission(self):
        identity = self.reserve(0)
        with self.assertRaisesRegex(RuntimeError, "Abort unsubmitted"):
            self.pool.drain_reset(8, "task")
        self.submit(identity)
        lease = self.pool.acquire_layers(identity)
        with self.assertRaisesRegex(RuntimeError, "release"):
            self.pool.drain_reset(8, "task")
        self.assertEqual(self.pool.epoch, 7)
        self.pool.release(lease, Event())
        self.pool.drain_reset(8, "task")

    def test_abort_unsubmitted_work_tracks_tail_and_never_publishes(self):
        first = self.reserve(0)
        tail = Event()
        self.pool.abort(first, producer_done=tail)
        self.assertEqual(self.pool._slot(first).state, "BUILDING")
        self.assertIsNone(self.pool.acquire())
        tail.done = True
        self.pool.poll()
        self.assertEqual(self.pool._slot(first).state, "FREE")
        self.assertEqual(self.pool.publications, 0)

    def test_abort_after_released_reader_waits_for_both_tails(self):
        identity, submission, _ = self.build(0)
        lease = self.pool.acquire_layers(identity)
        with self.assertRaisesRegex(RuntimeError, "no live leases"):
            self.pool.abort(identity, producer_done=Event())
        reader = Event()
        self.pool.release(lease, reader)
        with self.assertRaisesRegex(ValueError, "KV-ready"):
            self.pool.abort(identity, producer_done=submission.layer_ready[0])
        tail = Event(True)
        self.pool.abort(identity, producer_done=tail)
        self.assertEqual(self.pool._slot(identity).state, "RETIRED")
        reader.done = True
        self.pool.poll()
        self.assertEqual(self.pool._slot(identity).state, "FREE")
        self.assertEqual(self.pool.publications, 0)

    def test_abort_reset_does_not_wait_on_failed_producers_unrecorded_event(self):
        identity, submission, _ = self.build(0)
        tail = Event()
        self.pool.abort(identity, producer_done=tail)
        self.pool.drain_reset(8, "task")
        self.assertTrue(tail.done)
        self.assertEqual(submission.producer_done.syncs, 0)

    def test_host_owner_enforced(self):
        errors = []
        def other_thread():
            for operation in (self.pool.poll, self.pool.reserve, self.pool.acquire):
                try:
                    operation()
                except RuntimeError as error:
                    errors.append(str(error))
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 3)
        self.assertTrue(all("owning host" in error for error in errors))

    def test_explicit_retirement_requires_producer_lease_and_reader_completion(self):
        identity, submission, _ = self.build(0)
        lease = self.pool.acquire_layers(identity)
        self.assertFalse(self.pool.retire_complete(identity))
        submission.producer_done.done = True
        self.assertFalse(self.pool.retire_complete(identity))
        reader = Event()
        self.pool.release(lease, reader)
        self.assertEqual(self.pool.active, identity)
        self.assertFalse(self.pool.retire_complete(identity))
        reader.done = True
        self.assertTrue(self.pool.retire_complete(identity))
        self.assertIsNone(self.pool.active)
        self.assertEqual(self.pool.live_snapshots, 0)
        replacement = self.reserve(1)
        self.assertEqual(replacement.slot, identity.slot)
        with self.assertRaises(RuntimeError):
            self.pool.retire_complete(identity)

    def test_preparation_only_cannot_be_explicitly_retired(self):
        identity = self.reserve(0)
        self.pool.track_preparation(identity, Event(True))
        self.assertFalse(self.pool.retire_complete(identity))
        self.assertEqual(self.pool._slot(identity).state, "BUILDING")


if __name__ == "__main__":
    unittest.main()
