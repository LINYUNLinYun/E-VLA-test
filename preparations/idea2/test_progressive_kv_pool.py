"""CPU-only tests for progressive old/new reads and ahead-vision ownership."""
import threading
import unittest

from .progressive_kv_pool import OldKVOnlyLease, ProgressiveKVPool
from .scheduler import Identity, Observation
from .test_layer_scheduler import Event


class ProgressivePoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = ProgressiveKVPool(3, epoch=7, task="task")

    def observation(self, index):
        return Observation(index, self.pool.epoch, "task", object())

    def build(self, index, done=True):
        observation = self.observation(index)
        self.pool.offer(observation)
        identity, _ = self.pool.reserve()
        metadata = Event(done)
        layers = tuple(Event(done) for _ in range(3))
        full = Event(done, (metadata, *layers))
        value = object()
        self.pool.submitted_layers(identity, value, full, layers, metadata_ready=metadata)
        return identity, full, value

    def pair(self):
        old, old_done, old_value = self.build(0)
        old_lease = self.pool.acquire_old_kv()
        new, new_done, _ = self.build(1, done=False)
        new_lease = self.pool.acquire_layers(new)
        return old, old_done, old_value, old_lease, new, new_done, new_lease

    def register(self, old, observation=None, done=None):
        return self.pool.register_vision_ahead(old, observation or self.observation(2),
            done if done is not None else Event(), contract="old_kv_only")

    def test_ahead_during_two_leases_changes_no_old_kv_ownership(self):
        old, old_done, value, old_lease, new, new_done, new_lease = self.pair()
        slot = self.pool._slot(old)
        before = (slot.generation, slot.state, slot.value, slot.leases)
        ticket = self.register(old)
        self.assertEqual(before, (slot.generation, slot.state, slot.value, slot.leases))
        self.assertIs(old_lease.value, value)
        self.assertIs(ticket.producer_done, old_done)
        self.assertEqual(ticket.identity, Identity(7, old.slot, old.generation + 1))
        self.assertEqual(old_done.syncs + new_done.syncs + ticket.vision_done.syncs, 0)
        self.pool.offer(ticket.observation)
        self.assertIsNone(self.pool.reserve())
        self.assertEqual(self.pool.live_snapshots, 2)

    def test_preflight_returns_dependency_without_registering_or_waiting(self):
        old, old_done, _, _, _, _, _ = self.pair()
        observation, vision_done = self.observation(2), Event()
        dependency = self.pool.validate_vision_ahead(old, observation, vision_done,
                                                     contract="old_kv_only")
        self.assertIs(dependency, old_done)
        self.assertIsNone(self.pool.vision_ahead)
        self.assertEqual(old_done.syncs + vision_done.syncs, 0)
        self.assertEqual(self.pool.ticket_diagnostics["history"], [])

    def test_reservation_never_uses_other_free_slot_for_ahead_identity(self):
        self.build(0)
        old, _, _ = self.build(1)
        lease = self.pool.acquire_old_kv()
        self.assertEqual(old.slot, 1)
        ticket = self.register(old)
        self.assertEqual(self.pool.slots[0].state, "FREE")
        self.pool.offer(ticket.observation)
        self.assertIsNone(self.pool.reserve())
        self.pool.release(lease, Event(True))
        self.assertTrue(self.pool.retire_complete(old))
        self.assertEqual(self.pool.reserve()[0], ticket.identity)

    def test_claim_requires_reader_completion_and_preserves_new_active(self):
        old, _, _, old_lease, new, new_done, new_lease = self.pair()
        ticket = self.register(old)
        old_reader, new_reader = Event(), Event()
        self.pool.release(old_lease, old_reader)
        self.pool.release(new_lease, new_reader)
        new_done.done = True
        self.pool.offer(ticket.observation)
        self.assertIsNone(self.pool.reserve())
        old_reader.done = True
        reserved, observation = self.pool.reserve()
        self.assertEqual(reserved, ticket.identity)
        self.assertIs(observation, ticket.observation)
        self.assertEqual(self.pool.active, new)
        self.assertIs(self.pool.claim_vision_ahead(reserved), ticket)
        self.assertIs(self.pool._preparations[reserved], ticket.vision_done)
        self.assertIsNone(self.pool.claim_vision_ahead(reserved))
        self.assertEqual(self.pool.ticket_diagnostics["counts"],
                         dict(registered=1, claimed=1, reset_drained=0))

    def test_same_pending_reader_event_cannot_release_two_slots(self):
        old, _, _, old_lease, new, _, new_lease = self.pair()
        reader = Event()
        self.pool.release(old_lease, reader)
        with self.assertRaisesRegex(RuntimeError, "another live slot"):
            self.pool.release(new_lease, reader)
        self.assertFalse(new_lease.released)
        self.pool.release(new_lease, Event())

    def test_existing_ordinary_lease_blocks_ahead_workspace(self):
        old, _, _ = self.build(0)
        lease = self.pool.acquire()
        with self.assertRaisesRegex(RuntimeError, "metadata Lease"):
            self.register(old)
        self.pool.release(lease, Event(True))
        self.assertIsNotNone(self.register(old))

    def test_unfinished_released_ordinary_reader_still_blocks_ahead(self):
        old, _, _ = self.build(0)
        lease = self.pool.acquire()
        reader = Event()
        self.pool.release(lease, reader)
        with self.assertRaisesRegex(RuntimeError, "ordinary Prefix reader"):
            self.register(old)
        reader.done = True
        self.assertIsNotNone(self.register(old))

    def test_kv_only_released_reader_allows_ahead_but_pins_kv(self):
        old, _, _ = self.build(0)
        lease = self.pool.acquire_old_kv()
        self.assertIsInstance(lease, OldKVOnlyLease)
        reader = Event()
        self.pool.release(lease, reader)
        ticket = self.register(old)
        self.pool.offer(ticket.observation)
        self.assertIsNone(self.pool.reserve())
        self.assertEqual(reader.syncs, 0)

    def test_no_new_metadata_readers_after_ticket_registration(self):
        old, _, _ = self.build(0)
        self.register(old)
        with self.assertRaisesRegex(RuntimeError, "metadata readers"):
            self.pool.acquire()
        with self.assertRaisesRegex(RuntimeError, "only old K/V"):
            self.pool.acquire_layers(old)
        self.assertIsInstance(self.pool.acquire_old_kv(old), OldKVOnlyLease)

    def test_ticket_cannot_be_overwritten_or_observation_replaced(self):
        old, _, _ = self.build(0)
        ticket = self.register(old)
        with self.assertRaisesRegex(RuntimeError, "Only one"):
            self.register(old)
        with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
            self.pool.offer(self.observation(3))
        self.assertIs(self.pool.vision_ahead, ticket)

    def test_contract_and_observation_epoch_are_explicit(self):
        old, _, _ = self.build(0)
        with self.assertRaisesRegex(ValueError, "old_kv_only"):
            self.pool.register_vision_ahead(old, self.observation(2), Event(), contract="ordinary")
        with self.assertRaisesRegex(RuntimeError, "epoch/task"):
            self.register(old, Observation(2, 8, "task", None))
        with self.assertRaisesRegex(RuntimeError, "follow"):
            self.register(old, self.observation(0))

    def test_event_cannot_alias_old_generation_or_other_slot(self):
        old, _, _, _, new, _, _ = self.pair()
        with self.assertRaisesRegex(RuntimeError, "fresh event"):
            self.register(old, done=self.pool._submissions[old].metadata_ready)
        with self.assertRaisesRegex(RuntimeError, "another live slot"):
            self.register(old, done=self.pool._submissions[new].metadata_ready)

    def test_wrong_slot_generation_epoch_and_unreserved_claim_rejected(self):
        old, _, _, _, new, _, _ = self.pair()
        ticket = self.register(old)
        for identity in (old, new, ticket.identity, Identity(8, old.slot, old.generation)):
            with self.assertRaises(RuntimeError):
                self.pool.claim_vision_ahead(identity)
        self.assertIs(self.pool.vision_ahead, ticket)

    def test_pending_ticket_reset_drains_vision_not_just_old_producer(self):
        old, old_done, _ = self.build(0)
        vision_done = Event(False, (old_done,))
        self.register(old, done=vision_done)
        self.pool.drain_reset(8, "next")
        self.assertEqual(vision_done.syncs, 1)
        self.assertIsNone(self.pool.vision_ahead)
        self.assertEqual(self.pool.ticket_diagnostics["counts"]["reset_drained"], 1)
        self.assertEqual(self.pool.live_snapshots, 0)
        with self.assertRaises(RuntimeError):
            self.pool.claim_vision_ahead(old)

    def test_reset_with_live_lease_rejects_before_waiting_vision(self):
        old, _, _, _, _, _, _ = self.pair()
        ticket = self.register(old)
        with self.assertRaisesRegex(RuntimeError, "Finish and release"):
            self.pool.drain_reset(8, "task")
        self.assertEqual(ticket.vision_done.syncs, 0)

    def test_reset_after_reserve_before_claim_drains_ticket(self):
        old, _, _, old_lease, _, new_done, new_lease = self.pair()
        ticket = self.register(old)
        self.pool.release(old_lease, Event(True))
        self.pool.release(new_lease, Event(True))
        new_done.done = True
        self.pool.offer(ticket.observation)
        self.assertEqual(self.pool.reserve()[0], ticket.identity)
        self.pool.drain_reset(8, "task")
        self.assertTrue(ticket.vision_done.done)
        self.assertEqual(self.pool.live_snapshots, 0)

    def test_claimed_ticket_reset_still_drains_preparation(self):
        old, _, _, old_lease, _, new_done, new_lease = self.pair()
        ticket = self.register(old)
        self.pool.release(old_lease, Event(True))
        self.pool.release(new_lease, Event(True))
        new_done.done = True
        self.pool.offer(ticket.observation)
        identity, _ = self.pool.reserve()
        self.pool.claim_vision_ahead(identity)
        self.pool.drain_reset(8, "task")
        self.assertEqual(ticket.vision_done.syncs, 1)

    def test_unsubmitted_old_producer_and_wrong_host_are_rejected(self):
        self.pool.offer(self.observation(0))
        identity, _ = self.pool.reserve()
        with self.assertRaisesRegex(RuntimeError, "enqueued old producer"):
            self.register(identity)
        errors = []
        def other_host():
            try:
                self.pool.acquire_old_kv()
            except RuntimeError as error:
                errors.append(str(error))
        worker = threading.Thread(target=other_host)
        worker.start()
        worker.join()
        self.assertEqual(len(errors), 1)
        self.assertIn("owning host", errors[0])


if __name__ == "__main__":
    unittest.main()
