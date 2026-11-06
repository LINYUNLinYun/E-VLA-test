"""Two KV slots with explicit KV-only readers and one ahead-vision ticket.

An ahead ticket permits writing *vision workspace only*. Its producer must wait
on ``ticket.producer_done`` on the GPU. OldKVOnlyLease is a caller contract:
the reader may access old K/V, never old masks, positions, embeddings, state,
noise, or vision scratch. The new-frame metadata must be used by that reader.
No CUDA work or CPU synchronization is performed during ordinary dispatch.
"""
from dataclasses import dataclass

from .layer_scheduler import LayerPool
from .scheduler import Identity, Lease


@dataclass
class OldKVOnlyLease(Lease):
    """Read permission for complete K/V only, excluding Prefix metadata."""


@dataclass(frozen=True)
class VisionAheadTicket:
    epoch: int
    physical_slot: int
    expected_generation: int
    old_identity: Identity
    observation: object
    vision_done: object
    producer_done: object
    contract: str = "old_kv_only"

    @property
    def identity(self):
        return Identity(self.epoch, self.physical_slot, self.expected_generation)


class ProgressiveKVPool(LayerPool):
    def __init__(self, num_layers, epoch=0, task=""):
        self._vision_ahead = None
        self._kv_only_readers = {}
        self._ticket_history = []
        self._ticket_counts = dict(registered=0, claimed=0, reset_drained=0)
        super().__init__(num_layers, epoch, task)

    @property
    def vision_ahead(self):
        self._owner()
        return self._vision_ahead

    @property
    def ticket_diagnostics(self):
        self._owner()
        return dict(counts=dict(self._ticket_counts), pending=self._vision_ahead is not None,
                    history=[dict(row) for row in self._ticket_history])

    def _log_ticket(self, ticket, action):
        self._ticket_counts[action] += 1
        self._ticket_history.append(dict(action=action, epoch=ticket.epoch,
            slot=ticket.physical_slot, generation=ticket.expected_generation,
            old_generation=ticket.old_identity.generation,
            observation_id=ticket.observation.observation_id, contract=ticket.contract))

    def _protected(self, identity):
        ticket = self._vision_ahead
        return ticket is not None and identity == ticket.old_identity

    def acquire(self):
        self._owner()
        self.poll()
        if self.active is not None and self._protected(self.active):
            raise RuntimeError("Ahead vision forbids ordinary old Prefix metadata readers")
        return super().acquire()

    def acquire_layers(self, identity):
        self._owner()
        if self._protected(identity):
            raise RuntimeError("Ahead vision permits only old K/V readers for this generation")
        return super().acquire_layers(identity)

    def acquire_old_kv(self, identity=None):
        self._owner()
        self.poll()
        identity = self.active if identity is None else identity
        if identity is None:
            return None
        slot = self._slot(identity)
        if (slot.state not in ("ACTIVE", "RETIRED") or slot.value is None
                or slot.ready is None or not slot.ready.query()):
            raise RuntimeError("Old K/V must be a complete, pinned snapshot")
        slot.leases += 1
        lease = OldKVOnlyLease(identity, slot.value, slot.observation)
        self._leases[id(lease)] = lease
        return lease

    def _check_event_conflicts(self, identity, events):
        super()._check_event_conflicts(identity, events)
        ticket = self._vision_ahead
        if ticket is not None and any(event is ticket.vision_done for event in events):
            raise RuntimeError("Ahead vision completion belongs to a pending ticket")

    def release(self, lease, reader_done):
        self._live_lease(lease)
        kv_only = isinstance(lease, OldKVOnlyLease)
        if kv_only:
            self._kv_only_readers.setdefault(lease.identity, set()).add(id(reader_done))
        try:
            return super().release(lease, reader_done)
        except BaseException:
            if kv_only and not lease.released:
                self._kv_only_readers[lease.identity].discard(id(reader_done))
            raise

    def poll(self):
        super().poll()
        for identity in tuple(self._kv_only_readers):
            slot = self._slot(identity)
            live = {id(event) for event in slot.readers}
            self._kv_only_readers[identity].intersection_update(live)
            if slot.state == "FREE" or not self._kv_only_readers[identity]:
                del self._kv_only_readers[identity]

    def validate_vision_ahead(self, old_identity, observation, vision_done, *, contract):
        """Preflight before writing workspace; return the producer GPU dependency.

        Single-Host callers validate, enqueue the producer wait/vision/Event,
        then register without intervening Pool mutations. Registering alone
        cannot retroactively prevent an unsafe GPU launch.
        """
        self._owner()
        if contract != "old_kv_only":
            raise ValueError("Ahead vision requires the explicit old_kv_only contract")
        if self._vision_ahead is not None:
            raise RuntimeError("Only one ahead vision ticket may exist")
        slot = self._slot(old_identity)
        if slot.state not in ("BUILDING", "ACTIVE", "RETIRED") or slot.ready is None:
            raise RuntimeError("Ahead vision requires an enqueued old producer completion")
        if (observation.epoch, observation.task) != (self.epoch, self.task):
            raise RuntimeError("Ahead observation belongs to another epoch/task")
        if any(s.observation is not None and observation.observation_id <= s.observation.observation_id
               for s in self.slots if s.state != "FREE"):
            raise RuntimeError("Ahead observation must follow all admitted KV observations")
        if self.pending is not None and self.pending is not observation:
            raise RuntimeError("A different observation is already pending")
        readers = [lease for lease in self._leases.values() if lease.identity == old_identity]
        if any(not isinstance(lease, OldKVOnlyLease) for lease in readers):
            raise RuntimeError("Ordinary Prefix metadata Lease prevents ahead vision")
        allowed = self._kv_only_readers.get(old_identity, set())
        if any(not event.query() and id(event) not in allowed for event in slot.readers):
            raise RuntimeError("Unfinished ordinary Prefix reader prevents ahead vision")
        self._event(vision_done)
        submission = self._submissions.get(old_identity)
        owned = list(submission.events) if submission is not None else [slot.ready]
        owned.extend(slot.readers)
        if old_identity in self._preparations:
            owned.append(self._preparations[old_identity])
        if any(vision_done is event for event in owned):
            raise RuntimeError("Ahead vision requires a fresh event, not an old generation event")
        self._check_event_conflicts(old_identity, (vision_done,))
        return slot.ready

    def register_vision_ahead(self, old_identity, observation, vision_done, *, contract):
        """Register an already-enqueued vision job in old-slot vision workspace.

        Caller must use a fresh vision_done event, wait on old slot.ready before
        writing workspace, and establish that old readers use new metadata.
        This does not change the old KV value, generation, Lease, or Slot state.
        """
        producer_done = self.validate_vision_ahead(old_identity, observation, vision_done, contract=contract)
        ticket = VisionAheadTicket(self.epoch, old_identity.slot, old_identity.generation + 1,
                                   old_identity, observation, vision_done, producer_done)
        self._vision_ahead = ticket
        self._log_ticket(ticket, "registered")
        return ticket

    def offer(self, observation):
        self._owner()
        if self._vision_ahead is not None and observation is not self._vision_ahead.observation:
            raise RuntimeError("An admitted ahead observation cannot be replaced")
        return super().offer(observation)

    def reserve(self):
        self._owner()
        ticket = self._vision_ahead
        if ticket is None:
            return super().reserve()
        self.poll()
        if self.pending is None:
            return None
        if self.pending is not ticket.observation:
            raise RuntimeError("Reservation differs from the ahead observation")
        slot = self.slots[ticket.physical_slot]
        if ticket.epoch != self.epoch or slot.generation + 1 != ticket.expected_generation:
            raise RuntimeError("Ahead ticket generation no longer matches its physical slot")
        if slot.state != "FREE":
            self.no_free_slot += 1
            return None
        slot.generation += 1
        slot.state, slot.observation = "BUILDING", self.pending
        slot.ready_observed_at = slot.published_at = None
        self.pending = None
        return ticket.identity, slot.observation

    def claim_vision_ahead(self, new_identity):
        """Bind the completed-or-pending vision job to a freshly reserved KV slot.

        Return its metadata Event to the VLM producer; enqueue a GPU wait before
        reading vision outputs. The caller must not replay Encoder a second time.
        """
        slot = self._slot(new_identity)
        ticket = self._vision_ahead
        if ticket is None:
            return None
        if new_identity != ticket.identity:
            raise RuntimeError("Ahead ticket epoch/slot/generation mismatch")
        if (slot.state != "BUILDING" or slot.ready is not None or slot.leases or slot.readers
                or slot.observation is not ticket.observation):
            raise RuntimeError("Ahead ticket needs an exclusive matching new reservation")
        self._vision_ahead = None
        try:
            self.track_preparation(new_identity, ticket.vision_done)
        except BaseException:
            self._vision_ahead = ticket
            raise
        self._log_ticket(ticket, "claimed")
        return ticket

    def drain_reset(self, epoch, task):
        self._owner()
        if self._leases or any(slot.leases for slot in self.slots):
            raise RuntimeError("Finish and release generation jobs before reset")
        ticket = self._vision_ahead
        if ticket is not None:
            ticket.vision_done.synchronize()
            # A reserved-but-not-yet-claimed ticket still owns preparation work.
            slot = self.slots[ticket.physical_slot]
            if slot.generation == ticket.expected_generation and slot.state == "BUILDING" and slot.ready is None:
                self._vision_ahead = None
                try:
                    self.track_preparation(ticket.identity, ticket.vision_done)
                finally:
                    self._vision_ahead = ticket
        super().drain_reset(epoch, task)
        if ticket is not None:
            self._log_ticket(ticket, "reset_drained")
        self._vision_ahead = None
        self._kv_only_readers.clear()
