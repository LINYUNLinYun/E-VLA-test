"""Opt-in, two-slot ownership for same-observation KV-ready consumers.

The original Pool.acquire() still exposes only fully published prefixes. A
LayerLease explicitly pins one producer generation and permits GPU waits on
that generation's per-layer fences. Only the creating host thread may use this
protocol; GPU workers never mutate Pool state.

IMPORTANT: submitted_layers() is a host submission receipt, not a CUDA record.
The caller must first enqueue this generation's metadata/layer event records
and full producer-done record, then register them, then enqueue its consumers.
CUDA Event.query() alone cannot distinguish an old record from a new one.
For captured waits, validate_replay() must run before every replay; wait_layer()
is not invoked by graph replay. Re-recording events outside this protocol while
a generation is live is forbidden. No CPU wait is needed for normal dispatch.
"""
from dataclasses import dataclass
import threading

from .scheduler import Identity, Lease, Pool


@dataclass(frozen=True)
class LayerSubmission:
    identity: Identity
    metadata_ready: object
    layer_ready: tuple
    producer_done: object

    @property
    def events(self):
        return (self.metadata_ready, *self.layer_ready, self.producer_done)


@dataclass
class LayerLease(Lease):
    """Explicit partial-prefix read permission; never obtained by acquire()."""


class LayerPool(Pool):
    def __init__(self, num_layers, epoch=0, task=""):
        if not isinstance(num_layers, int) or isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self._owner_thread = threading.get_ident()
        self.num_layers = num_layers
        self._submissions = {}
        self._preparations = {}
        self._leases = {}
        self._consumed = set()
        self._aborted = set()
        super().__init__(epoch, task)

    def _owner(self):
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("LayerPool may only be used by its owning host thread")

    @staticmethod
    def _event(event):
        if event is None or not callable(getattr(event, "query", None)) or not callable(getattr(event, "synchronize", None)):
            raise ValueError("An event with query/synchronize is required")

    def _slot(self, identity):
        self._owner()
        if (not isinstance(identity, Identity) or not isinstance(identity.slot, int)
                or isinstance(identity.slot, bool) or not 0 <= identity.slot < len(self.slots)):
            raise RuntimeError("Invalid slot identity")
        return super()._slot(identity)

    def offer(self, observation):
        self._owner()
        return super().offer(observation)

    def reserve(self):
        self._owner()
        self.poll()
        if self.pending is None:
            return None
        # Unlike the old-KV Pool, the second slot may hold next-frame vision
        # preparation while the first slot's full prefix is still building.
        for index, slot in enumerate(self.slots):
            if slot.state == "FREE":
                slot.generation += 1
                slot.state, slot.observation = "BUILDING", self.pending
                slot.ready_observed_at = slot.published_at = None
                self.pending = None
                return Identity(self.epoch, index, slot.generation), slot.observation
        self.no_free_slot += 1
        return None

    def track_preparation(self, identity, metadata_ready):
        """Register an enqueued vision/metadata stage without publishing KV.

        No other unregistered GPU work may remain when reset is requested.
        If full-prefix submission raises after launching work, abort() must
        first register a tail covering those additional accesses.
        """
        slot = self._slot(identity)
        if slot.state != "BUILDING" or slot.ready is not None or identity in self._preparations:
            raise RuntimeError("Invalid preparation submission")
        self._event(metadata_ready)
        self._check_event_conflicts(identity, (metadata_ready,))
        self._preparations[identity] = metadata_ready

    def submitted(self, identity, value, ready):
        """Register a full-prefix-only producer; no partial reads are allowed."""
        self._owner()
        self._event(ready)
        if self._preparations.get(identity) is ready:
            raise ValueError("Full producer completion cannot be the preparation event")
        self._check_event_conflicts(identity, (ready,))
        return super().submitted(identity, value, ready)

    def _check_event_conflicts(self, identity, events):
        selected = {id(event) for event in events}
        for index, slot in enumerate(self.slots):
            other = Identity(self.epoch, index, slot.generation)
            if other == identity or slot.state == "FREE":
                continue
            submission = self._submissions.get(other)
            owned = list(submission.events) if submission else []
            if other in self._preparations:
                owned.append(self._preparations[other])
            owned.extend(slot.readers)
            if slot.ready is not None:
                owned.append(slot.ready)
            if selected.intersection(id(event) for event in owned):
                raise RuntimeError("Event belongs to another live slot generation")

    def submitted_layers(self, identity, value, producer_done, layer_ready, *, metadata_ready):
        """Register already-enqueued producer records, without waiting for them.

        Grouped release may use the same event for several adjacent layers.
        producer_done must be a distinct record after the entire VLM tail;
        the final KV-ready event is not sufficient to release producer storage.
        """
        slot = self._slot(identity)
        if slot.state != "BUILDING" or slot.ready is not None or identity in self._submissions:
            raise RuntimeError("Invalid layer producer submission")
        if value is None:
            raise ValueError("Layer producer must bind its slot-owned KV value")
        if identity in self._preparations and self._preparations[identity] is not metadata_ready:
            raise RuntimeError("Layer submission must use the tracked metadata event")
        layers = tuple(layer_ready)
        if len(layers) != self.num_layers:
            raise ValueError("Exactly one ready-event binding per layer is required")
        for event in (metadata_ready, *layers, producer_done):
            self._event(event)
        if any(producer_done is event for event in (metadata_ready, *layers)):
            raise ValueError("Full producer completion must have its own event")
        self._check_event_conflicts(identity, (metadata_ready, *layers, producer_done))
        submission = LayerSubmission(identity, metadata_ready, layers, producer_done)
        super().submitted(identity, value, producer_done)
        self._submissions[identity] = submission
        return submission

    def poll(self):
        self._owner()
        # A failed prefix must never become the active complete snapshot.
        for identity in self._aborted:
            slot = self._slot(identity)
            if slot.state == "BUILDING" and slot.ready is not None and slot.ready.query():
                slot.state = "RETIRED"
        super().poll()
        for identity in tuple(self._submissions):
            slot = self._slot(identity)
            if slot.state == "FREE":
                del self._submissions[identity]
        for identity in tuple(self._preparations):
            if self._slot(identity).state == "FREE":
                del self._preparations[identity]
        self._aborted = {identity for identity in self._aborted if self._slot(identity).state != "FREE"}
        self._consumed = {identity for identity in self._consumed if self._slot(identity).state != "FREE"}

    def acquire(self):
        """Compatibility path: only a complete, published prefix is visible."""
        self._owner()
        lease = super().acquire()
        if lease is not None:
            self._leases[id(lease)] = lease
        return lease

    def acquire_layers(self, identity):
        self._owner()
        self.poll()
        slot = self._slot(identity)
        if identity in self._aborted or identity not in self._submissions:
            raise RuntimeError("No valid layer producer submission for this generation")
        if slot.state not in ("BUILDING", "ACTIVE"):
            raise RuntimeError("Layer reads require the current building or active generation")
        slot.leases += 1
        lease = LayerLease(identity, slot.value, slot.observation)
        self._leases[id(lease)] = lease
        return lease

    def _live_lease(self, lease):
        self._owner()
        if self._leases.get(id(lease)) is not lease or lease.released:
            raise RuntimeError("A live lease issued by this Pool is required")
        slot = self._slot(lease.identity)
        if slot.leases < 1 or slot.value is not lease.value:
            raise RuntimeError("Lease no longer owns this slot value")
        return slot

    def validate_replay(self, lease, *, layer_events=None, metadata_event=None):
        """Validate a graph's generation and optional captured event addresses.

        Call after enqueueing producer records and before replaying consumer
        graphs. Returning a receipt never implies the GPU work is finished.
        """
        slot = self._live_lease(lease)
        if not isinstance(lease, LayerLease) or lease.identity in self._aborted:
            raise RuntimeError("An explicit live LayerLease is required")
        submission = self._submissions.get(lease.identity)
        if submission is None or slot.ready is not submission.producer_done:
            raise RuntimeError("Missing current-generation layer submission")
        if metadata_event is not None and metadata_event is not submission.metadata_ready:
            raise RuntimeError("Captured metadata event belongs to a different submission")
        if layer_events is not None:
            expected = tuple(layer_events)
            if len(expected) != self.num_layers or any(a is not b for a, b in zip(expected, submission.layer_ready)):
                raise RuntimeError("Captured layer events belong to a different submission")
        return submission

    def wait_metadata(self, lease, stream):
        stream.wait_event(self.validate_replay(lease).metadata_ready)

    def wait_layer(self, lease, layer, stream):
        submission = self.validate_replay(lease)
        if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < self.num_layers:
            raise ValueError("Invalid layer index")
        stream.wait_event(submission.layer_ready[layer])

    def release(self, lease, reader_done):
        slot = self._live_lease(lease)
        self._event(reader_done)
        submission = self._submissions.get(lease.identity)
        producer_events = submission.events if submission is not None else (slot.ready,)
        if lease.identity in self._preparations:
            producer_events = (*producer_events, self._preparations[lease.identity])
        if any(reader_done is event for event in (*producer_events, *slot.readers)):
            raise ValueError("Reader completion requires its own event record")
        self._check_event_conflicts(lease.identity, (reader_done,))
        # Remove registry entry before base release calls poll. A failed poll
        # does not undo submission of the reader-done fence or revive a lease.
        del self._leases[id(lease)]
        self._consumed.add(lease.identity)
        return super().release(lease, reader_done)

    def abort(self, identity, *, producer_done):
        """Retire a failed reservation using a caller-recorded producer tail.

        The tail must cover all submitted producer accesses. Finish/release
        consumers first; the caller must also resolve any queued waits on
        events the failed producer did not record. An unsubmitted reservation
        still needs an explicit completed event, preventing silent reuse after
        a launch raised part-way through submission.
        """
        slot = self._slot(identity)
        if slot.state != "BUILDING" or slot.leases:
            raise RuntimeError("Abort requires a building reservation with no live leases")
        self._event(producer_done)
        submission = self._submissions.get(identity)
        if submission is not None and any(producer_done is event for event in (submission.metadata_ready, *submission.layer_ready)):
            raise ValueError("Abort tail cannot be a metadata or KV-ready event")
        self._check_event_conflicts(identity, (producer_done,))
        slot.ready = producer_done
        self._aborted.add(identity)
        self.poll()

    def retire_complete(self, identity):
        """Opt-in fresh-frame retirement after all producer/readers complete.

        Unlike old-KV serving, a consumed fresh snapshot need not remain ACTIVE
        until another prefix publishes. Never waits: False means its slot must
        stay pinned. This does not change the inherited publication policy.
        """
        slot = self._slot(identity)
        if slot.state == "FREE":
            return True
        if identity not in self._consumed:
            return False
        if slot.ready is None or not slot.ready.query() or slot.leases:
            return False
        if any(not reader.query() for reader in slot.readers):
            return False
        if self.active == identity:
            self.active = None
        slot.state = "RETIRED"
        self.poll()
        return slot.state == "FREE"

    def drain_reset(self, epoch, task):
        self._owner()
        if self._leases or any(slot.leases for slot in self.slots):
            raise RuntimeError("Finish and release generation jobs before reset")
        unsubmitted = [Identity(self.epoch, index, slot.generation)
                       for index, slot in enumerate(self.slots)
                       if slot.state == "BUILDING" and slot.ready is None]
        if any(identity not in self._preparations for identity in unsubmitted):
            raise RuntimeError("Abort unsubmitted reservations with a producer tail before reset")
        for identity in unsubmitted:
            self._preparations[identity].synchronize()
        # Producer completion covers layer records. Abort tails may replace a
        # failed/unrecorded full-done event, so never wait on that stale receipt.
        super().drain_reset(epoch, task)
        self._submissions.clear()
        self._preparations.clear()
        self._aborted.clear()
        self._consumed.clear()
