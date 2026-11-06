"""Single-host two-slot ownership protocol; no torch dependency.

Events implement query()/synchronize(). Producers and consumers retain their
inputs until their events complete. Only the host owning Pool may mutate it.
"""
from dataclasses import dataclass, field
from typing import Any
import time


@dataclass(frozen=True)
class Observation:
    observation_id: int
    epoch: int
    task: str
    payload: Any
    image_times: dict = field(default_factory=dict)
    state_id: Any = None
    state_time: float | None = None
    host_receipt_time: float | None = None
    host_receipt_unix: float | None = None


@dataclass(frozen=True)
class Identity:
    epoch: int
    slot: int
    generation: int


@dataclass
class Slot:
    generation: int = 0
    state: str = "FREE"
    observation: Observation | None = None
    value: Any = None
    ready: Any = None
    leases: int = 0
    readers: list = field(default_factory=list)
    ready_observed_at: float | None = None
    published_at: float | None = None


@dataclass
class Lease:
    identity: Identity
    value: Any
    observation: Observation
    released: bool = False


class Pool:
    def __init__(self, epoch=0, task=""):
        self.epoch, self.task = epoch, task
        self.slots = [Slot(), Slot()]
        self.active = None
        self.pending = None
        self.pending_replacements = 0
        self.no_free_slot = 0
        self.publications = 0

    def offer(self, observation):
        if (observation.epoch, observation.task) != (self.epoch, self.task):
            return False
        if self.pending is not None:
            if observation.observation_id <= self.pending.observation_id:
                return False
            self.pending_replacements += 1
        self.pending = observation
        return True

    def _slot(self, identity):
        slot = self.slots[identity.slot]
        if identity.epoch != self.epoch or identity.generation != slot.generation:
            raise RuntimeError("Stale slot generation or episode callback")
        return slot

    def reserve(self):
        self.poll()
        if self.pending is None or any(s.state == "BUILDING" for s in self.slots):
            return None
        for index, slot in enumerate(self.slots):
            if slot.state == "FREE":
                slot.generation += 1
                slot.state, slot.observation = "BUILDING", self.pending
                self.pending = None
                return Identity(self.epoch, index, slot.generation), slot.observation
        self.no_free_slot += 1
        return None

    def submitted(self, identity, value, ready):
        slot = self._slot(identity)
        if slot.state != "BUILDING" or slot.ready is not None:
            raise RuntimeError("Invalid producer submission")
        slot.value, slot.ready = value, ready

    def poll(self):
        for index, slot in enumerate(self.slots):
            if slot.state == "BUILDING" and slot.ready is not None and slot.ready.query():
                identity = Identity(self.epoch, index, slot.generation)
                slot.ready_observed_at = time.perf_counter()
                old = self.slots[self.active.slot] if self.active is not None else None
                if old is None or slot.observation.observation_id > old.observation.observation_id:
                    if old is not None:
                        old.state = "RETIRED"
                    slot.state, self.active = "ACTIVE", identity
                    slot.published_at = time.perf_counter()
                    self.publications += 1
                else:
                    slot.state = "RETIRED"
        for slot in self.slots:
            slot.readers[:] = [event for event in slot.readers if not event.query()]
            if slot.state == "RETIRED" and not slot.leases and not slot.readers:
                slot.state = "FREE"
                slot.value = slot.observation = slot.ready = None

    def acquire(self):
        self.poll()
        if self.active is None:
            return None
        slot = self._slot(self.active)
        slot.leases += 1
        return Lease(self.active, slot.value, slot.observation)

    def release(self, lease, reader_done):
        if lease.released:
            raise RuntimeError("Lease already released")
        slot = self._slot(lease.identity)
        slot.readers.append(reader_done)
        slot.leases -= 1
        lease.released = True
        # A caller may keep a released lease for logging. Do not let that host
        # object retain an otherwise reclaimed full snapshot outside the pool.
        lease.value = lease.observation = None
        self.poll()

    def drain_reset(self, epoch, task):
        # Synchronize only work owned by this scheduler, never the whole device.
        if any(s.leases for s in self.slots):
            raise RuntimeError("Finish and release generation jobs before reset")
        for slot in self.slots:
            for event in ([slot.ready] if slot.ready is not None else []) + slot.readers:
                event.synchronize()
        self.epoch, self.task = epoch, task
        self.active = self.pending = None
        for slot in self.slots:
            slot.state = "FREE"
            slot.value = slot.observation = slot.ready = None
            slot.readers.clear()

    @property
    def live_snapshots(self):
        return sum(s.state != "FREE" for s in self.slots)
