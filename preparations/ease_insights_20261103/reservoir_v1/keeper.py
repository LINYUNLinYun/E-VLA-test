"""Retain 16 GiB between jobs, and 8 GiB alongside the model.

One owned CUDA context uses the unchanged small keeper's driver ABI and
64-MiB bootstrap allocation. Additional allocations can be resized without
destroying that context. No kernels, Torch, model, or GPU lease are involved.
Only the proven parent's nonce/sequence-bound pipe can request 8/16-GiB
targets. EOF or parent loss cleans up only this process's allocations.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal

from preparations.ease_insights_20261103 import gpu_owned_keeper as base

SCHEMA = "ease-resizable-vram-keeper-v1"
BASE_SHA = "c3719a70ed02b2ac42c8cfea261dba5436f59a10657e350c5d2687f5e2ca1566"
IDLE_MIB = 16384
RUNNING_MIB = 8192
ANCHOR_MIB = 64


def target_bytes(mib):
    if type(mib) is not int or mib not in (RUNNING_MIB, IDLE_MIB):
        raise ValueError("Target must be exactly 8192 or 16384 MiB")
    return mib * base.MIB


class Reservoir(base.DriverAllocation):
    def __init__(self, gpu_uuid, loader=None):
        super().__init__(gpu_uuid, ANCHOR_MIB * base.MIB, loader=loader)
        self.extra = []

    def acquire(self):
        super().acquire()
        if IDLE_MIB * base.MIB * 4 > self.total_memory_bytes * 3:
            raise ValueError("16-GiB target exceeds 75% of assigned device capacity")
        self.resize(IDLE_MIB)

    def resize(self, mib):
        wanted = target_bytes(mib)
        if self.context is None or self.pointer is None:
            raise RuntimeError("Reservoir has no live owned CUDA context")
        # Chunk boundaries are shared by both targets, so shrinking never
        # requires allocating a replacement buffer first.
        if wanted > self.allocated_bytes:
            added = []
            remaining = wanted - self.allocated_bytes
            try:
                while remaining:
                    current = self.allocated_bytes + sum(size for _, size in added)
                    size = min(1024 * base.MIB - current % (1024 * base.MIB), remaining)
                    pointer = base.DEVICE_POINTER()
                    self._call("cuMemAlloc_v2", ctypes.byref(pointer), size)
                    if not pointer.value:
                        raise RuntimeError("CUDA returned a null reservoir allocation")
                    added.append((pointer, size))
                    remaining -= size
            except BaseException:
                # Failed growth leaves the previous retained floor intact.
                self.extra.extend(added)
                self.allocated_bytes += sum(size for _, size in added)
                raise
            self.extra.extend(added)
            self.allocated_bytes = wanted
        else:
            while self.allocated_bytes > wanted:
                pointer, size = self.extra[-1]
                if self.allocated_bytes - size < wanted:
                    raise RuntimeError("Reservoir target is not an allocation boundary")
                self._call("cuMemFree_v2", pointer)
                self.extra.pop()
                self.allocated_bytes -= size
        return self.allocated_bytes

    def close(self):
        errors = []
        while self.extra:
            pointer, _ = self.extra.pop()
            try:
                self._call("cuMemFree_v2", pointer)
            except Exception as error:
                errors.append(str(error))
        return errors + super().close()


def command(record, nonce, previous):
    if (not isinstance(record, dict) or set(record) != {"schema", "op", "nonce", "sequence", "target_mib"}
            or record["schema"] != SCHEMA or record["op"] != "resize" or record["nonce"] != nonce
            or type(record["sequence"]) is not int or record["sequence"] != previous + 1):
        raise ValueError("Resize command identity or sequence differs")
    target_bytes(record["target_mib"])
    return record["sequence"], record["target_mib"]


def run(config, allocation_factory=Reservoir):
    config = base.validate_config(config)
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError("Original driver/lifecycle helper source changed")
    base._close_unneeded_fds({0, 1, 2, config.control_fd, config.ready_fd})
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    identity = dict(base._identity(config), schema=SCHEMA)
    allocation = allocation_factory(config.gpu_uuid)
    sequence = 0
    def receipt(phase, error=None):
        base._write_record(config.ready_fd, dict(identity, phase=phase, sequence=sequence,
            allocated_bytes=allocation.allocated_bytes if allocation.pointer is not None else 0,
            total_memory_bytes=allocation.total_memory_bytes, driver_version=allocation.driver_version,
            context_created=allocation.context is not None, error=error))
    try:
        if not base._control_open(config.control_fd):
            raise RuntimeError("Parent control pipe closed before initialization")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_uuid
        allocation.acquire()
        base._check_parent(config)
        if stopped[0]:
            raise RuntimeError("Stopped before reservoir readiness")
        receipt("ready")
        pending = b""
        while not stopped[0]:
            base._check_parent(config)
            readable, _, _ = select.select([config.control_fd], [], [], 0.25)
            if not readable:
                continue
            block = os.read(config.control_fd, 4096)
            if not block:
                return 0
            pending += block
            if len(pending) > 8192:
                raise ValueError("Parent command exceeds bounded size")
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                sequence, mib = command(json.loads(line), config.nonce, sequence)
                try:
                    allocation.resize(mib)
                    receipt("resized")
                except Exception as error:
                    # Keep all allocations still owned after failed growth.
                    receipt("resize_failed", str(error))
        return 0
    except Exception as error:
        receipt("error", str(error))
        return 1
    finally:
        errors = allocation.close()
        if errors:
            print(json.dumps({"schema": SCHEMA, "cleanup_errors": errors}), flush=True)
        os.close(config.control_fd)
        os.close(config.ready_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gpu-uuid", "control-fd", "ready-fd", "nonce", "parent-pid", "parent-start-ticks"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    return run(base.Config(**vars(args)))


if __name__ == "__main__":
    raise SystemExit(main())
