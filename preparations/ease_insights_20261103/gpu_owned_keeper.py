"""Keep one small, explicitly assigned CUDA allocation visible until lease EOF.

This module has no CUDA/driver side effects on import. Only its explicit CLI
loads libcuda. The session parent must launch it with ``close_fds=True`` and
``pass_fds=(control_read, ready_write)``; never pass the project GPU flock FD.
It must retain the sole control writer across empty queues/stage preparation,
and close it only on explicit release, pause, or fatal session shutdown.

CLI: --gpu-uuid FULL_UUID --control-fd READ_FD --ready-fd WRITE_FD --nonce TOKEN
     --parent-pid PID --parent-start-ticks TICKS [--allocation-mib 64]

The ready pipe receives exactly one newline-terminated JSON record, schema
``ease-gpu-owned-keeper-v1``, phase ``ready`` or ``error``, then closes. A ready
record includes PID/start ticks/nonce, actual GPU UUID, allocation/capacity,
context_created and driver_version. ``nspid`` is for audit only: /proc's PID
namespace may differ from nvidia-smi's; this module never asserts a host PID.
The control pipe is an EOF lease, not a command channel. Queue completion is
irrelevant to the keeper. EOF, SIGTERM, SIGINT or loss of the proven parent
releases only its own allocation/context. There are no kernels or subprocesses.
The parent must impose a bounded ready timeout and check the ready identity;
a CUDA driver call that hangs cannot be made cancellable by a Python pipe.

ABI source: NVIDIA CUDA 12.4 official cuda.h (CUdevice=int, CUcontext=pointer,
CUuuid=16 bytes, CUdeviceptr=pointer-sized unsigned, sizes=size_t), declarations
cuInit, cuDriverGetVersion, cuDeviceGet/GetCount/GetUuid_v2/TotalMem_v2,
cuCtxCreate_v2/Destroy_v2, cuMemAlloc_v2/Free_v2. UUID v2 deliberately rejects a
MIG instance whose UUID differs from the explicitly assigned full GPU UUID.
"""
import argparse
import ctypes
from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import sys
import uuid


SCHEMA = "ease-gpu-owned-keeper-v1"
DEFAULT_ALLOCATION_MIB = 64
MAX_ALLOCATION_MIB = 1024
MIB = 1024 * 1024
UUID_PATTERN = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")
DEVICE_POINTER = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]


@dataclass(frozen=True)
class Config:
    gpu_uuid: str
    control_fd: int
    ready_fd: int
    nonce: str
    parent_pid: int
    parent_start_ticks: int
    allocation_mib: int = DEFAULT_ALLOCATION_MIB


def proc_start_ticks(pid):
    """Read Linux field 22 without mistaking spaces/parentheses in comm fields."""
    tail = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(tail[19])


def _integer(value, name, lower, upper=None):
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise ValueError(f"{name} must be an integer")
    number = int(value)
    if number < lower or (upper is not None and number > upper):
        raise ValueError(f"{name} is outside its allowed range")
    return number


def validate_config(config):
    """Validate every supplied setting/FD before loading any CUDA library."""
    if not isinstance(config.gpu_uuid, str) or not UUID_PATTERN.fullmatch(config.gpu_uuid):
        raise ValueError("gpu_uuid must be one complete GPU UUID")
    if not isinstance(config.nonce, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", config.nonce):
        raise ValueError("nonce must contain 1..256 token characters")
    normalized = Config(
        gpu_uuid="GPU-" + str(uuid.UUID(config.gpu_uuid[4:])),
        control_fd=_integer(config.control_fd, "control_fd", 3),
        ready_fd=_integer(config.ready_fd, "ready_fd", 3),
        nonce=config.nonce,
        parent_pid=_integer(config.parent_pid, "parent_pid", 1),
        parent_start_ticks=_integer(config.parent_start_ticks, "parent_start_ticks", 1),
        allocation_mib=_integer(config.allocation_mib, "allocation_mib", 1, MAX_ALLOCATION_MIB),
    )
    if normalized.control_fd == normalized.ready_fd:
        raise ValueError("control_fd and ready_fd must be distinct")
    for fd, mode, name in ((normalized.control_fd, os.O_RDONLY, "control_fd"),
                           (normalized.ready_fd, os.O_WRONLY, "ready_fd")):
        if not stat.S_ISFIFO(os.fstat(fd).st_mode):
            raise ValueError(f"{name} must be a pipe")
        if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != mode:
            raise ValueError(f"{name} has the wrong pipe direction")
    _check_parent(normalized)
    return normalized


def _check_parent(config):
    if os.getppid() != config.parent_pid or proc_start_ticks(config.parent_pid) != config.parent_start_ticks:
        raise RuntimeError("Session parent PID/start identity no longer matches")


def _control_open(fd, timeout=0):
    readable, _, _ = select.select([fd], [], [], timeout)
    if not readable:
        return True
    if os.read(fd, 1):
        raise ValueError("control pipe accepts EOF only, not messages")
    return False


def _identity(config):
    record = dict(schema=SCHEMA, gpu_uuid=config.gpu_uuid, pid=os.getpid(),
                  proc_start_ticks=proc_start_ticks(os.getpid()), nonce=config.nonce)
    # These are audit fields, not a host-to-container PID mapping guarantee.
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                record["nspid"] = [int(value) for value in line.split()[1:]]
        record["pid_namespace_inode"] = os.stat("/proc/self/ns/pid").st_ino
    except (OSError, ValueError):
        pass
    return record


class DriverAllocation:
    """One context/allocation; the loader is injectable for CPU-only tests."""
    def __init__(self, gpu_uuid, allocated_bytes, loader=None):
        self.gpu_uuid = gpu_uuid
        self.allocated_bytes = allocated_bytes
        self.total_memory_bytes = None
        self.driver_version = None
        self.context = None
        self.pointer = None
        self.loader = loader
        self.functions = {}

    def _bind(self, library, name, arguments):
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
        self.functions[name] = function

    def _call(self, name, *arguments):
        result = self.functions[name](*arguments)
        if result != 0:
            raise RuntimeError(f"{name} failed with CUDA error {result}")

    def acquire(self):
        # No library is loaded until configuration, parent and lease checks pass.
        library = (self.loader or ctypes.CDLL)("libcuda.so.1")
        integer, pointer = ctypes.c_int, ctypes.POINTER
        signatures = {
            "cuInit": [ctypes.c_uint],
            "cuDriverGetVersion": [pointer(integer)],
            "cuDeviceGetCount": [pointer(integer)],
            "cuDeviceGet": [pointer(integer), integer],
            "cuDeviceGetUuid_v2": [pointer(CUuuid), integer],
            "cuDeviceTotalMem_v2": [pointer(ctypes.c_size_t), integer],
            "cuCtxCreate_v2": [pointer(ctypes.c_void_p), ctypes.c_uint, integer],
            "cuCtxDestroy_v2": [ctypes.c_void_p],
            "cuMemAlloc_v2": [pointer(DEVICE_POINTER), ctypes.c_size_t],
            "cuMemFree_v2": [DEVICE_POINTER],
        }
        for name, arguments in signatures.items():
            self._bind(library, name, arguments)
        self._call("cuInit", 0)
        version, count = integer(), integer()
        self._call("cuDriverGetVersion", ctypes.byref(version))
        self.driver_version = version.value
        if self.driver_version <= 0:
            raise RuntimeError("CUDA driver reported an invalid version")
        self._call("cuDeviceGetCount", ctypes.byref(count))
        if count.value != 1:
            raise RuntimeError("Assigned UUID must expose exactly one CUDA device")
        device, identity = integer(), CUuuid()
        self._call("cuDeviceGet", ctypes.byref(device), 0)
        self._call("cuDeviceGetUuid_v2", ctypes.byref(identity), device)
        actual_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(identity.bytes)))
        if actual_uuid != self.gpu_uuid:
            raise RuntimeError(f"CUDA device UUID mismatch: {actual_uuid}")
        capacity = ctypes.c_size_t()
        self._call("cuDeviceTotalMem_v2", ctypes.byref(capacity), device)
        self.total_memory_bytes = capacity.value
        if not 0 < self.allocated_bytes <= MAX_ALLOCATION_MIB * MIB:
            raise ValueError("Allocation exceeds the absolute memory bound")
        if self.allocated_bytes * 20 > self.total_memory_bytes:
            raise ValueError("Allocation exceeds 5% of total device memory")
        context = ctypes.c_void_p()
        self._call("cuCtxCreate_v2", ctypes.byref(context), 0, device)
        if not context.value:
            raise RuntimeError("CUDA returned a null context")
        self.context = context
        allocation = DEVICE_POINTER()
        self._call("cuMemAlloc_v2", ctypes.byref(allocation), self.allocated_bytes)
        if not allocation.value:
            raise RuntimeError("CUDA returned a null allocation")
        self.pointer = allocation

    def close(self):
        """Attempt both owned cleanups even if freeing memory fails; never signal."""
        errors = []
        for attribute, function in (("pointer", "cuMemFree_v2"), ("context", "cuCtxDestroy_v2")):
            resource = getattr(self, attribute)
            if resource is not None:
                setattr(self, attribute, None)
                try:
                    self._call(function, resource)
                except Exception as error:
                    errors.append(str(error))
        return errors


def _write_record(fd, record):
    payload = (json.dumps(record, ensure_ascii=True) + "\n").encode()
    while payload:
        written = os.write(fd, payload)
        if written <= 0:
            raise OSError("ready pipe write made no progress")
        payload = payload[written:]


def _diagnostic_pipe_fd(value, mode):
    """Recover only a valid directional pipe FD when another setting is bad."""
    try:
        fd = _integer(value, "fd", 3)
        if stat.S_ISFIFO(os.fstat(fd).st_mode) and fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == mode:
            return fd
    except (OSError, ValueError):
        pass
    return None


def run_keeper(config, allocation_factory=DriverAllocation):
    """Run the explicit helper lifecycle; injectable driver tests use no GPUs."""
    allocation = None
    ready_sent = False
    failure = None
    cleanup_errors = []
    old_handlers = {}
    stopped = [False]
    identity = None
    ready_fd = _diagnostic_pipe_fd(config.ready_fd, os.O_WRONLY)
    control_fd = _diagnostic_pipe_fd(config.control_fd, os.O_RDONLY)

    def stop(signum, frame):
        stopped[0] = True

    try:
        config = validate_config(config)
        identity = _identity(config)
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, stop)
        if not _control_open(config.control_fd):
            raise RuntimeError("Control lease closed before CUDA initialization")
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_uuid
        allocation = allocation_factory(config.gpu_uuid, config.allocation_mib * MIB)
        allocation.acquire()
        _check_parent(config)
        if stopped[0] or not _control_open(config.control_fd):
            raise RuntimeError("Control lease closed or stopped before ready")
        _write_record(config.ready_fd, dict(identity, phase="ready", context_created=True,
            allocated_bytes=allocation.allocated_bytes, total_memory_bytes=allocation.total_memory_bytes,
            driver_version=allocation.driver_version))
        ready_sent = True
        os.close(config.ready_fd)
        while not stopped[0] and _control_open(config.control_fd, timeout=0.25):
            _check_parent(config)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if allocation is not None:
            cleanup_errors = allocation.close()
        if not ready_sent:
            try:
                record = identity or _identity(config)
                if ready_fd is not None:
                    _write_record(ready_fd, dict(record, phase="error", context_created=False,
                        error=failure or "Stopped before ready", cleanup_errors=cleanup_errors))
            except (OSError, ValueError, IndexError):
                pass
            if ready_fd is not None:
                try:
                    os.close(ready_fd)
                except OSError:
                    pass
        if control_fd is not None:
            try:
                os.close(control_fd)
            except OSError:
                pass
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    if failure or cleanup_errors:
        if ready_sent:
            print(json.dumps(dict(schema=SCHEMA, phase="error", error=failure,
                                  cleanup_errors=cleanup_errors)), file=sys.stderr, flush=True)
        return 1
    return 0


def _close_unneeded_fds(keep):
    """Defensively drop inherited files, including any accidental GPU flock."""
    for entry in os.listdir("/proc/self/fd"):
        fd = int(entry)
        if fd not in keep:
            try:
                os.close(fd)
            except OSError:
                pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gpu-uuid", "control-fd", "ready-fd", "nonce", "parent-pid", "parent-start-ticks"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--allocation-mib", default=DEFAULT_ALLOCATION_MIB)
    args = parser.parse_args(argv)
    config = Config(**vars(args))
    try:
        config = validate_config(config)
    except (OSError, ValueError, RuntimeError):
        # run_keeper emits a structured error where the supplied ready FD works.
        return run_keeper(config)
    _close_unneeded_fds({0, 1, 2, config.control_fd, config.ready_fd})
    return run_keeper(config)


if __name__ == "__main__":
    raise SystemExit(main())
