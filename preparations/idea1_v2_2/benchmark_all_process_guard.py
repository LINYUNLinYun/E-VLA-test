"""Sampled all-process NVML guard; never claims continuous nanosecond isolation."""
import ctypes as C
import ctypes.util
import multiprocessing as mp
import math
import os
from pathlib import Path
import time

from benchmark_gpu_identity import local_identity

INTERVAL = .1
MAX_GAP = .5
MAX_EVENTS = 100000
HEADER = '/usr/local/cuda/targets/x86_64-linux/include/nvml.h'
HEADER_SHA256 = '899f3c7d6d5ead655d32d0bca566ecf1a32039067ccf45400f4341db9a18b04f'


class ProcessInfo(C.Structure):
    # nvml.h:285..298, v2 function prototypes:11034/11036. No v3 ABI guessing.
    _fields_ = [('pid', C.c_uint), ('usedGpuMemory', C.c_ulonglong),
                ('gpuInstanceId', C.c_uint), ('computeInstanceId', C.c_uint)]


def abi_identity():
    from common import sha
    if sha(HEADER) != HEADER_SHA256 or C.sizeof(ProcessInfo) != 24:
        raise RuntimeError('Reviewed local NVML header/LP64 process ABI changed')
    library = C.CDLL(ctypes_library())  # Loading symbols is CPU-only; never calls nvmlInit.
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if 'libnvidia-ml.so' in line and line.split()[-1].startswith('/')}
    if len(paths) != 1:
        raise RuntimeError('Cannot identify exactly one installed NVML shared library')
    binary = Path(next(iter(paths))).resolve()
    return dict(header=HEADER, header_sha256=HEADER_SHA256, structure_size=C.sizeof(ProcessInfo),
        offsets={name:getattr(ProcessInfo, name).offset for name, _ in ProcessInfo._fields_}, api='explicit _v2',
        library_path=str(binary), library_sha256=sha(binary))


class Scanner:
    def __init__(self, gpu=None, uuid=None):
        abi_identity()
        ctypes_path = ctypes_library()
        self.lib = C.CDLL(ctypes_path)
        def function(name, arguments):
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = arguments, C.c_int
            return fn
        self.init = function('nvmlInit_v2', [])
        self.shutdown = function('nvmlShutdown', [])
        self.by_index = function('nvmlDeviceGetHandleByIndex_v2', [C.c_uint, C.POINTER(C.c_void_p)])
        self.by_uuid = function('nvmlDeviceGetHandleByUUID', [C.c_char_p, C.POINTER(C.c_void_p)])
        self.get_uuid = function('nvmlDeviceGetUUID', [C.c_void_p, C.c_char_p, C.c_uint])
        self.compute = function('nvmlDeviceGetComputeRunningProcesses_v2', [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(ProcessInfo)])
        self.graphics = function('nvmlDeviceGetGraphicsRunningProcesses_v2', [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(ProcessInfo)])
        self.check(self.init())
        handle = C.c_void_p()
        self.check(self.by_uuid(uuid.encode(), C.byref(handle)) if uuid else self.by_index(int(gpu), C.byref(handle)))
        actual = C.create_string_buffer(96)
        self.check(self.get_uuid(handle, actual, len(actual)))
        self.uuid = actual.value.decode()
        if not self.uuid.startswith('GPU-') or (uuid and self.uuid != uuid):
            raise RuntimeError('NVML UUID identity differs')

    @staticmethod
    def check(code):
        if code != 0:
            raise RuntimeError(f'NVML query unavailable/failed: {code}')

    @staticmethod
    def processes(function, handle):
        capacity = 16
        for _ in range(5):
            count = C.c_uint(capacity)
            values = (ProcessInfo*capacity)()
            code = function(handle, C.byref(count), values)
            if code == 7:  # NVML_ERROR_INSUFFICIENT_SIZE, bounded race retry.
                capacity = max(capacity*2, count.value+16)
                if capacity > 65536:
                    raise RuntimeError('Unbounded NVML process list')
                continue
            Scanner.check(code)
            if count.value > capacity:
                raise RuntimeError('Malformed NVML process count')
            result = {}
            for value in values[:count.value]:
                if value.pid <= 0 or value.pid in result:
                    raise RuntimeError('Malformed/duplicate NVML PID')
                # Unavailable memory is not an empty process. Binding requires known memory.
                result[value.pid] = None if value.usedGpuMemory == 2**64-1 else value.usedGpuMemory/1024**2
            return result
        raise RuntimeError('NVML process list never stabilized')

    def scan(self):
        start = time.monotonic()
        handle = C.c_void_p()
        self.check(self.by_uuid(self.uuid.encode(), C.byref(handle)))
        value = C.create_string_buffer(96)
        self.check(self.get_uuid(handle, value, len(value)))
        if value.value.decode() != self.uuid:
            raise RuntimeError('Bound physical UUID changed')
        compute = self.processes(self.compute, handle)
        graphics = self.processes(self.graphics, handle)
        return dict(uuid=self.uuid, processes=compute, graphics=graphics, timestamp=time.time(),
            monotonic_start=start, monotonic_end=time.monotonic())

    def close(self):
        self.check(self.shutdown())


def ctypes_library():
    value = C.util.find_library('nvidia-ml')
    if not value:
        raise RuntimeError('Installed NVML library missing; no implicit installation')
    return value


def parent_identity(pid):
    tail = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=tail[19], pid_namespace=os.readlink(f'/proc/{pid}/ns/pid'))


def validate_sample(row, uuid, pid, previous=None):
    if any(not isinstance(row[field], (int,float)) or isinstance(row[field], bool) or not math.isfinite(row[field]) for field in ('monotonic_start','monotonic_end')) or row['monotonic_end'] < row['monotonic_start']:
        raise RuntimeError('Invalid/nonfinite/reversed NVML observation clock')
    if row['uuid'] != uuid or set(map(int, row['processes'])) != {pid} or row['graphics']:
        raise RuntimeError('Compute/graphics exclusivity failed')
    if row['monotonic_end']-row['monotonic_start'] > MAX_GAP:
        raise RuntimeError('NVML scan itself exceeded maximum coverage gap')
    if previous is not None and (row['monotonic_start'] < previous['monotonic_end']
            or row['monotonic_end']-previous['monotonic_start'] > MAX_GAP):
        raise RuntimeError('Monitor sample coverage gap exceeded bound')


def monitor_child(pipe, uuid, pid, local):
    scanner, events, error, previous, sequence = None, [], None, None, 0
    def sample(label):
        nonlocal error, previous, sequence
        if error:
            return
        try:
            if parent_identity(local['pid']) != local:
                raise RuntimeError('Benchmark owner identity changed')
            row = scanner.scan()
            sequence += 1
            row.update(sequence=sequence, label=label, local=local)
            events.append(row)
            validate_sample(row, uuid, pid, previous)
            previous = row
            if len(events) > MAX_EVENTS:
                raise RuntimeError('Bounded monitor event buffer exhausted')
        except BaseException as exc:
            error = repr(exc)
    try:
        scanner = Scanner(uuid=uuid)
        sample('monitor_start')
        pipe.send(dict(error=error, ready=True))
        while True:
            if pipe.poll(INTERVAL):
                request = pipe.recv()
                operation, label = request
                if operation in ('check', 'flush', 'stop'):
                    sample(label)
                else:
                    error = 'Unknown guard command'
                response = dict(error=error, sequence=sequence)
                if operation in ('flush', 'stop'):
                    response['events'], events = events, []
                pipe.send(response)
                if operation == 'stop':
                    return
            else:
                sample('periodic')
    except BaseException as exc:
        try: pipe.send(dict(error=repr(exc), events=events))
        except (BrokenPipeError, OSError): pass
    finally:
        if scanner:
            scanner.close()


class Monitor:
    def __init__(self, uuid, pid, local):
        context = mp.get_context('spawn')
        self.pipe, child = context.Pipe()
        self.process = context.Process(target=monitor_child, args=(child, uuid, pid, local), daemon=True)
        self.process.start()
        child.close()
        ready = self.receive()
        if ready.get('error'):
            raise RuntimeError(ready['error'])

    def receive(self):
        if not self.pipe.poll(10):
            raise RuntimeError('Monitor missing/crashed/unresponsive; attempt invalid')
        row = self.pipe.recv()
        return row

    def request(self, operation, label):
        if not self.process.is_alive():
            raise RuntimeError('Monitor died; attempt invalid')
        self.pipe.send((operation, label))
        return self.receive()

    def check(self, label):
        row = self.request('check', label)
        if row.get('error'):
            raise RuntimeError(row['error'])
        return row['sequence']

    def flush(self, label, stop=False):
        row = self.request('stop' if stop else 'flush', label)
        if stop:
            self.process.join(10)
            if self.process.is_alive() or self.process.exitcode != 0:
                raise RuntimeError('Monitor did not stop cleanly')
        return row
