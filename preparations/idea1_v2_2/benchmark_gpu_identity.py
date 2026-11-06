"""Fail-closed causal CUDA/NVML identity binding; not an OS PID mapping."""
import os
import subprocess
import time
from pathlib import Path


def local_identity():
    stat = Path('/proc/self/stat').read_text().split(') ', 1)[1].split()
    return dict(pid=os.getpid(), start_ticks=stat[19],
                pid_namespace=os.readlink('/proc/self/ns/pid'))


def scan(gpu):
    uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                    '--format=csv,noheader'], text=True).strip()
    if not uuid.startswith('GPU-') or '\n' in uuid:
        raise RuntimeError('Cannot resolve exactly one physical GPU UUID')
    output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    processes = {}
    for line in output.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 3:
            raise RuntimeError('Malformed NVML process record')
        if fields[0] == uuid:
            pid, memory = int(fields[1]), int(fields[2])
            if pid <= 0 or pid in processes or memory < 0:
                raise RuntimeError('Invalid/duplicate NVML process record')
            processes[pid] = memory
    return dict(gpu=str(gpu), uuid=uuid, processes=processes, timestamp=time.time())


class CausalIdentity:
    def __init__(self, gpu, record, scanner=scan, identity=local_identity):
        self.gpu, self.record, self.scanner, self.identity = str(gpu), record, scanner, identity
        self.local = identity()
        self.uuid = None
        self.pid = None

    def capture(self, label, expected=None):
        if self.identity() != self.local:
            raise RuntimeError('Local process identity changed')
        row = self.scanner(self.gpu)
        if self.uuid is None:
            self.uuid = row['uuid']
        if row['uuid'] != self.uuid:
            raise RuntimeError('Physical GPU UUID changed')
        if expected is not None and set(row['processes']) != set(expected):
            raise RuntimeError(f"GPU process set changed: {row['processes']}; expected {expected}")
        self.record(dict(label=label, local=self.local, **row))
        return row

    def bind(self, torch):
        if self.pid is not None:
            raise RuntimeError('Identity already bound')
        self.capture('empty_baseline', [])
        if torch.cuda.is_initialized():
            raise RuntimeError('CUDA initialized before empty baseline')
        torch.cuda.init()
        self.keepalive = torch.empty(1, dtype=torch.uint8, device='cuda:0')
        torch.cuda.synchronize()
        actual_uuid = str(torch.cuda.get_device_properties(0).uuid)
        normalize = lambda x: x.lower().replace('gpu-', '').replace('-', '')
        if normalize(actual_uuid) != normalize(self.uuid):
            raise RuntimeError('CUDA and NVML UUID differ')
        initial = self.capture('context_created')
        if len(initial['processes']) != 1:
            raise RuntimeError('CUDA context must produce exactly one NVML PID')
        pid = next(iter(initial['processes']))
        baseline = self.capture('context_stable', [pid])['processes'][pid]
        # Real allocation challenge before model load and all benchmark timing.
        probe = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device='cuda:0')
        probe.fill_(73)
        torch.cuda.synchronize()
        allocated = self.capture('challenge_allocated', [pid])['processes'][pid]
        if allocated - baseline < 64:
            raise RuntimeError('NVML memory did not respond to own allocation')
        del probe
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        released = self.capture('challenge_released', [pid])['processes'][pid]
        if allocated - released < 64:
            raise RuntimeError('NVML memory did not respond to own release')
        self.pid = pid
        self.capture('causal_identity_bound', [pid])

    def assert_exclusive(self, snapshot=False):
        if self.pid is None:
            raise RuntimeError('CUDA/NVML identity is not bound')
        row = self.capture('benchmark_snapshot' if snapshot else 'point_boundary', [self.pid])
        row.update(pids=list(row['processes']), local=self.local,
                   identity_method='empty baseline + CUDA UUID + allocation/release memory challenge',
                   cpu_isolation='not guaranteed; process snapshots retained')
        if snapshot:
            row['processes_snapshot'] = subprocess.check_output(['ps', '-eo', 'pid,pcpu,comm'], text=True)
        return row
