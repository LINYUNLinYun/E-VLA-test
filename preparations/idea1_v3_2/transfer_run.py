"""One/two persistent serialized policy shards; explicit assigned idle cards only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from .transfer_protocol import ROOT, MODES, load_manifest, root_path, complete_result, key, write_json, sha


def receive(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size-len(data))
        if not chunk:
            raise ConnectionError('Truncated localhost RPC')
        data.extend(chunk)
    return bytes(data)


def rpc(port, request, timeout=10):
    payload = pickle.dumps(request, protocol=5)
    with socket.create_connection(('127.0.0.1', port), timeout=timeout) as sock:
        sock.sendall(struct.pack('!I', len(payload))+payload)
        size = struct.unpack('!I', receive(sock, 4))[0]
        if size > 128*1024*1024:
            raise ValueError('Oversized RPC response')
        result = pickle.loads(receive(sock, size))
    if not result.get('ok', False):
        raise RuntimeError('Service technical error: '+str(result))
    return result


def gpu_inventory(gpu):
    def query(args):
        return subprocess.run(['nvidia-smi', '-i', str(gpu), *args], capture_output=True, text=True, check=True).stdout.strip()
    values = query(['--query-gpu=index,uuid,pci.bus_id,memory.used,memory.total', '--format=csv,noheader,nounits']).split(',')
    pids = query(['--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    return dict(index=int(values[0].strip()), uuid=values[1].strip(), pci_bus_id=values[2].strip(),
                used_mib=int(values[3]), total_mib=int(values[4]),
                compute_pids=[int(p.strip()) for p in pids.splitlines() if p.strip().isdigit()],
                observed_unix=time.time())


def process_identity(pid):
    path = Path('/proc')/str(pid)
    # comm in parentheses may contain spaces; starttime is field22, relative20.
    stat = (path/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=stat[19], command=(path/'cmdline').read_bytes().replace(b'\0', b' ').decode())


def stop_owned(process, identity):
    if process.poll() is not None:
        return dict(pid=process.pid, status='already_exited', returncode=process.returncode)
    actual = process_identity(process.pid)
    if actual != identity:
        raise RuntimeError('Owned PID identity changed; refusing to signal')
    process.terminate()
    try:
        code = process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if process_identity(process.pid) != identity:
            raise RuntimeError('PID changed before final termination')
        process.kill(); code = process.wait(timeout=10)
    return dict(pid=process.pid, status='stopped_exact_owned_child', returncode=code)


def namespace_ownership(process, identity, before, after, health_uuid):
    """NVML PIDs can be host IDs absent from this container's /proc.

    Accept direct NSpid identity when available, otherwise retain a constrained
    before/after proof: idle assigned UUID, one new compute PID, live exact child,
    and its authenticated localhost health reporting the same physical UUID.
    Never signal an NVML PID; cleanup uses only the Popen-owned local child.
    """
    if process.poll() is not None or process_identity(process.pid) != identity:
        raise RuntimeError('Local owned service identity is no longer live')
    if before['compute_pids'] or before['uuid'] != after['uuid'] or health_uuid != before['uuid']:
        raise RuntimeError('GPU UUID or initial-idle proof differs')
    if len(after['compute_pids']) != 1:
        raise RuntimeError('Expected exactly one new compute process on assigned idle card')
    status = (Path('/proc')/str(process.pid)/'status').read_text()
    namespace_line = next((line for line in status.splitlines() if line.startswith('NSpid:')), None)
    namespace_pids = [int(x) for x in namespace_line.split()[1:]] if namespace_line else [process.pid]
    method = 'explicit_NSpid_match' if after['compute_pids'][0] in namespace_pids else 'sole_new_NVML_process_on_previously_idle_UUID_plus_live_authenticated_child'
    return dict(local_child_identity=identity, namespace_status_line=namespace_line, namespace_pids=namespace_pids,
                nvml_compute_pids=after['compute_pids'], nvml_uuid=after['uuid'], method=method,
                before=before, after=after, signal_targets='Only exact local Popen child; never NVML host PID')


def child_environment(gpu_uuid, env_kind=None):
    env = dict(os.environ)
    paths = [str(ROOT), str(ROOT/'policies/openpi/src'), str(ROOT/'policies/openpi/packages/openpi-client/src'), str(ROOT/'LIBERO')]
    env.update(PYTHONPATH=':'.join(paths+[env.get('PYTHONPATH', '')]), PYTHONUNBUFFERED='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=gpu_uuid,
               XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.85',
               MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', TRANSFER_GPU_UUID=gpu_uuid)
    return env


def find_complete(directory, manifest, case, mode, gpu_uuid):
    for path in sorted(directory.glob('attempt*/result.json')):
        result = json.loads(path.read_text())
        if complete_result(result, manifest, case, mode, gpu_uuid):
            return path, result
    return None


def run(args):
    manifest = load_manifest(args.manifest)
    output = root_path(args.output)
    if Path(args.manifest).resolve() != output/'manifest.json':
        raise ValueError('Run output must own the frozen manifest')
    deadline = args.deadline_unix or manifest['budget']['deadline_unix']
    if deadline > manifest['budget']['deadline_unix']:
        raise ValueError('Cannot extend the frozen E4 deadline')
    environment_pythons = json.loads(Path(args.environment_pythons).read_text())
    if len(args.ports) < len(manifest['gpus']):
        raise ValueError('A distinct service port is required for every frozen shard')
    if set(environment_pythons) != {'calvin', 'simpler', 'pusht'}:
        raise ValueError('Explicit existing interpreter for each environment is required')
    for value in environment_pythons.values():
        if not Path(value).is_file():
            raise FileNotFoundError(value)
    lifecycle = dict(status='running', manifest_hash=manifest['manifest_hash'], controller_pid=os.getpid(),
                     started_unix=time.time(), deadline_unix=deadline, shards={}, failure_policy='technical never policy-zero')
    lifecycle_path = output/f'lifecycle_{time.time_ns()}.json'
    lock = threading.Lock()

    def update():
        with lock:
            write_json(lifecycle_path, lifecycle)

    def shard(shard_id):
        gpu, port = manifest['gpus'][shard_id], args.ports[shard_id]
        state = dict(gpu=gpu, port=port, children=[], results=[], waits=[])
        lifecycle['shards'][str(shard_id)] = state; update()
        selected = [c for c in manifest['cases'] if c['shard'] == shard_id and c['support_status'] == 'supported']
        if not selected:
            state['status'] = 'no_supported_cases'; update(); return
        owned = []
        try:
            # Pending work may wait for an externally occupied assigned card.
            # This loop ends once our service owns it and never treats own PID
            # occupancy as a reason to wait for GPU availability.
            while time.time() < deadline:
                inventory = gpu_inventory(gpu); state['gpu_inventory'] = inventory
                if not inventory['compute_pids']:
                    break
                state['waits'].append(inventory); update()
                time.sleep(min(30, max(0, deadline-time.time())))
            else:
                raise TimeoutError('Deadline while pending assigned idle GPU')
            gpu_uuid = inventory['uuid']
            # Bind check avoids accidentally sending requests to another job.
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', port))
            service_path = output/f'shard{shard_id}_service_{time.time_ns()}.json'
            service_log = service_path.with_suffix('.log')
            command = [args.service_python, '-m', 'preparations.idea1_v3_2.transfer_server', '--protocol', str(output/'manifest.json'),
                       '--port', str(port), '--output', str(service_path), '--deadline-unix', str(deadline), '--launch']
            with service_log.open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=child_environment(gpu_uuid), stdout=log, stderr=subprocess.STDOUT)
            identity = process_identity(process.pid); owned.append((process, identity))
            state['children'].append(dict(role='policy_service', identity=identity, command=command, log=str(service_log)))
            update()
            while True:
                if time.time() >= deadline or process.poll() is not None:
                    raise RuntimeError('Service exited or startup exceeded E4 deadline')
                try:
                    health = rpc(port, dict(op='health'), timeout=3)
                    protocol = health.get('protocol', health)
                    if protocol.get('server_pid') != process.pid or protocol.get('manifest_hash', health.get('manifest_hash')) != manifest['manifest_hash']:
                        raise RuntimeError('Service endpoint PID/manifest mismatch')
                    if not protocol.get('legacy_gate_passed'):
                        raise RuntimeError('Original real-input backend parity gate did not pass')
                    if protocol.get('gpu_uuid') != gpu_uuid:
                        raise RuntimeError('Service GPU UUID mismatch')
                    observed = gpu_inventory(gpu)
                    ownership = namespace_ownership(process, identity, inventory, observed, protocol['gpu_uuid'])
                    state['health'] = health; state['gpu_owned_inventory'] = observed
                    state['pid_namespace_evidence'] = ownership; update(); break
                except (ConnectionError, OSError):
                    time.sleep(1)
            for case in selected:
                # All six conditions for a case remain on this fixed GPU.
                for mode in MODES:
                    episode_key = key(case['environment'], case['case'], mode)
                    directory = output/'episodes'/episode_key
                    previous = find_complete(directory, manifest, case, mode, gpu_uuid)
                    if previous:
                        state['results'].append(dict(key=episode_key, status='resumed_complete', result=str(previous[0]))); update(); continue
                    if time.time() >= deadline:
                        raise TimeoutError('E4 total deadline reached')
                    attempts = sorted(directory.glob('attempt*'))
                    target = directory/f'attempt{len(attempts):03d}'
                    target.mkdir(parents=True, exist_ok=False)
                    command = [environment_pythons[case['environment']], '-m', 'preparations.idea1_v3_2.transfer_client',
                               '--environment', case['environment'], '--case', str(case['case']), '--mode', mode,
                               '--port', str(port), '--source-reference', manifest['source_reference_path'],
                               '--output', str(target), '--gpu', str(gpu), '--gpu-uuid', gpu_uuid,
                               '--manifest', str(output/'manifest.json'), '--deadline-unix', str(deadline), '--launch']
                    if case['case'] == 0:
                        command.append('--video')
                    with (target/'client.log').open('x') as log:
                        client = subprocess.Popen(command, cwd=ROOT, env=child_environment(gpu_uuid, case['environment']), stdout=log, stderr=subprocess.STDOUT)
                    client_id = process_identity(client.pid); owned.append((client, client_id))
                    state['children'].append(dict(role='episode_client', identity=client_id, command=command))
                    update()
                    try:
                        client.wait(timeout=max(1, deadline-time.time()))
                    except subprocess.TimeoutExpired:
                        state.setdefault('cleanup', []).append(stop_owned(client, client_id))
                    result_path = target/'result.json'
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        if result.get('status') == 'complete' and not complete_result(result, manifest, case, mode, gpu_uuid):
                            raise ValueError('Completed client identity does not match frozen case')
                    else:
                        result = dict(status='technical_failure', success=None, environment=case['environment'], case=case['case'], mode=mode,
                                      manifest_hash=manifest['manifest_hash'], gpu_uuid=gpu_uuid, failure_code='CLIENT_NO_RESULT', returncode=client.returncode)
                        write_json(result_path, result, exclusive=True)
                    state['results'].append(dict(key=episode_key, status=result['status'], result=str(result_path), returncode=client.returncode)); update()
            state['status'] = 'complete'
        except Exception:
            state['status'] = 'technical_failure'; state['error'] = traceback.format_exc()
        finally:
            for process, identity in reversed(owned):
                try:
                    if process.poll() is None and 'transfer_server' in identity['command']:
                        try:
                            rpc(port, dict(op='shutdown'), timeout=3)
                            process.wait(timeout=5)
                        except Exception:
                            pass
                    state.setdefault('cleanup', []).append(stop_owned(process, identity))
                except Exception:
                    state.setdefault('cleanup_errors', []).append(traceback.format_exc())
            update()

    with ThreadPoolExecutor(max_workers=len(manifest['gpus'])) as pool:
        futures = [pool.submit(shard, i) for i in range(len(manifest['gpus']))]
        for future in futures:
            future.result()
    lifecycle['status'] = 'complete' if all(s['status'] in ('complete', 'no_supported_cases') for s in lifecycle['shards'].values()) else 'partial_technical'
    lifecycle['completed_unix'] = time.time(); update()
    print(json.dumps(dict(status=lifecycle['status'], lifecycle=str(lifecycle_path)), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest'); p.add_argument('--output'); p.add_argument('--environment-pythons')
    p.add_argument('--service-python', default=str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'))
    p.add_argument('--ports', nargs='+', type=int, default=[18641, 18642]); p.add_argument('--deadline-unix', type=float)
    p.add_argument('--launch', action='store_true')
    a = p.parse_args()
    if not a.launch:
        print(json.dumps(dict(dry_run=True, gpu_calls=0, subprocess_launches=0, required='Frozen manifest, one/two explicitly assigned idle GPUs, existing environment Python mapping', **vars(a))))
        return
    if not all((a.manifest, a.output, a.environment_pythons)):
        p.error('--manifest --output --environment-pythons required for launch')
    run(a)


if __name__ == '__main__':
    main()
