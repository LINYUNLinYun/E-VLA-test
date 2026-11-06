"""One source checkpoint, one mode, many target cases; scheduler owns GPU allocation."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, REPO, CHECKPOINTS, SUITES, ARTIFACTS, PROTOCOL, atomic_json, normalize_cases, stage_artifacts

def stop_process(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill(); process.wait()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu')
    p.add_argument('--gpu-uuid')
    p.add_argument('--output', required=True)
    p.add_argument('--manifest', help='JSON list or object with cases; same meaning as --cases')
    p.add_argument('--cases')
    p.add_argument('--source-suite', choices=SUITES, default='libero_goal')
    p.add_argument('--target-suite', default='all', help='all, one suite, or comma-separated suite list')
    p.add_argument('--mode', choices=('flash', 'draft', 'full'), required=True)
    p.add_argument('--init-ids', default='20')
    p.add_argument('--port', type=int)
    p.add_argument('--egl-index', type=int, help='Optional scheduler-verified physical EGL index')
    p.add_argument('--ready-timeout', type=int, default=900)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if a.cases and a.manifest:
        p.error('Use only one of --cases / --manifest')
    gpu = a.gpu_uuid or a.gpu
    if not gpu and not a.dry_run:
        p.error('Explicit scheduler allocation --gpu or --gpu-uuid is required')
    cases = normalize_cases(a.cases or a.manifest, a.target_suite, tuple(int(i) for i in a.init_ids.split(',')))
    output = Path(a.output).resolve()
    if (output / 'result.json').exists():
        previous = json.loads((output / 'result.json').read_text())
        if previous.get('status') == 'complete':
            print(json.dumps(previous)); return
        raise RuntimeError('Existing failed/running output requires scheduler inspection, not blind restart')
    output.mkdir(parents=True, exist_ok=True)
    artifacts = stage_artifacts(output, a.source_suite)
    cases_path = output / 'cases.json'
    atomic_json(cases_path, cases)
    if a.port:
        port = a.port
    else:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    server_cmd = [str(REPO / '.venv/bin/python'), str(HERE / 'server.py'),
        '--mode', a.mode, '--source-suite', a.source_suite, '--artifacts', str(artifacts),
        '--output', str(output), '--port', str(port)]
    client_cmd = [str(REPO / 'examples/libero/.venv/bin/python'), str(HERE / 'client.py'),
        '--mode', a.mode, '--source-suite', a.source_suite, '--cases', str(cases_path),
        '--output', str(output), '--port', str(port)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENPI_DATA_HOME=str(ROOT / 'shared/checkpoints/openpi'),
        LIBERO_CONFIG_PATH=str(REPO / '.libero'), MUJOCO_GL='egl',
        PYTHONPATH=os.pathsep.join((str(ROOT), str(REPO), str(REPO / 'src'),
                                  str(REPO / 'third_party/libero'), str(REPO / 'packages/openpi-client/src'))),
        XDG_CACHE_HOME=str(ROOT / 'shared/caches/xdg'), TOKENIZERS_PARALLELISM='false',
        JAX_PLATFORMS='cpu', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    manifest = dict(protocol=PROTOCOL, source_suite=a.source_suite, mode=a.mode,
        target_suites=list(dict.fromkeys(c['suite'] for c in cases)), cases=cases,
        gpu=gpu, gpu_ordinal=a.gpu, gpu_uuid=a.gpu_uuid,
        base_checkpoint=str(CHECKPOINTS / 'base/base_weights.pkl'),
        draft_checkpoint=str(CHECKPOINTS / ARTIFACTS[a.source_suite] / 'draft_triton.pkl'),
        server_cmd=server_cmd, client_cmd=client_cmd, scheduler_owns_allocation=True,
        pure_draft_semantics='native FLASH head, no verify/fallback/initial Full; native input/output transforms',
        code_files=[str(p) for p in HERE.glob('*.py')])
    import hashlib
    manifest['code_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    atomic_json(output / 'manifest.json', manifest)
    if a.dry_run:
        print(json.dumps(dict(status='dry_run', cases=len(cases), output=str(output), commands=[server_cmd, client_cmd]))); return
    physical_gpu = a.gpu
    if physical_gpu is None:
        inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
        physical_gpu = next(row.split(',')[0].strip() for row in inventory.splitlines() if row.split(',')[1].strip() == a.gpu_uuid)
    physical_gpu = int(physical_gpu)
    if a.egl_index is None:
        sys.path.insert(0, str(ROOT))
        from preparations.idea1_v3_4.dispatch import render_environment
        client_env, devices = render_environment(env, physical_gpu)
    else:
        devices = {str(a.egl_index): physical_gpu}
        client_env = dict(env, CUDA_VISIBLE_DEVICES=','.join(map(str, dict.fromkeys((physical_gpu, a.egl_index)))), MUJOCO_EGL_DEVICE_ID=str(a.egl_index))
    atomic_json(output / 'render_identity.json', dict(physical_gpu=physical_gpu, egl_mapping=devices,
        client_cuda_mask=client_env['CUDA_VISIBLE_DEVICES'], egl_index=client_env['MUJOCO_EGL_DEVICE_ID']))
    server = client = None
    started = time.time()
    result = dict(status='running', started_unix=started, total_cases=len(cases), gpu=gpu,
                  source_suite=a.source_suite, mode=a.mode, output=str(output))
    atomic_json(output / 'result.json', result)
    def interrupted(signum, frame):
        raise RuntimeError('Runner interrupted by signal %d' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (output / 'server.log').open('w') as server_log, (output / 'client.log').open('w') as client_log:
            server = subprocess.Popen(server_cmd, cwd=str(REPO), env=env, stdout=server_log, stderr=subprocess.STDOUT)
            atomic_json(output / 'processes.json', dict(runner_pid=os.getpid(), server_pid=server.pid, port=port))
            deadline = time.time() + a.ready_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Server exited before health check: %d' % server.returncode)
                try:
                    with urllib.request.urlopen('http://127.0.0.1:%d/healthz' % port, timeout=2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    pass
                if time.time() >= deadline:
                    raise TimeoutError('Server readiness timeout; server handle was checked before stop')
                time.sleep(1)
            ready = time.time()
            result['preparation_seconds'] = ready - started
            atomic_json(output / 'result.json', result)
            client = subprocess.Popen(client_cmd, cwd=str(REPO), env=client_env, stdout=client_log, stderr=subprocess.STDOUT)
            atomic_json(output / 'processes.json', dict(runner_pid=os.getpid(), server_pid=server.pid, client_pid=client.pid, port=port))
            while client.poll() is None:
                if server.poll() is not None:
                    raise RuntimeError('Live evaluation server exited: %d' % server.returncode)
                time.sleep(1)
            if client.returncode != 0:
                raise RuntimeError('Client exited nonzero: %d' % client.returncode)
            receipt = json.loads((output / 'client_complete.json').read_text())
            if receipt['cases'] != len(cases):
                raise RuntimeError('Completed count differs from frozen cases')
            result.update(status='complete', completed_unix=time.time(),
                          closed_loop_seconds=time.time() - ready, **{k: receipt[k] for k in ('cases', 'successes')})
    except BaseException:
        result.update(status='technical_failure', completed_unix=time.time(), error=traceback.format_exc())
        raise
    finally:
        stop_process(client); stop_process(server)
        result['wall_seconds'] = time.time() - started
        result['server_exit_code'] = server.poll() if server else None
        result['client_exit_code'] = client.poll() if client else None
        atomic_json(output / 'result.json', result)
    print(json.dumps(result))

if __name__ == '__main__':
    main()
