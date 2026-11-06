"""Native serial B1 RPC service, with budget changes at Full boundaries."""
import json
import os
import pickle
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import traceback

from .runtime import BudgetedPolicy, checked_budget


class ClientSessions:
    """RPC state whose client/prompt registrations commit after valid calls."""
    def __init__(self, backend, budget, protocol, *, actors=1, cost_weighted=False,
                 artifact_root=None, policy_factory=BudgetedPolicy, events=None):
        self.backend, self.budget, self.protocol = backend, budget, protocol
        self.actors, self.cost_weighted = actors, cost_weighted
        self.artifact_root, self.policy_factory, self.events = artifact_root, policy_factory, events
        self.policies, self.group_prompt = {}, None

    def handle(self, request):
        request = dict(request)
        operation = request.pop('op')
        client = request.pop('client', 'default')
        if operation == 'health':
            return dict(protocol=self.protocol, clients={key: dict(
                current_budget=value.current_budget['budget_id'],
                queued_budget=None if value.queued_budget is None else value.queued_budget['budget_id'])
                for key, value in self.policies.items()})
        if operation == 'reset_all':
            if any(policy.pending is not None for policy in self.policies.values()):
                raise RuntimeError('Unacknowledged action during group reset')
            for policy in self.policies.values():
                policy.close()
            self.policies.clear()
            self.group_prompt = None
            return {}
        if operation not in ('configure', 'set_budget', 'predict', 'ack'):
            raise ValueError('Unknown RPC operation: ' + str(operation))
        created = client not in self.policies
        if created:
            if len(self.policies) >= self.actors:
                raise RuntimeError('Client count exceeds configured environment workers')
            policy = self.policy_factory(self.backend, self.budget,
                cost_weighted=self.cost_weighted, artifact_root=self.artifact_root)
        else:
            policy = self.policies[client]
        try:
            if operation == 'configure':
                response = policy.configure(**request)
            elif operation == 'set_budget':
                response = policy.set_budget(**request)
                if self.events is not None:
                    self.events.write(json.dumps(dict(client=client, **response, time=time.time()))+'\n')
            elif operation == 'predict':
                instruction = request['instruction']
                if self.group_prompt is not None and instruction != self.group_prompt:
                    raise ValueError('Clients in one group must share their task/prompt')
                response = policy.predict(**request)
                self.group_prompt = instruction
            else:
                response = policy.acknowledge(**request)
        except Exception:
            if created:
                policy.close()
            raise
        if created:
            self.policies[client] = policy
        return response


def serve(args):
    if args.gpu < 0 or not 0 <= args.port <= 65535:
        raise ValueError('Invalid GPU index or listening port')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4')
    budget = checked_budget(args.budget)
    active = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}',
        '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    if active:
        raise RuntimeError('Selected GPU is occupied; choose an available GPU')
    root = Path(__file__).resolve().parents[2]
    for path in (root, root/'policies/realtime-vla-flash/src', root/'policies/openpi/src',
                 root/'policies/openpi/packages/openpi-client/src', root/'LIBERO'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault('LIBERO_CONFIG_PATH', str(root/'policies/realtime-vla-flash/.libero'))
    import torch
    from preparations.idea1_v3.io import atomic
    from preparations.idea1_v3.safe_qk import install
    from preparations.idea1_v2_1.runtime_client import recv_exact
    from preparations.idea1_v3_4_E2_module_skip.backend import Backend
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    install(str(args.output / 'qk_receipt.json'))
    backend = Backend()
    source = json.loads(Path(budget['cost_table_path']).read_text())
    calibration = source.get('provenance', {}).get('protocol', {})
    for key in ('checkpoint_sha256', 'dtype', 'attention', 'backend'):
        if key in ('checkpoint_sha256', 'dtype') or key in calibration:
            if calibration.get(key) is None or calibration[key] != backend.protocol.get(key):
                raise ValueError('Budget hardware calibration model protocol mismatch: ' + key)
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('127.0.0.1', args.port))
        server.listen(16)
        port = server.getsockname()[1]
        protocol = dict(backend.protocol, experiment='adjustable_mixed_budget',
            environment_workers=args.actors, server_pid=os.getpid(), port=port,
            budget_artifact=budget['artifact_path'], initial_budget_hash=budget['budget_hash'],
            budget_scope='net VLM encoder time saved; Full-D10/Fast-D2x4 retained',
            budget_switch='next Full only, preserving the current Fast mask and Full history',
            service_mode='serial B1 with isolated same-task clients',
            model_load_s=time.perf_counter()-started)
        atomic(args.output / 'protocol.json', protocol)
        print(json.dumps(dict(status='ready', port=port, protocol_path=str(args.output/'protocol.json'))), flush=True)
        with (args.output / 'budget_events.jsonl').open('a', buffering=1) as events:
            sessions = ClientSessions(backend, budget, protocol, actors=args.actors,
                cost_weighted=args.weighted, artifact_root=args.output/'budgets', events=events)
            while True:
                connection, _ = server.accept()
                with connection:
                    try:
                        size = struct.unpack('!I', recv_exact(connection, 4))[0]
                        if size > 128 * 1024 * 1024:
                            raise ValueError('RPC request exceeds limit')
                        request = pickle.loads(recv_exact(connection, size))
                        response = sessions.handle(request)
                        response['ok'] = True
                    except Exception:
                        response = dict(ok=False, error=traceback.format_exc())
                    payload = pickle.dumps(response, protocol=5)
                    connection.sendall(struct.pack('!I', len(payload))+payload)
