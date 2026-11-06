"""Persistent native JAX policy service; feature extraction is loaded on demand."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import socket
import struct
import time
import traceback

import numpy as np

from preparations.idea1_v3_1.server import receive
from .common import CHECKPOINT, PROTOCOL, atomic_json, process_identity


class NativePolicy:
    def __init__(self):
        import jax
        from openpi.training import config
        from openpi.policies import policy_config
        self.jax = jax
        self.policy = policy_config.create_trained_policy(
            config.get_config('pi05_libero'), CHECKPOINT, sample_kwargs={'num_steps':10})
        self.model = self.policy._model
        assert self.model.pi05 and self.model.action_horizon == 10
        self.first_parity = None
        self.extractor = None

    def prepare(self, obs, instruction):
        import jax.numpy as jnp
        from openpi.models.model import Observation
        from preparations.idea1_v3_1.models import raw_observation
        data = self.policy._input_transform(raw_observation(obs, instruction))
        return Observation.from_dict(self.jax.tree.map(lambda x:jnp.asarray(x)[None], data))

    def decode(self, observation, raw):
        result = self.policy._output_transform(dict(
            state=np.asarray(observation.state[0]), actions=np.asarray(raw[0])))
        return np.asarray(result['actions'], np.float32)

    def predict(self, obs, instruction, seed, noise=None):
        import jax.numpy as jnp
        from preparations.idea1_v3_1.models import raw_observation
        from preparations.idea1_v2_1.runtime_client import input_digest, value_digest
        noise = (np.random.default_rng(seed).standard_normal((10,32), dtype=np.float32)
                 if noise is None else np.asarray(noise, np.float32))
        observation = self.prepare(obs, instruction)
        begin = time.perf_counter()
        raw = self.policy._sample_actions(self.jax.random.key(0), observation,
                                         noise=jnp.asarray(noise)[None], num_steps=10)
        raw = np.asarray(raw)
        actions = self.decode(observation, raw)
        if self.first_parity is None:
            reference = np.asarray(self.policy.infer(raw_observation(obs, instruction), noise=noise)['actions'],np.float32)
            self.first_parity = dict(passed=bool(np.array_equal(reference, actions)),
                max_abs=float(np.abs(np.asarray(reference)-actions).max()),
                scope='Same real observation/noise, direct native sampler vs official Policy.infer; both decoded to float32 controller commands')
        if not self.first_parity['passed']:
            raise ValueError(f'Official policy parity failed: {self.first_parity}')
        return dict(actions=actions, raw_actions=raw[0], noise=noise,
                    input_hash=input_digest(obs,instruction), noise_hash=value_digest(noise),
                    policy_ms=(time.perf_counter()-begin)*1000, parity=self.first_parity)

    def extract(self, request_path, output):
        if self.extractor is None:
            from .features import FeatureExtractor
            self.extractor = FeatureExtractor(self)
        return self.extractor.extract(Path(request_path), Path(output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--ready', required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    policy = NativePolicy()
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', args.port))
    server.listen(8)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.ready, dict(ready=True, protocol=PROTOCOL, source_sha256=source_hash,
                                load_s=time.perf_counter()-started, **process_identity()))
    print(json.dumps(dict(event='ready', port=args.port, **process_identity())), flush=True)
    while True:
        connection, _ = server.accept()
        with connection:
            try:
                size = struct.unpack('!I', receive(connection,4))[0]
                request = pickle.loads(receive(connection,size))
                op = request.pop('op')
                if op == 'health':
                    result = dict(protocol=PROTOCOL, parity=policy.first_parity, **process_identity())
                elif op == 'predict':
                    result = policy.predict(**request)
                elif op == 'extract':
                    result = policy.extract(**request)
                elif op == 'reload_features':
                    import importlib
                    from . import features
                    importlib.reload(features)
                    policy.extractor=None
                    result=dict(reloaded=True)
                elif op == 'stop':
                    result = dict(stopping=True)
                else:
                    raise ValueError(op)
                result = dict(ok=True, **result)
            except Exception:
                result = dict(ok=False, error=traceback.format_exc())
                print(result['error'], flush=True)
            payload = pickle.dumps(result, protocol=5)
            connection.sendall(struct.pack('!I',len(payload)) + payload)
        if result.get('stopping'):
            break


if __name__ == '__main__':
    main()
