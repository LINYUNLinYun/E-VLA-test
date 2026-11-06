"""Thin py3.8 adapter preserving full experiment metadata and playable H264 videos."""
from __future__ import annotations

import atexit
import dataclasses
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/spec'))
from scripts.experiment10.parallel_verify import parallel_infer as _parallel_infer
import spec_client_libero as client


def main():
    import tyro
    original_record = client._make_infer_record
    metadata_key = os.environ.get('EXP11_METADATA_KEY', 'experiment10')
    def record(*args, **kwargs):
        result = original_record(*args, **kwargs)
        result[metadata_key] = kwargs['policy_timing'].copy()
        return result
    client._make_infer_record = record
    save_root = os.environ.get('EXP11_SAVE_FULL_INPUTS')
    input_writer = ThreadPoolExecutor(max_workers=1) if save_root else None
    if input_writer is not None:
        atexit.register(input_writer.shutdown, wait=True)

    def save_full_input(element, result):
        if not save_root:
            return
        timing = result.get('policy_timing', {})
        if timing.get('candidate_skip_layers_1based') != [] or timing.get('denoise_steps') != 3:
            return
        task = int(element['__trace_task_id__'])
        episode = int(element['__trace_episode_idx__'])
        infer_id = int(element['__trace_infer_id__'])
        suite = str(timing.get('suite') or os.environ.get('EXP11_SUITE', 'unknown'))
        image = np.ascontiguousarray(element['observation/image'], dtype=np.uint8).copy()
        wrist = np.ascontiguousarray(element['observation/wrist_image'], dtype=np.uint8).copy()
        state = np.ascontiguousarray(element['observation/state'], dtype=np.float32).copy()
        prompt = str(element['prompt'])
        noise = np.random.default_rng(np.random.SeedSequence([12345, episode, infer_id])).standard_normal((50, 32)).astype(np.float32)
        digest = hashlib.sha256()
        for array in (image, wrist, state, noise):
            digest.update(array.tobytes())
        digest.update(prompt.encode('utf-8'))
        digest.update(suite.encode('utf-8'))
        directory = Path(save_root) / suite / ('task%02d' % task) / ('init%02d' % episode)
        destination = directory / ('replan%04d.npz' % infer_id)
        metadata_json = json.dumps({'suite': suite, 'task_id': task,
                                    'episode_idx': episode, 'infer_id': infer_id,
                                    'executed_steps_since_replan': int(element['__executed_steps__']),
                                    'input_sha256': digest.hexdigest(), 'checkpoint_sha256': timing.get('checkpoint_sha256'),
                                    'condition': timing.get('condition'), 'skip_layers': timing.get('candidate_skip_layers_1based'),
                                    'denoise_steps': timing.get('denoise_steps'), 'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest()},
                                    ensure_ascii=False)
        timing['full_input_path'] = str(destination)
        timing['full_input_sha256'] = digest.hexdigest()
        result['policy_timing'] = timing

        def persist():
            directory.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                with np.load(destination, allow_pickle=False) as saved:
                    old = json.loads(str(saved['metadata_json'].item()))
                if old.get('input_sha256') != digest.hexdigest():
                    raise RuntimeError('Existing full-observation artifact differs on retry: %s' % destination)
                return
            temp = destination.with_suffix('.npz.tmp')
            with temp.open('wb') as stream:
                np.savez_compressed(stream, agent_image=image, wrist_image=wrist, robot_state=state,
                                    prompt=np.asarray(prompt), noise=noise,
                                    metadata_json=np.asarray(metadata_json))
                stream.flush()
                os.fsync(stream.fileno())
            temp.replace(destination)

        input_writer.submit(persist)
    original_video = client.imageio.mimwrite
    def video(path, frames, **kwargs):
        kwargs.update({'codec': 'libx264', 'pixelformat': 'yuv420p', 'macro_block_size': 2,
                       'ffmpeg_params': ['-crf', '20', '-movflags', '+faststart']})
        return original_video(path, frames, **kwargs)
    client.imageio.mimwrite = video
    original_client = client._websocket_client_policy.WebsocketClientPolicy
    class CheckedClient(original_client):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            metadata = self.get_server_metadata()
            if metadata.get('protocol', {}).get('schema') != 1 or not metadata.get('checkpoint_sha256'):
                raise RuntimeError('Not an Experiment10 same-checkpoint server')
            parallel_json = os.environ.get('EXP10_PARALLEL_VERIFY_JSON')
            self._parallel_condition = json.loads(parallel_json) if parallel_json else None
            self._reference_client = None
            self._parallel_executor = None
            if self._parallel_condition is not None:
                reference_port = int(os.environ['EXP10_PARALLEL_REFERENCE_PORT'])
                host = kwargs.get('host', args[0] if args else '127.0.0.1')
                self._reference_client = original_client(host=host, port=reference_port)
                reference_metadata = self._reference_client.get_server_metadata()
                if (reference_metadata.get('protocol', {}).get('schema') != 1 or
                        reference_metadata.get('checkpoint_sha256') != metadata.get('checkpoint_sha256')):
                    raise RuntimeError('Parallel verifier must use the identical Experiment10 checkpoint')
                self._parallel_executor = ThreadPoolExecutor(max_workers=2)
        def infer(self, *args, **kwargs):
            try:
                if self._parallel_condition is not None:
                    result = _parallel_infer(
                        lambda: original_client.infer(self, *args, **kwargs),
                        lambda: self._reference_client.infer(*args, **kwargs),
                        self._parallel_condition, self._parallel_executor)
                else:
                    result = super().infer(*args, **kwargs)
                if save_root and args and isinstance(args[0], dict):
                    save_full_input(args[0], result)
                return result
            except Exception as exc:
                # The legacy rollout catches Exception and continues episodes.
                # A transport/server failure invalidates the batch, not task success.
                raise SystemExit('Fatal inference/transport error; batch invalidated: '+str(exc)) from exc
    client._websocket_client_policy.WebsocketClientPolicy = CheckedClient
    args = tyro.cli(client.Args)
    expected_inits = json.loads(os.environ.get('EXP11_INIT_INDICES_JSON', '[0,1,2,3,4]'))
    if (args.episode_index_start != min(expected_inits) or args.episode_index_end != max(expected_inits) + 1 or
            args.num_trials_per_task != len(expected_inits) or expected_inits != list(range(min(expected_inits), max(expected_inits)+1))
            or args.seed != 7 or args.num_steps_wait != 10 or not args.record_semantic_telemetry):
        raise ValueError('Rollout init range, seed, warmup or telemetry differs from the frozen protocol')
    client.eval_libero(args)


if __name__ == '__main__':
    main()
