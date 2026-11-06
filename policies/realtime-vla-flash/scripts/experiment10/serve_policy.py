"""Same-weight arbitrary-mask policy server for Experiment 10 closed-loop screens.

This first adapter covers static and one-prefix-then-Full18 recovery. It does not
claim to implement asynchronous speculative control or a trained independent draft.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.experiment10.backend import Backend, validate_mask
from scripts.experiment10.command_predictor import CommandPredictor, accepted_prefix, verified_prefix
from scripts.experiment10.common import RESULTS, SUITES, PROTOCOL


def memory_diagnostics():
    import torch
    import subprocess
    import os
    result = {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'pid': os.getpid()}
    try:
        result.update(allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved(),
                      free_total=torch.cuda.mem_get_info())
    except Exception as exc:
        result['cuda_diagnostic_error'] = str(exc)
    try:
        response = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                   '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        result['compute_process_snapshot'] = response.stdout
    except Exception as exc:
        result['process_diagnostic_error'] = str(exc)
    return result


class Policy:
    def __init__(self, backend, suite, skip, steps, mode, horizon, condition,
                 translation_threshold_m=None, rotation_threshold_rad=None,
                 predictor_process_variance=None, predictor_measurement_variance=None,
                 max_observation_age_ms=100.0, minimum_executed_history=4):
        self.backend, self.suite = backend, suite
        self.skip, self.steps = validate_mask(skip), steps
        self.mode, self.horizon, self.condition = mode, horizon, condition
        self.translation_threshold_m = translation_threshold_m
        self.rotation_threshold_rad = rotation_threshold_rad
        self.max_observation_age_ms = float(max_observation_age_ms)
        self.minimum_executed_history = int(minimum_executed_history)
        self.predictor = None
        if mode == 'kalman_gate':
            if predictor_process_variance is None or predictor_measurement_variance is None:
                raise ValueError('Kalman-gate mode requires frozen Q/R calibration')
            self.predictor = CommandPredictor(predictor_process_variance,
                                              predictor_measurement_variance,
                                              minimum_history=self.minimum_executed_history)
        self.reset()

    def reset(self):
        self.round = 0
        if self.predictor is not None:
            self.predictor.reset()

    def infer(self, observation):
        if observation.get('__reset_policy_state__', False):
            self.reset()
        episode = int(observation['__trace_episode_idx__'])
        infer_id = int(observation['__trace_infer_id__'])
        prompt = str(observation['prompt'])
        noise = np.random.default_rng(np.random.SeedSequence([12345, episode, infer_id])).standard_normal((50, 32)).astype(np.float32)
        meta = {'suite': self.suite, 'task_description': prompt}
        inputs = {k: observation[k] for k in ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')}
        started = time.monotonic()
        try:
            self.backend.prepare(inputs, meta, noise)
        except Exception:
            import logging
            logging.exception('Preparation failed: %s', memory_diagnostics())
            raise
        recover = self.mode == 'prefix_once' and self.round > 0
        verification_reason = None
        candidate_accepted = None
        candidate_cuda_ms = reference_cuda_ms = 0.
        execution_gate_reason = None
        predictor_history_count = None
        observation_age_ms = None
        if self.mode == 'verify_then_execute':
            candidate_raw, candidate_command, candidate_timing = self.backend.infer(self.skip, self.steps)
            reference_raw, reference_command, reference_timing = self.backend.infer((), 10)
            candidate_cuda_ms = float(candidate_timing['cuda_policy_ms'])
            reference_cuda_ms = float(reference_timing['cuda_policy_ms'])
            candidate_accepted, verification_reason = verified_prefix(
                candidate_command, reference_command, self.horizon,
                self.translation_threshold_m, self.rotation_threshold_rad)
            used_full_fallback = candidate_accepted == 0
            if used_full_fallback:
                raw, command, skip, steps, accepted = reference_raw, reference_command, (), 10, 12
            else:
                raw, command, skip, steps, accepted = candidate_raw, candidate_command, self.skip, self.steps, candidate_accepted
            timing = {'cuda_policy_ms': candidate_cuda_ms + reference_cuda_ms}
        elif self.mode == 'kalman_gate':
            executed = np.asarray(observation.get('__executed_actions_since_last_infer__', []), dtype=float)
            if executed.size:
                if executed.ndim != 2 or executed.shape[1] != 7 or not np.isfinite(executed).all():
                    raise ValueError('Invalid actual-action history supplied by the client')
                self.predictor.observe_executed(executed)
            predictor_history_count = self.predictor.history_count
            candidate_command = None
            candidate_timing = {'cuda_policy_ms': 0.}
            if predictor_history_count >= self.minimum_executed_history:
                candidate_raw, candidate_command, candidate_timing = self.backend.infer(self.skip, self.steps)
                candidate_cuda_ms = float(candidate_timing['cuda_policy_ms'])
                observed_at = float(observation.get('__observation_time_monotonic_ms__', float('nan')))
                observation_age_ms = time.monotonic() * 1000.0 - observed_at
                candidate_accepted, execution_gate_reason = accepted_prefix(
                    self.predictor, candidate_command, self.translation_threshold_m,
                    self.rotation_threshold_rad, observation_age_ms,
                    self.max_observation_age_ms, self.horizon)
            else:
                candidate_accepted, execution_gate_reason = 0, 'insufficient_executed_history'
            used_full_fallback = candidate_accepted == 0
            if used_full_fallback:
                raw, command, reference_timing = self.backend.infer((), 10)
                reference_cuda_ms = float(reference_timing['cuda_policy_ms'])
                skip, steps, accepted = (), 10, 12
            else:
                raw, command, skip, steps, accepted = candidate_raw, candidate_command, self.skip, self.steps, candidate_accepted
            timing = {'cuda_policy_ms': candidate_cuda_ms + reference_cuda_ms}
        else:
            skip, steps = ((), 10) if recover else (self.skip, self.steps)
            raw, command, timing = self.backend.infer(skip, steps)
            accepted = 12 if recover else self.horizon
            used_full_fallback = recover
        self.round += 1
        total = (time.monotonic() - started) * 1000
        has_dynamic_draft_route = self.mode in ('verify_then_execute', 'kalman_gate', 'prefix_once')
        layer_pattern = ('vlm_full18' if not skip and steps == 10 else
                         'vlm_full18_d%d' % steps if not skip else
                         'exp10_skip_' + '_'.join(map(str, skip)))
        metadata = {'sample_actions_ms': total, 'total_ms': total, 'did_prefill': 1,
                    'is_full_pipeline_round': int(used_full_fallback) if has_dynamic_draft_route else 1,
                    'used_full_fallback': int(used_full_fallback),
                    'include_in_draft_accept_metrics': int(has_dynamic_draft_route and not used_full_fallback),
                    'vlm_layer_pattern': layer_pattern,
                    'vlm_layer_indices_1based': [l for l in range(1, 19) if l not in skip],
                    'vlm_full_block_indices_1based': [l for l in range(1, 19) if l not in skip],
                    'vlm_kv_shell_indices_1based': list(skip), 'action_layer_pattern': 'full18',
                    'action_layer_indices_1based': list(range(1, 19)), 'denoise_steps': steps,
                    'denoise_times': [1-i/steps for i in range(steps)], 'flow_dt': -1/steps,
                    'ablation_noise_seed': 12345, 'noise_seed_mode': 'episode_infer',
                    'noise_seed_rule': 'SeedSequence(global_noise_seed, episode_idx, replan_idx)',
                    'noise_sha256': hashlib.sha256(noise.tobytes()).hexdigest(),
                    'trace_episode_idx': episode, 'trace_infer_id': infer_id,
                    'checkpoint_sha256': self.backend.weight_hash, 'condition': self.condition,
                    'suite': self.suite,
                    'mode': self.mode, 'recovery_round': recover, 'requested_horizon': self.horizon,
                    'candidate_skip_layers_1based': list(self.skip), 'candidate_denoise_steps': self.steps,
                    'reference_skip_layers_1based': [], 'reference_denoise_steps': 10,
                    'candidate_accepted_prefix_len': candidate_accepted,
                    'verification_reason': verification_reason,
                    'execution_gate_reason': execution_gate_reason,
                    'predictor_history_count': predictor_history_count,
                    'observation_age_ms_at_decision': observation_age_ms,
                    'max_observation_age_ms': self.max_observation_age_ms if self.mode == 'kalman_gate' else None,
                    'predictor_minimum_executed_history': self.minimum_executed_history if self.mode == 'kalman_gate' else None,
                    'verification_translation_threshold_m': self.translation_threshold_m,
                    'verification_rotation_threshold_rad': self.rotation_threshold_rad,
                    'command_prediction_translation_threshold': self.translation_threshold_m if self.mode == 'kalman_gate' else None,
                    'command_prediction_rotation_threshold': self.rotation_threshold_rad if self.mode == 'kalman_gate' else None,
                    'predictor_process_variance': self.predictor.q.tolist() if self.predictor is not None else None,
                    'predictor_measurement_variance': self.predictor.r.tolist() if self.predictor is not None else None,
                    'candidate_cuda_policy_ms': candidate_cuda_ms,
                    'reference_cuda_policy_ms': reference_cuda_ms,
                    'cuda_policy_ms': timing['cuda_policy_ms'],
                    'timing_is_dedicated_benchmark': False}
        return {'actions': command, 'accepted_prefix_len': accepted, 'policy_timing': metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--skip', default='')
    parser.add_argument('--steps', type=int, choices=(1, 2, 3, 4, 5, 7, 10), default=10)
    parser.add_argument('--mode', choices=('static', 'prefix_once', 'verify_then_execute', 'kalman_gate'), default='static')
    parser.add_argument('--horizon', type=int, choices=(1, 2, 4, 5, 8, 12), default=12)
    parser.add_argument('--translation-threshold-m', type=float)
    parser.add_argument('--rotation-threshold-rad', type=float)
    parser.add_argument('--predictor-process-variance', default='')
    parser.add_argument('--predictor-measurement-variance', default='')
    parser.add_argument('--max-observation-age-ms', type=float, default=100.0)
    parser.add_argument('--minimum-executed-history', type=int, default=4)
    parser.add_argument('--condition', required=True)
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    import torch
    # Initialize the actual worker context before CPU checkpoint preparation.
    # Do not reserve unrelated memory or displace another GPU user's process.
    torch.cuda.init()
    skip = tuple(int(l) for l in args.skip.split(',') if l)
    validate_mask(skip)
    metadata = []
    observation_root = Path(os.environ.get('EXP11_SOURCE_RESULTS', str(RESULTS)))
    entries = json.loads((observation_root / 'observations' / args.suite / 'manifest.json').read_text())['observations']
    for task in range(10):
        entry = next(e for e in entries if e['task_id'] == task)
        with np.load(entry['path'], allow_pickle=False) as data:
            metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(metadata)
    entry = next(e for e in entries if e['task_id'] == 0)
    with np.load(entry['path'], allow_pickle=False) as data:
        warm_observation = {'observation/image': data['agent_image'], 'observation/wrist_image': data['wrist_image'],
                            'observation/state': data['robot_state'], 'prompt': str(data['prompt'].item())}
        warm_meta = json.loads(str(data['metadata_json'].item()))
    warm_noise = np.random.default_rng(12345).standard_normal((50, 32)).astype(np.float32)
    try:
        backend.prepare(warm_observation, warm_meta, warm_noise)
        backend.infer(skip, args.steps)  # Includes the ordinary/manual D10 parity gate.
        if args.mode == 'verify_then_execute':
            backend.infer((), 10)  # Warm/capture the full reference path before timing real requests.
    except Exception:
        import logging
        logging.exception('Startup parity preparation failed: %s', memory_diagnostics())
        raise
    print(json.dumps({'event': 'model_ready_after_parity', 'cuda_allocated': torch.cuda.memory_allocated(),
                      'cuda_reserved': torch.cuda.memory_reserved(), 'cuda_free_total': torch.cuda.mem_get_info()}), flush=True)
    predictor_q = tuple(float(v) for v in args.predictor_process_variance.split(',') if v)
    predictor_r = tuple(float(v) for v in args.predictor_measurement_variance.split(',') if v)
    if args.mode in ('verify_then_execute', 'kalman_gate') and (args.translation_threshold_m is None or
                                                                args.rotation_threshold_rad is None):
        raise ValueError('Verification/gating requires frozen translation and rotation thresholds')
    if args.mode == 'kalman_gate' and (len(predictor_q) != 6 or len(predictor_r) != 6 or
                                       args.max_observation_age_ms <= 0 or args.minimum_executed_history < 2):
        raise ValueError('Kalman-gate requires six positive Q/R values, an age limit and minimum history')
    policy = Policy(backend, args.suite, skip, args.steps, args.mode, args.horizon, args.condition,
                    args.translation_threshold_m, args.rotation_threshold_rad,
                    predictor_q or None, predictor_r or None, args.max_observation_age_ms,
                    args.minimum_executed_history)
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    WebsocketPolicyServer(policy, host='127.0.0.1', port=args.port,
                          metadata={'protocol': PROTOCOL, 'checkpoint_sha256': backend.weight_hash,
                                    'condition': args.condition, 'skip': skip, 'steps': args.steps,
                                    'mode': args.mode, 'horizon': args.horizon, 'model_ready_after_parity': True},
                          reset_policy_on_connect=True).serve_forever()


if __name__ == '__main__':
    main()
