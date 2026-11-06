"""Same-observation parallel D3/Full18 verification helpers."""
from __future__ import annotations

import time

from scripts.experiment10.command_predictor import verified_prefix


def combine_parallel_outputs(candidate, reference, condition, wall_ms):
    """Select a verified D3 prefix or Full18 fallback and preserve both timings."""
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        raise TypeError('Both parallel policy responses must be dictionaries')
    candidate_timing = candidate.get('policy_timing', {})
    reference_timing = reference.get('policy_timing', {})
    if not isinstance(candidate_timing, dict) or not isinstance(reference_timing, dict):
        raise TypeError('Both policy responses must contain timing metadata')
    for key in ('checkpoint_sha256', 'noise_sha256'):
        left, right = candidate_timing.get(key), reference_timing.get(key)
        if left is None or right is None or left != right:
            raise RuntimeError('Parallel candidate/reference %s mismatch' % key)

    accepted, reason = verified_prefix(
        candidate['actions'], reference['actions'], condition['horizon'],
        condition['translation_threshold_m'], condition['rotation_threshold_rad'])
    used_full = accepted == 0
    selected = reference if used_full else candidate
    out = dict(selected)
    metadata = dict(selected.get('policy_timing', {}))
    candidate_skip = list(condition['skip'])
    reference_skip = []
    candidate_steps = int(condition['steps'])
    reference_steps = 10
    metadata.update({
        'condition': condition['name'],
        'mode': 'parallel_verify',
        'requested_horizon': int(condition['horizon']),
        'candidate_skip_layers_1based': candidate_skip,
        'candidate_denoise_steps': candidate_steps,
        'reference_skip_layers_1based': reference_skip,
        'reference_denoise_steps': reference_steps,
        'candidate_noise_sha256': candidate_timing['noise_sha256'],
        'reference_noise_sha256': reference_timing['noise_sha256'],
        'candidate_checkpoint_sha256': candidate_timing['checkpoint_sha256'],
        'reference_checkpoint_sha256': reference_timing['checkpoint_sha256'],
        'candidate_accepted_prefix_len': int(accepted),
        'verification_reason': reason,
        'verification_translation_threshold_m': float(condition['translation_threshold_m']),
        'verification_rotation_threshold_rad': float(condition['rotation_threshold_rad']),
        'used_full_fallback': int(used_full),
        'is_full_pipeline_round': int(used_full),
        'include_in_draft_accept_metrics': int(not used_full),
        'candidate_cuda_policy_ms': float(candidate_timing.get('cuda_policy_ms', 0.0)),
        'reference_cuda_policy_ms': float(reference_timing.get('cuda_policy_ms', 0.0)),
        'candidate_server_sample_actions_ms': float(candidate_timing.get('sample_actions_ms', 0.0)),
        'reference_server_sample_actions_ms': float(reference_timing.get('sample_actions_ms', 0.0)),
        'cuda_policy_ms': float(candidate_timing.get('cuda_policy_ms', 0.0)) +
                          float(reference_timing.get('cuda_policy_ms', 0.0)),
        'parallel_inference_wall_ms': float(wall_ms),
        'sample_actions_ms': float(wall_ms),
        'total_ms': float(wall_ms),
        'timing_is_dedicated_benchmark': False,
    })
    out['policy_timing'] = metadata
    out['accepted_prefix_len'] = 12 if used_full else int(accepted)
    candidate_server = candidate.get('server_timing', {})
    reference_server = reference.get('server_timing', {})
    out['server_timing'] = {
        'infer_ms': float(wall_ms),
        'policy_time_ms': float(wall_ms),
        'serve_time_ms': float(wall_ms),
        'ws_unpack_ms': 0.0,
        'ws_pack_ms': 0.0,
        'candidate_server_timing': candidate_server,
        'reference_server_timing': reference_server,
    }
    return out


def parallel_infer(candidate_call, reference_call, condition, executor):
    """Send the identical observation to distinct policy connections concurrently."""
    started = time.perf_counter()
    candidate_future = executor.submit(candidate_call)
    reference_future = executor.submit(reference_call)
    candidate = candidate_future.result()
    reference = reference_future.result()
    return combine_parallel_outputs(candidate, reference, condition,
                                    (time.perf_counter() - started) * 1000.0)
