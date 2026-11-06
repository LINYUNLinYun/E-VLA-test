"""Native FLASH service plus an explicit pure-draft dispatch, no shared edits."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CHECKPOINTS, REPO, PROTOCOL, noise_seed
sys.path[:0] = [str(REPO), str(REPO / 'src'), str(REPO / 'packages/openpi-client/src')]
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

def finite_json(value):
    import numpy as np
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist())
    if isinstance(value, np.generic):
        return finite_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('flash', 'draft', 'full'), required=True)
    p.add_argument('--source-suite', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--port', type=int, required=True)
    a = p.parse_args()
    import numpy as np
    import torch
    from scripts.spec import spec_serve_policy as serve
    from scripts.spec.triton import triton_pi0_runtime as runtime
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    request_log = (output / 'requests.jsonl').open('a', buffering=1)
    call_counts = {'full': 0, 'draft': 0, 'verify': 0}
    for method_name, label in (('run_full_with_timing', 'full'),
                               ('run_draft_with_timing', 'draft'),
                               ('run_verify_semantics_with_timing', 'verify')):
        original = getattr(runtime.SpecTritonRuntimeSession, method_name)
        def counted(self, *args, _original=original, _label=label, **kwargs):
            call_counts[_label] += 1
            if a.mode == 'draft' and _label != 'draft':
                raise RuntimeError('Pure draft entered forbidden ' + _label)
            return _original(self, *args, **kwargs)
        setattr(runtime.SpecTritonRuntimeSession, method_name, counted)

    original_sample = runtime.SpecTritonPolicyRuntime.sample_actions_with_timing
    def sample(self, **kwargs):
        initial = self._full_cache_snapshot is None
        pending = bool(self._pending_full_fallback)
        periodic = (not initial and not pending and self._periodic_full_every_n_draft_rounds > 0
                    and self._draft_rounds_since_full >= self._periodic_full_every_n_draft_rounds)
        if a.mode != 'draft':
            actions, timing = original_sample(self, **kwargs)
            full = bool(timing.get('is_full_pipeline_round', 0))
            timing.update(ease_initial_full=int(full and initial),
                          ease_rejection_full=int(full and pending),
                          ease_periodic_full=int(full and periodic),
                          ease_forced_full=int(full and a.mode == 'full'))
            return actions, timing
        batch_size = 1 if kwargs['state'].ndim == 1 else int(kwargs['state'].shape[0])
        self._init_runtime_state(batch_size)
        self._advance_runtime_state(kwargs.get('executed_steps'))
        session = self._runtime_pool.start_session(kwargs['prompt'])
        prepared = self._prepare_session_observation(
            session=session, images=kwargs['images'], state=kwargs['state'],
            tokenized_prompt=kwargs.get('tokenized_prompt'), tokenized_prompt_mask=kwargs.get('tokenized_prompt_mask'))
        actions, costs = session.run_draft_with_timing(prepared=prepared)
        action_batch = self._as_action_batch(actions)
        self._action_chunk_cache = action_batch.detach().clone()
        self._action_cache_ptr = 0
        encoder, draft = float(costs['encoder_ms']), float(costs['draft_ms'])
        timing = dict(encoder_ms=encoder, vision_ms=encoder, draft_ms=draft,
                      vlm_prefill_ms=0., action_verify_ms=0., action_denoise_ms=0.,
                      full_fallback_ms=0., used_full_fallback=0., scheduled_full_fallback=0.,
                      is_full_pipeline_round=0., include_in_draft_accept_metrics=0.,
                      accepted_prefix_len=12., accepted_prefix_len_mean=12.,
                      total_ms=encoder + draft, route_type='draft', did_prefill=0.,
                      gripper_verify_enabled=0., verify_mode_random=0.,
                      ease_initial_full=0, ease_rejection_full=0, ease_periodic_full=0,
                      ease_pure_draft=True, ease_actual_vlm_blocks=0, ease_actual_ae_layers=0)
        return self._maybe_unbatch_actions(action_batch), timing
    runtime.SpecTritonPolicyRuntime.sample_actions_with_timing = sample

    if a.mode == 'draft':
        # Avoid unused Full graph warmups; these are constructor calls, not actions.
        original_module_loader = runtime._spec_runtime_module
        def load_module(filename, name):
            module = original_module_loader(filename, name)
            if filename == 'pi0_infer.py':
                module.Pi0Inference.record_infer_graph = lambda self: None
            return module
        runtime._spec_runtime_module = load_module
        original_capture = runtime._StagedPi0Runtime._capture_stage_graph
        def capture(self, fn):
            if fn.__name__ != '_record_encoder_stage':
                return None
            return original_capture(self, fn)
        runtime._StagedPi0Runtime._capture_stage_graph = capture

    original_format = serve.TritonServerPolicy._format_outputs
    def format_outputs(self, *, transformed, actions):
        outputs = original_format(self, transformed=transformed, actions=actions)
        outputs['ease_raw_prefix'] = actions.detach().float().cpu().numpy().reshape(-1, 32)[:12, :7]
        return outputs
    serve.TritonServerPolicy._format_outputs = format_outputs
    original_infer = serve.TritonServerPolicy.infer
    def infer(self, obs, *, noise=None):
        obs = dict(obs)
        suite = str(obs.pop('__ease_suite__'))
        task = int(obs['__trace_task_id__'])
        init = int(obs['__trace_episode_idx__'])
        replan = int(obs['__trace_infer_id__'])
        seed = noise_seed(suite, task, init, replan)
        actual_noise = np.random.default_rng(seed).standard_normal((50, 32)).astype(np.float32)
        digest = hashlib.sha256()
        for key in ('observation/image', 'observation/wrist_image', 'observation/state'):
            value = np.ascontiguousarray(obs[key])
            digest.update(key.encode()); digest.update(str(value.shape).encode()); digest.update(value.tobytes())
        digest.update(str(obs['prompt']).encode())
        counts_before = dict(call_counts)
        started = time.perf_counter()
        out = original_infer(self, obs, noise=actual_noise)
        elapsed = (time.perf_counter() - started) * 1000
        counts = {key: call_counts[key] - counts_before[key] for key in call_counts}
        if a.mode == 'draft' and counts != dict(full=0, draft=1, verify=0):
            raise RuntimeError('Pure draft route counter violation: %r' % counts)
        timing = out['policy_timing']
        timing.update(ease_mode=a.mode, ease_source_suite=a.source_suite,
                      noise_seed_mode='case_replan_explicit', noise_seed_rule=PROTOCOL['noise_rule'],
                      noise_sha256=hashlib.sha256(actual_noise.tobytes()).hexdigest(),
                      ease_noise_seed=str(seed), ease_noise_used=a.mode != 'draft',
                      ease_input_sha256=digest.hexdigest(), ease_route_calls=counts)
        row = dict(suite=suite, task_id=task, init_id=init, replan_id=replan,
                   source_suite=a.source_suite, mode=a.mode, prompt=obs['prompt'],
                   state8=np.asarray(obs['observation/state']).tolist(),
                   input_sha256=digest.hexdigest(), noise_seed=str(seed),
                   noise_sha256=timing['noise_sha256'], noise_used=a.mode != 'draft',
                   raw_prefix=out.pop('ease_raw_prefix').tolist(),
                   decoded_prefix=np.asarray(out['actions'])[:12].tolist(),
                   route_calls=counts, policy_wall_ms=elapsed, timing=timing)
        request_log.write(json.dumps(finite_json(row), allow_nan=False) + '\n')
        return out
    serve.TritonServerPolicy.infer = infer
    logging.basicConfig(level=logging.WARNING, force=True)
    serve.main(serve.Args(host='127.0.0.1', port=a.port, config='pi0_libero',
                         backend='triton', base_triton_path=str(CHECKPOINTS / 'base'),
                         draft_triton_path=a.artifacts, tokenizer_source='local',
                         pytorch_device='cuda:0', max_exec_steps=12, tau_radius=.15,
                         periodic_full_every_n_draft_rounds=2,
                         force_full_each_round=a.mode == 'full'))

if __name__ == '__main__':
    main()
