"""V3 real Triton model adapter and explicit, execution-acknowledged routing."""
from pathlib import Path
import hashlib
import os
import pickle
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V14 = (3, 4, 9, 11)
MASKS = {'Full': (), 'transfer16': (3, 9), 'transfer15': (3, 9, 11),
         'V14_A03': V14, 'V13_Custom': (3, 4, 7, 12, 15)}


def verification_seed(seed):
    return int.from_bytes(hashlib.sha256(f'idea1-v3/verify/{seed}'.encode()).digest()[:4], 'little')


def acceptance(draft, hats, tau, last_gripper=None):
    """FLASH sign-switch semantics, evaluated only over the future Rmax5 window."""
    d = np.asarray(draft)[:5]
    h = np.asarray(hats)[:, :len(d)]
    delta = h[..., :6] - d[None, :, :6]
    dist = np.linalg.norm(delta, axis=-1) / np.sqrt(6.)
    length = int(np.cumprod(np.all(dist <= tau, axis=0)).sum())
    verify_switch = post_switch = False
    if last_gripper is not None:
        previous = np.concatenate([np.full((len(h), 1), last_gripper), h[:, :-1, 6]], axis=1)
        verify_switch = bool(np.any((previous < 0) != (h[..., 6] < 0)))
        prev = np.r_[last_gripper, d[:-1, 6]]
        switches = np.flatnonzero(((prev < 0) != (d[:, 6] < 0))[:length])
        if len(switches):
            post_switch = True
            length = int(switches[0])
    if verify_switch:
        length = 0
    return length, dict(distance=dist.tolist(), signed_error=delta.tolist(),
                        verify_gripper=verify_switch, post_verify_gripper=post_switch)


class TritonBackend:
    """Reuse released transforms/checkpoint; never enter identity/image-mean fallbacks."""
    def __init__(self):
        import sys
        sys.path.insert(0, str(ROOT / 'policies/realtime-vla-flash'))
        from preparations.idea1_v2_2.pi0_adapter import Pi0Adapter
        self.adapter = Pi0Adapter()
        os.environ['SPEC_TRITON_VERIFY_EXACT_INPUT'] = '1'
        self.torch = self.adapter.torch
        self.specs = {}
        self.startup_events = []
        self.protocol = dict(self.adapter.protocol, experiment='idea1_v3', R=5,
                             startup_accounting='constructor full graph capture disabled; explicit calls only')

    def prepare(self, obs, instruction, skip=(), encode=True):
        from scripts.spec.triton import pi0_infer as k
        from preparations.idea1_v2_1.runtime_models import native_observation
        a = self.adapter
        images, state = native_observation(obs)
        observation = {'observation/image': np.asarray(images[0]), 'observation/wrist_image': np.asarray(images[1]),
                       'observation/state': state, 'prompt': instruction}
        # Constructor normally executes four Full AE passes. No hidden Full bootstraps,
        # especially for E4. These eager paths do not use its monolithic graph.
        original = k.Pi0Inference.record_infer_graph
        k.Pi0Inference.record_infer_graph = lambda runtime: None
        tick = time.perf_counter()
        try:
            previous_runtime = a.backend.runtime
            a.backend.prepare(observation, dict(suite=a.prompt_suites[instruction], task_description=instruction),
                              np.zeros((50, 32), np.float32))
        finally:
            k.Pi0Inference.record_infer_graph = original
        r = a.backend.runtime
        if r is not previous_runtime:
            self.startup_events.append(dict(kind='prompt_runtime_construction',seconds=time.perf_counter()-tick,
                                           prompt=instruction,full_AE_calls=0))
        r._infer_module = k
        w, b = r.weights, r.buffers
        b['observation_images_normalized'].copy_(a.backend.inputs[0])
        b['observation_state_normalized'].copy_(a.backend.inputs[1])
        b['encoder_x'][512:].copy_(w['language_embeds'])
        timing = {'input_processing_ms': (time.perf_counter()-tick)*1000}
        _, timing['vision_ms'] = a.measure(lambda: k.vision_encoder(w, b, 2))
        seq = b['encoder_x'].shape[0]
        if encode:
            _, timing['vlm_ms'] = a.measure(lambda: k.transformer_encoder(w, b, seq,
                vlm_layer_indices=tuple(i for i in range(18) if i+1 not in skip)))
        c = dict(runtime=r, transformed_state=np.asarray(a.backend.transformed['state']).copy(),
                 state=a.backend.inputs[1].clone(), seq=seq, startup_events=[])
        if encode:
            c.update(K=b['encoder_K'][:, :seq].clone(), V=b['encoder_V'][:, :seq].clone())
        return c, timing

    def spec(self, c, suite):
        from scripts.spec.triton.pi0_spec_infer import Pi0SpecInference
        from scripts.experiment1.run_intermediate_action_probe import SUITE_ARTIFACT
        artifact = SUITE_ARTIFACT[suite]
        key = (suite, c['seq'])
        if key not in self.specs:
            tick = time.perf_counter()
            path = ROOT / 'shared/checkpoints/realtime-vla-flash/converted' / artifact / 'draft_triton.pkl'
            with path.open('rb') as stream:
                weights = pickle.load(stream)
            self.specs[key] = Pi0SpecInference(checkpoint=c['runtime'].weights,
                                              draft_checkpoint=weights, num_views=2, chunk_size=50)
            self.startup_events.append(dict(kind='learned_draft_weight_load',seconds=time.perf_counter()-tick,
                                           suite=suite,path=str(path),meta=self.specs[key].meta))
        spec = self.specs[key]
        # Prompt length alone is not prompt identity. Graph input copies below use
        # this fresh language tensor even when another task has identical length.
        spec.weights['language_embeds'].copy_(c['runtime'].weights['language_embeds'])
        return spec

    def sample(self, c, seed, steps):
        result = self.adapter.sample(c, seed, dict(kind='native', m=None, num_steps=steps), False)
        actions, raw, _, audit, ms, decode = result
        return actions, raw, dict(ae_ms=ms, decode_ms=decode, nodes=audit['nodes'])

    def learned(self, c, suite):
        from scripts.spec.triton.pi0_spec_infer import PrefixContext
        s = self.spec(c, suite)
        vision = s._project_vision_prefix(full_runtime=c['runtime'])
        language = s.weights['language_embeds'].float().unsqueeze(0)
        prefix = self.torch.cat([vision, language], dim=1)
        masks = self.torch.ones(prefix.shape[:2], device=prefix.device, dtype=self.torch.bool)
        context = PrefixContext(prefix_embs=prefix, prefix_pad_masks=masks, prefix_att_masks=~masks)
        raw, ms = self.adapter.measure(lambda: s._run_draft_block_maybe_graph(
            prefix=context, observation_state_normalized=c['state']))
        raw = raw[0].clone()
        actions = self.decode(c, raw)
        return actions, raw, dict(draft_head_ms=ms)

    def decode(self, c, raw):
        return np.asarray(self.adapter.backend.output_transform(dict(state=c['transformed_state'].copy(),
            actions=raw.float().cpu().numpy().copy()))['actions'], np.float32)

    def verify(self, full, current, raw, seed, suite):
        from scripts.spec.triton.pi0_spec_infer import FullCacheSnapshot
        s = self.spec(current, suite)
        if not s._verify_exact_input_supported(weights=current['runtime'].weights):
            raise RuntimeError('Verifier requires unfused time/input weights; refusing approximate-time fallback')
        snapshot = FullCacheSnapshot(encoder_seq_len=full['seq'], encoder_k=full['K'], encoder_v=full['V'])
        noise = self.torch.as_tensor(np.random.default_rng(verification_seed(seed)).standard_normal((50,32),dtype=np.float32),device='cuda')
        hats, ms = self.adapter.measure(lambda: s.run_verify(snapshot, self.adapter.backend.inputs[0],
            current['state'], noise, raw, (.10,.05), full_runtime=current['runtime']))
        return hats[0].float().cpu().numpy(), dict(verify_ae_ms=ms)


class Policy:
    def __init__(self, config, backend=None):
        if config['experiment'] not in ('E1','E4'):
            raise ValueError('Unsupported experiment; E2/E3 require their own implementation')
        if config['experiment']=='E1' and (config.get('draft_steps',2) not in (2,10) or config.get('period',2) not in (2,4)
                or config.get('tau',.15) not in (.1,.15,.25)):
            raise ValueError('E1 parameters outside frozen v3 grid')
        self.config = config
        self.backend = backend or TritonBackend()
        self.protocol = dict(self.backend.protocol, config=config)
        self.reset()

    def reset(self):
        self.full = None
        self.draft_rounds = 0
        self.round = self.step = 0
        self.last_gripper = None
        self.pending_gripper = False
        self.pending = None
        self.counts = dict(full=0, verify=0, draft=0)
        return {'ok': True}

    def acknowledge(self, executed):
        if self.pending is None:
            raise RuntimeError('No proposed actions to acknowledge')
        executed = np.asarray(executed, np.float32)
        if not 0 < len(executed) <= self.pending['execute_length']:
            raise ValueError('Executed prefix outside proposal')
        if not np.array_equal(executed, self.pending['actions'][:len(executed)]):
            raise ValueError('Actual command differs from proposed decoded actions')
        self.last_gripper = float(self.pending['raw'][len(executed)-1,6])
        self.step += len(executed)
        self.pending = None
        return {'ok': True}

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Previous execution must be acknowledged before replanning')
        tick = time.perf_counter()
        startup_offset = len(getattr(self.backend,'startup_events',[]))
        cfg = self.config
        timing = {}
        route = {}
        arrays = {}
        age = None if self.full is None else dict(rounds=self.round-self.full['round'],
            env_steps=self.step-self.full['step'], wall_s=time.perf_counter()-self.full['time'])
        def full(reason):
            c, t = self.backend.prepare(obs, instruction, ())
            actions, raw, costs = self.backend.sample(c, seed, 10)
            timing.update({f'full_{k}':v for k,v in dict(t, **costs).items()})
            self.full = dict(c, round=self.round, step=self.step, time=time.perf_counter())
            self.draft_rounds = 0
            self.pending_gripper = False
            self.counts['full'] += 1
            route.update(path='full', full_reason=reason)
            return actions, raw, 5
        if cfg['experiment'] == 'E4':
            c, t = self.backend.prepare(obs, instruction, (), encode=False)
            actions, raw, costs = self.backend.learned(c, suite)
            timing.update(t, **costs)
            self.counts['draft'] += 1
            route.update(path='learned_draft', full_reason=None)
            length = 5
        elif self.full is None:
            actions, raw, length = full('startup')
        elif self.pending_gripper:
            actions, raw, length = full('gripper_window')
        elif self.draft_rounds >= cfg.get('period',2):
            actions, raw, length = full('periodic')
        else:
            c, t = self.backend.prepare(obs, instruction, V14)
            actions, raw, costs = self.backend.sample(c, seed, cfg.get('draft_steps',2))
            timing.update({f'draft_{k}':v for k,v in dict(t, **costs).items()})
            self.counts['draft'] += 1
            hats, costs = self.backend.verify(self.full, c, raw, seed, suite)
            timing.update(costs)
            self.counts['verify'] += 1
            raw_np = raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
            if hats.shape!=(2,50,32) or not np.isfinite(hats).all(): raise ValueError('Invalid verifier reconstruction')
            length, details = acceptance(raw_np, hats, cfg.get('tau',.15), self.last_gripper)
            arrays.update(draft_future=raw_np[:5], reconstructed_future=hats[:,:5])
            route.update(path='v14_verified', accepted_length=length, **details)
            self.draft_rounds += 1
            switch = details['verify_gripper'] or details['post_verify_gripper']
            if length == 0:
                actions, raw, length = full('gripper' if switch else 'zero_accept')
            elif switch:
                self.pending_gripper = True
        raw_np = raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
        actions = np.asarray(actions,np.float32)
        if actions.shape != (50,7) or not np.isfinite(actions).all():
            raise ValueError('Invalid model action chunk')
        result = dict(actions=actions, execute_length=min(length,5), raw=raw_np,
            route=route, timing=timing, counts=dict(self.counts), full_cache_age=age,
            seed=seed, verification_seed=verification_seed(seed), round=self.round, step=self.step,
            arrays=arrays)
        result['startup_events'] = getattr(self.backend,'startup_events',[])[startup_offset:]
        result['timing']['policy_wall_ms'] = (time.perf_counter()-tick)*1000
        self.round += 1
        self.pending = result
        return result
