import hashlib
import time
import numpy as np
from .common import MASK
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian


class Policy:
    def __init__(self, backend, sampler=sample_gaussian):
        self.b, self.sampler = backend, sampler
        self.config = {}
        self.reset()

    def reset(self):
        self.pending = None
        self.round = self.step = 0
        self.history = []
        self.counts = dict(full=0, draft=0, verify=0, residual_refresh=0)

    def configure(self, config):
        if config['D'] != 2 or tuple(config['skip']) != MASK or config['full_ae'] or config['verifier']:
            raise ValueError('Frozen V12 D2, no Full AE or verifier')
        if config['refresh_interval'] != 3 or config['compensation'] not in (None, 'extrapolate'):
            raise ValueError('Frozen three-round residual-only schedule')
        self.config = config
        self.reset()
        return {}

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Previous execution must be acknowledged')
        start = time.perf_counter()
        compensated = self.config['compensation'] == 'extrapolate'
        refresh = compensated and self.round % 3 == 0
        timing = {}
        if refresh:
            context, costs = self.b.prepare_residual(obs, instruction, capture=True)
            self.history.append(dict(step=self.step, residuals=context['residuals']))
            self.history = self.history[-2:]
            self.counts['residual_refresh'] += 1
            timing.update({f'refresh_{k}':v for k,v in costs.items()})
            del context  # No action expert is invoked on the full context.
        last = self.history[-1] if self.history else None
        previous = self.history[-2] if len(self.history) == 2 else None
        age = None if last is None else self.step-last['step']
        gap = None if previous is None else last['step']-previous['step']
        ratio = 0. if gap is None else age/gap
        context, costs = self.b.prepare_residual(obs, instruction, MASK,
            history=None if last is None else last['residuals'],
            older=None if previous is None else previous['residuals'], ratio=ratio)
        timing.update(costs)
        actions, raw, costs = self.sampler(self.b, context, seed, 2)
        timing.update(costs)
        actions = np.asarray(actions, np.float32)
        raw = raw.float().cpu().numpy()
        if not np.isfinite(actions).all() or not np.isfinite(raw).all():
            raise ValueError('Nonfinite action output')
        self.counts['draft'] += 1
        timing['policy_wall_ms'] = (time.perf_counter()-start)*1000
        result = dict(actions=actions, raw=raw, arrays={}, execute_length=5,
            round=self.round, step=self.step, seed=seed, timing=timing,
            counts=dict(self.counts), full_cache_age=age,
            route=dict(path='v12_d2', action_source='v12_d2', D=2, skip=list(MASK),
                compensation=self.config['compensation'], residual_refresh=refresh,
                full_ae_executed=False, full_action_executed=False,
                history_steps=[h['step'] for h in self.history],
                has_second_history=previous is not None, extrapolation_ratio=ratio,
                fast_time_embedding='openpi_linspace', native_encoder_graph=False,
                native_ae_graph=False, verify_graph=False),
            model_noise_sha256=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32), dtype=np.float32).tobytes()).hexdigest())
        self.pending = result
        self.round += 1
        return result

    def acknowledge(self, executed, quaternions=None):
        if self.pending is None:
            raise RuntimeError('No pending actions')
        executed = np.asarray(executed, np.float32)
        if not 0 < len(executed) <= 5 or not np.array_equal(executed, self.pending['actions'][:len(executed)]):
            raise ValueError('Actual actions differ from proposed prefix')
        self.step += len(executed)
        self.pending = None
        return {}
