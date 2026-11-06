"""Mixed native module skips; Full-only error scoring and fixed cost windows."""
import time

import numpy as np

from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Base
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_module_skip.model import Selector, forecast_error
from .selection import select_mixed, mask_record, objective_scores


class Policy(Base):
    def __init__(self, backend, budgets):
        self.b = backend
        self.budgets = budgets
        self.selector = Selector(backend, structure='mixed')
        self.config = {}
        self.reset()

    def reset(self):
        super().reset()
        self.history = []
        self.error_buffer = None
        self.history_identity = None
        self.score_mode = None
        self.score_source_steps = []
        self.skip_attn = self.skip_ffn = None

    def configure(self, config):
        if (config['period'] != 5 or config['adjacency_rule'] != 'spaced'
                or config['family'] != 'mixed' or config['scoring'] != 'error'
                or config['compensation'] != 'extrapolate'):
            raise ValueError('Mixed comparison requires the frozen E+/period5 protocol')
        self.selector.clear()
        self.config = dict(config)
        self.reset()
        return {}

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Unacknowledged action')
        begin = time.perf_counter()
        cfg = self.config
        budget = self.budgets[cfg['budget']]
        weighted = bool(cfg['cost_weighted'])
        full = self.round % 5 == 0
        arrays = {}
        ratio, has_older = 0., False
        selection = dict(selector_total_ms=0., selector_hot_ms=0.,
                         selector_capture_ms=0., error_build_ms=0., mask_selection_ms=0.)
        if full:
            c, timing = self.b.prepare_modules(obs, instruction, capture=True)
            identity = (instruction, c['seq'])
            if self.history_identity is not None and identity != self.history_identity:
                raise ValueError('Residual prompt/shape identity changed inside episode')
            self.history_identity = identity
            actions, raw, costs = self.b.sample(c, seed, 10)
            residuals = c.pop('residuals')
            error = None
            if len(self.history) == 2:
                older, last = self.history
                gap = last['step'] - older['step']
                if gap <= 0:
                    raise ValueError('Non-increasing Full history')
                eratio = (self.step - last['step']) / gap
                self.error_buffer, ms = self.b.adapter.measure(lambda: forecast_error(
                    residuals, last['residuals'], older['residuals'], eratio,
                    out=self.error_buffer, structure='mixed'))
                selection['error_build_ms'] = ms
                error = self.error_buffer
                self.score_mode = 'comp_error'
                self.score_source_steps = [older['step'], last['step'], self.step]
                arrays['prediction_error_rms'] = error.square().mean(
                    tuple(range(2, error.ndim))).sqrt().cpu().numpy()
            else:
                self.score_mode = 'warm_original'
                self.score_source_steps = []
            gradient, score, st = self.selector.run(
                c, seed, coefficients(), error=error, structure='mixed')
            selection.update(st)
            tick = time.perf_counter()
            self.skip_attn, self.skip_ffn = select_mixed(score, budget, cost_weighted=weighted)
            self.skip_attn, self.skip_ffn = list(self.skip_attn), list(self.skip_ffn)
            self.skip = sorted(set(self.skip_attn) | set(self.skip_ffn))
            selection['mask_selection_ms'] = (time.perf_counter() - tick) * 1000
            arrays.update(gate_gradient=gradient, module_score=score,
                          selection_score=objective_scores(score, budget, cost_weighted=weighted))
            self.history = (self.history + [dict(step=self.step, residuals=residuals)])[-2:]
            self.last_full_step, self.last_full_round = self.step, self.round
            self.mask_step = self.step
            self.counts['full'] += 1
            self.counts['selector'] += 1
        else:
            if not self.history or self.skip_attn is None or self.skip_ffn is None:
                raise RuntimeError('Fast without Full module history/mask')
            last = self.history[-1]
            previous = self.history[-2] if len(self.history) == 2 else None
            has_older = previous is not None
            ratio = 0. if previous is None else (
                (self.step - last['step']) / (last['step'] - previous['step']))
            c, timing = self.b.prepare_modules(obs, instruction,
                skip_attn=self.skip_attn, skip_ffn=self.skip_ffn,
                history=last['residuals'],
                older=None if previous is None else previous['residuals'], ratio=ratio)
            if (instruction, c['seq']) != self.history_identity:
                raise ValueError('Fast history prompt/shape mismatch')
            actions, raw, costs = self.b.sample(c, seed, 2)
            self.fast_count += 1
            self.counts['draft'] += 1
        if any(b - a == 1 for a, b in zip(self.skip, self.skip[1:])):
            raise AssertionError('Adjacent skipped support layers')
        timing.update(costs, **selection)
        timing['policy_wall_ms'] = (time.perf_counter() - begin) * 1000
        timing['policy_hot_ms'] = timing['policy_wall_ms'] - timing['selector_capture_ms']

        def nbytes(value):
            return sum(map(nbytes, value.values())) if isinstance(value, dict) else value.numel() * value.element_size()

        route = dict(path='full' if full else 'fast', D=10 if full else 2,
            skip=[] if full else list(self.skip), selected_skip=list(self.skip),
            selected_attn=list(self.skip_attn), selected_ffn=list(self.skip_ffn),
            actual_skip_attn=[] if full else list(self.skip_attn),
            actual_skip_ffn=[] if full else list(self.skip_ffn),
            selected_af=sorted(set(self.skip_attn) & set(self.skip_ffn)),
            selected_module_count=len(self.skip_attn) + len(self.skip_ffn),
            mask_refresh=full, mask_age_env_steps=self.step-self.mask_step,
            full_age_rounds=self.round-self.last_full_round,
            full_age_env_steps=self.step-self.last_full_step,
            method=cfg['method'], family='mixed', budget=cfg['budget'],
            cost_weighted=weighted, objective='sum(score / standalone_saved_ms)' if weighted else 'sum(score)',
            compensation='extrapolate', period=5, adjacency_rule='spaced',
            steady_vjp_count=int(full), history_full_steps=[h['step'] for h in self.history],
            has_second_history=has_older, extrapolation_ratio=ratio,
            scoring_mode=self.score_mode, score_source_full_steps=list(self.score_source_steps),
            history_bytes=sum(nbytes(h['residuals']) for h in self.history),
            current_observation_kv=True,
            patch_layers=[] if full else list(self.skip),
            patch_attn=[] if full else list(self.skip_attn),
            patch_ffn=[] if full else list(self.skip_ffn),
            jacobian_model_steps=10, jacobian_output_steps=[1,2,3,4,5], jacobian_full_layers=18)
        route.update(mask_record(self.skip_attn, self.skip_ffn, budget))
        route['native_module_receipt'] = c['module_meta']
        result = dict(actions=np.asarray(actions, np.float32), raw=raw.float().cpu().numpy(),
            arrays=arrays, execute_length=5, route=route, timing=timing, seed=seed,
            round=self.round, step=self.step, counts=dict(self.counts),
            full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,
                                                   env_steps=self.step-self.last_full_step))
        self.pending = result
        self.round += 1
        return result
