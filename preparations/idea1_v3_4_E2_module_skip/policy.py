"""Full-D10/4xFast-D2, module histories and one Full-only VJP."""
import time
import numpy as np
from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Base
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_adjacency_200.common import choose, legal
from .block_backend import prepare_block
from .model import Selector, forecast_error
from .selection import select_ffn, fixed_episode_mask, mask_record


class Policy(Base):
    def __init__(self, backend, budgets):
        self.b = backend
        self.budgets = budgets
        self.selector = Selector(backend)
        self.config = {}
        self.reset()

    def reset(self):
        super().reset()
        self.history = []
        self.error_buffer = None
        self.history_identity = None
        self.score_mode = None
        self.score_source_steps = []

    def configure(self, config):
        assert config['period'] == 5 and config['adjacency_rule'] == 'spaced'
        assert config['family'] in ('block', 'ffn')
        assert config['scoring'] in ('raw', 'error', 'fixed')
        self.selector.clear()
        self.config = dict(config)
        self.reset()
        return {}

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Unacknowledged action')
        tick = time.perf_counter()
        cfg = self.config
        full = self.round % 5 == 0
        block = cfg['family'] == 'block'
        structure = 'block' if block else 'ffn'
        comp = cfg['compensation'] == 'extrapolate'
        arrays = {}
        ratio, has_older, refreshed = 0., False, False
        selection = dict(selector_total_ms=0., selector_hot_ms=0., selector_capture_ms=0.,
                         error_build_ms=0., mask_selection_ms=0.)
        if full:
            if block:
                c, timing = prepare_block(self.b, obs, instruction, capture=True)
            else:
                c, timing = self.b.prepare_modules(obs, instruction, capture=('ffn',))
            identity = (instruction, c['seq'])
            if self.history_identity is not None and identity != self.history_identity:
                raise ValueError('Residual prompt/shape identity changed inside episode')
            self.history_identity = identity
            actions, raw, costs = self.b.sample(c, seed, 10)
            residuals = c.pop('residuals')
            if cfg['scoring'] == 'fixed':
                if self.skip is None:
                    start = time.perf_counter()
                    self.skip = list(cfg['fixed_skip']) if 'fixed_skip' in cfg else fixed_episode_mask(
                        self.budgets[cfg['budget']], cfg['episode_case'])
                    selection['mask_selection_ms'] = (time.perf_counter()-start)*1000
                    refreshed = True
                self.score_mode = 'fixed_episode'
            else:
                error = None
                if cfg['scoring'] == 'error' and len(self.history) == 2:
                    older, last = self.history
                    gap = last['step']-older['step']
                    if gap <= 0:
                        raise ValueError('Non-increasing Full history')
                    eratio = (self.step-last['step'])/gap
                    self.error_buffer, build_ms = self.b.adapter.measure(lambda:forecast_error(
                        residuals, last['residuals'], older['residuals'], eratio,
                        out=self.error_buffer, structure=structure))
                    selection['error_build_ms'] = build_ms
                    error = self.error_buffer
                    self.score_mode = 'comp_error'
                    self.score_source_steps = [older['step'], last['step'], self.step]
                    arrays['prediction_error_rms'] = error.float().square().mean(tuple(range(1,error.ndim))).sqrt().cpu().numpy()
                else:
                    self.score_mode = 'warm_original' if cfg['scoring']=='error' else 'original'
                    self.score_source_steps = []
                grad, score, st = self.selector.run(c, seed, coefficients(), error=error, structure=structure)
                selection.update(st)
                start = time.perf_counter()
                self.skip = choose(score, cfg['block_skip_count'], 'spaced') if block else select_ffn(score, self.budgets[cfg['budget']])
                selection['mask_selection_ms'] = (time.perf_counter()-start)*1000
                arrays.update(gate_gradient=grad, module_score=score)
                refreshed = True
            # Read old histories to score first; commit this Full afterwards.
            self.history = (self.history+[dict(step=self.step, residuals=residuals)])[-2:]
            self.last_full_step, self.last_full_round = self.step, self.round
            self.counts['full'] += 1
            if refreshed:
                self.mask_step = self.step
                self.counts['selector'] += int(cfg['scoring']!='fixed')
        else:
            if not self.history or self.skip is None:
                raise RuntimeError('Fast without Full history/mask')
            last = self.history[-1]
            previous = self.history[-2] if len(self.history)==2 else None
            has_older = previous is not None
            ratio = 0. if previous is None else (self.step-last['step'])/(last['step']-previous['step'])
            hist = last['residuals'] if comp else None
            old = previous['residuals'] if previous is not None and comp else None
            if block:
                c, timing = prepare_block(self.b, obs, instruction, skip=self.skip,
                                           history=hist, older=old, ratio=ratio)
            else:
                c, timing = self.b.prepare_modules(obs, instruction, skip_ffn=self.skip,
                    history=hist, older=old, ratio=ratio)
            if (instruction, c['seq']) != self.history_identity:
                raise ValueError('Fast history prompt/shape mismatch')
            actions, raw, costs = self.b.sample(c, seed, 2)
            self.fast_count += 1
            self.counts['draft'] += 1
        assert legal(self.skip, len(self.skip), 'spaced')
        timing.update(costs, **selection)
        timing['policy_wall_ms'] = (time.perf_counter()-tick)*1000
        timing['policy_hot_ms'] = timing['policy_wall_ms']-timing['selector_capture_ms']
        def history_bytes(bank):
            if isinstance(bank, dict):
                return sum(history_bytes(v) for v in bank.values())
            return bank.numel()*bank.element_size()
        route = dict(path='full' if full else 'fast', D=10 if full else 2,
            skip=[] if full else list(self.skip), selected_skip=list(self.skip),
            selected_ffn=[] if block else list(self.skip), selected_attn=list(self.skip) if block else [],
            mask_refresh=refreshed, mask_age_env_steps=self.step-self.mask_step,
            full_age_rounds=self.round-self.last_full_round, full_age_env_steps=self.step-self.last_full_step,
            method=cfg['method'], family=cfg['family'], budget=cfg['budget'],
            compensation=cfg['compensation'], period=5, adjacency_rule='spaced',
            steady_vjp_count=int(full and cfg['scoring']!='fixed'),
            history_full_steps=[h['step'] for h in self.history], has_second_history=has_older,
            extrapolation_ratio=ratio, scoring_mode=self.score_mode,
            score_source_full_steps=list(self.score_source_steps),
            history_bytes=sum(history_bytes(h['residuals']) for h in self.history),
            selected_module_count=len(self.skip)*(2 if block else 1),
            current_observation_kv=True, patch_layers=[] if full or not comp else list(self.skip),
            jacobian_model_steps=10 if cfg['scoring']!='fixed' else 0,
            jacobian_output_steps=[1,2,3,4,5], jacobian_full_layers=18)
        if not block:
            budget = self.budgets[cfg['budget']]
            route['frozen_budget_id'] = cfg['budget']
            route.update(mask_record(self.skip, budget))
        result = dict(actions=np.asarray(actions,np.float32), raw=raw.float().cpu().numpy(),
            arrays=arrays, execute_length=5, route=route, timing=timing, seed=seed,
            round=self.round, step=self.step, counts=dict(self.counts),
            full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,
                                                  env_steps=self.step-self.last_full_step))
        # Keep a bounded pilot observation bank without any extra model requests.
        if cfg['id'].endswith('F-R+') and cfg.get('pilot', False) and self.round < 15:
            result['save_pilot_input'] = True
        self.pending = result
        self.round += 1
        return result
