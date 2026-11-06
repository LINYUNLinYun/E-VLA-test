"""The three independent v4.0 interventions on one acknowledged control path."""
import gc
import json
import time
from collections import OrderedDict
import numpy as np
import torch
from preparations.idea1_v3.policy import verification_seed
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v3_3.rules import acceptance
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture, prepare_patch
from preparations.idea1_v3_4_E2_comp_error_cost.model import forecast_error
from preparations.idea1_v3_4_selection_async.core import choose
from .common import OUT
from preparations.idea1_v4_0.rules import KalmanPrior, motion_features, skip_budget
from .schedule import mix
from .sampling import sample_initial, sample_gaussian
from preparations.idea1_v4_0.selection import Graph


class Policy:
    def __init__(self, backend):
        self.b = backend
        self.low = torch.cuda.Stream(priority=0)
        self.high = torch.cuda.Stream(priority=-1)
        self.graphs = OrderedDict()
        self.job = None
        self.pending = None
        self.config = {}
        self.thresholds = json.loads((OUT/'manifest.json').read_text())['phase_rule']['thresholds']
        self.reset()

    def reset(self):
        assert self.job is None and self.pending is None
        self.full = None
        self.history = []
        self.gradient = None
        self.skip = None
        self.mask_step = None
        self.round = self.step = self.draft_rounds = self.job_sequence = 0
        self.pending_gripper = False
        self.last_gripper = None
        self.commands, self.quaternions = [], []
        self.kalman = KalmanPrior() if self.config.get('initial') == 'kalman' else None
        self.error_buffer = None
        self.counts = dict(full=0, draft=0, verify=0, selector=0, completed_selector=0,
                           vjp=0, kalman=0)

    def configure(self, config):
        assert self.job is None and self.pending is None
        assert config['mode'] in ('full', 'verified') and config['period'] in (1, 3)
        assert config['fast_steps'] in (1, 2) and config['projections'] == 4
        assert config['scoring'] in ('original', 'compensation_error')
        assert config['initial'] in ('gaussian', 'kalman')
        assert config['tau'] == .10 and config['protect_gripper']
        assert not config['native_inference_graph'] and not config['verify_graph']
        if config.get('warmstart'):
            assert config['initial']=='kalman' and 0 < config['start_time'] <= 1
            assert 0 <= config['noise_scale'] <= 1 and not config['dynamic']
        self.config = dict(config)
        self.reset()
        return {}

    def clear(self):
        assert self.job is None and self.pending is None
        for graph in self.graphs.values():
            graph.close()
        self.graphs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        self.reset()

    def acknowledge(self, executed, quaternions):
        executed = np.asarray(executed, np.float32)
        quaternions = np.asarray(quaternions)
        assert self.pending is not None and 0 < len(executed) <= self.pending['execute_length']
        assert np.array_equal(executed, self.pending['actions'][:len(executed)])
        assert quaternions.shape == (len(executed), 4) and np.isfinite(quaternions).all()
        for command, pose in zip(executed, quaternions):
            effective = command.copy()
            effective[:6] = np.clip(effective[:6], -1, 1)
            self.commands.append(effective)
            self.quaternions.append(pose.copy())
            if self.kalman is not None:
                self.kalman.observe(effective)
        self.last_gripper = float(self.pending['raw'][len(executed)-1, 6])
        self.step += len(executed)
        self.pending = None
        return {}

    def _graph(self, context, seed, kind):
        key = (id(context['runtime']), kind)
        cold = 0.
        if key not in self.graphs:
            # The two scoring modes can share frozen model weights, but retain
            # independent graph inputs. Bound prompt/shape graph pools to two.
            while len(self.graphs) >= 2:
                _, graph = self.graphs.popitem(last=False)
                graph.close()
                gc.collect()
                torch.cuda.empty_cache()
            tick = time.perf_counter()
            self.graphs[key] = Graph(self.b, context, seed, kind)
            cold = (time.perf_counter()-tick)*1000
        self.graphs.move_to_end(key)
        return self.graphs[key], cold

    def _collect(self, arrays, events, timing, reason):
        if self.job is None:
            return
        tick = time.perf_counter()
        job = dict(self.job)
        graph = job.pop('graph')
        ready = graph.end.query()
        result = graph.collect(6, 'layer_rms_sum')
        self.gradient = result['gradient']
        self.skip = result['mask']
        self.mask_step = job['origin_step']
        self.counts['completed_selector'] += 1
        for name, source in (('gate_gradient', 'gradient'), ('layer_score', 'layer_score')):
            arrays[f"{name}_job{job['id']}"] = result[source]
        event = dict(job, reason=reason, completion_step=self.step,
            ready_before_wait=ready, wait_ms=result['wait_ms'], gpu_ms=result['gpu_ms'],
            score_ms=result['score_ms'], selected_skip=list(self.skip),
            collect_wall_ms=(time.perf_counter()-tick)*1000)
        events.append(event)
        timing['selector_wait_ms'] = timing.get('selector_wait_ms', 0.)+result['wait_ms']
        timing['selector_collect_ms'] = timing.get('selector_collect_ms', 0.)+event['collect_wall_ms']
        self.job = None

    def finish(self):
        assert self.pending is None
        arrays, events, timing = {}, [], {}
        self._collect(arrays, events, timing, 'episode_end')
        return dict(arrays=arrays, selector_events=events, timing=timing, counts=dict(self.counts))

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Previous execution was not acknowledged')
        cfg, backend = self.config, self.b
        start = time.perf_counter()
        timing, arrays, events = {}, {}, []
        age = None if self.full is None else dict(rounds=self.round-self.full['round'],
                                                 env_steps=self.step-self.full['step'])
        route = dict(path=None, draft_attempted=False, verify_performed=False,
            accepted_length=None, fallback=False, full_reason=None,
            configured_fast_steps=cfg['fast_steps'], configured_initial=cfg['initial'],
            configured_scoring=cfg['scoring'], dynamic_budget=cfg['dynamic'],
            period=cfg['period'], mask_refresh=False, steady_vjp_count=0,
            selector_forward_count=0, draft_skip=[], full_age_at_entry=age,
            native_encoder_graph=False, native_ae_graph=False, verify_graph=False,
            selector_cuda_graph=True, scoring_mode=None,
            kalman_observations=0 if self.kalman is None else self.kalman.filter.count)

        def full(reason):
            self._collect(arrays, events, timing, 'before_full')
            tick = time.perf_counter()
            patch = cfg['mode'] == 'verified'
            context, costs = prepare_full(backend, obs, instruction, trace=patch)
            if patch:
                residuals, capture_ms = backend.adapter.measure(lambda: capture(context.pop('hidden_trace')))
                costs['residual_capture_ms'] = capture_ms
                kind, score_mode, source_steps, eratio = 'original', 'original', [], None
                if cfg['scoring'] == 'compensation_error' and self.history:
                    last = self.history[-1]
                    older = self.history[-2] if len(self.history) > 1 else last
                    eratio = (self.step-last['step'])/(last['step']-older['step']) if len(self.history)>1 else 0.
                    self.error_buffer, build_ms = backend.adapter.measure(lambda: forecast_error(
                        residuals, last['residuals'], older['residuals'], eratio, out=self.error_buffer))
                    costs['error_build_ms'] = build_ms
                    context['selection_error'] = self.error_buffer
                    kind = 'error'
                    score_mode = 'comp_error_extrapolate' if len(self.history)>1 else 'comp_error_hold'
                    source_steps = ([older['step']] if len(self.history)>1 else [])+[last['step'], self.step]
                elif cfg['scoring'] == 'compensation_error':
                    score_mode = 'warm_original'
                self.history = (self.history+[dict(step=self.step, residuals=residuals)])[-2:]
                graph, cold = self._graph(context, seed, kind)
                timing['selector_capture_ms'] = cold
                launch = time.perf_counter()
                graph.launch(graph.stage(context, seed), self.low)
                self.job = dict(graph=graph, id=self.job_sequence, origin_round=self.round,
                    origin_step=self.step, actual_full_index=self.counts['full'], seed=seed,
                    scoring_mode=score_mode, score_source_full_steps=source_steps, score_error_ratio=eratio)
                self.job_sequence += 1
                self.counts['selector'] += 1
                self.counts['vjp'] += 4
                timing['selector_launch_ms'] = (time.perf_counter()-launch)*1000
                route.update(mask_refresh=True, steady_vjp_count=4, selector_forward_count=1,
                    scoring_mode=score_mode, score_source_full_steps=source_steps, score_error_ratio=eratio)
                self.high.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.high):
                    actions, raw, ae_costs = backend.sample(context, seed, 10)
            else:
                actions, raw, ae_costs = backend.sample(context, seed, 10)
            self.full = dict(context, round=self.round, step=self.step)
            self.draft_rounds = 0
            self.pending_gripper = False
            self.counts['full'] += 1
            timing.update({f'full_{k}':v for k, v in dict(costs, **ae_costs).items()})
            timing['full_path_wall_ms'] = (time.perf_counter()-tick)*1000
            route.update(path='full', full_reason=reason, full_steps=10,
                         actual_full_index=self.counts['full']-1)
            return actions, raw, 5

        if cfg['mode'] == 'full':
            actions, raw, length = full('baseline')
        elif self.full is None:
            actions, raw, length = full('startup')
        elif self.pending_gripper:
            actions, raw, length = full('pending_gripper')
        elif self.draft_rounds >= cfg['period']-1:
            actions, raw, length = full('periodic')
        else:
            self._collect(arrays, events, timing, 'before_fast')
            assert self.gradient is not None
            features = motion_features(self.commands, self.quaternions) if cfg['dynamic'] else None
            k, phase = skip_budget(features, self.thresholds) if cfg['dynamic'] else (6, 'fixed')
            if cfg['dynamic']:
                tick = time.perf_counter()
                self.skip, _, _ = choose(self.gradient, k, 'layer_rms_sum')
                timing['budget_score_ms'] = (time.perf_counter()-tick)*1000
            route.update(draft_attempted=True, draft_skip=list(self.skip), skip_count=k,
                budget=18-k, phase=phase, phase_features=features, mask_origin_step=self.mask_step,
                draft_steps=cfg['fast_steps'], kalman_used=cfg['initial']=='kalman')
            prior = None
            if self.kalman is not None:
                tick = time.perf_counter()
                prior = self.kalman.forecast()
                timing['kalman_predict_ms'] = (time.perf_counter()-tick)*1000
                assert self.kalman.filter.count == self.step
                route['kalman_source_last_step'] = self.step-1
            tick = time.perf_counter()
            last = self.history[-1]
            older = self.history[-2] if len(self.history)>1 else None
            gap = None if older is None else last['step']-older['step']
            ratio = 0. if gap is None else (self.step-last['step'])/gap
            context, costs = prepare_patch(backend, obs, instruction, self.skip, last['residuals'],
                None if older is None else older['residuals'], ratio)
            if prior is None:
                actions, raw, ae_costs = sample_gaussian(backend, context, seed, cfg['fast_steps'])
            else:
                encode_tick = time.perf_counter()
                coordinates = DecoderCoordinates(backend.adapter.backend.output_transform, context['transformed_state'])
                initial = coordinates.encode_commands(prior, np.zeros((50, 32), np.float32))
                timing['kalman_encode_ms'] = (time.perf_counter()-encode_tick)*1000
                mix_tick = time.perf_counter()
                initial = mix(initial, seed, cfg['start_time'], cfg['noise_scale'])
                timing['warmstart_mix_ms'] = (time.perf_counter()-mix_tick)*1000
                actions, raw, ae_costs = sample_initial(backend, context, initial, cfg['fast_steps'], cfg['start_time'])
                self.counts['kalman'] += 1
                arrays['kalman_initial_prefix'] = initial[:5].copy()
                route.update(initial_state='kalman', prior_unused_dims='zeros25', initial_unused_dims='scaled_gaussian25',
                             initial_time_nodes=ae_costs['nodes']['timesteps'],
                             warmstart_time=cfg['start_time'], warmstart_noise_scale=cfg['noise_scale'])
            self.counts['draft'] += 1
            self.draft_rounds += 1
            route.update(draft_history_full_steps=[h['step'] for h in self.history],
                draft_extrapolation_ratio=ratio, captured_layers=list(range(1,18)),
                fast_time_embedding=ae_costs['time_embedding'],
                actual_fast_nfe=(ae_costs['nodes']['actual_NFE'] if isinstance(ae_costs['nodes'],dict)
                                 else len(ae_costs['nodes'])-1))
            timing.update({f'draft_{k}':v for k, v in dict(costs, **ae_costs).items()})
            timing['draft_path_wall_ms'] = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            hats, verify_costs = backend.verify(self.full, context, raw, seed, suite)
            raw_np = raw.float().cpu().numpy()
            length, details = acceptance(raw_np, hats, cfg['tau'], 5,
                last_gripper=self.last_gripper, protect=True)
            timing.update(verify_costs)
            timing['verify_path_wall_ms'] = (time.perf_counter()-tick)*1000
            self.counts['verify'] += 1
            route.update(verify_performed=True, accepted_length=length, **details)
            arrays.update(draft_future=raw_np[:5].copy(), reconstructed_future=hats[:, :5].copy())
            switch = details['verify_gripper'] or details['post_verify_gripper']
            if length == 0:
                route['fallback'] = True
                actions, raw, length = full('reject_gripper' if switch else 'reject_spatial')
            else:
                route['path'] = 'fast_verified'
                if switch:
                    self.pending_gripper = True
        raw_np = raw.float().cpu().numpy()
        actions = np.asarray(actions, np.float32)
        assert raw_np.shape == (50,32) and actions.shape == (50,7)
        assert np.isfinite(raw_np).all() and np.isfinite(actions).all()
        category = ('full_'+route['full_reason']) if route['path']=='full' else ('fast_accept_5' if length==5 else 'fast_accept_partial')
        route.update(request_category=category, selected_skip=[] if self.skip is None else list(self.skip),
            history_full_steps_after=[h['step'] for h in self.history], selection_stride=1,
            actual_fulls_after=self.counts['full'], projections=4, batched=True,
            aggregation='layer_rms_sum', asynchronous=True)
        timing.setdefault('selector_capture_ms', 0.)
        timing.setdefault('selector_wait_ms', 0.)
        timing['policy_wall_ms'] = (time.perf_counter()-start)*1000
        result = dict(actions=actions, raw=raw_np, execute_length=length, route=route, timing=timing,
            arrays=arrays, selector_events=events, counts=dict(self.counts), full_cache_age=age,
            round=self.round, step=self.step, seed=seed, verification_seed=verification_seed(seed))
        self.round += 1
        self.pending = result
        return result
