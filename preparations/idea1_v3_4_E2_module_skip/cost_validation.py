"""One-input native Fast cost pairing and instrumented module-route checks.

At most 51 native requests plus two eager VJPs. This is a cost/numerical probe
using a saved real observation, not evidence about free closed-loop quality.
Masks and tolerances come from the frozen budget; results never retune them.
"""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.common import choose
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from .block_backend import prepare_block
from .common import ROOT
from .model import Selector
from .selection import mask_record, select_ffn


MAX_NATIVE_REQUESTS = 51
MAX_EAGER_VJPS = 2


def _kv(c):
    buffers, seq = c['runtime'].buffers, c['seq']
    return tuple(buffers[name][:, :seq].clone() for name in ('encoder_K', 'encoder_V'))


def _assert_equal(reference, candidate, label):
    for name in ('raw', 'K', 'V'):
        if not torch.allclose(reference[name], candidate[name], atol=1e-5, rtol=1e-5):
            raise AssertionError((label, name, 'atol=rtol=1e-5'))
    if not np.allclose(reference['actions'], candidate['actions'], atol=1e-5, rtol=1e-5):
        raise AssertionError((label, 'actions', 'atol=rtol=1e-5'))


def _stages(rows, layer=None):
    totals = {}
    for row in rows:
        if layer is not None and row.get('layer') != layer:
            continue
        name = row['stage']
        totals[name] = totals.get(name, 0.) + float(row['ms'])
    return totals


def _median_stages(rows, field):
    names = sorted({name for row in rows for name in row[field]})
    return {name: float(np.median([row[field].get(name, 0.) for row in rows])) for name in names}


def run(backend, out, budgets):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from preparations.kv_reuse_probe.probe import load_observation
    manifest_path = ROOT / 'results/idea1_v3_4_E2_comp_error_cost/manifest.json'
    sample = json.loads(manifest_path.read_text())['samples'][0]
    observation, prompt = load_observation(sample['current'])
    seed = int(sample['current']['seed'])
    counts = dict(native_requests=0, eager_vjps=0)
    started = time.time()
    selectors = []

    def status(stage):
        atomic(out / 'cost_validation_status.json', dict(status=stage, **counts,
            maximum_native_requests=MAX_NATIVE_REQUESTS, maximum_eager_vjps=MAX_EAGER_VJPS,
            started=started, updated=time.time()))

    def call(kind='full', mask=(), history=None, older=None, capture=False,
             profile=False, snapshot=False, direct=False):
        if counts['native_requests'] >= MAX_NATIVE_REQUESTS:
            raise RuntimeError('Bounded cost validation native request limit reached')
        counts['native_requests'] += 1
        torch.cuda.synchronize()
        tick = time.perf_counter()
        if direct:
            c, timing = backend.prepare(observation, prompt)
        elif kind == 'block':
            c, timing = prepare_block(backend, observation, prompt, skip=mask,
                history=history, older=older, ratio=.8, capture=capture)
        else:
            c, timing = backend.prepare_modules(observation, prompt,
                skip_attn=mask if kind in ('A', 'AF') else (),
                skip_ffn=mask if kind in ('F', 'AF') else (),
                history=history, older=older, ratio=.8, capture=capture, profile=profile)
        actions, raw, costs = backend.sample(c, seed, 2)
        torch.cuda.synchronize()
        result = dict(timing, **costs, request_wall_ms=(time.perf_counter()-tick)*1000)
        payload = None
        if snapshot:
            k, v = _kv(c)
            payload = dict(raw=raw.detach().clone(), actions=np.asarray(actions).copy(), K=k, V=v)
        return c, result, payload

    def eager_scores(c, structure):
        if counts['eager_vjps'] >= MAX_EAGER_VJPS:
            raise RuntimeError('Bounded cost validation eager VJP limit reached')
        counts['eager_vjps'] += 1
        selector = Selector(backend, structure=structure, use_graph=False)
        selectors.append(selector)
        try:
            grad, scores, timing = selector.run(c, seed, coefficients())
            return scores, dict(gradient=grad.tolist(), scores=scores.tolist(), timing=timing)
        finally:
            selector.clear()
            selectors.remove(selector)

    status('running')
    try:
        # Setup capture is never included in the measured Fast comparisons.
        current, _, full = call(capture=True, snapshot=True)
        module_history = current.pop('residuals')
        block, _, block_full = call(kind='block', capture=True, snapshot=True)
        block_history = block.pop('residuals')
        _, _, direct_full = call(direct=True, snapshot=True)
        _assert_equal(direct_full, full, 'module all-keep/capture versus old Full')
        _assert_equal(direct_full, block_full, 'block all-keep/capture versus old Full')
        ffn_score, ffn_selector = eager_scores(current, 'ffn')
        block_score, block_selector = eager_scores(block, 'block')
        del block_full, direct_full, current, block
        status('setup_complete')

        costs = []
        for budget_id, count in (('B4', 4), ('B6', 6)):
            budget = budgets[budget_id]
            ffn_mask = select_ffn(ffn_score, budget)
            block_mask = choose(block_score, count, 'spaced')
            branches = dict(F=dict(kind='F', mask=ffn_mask, history=module_history, older=module_history),
                            B=dict(kind='block', mask=block_mask, history=block_history, older=block_history))
            for name in ('B', 'F'):
                for _ in range(2):
                    call(**branches[name])
            pairs = []
            for repeat in range(4):
                # Four two-request pairs form ABBAABBA; each route is equally
                # often first. All pairs use the same input and explicit noise.
                order = ('B', 'F') if repeat % 2 == 0 else ('F', 'B')
                values = {}
                for name in order:
                    _, values[name], _ = call(**branches[name])
                pairs.append(dict(repeat=repeat, order=list(order), block=values['B'], ffn=values['F'],
                    request_ffn_minus_block_ms=values['F']['request_wall_ms']-values['B']['request_wall_ms'],
                    vlm_ffn_minus_block_ms=values['F']['vlm_ms']-values['B']['vlm_ms']))
            wall_delta = float(np.median([row['request_ffn_minus_block_ms'] for row in pairs]))
            vlm_delta = float(np.median([row['vlm_ffn_minus_block_ms'] for row in pairs]))
            tau = float(budget['tau_ms'])
            row = dict(budget=budget_id, budget_hash=budget['budget_hash'], tau_ms=tau,
                block_mask=block_mask, block_skip_count=len(block_mask),
                ffn_mask=ffn_mask, ffn_skip_count=len(ffn_mask),
                frozen_ffn_mask_record=mask_record(ffn_mask, budget),
                frozen_budget_nearest_tier=budget['nearest_tier'],
                warmup_requests_per_branch=2, hot_pairs=4, pairs=pairs,
                block_fast_request_ms=float(np.median([r['block']['request_wall_ms'] for r in pairs])),
                ffn_fast_request_ms=float(np.median([r['ffn']['request_wall_ms'] for r in pairs])),
                block_fast_vlm_ms=float(np.median([r['block']['vlm_ms'] for r in pairs])),
                ffn_fast_vlm_ms=float(np.median([r['ffn']['vlm_ms'] for r in pairs])),
                paired_request_ffn_minus_block_ms=wall_delta, paired_vlm_ffn_minus_block_ms=vlm_delta,
                matched_actual_cost=bool(wall_delta <= tau),
                matched_actual_vlm_cost=bool(vlm_delta <= tau),
                same_or_lower_request_cost=bool(wall_delta <= 0),
                actual_cost_criterion='median paired full Fast request wall difference <= frozen tau',
                masks_retuned_after_measurement=False)
            costs.append(row)
            print(json.dumps(dict(cost_validation=budget_id, native_requests=counts['native_requests'],
                request_ffn_minus_block_ms=wall_delta, tau_ms=tau,
                matched_actual_cost=row['matched_actual_cost'])), flush=True)
            status('paired_' + budget_id)

        profiles, restores = [], []
        for kind in ('full', 'A', 'F', 'AF'):
            kwargs = {} if kind == 'full' else dict(kind=kind, mask=(8,), history=module_history, older=module_history)
            for _ in range(2):
                call(**kwargs)
            observations = []
            for repeat in range(4):
                c, timing, candidate = call(**kwargs, profile=True, snapshot=True)
                _assert_equal(full, candidate, f'{kind} L8 true-residual injection restores Full')
                meta = c['module_meta']
                if not meta['current_observation_kv']:
                    raise AssertionError((kind, 'current-observation K/V'))
                layer8 = next(route for route in meta['routes'] if route['layer'] == 8)
                expected_attn, expected_ffn = kind not in ('A', 'AF'), kind not in ('F', 'AF')
                if (layer8['attn'] != expected_attn or layer8['ffn'] != expected_ffn or
                    layer8['projection'] != ('qkv' if expected_attn else 'kv')):
                    raise AssertionError((kind, 'wrong native module route', layer8))
                rows = meta['profile_ms']
                observations.append(dict(repeat=repeat, timing=timing, events=rows,
                    stage_total_ms=_stages(rows), layer8_ms=_stages(rows, layer=8),
                    layer8_route=layer8))
                del candidate, c
            profiles.append(dict(route=kind, layer=8, warmup_requests=2, measured_requests=4,
                median_stage_total_ms=_median_stages(observations, 'stage_total_ms'),
                median_layer8_ms=_median_stages(observations, 'layer8_ms'), observations=observations))
            restores.append(dict(route=kind, passed=True, raw_actions_K_V=True,
                                 measured_requests=4, atol=1e-5, rtol=1e-5))
            print(json.dumps(dict(cost_validation='profile_' + kind,
                                  native_requests=counts['native_requests'], passed=True)), flush=True)
            status('profile_' + kind)

        by_kind = {row['route']: row for row in profiles}
        layer8 = by_kind['full']['median_layer8_ms']
        attention_complete = layer8['qkv'] + layer8['attn']
        kv_only = by_kind['A']['median_layer8_ms']['kv']
        ffn = layer8['ffn']
        result = dict(passed=True, **counts, maximum_native_requests=MAX_NATIVE_REQUESTS,
            maximum_eager_vjps=MAX_EAGER_VJPS, started=started, completed=time.time(),
            input=dict(manifest=str(manifest_path), suite=sample['suite'], task=sample['task'],
                       init=sample['init'], side='current', seed=seed),
            protocol=dict(same_observation=True, same_noise=True, native_fast_D=2,
                compensation='actual current residual; older=last, ratio=0.8',
                history_scope='constant observation for costs and numerical checks, not period-five forecast quality',
                timing='synchronized full request wall and native VLM elapsed time; timed Fast calls include compensation',
                pairing='four hot reversed-order B/F pairs per frozen budget',
                profile_scope='CUDA events around native stages, without inter-stage synchronization; instrumentation cost applies'),
            all_keep_matches_old_full=True, captures_preserve_K_V_raw_actions=True,
            selectors=dict(ffn=ffn_selector, block=block_selector), paired_costs=costs,
            profiles=profiles, current_residual_restoration=restores,
            layer8_costs=dict(complete_attention_ms=attention_complete, retained_kv_only_ms=kv_only,
                removable_attention_update_ms=attention_complete-kv_only, ffn_ms=ffn,
                attention_to_ffn_ratio=attention_complete/ffn if ffn > 0 else None,
                original_signed_measurements_preserved=True),
            quality_claim='none; slower-than-anchor candidates are explicitly not accepted as same-budget advantages')
        atomic(out / 'cost_validation.json', result)
        status('complete')
        return result
    except Exception as error:
        atomic(out / 'cost_validation.json', dict(passed=False, **counts,
            maximum_native_requests=MAX_NATIVE_REQUESTS, maximum_eager_vjps=MAX_EAGER_VJPS,
            error=repr(error), started=started, failed=time.time()))
        status('failed')
        raise
    finally:
        for selector in selectors:
            selector.clear()
        gc.collect()
        torch.cuda.empty_cache()
