"""One bounded, same-input comparison of actual complete Fast requests."""
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_4_E2_module_skip.selection import freeze_budget, select_ffn
from .selection import select_mixed, mask_record


def run(backend, out, budgets, table, observation, prompt, seed, score, histories):
    out = Path(out)
    marker = out / 'cost_validation.json'
    with lock(out / 'cost_validation.lock'):
        if marker.exists():
            result = json.loads(marker.read_text())
            if result.get('passed') is not True:
                raise ValueError('Saved same-input Fast cost check did not pass')
            return result
        count = 0
        measured, comparisons = [], []

        def call(a, f):
            nonlocal count
            if count >= 48:
                raise RuntimeError('Bounded same-input Fast check exceeded 48 calls')
            count += 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            c, timing = backend.prepare_modules(observation, prompt, skip_attn=a, skip_ffn=f,
                history=histories[1], older=histories[0], ratio=.8)
            actions, raw, costs = backend.sample(c, seed, 2)
            torch.cuda.synchronize()
            wall = (time.perf_counter() - start) * 1000
            if not torch.isfinite(raw).all() or not np.isfinite(actions).all():
                raise ValueError('Nonfinite Fast cost-check output')
            receipt = c['module_meta']
            if (receipt['skip_attn'] != list(a) or receipt['skip_ffn'] != list(f)
                    or receipt['patched']['attn'] != list(a) or receipt['patched']['ffn'] != list(f)):
                raise ValueError('Fast cost-check native operator/patch receipt mismatch')
            return dict(request=count, wall_ms=wall, vlm_ms=timing['vlm_ms'],
                        native_timing=dict(timing, **costs))

        for name, budget in budgets.items():
            masks = {'keep': ([], []),
                     'U': select_mixed(score, budget, cost_weighted=False),
                     'W': select_mixed(score, budget, cost_weighted=True)}
            ffnb = freeze_budget(table, name)
            if ffnb.get('nearest_tier'):
                raise ValueError('Fresh FFN comparator has no masks in the exact cost window')
            masks['F'] = ([], select_ffn(np.asarray(score)[:, 1], ffnb))
            for route, (a, f) in masks.items():
                if route != 'keep':
                    mask_record(a, f, budget)
            for _ in range(2):
                for a, f in masks.values():
                    call(a, f)
            rows = []
            for repeat in range(4):
                order = list(masks) if repeat % 2 == 0 else list(masks)[::-1]
                values = {}
                for route in order:
                    a, f = masks[route]
                    value = call(a, f)
                    value.update(budget=name, route=route, repeat=repeat,
                                 order=order, attn=list(a), ffn=list(f))
                    rows.append(value)
                    values[route] = value
                comparisons.append(dict(budget=name, repeat=repeat,
                    W_minus_U_wall_ms=values['W']['wall_ms'] - values['U']['wall_ms'],
                    W_minus_U_vlm_ms=values['W']['vlm_ms'] - values['U']['vlm_ms']))
            for route, (a, f) in masks.items():
                selected = [row for row in rows if row['route'] == route]
                measured.append(dict(budget=name, route=route, attn=list(a), ffn=list(f),
                    mask_cost=None if route == 'keep' else mask_record(a, f, budget),
                    median_wall_ms=float(np.median([r['wall_ms'] for r in selected])),
                    median_vlm_ms=float(np.median([r['vlm_ms'] for r in selected])),
                    samples=selected))
        physical = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
        uuid = subprocess.check_output(['nvidia-smi', f'--id={physical}', '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
        result = dict(passed=True, native_requests=count, maximum_native_requests=48,
            gpu_uuid=uuid, seed=seed, input_scope='first fixed real bank input; current observation and same seed for all routes',
            score_scope='same current E score from the mixed 34-gate Full-D10 replay',
            compensation='same two prior native module residual banks, ratio0.8',
            warmup_requests=16, measured_requests=32, repeats=4,
            routes=measured, paired=comparisons,
            limitations='one input and four balanced-order repeats; actual wall differences are descriptive, not general matched-budget proof',
            time=time.time())
        atomic(marker, result)
        return result
