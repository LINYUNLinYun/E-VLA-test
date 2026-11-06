"""Bounded, same-service pilot mechanism analysis; never an online oracle.

run(backend, manifest, output=None) consumes only each pilot's first 15 saved
inputs. Own-episode Full residuals are read before being updated. Native
Full-VLM/D2 is every action reference. The first three Fulls are scored; the
third Full and its first following Fast are the predeclared mask interventions.
The B4 task1/init19 third Full in each suite owns single-module diagnostics;
missing observations are reported and never replaced according to outcomes.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from preparations.idea1_v3.io import atomic, digest


MAX_REQUESTS = 1000
MAX_INPUTS = 240
MAX_REPLANS = 15


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _physical(actions, controller):
    """Match OSC's clipping and linear scaling of delta-pose commands."""
    low, high = (np.asarray(controller[key], dtype=np.float64) for key in ('input_min', 'input_max'))
    outlow, outhigh = (np.asarray(controller[key], dtype=np.float64) for key in ('output_min', 'output_max'))
    if any(value.shape != (6,) for value in (low, high, outlow, outhigh)) or np.any(high <= low):
        raise ValueError('Expected six-dimensional frozen OSC controller bounds')
    clipped = np.clip(np.asarray(actions, dtype=np.float64)[:, :6], low, high)
    return (clipped - (high + low) / 2) * np.abs((outhigh - outlow) / (high - low)) + (outhigh + outlow) / 2


def action_error(reference_raw, candidate_raw, reference_actions, candidate_actions, controller):
    """Normalized raw coordinates and physical controller-command errors."""
    ref, raw = np.asarray(reference_raw), np.asarray(candidate_raw)
    ra, ca = np.asarray(reference_actions), np.asarray(candidate_actions)
    if (ref.shape != (50, 32) or raw.shape != ref.shape or ra.shape != (50, 7)
            or ca.shape != ra.shape or not all(np.isfinite(v).all() for v in (ref, raw, ra, ca))):
        raise ValueError('Expected finite H50 raw32/actions7 arrays')
    delta = raw.astype(np.float64) - ref.astype(np.float64)
    physical = _physical(ca, controller) - _physical(ra, controller)
    flips = (ra[:, 6] < 0) != (ca[:, 6] < 0)
    result = {}
    for prefix, length in (('first5', 5), ('all50', 50)):
        d, p = delta[:length, :7], physical[:length]
        result[prefix] = dict(raw35_rmse=float(np.sqrt(np.mean(d * d))),
            normalized_raw_dim_rmse=np.sqrt(np.mean(d * d, axis=0)).tolist(),
            raw32_rmse=float(np.sqrt(np.mean(delta[:length] ** 2))),
            decoded_action_dim_rmse=np.sqrt(np.mean((ca[:length] - ra[:length]).astype(np.float64) ** 2, axis=0)).tolist(),
            physical_translation_l2_rmse_m=float(np.sqrt(np.mean(np.sum(p[:, :3] ** 2, axis=1)))),
            physical_translation_dim_rmse_m=np.sqrt(np.mean(p[:, :3] ** 2, axis=0)).tolist(),
            physical_rotation_vector_l2_rmse_rad=float(np.sqrt(np.mean(np.sum(p[:, 3:] ** 2, axis=1)))),
            physical_rotation_dim_rmse_rad=np.sqrt(np.mean(p[:, 3:] ** 2, axis=0)).tolist(),
            gripper_sign_flips=int(flips[:length].sum()), gripper_flip_fraction=float(flips[:length].mean()))
    result['tail_steps6_50_raw7_rmse'] = float(np.sqrt(np.mean(delta[5:, :7] ** 2)))
    result['tail_steps6_50_gripper_sign_flips'] = int(flips[5:].sum())
    result['raw_max_abs'] = float(np.max(np.abs(delta)))
    return result


def _ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + end - 1) / 2
        start = end
    return result


def _correlation(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class _Budget:
    """Append before work, so retries retain consumed requests."""
    def __init__(self, path):
        self.path = Path(path)
        self.rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
        if len(self.rows) > MAX_REQUESTS:
            raise ValueError('Existing offline request ledger exceeds the limit')
        self.stream = self.path.open('a', buffering=1)

    def consume(self, kind, episode, purpose):
        if len(self.rows) >= MAX_REQUESTS:
            raise RuntimeError('Offline total model-request limit reached')
        row = dict(number=len(self.rows) + 1, kind=kind, episode=episode, purpose=purpose, time=time.time())
        self.stream.write(json.dumps(row) + '\n')
        self.rows.append(row)
        return row['number']

    def counts(self):
        return dict(total_model_requests=len(self.rows), native_requests=sum(r['kind'] == 'native' for r in self.rows),
                    selector_eager_forwards=sum(r['kind'] == 'selector' for r in self.rows),
                    selector_warmup_forwards=0, maximum_model_requests=MAX_REQUESTS)

    def close(self):
        self.stream.close()


def _sources(manifest, base):
    episodes = []
    for budget in ('B4', 'B6'):
        for case in manifest['pilot_cases']:
            key = f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"
            directory = base / f'{budget}_F-R+' / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"
            if not (directory / 'complete.json').exists():
                raise ValueError(f'Pilot is not complete: {directory}')
            identity = json.loads((directory / 'identity.json').read_text())
            if identity['case'] != case or identity['manifest_hash'] != manifest['manifest_hash']:
                raise ValueError('Pilot identity differs from the frozen manifest')
            records = [json.loads(line) for line in (directory / 'infer.jsonl').read_text().splitlines()]
            saved = [row for row in records[:MAX_REPLANS] if 'input_path' in row]
            if len(saved) != min(MAX_REPLANS, len(records)):
                raise ValueError('Pilot is missing a required first-fifteen saved input')
            if [int(row['replan']) for row in saved] != list(range(len(saved))):
                raise ValueError('Pilot saved replans are not the original contiguous prefix')
            episodes.append(dict(key=f'{budget}_{key}', budget=budget, case=case,
                                 directory=str(directory), rows=saved, controller=identity['controller'],
                                 diagnostic=budget == 'B4' and case['task'] == 1 and case['init'] == 19,
                                 infer_sha256=_sha(directory / 'infer.jsonl')))
    if len(episodes) != 16 or sum(len(item['rows']) for item in episodes) > MAX_INPUTS:
        raise ValueError('Expected sixteen declared pilot episodes and at most 240 observations')
    return episodes


def run(backend, manifest, output=None):
    """Run after the sixteen pilots, using the existing serialized B1 backend.

    The caller resets online clients before entry and serializes this operation
    with all policy work. Native references/counterfactuals and every eager D10
    VJP count toward the hard 1000-request limit; there are no graph warmups.
    Completed episode analyses resume without replay; interrupted partial work
    remains charged in requests.jsonl. This function never writes online masks,
    policy history, or the experiment's frozen cost table.
    """
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3.common import coefficients
    from preparations.idea1_v3_4_E2_adjacency_200.common import choose
    from preparations.idea1_v3_1.common import lock
    from .common import OUT
    from .model import Selector, forecast_error
    from .selection import mask_record, select_ffn
    base = Path(manifest.get('output_directory', OUT))
    output = Path(output) if output is not None else base / 'offline'
    output.mkdir(parents=True, exist_ok=True)
    episodes = _sources(manifest, base)
    source_identity = dict(manifest_hash=manifest['manifest_hash'],
        sources=[dict(key=e['key'], infer_sha256=e['infer_sha256'],
                      saved_inputs=[dict(replan=r['replan'], input_file_sha256=r['input_file_sha256']) for r in e['rows']]) for e in episodes],
        source_sha256=_sha(__file__))
    protocol = dict(source_identity=source_identity, maximum_model_requests=MAX_REQUESTS,
        maximum_saved_observations=MAX_INPUTS, per_episode_saved_replans=MAX_REPLANS,
        predeclared_targets='third Full, then first saved Fast following that Full',
        predeclared_single_modules='B4/task1/init19 third Full in each suite; all 17 FFN and Attention individually, no outcome-based replacement',
        score='first three Fulls: one raw D10 VJP each, and an error D10 VJP only on third Full; no Fast VJP',
        counterfactual_reference='complete current-observation native VLM plus D2, same seed',
        request_upper_bound_without_retries=832,
        normalization='raw action coordinates are checkpoint-normalized; per-dimension RMSE has no outcome-dependent rescaling',
        physical_units='OSC clipped/scaled delta translation in meters and axis-angle command vector in radians; not rollout end-effector displacement',
        history='own episode, actual Full env steps; read previous banks before appending current Full',
        replay_boundary='error direction is native BF16-state residual difference; VJP is approximate PyTorch Full-D10 replay',
        no_online_oracle=True)
    with lock(output / 'offline.lock', blocking=False):
        frozen = output / 'manifest.json'
        if frozen.exists():
            prior = json.loads(frozen.read_text())
            if prior['source_identity'] != source_identity:
                raise ValueError('Offline sources changed; use a new output revision')
        else:
            atomic(frozen, protocol)
        if (output / 'summary.json').exists():
            return json.loads((output / 'summary.json').read_text())
        budget = _Budget(output / 'requests.jsonl')
        selector = Selector(backend, structure='ffn', use_graph=False)
        episode_reports = []
        try:
            for episode in episodes:
                report_path = output / f"{episode['key']}.json"
                if report_path.exists():
                    episode_reports.append(json.loads(report_path.read_text()))
                    continue
                rows = episode['rows']
                full_rows = [r for r in rows if int(r['replan']) % 5 == 0][:3]
                third = int(full_rows[2]['replan']) if len(full_rows) == 3 else None
                first_fast = next((int(r['replan']) for r in rows if third is not None and int(r['replan']) > third and int(r['replan']) % 5), None)
                targets = {r for r in (third, first_fast) if r is not None}
                arrays, observed, scores, interventions, diagnostics = {}, [], [], [], []
                history = []
                active_masks = {}
                active_score = {}
                k = 4 if episode['budget'] == 'B4' else 6
                frozen_budget = manifest['budgets'][episode['budget']]

                def native(obs, prompt, seed, purpose, **kwargs):
                    number = budget.consume('native', episode['key'], purpose)
                    c, timing = backend.prepare_modules(obs, prompt, **kwargs)
                    actions, raw, cost = backend.sample(c, seed, 2)
                    return c, np.asarray(actions, dtype=np.float32).copy(), raw.float().cpu().numpy().copy(), dict(timing, **cost), number

                for row in rows:
                    index, step = int(row['replan']), int(row['step'])
                    full = index % 5 == 0
                    side = dict(path=row['input_path'], sha256=row['input_file_sha256'], seed=row['seed'], input_hash=row['input_hash'])
                    obs, prompt = load_observation(side)
                    capture = ('attn', 'ffn') if full and episode['diagnostic'] else ('ffn',)
                    c, ref_actions, ref_raw, costs, ref_number = native(obs, prompt, int(row['seed']), f'reference/replan{index}', capture=capture)
                    current = c.pop('residuals')
                    arrays[f'reference_raw_{index}'] = ref_raw
                    arrays[f'reference_actions_{index}'] = ref_actions
                    previous = history[-2] if len(history) == 2 else None
                    last = history[-1] if history else None
                    ratio = 0. if previous is None else (step - last['step']) / (last['step'] - previous['step'])
                    error = None
                    observation = dict(replan=index, step=step, full=full, seed=int(row['seed']),
                        reference_request=ref_number, reference_timing=costs,
                        history_full_steps=[h['step'] for h in history], extrapolation_ratio=ratio)
                    if last is not None:
                        prediction = {}
                        for layer in range(1, 18):
                            now, recent = current['ffn'][layer], last['residuals']['ffn'][layer]
                            estimate = recent if previous is None else recent + ratio * (recent - previous['residuals']['ffn'][layer])
                            prediction[layer] = dict(residual_rms=float(now.square().mean().sqrt()),
                                change_from_last_full_rms=float((now - recent).square().mean().sqrt()),
                                prediction_error_rms=float((now - estimate).square().mean().sqrt()))
                        observation['ffn_prediction'] = prediction
                        if previous is not None:
                            error = forecast_error(current, last['residuals'], previous['residuals'], ratio, structure='ffn')
                    observed.append(observation)
                    if full:
                        modes = ('raw', 'error') if error is not None else ('raw',)
                        full_scores = {}
                        for mode in modes:
                            number = budget.consume('selector', episode['key'], f'{mode}/replan{index}')
                            gradient, score, timing = selector.run(c, int(row['seed']), coefficients(),
                                error=error if mode == 'error' else None, structure='ffn', use_graph=False)
                            full_scores[mode] = np.asarray(score)
                            scores.append(dict(replan=index, step=step, mode=mode, gradient=gradient.tolist(),
                                score=score.tolist(), request=number, timing=timing,
                                source_full_steps=[h['step'] for h in history] + [step]))
                        active_score = full_scores
                        active_masks = {}
                        for mode, score in full_scores.items():
                            active_masks[f'{mode}_same_count'] = choose(score, k, 'spaced')
                            active_masks[f'{mode}_cost'] = select_ffn(score, frozen_budget)
                    if index in targets:
                        memo = {}
                        for name, mask in active_masks.items():
                            for compensate in (False, True):
                                identity = (tuple(mask), compensate)
                                if identity not in memo:
                                    _, actions, raw, timing, number = native(obs, prompt, int(row['seed']),
                                        f'{name}/comp{int(compensate)}/replan{index}', skip_ffn=mask,
                                        history=None if not compensate or last is None else last['residuals'],
                                        older=None if not compensate or previous is None else previous['residuals'], ratio=ratio)
                                    array_key = f'intervention_{len(interventions)}_{index}'
                                    arrays[array_key + '_raw'] = raw
                                    arrays[array_key + '_actions'] = actions
                                    memo[identity] = (number, timing, array_key,
                                        action_error(ref_raw, raw, ref_actions, actions, episode['controller']))
                                number, timing, array_key, metrics = memo[identity]
                                record = dict(replan=index, step=step, method=name, compensated=compensate,
                                    mask=list(mask), skip_count=len(mask), request=number, array_key=array_key,
                                    history_full_steps=[h['step'] for h in history], extrapolation_ratio=ratio,
                                    metrics=metrics, timing=timing)
                                if name.endswith('_cost'):
                                    record['frozen_cost'] = mask_record(mask, frozen_budget)
                                else:
                                    costs_by_layer = {r['layer']: r['ffn_saved_ms'] for r in frozen_budget['measured_layer_costs']}
                                    record['estimated_saved_ms'] = float(sum(costs_by_layer[layer] for layer in mask))
                                interventions.append(record)
                    if episode['diagnostic'] and index == third:
                        for kind in ('ffn', 'attn'):
                            for layer in range(1, 18):
                                for compensate in (False, True):
                                    _, actions, raw, timing, number = native(obs, prompt, int(row['seed']),
                                        f'single/{kind}/L{layer}/comp{int(compensate)}',
                                        skip_ffn=(layer,) if kind == 'ffn' else (),
                                        skip_attn=(layer,) if kind == 'attn' else (),
                                        history=last['residuals'] if compensate else None,
                                        older=previous['residuals'] if compensate else None, ratio=ratio)
                                    array_key = f'single_{kind}_{layer}_{int(compensate)}'
                                    arrays[array_key + '_raw'] = raw
                                    arrays[array_key + '_actions'] = actions
                                    diagnostics.append(dict(kind=kind, layer=layer, compensated=compensate,
                                        replan=index, step=step, request=number, array_key=array_key,
                                        raw_score=float(active_score['raw'][layer - 1]) if kind == 'ffn' else None,
                                        error_score=float(active_score['error'][layer - 1]) if kind == 'ffn' else None,
                                        metrics=action_error(ref_raw, raw, ref_actions, actions, episode['controller']), timing=timing))
                    # Commit only real Full residuals, after forecast/scoring/interventions.
                    if full:
                        history = (history + [dict(step=step, residuals=current)])[-2:]
                    del c, current, error
                correlations = {}
                for mode, compensated in (('raw', False), ('error', True)):
                    subset = [d for d in diagnostics if d['kind'] == 'ffn' and d['compensated'] == compensated]
                    if subset:
                        x = [d[mode + '_score'] for d in subset]
                        y = [d['metrics']['first5']['raw35_rmse'] for d in subset]
                        correlations[mode] = dict(n=len(x), pearson=_correlation(x, y), spearman=_correlation(_ranks(x), _ranks(y)))
                report = dict(key=episode['key'], budget=episode['budget'], case=episode['case'],
                    source_infer_sha256=episode['infer_sha256'], saved_observations=len(rows),
                    targets=sorted(targets), missing_third_full=third is None,
                    missing_first_fast_after_third_full=first_fast is None,
                    single_module_predeclared=episode['diagnostic'],
                    single_module_missing=episode['diagnostic'] and third is None,
                    observations=observed, scores=scores, interventions=interventions,
                    single_modules=diagnostics, single_module_correlations=correlations,
                    action_arrays=f"{episode['key']}.npz", request_counts=budget.counts())
                array_path = output / report['action_arrays']
                temp = array_path.with_name(array_path.name + f'.{os.getpid()}.tmp')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                os.replace(temp, array_path)
                atomic(report_path, report)
                episode_reports.append(report)
                atomic(output / 'status.json', dict(status='running', completed_episodes=len(episode_reports),
                    total_episodes=len(episodes), **budget.counts(), time=time.time()))
            grouped = {}
            for report in episode_reports:
                for item in report['interventions']:
                    key = (report['budget'], item['method'], item['compensated'])
                    grouped.setdefault(key, []).append(item)
            variant_summary = []
            for (budget_id, method, compensated), items in sorted(grouped.items()):
                variant_summary.append(dict(budget=budget_id, method=method, compensated=compensated,
                    n=len(items), skip_counts=sorted(set(item['skip_count'] for item in items)),
                    first5_raw35_rmse_mean=float(np.mean([item['metrics']['first5']['raw35_rmse'] for item in items])),
                    first5_raw35_rmse_median=float(np.median([item['metrics']['first5']['raw35_rmse'] for item in items])),
                    first5_translation_l2_rmse_m_mean=float(np.mean([item['metrics']['first5']['physical_translation_l2_rmse_m'] for item in items])),
                    first5_rotation_vector_l2_rmse_rad_mean=float(np.mean([item['metrics']['first5']['physical_rotation_vector_l2_rmse_rad'] for item in items])),
                    first5_gripper_sign_flips=sum(item['metrics']['first5']['gripper_sign_flips'] for item in items),
                    tail_steps6_50_raw7_rmse_mean=float(np.mean([item['metrics']['tail_steps6_50_raw7_rmse'] for item in items]))))
            summary = dict(status='complete', manifest_hash=manifest['manifest_hash'],
                completed_episodes=len(episode_reports), saved_observations=sum(r['saved_observations'] for r in episode_reports),
                target_observations=sum(len(r['targets']) for r in episode_reports),
                single_module_observations=sum(bool(r['single_modules']) for r in episode_reports),
                missing_single_module_observations=[r['key'] for r in episode_reports if r['single_module_missing']],
                reports=[f"{r['key']}.json" for r in episode_reports], variants=variant_summary,
                single_module_correlations={r['key']: r['single_module_correlations'] for r in episode_reports if r['single_modules']},
                **budget.counts(), protocol=protocol, time=time.time())
            atomic(output / 'summary.json', summary)
            atomic(output / 'status.json', summary)
            return summary
        except Exception:
            import traceback
            atomic(output / 'failure.json', dict(error=traceback.format_exc(), **budget.counts(), time=time.time()))
            raise
        finally:
            selector.clear()
            budget.close()
