"""Post-hoc CPU-only U/W choices on each stored mixed Full score.

This does not run a model. Each saved score/history/input is held fixed while
both frozen objectives reselect from the same exact candidate set. Warm raw
scores and later error scores are always reported separately. Run once on a
single --episode for development, then once with --require-complete after the
160-episode matrix and its audit are complete.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from preparations.idea1_v3.io import digest
from preparations.idea1_v3_4_E2_mixed_cost.selection import (
    mask_record, objective_scores, select_mixed,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'results/idea1_v3_4_E2_mixed_cost'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(values)) if values else None


def selection_terms(scores, budget, attn, ffn):
    density = objective_scores(scores, budget, cost_weighted=True)
    raw = math.fsum([float(scores[layer - 1, 0]) for layer in attn] +
                    [float(scores[layer - 1, 1]) for layer in ffn])
    weighted = math.fsum([float(density[layer - 1, 0]) for layer in attn] +
                         [float(density[layer - 1, 1]) for layer in ffn])
    if not math.isfinite(weighted):
        raise ValueError('Selected mask has an undefined standalone-cost normalizer')
    return raw, weighted


def analyze_episode(path, cfg, manifest, first_load):
    result = json.loads((path / 'result.json').read_text())
    if (not (path / 'complete.json').exists() or not result.get('valid')
            or result['manifest_hash'] != manifest['manifest_hash']
            or result['condition'] != cfg['id']):
        raise ValueError(f'Not a completed frozen mixed episode: {path}')
    requests = [json.loads(line) for line in (path / 'infer.jsonl').read_text().splitlines() if line.strip()]
    if len(requests) != result['predictions']:
        raise ValueError('Episode request count differs from its completion record')
    budget_id = cfg['budget']
    budget = manifest['budgets'][budget_id]
    rows = []
    with np.load(path / 'arrays.npz', allow_pickle=False) as arrays:
        for request in requests:
            route = request['route']
            if route['path'] != 'full':
                continue
            index = int(request['replan'])
            mode = route['scoring_mode']
            if mode != ('warm_original' if index < 10 else 'comp_error'):
                raise ValueError('Unexpected scoring phase in the saved Full request')
            scores = np.asarray(arrays[f'module_score_{index}'], dtype=np.float64)
            gradient = arrays[f'gate_gradient_{index}']
            if scores.shape != (17, 2) or not np.array_equal(scores, np.abs(gradient)):
                raise ValueError('Saved score is not the absolute 34-gate gradient')
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite saved module score')
            if budget_id not in first_load:
                tick = time.perf_counter()
                select_mixed(scores, budget, cost_weighted=False)
                first_load[budget_id] = (time.perf_counter() - tick) * 1000
            episode_id = f"{cfg['id']}/{result['case']['suite']}/task{result['case']['task']:02d}/init{result['case']['init']:02d}"
            row = dict(source_condition=cfg['id'], budget=budget_id, source_cost_weighted=bool(cfg['cost_weighted']),
                episode_id=episode_id, **result['case'], episode_success=bool(result['success']),
                replan=index, step=int(request['step']), scoring_mode=mode,
                input_hash=request.get('input_hash'),
                score_sha256=hashlib.sha256(scores.tobytes()).hexdigest(),
                score_source_full_steps=json.dumps(route['score_source_full_steps']),
                online_mask_selection_ms=float(request['timing']['mask_selection_ms']))
            selections = {}
            for weighted, name in ((False, 'U'), (True, 'W')):
                tick = time.perf_counter()
                attn, ffn = select_mixed(scores, budget, cost_weighted=weighted)
                elapsed = (time.perf_counter() - tick) * 1000
                record = mask_record(attn, ffn, budget)
                raw, density = selection_terms(scores, budget, attn, ffn)
                support, af = set(attn) | set(ffn), set(attn) & set(ffn)
                selections[name] = (attn, ffn)
                row.update({name + '_attn': json.dumps(attn), name + '_ffn': json.dumps(ffn),
                    name + '_attn_count': len(attn), name + '_ffn_count': len(ffn), name + '_af_count': len(af),
                    name + '_a_only_count': len(set(attn) - set(ffn)), name + '_f_only_count': len(set(ffn) - set(attn)),
                    name + '_support_count': len(support), name + '_module_count': len(attn) + len(ffn),
                    name + '_raw_sensitivity': raw, name + '_density_objective': density,
                    name + '_estimated_saved_ms': record['saved_ms'], name + '_shadow_selection_cpu_ms': elapsed})
            actual = selections['W' if cfg['cost_weighted'] else 'U']
            if (actual[0] != route['selected_attn'] or actual[1] != route['selected_ffn']
                    or route['budget_hash'] != budget['budget_hash']):
                raise ValueError('Stored actual policy choice differs from exact frozen replay')
            stored_objective = f'selection_score_{index}'
            if stored_objective in arrays and not np.array_equal(arrays[stored_objective],
                    objective_scores(scores, budget, cost_weighted=cfg['cost_weighted'])):
                raise ValueError('Stored objective coefficients differ from frozen U/W definition')
            ua, uf = map(set, selections['U'])
            wa, wf = map(set, selections['W'])
            row.update(masks_differ=selections['U'] != selections['W'], support_differs=(ua | uf) != (wa | wf),
                attn_symmetric_difference=len(ua ^ wa), ffn_symmetric_difference=len(uf ^ wf),
                W_minus_U_raw_sensitivity=row['W_raw_sensitivity'] - row['U_raw_sensitivity'],
                W_minus_U_density_objective=row['W_density_objective'] - row['U_density_objective'],
                W_minus_U_estimated_saved_ms=row['W_estimated_saved_ms'] - row['U_estimated_saved_ms'])
            raw_scale = max(1., row['U_raw_sensitivity'], row['W_raw_sensitivity'])
            density_scale = max(1., row['U_density_objective'], row['W_density_objective'])
            if row['W_minus_U_raw_sensitivity'] < -1e-12 * raw_scale:
                raise AssertionError('U did not minimize its raw objective')
            if row['W_minus_U_density_objective'] > 1e-12 * density_scale:
                raise AssertionError('W did not minimize its density objective')
            rows.append(row)
    return rows


def summarize(rows, group_fields):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in group_fields)].append(row)
    output = []
    for key, records in sorted(grouped.items()):
        episodes = collections.defaultdict(list)
        for row in records:
            episodes[row['episode_id']].append(row['masks_differ'])
        result = dict(zip(group_fields, key))
        result.update(full_inputs=len(records), source_episodes=len(episodes),
            different_masks=sum(row['masks_differ'] for row in records),
            mask_difference_fraction=mean([row['masks_differ'] for row in records]),
            episode_equal_weighted_mask_difference_fraction=mean([np.mean(values) for values in episodes.values()]),
            support_difference_fraction=mean([row['support_differs'] for row in records]),
            mean_attn_symmetric_difference=mean([row['attn_symmetric_difference'] for row in records]),
            mean_ffn_symmetric_difference=mean([row['ffn_symmetric_difference'] for row in records]),
            mean_online_mask_selection_ms=mean([row['online_mask_selection_ms'] for row in records]))
        for name in ('U', 'W'):
            result[name] = {field: mean([row[name + '_' + field] for row in records])
                for field in ('attn_count', 'ffn_count', 'af_count', 'a_only_count', 'f_only_count',
                              'support_count', 'module_count', 'raw_sensitivity', 'density_objective',
                              'estimated_saved_ms', 'shadow_selection_cpu_ms')}
        for field in ('W_minus_U_raw_sensitivity', 'W_minus_U_density_objective', 'W_minus_U_estimated_saved_ms'):
            result['mean_' + field] = mean([row[field] for row in records])
        output.append(result)
    return output


def run(source=DEFAULT_SOURCE, output=None, episode=None, require_complete=False):
    source = Path(source).resolve()
    output = Path(output).resolve() if output else source / 'shadow_selection'
    if require_complete and episode:
        raise ValueError('A single-episode development probe cannot be a complete matrix analysis')
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    scientific = dict(manifest)
    expected_hash = scientific.pop('manifest_hash')
    if digest(scientific) != expected_hash:
        raise ValueError('Frozen mixed manifest hash mismatch')
    selection_path = ROOT / 'preparations/idea1_v3_4_E2_mixed_cost/selection.py'
    relative = str(selection_path.relative_to(ROOT))
    if sha(selection_path) != manifest['source_files'][relative]:
        raise ValueError('Current selection source differs from frozen source')
    if require_complete:
        audit = json.loads((source / 'audit.json').read_text())
        if not audit.get('passed') or not audit.get('matrix_complete') or not (source / 'complete.json').exists():
            raise ValueError('Formal matrix has not completed and passed its audit')
    cfgs = {cfg['id']: cfg for cfg in manifest['conditions']}
    selected, missing = [], []
    if episode:
        path = Path(episode).resolve()
        cfg = cfgs[json.loads((path / 'result.json').read_text())['condition']]
        selected.append((cfg, path))
    else:
        for cfg in manifest['conditions']:
            for case in manifest['small_cases']:
                path = source / cfg['id'] / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"
                if not (path / 'complete.json').exists():
                    missing.append(dict(condition=cfg['id'], case=case))
                else:
                    selected.append((cfg, path))
    if require_complete and (missing or len(selected) != 160):
        raise ValueError('Expected all 160 completed frozen episode records')
    first_load, rows = {}, []
    started = time.perf_counter()
    for cfg, path in selected:
        rows.extend(analyze_episode(path, cfg, manifest, first_load))
    result = dict(passed=True, formal_matrix_complete=bool(require_complete), single_episode_development=bool(episode),
        source=str(source), manifest_hash=manifest['manifest_hash'], source_episodes=len(selected), full_inputs=len(rows),
        missing_episodes=missing, cpu_analysis_wall_s=time.perf_counter()-started,
        candidate_first_load_and_U_selection_ms=first_load,
        scoring_phases=['warm_original', 'comp_error'],
        by_budget_and_phase=summarize(rows, ('budget', 'scoring_mode')),
        by_source_condition_and_phase=summarize(rows, ('source_condition', 'budget', 'scoring_mode')),
        interpretation=dict(
            comparison='For every saved Full score, exact U and W replay use the identical scores and frozen legal candidate set.',
            phase='W also weights the first two original-score Fulls; later E-only agreement does not imply matching policy prefixes.',
            score='One positive VJP sensitivity is a proxy, not measured action damage; U/W objective inequalities hold by construction.',
            trajectories='Source U/W policies generate different free trajectories. Summary inputs are pooled with explicit source-condition strata.',
            weighting='Main fractions weight saved Full requests; episode-equal fractions are also provided. Common-prefix duplicates are not deduplicated.',
            time='Shadow timings measure this CPU post-hoc process; online_mask_selection_ms is the actual stored policy timing. No GPU or model was run.',
            quality='No success-rate, causal quality, noninferiority, or action-counterfactual claim follows from mask or score differences.'),
        provenance=dict(manifest_path=str(manifest_path), manifest_sha256=sha(manifest_path),
            analysis_script_sha256=sha(Path(__file__)), frozen_selection_sha256=sha(selection_path)))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if rows:
        with (output / 'full_inputs.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(passed=True, source_episodes=len(selected), full_inputs=len(rows),
                         formal_matrix_complete=bool(require_complete), output=str(output))), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--episode', type=Path)
    parser.add_argument('--require-complete', action='store_true')
    arguments = parser.parse_args()
    run(arguments.source, arguments.output, arguments.episode, arguments.require_complete)
