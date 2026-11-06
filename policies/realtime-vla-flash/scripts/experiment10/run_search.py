"""Launch bounded real-model heuristic search only after four complete phase probes."""
from __future__ import annotations

import argparse
import json

from scripts.experiment10.common import RESULTS, PHASES, SUITES, dump_json, sha256
from scripts.experiment10.search_rules import Rules, heuristic_search, pareto_select
from scripts.experiment10.search_eval import GpuPool, load_entries


def serialize(result):
    return {**{k: v for k, v in result.items() if k not in ('scores', 'origins')},
            'candidates': [{'skip_layers': m, 'objectives': v, 'origin': result['origins'][m]}
                           for m, v in sorted(result['scores'].items())]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    prior_path = RESULTS / 'phase_analysis/layer_prior.json'
    prior = json.loads(prior_path.read_text())
    if not prior['complete_four_suites'] or prior['selection_split'] != 'search':
        raise RuntimeError('Only search-only, four-suite completed priors are allowed')
    low = frozenset(r['layer'] for r in prior['layers'] if r['protected_low_cosine'])
    rules = Rules(low)
    importance = {r['layer']: r['single_layer_task_mean_rmse'] for r in prior['layers']}
    feasibility = {str(k): len(list(rules.legal_masks(k))) for k in range(2, 7)}
    print(json.dumps({'legal_mask_counts': feasibility, 'low_cosine_protected': sorted(low),
                      'global_budget': 1000, 'phase_budget': 256}, indent=2), flush=True)
    if not args.launch:
        return
    pool = GpuPool()
    output = RESULTS / 'search'
    output.mkdir(parents=True, exist_ok=True)
    try:
        batch = lambda masks: pool.evaluate(masks)
        global_result = heuristic_search(rules, lambda m: batch([m])[m], importance,
                                         budget=1000, evaluate_batch=batch)
        dump_json(output / 'global_search.json', {**serialize(global_result), 'prior_sha256': sha256(prior_path),
                                                 'legal_mask_counts': feasibility})
        scores = global_result['scores']
        chosen = []
        for count in range(2, 7):
            chosen.extend(pareto_select({m: v for m, v in scores.items() if len(m) == count}, width=2))
        chosen = list(dict.fromkeys(chosen + pareto_select(scores, width=16)))[:16]
        # Keep V13 and V16 references outside the 90/10 proposal allocation.
        references = [(3, 4, 7, 12, 15), (9, 10)]
        refined = pool.evaluate(list(dict.fromkeys(chosen + references)), coarse=False)
        dump_json(output / 'refined_search.json', {'selection_split': 'search', 'prior_sha256': sha256(prior_path),
                    'candidates': [{'skip_layers': m, 'objectives': v} for m, v in sorted(refined.items())],
                    'references': references})
        validation_masks = []
        for count in range(2, 7):
            validation_masks.extend(pareto_select({m: v for m, v in refined.items()
                                                   if len(m) == count and m not in references}, width=2))
        validation = pool.evaluate(list(dict.fromkeys(validation_masks + references)), role='validation', coarse=False)
        dump_json(output / 'validation.json', {'role': 'validation', 'candidates': [
                  {'skip_layers': m, 'objectives': v} for m, v in sorted(validation.items())]})
        frozen = pareto_select({m: v for m, v in validation.items() if m not in references}, width=2)
        phase_masks, skipped = {}, []
        coverage = {(r['suite'], r['phase']): r['tasks'] for r in prior['phase_task_coverage']}
        for phase in PHASES:
            if any(coverage.get((suite, phase), 0) < 3 for suite in SUITES):
                skipped.append({'phase': phase, 'reason': 'fewer_than_three_search_tasks_in_a_suite'})
                continue
            validation_coverage = {s: {e['task_id'] for e in load_entries('validation', phase=phase) if e['suite'] == s}
                                   for s in SUITES}
            if any(not v for v in validation_coverage.values()):
                skipped.append({'phase': phase, 'reason': 'no_validation_task_in_a_suite'})
                continue
            phase_batch = lambda masks, p=phase: pool.evaluate(masks, phase=p)
            phase_prior = prior['phase_layers'][phase]
            phase_rules = Rules(frozenset(r['layer'] for r in phase_prior if r['protected_low_cosine']))
            phase_importance = {r['layer']: r['single_layer_task_mean_rmse'] for r in phase_prior}
            result = heuristic_search(phase_rules, lambda m: phase_batch([m])[m], phase_importance,
                                       budget=256, evaluate_batch=phase_batch)
            dump_json(output / ('phase_' + phase + '.json'), serialize(result))
            frontier = pareto_select(result['scores'], width=4)
            verified = pool.evaluate(frontier, role='validation', coarse=False, phase=phase)
            if verified:
                phase_masks[phase] = pareto_select(verified, width=1)[0]
        dump_json(output / 'frozen_candidates.json', {'complete': True, 'prior_sha256': sha256(prior_path),
                  'static_masks': frozen, 'phase_masks': phase_masks, 'skipped_phases': skipped,
                  'validation_basis': 'offline actions; closed-loop recovery screening still required',
                  'evaluation_tasks_not_used': True, 'references': references})
    finally:
        pool.close()


if __name__ == '__main__':
    main()
