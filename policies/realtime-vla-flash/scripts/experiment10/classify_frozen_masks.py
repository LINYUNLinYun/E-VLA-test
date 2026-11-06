"""Record whether frozen masks came from legal search or a one-rule challenge."""
import json
from scripts.experiment10.common import RESULTS, dump_json, sha256
from scripts.experiment10.search_rules import Rules


def main():
    root = RESULTS/'search'
    frozen_path = root/'frozen_candidates.json'
    global_path = root/'global_search.json'
    prior_path = RESULTS/'phase_analysis/layer_prior.json'
    frozen, global_result, prior = [json.loads(p.read_text()) for p in
                                    (frozen_path, global_path, prior_path)]
    if not frozen.get('complete') or prior.get('selection_split') != 'search':
        raise RuntimeError('Frozen masks require a complete search-only result and prior')
    low = frozenset(r['layer'] for r in prior['layers'] if r['protected_low_cosine'])
    rules = Rules(low)
    global_origins = {tuple(row['skip_layers']): row['origin'] for row in global_result['candidates']}
    validation_path = root/'validation.json'
    validation = json.loads(validation_path.read_text())

    def describe(mask, phase=None):
        key = tuple(mask)
        violations = sorted(rules.violations(mask))
        origin = global_origins.get(key)
        phase_origin = None
        if phase:
            phase_path = root/('phase_'+phase+'.json')
            phase_result = json.loads(phase_path.read_text())
            phase_origin = next((row['origin'] for row in phase_result['candidates']
                                 if tuple(row['skip_layers']) == key), None)
        if violations and (len(violations) != 1 or 'permanent' in violations):
            category = 'unexpected_multiple_or_permanent_constraint_violations'
        elif violations:
            category = 'one_rule_challenge'
        else:
            category = 'heuristic_legal'
        return {'skip_layers': list(key), 'category': category, 'violations': violations,
                'global_origin': origin, 'phase_origin': phase_origin}

    output = {'complete': True, 'global_candidate_count': len(global_result['candidates']),
              'main_evaluations': global_result.get('actual_main_evaluations'),
              'challenge_evaluations': global_result.get('actual_challenge_evaluations'),
              'challenge_origin_count': sum(x['origin'] == 'one_rule_challenge'
                                            for x in global_result['candidates']),
              'static_masks': [describe(mask) for mask in frozen['static_masks']],
              'validation_candidate_count': len(validation['candidates']),
              'validation_candidates': [describe(row['skip_layers']) for row in validation['candidates']],
              'validation_sha256': sha256(validation_path),
              'phase_masks': {phase: describe(mask, phase)
                              for phase, mask in frozen.get('phase_masks', {}).items()},
              'global_sha256': sha256(global_path), 'frozen_sha256': sha256(frozen_path),
              'prior_sha256': sha256(prior_path),
              'interpretation': 'Challenge-origin winners deliberately violate one heuristic and must be reported as challenge candidates; absolute protections remain enforced.'}
    dump_json(root/'frozen_mask_classification.json', output)


if __name__ == '__main__':
    main()
