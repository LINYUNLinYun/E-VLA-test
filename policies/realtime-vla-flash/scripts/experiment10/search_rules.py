"""CPU-only heuristic constraints and Pareto beam selection; never scores masks by cosine alone."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


@dataclass(frozen=True)
class Rules:
    low_cosine: frozenset
    permanently_protected: frozenset = frozenset((1, 17, 18))

    def violations(self, mask):
        mask = frozenset(mask)
        if len(mask) == 0:
            return set()
        if any(l < 1 or l > 18 for l in mask) or mask & self.permanently_protected:
            return {'permanent'}
        result = set()
        if 2 in mask:
            result.add('start_l2')
        if any(l + 1 in mask for l in mask):
            result.add('adjacent')
        if mask & self.low_cosine:
            result.add('low_cosine')
        return result

    def allowed(self, mask, *, challenge=False):
        violations = self.violations(mask)
        if not violations:
            return True
        if not challenge or len(violations) != 1 or 'permanent' in violations:
            return False
        mask = frozenset(mask)
        if 'adjacent' in violations:
            return sum(l + 1 in mask for l in mask) == 1
        if 'low_cosine' in violations:
            return len(mask & self.low_cosine) == 1
        return True

    def legal_masks(self, count):
        """Enumerate only legal independent sets on CPU; no model forwards/full parameter enumeration."""
        candidates = [l for l in range(3, 17) if l not in self.low_cosine]
        def visit(start, chosen):
            if len(chosen) == count:
                yield tuple(chosen)
                return
            for index in range(start, len(candidates)):
                layer = candidates[index]
                if not chosen or layer > chosen[-1] + 1:
                    yield from visit(index + 1, chosen + [layer])
        if count < 0:
            raise ValueError('Deletion count must be nonnegative')
        yield from visit(0, [])

    def nearest_v13(self, maximum=8):
        reference = {3, 4, 7, 12, 15}
        legal = list(self.legal_masks(5))
        if not legal:
            return []
        distance = min(len(set(m) ^ reference) for m in legal)
        return [m for m in legal if len(set(m) ^ reference) == distance][:maximum]


def pareto_select(scored, width=8):
    """Minimization of all recorded objectives, nondominated fronts then crowding distance."""
    if not scored:
        return []
    masks = sorted(scored)
    objectives = np.asarray([scored[m] for m in masks], dtype=np.float64)
    if objectives.ndim != 2 or not np.isfinite(objectives).all():
        raise ValueError('Pareto objectives must be finite vectors')
    remaining = list(range(len(masks)))
    selected = []
    while remaining and len(selected) < width:
        front = [i for i in remaining if not any(
            np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i])
            for j in remaining if j != i)]
        if len(front) <= width - len(selected):
            selected.extend(front)
        else:
            distance = {i: 0. for i in front}
            for column in range(objectives.shape[1]):
                order = sorted(front, key=lambda i: (objectives[i, column], masks[i]))
                span = objectives[order[-1], column] - objectives[order[0], column]
                if span == 0:
                    continue
                distance[order[0]] = distance[order[-1]] = float('inf')
                for k in range(1, len(order) - 1):
                    distance[order[k]] += (objectives[order[k+1], column] - objectives[order[k-1], column]) / span
            selected.extend(sorted(front, key=lambda i: (-distance[i], masks[i]))[:width-len(selected)])
        remaining = [i for i in remaining if i not in front]
    return [masks[i] for i in selected]


def heuristic_search(rules, evaluate, importance, *, budget=1000, beam_width=8, seed=20261009, evaluate_batch=None):
    """Actual combination evaluations; fixed 90/10 budgets, references excluded from allocation.

    evaluate(mask) must return measured finite objective vectors on search-only
    observations. There is deliberately no synthetic/additive single-layer evaluator.
    """
    rng = np.random.default_rng(seed)
    scores, origins, infeasible = {}, {}, []
    main_budget = int(.9 * budget)
    challenge_budget = budget - main_budget
    main_count = 0
    def measure(mask, origin):
        nonlocal main_count
        mask = tuple(sorted(mask))
        if mask in scores or main_count >= main_budget:
            return False
        if not rules.allowed(mask):
            return False
        value = tuple(float(v) for v in evaluate(mask))
        if not value or not np.isfinite(value).all():
            raise ValueError('Invalid actual mask score')
        scores[mask], origins[mask] = value, origin
        main_count += 1
        return True
    def measure_many(masks, origin):
        nonlocal main_count
        masks = list(dict.fromkeys(tuple(sorted(m)) for m in masks))
        masks = [m for m in masks if m not in scores and rules.allowed(m)][:max(0, main_budget-main_count)]
        if evaluate_batch is None:
            return any([measure(m, origin) for m in masks])
        evaluated = evaluate_batch(masks)
        if set(evaluated) != set(masks):
            raise ValueError('Parallel evaluation coverage mismatch')
        for mask in masks:
            value = tuple(float(v) for v in evaluated[mask])
            if not value or not np.isfinite(value).all():
                raise ValueError('Invalid parallel combination score')
            scores[mask], origins[mask] = value, origin
        main_count += len(masks)
        return bool(masks)
    beam = [()]
    by_count = {}
    for count in range(1, 7):
        legal = list(rules.legal_masks(count))
        if not legal:
            infeasible.append(count)
            continue
        proposals = {tuple(sorted((*parent, layer))) for parent in beam for layer in range(3, 17)
                     if layer not in parent and rules.allowed((*parent, layer))}
        ranked = sorted(proposals, key=lambda m: (sum(importance.get(l, float('inf')) for l in m), m))
        # Importance orders proposals; actual measured combination scores choose the beam.
        random_legal = [legal[i] for i in rng.permutation(len(legal))[:8]]
        shortlist = ranked[:120] + random_legal
        if count == 5:
            shortlist = rules.nearest_v13() + shortlist
        shortlist = list(dict.fromkeys(shortlist))[:128]
        measure_many(shortlist, 'beam/random/v13_repair')
        current = {m: v for m, v in scores.items() if len(m) == count}
        beam = pareto_select(current, beam_width)
        by_count[count] = beam
        if not beam:
            break
    # Same-count local exchanges, no adjacency/low-cosine constraint relaxation here.
    for _ in range(3):
        changed = False
        for count, parents in list(by_count.items()):
            proposals = []
            for parent in parents:
                for restored in parent:
                    for deleted in range(3, 17):
                        if deleted not in parent:
                            proposals.append((*[l for l in parent if l != restored], deleted))
            changed |= measure_many(proposals, 'local_swap')
            by_count[count] = pareto_select({m: v for m, v in scores.items() if len(m) == count}, beam_width)
        if not changed or main_count >= main_budget:
            break
    challenges = []
    for count in range(2, 7):
        for mask in combinations(range(2, 17), count):
            if rules.violations(mask) and rules.allowed(mask, challenge=True):
                challenges.append(mask)
    selected_challenges = [challenges[i] for i in rng.permutation(len(challenges))[:challenge_budget]]
    evaluated = evaluate_batch(selected_challenges) if evaluate_batch else {m: evaluate(m) for m in selected_challenges}
    if set(evaluated) != set(selected_challenges):
        raise ValueError('Challenge evaluation coverage mismatch')
    for mask, raw_value in evaluated.items():
        value = tuple(float(v) for v in raw_value)
        if not value or not np.isfinite(value).all():
            raise ValueError('Invalid challenge score')
        scores[mask], origins[mask] = value, 'one_rule_challenge'
    return {'scores': scores, 'origins': origins, 'by_count': by_count,
            'infeasible_main_counts': infeasible, 'actual_main_evaluations': main_count,
            'actual_challenge_evaluations': min(len(challenges), challenge_budget)}
