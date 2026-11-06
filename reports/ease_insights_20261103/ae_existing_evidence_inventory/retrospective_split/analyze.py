"""One fixed retrospective trial split; CPU only, no experimental inference."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'taskwise/paired_case_outcomes.jsonl'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def choose(rows):
    return 'L2' if sum(r['L2_success'] for r in rows) >= sum(r['L8_success'] for r in rows) else 'L8'


def main():
    raw = SOURCE.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    tasks = defaultdict(list)
    seen = set()
    for row in rows:
        key = row['suite'], row['task_id'], row['trial_index'], row['run_seed']
        assert key not in seen
        seen.add(key)
        assert type(row['L2_success']) is bool and type(row['L8_success']) is bool
        tasks[key[:2]].append(row)
    assert len(rows) == 400 and len(tasks) == 40
    for group in tasks.values():
        assert sorted(r['trial_index'] for r in group) == list(range(10))
        assert len({r['taxonomy_category'] for r in group}) == 1
    protocol = dict(
        status='retrospective exploratory check, specified after aggregate/task summaries were available',
        input_sha256=hashlib.sha256(raw).hexdigest(),
        train_trial_indices=list(range(5)), evaluate_trial_indices=list(range(5, 10)),
        tie_choice='L2', grouping_modes=['fixed_metadata_category', 'task'],
        baseline='single fixed path selected only from the training half; tie L2',
        no_split_search=True, bootstrap_draws=20000, bootstrap_seed=20261105,
        bootstrap_scope='heldout task means conditional on this fitted route; not training-selection uncertainty',
        limitations=['Historical aggregate/task summaries were already inspected; this is not independent confirmation.',
                     'Trial labels and seed match, but physical initial state hashes are absent.',
                     'Only the same 40 tasks are evaluated; not unseen-task generalization.',
                     'No paired Full baseline or measured latency; no adaptive speedup claim.'])
    protocol_path = HERE / 'protocol.json'
    encoded = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
    if protocol_path.exists():
        assert protocol_path.read_text() == encoded, 'Do not alter the split after evaluating it'
    else:
        with protocol_path.open('x') as stream:
            stream.write(encoded)
            stream.flush()
    train = [r for r in rows if r['trial_index'] < 5]
    test = [r for r in rows if r['trial_index'] >= 5]
    fixed = choose(train)
    result = dict(protocol=protocol, train_pairs=len(train), evaluation_pairs=len(test),
                  training_selected_fixed=fixed,
                  train_successes={p: sum(r[p+'_success'] for r in train) for p in ('L2', 'L8')},
                  heldout_successes={p: sum(r[p+'_success'] for r in test) for p in ('L2', 'L8')},
                  routes={})
    for mode in protocol['grouping_modes']:
        def key(row):
            return row['taxonomy_category'] if mode == 'fixed_metadata_category' else (row['suite'], row['task_id'])
        groups = defaultdict(list)
        for row in train:
            groups[key(row)].append(row)
        choices = {group: choose(items) for group, items in groups.items()}
        route = [bool(r[choices[key(r)]+'_success']) for r in test]
        base = [bool(r[fixed+'_success']) for r in test]
        differences = defaultdict(list)
        for row, a, b in zip(test, route, base):
            differences[(row['suite'], row['task_id'])].append(int(a)-int(b))
        means = np.array([np.mean(differences[t]) for t in sorted(differences)])
        rng = np.random.default_rng(protocol['bootstrap_seed'])
        samples = means[rng.integers(len(means), size=(protocol['bootstrap_draws'], len(means)))].mean(axis=1)*100
        result['routes'][mode] = dict(
            choices=[dict(group=list(k) if isinstance(k, tuple) else k, path=v) for k, v in sorted(choices.items())],
            heldout_successes=sum(route), n=len(test), baseline_successes=sum(base),
            difference_pp=100*(sum(route)-sum(base))/len(test),
            route_only=sum(a and not b for a, b in zip(route, base)),
            baseline_only=sum(b and not a for a, b in zip(route, base)),
            conditional_task_cluster_ci95_pp=np.percentile(samples, [2.5, 97.5]).tolist())
    stability = defaultdict(int)
    for group in tasks.values():
        a = sum(int(r['L2_success'])-int(r['L8_success']) for r in group if r['trial_index'] < 5)
        b = sum(int(r['L2_success'])-int(r['L8_success']) for r in group if r['trial_index'] >= 5)
        label = 'same_nonzero_direction' if a*b > 0 else 'opposite_nonzero_direction' if a*b < 0 else 'tie_in_at_least_one_half'
        stability[label] += 1
    result['task_direction_stability'] = dict(stability)
    save(HERE/'analysis.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('protocol', 'routes')}, ensure_ascii=False))
    for mode, values in result['routes'].items():
        print(mode, json.dumps({k: v for k, v in values.items() if k != 'choices'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
