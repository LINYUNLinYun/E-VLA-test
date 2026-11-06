"""Prepare an outcome-free task-pool proposal; never creates runnable jobs."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / 'task_inventory.csv'
NAMESPACE = 'ease-libero90-state-placement-metadata-proposal-v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    raw = INVENTORY.read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    pool = [r for r in rows if r['suite'] == 'libero_90']
    assert len(pool) == 90 and {int(r['task']) for r in pool} == set(range(90))
    states = [r for r in pool if r['category'] == 'state_change_only']
    placement = [r for r in pool if r['category'] == 'single_object_placement']
    assert len(states) == 12 and len(placement) == 69
    def rank(row):
        return digest((NAMESPACE+'|'+row['suite']+'|'+row['task_name']).encode())
    selected_placement = sorted(placement, key=lambda r: (rank(r), int(r['task'])))[:12]
    selected = sorted(states+selected_placement, key=lambda r: (r['category'], int(r['task'])))
    assert len(selected) == len({r['full_bddl_identity'] for r in selected}) == 24
    tasks = []
    for row in selected:
        item = dict(row)
        item['task'] = int(item['task'])
        item['selection_digest'] = rank(row)
        tasks.append(item)
    proposal = dict(
        schema=NAMESPACE, status='metadata_only_proposal_not_runnable_not_enqueued',
        input_inventory=str(INVENTORY.resolve()), input_sha256=digest(raw),
        selection='All12 state-only tasks plus the first12 of69 single-placement tasks ordered by sha256(namespace|suite|task_name); no outcomes are inputs.',
        classes={'state_change_only':12, 'single_object_placement':12},
        total_tasks=24, tasks=tasks,
        illustrative_budget_not_execution_plan=dict(
            discovery_if_six_conditions_three_inits=6*24*3,
            confirmation_if_full_and_two_paths_five_inits=3*24*5,
            total_if_both_stages_run=6*24*3+3*24*5),
        scientific_configuration_frozen=False, initial_state_indices=None,
        required_before_execution=[
            'Finish and report the current frozen VLM local-search branch; this proposal does not change its stopping rule.',
            'Freeze a separate scientific protocol, numerical gates, conditions/budget, selection and heldout confirmation rules.',
            'Audit prior case usage and choose new discovery/confirmation initial states; file labels alone do not prove physical state equality.',
            'Implement and validate the LIBERO90 runner/step-limit/input-bank path; metadata CPU validity is not render/runtime validity.',
            'Extend resource execution at a safe boundary while retaining owned GPU occupancy; current v3 plan only admits its fixed stage outputs.'],
        limitations=[
            'Task names/full BDDL identities are distinct; statistical independence is not implied.',
            'Classes describe instructions/goals, not measured difficulty or path preference.',
            'One suite reduces suite confounding, but assets/scenes/task composition remain possible confounders.',
            'No success rate, path complementarity, latency or adaptive benefit is established.'])
    encoded = json.dumps(proposal, ensure_ascii=False, indent=2)+'\n'
    output = HERE/'candidate.json'
    if output.exists():
        assert output.read_text() == encoded, 'Metadata proposal changed; use a new revision'
    else:
        with output.open('x') as stream:
            stream.write(encoded)
    print(json.dumps({'status':proposal['status'], 'classes':proposal['classes'],
                      'candidate_sha256':digest(encoded.encode()),
                      'selected_task_ids':[r['task'] for r in tasks]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
