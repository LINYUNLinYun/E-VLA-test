"""Same eight cases, different KV semantics: quality smoke, never a timing ranking."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / 'reports/idea2_d10_dynamic_sm_20261024'
fresh_dir = ROOT / 'closedloop_d10_fresh_gpu0'
old_dir = ROOT / 'closedloop_d10_o0_o1'
states = {label: json.loads((path / 'execution.json').read_text())
          for label, path in [('fresh', fresh_dir), ('old', old_dir)]}
for state in states.values():
    assert state['status'] == 'passed' and state['source_unchanged']
    assert all(case['status'] == 'valid' for case in state['cases'])
assert len(states['fresh']['cases']) == 8 and len(states['old']['cases']) == 16
cases = {}
for label, state in states.items():
    for case in state['cases']:
        key = (case['suite'], case['task'], case['init'])
        cases.setdefault(key, {})[case['label']] = case
rows = []
for (suite, task, init), arms in cases.items():
    assert set(arms) == {'F0', 'O0', 'O1'}
    first = {}
    for label, case in arms.items():
        directory = fresh_dir if label == 'F0' else old_dir
        first[label] = json.loads((directory / case['relative_directory'] / 'infer.jsonl').read_text().splitlines()[0])
    identical_initial_input = len({x['input_hash'] for x in first.values()}) == 1
    identical_seed = len({x['seed'] for x in first.values()}) == 1
    assert identical_initial_input and identical_seed
    assert all(x['current_observation_id'] == x['prefix_observation_id'] for x in first.values())
    a = np.asarray(first['F0']['chunk_actions'])
    b = np.asarray(first['O0']['chunk_actions'])
    row = dict(suite=suite, task=task, init=init,
               identical_initial_input=identical_initial_input, identical_initial_seed=identical_seed,
               bootstrap_action_max_abs=float(np.abs(a-b).max()),
               bootstrap_action_mae=float(np.abs(a-b).mean()))
    for label, case in arms.items():
        row[label+'_success'] = case['result']['success']
        row[label+'_steps'] = case['result']['steps']
    rows.append(row)
summary = {label: dict(episodes=8, successes=sum(x[label+'_success'] for x in rows),
                      total_steps=sum(x[label+'_steps'] for x in rows)) for label in ('F0','O0','O1')}
result = dict(status='passed', summary=summary, cases=rows,
              scope='Eight fixed cases only; fresh current KV versus old complete KV. Not a full LIBERO quality estimate.',
              limitations=['Different GPUs: no absolute latency ranking.',
                           'After bootstrap KV semantics differ and trajectories diverge. Do not require same future observations or claim numerical equivalence.',
                           'Common initial inputs, original model seed construction, H10/D10/R5 and protocol are preserved.',
                           'No robot real-time/SLO claim: LIBERO pauses during inference.'],
              bootstrap_within_tolerance=all(x['bootstrap_action_max_abs'] <= 1e-5 and x['bootstrap_action_mae'] <= 1e-6 for x in rows))
with (OUT/'closedloop_fresh_vs_old.csv').open('w') as f:
    writer=csv.DictWriter(f,list(rows[0])); writer.writeheader(); writer.writerows(rows)
(OUT/'closedloop_fresh_vs_old.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
