"""Compare the prespecified paired smoke without pretending it is a full eval."""
from pathlib import Path
import csv
import json
import numpy as np

root=Path(__file__).resolve().parent
dest=root/'closedloop_d10_o0_o1'
state=json.loads((dest/'execution.json').read_text())
assert state['status']=='passed' and state['source_unchanged']
assert len(state['cases'])==16 and all(c['status']=='valid' for c in state['cases'])
pairs={}
rows=[]
for c in state['cases']:
    key=(c['suite'],c['task'],c['init'])
    pairs.setdefault(key,{})[c['label']]=c
    result=c['result']
    rows.append(dict(label=c['label'],suite=c['suite'],task=c['task'],init=c['init'],
                     success=result['success'],steps=result['steps'],predictions=result['predictions'],
                     cold_capture_requests=result['cold_capture_requests'],
                     all_work_wall_s=result['all_work_wall_s'],drain_wall_s=result['drain_wall_s'],
                     steady_rpc_n=result['steady_oldkv_roundtrip_ms']['n'],
                     steady_rpc_mean_ms=result['steady_oldkv_roundtrip_ms']['mean'],
                     steady_rpc_p95_ms=result['steady_oldkv_roundtrip_ms']['p95']))
pair_rows=[]
for key,arms in pairs.items():
    assert set(arms)=={'O0','O1'}
    logs={label:[json.loads(line) for line in (dest/c['relative_directory']/'infer.jsonl').read_text().splitlines()]
          for label,c in arms.items()}
    prefix_inputs={label:{x['current_observation_id']:x['input_hash'] for x in log}
                   for label,log in logs.items()}
    matched=[]
    for a,b in zip(logs['O0'],logs['O1']):
        same_prefix=(prefix_inputs['O0'][a['prefix_observation_id']]
                     ==prefix_inputs['O1'][b['prefix_observation_id']])
        if same_prefix and a['input_hash']==b['input_hash'] and a['seed']==b['seed']:
            aa,bb=np.array(a['chunk_actions']),np.array(b['chunk_actions'])
            matched.append(dict(replan=a['replan'],max_abs=float(np.abs(aa-bb).max()),
                                mae=float(np.abs(aa-bb).mean()),bitwise=bool(np.array_equal(aa,bb))))
    same_sequence=(len(logs['O0'])==len(logs['O1'])==len(matched))
    pair_rows.append(dict(suite=key[0],task=key[1],init=key[2],
                          O0_success=arms['O0']['result']['success'],O1_success=arms['O1']['result']['success'],
                          O0_steps=arms['O0']['result']['steps'],O1_steps=arms['O1']['result']['steps'],
                          O0_requests=len(logs['O0']),O1_requests=len(logs['O1']),
                          identical_input_prefix_noise_sequence=same_sequence,matched_requests=len(matched),
                          matched_action_max_abs=max((x['max_abs'] for x in matched),default=None),
                          matched_action_max_mae=max((x['mae'] for x in matched),default=None),
                          matched_actions_bitwise=bool(matched) and all(x['bitwise'] for x in matched)))
summary={label:dict(episodes=8,successes=sum(x['success'] for x in rows if x['label']==label),
                    total_steps=sum(x['steps'] for x in rows if x['label']==label),
                    all_work_wall_s=sum(x['all_work_wall_s'] for x in rows if x['label']==label),
                    cold_capture_requests=sum(x['cold_capture_requests'] for x in rows if x['label']==label))
         for label in ['O0','O1']}
result=dict(status='passed',scope='Eight fixed cases per mode; old complete KV only. Not full LIBERO evaluation, not fresh-KV equivalence, not real-time robot control or camera SLO.',
            timing_caveat='Episode wall includes environment startup/rendering/RPC/I/O/drain and asymmetric first cold capture. Do not use it as a formal isolated inference speedup.',
            summary=summary,pairs=pair_rows,
            all_matched_actions_within_tolerance=all(x['matched_action_max_abs'] is not None and x['matched_action_max_abs']<=1e-5 and x['matched_action_max_mae']<=1e-6 for x in pair_rows))
out=root.parents[2]/'reports/idea2_d10_dynamic_sm_20261024'
for name,data in [('closedloop_episodes.csv',rows),('closedloop_pairs.csv',pair_rows)]:
    with (out/name).open('w') as f:
        w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
(out/'closedloop_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
