"""Read existing audited V14-D2 logs; no model/GPU imports or reruns."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/root/nfs/embodied/benchmark')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'reports/idea1_libero_plus_d2_final_20261023T1200Z'
index = json.loads((SOURCE / 'summary.json').read_text())
audit = json.loads((SOURCE / 'root_all195_execution_audit.json').read_text())
assert audit['passed'] and index['completed'] == audit['completed'] == 195
bound = {str(Path(i['path']).resolve()): i['sha256']
         for episode in audit['episodes'] for i in episode['inputs']}
fields = ('input_processing_ms', 'vision_ms', 'vlm_ms', 'ae_ms', 'decode_ms')
metrics = (*fields, 'model_event_sum_ms', 'recorded_stage_sum_ms',
           'unitemized_request_wall_ms', 'request_wall_ms')
measurements, receipts = [], []
seen = set()
for entry in index['completed_results']:
    key = (entry['group'], entry['source_index'])
    assert key not in seen
    seen.add(key)
    result_path = Path(entry['path'])
    result_bytes = result_path.read_bytes()
    assert hashlib.sha256(result_bytes).hexdigest() == entry['sha256']
    result = json.loads(result_bytes)
    assert result['mode'] == 'v14_d2' and result['D'] == 2 and result['H'] == 50
    path = result_path.parent / 'infer.jsonl'
    digest = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            response = row['response']
            timing = response['timing']
            assert response['mode'] == 'v14_d2' and response['category'] == 'quality'
            assert timing['actual_NFE'] == 2 and timing['nodes'] == [1.0, 0.5, 0.0]
            assert response['observed_calls']['ae_layer_calls'] == 36
            assert row['request'] == count
            vals = {k: float(timing[k]) for k in (*fields, 'request_wall_ms')}
            assert all(np.isfinite(v) and v >= 0 for v in vals.values())
            vals['model_event_sum_ms'] = sum(vals[k] for k in ('vision_ms', 'vlm_ms', 'ae_ms'))
            vals['recorded_stage_sum_ms'] = sum(vals[k] for k in fields)
            vals['unitemized_request_wall_ms'] = vals['request_wall_ms'] - vals['recorded_stage_sum_ms']
            measurements.append(dict(namespace=entry['namespace'], gpu_index=entry['actual_gpu'],
                gpu_uuid=result['gpu_uuid'], dataset_group=entry['group'], source_index=entry['source_index'],
                request=row['request'], first_in_episode=(count == 0), **vals))
            count += 1
    assert count == result['requests']
    assert digest.hexdigest() == bound[str(path.resolve())]
    receipts.append(dict(namespace=entry['namespace'], source_index=entry['source_index'],
                         path=str(path), sha256=digest.hexdigest(), requests=count))
assert len(seen) == 195 and len(measurements) == audit['requests'] == 14754

groups = defaultdict(list)
for row in measurements:
    scopes = ('all_requests', 'first_in_episode' if row['first_in_episode'] else 'after_first_in_episode')
    for dataset in (row['dataset_group'], 'all_datasets'):
        for scope in scopes:
            groups[(row['namespace'], row['gpu_uuid'], dataset, scope)].append(row)
summary = []
for (namespace, gpu, dataset, scope), values in sorted(groups.items()):
    for metric in metrics:
        arr = np.array([r[metric] for r in values], dtype=np.float64)
        summary.append(dict(namespace=namespace, gpu_uuid=gpu, dataset_group=dataset, scope=scope,
            metric=metric, n=len(arr), mean=float(arr.mean()), p50=float(np.percentile(arr, 50)),
            p95=float(np.percentile(arr, 95)), minimum=float(arr.min()), maximum=float(arr.max())))

for name, rows in [('requests.csv', measurements), ('stage_summary.csv', summary), ('source_receipts.csv', receipts)]:
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

report = dict(status='complete', analysis_only=True, source_index=str(SOURCE / 'summary.json'),
    source_execution_audit=str(SOURCE / 'root_all195_execution_audit.json'),
    source_index_sha256=hashlib.sha256((SOURCE / 'summary.json').read_bytes()).hexdigest(),
    source_logs_verified=len(receipts), episodes=195, requests=14754,
    first_in_episode=sum(r['first_in_episode'] for r in measurements),
    after_first_in_episode=sum(not r['first_in_episode'] for r in measurements),
    method='Per-request statistics, separated by physical GPU and original run namespace; no outcome selection.',
    timing_definitions=dict(input_processing_ms='Host wall from after ensure_prompt to static input staging; excludes ensure_prompt/tokenizer work.',
        vision_ms='CUDA Event span around original vision_encoder; includes dispatch gaps, not kernel busy time.',
        vlm_ms='CUDA Event span around 14-active-layer transformer_encoder.',
        ae_ms='CUDA Event span for both full18-layer AE iterations and Euler updates; not measured separately per iteration.',
        decode_ms='Host wall for raw D2H and output transform.',
        model_event_sum_ms='Per-request sum of vision/vlm/ae Event spans.',
        request_wall_ms='Server policy wall; includes prompt work, input hashing, other setup/copies/validation; excludes full response serialization and client RPC/network.',
        unitemized_request_wall_ms='Per-request request_wall minus five recorded stages; not a directly measured CPU stage or pure compute time.'),
    limitations=['Existing closed-loop observations, not fixed-input exclusive performance experiments.',
        'Removing each episode first request is an explicit descriptive view, not proof every remaining sample is fully warm.',
        'Do not add stage medians to obtain request median; sum computed per request.',
        'No pure-kernel time, AE1/AE2 split or exact attribution of unitemized time without new instrumentation.',
        'No GPU experiments, source changes or changes to the frozen original artifacts.'],
    source_receipts=receipts, summary=summary)
(OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('status', 'episodes', 'requests', 'first_in_episode', 'after_first_in_episode')}))
for namespace in sorted({r['namespace'] for r in measurements}):
    print(namespace)
    for row in summary:
        if row['namespace'] == namespace and row['dataset_group'] == 'all_datasets' and row['scope'] == 'after_first_in_episode':
            print(row['metric'], 'n=', row['n'], 'mean/P50/P95=', *(round(row[k], 3) for k in ('mean', 'p50', 'p95')))
