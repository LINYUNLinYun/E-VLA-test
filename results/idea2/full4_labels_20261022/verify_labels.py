"""Verify captured NVTX -> original graph nodes -> replayed GPU kernels."""
import collections
import hashlib
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent
TRACE = Path('nsys/idea2_experiment/cewp_d10_full4_labeled_20261022')
c = sqlite3.connect(str(TRACE.with_suffix('.sqlite')))
c.row_factory = sqlite3.Row
ranges = c.execute('''select n.start,n.end,n.globalTid,coalesce(n.text,s.value) label
    from NVTX_EVENTS n left join StringIds s on n.textId=s.id
    where coalesce(n.text,s.value) like 'Prefill%'
       or coalesce(n.text,s.value) like 'AE/%' ''').fetchall()
nodes = {r['graphNodeId']: r for r in c.execute(
    'select * from CUDA_GRAPH_NODE_EVENTS where originalGraphNodeId is null')}
links = {r['graphNodeId']: r['originalGraphNodeId'] for r in c.execute(
    'select * from CUDA_GRAPH_NODE_EVENTS where originalGraphNodeId is not null')}
labels = {}
for identifier in nodes:
    source = identifier
    seen = set()
    # Instantiation emits both a creation and clone record for the same node.
    # Always follow clone ancestry, even when a creation record is also present.
    while source in links:
        assert source not in seen
        seen.add(source)
        source = links[source]
    node = nodes[source]
    enclosing = [r for r in ranges if r['globalTid'] == node['globalTid']
                 and r['start'] <= node['start'] <= r['end']]
    if enclosing:
        labels[identifier] = min(enclosing, key=lambda r:r['end']-r['start'])['label']
counts = collections.Counter()
streams = collections.defaultdict(collections.Counter)
kernels = list(c.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null'))
for kernel in kernels:
    label = labels[kernel['graphNodeId']]
    counts[label] += 1
    streams[kernel['streamId']][label] += 1
assert counts['Prefill/VisionEncoder'] and counts['Prefill/VLMDecoder']
assert all(counts[f'AE/Denoise/step_{i:02d}'] for i in range(10))

def sequences(database):
    db = sqlite3.connect(database)
    groups = collections.defaultdict(list)
    for row in db.execute('''select correlationId,s.value,gridX,gridY,gridZ,
        blockX,blockY,blockZ,registersPerThread,staticSharedMemory,dynamicSharedMemory
        from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id
        where graphNodeId is not null order by start'''):
        groups[row[0]].append(tuple(row[1:]))
    return {size: {tuple(v) for v in groups.values() if len(v) == size}
            for size in (2309,11166)}

original = sequences('nsys/idea2_experiment/cewp_d10_full.4.sqlite')
current = sequences(str(TRACE.with_suffix('.sqlite')))
sequence_checks = {str(size): original[size] == current[size] and bool(current[size])
                   for size in original}
assert all(sequence_checks.values())
result = json.loads((ROOT/'profile/result.json').read_text())
assert len(result['rows']) == 5 and all(g['correct'] for g in result['groups'])
assert all(r['steady_recapture'] == 0 and r['pool']['live_snapshots'] <= 2
           and r['normalized_error']['max_abs'] == r['decoded_error']['max_abs'] == 0
           for r in result['rows'])
receipt = dict(passed=True, report=str(TRACE.with_suffix('.nsys-rep')),
    original_report='nsys/idea2_experiment/cewp_d10_full.4.nsys-rep',
    graph_kernels=len(kernels), all_graph_kernels_labeled=sum(counts.values()) == len(kernels),
    phase_kernel_counts=counts, stream_labels={k: dict(v) for k,v in streams.items()},
    original_kernel_sequence_and_launch_geometry_equal=sequence_checks,
    prefill_kernels_per_replay=2309, ae_kernels_per_replay=11166,
    numerical_max_abs=0, measured_requests=5, steady_recaptures=0,
    profile_only=True, split_graph=False, extra_cuda_sync=False,
    collection_mode='node:nvtx-precapture', nsys_version='2026.2.1',
    scope='New label diagnostic on shared GPU3; not a new latency/resource-contention benchmark. Original full.4 is preserved for performance comparison.',
    viewing='Open with Nsight Systems 2026.2; focus cewp/continuous/graph_ae_high/None/1. Expand CUDA HW GPU NVTX projection. Prefill stream 13, AE stream 14. Prefill/VisionEncoder repeats for three camera inputs, followed by Prefill/VLMDecoder.',
    source_sha256=hashlib.sha256(Path('preparations/idea2/cewp_profile_labels.py').read_bytes()).hexdigest(),
    official_reference='https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cuda-graph-trace')
(ROOT/'receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
