"""Read-only live evidence plus draft artifacts; never applies or launches anything."""
from pathlib import Path
import collections
import copy
import hashlib
import json
import os
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/root/nfs/embodied/benchmark')
sys.path.insert(0, str(ROOT))
R = ROOT / 'results/ease_insights_20261103'
OUT = Path(__file__).resolve().parent
DISC = R / 'vlm_local_search_v1/discovery'
IDS = [
    'vlm_local_v1_discovery_full_d10_all120__wait_reassign_5f9688a613c6__gate_policy_5s2s_aa673ea2c916',
    'vlm_local_v1_discovery_local_spatial_all120__gate_policy_5s2s_ece0430b8b3a',
]
RUNS = [
    DISC / 'full_d10/runs/discovery_shard00of01_1793818662106423845',
    DISC / 'local_spatial/runs/discovery_shard00of01_1793818662086250240',
]

def sha(b):
    return hashlib.sha256(b).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_bytes())

from preparations.ease_insights_20261103.pipeline import proc_identity
from preparations.ease_insights_20261103.vlm_local_search_v1.selection import load_stage
from preparations.ease_insights_20261103.vlm_local_search_v1.supervisor import Paths, Supervisor

started = time.time()
archive = []
inputs = [R / n for n in [
    'queue.json', 'scheduler/status.json', 'scheduler/jobs_state.json',
    'scheduler/controller_process.json',
    'vlm_local_search_v1/supervisor/state.json',
    'vlm_local_search_v1/supervisor/status.json',
    'vlm_local_search_v1/supervisor/supervisor_process.json',
    'vlm_local_search_v1/supervisor/daemon.log',
    'gate_opportunity_coordinator/state.json',
    'gate_opportunity_coordinator/status.json',
    'gate_opportunity_coordinator/process.json',
    'gate_opportunity_coordinator/daemon.log',
    'gate_opportunity_coordinator/events.jsonl',
]]
inputs.extend(R / 'scheduler' / (job + '.log') for job in IDS)
inputs.extend(R / 'scheduler/launch_gates' / (job + '.json') for job in IDS)
inputs.append(DISC / 'manifest.json')
inputs.extend(path for run in RUNS for path in sorted(run.iterdir()) if path.is_file())
for src in inputs:
    data = src.read_bytes()
    dst = OUT / 'archive' / src.relative_to(R)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError('Refusing to overwrite earlier evidence: ' + str(dst))
    dst.write_bytes(data)
    archive.append(dict(source=str(src), copy=str(dst), sha256=sha(data), bytes=len(data), copied_unix=time.time()))
write(OUT / 'archive_manifest.json', archive)

queue = read(OUT / 'archive/queue.json')
states = read(OUT / 'archive/scheduler/jobs_state.json')
sup = read(OUT / 'archive/vlm_local_search_v1/supervisor/state.json')
coor = read(OUT / 'archive/gate_opportunity_coordinator/state.json')

known = [335208, 338645, 415450, 432881, 432890]
processes = [proc_identity(pid) for pid in known]
exact_service_matches = []
all_local_runtime_matches = []
matching_schedulers = []
modules = {'preparations.ease_insights_20261103.vlm_local_search_v1.' + x for x in ('runner', 'server', 'worker')}
from preparations.ease_insights_20261103.pipeline import scheduler_match
for entry in Path('/proc').iterdir():
    if not entry.name.isdecimal():
        continue
    p = proc_identity(int(entry.name))
    if p['status'] not in ('live', 'unknown'):
        continue
    argv = p.get('argv', [])
    if any(module in argv for module in modules):
        all_local_runtime_matches.append(p)
    if any(str(run) in argv for run in RUNS):
        exact_service_matches.append(p)
    if p['status'] == 'live' and scheduler_match(p, R / 'queue.json', R / 'scheduler'):
        matching_schedulers.append(p)
proc = dict(observed_unix=time.time(), known_processes=processes,
    exact_failed_service_matches=exact_service_matches, all_local_runtime_matches=all_local_runtime_matches,
    matching_live_schedulers=matching_schedulers,
    source='local /proc only; no GPU query and no host PID signal or process mutation')
write(OUT / 'process_observation.json', proc)

assert len(matching_schedulers) == 1 and matching_schedulers[0]['pid'] == 335208
assert all(p['status'] in ('absent', 'dead') for p in processes if p['pid'] != 335208)
assert not exact_service_matches
assert all(states[j]['status'] == 'failed' for j in IDS)
assert sup['phase'] == 'needs_attention' and sup['actions'] == {}
assert coor['phase'] == 'supervisor_terminal' and coor['pending'] is None
print('Archived live evidence; old job, supervisor and coordinator processes are absent/dead.', flush=True)

stage = load_stage(DISC, 'discovery')
counts = {}
for cid in stage['manifest']['planned_conditions']:
    rows = [r for (condition, _), r in stage['records'].items() if condition == cid]
    counts[cid] = dict(valid=len(rows), success=sum(r['success'] for r in rows),
        valid_policy_failure=sum(not r['success'] for r in rows),
        missing=sum(r['condition'] == cid for r in stage['missing']),
        invalid=sum(r.get('condition') == cid for r in stage['invalid']))
strict = dict(loader='preparations.ease_insights_20261103.vlm_local_search_v1.selection.load_stage',
    sources_verified=True, scientific_manifest_hash=stage['manifest']['manifest_hash'],
    expected=stage['expected'], valid=len(stage['records']), complete=stage['complete'],
    conditions=counts, invalid=stage['invalid'], unexpected=stage['unexpected'],
    missing=stage['missing'], episode_evidence=stage['episode_evidence'],
    started_unix=started, finished_unix=time.time(),
    interpretation='120/720 is incomplete discovery; all valid successes and policy failures are retained. The two startup OOM conditions each have zero valid cases and are eligible only for technical retry.')
write(OUT / 'strict_stage_audit.json', strict)
assert strict['valid'] == 120 and strict['expected'] == 720
assert not strict['invalid'] and not strict['unexpected']
assert counts['jac_anchor']['valid'] == 120
assert all(counts[c]['valid'] == 0 and counts[c]['missing'] == 120 for c in ('full_d10', 'local_spatial'))
print('Strict loader verified:', json.dumps(counts), flush=True)

before = (OUT / 'archive/queue.json').read_bytes()
(OUT / 'queue.before.json').write_bytes(before)
proposed = copy.deepcopy(queue)
diffs = []
for index, job in enumerate(proposed['jobs']):
    if job['id'] not in IDS:
        continue
    old = copy.deepcopy(job)
    new_id = job['id'] + '__startup_retry1'
    assert new_id not in states
    job['id'] = new_id
    flag = job['argv'].index('--status-file') + 1
    job['argv'][flag] = str(R / 'scheduler/launch_gates' / (new_id + '.json'))
    assert old['argv'][old['argv'].index('--') + 1:] == job['argv'][job['argv'].index('--') + 1:]
    assert all(old[k] == job[k] for k in old if k not in ('id', 'argv'))
    assert not Path(job['argv'][flag]).exists()
    diffs.append(dict(queue_job_index=index, old_id=old['id'], new_id=new_id,
        changes=[dict(field='id', before=old['id'], after=new_id),
                 dict(field=f'argv[{flag}] (--status-file)', before=old['argv'][flag], after=job['argv'][flag])],
        invariant_fields={k: v for k, v in old.items() if k not in ('id', 'argv')},
        invariant_inner_argv=job['argv'][job['argv'].index('--') + 1:]))
assert len(diffs) == 2
assert len({j['id'] for j in proposed['jobs']}) == len(proposed['jobs'])
assert all(set(j.get('depends_on', [])).issubset({x['id'] for x in proposed['jobs']}) for j in proposed['jobs'])
write(OUT / 'queue.proposed.json', proposed)
write(OUT / 'queue.field_diff.json', diffs)

sup_proposed = copy.deepcopy(sup)
sup_proposed['phase'] = 'waiting_discovery'
sup_removed = {key: sup_proposed.pop(key) for key in ('error', 'attention_unix') if key in sup_proposed}
write(OUT / 'supervisor.state.proposed.json', sup_proposed)
write(OUT / 'supervisor.state.field_diff.json', dict(phase=dict(before=sup['phase'], after='waiting_discovery'),
    removed=sup_removed, preserved='All other fields, including scheduler identity, actions, gate and creation timestamps'))
coor_proposed = copy.deepcopy(coor)
coor_removed = coor_proposed.pop('controllers')
write(OUT / 'coordinator.state.proposed.json', coor_proposed)
write(OUT / 'coordinator.state.field_diff.json', dict(removed=dict(controllers=coor_removed),
    preserved='schema, pending=null, cooldown_until, sequence and phase; phase supervisor_terminal is not sticky in Coordinator.step'))

# A pure in-memory completion-gate call reads science/markers but never ticks, saves or ensures a scheduler.
supervisor = Supervisor(Paths.defaults(DISC), scheduler_pid=335208)
gate = supervisor.completion_gate(DISC, 'discovery', proposed, states)
assert gate['complete'] is False and gate['completed'] == 1 and gate['expected'] == 6
write(OUT / 'proposed_supervisor_gate.json', gate)
source_paths = [ROOT / p for p in [
    'preparations/ease_insights_20261103/vlm_local_search_v1/supervisor.py',
    'preparations/ease_insights_20261103/vlm_local_search_v1/selection.py',
    'preparations/ease_insights_20261103/pipeline.py',
    'preparations/ease_insights_20261103/scheduler.py',
    'preparations/ease_insights_20261103/gate_opportunity_coordinator.py',
    'preparations/ease_insights_20261103/reassign_waiting_gate_v2.py']]
write(OUT / 'source_hashes.json', {str(p): sha(p.read_bytes()) for p in source_paths})
receipt = dict(mode='prepared_not_applied', prepared_unix=time.time(), directory=str(OUT),
    queue_before_sha256=sha(before), queue_proposed_sha256=sha((OUT / 'queue.proposed.json').read_bytes()),
    failed_job_ids=IDS, retry_job_ids=[j['new_id'] for j in diffs],
    valid=120, expected=720, science_complete=False,
    resource_env_decision='Proposed queue preserves EASE_GPU_GATE_IDLE_SECONDS=5 and EASE_GPU_GATE_POLL_SECONDS=2; root must resolve startup resource diagnosis before application.',
    live_allocator_unchanged=dict(pid=335208, start_ticks=matching_schedulers[0]['start_ticks']),
    no_live_mutations=True, no_signals=True, no_gpu_queries=True,
    live_queue_still_equals_archived=(R / 'queue.json').read_bytes() == before)
write(OUT / 'preparation_receipt.json', receipt)
assert receipt['live_queue_still_equals_archived']
print(json.dumps(receipt, indent=2), flush=True)
