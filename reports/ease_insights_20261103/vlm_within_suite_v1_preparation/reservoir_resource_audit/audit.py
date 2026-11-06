"""Read-only proof for the resource amendment; scientific432 audit stays separate.

--preflight checks source pins, retained old markers and live original/guardian
identities. It reports pending if real16-GiB allocation is absent. Final mode
requires the completed independent432 acceptance report, then binds each new
accepted serving run to its8-GiB reservoir receipt and original GPU lease.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/root/nfs/embodied/benchmark')
sys.path.insert(0, str(ROOT))
from preparations.ease_insights_20261103.reservoir_v1 import guardian as g

RESOURCE = g.RESOURCE_ROOT
REPORT = Path(__file__).resolve().parent
AMENDMENT_SHA = '524357cd305ecde913362f023474465d1067711e9f214970a1729b219d8bca83'
SCIENCE_AUDIT_SHA = '00b5a393f9fa7b175ae9ffe229d78f254e2d917d228157a771c17b2e99514f62'
FULL_AUDIT = REPORT.parent / 'full_discovery_audit/audit.json'
require, read, sha = g.require, g.read, g.sha


def validate_initial(value, gpu_uuid):
    require(value.get('schema') == g.km.SCHEMA and value.get('phase') == 'ready'
            and value.get('sequence') == 0 and value.get('gpu_uuid') == gpu_uuid
            and type(value.get('pid')) is int and value['pid'] > 0
            and str(value.get('proc_start_ticks', '')).isascii()
            and str(value.get('proc_start_ticks', '')).isdigit()
            and int(value['proc_start_ticks']) > 0
            and isinstance(value.get('nonce'), str) and len(value['nonce']) == 32
            and all(c in '0123456789abcdef' for c in value['nonce'])
            and value.get('allocated_bytes') == 16384 * g.km.base.MIB
            and type(value.get('total_memory_bytes')) is int
            and value['total_memory_bytes'] > value['allocated_bytes']
            and type(value.get('driver_version')) is int and value['driver_version'] > 0
            and value.get('context_created') is True and value.get('error') is None,
            'Initial16GiB CUDA ready receipt identity differs')


def validate_revision(identity, result, launch, initial, binding, amendment, original_binding):
    gpu_uuid = launch['gpu_uuid']
    validate_initial(initial, gpu_uuid)
    revision = identity.get('resource_execution_revision', {})
    expected_amendment = dict(path=str(RESOURCE / 'resource_amendment.json'), sha256=AMENDMENT_SHA)
    require(revision.get('schema') == 'ease-reservoir-science-resource-v1'
            and identity.get('controller_pid') == revision.get('controller_pid') == launch['pid']
            and revision.get('controller_source_sha256') == amendment['source_hashes'][
                'preparations/ease_insights_20261103/reservoir_v1/guardian.py']
            and revision.get('reused_borrowed_controller_sha256') == amendment['source_hashes'][
                'preparations/ease_insights_20261103/vlm_within_suite_v1/borrowed_controller.py']
            and revision.get('scientific_manifest_hash') == identity.get('manifest_hash') == g.MANIFEST
            and revision.get('resource_amendment') == expected_amendment
            and revision.get('scientific_runtime_unchanged') is True
            and revision.get('original_holder_resources_unchanged') is True
            and revision.get('original_scheduler_session_owns_lease') is True
            and revision.get('borrower_did_not_take_or_duplicate_lease') is True
            and result.get('resource_execution_revision') == revision,
            'Scientific run resource amendment/source/result identity differs')
    keeper = revision.get('reservoir_keeper', {})
    require(all(keeper.get(k) == initial.get(k) for k in (
                'schema','gpu_uuid','pid','proc_start_ticks','nonce','driver_version','total_memory_bytes'))
            and keeper.get('phase') == 'resized' and type(keeper.get('sequence')) is int
            and keeper['sequence'] > 0 and keeper.get('context_created') is True
            and keeper.get('allocated_bytes') == 8192 * g.km.base.MIB
            and keeper.get('error') is None,
            'Serving run does not bind to the original resized8GiB helper')
    require(binding.get('keeper') == initial and binding.get('gpu_uuid') == gpu_uuid
            and binding.get('guardian', {}).get('pid') == launch['pid']
            and binding.get('guardian', {}).get('start_ticks') == launch['start_ticks']
            and binding.get('guardian', {}).get('argv') == launch['argv'],
            'Initial16GiB context binding is not bound to guardian launch')
    host_pid = binding.get('reservoir_host_pid')
    require(type(host_pid) is int and host_pid > 0, 'Missing reservoir host context PID')
    for value, size in ((binding, 16384), (revision.get('assigned_gpu_consistency', {}), 8192)):
        memory = value.get('host_process_memory_mib', {})
        require(value.get('gpu_uuid') == gpu_uuid and value.get('retained_allocation_mib') == size
                and value.get('reservoir_host_pid') == host_pid
                and value.get('original_keeper_host_pid') == g.HOST_KEEPERS[gpu_uuid]
                and set(memory) == {str(host_pid), str(g.HOST_KEEPERS[gpu_uuid])},
                'Assigned16/8GiB context observation identity differs')
        g.owned_observation(gpu_uuid, [(int(pid), used) for pid, used in memory.items()], host_pid, size)
    original = revision.get('keeper', {})
    proof = revision.get('holder_proof', {})
    require(original == proof.get('keeper_ready') and original.get('allocated_bytes') == 64 * g.km.base.MIB
            and original.get('gpu_uuid') == gpu_uuid and proof.get('remote_lease_fds')
            and proof.get('binding') == original_binding
            and identity.get('global_gpu_lock') == original_binding['lease_path'],
            'Original64MiB keeper/remote scheduler lease proof differs')
    return dict(controller_pid=launch['pid'], helper_pid=initial['pid'], helper_start_ticks=initial['proc_start_ticks'],
                gpu_uuid=gpu_uuid, initial_retained_mib=16384, serving_retained_mib=8192,
                reservoir_host_pid=host_pid, host_PID_binding_scope='unique sized additional-context consistency inference')


def verify_no_lease_copy(pid, original_binding):
    forbidden = (original_binding['lease_device'], original_binding['lease_inode'])
    for path in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        require((stat.st_dev, stat.st_ino) != forbidden, 'New process duplicated the original GPU lease FD')


def run(preflight):
    amendment_path = RESOURCE / 'resource_amendment.json'
    amendment = g.verify_amendment(amendment_path, AMENDMENT_SHA)
    retained = []
    for cid, pin in amendment['retained_completed_conditions'].items():
        require(sha(pin['path']) == pin['sha256'], 'Retained old completion marker changed')
        marker = read(pin['path'])
        require(marker.get('condition') == cid and marker.get('complete') is True
                and marker.get('completed_after') == 72 and marker.get('manifest_hash') == g.MANIFEST,
                'Retained old completion marker identity differs')
        retained.append(dict(condition=cid, sha256=pin['sha256'], cases=72))
    guard = g.bc.HolderGuard()
    current_holders = guard.full()
    observer = g.bc.legacy.identity.LocalRuntime()
    launches = read(RESOURCE / 'launches.json')['processes']
    require(len(launches) == 2 and {p['gpu_uuid'] for p in launches} == set(g.HOST_KEEPERS),
            'Guardian launch matrix differs')
    slots = {}
    for launch in launches:
        gpu_uuid = launch['gpu_uuid']
        actual = observer.identity(launch['pid'])
        require(actual.get('start_ticks') == launch['start_ticks'] and actual.get('argv') == launch['argv']
                and actual.get('state') not in g.bc.legacy.identity.DEAD_STATES
                and actual.get('uid') == os.getuid(), 'Guardian live identity differs')
        slot = RESOURCE / 'sessions' / gpu_uuid
        cpu_receipt = read(slot / 'guardian.json')
        require(cpu_receipt.get('pid') == launch['pid'] and cpu_receipt.get('start_ticks') == launch['start_ticks']
                and cpu_receipt.get('argv') == launch['argv']
                and cpu_receipt.get('amendment_sha256') == AMENDMENT_SHA,
                'Guardian source launch receipt differs')
        original_binding = current_holders['holders'][gpu_uuid]['binding']
        verify_no_lease_copy(launch['pid'], original_binding)
        entry = dict(launch=launch, current_process=actual, initial_allocation=None, context_binding=None)
        if (slot / 'keeper_ready.json').exists():
            initial = read(slot / 'keeper_ready.json')
            validate_initial(initial, gpu_uuid)
            helper = observer.identity(initial['pid'])
            require(helper.get('start_ticks') == str(initial['proc_start_ticks'])
                    and helper.get('ppid') == launch['pid'] and helper.get('uid') == os.getuid()
                    and helper.get('state') not in g.bc.legacy.identity.DEAD_STATES,
                    'Reservoir live helper PID/start/parent differs')
            verify_no_lease_copy(initial['pid'], original_binding)
            entry['initial_allocation'] = initial
            if (slot / 'context_binding.json').exists():
                entry['context_binding'] = read(slot / 'context_binding.json')
        slots[gpu_uuid] = entry
    report = dict(schema='within-suite-reservoir-resource-acceptance-v1', status='pending', time=time.time(),
        preflight=preflight, amendment=dict(path=str(amendment_path), sha256=AMENDMENT_SHA),
        retained_completed_conditions=retained, original_four_markers_for288_cases_unchanged=True,
        sources_match_frozen_amendment=True, original_lease_proof=current_holders, slots=slots,
        scope='Resource amendment only; scientific outcomes and numerics require the independent complete432 audit',
        live_GPU_driver_queried=False, new_model_or_environment_started=False,
        real16GiB_receipts_present=sum(s['initial_allocation'] is not None for s in slots.values()), accepted_new_runs=[])
    if preflight:
        report['status'] = 'resource_receipts_present' if report['real16GiB_receipts_present'] == 2 else 'pending_real_allocation'
        return report
    full = read(FULL_AUDIT)
    require(full.get('status') == 'pass' and full.get('manifest_hash') == g.MANIFEST
            and full.get('audit_script', {}).get('sha256') == SCIENCE_AUDIT_SHA
            and sha(full['audit_script']['path']) == SCIENCE_AUDIT_SHA
            and full.get('scope', {}).get('expected_episodes') == 432
            and full.get('episode_evidence', {}).get('count') == 432
            and full.get('serving_runs', {}).get('passed') is True,
            'Completed scientific432 acceptance is missing or differs')
    for cid in ('local_goal', 'local_long'):
        section = full['serving_runs']['conditions'][cid]
        require(section.get('passed') is True and section.get('runs'), 'No accepted new serving runs')
        for record in section['runs']:
            run_dir = Path(record['path']).resolve()
            require(run_dir.parent == (g.bc.NEW_ROOT / 'discovery' / cid / 'runs').resolve(), 'Run escaped condition')
            for filename in ('identity.json','protocol.json','result.json'):
                require(sha(run_dir / filename) == record['artifact_sha256'][filename],
                        'Scientific acceptance run artifact changed')
            identity, result = read(run_dir / 'identity.json'), read(run_dir / 'result.json')
            gpu_uuid = record['gpu_uuid']
            slot = slots[gpu_uuid]
            proof = validate_revision(identity, result, slot['launch'], slot['initial_allocation'],
                slot['context_binding'], amendment, current_holders['holders'][gpu_uuid]['binding'])
            report['accepted_new_runs'].append(dict(path=str(run_dir), **proof))
    report.update(status='pass', scientific432_acceptance=dict(path=str(FULL_AUDIT), sha256=sha(FULL_AUDIT)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        result = run(args.preflight)
    except Exception as error:
        result = dict(schema='within-suite-reservoir-resource-acceptance-v1', status='failed', error=repr(error), time=time.time())
    result['audit_source_sha256'] = sha(__file__)
    target = REPORT / ('preflight.json' if args.preflight else 'audit.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], report=str(target), error=result.get('error'))))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
