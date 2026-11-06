"""Freeze a bounded40-case Full calibration before creating any new service."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .common import ROOT, SUITES, LIMITS, PHASE, HEALTH_RULE, atomic_json, digest, sha
from .common import conditions, validate_output, read_manifest

PACKAGE=Path(__file__).resolve().parent
BASE=ROOT/'results/ease_insights_20261103/vlm_within_suite_v1/discovery/manifest.json'
OUTPUT=ROOT/'results/ease_insights_20261103/baseline_recovery_v1/calibration'
REPORT=ROOT/'reports/ease_insights_20261103/baseline_recovery_v1'


def official_assets():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['LIBERO_CONFIG_PATH']=str(ROOT/'policies/realtime-vla-flash/.libero')
    sys.path.insert(0,str(ROOT/'LIBERO'))
    from libero.libero import benchmark,get_libero_path
    from preparations.idea1_v2_1.runtime_client import official_init_states
    import numpy as np
    rows=[]
    for suite in SUITES:
        selected=benchmark.get_benchmark_dict()[suite]()
        for task in range(10):
            definition=selected.get_task(task)
            bddl=Path(get_libero_path('bddl_files'))/definition.problem_folder/definition.bddl_file
            initial=Path(get_libero_path('init_states'))/definition.problem_folder/definition.init_states_file
            vectors=official_init_states(selected,task)
            if len(vectors)<=49:
                raise ValueError('Need init44 calibration and untouched45-49 for '+suite+str(task))
            rows.append(dict(suite=suite,task=task,task_name=definition.name,prompt=definition.language,
                bddl_path=str(bddl),bddl_sha256=sha(bddl),init_file=str(initial),init_file_sha256=sha(initial),
                available=len(vectors),init_sha256={str(i):hashlib.sha256(np.asarray(vectors[i]).tobytes()).hexdigest() for i in range(44,50)}))
    return dict(passed=True,cases=rows,task_count=40,calibration_init=44,
        reserved_future_inits=list(range(45,50)),scope='Official metadata and CPU state vectors; no simulator/model/CUDA context')


def prior_use(assets):
    """Check known EASE records by official physical-task aliases, independent of outcome."""
    catalog=ROOT/'reports/ease_insights_20261103/inputs/libero_task_inventory.csv'
    if not catalog.exists():
        # Resolve the already accepted exact inventory through its frozen manifest.
        base=json.loads(BASE.read_text())
        candidate=json.loads(Path(base['design_evidence']['path']).read_text())
        catalog=Path(candidate['input_inventory'])
    with catalog.open() as stream:
        records=list(csv.DictReader(stream))
    mapping={(r['suite'],int(r['task'])):r for r in records}
    targets={mapping[(r['suite'],r['task'])]['full_bddl_identity'] for r in assets['cases']}
    raw_targets={r['bddl_sha256'] for r in assets['cases']}
    known=ROOT/'results/ease_insights_20261103'
    proc=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','identity.json','-g','result.json',
        '-g','episode_receipt.json',str(known)],capture_output=True,text=True,check=False)
    if proc.returncode not in (0,1) or proc.stderr.strip():
        raise RuntimeError('Known-outcome inventory failed: '+proc.stderr)
    found=[];unresolved=[];inspected=0
    for name in sorted(set(proc.stdout.splitlines())):
        path=Path(name)
        init_parts=[s for s in path.parts if s.startswith('init') and s[4:].isdigit()]
        if init_parts and not any(int(s[4:]) in range(44,50) for s in init_parts):
            continue
        value=json.loads(path.read_text());scene=value.get('case',{})
        if not all(k in scene for k in ('suite','task','init')) or scene['init'] not in range(44,50):
            continue
        inspected+=1;row=mapping.get((scene['suite'],scene['task']))
        if (row and row['full_bddl_identity'] in targets) or value.get('bddl_sha256') in raw_targets:
            found.append(dict(path=name,sha256=sha(path),case=scene))
        elif scene['suite'] in SUITES:
            unresolved.append(dict(path=name,case=scene))
    value=dict(passed=not found and not unresolved,known_root=str(known),catalog=dict(path=str(catalog),sha256=sha(catalog)),
        requested_inits=list(range(44,50)),inventoried_files=len(set(proc.stdout.splitlines())),
        matching_existing_outcomes=found,unresolved=unresolved,inspected_reserved_init_documents=inspected,
        scope='Finite known EASE JSON records and official physical-task aliases; no global absence claim',
        outcome_success_not_used=True,inspected_at=time.time())
    if not value['passed']:
        raise ValueError('Reserved initial states already used or unresolved: '+json.dumps(value))
    return value


def prepare():
    destination=OUTPUT/'manifest.json'
    if destination.exists():
        value=read_manifest(destination,verify_sources=True)
        return dict(reused=True,manifest_hash=value['manifest_hash'],manifest_file_sha256=sha(destination))
    base=json.loads(BASE.read_text())
    # Both new evidence and the new independent case list precede any execution.
    assets=official_assets();unused=prior_use(assets)
    protocol=dict(schema='ease-full-health-calibration-v1',frozen_at=time.time(),status='before_new_outcomes',
        suite_order=list(SUITES),tasks=list(range(10)),inits=[44],condition=conditions()[0],
        total_episodes=40,policy_action_limits=dict(LIMITS),settling_actions=10,environment_seed=7,
        health_rule=HEALTH_RULE,no_task_filtering=True,no_fast_path_inference=True,
        reason='LIBERO90 Full7/72 makes that pool unsuitable for a usable-path search; prior four-suite Full190/200 motivates a fresh compatibility check.',
        inherited_science='Identical Pi0 checkpoint, normalization, H50/R5, FullD10 arithmetic and seed contract; suite budgets restore the accepted common-suite protocol.',
        future_search='Separate protocol and unused initial states; this calibration alone selects no mask or complement hypothesis.')
    protocol_path=REPORT/'protocol.json';atomic_json(protocol_path,protocol)
    pin=dict(path=str(protocol_path),sha256=sha(protocol_path))
    keys=('checkpoints','bank','environment','noise','science','quality_environment_reuse','technical_max_attempts','preflight')
    manifest={k:copy.deepcopy(base[k]) for k in keys}
    manifest['preflight'].pop('cross_service_reference',None)
    manifest['science'].pop('policy_action_limit',None)
    manifest['science'].update(policy_action_limits=dict(LIMITS),settling_actions=10,environment_seed=7,training=False)
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(base['source_files']);sources.update(local)
    pool_plan=ROOT/'results/ease_insights_20261103/resource_pool_v2/plan.json'
    pool=json.loads(pool_plan.read_text());sources.update(pool['source_hashes'])
    manifest.update(revision='ease-common-suite-full-calibration-v1',phase=PHASE,output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=['full_d10'],suites=list(SUITES),tasks=list(range(10)),
        case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],calibration_inits=[44],
        cases_per_condition=40,total_calibration_episodes=40,health_rule=copy.deepcopy(HEALTH_RULE),
        official_init_availability=assets,initial_state_prior_use_audit=unused,protocol_evidence=pin,design_evidence=pin,
        local_runtime_sources=local,source_files=sources,analysis_sources=local,
        legacy_asset_reference=dict(path=str(BASE),sha256=sha(BASE),manifest_hash=base['manifest_hash']),
        resource_execution_revision=dict(name='resource-pool-v2',plan=dict(path=str(pool_plan),sha256=sha(pool_plan)),
            original_GPU_leases_unchanged=True,idle_mib=16384,running_placeholder_mib=8192),
        fixed_input_cost=dict(conditions='Full only',warmup_requests=24,measured_requests=48,
            scope='Finite old bank; descriptive timing only, no acceleration claim'))
    manifest['manifest_hash']=digest(manifest)
    validate_output(OUTPUT,manifest);atomic_json(destination,manifest)
    read_manifest(destination,verify_sources=True)
    return dict(reused=False,manifest_hash=manifest['manifest_hash'],manifest_file_sha256=sha(destination),
        episodes=40,source_files=len(sources),GPU_started=False)


if __name__=='__main__':
    print(json.dumps(prepare()))
