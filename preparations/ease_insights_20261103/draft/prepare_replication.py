"""Freeze independent init21-24 validation only after the complete 360-case discovery."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import ROOT, CHECKPOINTS, ARTIFACTS, SUITES, PROTOCOL, atomic_json, case_key

DEFAULT_ROOT = ROOT / 'results/ease_insights_20261103'
HOLDOUT_INITS = (21, 22, 23, 24)
RUNTIME_FILES = ('common.py', 'run.py', 'server.py', 'client.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def identity(mode, source, case):
    return (mode, source, case['suite'], int(case['task_id']), int(case['init_id']))


def key_text(key):
    return '|'.join(map(str, key))


def full_source_from_master(master):
    jobs = [name for name in master['jobs'] if name.startswith('discovery_full_')]
    if len(jobs) != 1 or jobs[0][len('discovery_full_'):] not in SUITES:
        raise ValueError('Expected one explicit Full source artifact in frozen discovery jobs')
    return jobs[0][len('discovery_full_'):]


def expected_keys(full_source):
    cases = [dict(suite=s, task_id=t, init_id=20) for s in SUITES for t in range(10)]
    return {identity(mode, source, case) for mode in ('flash', 'draft') for source in SUITES for case in cases} | {
        identity('full', full_source, case) for case in cases}


def validate_complete_keys(keys, full_source):
    expected = expected_keys(full_source)
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate discovery scientific condition')
    actual = set(keys)
    return dict(expected=360, complete=len(actual),
                missing=sorted(key_text(x) for x in expected-actual),
                unexpected=sorted(key_text(x) for x in actual-expected))


def find_runs(roots):
    found = set()
    for root in map(Path, roots):
        if (root / 'cases.json').is_file() and (root / 'manifest.json').is_file():
            found.add(root.resolve())
        elif root.exists():
            for path in root.rglob('cases.json'):
                if (path.parent / 'manifest.json').is_file():
                    found.add(path.parent.resolve())
    return sorted(found)


def holdout_inventory(roots):
    """Read planning metadata and file names only; never read held-out outcomes."""
    found = []
    for run in find_runs(roots):
        metadata = json.loads((run / 'manifest.json').read_text())
        if metadata.get('mode') not in ('flash', 'draft', 'full') or metadata.get('source_suite') not in SUITES:
            continue
        rows = json.loads((run / 'cases.json').read_text())
        holdouts = [r for r in rows if int(r.get('init_id', -1)) in HOLDOUT_INITS]
        if not holdouts:
            continue
        recorded = []
        for case in holdouts:
            folder = run / 'episodes' / case_key(case)
            # Existence is sufficient to flag preregistration contamination.
            artifacts = [str(folder / name) for name in ('result.json', 'first_policy_observation.npz', 'initial_state.npz')
                         if (folder / name).exists()]
            if artifacts:
                recorded.append(dict(case=case, existing_files=artifacts))
        request_file = run / 'requests.jsonl'
        requests_exist = request_file.is_file() and request_file.stat().st_size > 0
        found.append(dict(run=str(run), planned_cases=len(holdouts), recorded_cases=recorded,
                          request_data_exists=requests_exist,
                          outcomes_read=False, evidence_exists=bool(recorded or requests_exist)))
    return found


def audit_discovery(master_path, roots):
    master_path = Path(master_path)
    master = json.loads(master_path.read_text())
    if (tuple(master['source_suites']) != SUITES or tuple(master['target_suites']) != SUITES or
        master['tasks'] != list(range(10)) or master['inits'] != [20] or master['expected_episodes'] != 360 or
        master['flash_episodes'] != 160 or master['draft_episodes'] != 160 or master['full_episodes'] != 40 or
        master['reused_smoke_episodes'] != 8):
        raise ValueError('Frozen discovery matrix differs from the requested design')
    frozen_hashes = {Path(p).name: h for p,h in master['source_sha256'].items()}
    if set(frozen_hashes) != set(RUNTIME_FILES):
        raise ValueError('Frozen runtime file set is incomplete')
    for name, expected in frozen_hashes.items():
        if sha(HERE/name) != expected:
            raise ValueError('Current runtime differs from discovery freeze: ' + name)
    full_source = full_source_from_master(master)
    rows, issues, files, planned, shard_metadata = [], [], [], [], []
    files.append(dict(path=str(master_path.resolve()), sha256=sha(master_path)))
    reused = 0
    for run in find_runs(roots):
        manifest_path, cases_path = run/'manifest.json', run/'cases.json'
        m = json.loads(manifest_path.read_text())
        if m.get('mode') not in ('flash', 'draft', 'full'):
            raise ValueError('Unexpected run mode: ' + str(run))
        mode, source = m['mode'], m['source_suite']
        if source not in SUITES or (mode == 'full' and source != full_source):
            raise ValueError('Unexpected source checkpoint: ' + str(run))
        cases = json.loads(cases_path.read_text())
        if cases != m['cases'] or len({case_key(c) for c in cases}) != len(cases):
            raise ValueError('Manifest/cases inconsistency or duplicate case: ' + str(run))
        if any(c['suite'] not in SUITES or c['task_id'] not in range(10) or c['init_id'] != 20 for c in cases):
            raise ValueError('Non-discovery case in selection inputs; no outcomes read: ' + str(run))
        if m['protocol'] != PROTOCOL:
            raise ValueError('Protocol mismatch in shard: ' + str(run))
        actual_hashes = {Path(p).name: h for p,h in m['code_sha256'].items() if Path(p).name in RUNTIME_FILES}
        if actual_hashes != frozen_hashes:
            raise ValueError('Runtime revision mismatch in shard: ' + str(run))
        if Path(m['base_checkpoint']).resolve() != (CHECKPOINTS/'base/base_weights.pkl').resolve():
            raise ValueError('Changed Pi0 target in ' + str(run))
        if Path(m['draft_checkpoint']).resolve() != (CHECKPOINTS/ARTIFACTS[source]/'draft_triton.pkl').resolve():
            raise ValueError('Changed draft source in ' + str(run))
        actual_targets = list(dict.fromkeys(c['suite'] for c in cases))
        if m['target_suites'] != actual_targets:
            raise ValueError('Target suite list differs from cases: ' + str(run))
        for path in (manifest_path,cases_path):
            files.append(dict(path=str(path),sha256=sha(path)))
        run_receipt_path = run/'result.json'
        run_complete = False
        if run_receipt_path.exists():
            rr = json.loads(run_receipt_path.read_text())
            files.append(dict(path=str(run_receipt_path),sha256=sha(run_receipt_path)))
            run_complete = rr.get('status') == 'complete'
            if run_complete and (rr.get('cases') != len(cases) or rr.get('total_cases') != len(cases) or
                rr.get('mode') != mode or rr.get('source_suite') != source or rr.get('client_exit_code') != 0):
                raise ValueError('Invalid complete run receipt: ' + str(run))
        if not run_complete:
            issues.append(dict(run=str(run), issue='run_not_complete'))
        shard_metadata.append(dict(run=str(run), mode=mode, source_suite=source, n=len(cases),
                                   protocol_hash=canonical_hash(m['protocol']),runtime_hashes=actual_hashes,
                                   run_complete=run_complete))
        run_rows=[]
        for case in cases:
            key=identity(mode,source,case)
            planned.append(key)
            receipt_path=run/'episodes'/case_key(case)/'result.json'
            if not receipt_path.exists():
                issues.append(dict(key=key_text(key),issue='case_result_missing'));continue
            receipt=json.loads(receipt_path.read_text())
            if receipt.get('status')!='complete':
                issues.append(dict(key=key_text(key),issue='case_not_valid_complete'));continue
            if identity(receipt['mode'],receipt['source_suite'],receipt)!=key:
                raise ValueError('Receipt scientific identity mismatch: '+str(receipt_path))
            episode=receipt['episode']
            if ((episode['task_suite_name'],episode['task_id'],episode['episode_idx']) != (case['suite'],case['task_id'],20) or
                episode['failure_reason'] is not None or not isinstance(receipt['success'],bool) or
                episode['success'] != receipt['success'] or episode['seed']!=7 or episode['num_steps_wait']!=10 or
                episode['max_steps']!=PROTOCOL['max_steps'][case['suite']]):
                raise ValueError('Official episode protocol/identity mismatch: '+str(receipt_path))
            native_manifest_path=run/'episodes'/case_key(case)/'native/manifest.json'
            nm=json.loads(native_manifest_path.read_text())
            if (nm['task_suite_name']!=case['suite'] or nm['selected_task_ids']!=[case['task_id']] or
                nm['episode_index_start']!=20 or nm['episode_index_end']!=21 or nm['seed']!=7 or
                nm['num_steps_wait']!=10 or nm['replan_steps']!=12 or nm['resize_size']!=224):
                raise ValueError('Client manifest differs from frozen protocol: '+str(native_manifest_path))
            evidence=dict(key=key_text(key),mode=mode,source_suite=source,target_suite=case['suite'],
                          task_id=case['task_id'],init_id=20,success=receipt['success'],result=str(receipt_path))
            rows.append(evidence);run_rows.append(evidence)
            if run.parent.name=='draft_smoke': reused+=1
            for path in (receipt_path,native_manifest_path):
                files.append(dict(path=str(path),sha256=sha(path)))
        if run_complete and rr['successes'] != sum(r['success'] for r in run_rows):
            raise ValueError('Run aggregate success disagrees with case receipts: '+str(run))
    if len(planned)!=len(set(planned)):
        raise ValueError('Duplicate planned scientific condition across smoke/discovery shards')
    completeness=validate_complete_keys([tuple(r['key'].split('|')[:3])+ (r['task_id'],r['init_id']) for r in rows],full_source)
    if completeness['unexpected']:
        raise ValueError('Unexpected scientific condition in discovery')
    complete=not completeness['missing'] and not issues and reused==8 and len(planned)==360
    if len(rows)==360 and reused!=8:
        issues.append(dict(issue='expected_eight_reused_smoke_cases',actual=reused))
    report=dict(status='complete' if complete else 'incomplete',**completeness,issues=issues,
                reused_smoke_cases=reused,planned_conditions=len(planned),full_source_suite=full_source,
                shards=shard_metadata,discovery_manifest=str(master_path.resolve()),
                discovery_manifest_sha256=sha(master_path), evidence_files=files,
                evidence_sha256=canonical_hash(sorted(files,key=lambda r:r['path'])),
                no_holdout_outcomes_read=True)
    return report,rows,master


def comparisons_and_selection(rows):
    index={(r['mode'],r['source_suite'],r['target_suite'],r['task_id']):int(r['success']) for r in rows}
    comparisons=[]
    for mode in ('draft','flash'):
        for target in SUITES:
            matched=[index[(mode,target,target,t)] for t in range(10)]
            for source in SUITES:
                if source==target:continue
                foreign=[index[(mode,source,target,t)] for t in range(10)]
                comparisons.append(dict(mode=mode,target_suite=target,matched_source_suite=target,
                    foreign_source_suite=source,n=10,matched_successes=sum(matched),foreign_successes=sum(foreign),
                    matched_only=sum(a and not b for a,b in zip(matched,foreign)),
                    foreign_only=sum(b and not a for a,b in zip(matched,foreign)),
                    matched_minus_foreign_count=sum(matched)-sum(foreign),
                    matched_minus_foreign_rate=(sum(matched)-sum(foreign))/10,
                    p_value=None,p_value_reason='Discovery selection; inferential p-values reserved for independent confirmation'))
    ranked=sorted((r for r in comparisons if r['mode']=='draft'),
        key=lambda r:(-r['matched_minus_foreign_count'],SUITES.index(r['target_suite']),SUITES.index(r['foreign_source_suite'])))
    selected=[]
    for row in ranked:
        if row['matched_minus_foreign_count']<=0:continue
        if row['target_suite'] in {s['target_suite'] for s in selected}:continue
        selected.append(row)
        if len(selected)==2:break
    return comparisons,ranked,selected


def prepare_jobs(selected,output,run_output_root,full_source):
    jobs=[]
    for pair in selected:
        target,foreign=pair['target_suite'],pair['foreign_source_suite']
        cases=[dict(suite=target,task_id=t,init_id=i) for t in range(10) for i in HOLDOUT_INITS]
        for mode,source in [('draft',target),('draft',foreign),('flash',target),('flash',foreign),('full',full_source)]:
            job_id='replication_%s_from_%s_to_%s'%(mode,source,target)
            cases_path=output/'cases'/(job_id+'.json')
            atomic_json(cases_path,cases)
            run_output=run_output_root/job_id
            jobs.append(dict(id=job_id,argv=['python3',str(HERE/'run.py'),'--gpu','{gpu}',
                '--gpu-uuid','{gpu_uuid}','--port','{port}','--output',str(run_output),'--mode',mode,
                '--source-suite',source,'--target-suite',target,'--cases',str(cases_path)],
                complete_file=str(run_output/'result.json'),ready_files=[str(output/'selection.json')],
                depends_on=[],condition=dict(mode=mode,source_suite=source,target_suite=target,
                    role='Full baseline' if mode=='full' else ('matched source' if source==target else 'foreign source'),
                    n=40,init_ids=list(HOLDOUT_INITS)),cases_sha256=sha(cases_path)))
    return jobs


def self_test():
    # Test selection independently of filesystem/model code with exact 360 identities.
    full_source=SUITES[0]
    keys=sorted(expected_keys(full_source))
    assert validate_complete_keys(keys,full_source)['missing']==[]
    assert len(validate_complete_keys(keys[:-1],full_source)['missing'])==1
    try:validate_complete_keys(keys+[keys[0]],full_source)
    except ValueError:pass
    else:raise AssertionError('duplicate conditions not rejected')
    def fixture(gaps=False):
        rows=[]
        for mode,source,target,task,init in keys:
            success=True
            if gaps and mode=='draft' and source!=target:
                success=task>=3
            rows.append(dict(mode=mode,source_suite=source,target_suite=target,task_id=task,init_id=init,
                             success=success))
        return rows
    c,r,s=comparisons_and_selection(fixture())
    assert len(c)==24 and len(r)==12 and s==[]
    c,r,s=comparisons_and_selection(fixture(True))
    assert [(x['target_suite'],x['foreign_source_suite']) for x in s]==[
        ('libero_spatial','libero_object'),('libero_object','libero_spatial')]
    assert all(x['p_value'] is None for x in c)
    changed=fixture(True)
    for row in changed:
        if row['mode']=='draft' and row['target_suite']=='libero_goal' and row['source_suite']=='libero_10':
            row['success']=row['task_id']>=8
    assert comparisons_and_selection(changed)[2][0]['target_suite']=='libero_goal'
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ease-holdout-inventory-') as temp:
        run=Path(temp)/'heldout'
        case=dict(suite='libero_goal',task_id=0,init_id=21)
        atomic_json(run/'manifest.json',dict(mode='draft',source_suite='libero_goal'))
        atomic_json(run/'cases.json',[case])
        outcome=run/'episodes'/case_key(case)/'result.json'
        outcome.parent.mkdir(parents=True)
        outcome.write_text('intentionally unreadable outcome; inventory must not parse it')
        inventory=holdout_inventory([Path(temp)])
        assert inventory[0]['evidence_exists'] and inventory[0]['outcomes_read'] is False
    print(json.dumps(dict(status='passed',tests=['complete360','missing','duplicate','null-selection','ties','ranking','no-discovery-p','heldout-outcomes-not-read'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--discovery-manifest',default=str(DEFAULT_ROOT/'draft_discovery_manifest.json'))
    p.add_argument('--runs',nargs='+',default=[str(DEFAULT_ROOT/'draft_smoke'),str(DEFAULT_ROOT/'draft_discovery')])
    p.add_argument('--output',default=str(HERE/'replication'))
    p.add_argument('--run-output-root',default=str(DEFAULT_ROOT/'draft_replication'))
    p.add_argument('--heldout-search-roots',nargs='+',default=[str(DEFAULT_ROOT)])
    p.add_argument('--audit-only',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    output=Path(a.output).resolve()
    if (output/'selection.json').exists():
        raise FileExistsError('Selection is already frozen; refusing overwrite or post-holdout reselection')
    output.mkdir(parents=True,exist_ok=True)
    inventory=holdout_inventory(a.heldout_search_roots)
    try:
        audit,rows,master=audit_discovery(a.discovery_manifest,a.runs)
    except Exception as error:
        atomic_json(output/'discovery_audit.json',dict(status='invalid',error=str(error),heldout_inventory=inventory))
        raise
    audit['heldout_inventory']=inventory
    atomic_json(output/'discovery_audit.json',audit)
    if audit['status']!='complete':
        print(json.dumps(dict(status='incomplete',complete=audit['complete'],expected=360,
            selection_created=False,output=str(output))))
        sys.exit(2)
    if any(row['evidence_exists'] for row in inventory):
        atomic_json(output/'heldout_evidence_block.json',dict(status='preexisting_holdout_evidence',
            inventory=inventory,outcomes_read=False,selection_created=False))
        raise RuntimeError('Held-out evidence already exists before this selection freeze; independent preregistration cannot be established')
    if a.audit_only:
        print(json.dumps(dict(status='complete_audit_only',complete=360,selection_created=False)));return
    comparisons,ranked,selected=comparisons_and_selection(rows)
    atomic_json(output/'all_discovery_comparisons.json',comparisons)
    atomic_json(output/'discovery_evidence.json',rows)
    selection=dict(status='selected' if selected else 'none',created_unix=time.time(),
        discovery_manifest_sha256=audit['discovery_manifest_sha256'],discovery_evidence_sha256=audit['evidence_sha256'],
        discovery_audit_sha256=sha(output/'discovery_audit.json'),discovery_conditions=360,
        rule=master['selection_rule'],tie_order=dict(target=list(SUITES),foreign_source=list(SUITES)),
        ranked_pure_draft_pairs=ranked,selected_pairs=selected,heldout_init_ids=list(HOLDOUT_INITS),
        selected_direction='matching-source success greater than foreign-source success',
        no_holdout_outcomes_read=True,preexisting_holdout_evidence=False,
        null_reason=None if selected else 'No positive matching-source minus foreign-source success difference in complete discovery',
        statistical_reporting='Report all24 discovery comparisons. Discovery p-values omitted; inferential comparisons use held-out cases only.',
        producer=str(Path(__file__).resolve()),producer_sha256=sha(__file__))
    atomic_json(output/'selection.json',selection)
    jobs=prepare_jobs(selected,output,Path(a.run_output_root).resolve(),audit['full_source_suite'])
    atomic_json(output/'jobs_draft.json',dict(status='draft_only_not_scheduled',selection_file=str(output/'selection.json'),
        selection_sha256=sha(output/'selection.json'),jobs=jobs,expected_episodes=len(jobs)*40,
        instruction='Parent controller reviews and appends these jobs; this script never mutates the live queue or launches a process.'))
    print(json.dumps(dict(status=selection['status'],selected_pairs=len(selected),jobs=len(jobs),
                          replication_episodes=len(jobs)*40,output=str(output),queue_modified=False)))

if __name__=='__main__':main()
