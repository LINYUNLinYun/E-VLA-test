"""Recorded incomplete-service handling; all scientific cases and gates stay fixed."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/root/nfs/embodied/benchmark');HERE=Path(__file__).resolve().parent
PLAN_SHA='486abfff778a15f76e013f37a37609a3841c2fa7df82abfe19404c2e12a4d6fd'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def plan():
    p=HERE/'resume_plan.json';require(sha(p)==PLAN_SHA,'Technical resume plan changed');return read(p)


def incomplete_service(run,manifest):
    """Recognize exactly the one recorded pre-prediction pairing-gate failure."""
    value=plan();recorded=value['failed_service'];run=Path(run).resolve()
    if run!=Path(recorded['directory']).resolve():return False
    require(manifest['manifest_hash']==value['scientific_manifest_hash'],'Technical recovery scientific identity differs')
    require(not (run/'result.json').exists() and recorded['complete_service_receipt_present'] is False,'Recorded failed service unexpectedly has a completion result')
    for name,digest in recorded['artifacts'].items():require(sha(run/name)==digest,'Recorded failed service artifact changed: '+name)
    for name in ('identity.json','protocol.json','gate.json','fixed_input_cost.json'):
        require(read(run/name)['manifest_hash']==manifest['manifest_hash'],'Failed service numerical identity differs')
    gate=read(run/'gate.json');cross=read(run/'cross_service.json')
    require(gate['passed'] and all(r['passed'] for r in gate['checks']) and cross['passed'],'Recorded service had a failed numerical gate')
    diag=read(value['fresh_environment_diagnostic']['path']);require(sha(value['fresh_environment_diagnostic']['path'])==value['fresh_environment_diagnostic']['sha256']
        and diag['status']=='pass' and diag['actual_input_hash']==diag['expected_input_hash'] and not diag['VLA_inference_performed'],'Strict reset diagnostic changed')
    require(value['failure']['VLA_case_prediction_performed'] is False and value['valid_outcomes_to_repeat']==0,'Technical failure scope differs')
    return True


def allowed_cost_exclusions(costs,manifest):
    """Keep only completed-service costs, exactly as the original selection rule."""
    failed=Path(plan()['failed_service']['directory']);require(incomplete_service(failed,manifest),'Missing recorded incomplete service')
    expected=str(failed/'fixed_input_cost.json');actual=[v['path'] for v in costs['excluded_or_pending']]
    require(actual==[expected],'Unexpected excluded or pending cost services: '+str(actual))
    require(costs['valid_services']==8,'Need the original eight completed cost services')
    return [dict(path=expected,sha256=sha(expected),reason='Recorded pre-prediction technical failure; no completed service receipt. Valid committed scientific cases remain included.')]


def verify_accepted_science_preserved():
    """Frozen recovery may rebuild administrative receipts, never raw outcomes."""
    value=plan();checked=[];receipt_changes=[]
    failed=Path(value['failed_service']['directory']);old_receipts={}
    require(sha(failed/'worker.log')==value['failed_service']['artifacts']['worker.log'],'Original failure log changed')
    for line in (failed/'worker.log').read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        if r.get('valid') is True and r.get('condition')=='ae12_d4' and r.get('manifest_hash')==value['scientific_manifest_hash']:
            s=r['case'];old_receipts[(s['suite'],s['task'],s['init'])]=r
    for item in value['accepted_cases_before_resume']:
        p=Path(item['receipt']['path']);current=read(p)
        require(current['artifacts']==item['artifacts'],'Accepted scientific artifact map changed')
        for name,digest in item['artifacts'].items():require(sha(p.parent/name)==digest,'Accepted scientific artifact changed: '+str(p.parent/name))
        if sha(p)!=item['receipt']['sha256']:
            s=item['case'];old=old_receipts.get((s['suite'],s['task'],s['init']))
            require(item['condition']=='ae12_d4' and old is not None,'Unexpected accepted receipt rewrite')
            require({k:v for k,v in current.items() if k!='worker_pid'}=={k:v for k,v in old.items() if k!='worker_pid'},'Administrative recovery changed more than receipt writer PID')
            require((p.parent/'recovery_receipt.json').exists(),'Rebuilt receipt lacks the frozen recovery ledger')
            receipt_changes.append(dict(path=str(p),before_sha256=item['receipt']['sha256'],after_sha256=sha(p),changed_field='worker_pid only; frozen recover_case rebuilt the administrative receipt',raw_scientific_artifacts_unchanged=True))
        checked.append(dict(path=str(p),sha256=sha(p),raw_scientific_artifacts_unchanged=True))
    return dict(accepted_before_resume=len(checked),all_raw_scientific_artifacts_unchanged=True,administrative_receipt_changes=receipt_changes,
        accepted_evidence=checked,no_valid_outcomes_repeated=True,plan_sha256=PLAN_SHA)
