"""CPU-only analysis after BOTH GPU0 queue exit=0 and final profile pass."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
R=ROOT/'results/idea2/progressive_kv_20261025'
OUT=R/'trace_analysis'
PARENT=R/'gpu0_selected_resource_confirmation_matrix.execution.json'
CASES=(
    ('gpu0_full_lead3_profile','progressive_kv_full_lead3_encoder_gpu0_20261025_v1','gpu0_full',0),
    ('gpu0_sm16_lead3_profile','progressive_kv_sm16_lead3_encoder_gpu0_20261025_v1','gpu0_sm16',16),
    ('gpu0_sm32_exposure_profile','progressive_kv_sm32_lead3_encoder_gpu0_20261025_v1','gpu0_sm32',32),
)


def now():return datetime.now(timezone.utc).isoformat()


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timeout-hours',type=float,default=12)
    parser.add_argument('--poll-seconds',type=float,default=20)
    args=parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.sched_setaffinity(0,set(range(64,96)))
    OUT.mkdir(parents=True,exist_ok=True)
    state=dict(status='waiting',started=now(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        gpu_launched=False,parent_execution=str(PARENT),source='GPU0 only; no cross-GPU duration ranking',cases=[])
    receipt=OUT/'gpu0_execution_receipt.json'
    deadline=time.monotonic()+args.timeout_hours*3600
    failed=False
    for name,basename,short,expected_sms in CASES:
        profile=R/name/'result.json'
        trace=ROOT/'nsys/idea2_experiment'/f'{basename}.nsys-rep'
        dest=OUT/short
        case=dict(name=name,status='waiting_parent_case',profile_result=str(profile),trace=str(trace),output=str(dest),expected_ae_sm_count=expected_sms)
        state['cases'].append(case)
        previous_signature=None
        while True:
            if time.monotonic()>deadline:
                case.update(status='timed_out_not_analyzed',finished=now())
                state.update(status='timed_out_incomplete',finished=now())
                write(receipt,state)
                return 2
            parent=read(PARENT)
            matches=[c for c in (parent or {}).get('cases',[]) if c.get('name')==name]
            parent_case=matches[0] if len(matches)==1 else None
            value=read(profile)
            if parent_case and parent_case.get('status')=='passed' and parent_case.get('exit_code')==0:
                case['parent_evidence']={k:parent_case.get(k) for k in ('name','status','exit_code','pid','start_utc','end_utc','runtime_sources_unchanged')}
                if value and value.get('status')=='passed':
                    rows=[r for r in value.get('rows',[]) if r.get('phase')=='profile']
                    cfg=value.get('args',{})
                    resource=value.get('resource_adapter') or {}
                    if (len(rows)!=8 or cfg.get('profile_lead')!=3 or cfg.get('profile_variant')!='encoder'
                            or resource.get('ae_sm_min',0)!=expected_sms):
                        case.update(status='profile_contract_failed',profile_rows=len(rows),finished=now())
                        failed=True
                        break
                    if trace.is_file() and trace.stat().st_size>0:
                        signature=(trace.stat().st_size,trace.stat().st_mtime_ns)
                        if signature==previous_signature:
                            dest.mkdir(parents=True,exist_ok=True)
                            command=['taskset','-c','64-95',sys.executable,str(R/'analyze_traces.py'),
                                '--trace',str(trace),'--profile-result',str(profile),'--output',str(dest)]
                            case.update(status='analyzing',started=now(),command=command,profile_rows=8,
                                profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),trace_size=signature[0],trace_mtime_ns=signature[1])
                            write(receipt,state)
                            print(json.dumps(dict(name=name,status='analyzing',time=now())),flush=True)
                            with (dest/'analysis.log').open('w') as log:
                                process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
                            case.update(status='analyzed' if process.returncode==0 else 'analysis_failed',exit_code=process.returncode,finished=now())
                            if process.returncode==0:
                                report=read(dest/'report.json')
                                resource=report['resource_audit']
                                expected_pair=[expected_sms,128-expected_sms] if expected_sms else [None,None]
                                if not resource['passed'] or resource['expected_sm_pair']!=expected_pair:
                                    case['status']='resource_audit_failed'
                                case.update(attribution=report['attribution'],distributions=report['distributions'],
                                    resource_assignments=resource['assignments'],resource_audit_passed=resource['passed'],
                                    ae_tail_before_or_at_vlm_requests=report['ae_tail_before_or_at_vlm_requests'])
                            if case['status']!='analyzed':failed=True
                            write(dest/'execution_receipt.json',case)
                            print(json.dumps(dict(name=name,status=case['status'],output=str(dest),time=now())),flush=True)
                            break
                        previous_signature=signature
                        case['status']='waiting_trace_stability'
                    else:case['status']='waiting_completed_trace'
                elif value and value.get('status')=='failed':
                    case.update(status='profile_failed_not_analyzed',finished=now(),error=value.get('error'))
                    failed=True
                    break
                else:case['status']='waiting_final_profile_result'
            elif parent_case and parent_case.get('status') not in ('running','waiting','pending','passed',None):
                case.update(status='parent_case_failed_not_analyzed',parent_status=parent_case.get('status'),parent_exit_code=parent_case.get('exit_code'),finished=now())
                failed=True
                break
            state['updated']=now()
            write(receipt,state)
            time.sleep(args.poll_seconds)
        write(receipt,state)
    state.update(status='completed_with_failures' if failed else 'all_three_analyzed',finished=now())
    write(receipt,state)
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
