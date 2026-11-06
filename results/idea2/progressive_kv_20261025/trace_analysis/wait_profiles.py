"""Wait for at most four authorized GPU2 profiles, then analyze using CPU only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
EXPERIMENT=ROOT/'results/idea2/progressive_kv_20261025'
OUT=EXPERIMENT/'trace_analysis'
LEADS=(0,3,6,18)


def now():
    return datetime.now(timezone.utc).isoformat()


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
               gpu_launched=False,authorized_gpu_profile_source='GPU2 same_task only',cases=[])
    deadline=time.monotonic()+args.timeout_hours*3600
    state_path=OUT/'execution_receipt.json'
    write(state_path,state)
    failed=False
    for lead in LEADS:
        profile=EXPERIMENT/f'profile_lead{lead}_encoder_same_task_gpu2/result.json'
        trace=ROOT/f'nsys/idea2_experiment/progressive_kv_lead{lead}_encoder_20261025_v1.nsys-rep'
        dest=OUT/f'lead{lead}_encoder'
        case=dict(lead=lead,status='waiting_profile',profile_result=str(profile),trace=str(trace),output=str(dest))
        state['cases'].append(case)
        previous_signature=None
        while True:
            if time.monotonic()>deadline:
                case.update(status='timed_out_not_analyzed',finished=now())
                state.update(status='timed_out_incomplete',finished=now())
                write(state_path,state)
                print(json.dumps(case),flush=True)
                return 2
            value=None
            if profile.is_file():
                try:value=json.loads(profile.read_text())
                except (OSError,json.JSONDecodeError):pass
            if value and value.get('status')=='failed':
                case.update(status='profile_failed_not_analyzed',finished=now(),error=value.get('error'))
                failed=True
                print(json.dumps(case),flush=True)
                break
            if value and value.get('status')=='passed':
                rows=[r for r in value.get('rows',[]) if r.get('phase')=='profile']
                if len(rows)!=8:
                    case.update(status='profile_contract_failed',finished=now(),profile_rows=len(rows),error='Expected exactly 8 profile rows; verification excluded')
                    failed=True
                    print(json.dumps(case),flush=True)
                    break
                cfg=value.get('args',{})
                if cfg.get('profile_lead')!=lead or cfg.get('profile_variant')!='encoder':
                    case.update(status='profile_contract_failed',finished=now(),error='Profile lead/variant mismatch')
                    failed=True
                    print(json.dumps(case),flush=True)
                    break
                if trace.is_file() and trace.stat().st_size>0:
                    signature=(trace.stat().st_size,trace.stat().st_mtime_ns)
                    if signature==previous_signature:
                        dest.mkdir(parents=True,exist_ok=True)
                        command=['taskset','-c','64-95',sys.executable,str(EXPERIMENT/'analyze_traces.py'),
                                 '--trace',str(trace),'--profile-result',str(profile),'--output',str(dest)]
                        case.update(status='analyzing',started=now(),command=command,profile_rows=8,
                            profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),trace_size=signature[0],trace_mtime_ns=signature[1])
                        write(state_path,state)
                        print(json.dumps(dict(lead=lead,status='analyzing',time=now())),flush=True)
                        with (dest/'analysis.log').open('w') as log:
                            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
                        case.update(status='analyzed' if process.returncode==0 else 'analysis_failed',exit_code=process.returncode,finished=now())
                        if process.returncode==0:
                            report=json.loads((dest/'report.json').read_text())
                            case.update(attribution=report['attribution'],distributions=report['distributions'])
                        else:failed=True
                        write(dest/'execution_receipt.json',case)
                        print(json.dumps(dict(lead=lead,status=case['status'],output=str(dest),time=now())),flush=True)
                        break
                    previous_signature=signature
                    case['status']='waiting_trace_stability'
                else:
                    case['status']='waiting_completed_trace'
            state['updated']=now()
            write(state_path,state)
            time.sleep(args.poll_seconds)
        write(state_path,state)
    state.update(status='completed_with_failures' if failed else 'all_four_analyzed',finished=now())
    write(state_path,state)
    return 1 if failed else 0


if __name__=='__main__':
    raise SystemExit(main())
