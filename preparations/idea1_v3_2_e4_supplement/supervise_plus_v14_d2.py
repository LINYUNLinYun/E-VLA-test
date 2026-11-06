"""Resume D2 queue and release only the explicitly identified owned service."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
from .native_envs import sha,write_json
from .plus_v14_d2 import verify_manifest,dispatch
from .parallel_continue import process_identity,same_process
from preparations.idea1_v3_2.transfer_client import rpc


def assert_service(a,identity):
    health=rpc(a.port,dict(op='health'))['protocol']
    if health.get('manifest_hash')!=identity or health.get('server_pid')!=a.expected_server_pid or health.get('gpu_uuid')!=a.gpu_uuid:
        raise RuntimeError('Owned service PID/manifest/UUID mismatch; no shutdown sent')
    if health.get('modes')!=['v14_d2'] or health.get('D')!=2 or not health.get('legacy_gate_passed'):
        raise RuntimeError('Expected ready D2-only gated service')
    return health


def supervise(a):
    m=verify_manifest(Path(a.output)/'manifest.json');receipt_path=Path(a.receipt)
    if receipt_path.exists():raise FileExistsError(receipt_path)
    receipt=dict(status='verifying_ready_service',pid=os.getpid(),manifest_hash=m['manifest_hash'],
        manifest_sha256=sha(Path(a.output)/'manifest.json'),supervisor_source_sha256=sha(__file__),
        expected_server_pid=a.expected_server_pid,gpu_uuid=a.gpu_uuid,port=a.port,started_unix=time.time(),deadline_unix=a.deadline_unix,
        resume='Existing frozen worker skips completed D2 cases; only remaining D2 conditions dispatched')
    server=process_identity(a.expected_server_pid)
    if not server:raise RuntimeError('Expected server process missing')
    if 'policy_plus_v14_d2' not in server['argv']:raise RuntimeError('Expected server command identity missing')
    receipt['server_process_identity']=server;receipt['initial_health']=assert_service(a,m['manifest_hash']);write_json(receipt_path,receipt)
    claim_owner=Path(a.output)/'worker_claim/owner.json'
    if claim_owner.exists():
        owner=json.loads(claim_owner.read_text())
        if same_process(owner):raise RuntimeError('Another live worker owns queue; supervisor will not dispatch or shutdown')
    worker_error=None
    try:
        receipt['status']='dispatching';write_json(receipt_path,receipt);dispatch(a,m)
        receipt['status']='worker_returned'
    except Exception:
        worker_error=traceback.format_exc();receipt.update(status='worker_technical_failure',worker_error=worker_error)
    finally:
        # Dispatch drains its child before returning, including deadline cases.
        # Independently inspect this process's direct children before shutdown.
        children_path=Path('/proc')/str(os.getpid())/'task'/str(os.getpid())/'children'
        children=[int(v) for v in children_path.read_text().split()] if children_path.exists() else []
        live=[process_identity(pid) for pid in children]
        live=[identity for identity in live if identity and identity['state']!='Z']
        if claim_owner.exists():
            owner=json.loads(claim_owner.read_text())
            if owner['pid']!=os.getpid() and same_process(owner):live.append(owner)
        receipt['live_direct_children_before_shutdown']=live
        if live:
            receipt['shutdown']='withheld_live_owned_child';write_json(receipt_path,receipt)
            raise RuntimeError('Owned child still live; preserve service for natural drain')
        if not same_process(server):
            receipt['shutdown']='service_already_exited_or_pid_reused';write_json(receipt_path,receipt)
        else:
            try:
                receipt['final_health']=assert_service(a,m['manifest_hash'])
                receipt['shutdown_response']=rpc(a.port,dict(op='shutdown'));receipt['shutdown']='sent_to_exact_owned_service'
                until=time.time()+30
                while same_process(server) and process_identity(server['pid'])['state']!='Z' and time.time()<until:time.sleep(.2)
                receipt['server_exited']=not same_process(server) or process_identity(server['pid'])['state']=='Z'
            except Exception:receipt.update(shutdown='technical_failure_no_signals_sent',shutdown_error=traceback.format_exc())
        receipt['finished_unix']=time.time();write_json(receipt_path,receipt)
    if a.analysis_output:
        try:
            from .analyze_plus_v14_d2 import analyze
            report=analyze(a.output,a.analysis_output,allow_partial=True,include_costs=not a.skip_request_costs)
            receipt['analysis']=dict(output=a.analysis_output,status=report['status'],completed=report['completed_new_conditions'])
        except Exception:receipt['analysis_error']=traceback.format_exc()
        write_json(receipt_path,receipt)
    if worker_error:raise RuntimeError(worker_error)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--expected-server-pid',type=int,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--max-new-cases',type=int,default=195);p.add_argument('--analysis-output');p.add_argument('--skip-request-costs',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:print(json.dumps(dict(dry_run=True,**vars(a))));return
    result=supervise(a);print(json.dumps(dict(status=result['status'],shutdown=result.get('shutdown'))))


if __name__=='__main__':main()
