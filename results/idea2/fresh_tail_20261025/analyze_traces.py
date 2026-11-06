"""CPU-only fresh-tail trace wrapper; legacy analyzer source remains unchanged.

Kernel attribution/overlap uses the existing recursive graph ancestry and
launch correlation implementation. The only resource-routing extension is
fresh_tail_adapter as a fallback for resource_adapter. Embedded wait duration
comes from matching profile runtime CUDA Events, never guessed from nsys gaps.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
ROOT=R.parents[2]
LEGACY_PATH=ROOT/'results/idea2/progressive_kv_20261025/analyze_traces.py'
HISTORICAL=ROOT/'results/idea2/progressive_kv_20261025/trace_analysis/gpu0_sm32/report.json'
os.environ['CUDA_VISIBLE_DEVICES']=''
spec=importlib.util.spec_from_file_location('fresh_tail_legacy_trace_analyzer',LEGACY_PATH)
legacy=importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
_resource_audit=legacy.resource_audit


def now():return datetime.now(timezone.utc).isoformat()


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,json.JSONDecodeError):return None


def parent_case(name):
    matches=[]
    paths=[R/'profile_matrix.execution.json'] if name.startswith('profile_') else R.glob('*.execution.json')
    for path in paths:
        value=read(path) or {}
        matches.extend((path,row) for row in value.get('cases',[]) if row.get('name')==name)
    if len(matches)>1:raise ValueError(f'Ambiguous parent execution receipts for {name}')
    return matches[0] if matches else (None,None)


def fresh_resource_audit(sqlite_path,kernels,profile):
    metadata=(profile or {}).get('fresh_tail_adapter')
    if (profile or {}).get('resource_adapter') or metadata is None:
        return _resource_audit(sqlite_path,kernels,profile)
    normalized=dict(profile)
    inherited=dict(metadata)
    source=ROOT/'results/idea2/progressive_kv_20261025/resource_adapter.py'
    case_name=Path(profile.get('args',{}).get('output','')).name
    receipt,case=parent_case(case_name)
    # The fresh adapter hash does not identify the inherited resource adapter.
    # Use its recorded parent-run hash if present; otherwise leave it unknown.
    inherited['sha256']=(case or {}).get('runtime_sources_before',{}).get(str(source))
    normalized['resource_adapter']=inherited
    result=_resource_audit(sqlite_path,kernels,normalized)
    adapter=R/'adapter.py'
    result['adapter_metadata_source']='fresh_tail_adapter'
    result['fresh_tail_adapter_provenance']=dict(recorded_sha256=metadata.get('sha256'),
        current_sha256=digest(adapter),recorded_source_matches=metadata.get('sha256')==digest(adapter),
        parent_execution=str(receipt) if receipt else None,
        ready_semantics=metadata.get('ready_semantics'),cleanup=metadata.get('cleanup'))
    return result


def analyze(sqlite_path,output,profile,provenance):
    metadata=profile.get('fresh_tail_adapter')
    if not metadata:raise ValueError('Matching final fresh_tail_adapter metadata is required')
    tail=metadata['fresh_tail_steps']
    rows=[r for r in profile.get('rows',[]) if r.get('phase')=='profile']
    by_key={(r['request'],r['lead'],r['slot']):r for r in rows}
    if len(by_key)!=len(rows):raise ValueError('Ambiguous profile row identity')
    addresses=profile.get('kv_addresses') or []
    layers=len(addresses[0])//2 if addresses and len(addresses[0])%2==0 else 18
    if not rows:raise ValueError('Profile rows absent')
    if tail not in (0,1,2):raise ValueError('Unsupported fresh-tail experiment condition')
    # Reject source-contract failures before the reused analyzer writes reports.
    for source in rows:
        frontiers=source['frontiers']
        if source.get('fresh_tail_steps')!=tail or len(frontiers)!=10:
            raise ValueError('Fresh-tail condition or D10 schedule does not match profile')
        if tail and frontiers[-tail:]!=[layers]*tail:
            raise ValueError('Mandatory fresh-tail frontier was incomplete')
        if source.get('fresh_tail_wait_ms') is None:
            raise ValueError('Missing runtime fresh_tail_wait_ms; do not infer it from trace gaps')
    previous=legacy.resource_audit
    try:
        legacy.resource_audit=fresh_resource_audit
        report=legacy.analyze(sqlite_path,output,profile,provenance)
    finally:
        legacy.resource_audit=previous
    samples=[]
    for row in report['requests']:
        source=by_key[row['request'],row['lead'],row['slot']]
        frontiers=source['frontiers']
        if source.get('fresh_tail_steps')!=tail or len(frontiers)!=10:
            raise ValueError('Fresh-tail condition or D10 schedule does not match profile')
        fresh=(frontiers[-tail:]==[layers]*tail) if tail else True
        if not fresh:raise ValueError('Mandatory fresh-tail frontier was incomplete')
        wait=source.get('fresh_tail_wait_ms')
        if wait is None:raise ValueError('Missing runtime fresh_tail_wait_ms; do not infer it from trace gaps')
        row.update(fresh_tail_steps=tail,fresh_tail_wait_ms_runtime_event=wait,
                   final_tail_frontiers=frontiers[-tail:] if tail else [],mandatory_tail_frontiers_complete=fresh)
        samples.append(dict(request=row['request'],source_id=row['source_id'],fresh_tail_steps=tail,
                            frontiers=frontiers,runtime_event_wait_span_ms=wait))
    report['fresh_tail_analysis']=dict(adapter_metadata=metadata,expected_num_layers=layers,
        samples=samples,runtime_event_wait_span_ms=legacy.stats([r['runtime_event_wait_span_ms'] for r in samples]),
        mandatory_tail_frontiers_complete=True,
        wait_measurement='Matching profile fresh_tail_wait_ms: CUDA Event span from iteration_marks[D-tail] before captured wait to tail_wait_complete after captured wait. Includes marker/node overhead; not a pure kernel duration or a fabricated nsys Event timestamp.',
        nsys_wait_timestamps_inferred=False,
        historical_reference=str(HISTORICAL),historical_scope='Old SM32 tail0 profile is mechanism history only; do not pool with new formal timings.')
    report['definitions']['fresh_tail']='All current K/V ready, potentially before final VLM FFN. Earlier mixed rounds remain part of the trajectory; last fresh rounds do not imply equivalence to all-fresh D10.'
    with (output/'kernels.csv').open() as handle:
        kernels=list(csv.DictReader(handle))
    ordering=[]
    for row in report['requests']:
        own=[k for k in kernels if k['request_id']==row['request_id']]
        published=sorted((k for k in own if k['stage']=='VLM' and k['helper']=='_publish'),key=lambda k:int(k['start']))
        snapshots=sorted((k for k in own if k['stage']=='AE' and k['helper']=='_snapshot'),key=lambda k:int(k['start']))
        evidence=dict(request=row['request'],request_id=row['request_id'],publish_kernels=len(published),snapshot_kernels=len(snapshots),fresh_tail_steps=tail)
        if len(published)!=layers or len(snapshots)!=10:
            evidence.update(available=False,passed=False,reason='Expected one publish per layer and one snapshot per D10 round; incomplete kernel evidence')
        else:
            publish_end=int(published[-1]['end'])
            fresh_snapshots=[int(k['start']) for k in snapshots[10-tail:]] if tail else []
            evidence.update(available=True,last_KV_publish_kernel_end_ns=publish_end,
                first_mandatory_fresh_snapshot_start_ns=fresh_snapshots[0] if fresh_snapshots else None,
                fresh_snapshot_starts_ns=fresh_snapshots,
                fresh_snapshot_after_last_publish_ms=[(stamp-publish_end)/1e6 for stamp in fresh_snapshots],
                passed=all(stamp>=publish_end for stamp in fresh_snapshots))
        ordering.append(evidence)
    report['fresh_tail_analysis']['kernel_ready_ordering']=dict(passed=all(r['passed'] for r in ordering),rows=ordering,
        definition='Ordered single-stream D10 snapshot kernels checked against final current-VLM release-publish kernel end. This corroborates readiness; it does not assign a timestamp or duration to the captured Event wait.')
    legacy.save(output/'report.json',report)
    legacy.csv_write(output/'requests.csv',report['requests'])
    legacy.csv_write(output/'fresh_tail_waits.csv',samples)
    legacy.csv_write(output/'fresh_tail_ready_ordering.csv',ordering)
    return report


def mechanism_summary(output_root):
    sources=[]
    if HISTORICAL.exists():sources.append(('historical_sm32_tail0',HISTORICAL,True))
    for tail in (1,2):
        path=output_root/f'sm32_tail{tail}'/'report.json'
        if path.exists():sources.append((f'fresh_sm32_tail{tail}',path,False))
    rows=[]
    for name,path,historical in sources:
        report=read(path)
        if not report or report.get('status')!='analyzed_existing_trace':continue
        if not historical and not report.get('fresh_tail_analysis'):continue
        dist=report['distributions']
        row=dict(condition=name,source=str(path),source_sha256=digest(path),historical_reference_only=historical,
            requests=report['attribution']['requests'],
            AE_x_VLM_ms=dist['AE_x_currentVLM_ms']['mean'],AE_x_nextVision_ms=dist['AE_x_nextVision_ms']['mean'],
            AE_end_minus_VLM_end_kernel_ms=dist['ae_finish_minus_currentVLM_finish_kernel_ms']['mean'],
            AE_end_before_VLM_requests=report['ae_tail_before_or_at_vlm_requests'],
            AE_busy_ms=dist['ae_busy_ms']['mean'],AE_envelope_ms=dist['ae_envelope_ms']['mean'],
            VLM_busy_ms=dist['vlm_busy_ms']['mean'],VLM_envelope_ms=dist['vlm_envelope_ms']['mean'],
            runtime_event_wait_span_ms=(report.get('fresh_tail_analysis') or {}).get('runtime_event_wait_span_ms'),
            resource_verified=report['resource_audit']['passed'],
            eligible_sm_pair=report['resource_audit']['expected_sm_pair'],
            gpu_uuid=(report.get('profile_args') or {}).get('gpu_uuid'),
            fresh_tail_ready_ordering=(report.get('fresh_tail_analysis') or {}).get('kernel_ready_ordering'),
            observed_frontiers=[r['frontiers'] for r in report['requests']])
        rows.append(row)
    summary=dict(status='mechanism_profiles_complete' if sum(not x['historical_reference_only'] for x in rows)==2 else 'awaiting_remaining_profiles',
        rows=rows,gpu_launched=False,
        boundaries=['Overlap is actual GPU kernel intersection; Host NVTX ends never clip background work.',
            'Fresh-tail wait duration is the profile runtime CUDA Event span, not nsys kernel-gap inference.',
            'Historical tail0 is mechanism-only; these profiles do not rank formal steady-state latency.',
            'Green SM counts are resource eligibility, not measured occupancy.',
            'Numerical/quality conclusions belong to matched-schedule gates and paired/closed-loop runs, not these traces.'])
    legacy.save(output_root/'mechanism_summary.json',summary)
    legacy.csv_write(output_root/'mechanism_summary.csv',rows)
    return summary


def process_trace(trace,profile_path,output):
    if not trace.is_file():raise FileNotFoundError(f'Trace not ready: {trace}')
    profile=read(profile_path)
    if not profile or profile.get('status')!='passed':raise ValueError('Profile must be passed')
    if not any(r.get('phase')=='profile' for r in profile.get('rows',[])):raise ValueError('Profile rows absent')
    output.mkdir(parents=True,exist_ok=True)
    source=trace
    if trace.suffix!='.sqlite':
        source=output/'trace.sqlite'
        receipt=output/'export_receipt.json'
        signature=dict(path=str(trace.resolve()),size=trace.stat().st_size,mtime_ns=trace.stat().st_mtime_ns)
        existing=read(receipt)
        if not(source.exists() and existing and existing.get('source')==signature):
            if source.exists():raise ValueError('Unmatched existing trace.sqlite; use a fresh output directory')
            command=['nsys','export','--type','sqlite','--output',str(source),str(trace)]
            with (output/'export.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            legacy.save(receipt,dict(source=signature,command=command,cpu_only=True))
    provenance=dict(trace=str(trace.resolve()),profile_result=str(profile_path.resolve()),
        analyzer_sha256=digest(Path(__file__)),legacy_analyzer_sha256=digest(LEGACY_PATH),
        reused_layer_trace_sha256=digest(Path(legacy.base.__file__)),profile_sha256=digest(profile_path),gpu_launched=False)
    report=analyze(source,output,profile,provenance)
    if not report['resource_audit']['passed']:raise ValueError('Green resource audit failed; evidence retained')
    if not report['fresh_tail_analysis']['kernel_ready_ordering']['passed']:raise ValueError('Fresh-tail kernel readiness order unproved; evidence retained')
    return report


def watch(minutes):
    os.sched_setaffinity(0,set(range(64,96)))
    output=R/'trace_analysis';output.mkdir(parents=True,exist_ok=True)
    state=dict(status='waiting',started=now(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),
               maximum_wait_minutes=minutes,gpu_launched=False,cases=[])
    path=output/'execution_receipt.json'
    deadline=time.monotonic()+minutes*60
    for tail in (1,2):
        state['cases'].append(dict(tail=tail,status='waiting',profile=str(R/f'profile_sm32_tail{tail}'/'result.json'),
            trace=str(ROOT/f'nsys/idea2_experiment/fresh_tail_sm32_tail{tail}_lead3_encoder_20261025_v1.nsys-rep'),
            output=str(output/f'sm32_tail{tail}')))
    prior={}
    while any(c['status'].startswith('waiting') for c in state['cases']):
        if time.monotonic()>deadline:
            for case in state['cases']:
                if case['status'].startswith('waiting'):case['status']='timed_out_not_analyzed'
            state.update(status='timed_out_incomplete',finished=now())
            legacy.save(path,state);mechanism_summary(output)
            return 2
        for case in state['cases']:
            if not case['status'].startswith('waiting'):continue
            profile_path=Path(case['profile']);profile=read(profile_path)
            if not profile or profile.get('status') not in ('passed','failed'):continue
            if profile['status']=='failed':case.update(status='profile_failed_not_analyzed',finished=now());continue
            meta=profile.get('fresh_tail_adapter') or {}
            rows=[r for r in profile.get('rows',[]) if r.get('phase')=='profile']
            if (meta.get('ae_sm_min')!=32 or meta.get('fresh_tail_steps')!=case['tail']
                    or not(meta.get('cleanup') or {}).get('passed')):
                continue  # Wait for adapter finally/cleanup to finish writing.
            if len(rows)!=8 or any(r['lead']!=3 or r['variant']!='encoder' for r in rows):
                case.update(status='profile_contract_failed',profile_rows=len(rows));continue
            receipt,parent=parent_case(profile_path.parent.name)
            if not parent or parent.get('status')!='passed' or parent.get('exit_code')!=0:continue
            trace=Path(case['trace'])
            if not trace.exists() or not trace.stat().st_size:continue
            signature=(trace.stat().st_size,trace.stat().st_mtime_ns)
            if prior.get(case['tail'])!=signature:
                prior[case['tail']]=signature;case['status']='waiting_trace_stability';continue
            dest=Path(case['output']);dest.mkdir(parents=True,exist_ok=True)
            command=['taskset','-c','64-95',sys.executable,str(Path(__file__).resolve()),'--trace',str(trace),
                     '--profile-result',str(profile_path),'--output',str(dest)]
            case.update(status='analyzing',started=now(),command=command,parent_execution=str(receipt) if receipt else None)
            legacy.save(path,state)
            with (dest/'analysis.log').open('w') as log:
                run=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
            case.update(status='analyzed' if run.returncode==0 else 'analysis_failed',exit_code=run.returncode,finished=now())
            legacy.save(dest/'execution_receipt.json',case)
            legacy.save(path,state);mechanism_summary(output)
            print(json.dumps(dict(tail=case['tail'],status=case['status'],time=now())),flush=True)
        state['updated']=now();legacy.save(path,state)
        if any(c['status'].startswith('waiting') for c in state['cases']):time.sleep(10)
    state.update(status='both_analyzed' if all(c['status']=='analyzed' for c in state['cases']) else 'completed_with_failures',finished=now())
    legacy.save(path,state);mechanism_summary(output)
    return 0 if state['status']=='both_analyzed' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--profile-result',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--watch-minutes',type=float,default=30)
    args=parser.parse_args()
    if args.watch:
        if not 0<args.watch_minutes<=30:parser.error('Watch must be bounded to (0,30] minutes')
        return watch(args.watch_minutes)
    if not(args.trace and args.profile_result and args.output):parser.error('--trace, --profile-result, --output are required without --watch')
    report=process_trace(args.trace,args.profile_result,args.output)
    print(json.dumps(dict(status=report['status'],attribution=report['attribution'],
        resource_passed=report['resource_audit']['passed'],fresh_tail=report['fresh_tail_analysis']),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
