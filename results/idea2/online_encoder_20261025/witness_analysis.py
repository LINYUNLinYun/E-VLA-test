"""CPU-only actual-kernel overlap audit for witness.py; no Event-envelope proof."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[3]
LEGACY=ROOT/'results/idea2/progressive_kv_20261025/analyze_traces.py'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


legacy=load('witness_existing_kernel_analyzer',LEGACY)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence(report):
    rows=report['requests']
    paired=[r for r in rows if r.get('next_encoder_verified')]
    both=[r for r in paired if r['AE_x_currentVLM_ms']>0 and r['AE_x_nextVision_ms']>0]
    return dict(status='overlap_observed' if both else 'overlap_not_observed',
        requests=len(rows),certified_next_encoder_requests=len(paired),
        requests_with_both_actual_kernel_intersections=len(both),
        AE_x_currentVLM_ms=report['distributions'].get('AE_x_currentVLM_ms'),
        AE_x_nextEncoder_ms=report['distributions'].get('AE_x_nextVision_ms'),
        matching_request_ids=[r['request_id'] for r in both],resource_audit_passed=report['resource_audit']['passed'],
        definition='Union of actual kernel interval intersections for independently attributed AE/VLM/next-Vision graph replays, same process/device and distinct streams. Host NVTX ranges never clip GPU work; CUDA Event spans are auxiliary only.',
        boundary='Offline preprepared recorded observations; no online causality, E2E speedup or quality claim.')


def self_test():
    fixture=load('witness_synthetic_trace_fixture',LEGACY.parent/'analysis_selftest/test_analyzer.py')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);sql=root/'synthetic.sqlite';profile=fixture.fixture(sql)
        report=legacy.analyze(sql,root/'out',profile,dict(synthetic=True))
        result=evidence(report)
        assert result['requests_with_both_actual_kernel_intersections']==1
        assert report['requests'][0]['AE_x_currentVLM_ms']==20
        assert report['requests'][0]['AE_x_nextVision_ms']==15
        assert report['requests'][0]['graph_kernels_after_host_NVTX']>0
        with sqlite3.connect(sql) as db:
            db.execute('update CUPTI_ACTIVITY_KIND_KERNEL set graphNodeId=9999 where correlationId=5')
        try:legacy.analyze(sql,root/'invalid',profile)
        except ValueError as exc:assert 'attribution incomplete' in str(exc)
        else:raise AssertionError('Incomplete graph evidence was accepted')
    return dict(status='passed',cpu_only=True,gpu_launched=False,
        checks=['actual AE/VLM and AE/next-Vision intersections','no Host NVTX clipping','missing graph ancestry rejected'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path)
    parser.add_argument('--profile-result',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test()));return 0
    if not(args.trace and args.profile_result and args.output):parser.error('--trace, --profile-result and --output are required')
    args.output.mkdir(parents=True,exist_ok=True)
    try:
        if not args.trace.is_file():raise ValueError('Trace absent; kernel evidence unavailable')
        profile=json.loads(args.profile_result.read_text())
        rows=[r for r in profile.get('rows',[]) if r.get('phase')=='profile']
        if profile.get('status')!='passed' or not profile.get('cleanup',{}).get('passed') or not profile.get('oracle',{}).get('passed'):
            raise ValueError('Witness execution, matched-schedule oracle or cleanup incomplete')
        if len(rows)!=8 or sum(bool(r['next_encoder']) for r in rows)!=7:
            raise ValueError('Expected eight measured requests with seven next-Encoder jobs')
        source=args.trace
        if source.suffix!='.sqlite':
            source=args.output/'trace.sqlite';receipt=args.output/'export_receipt.json'
            signature=dict(path=str(args.trace.resolve()),size=args.trace.stat().st_size,mtime_ns=args.trace.stat().st_mtime_ns)
            if source.exists():
                if not receipt.exists() or json.loads(receipt.read_text()).get('source')!=signature:raise ValueError('Existing SQL export does not match trace')
            else:
                command=['nsys','export','--type','sqlite','--output',str(source),str(args.trace)]
                with (args.output/'export.log').open('w') as log:subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
                legacy.save(receipt,dict(source=signature,command=command,cpu_only=True))
        report=legacy.analyze(source,args.output,profile,dict(trace=str(args.trace.resolve()),profile_result=str(args.profile_result.resolve()),
            analyzer_sha256=sha(Path(__file__)),legacy_analyzer_sha256=sha(LEGACY),profile_sha256=sha(args.profile_result),gpu_launched=False))
        summary=evidence(report)
        if report['attribution']['requests']!=8 or summary['certified_next_encoder_requests']!=7:
            raise ValueError('Incomplete independent request/next-observation attribution')
        if not summary['resource_audit_passed']:raise ValueError('Green resource assignment unverified')
        legacy.save(args.output/'witness_summary.json',summary)
        print(json.dumps(summary,indent=2));return 0
    except (ValueError,OSError,KeyError,subprocess.CalledProcessError) as exc:
        legacy.save(args.output/'witness_summary.json',dict(status='missing_or_invalid_evidence',error=str(exc),gpu_launched=False,
            claim='No independent actual-kernel overlap conclusion can be drawn.'))
        raise


if __name__=='__main__':raise SystemExit(main())
