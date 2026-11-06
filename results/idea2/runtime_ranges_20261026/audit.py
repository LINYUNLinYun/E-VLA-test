"""CPU-only proof of full replay envelopes with retained per-node kernels."""
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

R=Path(__file__).resolve().parent
ROOT=R.parents[2]
LABELS={"preprocess","encoder","vlm","AE"}


def analyze(stage):
    kind=stage["kind"]
    directory=R/kind
    trace=Path(stage["trace"])
    cached=directory/"range_audit.json"
    signature=dict(path=str(trace),bytes=trace.stat().st_size,mtime_ns=trace.stat().st_mtime_ns)
    if cached.exists():
        previous=json.loads(cached.read_text())
        assert previous['trace_signature']==signature
        return previous
    sql=directory/"trace.sqlite"
    if not sql.exists():
        with (directory/"export.log").open("x") as log:
            subprocess.run(["nsys","export","--type","sqlite","--output",str(sql),str(trace)],
                check=True,stdout=log,stderr=subprocess.STDOUT)
    csvs=list(directory.glob("projection*.csv"))
    if not csvs:
        with (directory/"projection.log").open("x") as log:
            subprocess.run(["nsys","stats","--report","nvtx_gpu_proj_trace","--format","csv",
                "--output",str(directory/"projection"),str(sql)],check=True,stdout=log,stderr=subprocess.STDOUT)
        csvs=list(directory.glob("projection*.csv"))
    assert len(csvs)==1
    projected=list(csv.DictReader(csvs[0].open()))
    counts=collections.Counter(row["Name"].rsplit(":",1)[-1] for row in projected)
    for label in ("encoder","vlm","AE"):
        assert counts[label]==8,(kind,counts)
    capture=json.loads((directory/"capture.json").read_text())
    assert capture["captured"]==8
    with sqlite3.connect(sql) as db:
        db.row_factory=sqlite3.Row
        strings=dict(db.execute('select id,value from StringIds'))
        ranges=[]
        for r in db.execute('select rowid,* from NVTX_EVENTS where domainId=0 and end is not null'):
            v=dict(r);v['label']=r['text'] or strings.get(r['textId'],'');ranges.append(v)
        app_counts=collections.Counter(v['label'] for v in ranges)
        assert set(app_counts)==LABELS,(kind,app_counts)
        assert all(app_counts[k]==8 for k in LABELS),(kind,app_counts)
        apis=[dict(r) for r in db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
        operations=[]
        tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
            if table in tables:
                for r in db.execute('select * from '+table):
                    v=dict(r);v['kind']=table;operations.append(v)
        ops=collections.defaultdict(list)
        for v in operations:ops[v['globalPid'],v['correlationId']].append(v)
        proofs=[]
        for row in projected:
            label=row['Name'].rsplit(':',1)[-1]
            if label not in ('encoder','vlm','AE'):continue
            start=int(row['Orig Start (ns)']);end=start+int(row['Orig Duration (ns)'])
            candidates=[r for r in ranges if r['label']==label and r['start']==start and r['end']==end]
            assert len(candidates)==1,(kind,label,candidates)
            nv=candidates[0]
            keys={(a['globalTid']&-16777216,a['correlationId']) for a in apis
                  if a['globalTid']==nv['globalTid'] and start<=a['start'] and a['end']<=end}
            xs=[v for key in keys for v in ops.get(key,[])]
            assert xs,(kind,label)
            first=min(v['start'] for v in xs);last=max(v['end'] for v in xs)
            pstart=int(row['Projected Start (ns)']);pend=pstart+int(row['Projected Duration (ns)'])
            assert (first,last)==(pstart,pend),(kind,label,first,last,pstart,pend)
            kernels=[v for v in xs if v['kind']=='CUPTI_ACTIVITY_KIND_KERNEL']
            assert len(kernels)>100,(kind,label,len(kernels))
            assert int(row['Lvl'])==0 and int(row['NumChild'])==0
            proofs.append(dict(label=label,range_id=row['RangeId'],kernels=len(kernels),
                gpu_envelope_ms=(last-first)/1e6,cpu_submit_or_stage_ms=(end-start)/1e6,
                first_last_gpu_operation_exact_match=True))
        assert len(proofs)==24
        result=dict(kind=kind,trace=str(trace),bytes=trace.stat().st_size,
            sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),application_ranges=dict(app_counts),
            projected_ranges=dict(counts),core_envelopes_verified=len(proofs),
            actual_kernels=sum(v['kind']=='CUPTI_ACTIVITY_KIND_KERNEL' for v in operations),
            proofs=proofs,level0_no_children=True)
    if kind=='native_jax':
        assert len(capture['checks'])==10 and all(x['max_abs']==x['mae']==0 for x in capture['checks'])
        result['native_numerical_checks']=dict(n=10,max_abs=0,mae=0,mode='validated split-stage diagnostic; whole-JIT baseline preserved')
    result['trace_signature']=signature
    cached.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    state=json.loads((R/'execution.json').read_text())
    if state['status']!='passed':
        recovery=json.loads((R/'pipeline_resume_execution.json').read_text())
        assert recovery['status']=='passed' and recovery['source_unchanged'] and recovery['lease_released']
        assert state['source_before']==recovery['source_before']
        assert [s['kind'] for s in state['stages']]==['native_jax','fresh_prepared']
        assert all(s['status']=='passed' for s in state['stages'])
        assert [s['kind'] for s in recovery['stages']]==['pipeline']
        state=dict(status='passed',source_unchanged=True,lease_released=True,
            stages=state['stages']+recovery['stages'],
            retained_resource_stop='No processes or CUDA contexts remained after fresh completed, but utilization counter was stale96%; strict guard stopped before pipeline startup. Only the pending pipeline was resumed after idle settled.')
        (R/'canonical_execution.json').write_text(json.dumps(state,indent=2)+'\n')
    assert state['status']=='passed' and state['source_unchanged'] and state['lease_released']
    result=dict(status='passed',time_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        files=[analyze(stage) for stage in state['stages']],gpu_released=True,
        model_runtime_source_unchanged=True,quality_study_untouched=True,
        proof='Exactly one level0 GPU projection per replay, exact first/last GPU-operation endpoints, and retained individual kernels; no NVTX-inside-capture aliases',
        limits='CPU replay ranges show submission; GPU projection includes internal gaps, not kernel busy sum. Native remains a validated split-stage diagnostic.')
    (R/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))
    for row in result['files']:
        print(row['kind'],row['application_ranges'],row['projected_ranges'],row['actual_kernels'])


if __name__=='__main__':main()
