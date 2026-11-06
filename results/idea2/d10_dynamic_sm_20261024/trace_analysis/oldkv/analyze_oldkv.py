"""CPU-only old-KV trace analysis; runtime Graph-launch attribution, no model imports.

Kernels inherit the enclosing request/module of their runtime launch API, not
the GPU timestamp's enclosing Host range. Thus asynchronous trailing VLM work
is retained after the response. No layer/iteration labels are inferred.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys

ROOT = Path('/root/nfs/embodied/benchmark')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
from preparations.idea2.layer_trace import GEOMETRY, RangeIndex, overlap_metrics, _busy, _envelope

DEFAULTS = (
    'd10_oldkv_graph_serial_20261024_v1',
    'd10_oldkv_graph_encoder_overlap_20261024_v1',
    'd10_oldkv_graph_ae_first_20261024_v1',
)
MODULE_LABELS = {'ae', 'prefill', 'prefill/vision', 'prefill/vlm'}
STAGES = ('AE', 'Vision', 'VLM', 'Prefill_other')
PAIRS = {'AE_x_Vision': ('Vision',), 'AE_x_VLM': ('VLM',),
         'AE_x_Prefill': ('Vision', 'VLM', 'Prefill_other')}


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
                         for k, v in row.items()} for row in rows)


def is_request(label):
    return label.startswith(('cewp/paired/', 'cewp/continuous/', 'cewp/isolated_'))


def classify(labels):
    # RangeIndex returns deepest first. Nested prefill/vlm/vision refine Prefill.
    for label in labels:
        if label == 'ae':
            return 'AE'
        if label == 'prefill/vision':
            return 'Vision'
        if label == 'prefill/vlm':
            return 'VLM'
    return 'Prefill_other' if 'prefill' in labels else None


def read_trace(source):
    db = sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {'StringIds', 'NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL',
                'CUPTI_ACTIVITY_KIND_RUNTIME'}
    if not required <= tables:
        raise ValueError(f'Missing tables: {sorted(required-tables)}')
    strings = dict(db.execute('SELECT id,value FROM StringIds'))
    ranges = []
    for r in db.execute('SELECT rowid,* FROM NVTX_EVENTS WHERE end IS NOT NULL'):
        label = r['text'] or strings.get(r['textId'], '')
        if label:
            ranges.append(dict(rowid=r['rowid'], start=r['start'], end=r['end'],
                               tid=r['globalTid'], label=label))
    requests = [r for r in ranges if is_request(r['label'])]
    req_index = RangeIndex(requests)
    module_index = RangeIndex([r for r in ranges if r['label'] in MODULE_LABELS])
    APIs = defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        for r in db.execute(f'SELECT rowid,* FROM {table}'):
            if r['correlationId'] is not None:
                APIs[r['globalTid'] & -16777216, r['correlationId']].append(dict(
                    table=table, rowid=r['rowid'], start=r['start'], end=r['end'],
                    tid=r['globalTid'], name=strings.get(r['nameId'], str(r['nameId']))))
    launch_cache, kernels = {}, []
    for r in db.execute('SELECT rowid,* FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start,rowid'):
        raw = dict(r)
        key = (r['globalPid'], r['correlationId'])
        if key not in launch_cache:
            choices = APIs.get(key, [])
            candidates = [a for a in choices if a['table'] == 'CUPTI_ACTIVITY_KIND_RUNTIME'] or choices
            # Do not guess a nearby API if correlation is absent or ambiguous.
            api = candidates[0] if len(candidates) == 1 else None
            issue = ('missing_launch_correlation' if not candidates else
                     'ambiguous_launch_correlation' if len(candidates) > 1 else None)
            hosts = req_index.containing(api['tid'], api['start'], api['end']) if api else []
            labels = [x['label'] for x in module_index.containing(api['tid'], api['start'], api['end'])] if api else []
            launch_cache[key] = dict(api=api, issue=issue, host=hosts[0] if hosts else None,
                                     stage=classify(labels), module_labels=labels)
        mapped = launch_cache[key]
        api, host, stage = mapped['api'], mapped['host'], mapped['stage']
        request_id = f"{r['globalPid']}|host_nvtx_row={host['rowid']}" if host else None
        labels = host['label'].split('/') if host else []
        graph_node = raw.get('graphNodeId')
        errors = [mapped['issue']] if mapped['issue'] else []
        if host and stage is None:
            errors.append('request_kernel_outside_module_range')
        kernels.append(dict(kernel_rowid=r['rowid'], start=r['start'], end=r['end'],
            duration_us=(r['end']-r['start'])/1000, process=r['globalPid'], device=r['deviceId'],
            context=r['contextId'], green_context=raw.get('greenContextId'), stream=r['streamId'],
            graph=raw.get('graphId'), graph_node=graph_node, correlation=r['correlationId'],
            name=strings.get(r['demangledName'], str(r['demangledName'])),
            geometry=[raw.get(k) for k in GEOMETRY], stage=stage,
            branch='AE' if stage=='AE' else 'Prefill' if stage else None,
            request_id=request_id, frame_id=request_id, group='/'.join(labels[:3]) if host else None,
            phase=labels[1] if host else None, mode=labels[2] if host else None,
            request_label=host['label'] if host else None,
            host_start=host['start'] if host else None, host_end=host['end'] if host else None,
            launch_start=api['start'] if api else None, launch_end=api['end'] if api else None,
            launch_name=api['name'] if api else None, launch_table=api['table'] if api else None,
            module_labels=mapped['module_labels'], mapping_errors=errors))
    api_stats = [dict(r) for r in db.execute(
        "SELECT s.value AS name,count(*) AS calls,sum(r.end-r.start)/1e6 AS host_api_ms "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id "
        "GROUP BY r.nameId ORDER BY host_api_ms DESC")]
    contexts = [dict(r) for r in db.execute('SELECT * FROM TARGET_INFO_CUDA_CONTEXT_INFO')] if 'TARGET_INFO_CUDA_CONTEXT_INFO' in tables else []
    db.close()
    return kernels, requests, dict(tables=sorted(tables), geometry_fields=list(GEOMETRY),
                                  runtime_api=api_stats, contexts=contexts)


def select_stage(rows, stage):
    return [r for r in rows if r['branch']=='Prefill'] if stage=='Prefill' else [r for r in rows if r['stage']==stage]


def signatures(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['phase'], r['stage'], r['name'], tuple(r['geometry'])].append(r['duration_us'])
    return [dict(phase=p, stage=s, name=n, geometry=list(g), calls=len(v),
                 mean_us=statistics.mean(v), p50_us=statistics.median(v), sum_kernel_elapsed_ms=sum(v)/1000)
            for (p,s,n,g),v in sorted(grouped.items(), key=lambda x:-sum(x[1]))]


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=True)
    kernels, hosts, schema = read_trace(source)
    scoped = [r for r in kernels if r['request_id']]
    model = [r for r in scoped if r['stage']]
    by_request, by_group = defaultdict(list), defaultdict(list)
    for r in model:
        by_request[r['request_id']].append(r)
        by_group[r['group']].append(r)
    req_rows, module_rows = [], []
    for req, rows in sorted(by_request.items(), key=lambda x:min(r['host_start'] for r in x[1])):
        first = rows[0]
        late = [r for r in rows if r['end'] > first['host_end']]
        req_rows.append(dict(request_id=req, group=first['group'], request_label=first['request_label'],
            kernel_count=len(rows), graph_kernel_count=sum(r['graph_node'] not in (0,None) for r in rows),
            kernel_busy_ms=_busy(rows), kernel_envelope_ms=_envelope(rows),
            model_end_after_host_response_ms=(max(r['end'] for r in rows)-first['host_end'])/1e6,
            kernels_ending_after_host_response=len(late),
            prefill_kernels_ending_after_host_response=sum(r['branch']=='Prefill' for r in late)))
        for stage in ('AE','Prefill','Vision','VLM','Prefill_other'):
            selected = select_stage(rows, stage)
            if not selected:
                continue
            launches = {(r['process'],r['correlation']) for r in selected if 'GraphLaunch' in (r['launch_name'] or '')}
            module_rows.append(dict(request_id=req, group=first['group'], stage=stage,
                kernels=len(selected), graph_launches=len(launches), kernel_busy_ms=_busy(selected),
                gpu_envelope_ms=_envelope(selected),
                module_end_after_host_response_ms=(max(r['end'] for r in selected)-first['host_end'])/1e6))
    groups = {}
    for group, rows in by_group.items():
        modules = []
        for stage in ('AE','Prefill','Vision','VLM','Prefill_other'):
            selected = [r for r in module_rows if r['group']==group and r['stage']==stage]
            modules.append(dict(stage=stage, requests=len(selected),
                kernel_busy_mean_ms=statistics.mean(r['kernel_busy_ms'] for r in selected) if selected else None,
                gpu_envelope_mean_ms=statistics.mean(r['gpu_envelope_ms'] for r in selected) if selected else None,
                kernel_count=sum(r['kernels'] for r in selected)))
        ae = [r for r in rows if r['stage']=='AE']
        groups[group] = dict(requests=len({r['request_id'] for r in rows}), modules=modules,
            pairs={p:overlap_metrics(ae,[r for r in rows if r['stage'] in stages]) for p,stages in PAIRS.items()})
    graph = [r for r in kernels if r['graph_node'] not in (0,None)]
    scoped_graph = [r for r in scoped if r['graph_node'] not in (0,None)]
    all_module = [r for r in kernels if r['stage']]
    graph_launches = defaultdict(list)
    for r in kernels:
        if 'GraphLaunch' in (r['launch_name'] or ''):
            graph_launches[r['process'],r['correlation']].append(r)
    launch_rows = []
    for (pid,corr), rows in graph_launches.items():
        first = rows[0]
        launch_rows.append({k:first[k] for k in ('request_id','group','stage','module_labels','launch_name',
            'launch_start','launch_end','host_end')} | dict(process=pid,correlation=corr,
            kernels=len(rows),first_kernel=min(r['start'] for r in rows),last_kernel=max(r['end'] for r in rows)))
    attribution = dict(total_kernels=len(kernels), graph_kernels=len(graph), scoped_kernels=len(scoped),
        scoped_graph_kernels=len(scoped_graph), scoped_model_kernels=len(model),
        unscoped_kernels=len(kernels)-len(scoped), unscoped_graph_kernels=len(graph)-len(scoped_graph),
        module_kernels_without_request=sum(not r['request_id'] for r in all_module),
        scoped_graph_missing_module=sum(not r['stage'] for r in scoped_graph),
        scoped_nongraph_outside_module=sum(not r['stage'] for r in scoped if r['graph_node'] in (0,None)),
        missing_launch_correlation=sum('missing_launch_correlation' in r['mapping_errors'] for r in kernels),
        ambiguous_launch_correlation=sum('ambiguous_launch_correlation' in r['mapping_errors'] for r in kernels),
        all_scoped_graph_modules_mapped=bool(scoped_graph) and all(r['stage'] for r in scoped_graph),
        request_ranges=len(hosts), attributed_requests=len(by_request),
        model_kernels_after_host_response=sum(r['end']>r['host_end'] for r in model),
        prefill_kernels_after_host_response=sum(r['end']>r['host_end'] for r in model if r['branch']=='Prefill'),
        graph_resource_counts=[dict(stage=s,device=d,context=c,green_context=g,kernels=n)
            for (s,d,c,g),n in Counter((r['stage'],r['device'],r['context'],r['green_context']) for r in scoped_graph).items()])
    report = dict(source=str(source.resolve()), attribution=attribution, schema=schema, groups=groups,
        requests=req_rows, request_modules=module_rows, signatures=signatures(model),
        definition='Runtime/driver API correlation identifies the Host request and deepest ae/prefill/vision/vlm NVTX. Actual GPU intervals are never clipped to Host ranges. Vision means the entire prefill/vision Graph, including image encoding/projector, language embedding and mask/position metadata preparation; it is not pure encoder arithmetic. Prefill_other retains non-stage work. No layer or AE-iteration attribution is inferred. Same/cross-frame fields mean same/different Host submission request, not KV observation identity. Overlap counts same physical GPU distinct streams, including distinct Green contexts. Attention stages cannot be inferred from these module labels. Profile timings are diagnostic only.',
        input_preparation_scope='Whole-trace API statistics include input, output and lifecycle waits; no pure-CPU preparation attribution is claimed.')
    save(output/'report.json',report)
    write_csv(output/'kernels.csv',kernels)
    write_csv(output/'requests.csv',req_rows)
    write_csv(output/'request_modules.csv',module_rows)
    write_csv(output/'kernel_signatures.csv',report['signatures'])
    write_csv(output/'graph_launch_map.csv',launch_rows)
    return report


def compare(baseline, candidate):
    key = lambda r:(r['phase'],r['stage'],r['name'],tuple(r['geometry']))
    a,b = ({key(r):r for r in report['signatures']} for report in (baseline,candidate))
    matches=[]
    for k in sorted(a.keys() & b.keys(),key=str):
        x,y=a[k],b[k]
        calls=min(x['calls'],y['calls'])
        matches.append(dict(phase=k[0],stage=k[1],name=k[2],geometry=list(k[3]),
            baseline_calls=x['calls'],candidate_calls=y['calls'],common_calls=calls,
            baseline_mean_us=x['mean_us'],candidate_mean_us=y['mean_us'],
            candidate_over_baseline=y['mean_us']/x['mean_us'] if x['mean_us'] else None))
    summary=[]
    for phase,stage in sorted({(r['phase'],r['stage']) for r in matches}):
        for category in ('All_matched','GEMM_named_kernels'):
            rows=[r for r in matches if r['phase']==phase and r['stage']==stage
                  and (category=='All_matched' or re.search(r'gemm|gemv|matmul',r['name'],re.I))]
            if not rows:
                continue
            x=sum(r['baseline_mean_us']*r['common_calls'] for r in rows)
            y=sum(r['candidate_mean_us']*r['common_calls'] for r in rows)
            summary.append(dict(phase=phase,stage=stage,category=category,signature_types=len(rows),
                common_calls=sum(r['common_calls'] for r in rows),
                count_mismatched_types=sum(r['baseline_calls']!=r['candidate_calls'] for r in rows),
                weighted_baseline_ms=x/1000,weighted_candidate_ms=y/1000,candidate_over_baseline=y/x))
    return dict(matches=matches,summary=summary,
        baseline_unmatched=[a[k] for k in a.keys()-b.keys()],candidate_unmatched=[b[k] for k in b.keys()-a.keys()],
        definition='Exact phase/module/name/launch-geometry signature; modes differ. Weighted by common call counts. No layer/iteration or pure-Attention slowdown claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces',nargs='*',default=list(DEFAULTS),help='Basenames; first must be the serial baseline.')
    parser.add_argument('--trace-dir',type=Path,default=ROOT/'nsys/idea2_experiment')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'analysis')
    parser.add_argument('--skip-missing',action='store_true')
    parser.add_argument('--no-export',action='store_true',help='Require existing SQLite; never invoke nsys export.')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    available,missing=[],[]
    for name in args.traces:
        source=args.trace_dir/(name+'.sqlite')
        rep=args.trace_dir/(name+'.nsys-rep')
        if not source.exists() and (args.no_export or not rep.exists()):
            missing.append(name)
        else:
            available.append((name,source,rep))
    if missing and not args.skip_missing:
        parser.error('Missing completed trace/SQLite: '+', '.join(missing))
    reports,receipts={},[]
    for name,source,rep in available:
        dest=args.output/name
        dest.mkdir(parents=True,exist_ok=True)
        reused=source.exists()
        if not reused:
            command=['nsys','export','--type','sqlite','--output',str(source),str(rep)]
            with (dest/'export.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
        report=analyze(source,dest)
        reports[name]=report
        receipts.append(dict(name=name,source=str(source),export_reused=reused,attribution=report['attribution']))
        print(json.dumps(dict(name=name,attribution=report['attribution'])),flush=True)
    baseline=reports.get(args.traces[0])
    comparisons={}
    if baseline:
        for name,report in reports.items():
            if name==args.traces[0]:
                continue
            comparison=compare(baseline,report)
            comparisons[name]=comparison
            save(args.output/name/'signature_comparison.json',comparison)
            write_csv(args.output/name/'signature_comparison.csv',comparison['matches'])
            write_csv(args.output/name/'slowdown_summary.csv',comparison['summary'])
    aggregate=[]
    lines=['# Old-KV D10 trace analysis','',
        'CPU-only analysis of existing traces. GPU intervals are never clipped to the Host response. No layer/iteration labels inferred.','',
        '| Trace / group | Requests | AE busy ms | Vision busy ms | VLM busy ms | AE×Vision ms/request | AE×VLM ms/request |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,report in reports.items():
        for group,data in report['groups'].items():
            modules={r['stage']:r for r in data['modules']}
            row=dict(case=name,group=group,requests=data['requests'])
            for stage in ('AE','Prefill','Vision','VLM','Prefill_other'):
                row[stage+'_busy_mean_ms']=modules[stage]['kernel_busy_mean_ms']
                row[stage+'_envelope_mean_ms']=modules[stage]['gpu_envelope_mean_ms']
            for pair,value in data['pairs'].items():
                row[pair+'_total_ms']=value['overlap_ms']
                row[pair+'_per_request_ms']=value['overlap_ms']/data['requests']
            aggregate.append(row)
            fmt=lambda x:'n/a' if x is None else f'{x:.3f}'
            lines.append(f"| {name} / {group} | {data['requests']} | {fmt(row['AE_busy_mean_ms'])} | {fmt(row['Vision_busy_mean_ms'])} | {fmt(row['VLM_busy_mean_ms'])} | {fmt(row['AE_x_Vision_per_request_ms'])} | {fmt(row['AE_x_VLM_per_request_ms'])} |")
    lines+=['','All measured and unscoped kernels are retained in kernels.csv. Completeness, trailing VLM kernels, Graph launch mapping, resources and API statistics are in each report.json. Prefill_other is retained rather than guessed as Vision/VLM. A missing module is not silently treated as measured zero.','',
            'Matched signature comparisons use phase + module + kernel name + launch geometry; verify unmatched signatures and call-count differences before interpreting slowdown. These short profile measurements are not formal latency/throughput results.','',
            'Host request identity is a submission identity, not the KV observation identity; numerical equivalence and old-KV freshness need the benchmark result.json, not this trace alone.']
    if missing:
        lines+=['','Missing traces: '+', '.join(missing)]
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    write_csv(args.output/'module_overlap_summary.csv',aggregate)
    save(args.output/'receipt.json',dict(cpu_only=True,missing=missing,baseline=args.traces[0],cases=receipts,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__=='__main__':
    main()
