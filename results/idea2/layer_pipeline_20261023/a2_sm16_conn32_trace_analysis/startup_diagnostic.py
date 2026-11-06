"""CPU-only derived evidence; run from repository root after layer_trace."""
import csv
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('results/idea2/layer_pipeline_20261023')


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows, summaries = [], []
    for name, profile in [('a2_sm16_trace_analysis', 'a2_sm16_profile'),
                          ('a2_sm16_conn32_trace_analysis', 'a2_sm16_conn32_profile')]:
        directory = BASE/name
        report = json.loads((directory/'report.json').read_text())
        result = json.loads((BASE/profile/'result.json').read_text())
        bench = {(r['epoch'], r['request'], r['slot'], r['generation']): r
                 for r in result['rows'] if r['phase'] == 'continuous'}
        by_request = defaultdict(list)
        with (directory/'kernels.csv').open() as f:
            for row in csv.DictReader(f):
                if row['branch'] and row['request_id']:
                    for key in ('start', 'end', 'launch_start', 'launch_end', 'frame', 'epoch', 'slot', 'generation'):
                        row[key] = int(row[key])
                    by_request[row['request_id']].append(row)
        derived = []
        for identity, kernels in by_request.items():
            ae = [r for r in kernels if r['branch'] == 'AE']
            vlm = [r for r in kernels if r['stage'] == 'VLM']
            pre = [r for r in kernels if r['branch'] == 'Prefill']
            first = min(ae, key=lambda r: r['start'])
            origin = min(r['start'] for r in pre)
            measured = bench[tuple(first[k] for k in ('epoch', 'frame', 'slot', 'generation'))]
            derived.append(dict(analysis=name, request_id=identity,
                first_ae_kernel_after_first_prefix_kernel_ms=(first['start']-origin)/1e6,
                first_ae_iteration1_after_first_prefix_kernel_ms=(min(r['start'] for r in ae if r['iteration']=='1')-origin)/1e6,
                first_vlm_kernel_after_first_prefix_kernel_ms=(min(r['start'] for r in vlm)-origin)/1e6,
                first_ae_kernel_after_first_vlm_kernel_ms=(first['start']-min(r['start'] for r in vlm))/1e6,
                vlm_kernel_fraction_completed_before_first_ae=sum(r['end']<=first['start'] for r in vlm)/len(vlm),
                ae_begin_event_after_prefix_begin_ms=(measured['ae_completion_event_ms']-measured['ae_gpu_event_ms'])-(measured['prefill_completion_event_ms']-measured['prefill_gpu_event_ms']),
                ae_graph_host_launch_start_after_first_prefix_kernel_ms=(first['launch_start']-origin)/1e6,
                ae_graph_host_launch_end_after_first_prefix_kernel_ms=(first['launch_end']-origin)/1e6,
                first_ae_kernel_after_graph_launch_return_ms=(first['start']-first['launch_end'])/1e6))
        all_rows.extend(derived)
        statistics_by_metric = {key: dict(mean=statistics.mean(r[key] for r in derived),
                                         p50=statistics.median(r[key] for r in derived))
                                for key in derived[0] if key not in ('analysis', 'request_id')}
        group = next(iter(report['groups'].values()))
        connection = result.get('environment', {}).get('CUDA_DEVICE_MAX_CONNECTIONS', 'unrecorded')
        db = sqlite3.connect(Path(report['source']).as_uri()+'?mode=ro', uri=True)
        db.row_factory = sqlite3.Row
        contexts = [dict(r) for r in db.execute('select * from TARGET_INFO_CUDA_CONTEXT_INFO')]
        db.close()
        summaries.append(dict(analysis=name, profile_result=str(BASE/profile/'result.json'),
            trace=report['source'], environment=result.get('environment'),
            CUDA_DEVICE_MAX_CONNECTIONS=connection, request_count=len(derived),
            statistics=statistics_by_metric,
            overlap_per_request_ms={k:v['overlap_ms']/len(derived) for k,v in group['pairs'].items()},
            context_metadata=contexts, streams=report['streams'],
            attribution=report['attribution']))
    output = dict(scope='Separate profiling runs, old 5 requests with unrecorded connections environment; new 8 requests with explicit 32. No steady performance inference from trace durations.',
        clock_semantics='Kernel/API timing uses nsys-normalized CUPTI timestamps. Negative first-kernel-minus-launch-return is valid asynchronous execution during the API call. Event-start offset is separate GPU Event evidence, not first kernel.',
        causal_limit='Trace localizes changed startup scheduling; hardware workqueue identity/count is not established by stream IDs or zero workqueue metadata. Formal explicit 8/32 results provide separate single-setting evidence.',
        steady_evidence=str(BASE/'formal_resource_comparison.json'), summaries=summaries, rows=all_rows)
    destination=BASE/'a2_sm16_conn32_trace_analysis'
    (destination/'startup_diagnostic.json').write_text(json.dumps(output,indent=2))
    write_csv(destination/'startup_diagnostic.csv',all_rows)
    old=BASE/'green_marker_trace_comparison.json'
    merged=json.loads(old.read_text())
    merged['rows']=[r for r in merged['rows'] if r['analysis']!='a2_sm16_conn32_trace_analysis']+[
        r for r in all_rows if r['analysis']=='a2_sm16_conn32_trace_analysis']
    merged['conn32_diagnostic']=output
    old.write_text(json.dumps(merged,indent=2))
    write_csv(BASE/'green_marker_trace_comparison.csv',merged['rows'])
    print(json.dumps([{k:v for k,v in s.items() if k in ('analysis','request_count','CUDA_DEVICE_MAX_CONNECTIONS','statistics','overlap_per_request_ms')} for s in summaries],indent=2))


if __name__=='__main__':
    main()
