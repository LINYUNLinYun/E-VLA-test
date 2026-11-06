"""CPU-only validation and matched-work cross-frame trace summaries."""
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from preparations.idea2.layer_trace import overlap_metrics

BASE=Path('results/idea2/layer_pipeline_20261023')


def read_kernels(directory):
    with (directory/'kernels.csv').open() as f:
        rows=list(csv.DictReader(f))
    for row in rows:
        for key in ('process','device','context','green_context','stream','start','end','frame','epoch','slot','generation','iteration'):
            row[key]=int(row[key]) if row[key] else None
    return rows


def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main(name):
    directory=BASE/name
    report=json.loads((directory/'report.json').read_text())
    kernels=read_kernels(directory)
    graph=[r for r in kernels if r['branch'] and r['request_id']]
    requests=defaultdict(list)
    for r in graph:requests[(r['process'],r['epoch'],r['frame'])].append(r)
    evidence=[]
    same_frame=[]
    for (pid,epoch,frame),rows in sorted(requests.items()):
        ae=[r for r in rows if r['branch']=='AE']
        vlm=[r for r in rows if r['stage']=='VLM']
        same_frame.append(dict(process=pid,epoch=epoch,frame=frame,
            ae1_x_same_vlm_ms=overlap_metrics([r for r in ae if r['iteration']==1],vlm)['overlap_ms'],
            ae2_x_same_vlm_ms=overlap_metrics([r for r in ae if r['iteration']==2],vlm)['overlap_ms'],
            all_ae_x_same_vlm_ms=overlap_metrics(ae,vlm)['overlap_ms']))
        following=requests.get((pid,epoch,frame+1),[])
        nextvlm=[r for r in following if r['stage']=='VLM']
        nextvision=[r for r in following if r['stage']=='VisionEncoder']
        if not ae or not nextvlm:continue
        ae_last=max(r['end']for r in ae)
        vlm_first=min(r['start']for r in nextvlm)
        values=dict(process=pid,epoch=epoch,current_frame=frame,next_frame=frame+1,
            current_ae_last_kernel_end_ns=ae_last,next_vlm_first_kernel_start_ns=vlm_first,
            next_vlm_minus_current_ae_ms=(vlm_first-ae_last)/1e6,
            next_vlm_waits_for_current_ae=vlm_first>=ae_last,
            next_vision_first_kernel_start_ns=min(r['start']for r in nextvision),
            next_vision_last_kernel_end_ns=max(r['end']for r in nextvision))
        for iteration in (1,2):
            selected=[r for r in ae if r['iteration']==iteration]
            v=overlap_metrics(selected,nextvision)
            values[f'ae{iteration}_x_next_vision_ms']=v['overlap_ms']
            values[f'ae{iteration}_busy_ms']=v['left_busy_ms']
        values['all_ae_x_next_vision_ms']=overlap_metrics(ae,nextvision)['overlap_ms']
        evidence.append(values)
    write_csv(directory/'frame_dependency_evidence.csv',evidence)
    write_csv(directory/'same_frame_overlap.csv',same_frame)
    vision=defaultdict(list)
    for r in graph:
        if r['submit_kind']=='vision':vision[r['frame'],r['epoch'],r['slot'],r['generation']].append(r)
    vision_rows=[]
    for key,rows in sorted(vision.items()):
        hosts=sorted({r['host_request']for r in rows if r['host_request']})
        vision_rows.append(dict(frame=key[0],epoch=key[1],slot=key[2],generation=key[3],
            graph_kernels=len(rows),host_frames=[int(re.search(r'/frame_(\d+)',h)[1])for h in hosts],
            group_assignments=dict(Counter(r['group_assignment']for r in rows)),
            source_groups=sorted({r['source_group']for r in rows})))
    check=dict(all_graphs_mapped=report['attribution']['all_graph_stages_mapped']and report['attribution']['all_graph_requests_mapped'],
        no_unresolved_unscoped=report['attribution']['unresolved_unscoped_submissions']==0,
        eight_complete_frames=len(report['requests'])==8 and all(r['kernel_count']==4439 for r in report['requests']),
        seven_adjacent_frame_pairs=len(evidence)==7,
        next_vlm_not_before_current_ae=bool(evidence)and all(r['next_vlm_waits_for_current_ae']for r in evidence),
        prefetched_vision_logical_identity=all(r['host_frames']==[r['frame']-1]for r in vision_rows if r['host_frames']))
    stages=defaultdict(list)
    for req in report['requests']:
        for s in req['stages']:
            stages[s['stage'],s['iteration']].append(s)
    stage_summary=[dict(stage=stage,iteration=iteration,count=len(rows),
                       **{metric:statistics.mean(r[metric]for r in rows)for metric in
                          ('kernel_union_ms','kernel_envelope_ms','nonkernel_gap_ms')})
                   for (stage,iteration),rows in stages.items()]
    signatures=[]
    comparison=directory/'signature_comparison.json'
    if comparison.exists():
        c=json.loads(comparison.read_text());pairs=defaultdict(list)
        for r in c['matches']:pairs[r['stage'],r['iteration'],r['operation']].append(r)
        for (stage,iteration,op),rs in pairs.items():
            left=sum(r['baseline_mean_us']*r['baseline_calls']for r in rs)/8/1000
            right=sum(r['candidate_mean_us']*r['candidate_calls']for r in rs)/8/1000
            signatures.append(dict(stage=stage,iteration=iteration,operation=op,
                signature_count=len(rs),calls_equal=all(r['baseline_calls']==r['candidate_calls']for r in rs),
                serial_sum_elapsed_ms_per_request=left,candidate_sum_elapsed_ms_per_request=right,
                duration_ratio=right/left if left else None))
        check['all_signature_calls_equal']=all(r['calls_equal']for r in signatures)
        check['no_unmatched_signatures']=c['baseline_unmatched_types']==c['candidate_unmatched_types']==0
    summary=dict(profile_only=True,source=report['source'],checks=check,
        measured_logical_requests=len(report['requests']),adjacent_frame_pair_count=len(evidence),
        per_request_same_frame={key:dict(mean=statistics.mean(r[key]for r in same_frame),p50=statistics.median(r[key]for r in same_frame))
                                for key in ('ae1_x_same_vlm_ms','ae2_x_same_vlm_ms','all_ae_x_same_vlm_ms')},
        per_adjacent_pair={key:dict(mean=statistics.mean(r[key]for r in evidence),p50=statistics.median(r[key]for r in evidence))
                           for key in ('next_vlm_minus_current_ae_ms','ae1_x_next_vision_ms','ae2_x_next_vision_ms','all_ae_x_next_vision_ms')},
        vision_identity=vision_rows,stage_mean_per_request=stage_summary,matched_signatures=signatures,
        limitations=['Profiler durations are not steady request latency. Cross-run duration ratios may include GPU frequency/thermal and profiling drift.',
                    'Seven adjacent frame pairs are observable in eight requests; do not divide pair sums by eight when describing per-opportunity overlap.',
                    'Actual kernel interval intersection only. Nonkernel gap does not establish an exclusive Event-wait cause.'])
    (directory/'crossframe_findings.json').write_text(json.dumps(summary,indent=2))
    write_csv(directory/'matched_stage_signatures.csv',signatures)
    print(json.dumps({k:summary[k]for k in ('checks','per_adjacent_pair','stage_mean_per_request')},indent=2))
    if not all(check.values()):raise AssertionError(check)


if __name__=='__main__':main(sys.argv[1])
