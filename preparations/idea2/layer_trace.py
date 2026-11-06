"""CPU-only nsys SQLite attribution for same-frame layer/encoder pipelines.

Capture NVTX labels identify graph nodes through clone ancestry; runtime launch
correlations identify the logical frame that replayed them. GPU kernel intervals
alone define busy/overlap. Host NVTX, Events, memcopies and memsets never do.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics


GEOMETRY = ('gridX','gridY','gridZ','blockX','blockY','blockZ',
            'registersPerThread','staticSharedMemory','dynamicSharedMemory')
PAIRS = {'AE_x_VLM': ('VLM',), 'AE_x_Vision': ('VisionEncoder','VisionProjector'),
         'AE_x_VisionEncoder': ('VisionEncoder',), 'AE_x_VisionProjector': ('VisionProjector',),
         'AE_x_Prefill': ('VLM','VisionEncoder','VisionProjector','MetadataEmbedding')}


def process_id(global_tid):
    return int(global_tid) & -16777216 if global_tid is not None else None


def merged(intervals):
    output=[]
    for begin,end in sorted(intervals):
        if end<=begin:
            continue
        if output and begin<=output[-1][1]:
            output[-1][1]=max(output[-1][1],end)
        else:
            output.append([begin,end])
    return output


def interval_duration(intervals):
    return sum(e-b for b,e in merged(intervals))


class RangeIndex:
    """Thread-local containment index; handles nested and completed siblings."""
    def __init__(self,ranges):
        self.rows=defaultdict(list)
        for row in ranges:
            self.rows[row['tid']].append(row)
        self.starts,self.max_ends={},{}
        for tid,rows in self.rows.items():
            rows.sort(key=lambda r:(r['start'],-r['end']))
            self.starts[tid]=[r['start'] for r in rows]
            high=-float('inf'); prefix=[]
            for row in rows:
                high=max(high,row['end']);prefix.append(high)
            self.max_ends[tid]=prefix

    def containing(self,tid,begin,end=None):
        end=begin if end is None else end
        rows=self.rows.get(tid,())
        i=bisect_right(self.starts.get(tid,()),begin)-1
        output=[]
        while i>=0 and self.max_ends[tid][i]>=end:
            row=rows[i]
            if row['start']<=begin and row['end']>=end:
                output.append(row)
            i-=1
        return sorted(output,key=lambda r:(r['end']-r['start'],-r['start'],r['rowid']))


def classify_labels(labels):
    """Exclusive deepest operation, with independently retained iteration/layer.

    Attention_non_projection includes RoPE/reshape/repeat-KV/masking inside the
    self-attention wrapper. It is NOT claimed to be pure attention arithmetic.
    """
    semantic=[x for x in labels if x.startswith(('Prefill','AE/'))]
    if not semantic:
        return dict(branch=None,stage=None,operation=None,layer=None,iteration=None,label=None)
    label=semantic[0]
    iteration=None
    for text in semantic:
        match=re.search(r'AE/iteration_(\d+)',text)
        legacy=re.search(r'AE/Denoise/step_(\d+)',text)
        if match or legacy:
            iteration=int(match[1]) if match else int(legacy[1])+1
            break
    layer=next((int(m[1]) for text in semantic if (m:=re.search(r'/layer_(\d+)',text))),None)
    branch='AE' if any(x.startswith('AE/') for x in semantic) else 'Prefill'
    if branch=='AE':
        stage='AE'
    elif any(x=='Prefill/VisionEncoder' or x.startswith('Prefill/VisionEncoder/') for x in semantic):
        stage='VisionEncoder'
    elif any(x=='Prefill/VisionProjector' or x.startswith('Prefill/VisionProjector/') for x in semantic):
        stage='VisionProjector'
    elif any(x.startswith(('Prefill/VLM/','Prefill/VLMDecoder')) or x=='Prefill/VLM' for x in semantic):
        stage='VLM'
    else:
        stage='MetadataEmbedding'
    projection=next((x.rsplit('/',1)[-1] for x in semantic
                     if x.rsplit('/',1)[-1] in ('q_proj','k_proj','v_proj','o_proj')),None)
    if projection:
        operation={'q_proj':'Q','k_proj':'K','v_proj':'V','o_proj':'O'}[projection]
    elif any(x.endswith('/FFN') for x in semantic):
        operation='FFN'
    elif any(x.endswith('/Attention') for x in semantic):
        operation='Attention_non_projection'
    elif layer is not None:
        operation='Layer_other'
    elif stage=='AE':
        operation='Iteration_other' if iteration is not None else 'Preparation_or_Euler_update'
    else:
        operation='Other'
    return dict(branch=branch,stage=stage,operation=operation,layer=layer,
                iteration=iteration,label=label)


def parse_submission(label):
    match=re.match(r'^submit/(vision|vlm|ae)/',label or '')
    if not match:
        return None
    fields=dict(re.findall(r'(frame|epoch|slot|gen)=([^/]+)',label))
    if set(fields)!={'frame','epoch','slot','gen'}:
        return None
    return dict(kind=match[1],**{key:int(value) if re.fullmatch(r'-?\d+',value) else value
                                for key,value in fields.items()})


def section_group(label):
    if label.startswith(('layer/','crossframe/','cewp/')):
        return '/'.join(label.split('/')[:3])
    return 'unscoped_submit'


def is_request(label):
    return label.startswith(('layer/','crossframe/','cewp/continuous/','cewp/paired/','cewp/isolated_'))


def reconcile_unscoped_crossframe(kernels):
    """Attach out-of-range Encoder submits without inventing a Host range.

    Prefer an unambiguous identical submit identity. A single scoped group in
    this process permits a weaker explicit fallback (e.g. trailing lookahead).
    Mixed-mode/phase ambiguity remains unscoped, never guessed from timestamps.
    """
    by_identity=defaultdict(set)
    by_process=defaultdict(set)
    identity=lambda r:tuple(r[k] for k in ('process','epoch','frame','slot','generation'))
    for row in kernels:
        row['source_group']=row['group']
        row['host_section_missing']=row['host_request'] is None
        row['group_assignment']='host_section' if row['host_request'] else 'unscoped'
        if row['host_request']:
            by_process[row['process']].add(row['group'])
            if row['submit_kind']:
                by_identity[identity(row)].add(row['group'])
    for row in kernels:
        if row['group']!='unscoped_submit' or not row['submit_kind']:
            continue
        candidates=by_identity[identity(row)]
        reason='matching_submit_identity'
        if not candidates:
            candidates=by_process[row['process']]
            reason='unique_crossframe_process_group'
        if len(candidates)!=1:
            continue
        group=next(iter(candidates))
        if not group.startswith('crossframe/'):
            continue
        row['group']=group
        row['group_assignment']=reason
        row['request_id']=(f'{row["process"]}|{group}|epoch={row["epoch"]}/frame={row["frame"]}'
                           f'/slot={row["slot"]}/gen={row["generation"]}')
        row['frame_id']=f'{row["process"]}|{group}|epoch={row["epoch"]}/frame={row["frame"]}'


def _read_database(path):
    db=sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    required={'NVTX_EVENTS','StringIds','CUPTI_ACTIVITY_KIND_KERNEL'}
    if not required<=tables:
        raise ValueError(f'Missing nsys tables: {sorted(required-tables)}')
    strings=dict(db.execute('select id,value from StringIds'))
    ranges=[]
    for r in db.execute('select rowid,* from NVTX_EVENTS where end is not null'):
        label=r['text'] or strings.get(r['textId'],'')
        if label:
            ranges.append(dict(rowid=r['rowid'],start=r['start'],end=r['end'],tid=r['globalTid'],label=label))
    captures=RangeIndex([r for r in ranges if r['label'].startswith(('Prefill','AE/'))])
    submits=RangeIndex([r for r in ranges if r['label'].startswith('submit/')])
    sections=RangeIndex([r for r in ranges if is_request(r['label'])])
    apis=defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in tables:
            continue
        for r in db.execute(f'select rowid,* from {table}'):
            if r['correlationId'] is not None:
                apis[process_id(r['globalTid']),r['correlationId']].append(dict(
                    start=r['start'],end=r['end'],tid=r['globalTid'],table=table,
                    rowid=r['rowid'],name=strings.get(r['nameId'],str(r['nameId']))))
    origins,links={},{}
    conflicts=[]
    if 'CUDA_GRAPH_NODE_EVENTS' in tables:
        for r in db.execute('select * from CUDA_GRAPH_NODE_EVENTS'):
            key=(process_id(r['globalTid']),r['graphNodeId'])
            if r['originalGraphNodeId'] is None:
                # A clone can also have a creation entry. Link ancestry wins below.
                origins.setdefault(key,dict(start=r['start'],end=r['end'],tid=r['globalTid']))
            else:
                source=(key[0],r['originalGraphNodeId'])
                if key in links and links[key]!=source:
                    conflicts.append(dict(node=key,first=links[key],second=source))
                links[key]=source
    ancestry={}

    def original(key):
        if key in ancestry:
            return ancestry[key]
        visited=[];seen=set();current=key
        while current in links:
            if current in seen:
                result=(None,'graph_ancestry_cycle');break
            if current in ancestry:
                result=ancestry[current];break
            visited.append(current);seen.add(current);current=links[current]
        else:
            result=(current,None) if current in origins else (None,'missing_original_graph_node')
        for item in [key,*visited]:
            ancestry[item]=result
        return result

    capture_cache={}
    launch_cache={}
    kernels=[]
    for raw in db.execute('select rowid,* from CUPTI_ACTIVITY_KIND_KERNEL order by start,rowid'):
        r=dict(raw);pid=r.get('globalPid');graph_id=r.get('graphNodeId')
        errors=[];capture_labels=[];source_id=None
        if graph_id not in (None,0):
            key=(pid,graph_id)
            if key not in capture_cache:
                source,error=original(key)
                labels=[]
                if source:
                    node=origins[source]
                    labels=[x['label'] for x in captures.containing(node['tid'],node['start'])]
                capture_cache[key]=(source,error,labels)
            source,error,capture_labels=capture_cache[key]
            source_id=source[1] if source else None
            if error: errors.append(error)
        launch_key=(pid,r.get('correlationId'))
        matches=apis.get(launch_key,[])
        runtime=[x for x in matches if x['table']=='CUPTI_ACTIVITY_KIND_RUNTIME']
        candidates=runtime or matches
        if len(candidates)==1:
            launch=candidates[0]
        elif candidates:
            possible=[x for x in candidates if x['start']<=r['start']]
            launch=max(possible,key=lambda x:x['start']) if possible else None
            errors.append('ambiguous_launch_correlation')
        else:
            launch=None;errors.append('missing_launch_correlation')
        submit=None;host=None
        if launch:
            cache_key=(launch['table'],launch['rowid'])
            if cache_key not in launch_cache:
                ranges_submit=submits.containing(launch['tid'],launch['start'],launch['end'])
                ranges_host=sections.containing(launch['tid'],launch['start'],launch['end'])
                launch_cache[cache_key]=(ranges_submit[0] if ranges_submit else None,
                                         ranges_host[0] if ranges_host else None)
            submit,host=launch_cache[cache_key]
        parsed=parse_submission(submit['label']) if submit else None
        group=section_group(host['label']) if host else ('unscoped_submit' if parsed else None)
        if parsed:
            frame,epoch,slot,generation=(parsed[k] for k in ('frame','epoch','slot','gen'))
            request_id=f'{pid}|{group}|epoch={epoch}/frame={frame}/slot={slot}/gen={generation}'
            frame_id=f'{pid}|{group}|epoch={epoch}/frame={frame}'
            label_format='layer_submit'
        elif host:
            frame=epoch=slot=generation=None
            request_id=frame_id=f'{pid}|legacy_host_range={host["rowid"]}'
            label_format='legacy_host_request'
        else:
            frame=epoch=slot=generation=None
            request_id=frame_id=None;label_format='unassigned'
        classification=classify_labels(capture_labels)
        if graph_id not in (None,0) and classification['branch'] is None:
            errors.append('missing_capture_stage_label')
        # Non-Graph kernels may use eager operation NVTX, but are not guessed from
        # neighbouring streams. Input/copy helpers outside labels remain visible.
        if graph_id in (None,0) and launch:
            eager_labels=[x['label'] for x in captures.containing(launch['tid'],launch['start'],launch['end'])]
            eager=classify_labels(eager_labels)
            if eager['branch']:
                classification=eager;capture_labels=eager_labels
        kernels.append(dict(kernel_rowid=r['rowid'],start=r['start'],end=r['end'],
            duration_us=(r['end']-r['start'])/1000,process=pid,device=r['deviceId'],context=r['contextId'],
            green_context=r.get('greenContextId'),stream=r['streamId'],
            correlation=r.get('correlationId'),graph_node=graph_id,original_graph_node=source_id,
            name=strings.get(r['demangledName'],str(r['demangledName'])),
            geometry=[r.get(key) for key in GEOMETRY],**classification,
            capture_labels=capture_labels,request_id=request_id,frame_id=frame_id,
            frame=frame,epoch=epoch,slot=slot,generation=generation,group=group,label_format=label_format,
            submit_kind=parsed['kind'] if parsed else None,submit_label=submit['label'] if submit else None,
            host_request=host['label'] if host else None,host_request_rowid=host['rowid'] if host else None,
            launch_start=launch['start'] if launch else None,launch_end=launch['end'] if launch else None,
            launch_name=launch['name'] if launch else None,launch_table=launch['table'] if launch else None,
            mapping_errors=errors))
    db.close()
    reconcile_unscoped_crossframe(kernels)
    return kernels,dict(tables=sorted(tables),graph_origin_count=len(origins),graph_clone_link_count=len(links),
                        graph_link_conflicts=conflicts,geometry_fields=list(GEOMETRY))


def _stream(row):
    # Different green contexts remain on the SAME physical GPU. Never exclude
    # their overlap by grouping on greenContextId.
    return row['context'],row['green_context'],row['stream']


def overlap_metrics(left,right):
    """Unique interval measure, with same-frame and cross-frame attribution.

    A time segment can simultaneously have same- and cross-frame pairs; those
    subsets are reported separately and their union defines total overlap.
    Device/process time domains are measured separately before summing.
    """
    devices=defaultdict(lambda:[[],[]])
    for side,rows in enumerate((left,right)):
        for row in rows:
            devices[row['process'],row['device']][side].append(row)
    totals=Counter();pair_times=Counter();same_stream=0
    for _,(a,b) in devices.items():
        totals['left_busy_ns']+=interval_duration((r['start'],r['end']) for r in a)
        totals['right_busy_ns']+=interval_duration((r['start'],r['end']) for r in b)
        totals['kernel_union_ns']+=interval_duration((r['start'],r['end']) for r in a+b)
        events=defaultdict(list)
        for side,rows in enumerate((a,b)):
            for row in rows:
                key=(row['frame_id'],_stream(row))
                events[row['start']].append((side,key,1))
                events[row['end']].append((side,key,-1))
        active=[Counter(),Counter()];previous=None
        for instant,updates in sorted(events.items()):
            if previous is not None and instant>previous and active[0] and active[1]:
                pairs={(ka[0],kb[0]) for ka in active[0] for kb in active[1] if ka[1]!=kb[1]}
                if any(ka[1]==kb[1] for ka in active[0] for kb in active[1]):
                    same_stream+=instant-previous
                if pairs:
                    delta=instant-previous;totals['overlap_ns']+=delta
                    same=any(x==y for x,y in pairs)
                    cross=any(x!=y for x,y in pairs)
                    if same:totals['same_frame_overlap_ns']+=delta
                    if cross:totals['cross_frame_overlap_ns']+=delta
                    if same and cross:totals['same_and_cross_simultaneous_ns']+=delta
                    for pair in pairs: pair_times[pair]+=delta
            for side,key,change in updates:
                active[side][key]+=change
                if not active[side][key]:del active[side][key]
            previous=instant
    fields=('left_busy','right_busy','kernel_union','overlap','same_frame_overlap',
            'cross_frame_overlap','same_and_cross_simultaneous')
    result={key+'_ms':totals[key+'_ns']/1e6 for key in fields}
    result['overlap_over_union']=totals['overlap_ns']/totals['kernel_union_ns'] if totals['kernel_union_ns'] else 0
    result['same_stream_interval_overlap_ms']=same_stream/1e6
    result['frame_pairs']=[dict(ae_frame=a,prefill_frame=b,same_frame=a==b,overlap_ms=t/1e6)
                           for (a,b),t in sorted(pair_times.items())]
    return result


def _busy(rows):
    devices=defaultdict(list)
    for row in rows:
        devices[row['process'],row['device']].append((row['start'],row['end']))
    return sum(interval_duration(intervals) for intervals in devices.values())/1e6


def _envelope(rows):
    devices=defaultdict(list)
    for row in rows:
        devices[row['process'],row['device']].append(row)
    return sum(max(r['end'] for r in values)-min(r['start'] for r in values)
               for values in devices.values())/1e6


def summarize_group(rows):
    ae=[r for r in rows if r['branch']=='AE']
    pairs={name:overlap_metrics(ae,[r for r in rows if r['stage'] in stages]) for name,stages in PAIRS.items()}
    iteration_pairs={str(step):{
        name:overlap_metrics([r for r in ae if r['iteration']==step],
                             [r for r in rows if r['stage'] in PAIRS[name]])
        for name in ('AE_x_VLM','AE_x_Vision')}
        for step in sorted({r['iteration'] for r in ae if r['iteration'] is not None})}
    stages=[]
    for key in sorted({(r['stage'],r['iteration']) for r in rows},key=str):
        selected=[r for r in rows if (r['stage'],r['iteration'])==key]
        busy,envelope=_busy(selected),_envelope(selected)
        stages.append(dict(stage=key[0],iteration=key[1],kernel_count=len(selected),
                           kernel_union_ms=busy,kernel_envelope_ms=envelope,
                           nonkernel_gap_ms=envelope-busy,
                           sum_kernel_elapsed_ms=sum(r['duration_us'] for r in selected)/1000))
    return dict(kernel_count=len(rows),kernel_union_ms=_busy(rows),pairs=pairs,
                iteration_pairs=iteration_pairs,stages=stages)


def signature_statistics(rows):
    values=defaultdict(list)
    for r in rows:
        if not r['branch'] or not r['request_id']:
            continue
        key=(r['group'],r['branch'],r['stage'],r['iteration'],r['layer'],r['operation'],r['name'],tuple(r['geometry']))
        values[key].append(r['duration_us'])
    output=[]
    for key,durations in sorted(values.items(),key=lambda item:-sum(item[1])):
        group,branch,stage,iteration,layer,op,name,geometry=key
        output.append(dict(group=group,branch=branch,stage=stage,iteration=iteration,layer=layer,
            operation=op,name=name,geometry=list(geometry),calls=len(durations),
            mean_us=statistics.mean(durations),p50_us=statistics.median(durations),
            sum_kernel_elapsed_ms=sum(durations)/1000))
    return output


def analyze(path):
    kernels,schema=_read_database(path)
    labelled=[r for r in kernels if r['branch']]
    measured=[r for r in labelled if r['request_id']]
    graph=[r for r in kernels if r['graph_node'] not in (None,0)]
    groups={group:summarize_group([r for r in measured if r['group']==group])
            for group in sorted({r['group'] for r in measured})}
    requests=[]
    for request in sorted({r['request_id'] for r in measured}):
        rows=[r for r in measured if r['request_id']==request];first=rows[0]
        requests.append(dict(request_id=request,frame_id=first['frame_id'],frame=first['frame'],epoch=first['epoch'],
            slot=first['slot'],generation=first['generation'],group=first['group'],
            host_request_ranges=sorted({r['host_request'] for r in rows if r['host_request']}),
            first_kernel_ns=min(r['start'] for r in rows),last_kernel_ns=max(r['end'] for r in rows),
            **summarize_group(rows)))
    phase_rows=defaultdict(list)
    for row in measured:
        phase_rows[row['group'],row['stage'],row['iteration'],row['layer'],row['operation']].append(row)
    phases=[dict(group=g,stage=s,iteration=i,layer=l,operation=o,kernel_count=len(rows),
                 kernel_union_ms=_busy(rows),sum_kernel_elapsed_ms=sum(r['duration_us'] for r in rows)/1000)
            for (g,s,i,l,o),rows in sorted(phase_rows.items(),key=lambda item:str(item[0]))]
    bad=[r for r in kernels if r['mapping_errors'] or (r['graph_node'] not in (None,0) and not r['request_id'])]
    report=dict(source=str(Path(path).resolve()),profile_only=True,schema=schema,
        attribution=dict(total_kernels=len(kernels),graph_kernels=len(graph),
            graph_stage_mapped=sum(bool(r['branch']) for r in graph),
            graph_request_mapped=sum(bool(r['request_id']) for r in graph),
            all_graph_stages_mapped=all(r['branch'] is not None for r in graph),
            all_graph_requests_mapped=all(r['request_id'] is not None for r in graph),
            missing_launch_correlations=sum('missing_launch_correlation' in r['mapping_errors'] for r in kernels),
            ambiguous_launch_correlations=sum('ambiguous_launch_correlation' in r['mapping_errors'] for r in kernels),
            unlabelled_nongraph_kernels=sum(not r['branch'] and r['graph_node'] in (None,0) for r in kernels),
            classified_outside_request=sum(not r['request_id'] for r in labelled),
            group_assignments=dict(Counter(r['group_assignment'] for r in kernels)),
            unresolved_unscoped_submissions=sum(r['group']=='unscoped_submit' for r in kernels),
            measured_labelled_kernels=len(measured),label_formats=dict(Counter(r['label_format'] for r in graph))),
        groups=groups,requests=requests,phase_summaries=phases,kernel_signatures=signature_statistics(kernels),
        unmapped=bad,unlabelled_kernel_names=dict(Counter(r['name'] for r in kernels if not r['branch'])),
        streams=[dict(process=p,device=d,context=c,green_context=g,stream=s,branch=b,kernel_count=n)
                 for (p,d,c,g,s,b),n in sorted(Counter((r['process'],r['device'],r['context'],r['green_context'],r['stream'],r['branch']) for r in labelled).items(),key=str)],
        iterations_seen=sorted({r['iteration'] for r in labelled if r['iteration'] is not None}),
        definitions=[
          'Capture labels follow originalGraphNodeId recursively, even if a clone also has a creation record.',
          'Runtime launch correlation plus process identifies replay submission; driver API is fallback only when runtime correlation is absent.',
          'Logical frame comes from submit frame/epoch/slot/generation; host request range is retained separately because future-frame encoding may be submitted inside the preceding host request.',
          'Out-of-range crossframe submits use an unambiguous matching submit identity or, if none exists, the sole scoped group in the same process. source_group, host_section_missing and group_assignment preserve this inference; Host ranges are never fabricated. Ambiguous groups remain unscoped.',
          'Overlap measures actual GPU kernel intervals on distinct streams of the same process and physical device. Different greenContextIds are deliberately included.',
          'Same-frame and cross-frame overlap subsets can overlap each other; their union is total overlap, so do not simply add them.',
          'Event waits, Host NVTX spans, memcpy and memset are excluded. Kernel elapsed intervals include stalls and are not hardware utilization counters.',
          'Kernel envelope is first-to-last kernel timestamp of the selected rows. Envelope minus union is non-kernel gap, not proven Event-wait duration; group-wide envelopes also include gaps between requests.',
          'Attention_non_projection excludes separately labelled Q/K/V/O launches, but includes RoPE/reshape/repeat-KV and is not pure attention arithmetic.',
          'All unmapped kernels remain in kernels.csv; graph mapping failures remain explicit in unmapped and attribution. No performance inference from legacy label-schema validation traces.',
          'Profiler timings are not steady CPU-to-CPU request latency. Compare matching work, input shapes and model settings before interpreting signature duration ratios.'])
    return report,kernels


def compare_signatures(baseline,candidate,baseline_group=None,candidate_group=None):
    if baseline['iterations_seen']!=candidate['iterations_seen']:
        raise ValueError('Different labelled AE iteration counts; do not mix legacy D10 and new D2 traces')
    def choose(report,selected):
        if selected is None:
            if len(report['groups'])!=1:
                raise ValueError('Select an explicit group when a trace contains multiple modes/phases')
            selected=next(iter(report['groups']))
        if selected not in report['groups']:
            raise ValueError(f'Unknown group {selected}')
        return selected
    baseline_group,candidate_group=choose(baseline,baseline_group),choose(candidate,candidate_group)
    def key(row):
        return row['branch'],row['stage'],row['iteration'],row['layer'],row['operation'],row['name'],tuple(row['geometry'])
    left={key(r):r for r in baseline['kernel_signatures'] if r['group']==baseline_group}
    right={key(r):r for r in candidate['kernel_signatures'] if r['group']==candidate_group}
    matches=[]
    for signature in left.keys() & right.keys():
        a,b=left[signature],right[signature]
        matches.append({k:b[k] for k in ('branch','stage','iteration','layer','operation','name','geometry')} |
            dict(baseline_calls=a['calls'],candidate_calls=b['calls'],baseline_mean_us=a['mean_us'],
                 candidate_mean_us=b['mean_us'],duration_ratio=b['mean_us']/a['mean_us'] if a['mean_us'] else None,
                 candidate_sum_elapsed_ms=b['sum_kernel_elapsed_ms']))
    matches.sort(key=lambda r:-r['candidate_sum_elapsed_ms'])
    return dict(baseline_source=baseline['source'],candidate_source=candidate['source'],
                baseline_group=baseline_group,candidate_group=candidate_group,matches=matches,
                baseline_unmatched_types=len(left.keys()-right.keys()),candidate_unmatched_types=len(right.keys()-left.keys()),
                scope='Matched name+launch geometry+labelled phase, elapsed per-call ratio, not hardware utilization or end-to-end speedup.')


def write_csv(path,rows):
    if not rows:
        path.write_text('');return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})


def plot_timeline(report,kernels,path,group=None,request_id=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    requests=[r for r in report['requests'] if group is None or r['group']==group]
    if not requests:
        raise ValueError('No attributed requests available for a GPU timeline')
    requests.sort(key=lambda r:r['first_kernel_ns'])
    selected=next((r for r in requests if r['request_id']==request_id),None) if request_id else requests[len(requests)//2]
    if selected is None:
        raise ValueError(f'Unknown timeline request {request_id}')
    origin,end=selected['first_kernel_ns'],selected['last_kernel_ns']
    rows=[r for r in kernels if r['group']==selected['group'] and r['end']>origin and r['start']<end and r['branch']]
    lanes=defaultdict(list)
    for row in rows:
        stage=f'AE iteration {row["iteration"]}' if row['stage']=='AE' and row['iteration'] is not None else row['stage']
        if stage=='AE':stage='AE prep/Euler'
        lanes[stage,row['stream'],row['green_context']].append(row)
    order={'VisionEncoder':0,'VisionProjector':1,'MetadataEmbedding':2,'VLM':3}
    keys=sorted(lanes,key=lambda k:(order.get(k[0],4),k[0],k[1]))
    fig,ax=plt.subplots(figsize=(14,max(4,0.65*len(keys)+2)))
    palette=plt.get_cmap('tab10')
    for y,key in enumerate(keys):
        for same in (True,False):
            values=[r for r in lanes[key] if (r['frame_id']==selected['frame_id'])==same]
            ax.broken_barh([((max(origin,r['start'])-origin)/1e6,(min(end,r['end'])-max(origin,r['start']))/1e6)
                           for r in values],(y-.34,.68),facecolors=palette(y%10),alpha=1 if same else .3,
                          linewidth=0,rasterized=True)
    ax.set_yticks(range(len(keys)),[f'{stage} | stream {stream} / green {green}' for stage,stream,green in keys])
    ax.invert_yaxis();ax.set_xlim(0,(end-origin)/1e6);ax.set_xlabel('GPU kernel timestamps relative to selected frame (ms)')
    ax.grid(axis='x',alpha=.2)
    ax.set_title(f'{selected["group"]}; frame={selected["frame"]}, epoch={selected["epoch"]}\n'
                 'Actual CUPTI kernels; faint = other frame; gaps are not GPU busy; profiling is not steady latency')
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)
    return dict(request_id=selected['request_id'],source=report['source'],origin_ns=origin,end_ns=end,
                displayed_kernels=len(rows),other_frame_kernels=sum(r['frame_id']!=selected['frame_id'] for r in rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--group')
    parser.add_argument('--timeline-request')
    parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--baseline-group')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report,kernels=analyze(args.sqlite)
    report['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_csv(args.output/'kernels.csv',kernels)
    write_csv(args.output/'kernel_signatures.csv',report['kernel_signatures'])
    write_csv(args.output/'phase_summary.csv',report['phase_summaries'])
    flat=[]
    for group,value in report['groups'].items():
        for pair,metrics in value['pairs'].items():
            flat.append(dict(group=group,pair=pair,**{k:v for k,v in metrics.items() if k!='frame_pairs'}))
    write_csv(args.output/'overlap.csv',flat)
    write_csv(args.output/'requests.csv',report['requests'])
    write_csv(args.output/'request_stages.csv',[
        dict(request_id=request['request_id'],frame=request['frame'],epoch=request['epoch'],group=request['group'],**stage)
        for request in report['requests'] for stage in request['stages']])
    if args.baseline:
        comparison=compare_signatures(json.loads(args.baseline.read_text()),report,args.baseline_group,args.group)
        (args.output/'signature_comparison.json').write_text(json.dumps(comparison,indent=2))
        write_csv(args.output/'signature_comparison.csv',comparison['matches'])
    if not args.no_plot:
        try:
            report['timeline']=plot_timeline(report,kernels,args.output/'timeline.png',args.group,args.timeline_request)
        except ImportError as error:
            report['timeline']=dict(unavailable=str(error))
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    lines=[f'Source: {report["source"]}',f'Attribution: {json.dumps(report["attribution"])}',
           'Kernel interval intersection only. Profiler timing is not steady request latency.']
    for group,value in report['groups'].items():
        for pair in ('AE_x_VLM','AE_x_Vision'):
            metrics=value['pairs'][pair]
            lines.append(f'{group} {pair}: overlap={metrics["overlap_ms"]:.6f} ms; union={metrics["kernel_union_ms"]:.6f} ms; '
                         f'same-frame={metrics["same_frame_overlap_ms"]:.6f} ms; cross-frame={metrics["cross_frame_overlap_ms"]:.6f} ms')
    (args.output/'summary.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':
    main()
