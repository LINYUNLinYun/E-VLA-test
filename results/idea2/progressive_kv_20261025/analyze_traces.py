"""CPU-only progressive-KV nsys analysis; graph ancestry + launch correlation.

Use --trace report.nsys-rep --profile-result profile/result.json --output DIR.
SQLite export is CPU-only. Host NVTX ends NEVER clip GPU kernel intervals.
The request NVTX encloses only AE submission in this runtime. VLM and Vision
are joined by strict same-thread graph-launch order, with ambiguities rejected.
Profile rows certify which following Vision is next-observation lookahead.
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

ROOT = Path(__file__).resolve().parents[3]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, str(ROOT))
from preparations.idea2 import layer_trace as base

REQUEST = re.compile(r'^progressive/request_(\d+)/lead_(\d+)/slot_(\d+)$')
SUBMISSION = re.compile(r'^submit/(vision|vlm|ae)/frame=(\d+)/slot=(\d+)/gen=(\d+)(/ahead)?$')


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict, tuple)) else v
                          for k, v in row.items()} for row in rows)


def stats(values):
    values = sorted(x for x in values if x is not None)
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None, min=None, max=None)
    def percentile(p):
        position = (len(values)-1)*p
        lo = int(position)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(position-lo)
    return dict(n=len(values), mean=statistics.mean(values), p50=percentile(.5),
                p95=percentile(.95), min=values[0], max=values[-1])


def classify(labels, original):
    item = original(labels)
    item['detail_stage'] = item['stage']
    if any(x.startswith('AE/') for x in labels):
        item.update(branch='AE', stage='AE')
    elif any(x.startswith('Prefill/VLM') for x in labels):
        item.update(branch='Prefill', stage='VLM')
    elif any(x.startswith('Prefill/Vision') for x in labels):
        item.update(branch='Prefill', stage='Vision')
    return item


def read_trace(path):
    # Reuse tested recursive clone ancestry and interval code. Changes are local
    # to this offline process and restored immediately; no source file edits.
    original_request, original_classifier = base.is_request, base.classify_labels
    try:
        base.is_request = lambda label: bool(REQUEST.fullmatch(label))
        base.classify_labels = lambda labels: classify(labels, original_classifier)
        kernels, schema = base._read_database(path)
    finally:
        base.is_request, base.classify_labels = original_request, original_classifier
    db = sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    strings = dict(db.execute('select id,value from StringIds'))
    hosts, submissions = {}, []
    for raw in db.execute('select rowid,* from NVTX_EVENTS where end is not null'):
        label = raw['text'] or strings.get(raw['textId'], '')
        if match := REQUEST.fullmatch(label):
            hosts[raw['rowid']] = dict(rowid=raw['rowid'], start=raw['start'], end=raw['end'],
                tid=raw['globalTid'], label=label, request=int(match[1]), lead=int(match[2]), slot=int(match[3]))
        if match := SUBMISSION.fullmatch(label):
            submissions.append(dict(rowid=raw['rowid'],start=raw['start'],end=raw['end'],tid=raw['globalTid'],
                label=label,kind=match[1],frame=int(match[2]),slot=int(match[3]),generation=int(match[4]),ahead=bool(match[5])))
    submission_index=base.RangeIndex(submissions)
    schema['progressive_submit_ranges']=len(submissions)
    api_map = defaultdict(list)
    for table in ('CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER'):
        if table not in schema['tables']:
            continue
        for raw in db.execute(f'select rowid,* from {table}'):
            if raw['correlationId'] is not None:
                api_map[base.process_id(raw['globalTid']), raw['correlationId']].append(dict(
                    start=raw['start'], end=raw['end'], tid=raw['globalTid'], table=table,
                    name=strings.get(raw['nameId'], str(raw['nameId']))))
    def api_for(key):
        choices = api_map.get(key, ())
        runtime = [r for r in choices if r['table']=='CUPTI_ACTIVITY_KIND_RUNTIME']
        selected = runtime or choices
        return selected[0] if len(selected)==1 else None
    event_rows = []
    if 'CUPTI_ACTIVITY_KIND_CUDA_EVENT' in schema['tables']:
        for raw in db.execute('select * from CUPTI_ACTIVITY_KIND_CUDA_EVENT'):
            row = dict(raw)
            api = api_for((row.get('globalPid'), row.get('correlationId')))
            if api:
                row.update(api_start=api['start'], api_end=api['end'], tid=api['tid'], api_name=api['name'])
            event_rows.append(row)
    db.close()
    launches = {}
    for kernel in kernels:
        kernel['helper'] = next((name for name in ('_publish', '_snapshot', '_select')
                                 if kernel['name'].startswith(name)), None)
        if kernel['graph_node'] in (None, 0):
            continue
        if kernel['mapping_errors']:
            raise ValueError(f"Graph kernel attribution incomplete: {kernel['mapping_errors']} row={kernel['kernel_rowid']}")
        if kernel['stage'] not in ('AE', 'VLM', 'Vision'):
            raise ValueError(f"Graph node lacks progressive module label: row={kernel['kernel_rowid']} {kernel['capture_labels']}")
        key = (kernel['process'], kernel['correlation'])
        api = api_for(key)
        if api is None or 'GraphLaunch' not in api['name']:
            raise ValueError(f'Graph node has no unique GraphLaunch correlation: {key}')
        matching_submits=submission_index.containing(api['tid'],api['start'],api['end'])
        launch = launches.setdefault(key, dict(key=list(key), stage=kernel['stage'], api=api,
            submit=matching_submits[0] if matching_submits else None,
            host=hosts.get(kernel['host_request_rowid']), kernels=[], request_key=None, mapping=None))
        if launch['stage'] != kernel['stage']:
            raise ValueError(f'One GraphLaunch contains multiple top-level stages: {key}')
        launch['kernels'].append(kernel)
    if not launches:
        raise ValueError('No graph node kernels; collect --cuda-graph-trace=node with NVTX precapture')
    return kernels, list(launches.values()), hosts, event_rows, schema


def assign_requests(launches, profile_rows):
    """AE host anchors plus strict module order; never GPU-time containment."""
    by_thread = defaultdict(list)
    for launch in launches:
        by_thread[launch['key'][0], launch['api']['tid']].append(launch)
    by_profile = defaultdict(list)
    for row in profile_rows:
        by_profile[row['request'], row['lead'], row['slot']].append(row)
    requests = []
    for _, sequence in by_thread.items():
        sequence.sort(key=lambda r:r['api']['start'])
        aes = [r for r in sequence if r['stage']=='AE' and r['host']]
        previous_ae = None
        local = []
        for ae in aes:
            host = ae['host']
            key = f"{ae['key'][0]}:host_nvtx_row={host['rowid']}"
            lo = previous_ae['api']['end'] if previous_ae else -float('inf')
            vlms = [r for r in sequence if r['stage']=='VLM'
                    and lo <= r['api']['start'] < ae['api']['start']]
            explicit_vlms=[r for r in vlms if r.get('submit')]
            if explicit_vlms:
                vlms=[r for r in explicit_vlms if r['submit']['frame']==host['request'] and r['submit']['slot']==host['slot']]
            if not vlms or (previous_ae is not None and len(vlms)!=1):
                raise ValueError(f"Cannot uniquely join current VLM for {host['label']}: {len(vlms)} candidates")
            # First measured request may be preceded by the explicit bootstrap.
            vlm = vlms[-1]
            earlier_vlm = [r for r in sequence if r['stage']=='VLM' and r['api']['start'] < vlm['api']['start']]
            vlo = earlier_vlm[-1]['api']['end'] if earlier_vlm else -float('inf')
            visions = [r for r in sequence if r['stage']=='Vision'
                       and vlo <= r['api']['start'] < vlm['api']['start']]
            if vlm.get('submit'):
                selected=vlm['submit']
                visions=[r for r in visions if r.get('submit') and all(
                    r['submit'][field]==selected[field] for field in ('frame','slot','generation'))]
            if len(visions)!=1:
                raise ValueError(f"Cannot uniquely join current Vision for {host['label']}: {len(visions)} candidates")
            profile = by_profile.get((host['request'], host['lead'], host['slot']), [])
            if len(profile)>1:
                raise ValueError(f"Ambiguous profile rows for {host['label']}; pass one profile sequence")
            if profile_rows and not profile:
                raise ValueError(f"Missing profile row for {host['label']}")
            if profile and vlm.get('submit') and profile[0].get('generation') is not None:
                if profile[0]['generation']!=vlm['submit']['generation']:
                    raise ValueError(f"Profile/submit generation mismatch for {host['label']}")
            request = dict(key=key, host=host, ae=ae, vlm=vlm, vision=visions[0],
                           profile=profile[0] if profile else None, next_vision=None)
            for name in ('ae','vlm','vision'):
                launch = request[name]
                if launch['request_key'] is not None:
                    raise ValueError(f'Graph launch assigned to two current requests: {launch["key"]}')
                method=('AE_host_correlation' if name=='ae' else 'explicit_submit_frame_slot_generation'
                        if launch.get('submit') else 'same_host_graph_launch_sequence')
                launch.update(request_key=key, mapping=method)
                for kernel in launch['kernels']:
                    kernel.update(request_id=key, frame_id=key, frame=host['request'], slot=host['slot'],
                        request_label=host['label'], host_start=host['start'], host_end=host['end'])
            local.append(request)
            previous_ae = ae
        for i, request in enumerate(local):
            profile = request['profile']
            if profile is None or not profile.get('next_encoder'):
                continue
            if i+1 == len(local):
                following = [r for r in sequence if r['stage']=='Vision'
                             and r['api']['start'] > request['ae']['api']['end']]
                if len(following)!=1:
                    raise ValueError('Trailing nextEncoder cannot be uniquely attributed')
                nxt = following[0]
            else:
                following_request = local[i+1]
                if following_request['host']['request'] != request['host']['request']+1:
                    raise ValueError('Nonconsecutive request IDs: cannot infer nextEncoder source')
                if following_request['profile']['source_id'] != profile['source_id']+1:
                    raise ValueError('Profile source IDs do not validate lookahead identity')
                nxt = following_request['vision']
            if nxt['api']['start'] <= request['ae']['api']['end']:
                raise ValueError('Profile claims nextEncoder but its graph was submitted before current AE')
            if nxt.get('submit') and not nxt['submit']['ahead']:
                raise ValueError('Profile claims nextEncoder but Vision submit lacks /ahead')
            request['next_vision'] = nxt
        requests.extend(local)
    if not requests:
        raise ValueError('No AE GraphLaunch maps to progressive/request_i/lead_k/slot_s NVTX')
    if profile_rows and len(requests)!=len(profile_rows):
        raise ValueError(f'Profile/trace request count mismatch: {len(profile_rows)} vs {len(requests)}')
    return sorted(requests, key=lambda r:r['ae']['api']['start'])


def phase_metrics(launch):
    if launch is None:
        return None
    rows = launch['kernels']
    busy, envelope = base._busy(rows), base._envelope(rows)
    return dict(kernels=len(rows), helper_kernels=sum(bool(r['helper']) for r in rows),
                busy_ms=busy, envelope_ms=envelope, nonkernel_gap_ms=envelope-busy,
                start_ns=min(r['start'] for r in rows), end_ns=max(r['end'] for r in rows),
                sum_kernel_elapsed_ms=sum(r['duration_us'] for r in rows)/1000)


def graph_events(launch, all_launches, events):
    """Locate explicit runtime record fences surrounding a graph on its stream.

Missing device timestamps remain unknown. Graph-internal mark Events are not
guessed from their order; only explicit EventRecord API correlations qualify.
"""
    rows = launch['kernels']
    streams = {(r['process'], r['device'], r['context'], r['stream']) for r in rows}
    if len(streams)!=1:
        return dict(available=False, reason='graph_spans_multiple_streams')
    pid, device, context, stream = next(iter(streams))
    candidates = [r for r in events if r.get('globalPid')==pid and r.get('deviceId')==device
                  and r.get('contextId')==context and r.get('streamId')==stream
                  and (r.get('timestamp') or 0)>0 and 'EventRecord' in r.get('api_name','')]
    siblings = [r for r in all_launches if r is not launch and any(
        (k['process'],k['device'],k['context'],k['stream'])==(pid,device,context,stream) for k in r['kernels'])]
    lower = max((r['api']['end'] for r in siblings if r['api']['end']<=launch['api']['start']),default=-float('inf'))
    upper = min((r['api']['start'] for r in siblings if r['api']['start']>=launch['api']['end']),default=float('inf'))
    before = [r for r in candidates if lower <= r['api_end'] <= launch['api']['start']]
    after = [r for r in candidates if launch['api']['end'] <= r['api_start'] < upper]
    if launch['stage']=='AE' and launch['host']:
        after = [r for r in after if r['api_end'] <= launch['host']['end']]
    if not before or not after:
        return dict(available=False, reason='explicit_event_device_timestamp_unavailable')
    start = max(before,key=lambda r:r['api_start'])
    # AE records old_read then new_read; metric uses the latter. Other stages
    # have one done Event immediately after replay.
    end = (max if launch['stage']=='AE' else min)(after,key=lambda r:r['api_start'])
    if start['timestamp'] > min(r['start'] for r in rows) or end['timestamp'] < max(r['end'] for r in rows):
        return dict(available=False, reason='record_fences_do_not_enclose_all_graph_kernels')
    return dict(available=True, start_ns=start['timestamp'], end_ns=end['timestamp'],
                span_ms=(end['timestamp']-start['timestamp'])/1e6,
                start_event_id=start['eventId'], end_event_id=end['eventId'])


def resource_audit(sqlite_path, kernels, profile):
    """Match Kernel.greenContextId to Green context SM metadata, not contextId."""
    profile=profile or {}
    adapter=profile.get('resource_adapter') or {}
    layouts=adapter.get('actual_resources',[])
    requested=adapter.get('ae_sm_min',0)
    errors=[]
    if requested:
        if not layouts or any(len(x.get('sm_counts',[]))!=2 for x in layouts):
            errors.append(dict(kind='missing_driver_queried_resource_counts'))
        pairs={tuple(x['sm_counts']) for x in layouts if len(x.get('sm_counts',[]))==2}
        if len(pairs)!=1:
            errors.append(dict(kind='ambiguous_driver_resource_layout',layouts=layouts))
        expected_pair=next(iter(pairs)) if len(pairs)==1 else (None,None)
        for layout in layouts:
            if sum(layout.get('sm_counts',[]))!=layout.get('device_sm_count'):
                errors.append(dict(kind='driver_partition_does_not_cover_device',layout=layout))
    else:
        expected_pair=(None,None)
    db=sqlite3.connect(sqlite_path.resolve().as_uri()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    contexts=[dict(r) for r in db.execute('select * from TARGET_INFO_CUDA_CONTEXT_INFO')] if 'TARGET_INFO_CUDA_CONTEXT_INFO' in tables else []
    streams=[dict(r) for r in db.execute('select * from TARGET_INFO_CUDA_STREAM')] if 'TARGET_INFO_CUDA_STREAM' in tables else []
    db.close()
    index=defaultdict(list)
    for context in contexts:
        index[context['processId'],context['deviceId'],context['contextId']].append(context)
    groups=defaultdict(list)
    selected=[k for k in kernels if k['graph_node'] not in (None,0) and k['stage'] in ('AE','VLM','Vision')]
    for row in selected:
        groups[row['stage'],row['process'],row['device'],row['context'],row['green_context'],row['stream']].append(row)
    resources=[]
    for (stage,pid,device,primary,green,stream),rows in groups.items():
        local_pid=(pid>>24)&0xffffff
        matches=index.get((local_pid,device,green),[]) if green not in (None,0) else []
        context=matches[0] if len(matches)==1 else None
        actual=context.get('numMultiprocessors') if context else None
        expected=expected_pair[0 if stage=='AE' else 1] if requested else None
        row=dict(stage=stage,process=pid,local_process_id=local_pid,device=device,
            kernel_primary_context_id=primary,kernel_green_context_id=green,stream_id=stream,
            kernel_count=len(rows),actual_eligible_sms=actual,expected_eligible_sms=expected,
            context_metadata=context,includes_helpers=sum(bool(k.get('helper')) for k in rows))
        row['passed']=(bool(context) and bool(context.get('isGreenContext'))
                       and context.get('parentContextId')==primary and actual==expected) if requested else green==0
        resources.append(row)
        if not row['passed']:errors.append(dict(kind='unexpected_kernel_resource_assignment',**row))
    if not selected:errors.append(dict(kind='no_attributed_graph_kernels'))
    ae_green={r['kernel_green_context_id'] for r in resources if r['stage']=='AE'}
    vlm_green={r['kernel_green_context_id'] for r in resources if r['stage']=='VLM'}
    vision_green={r['kernel_green_context_id'] for r in resources if r['stage']=='Vision'}
    if requested and (ae_green & (vlm_green|vision_green) or vlm_green!=vision_green):
        errors.append(dict(kind='noncomplementary_stage_green_contexts',ae=sorted(ae_green,key=str),vlm=sorted(vlm_green,key=str),vision=sorted(vision_green,key=str)))
    source_file=ROOT/'results/idea2/progressive_kv_20261025/resource_adapter.py' if requested else ROOT/'preparations/idea2/progressive_kv.py'
    expected_hash=adapter.get('sha256') if requested else profile.get('source_before',{}).get('preparations/idea2/progressive_kv.py')
    actual_hash=hashlib.sha256(source_file.read_bytes()).hexdigest() if source_file.exists() else None
    pool_proof=dict(kind='source_invariant_not_nsys_pool_measurement',source=str(source_file),
        recorded_sha256=expected_hash,current_sha256=actual_hash,
        recorded_source_matches=bool(expected_hash) and expected_hash==actual_hash,
        profile_completed=profile.get('status')=='passed',
        constructor_contains_private_pool_check=source_file.exists() and 'len(set(pools))' in source_file.read_text(),
        explanation='Frozen constructor checks six distinct private Graph pools and raises on alias before measured requests. Allocator pool IDs are not exported by this nsys trace; this is source/finished-run evidence, not a measured pool-ID claim.')
    return dict(passed=not errors,partitioned=bool(requested),requested_ae_sm_min=requested,
        driver_queried_layouts=layouts,expected_sm_pair=list(expected_pair),assignments=resources,
        contexts=contexts,stream_metadata=streams,private_graph_pool_evidence=pool_proof,errors=errors,
        definition='Join CUPTI Kernel.greenContextId (not Kernel.contextId) with TARGET_INFO_CUDA_CONTEXT_INFO.contextId for same process/device; require isGreenContext and matching parentContextId. SM counts are resource eligibility limits queried by driver/trace metadata, not measured occupancy. Opaque driver stream handles cannot be numerically equated with CUPTI stream IDs. Primary greenContextId=0 means unrestricted, not zero-SM occupancy.')


def analyze(sqlite_path, output, profile=None, provenance=None):
    output.mkdir(parents=True, exist_ok=True)
    kernels, launches, hosts, event_rows, schema = read_trace(sqlite_path)
    profile_rows = [r for r in (profile or {}).get('rows',[]) if r.get('phase')=='profile']
    requests = assign_requests(launches, profile_rows)
    summary_rows, phase_rows, crosschecks = [], [], []
    for request in requests:
        key, host, measured = request['key'], request['host'], request['profile']
        phases = {name:phase_metrics(request[name]) for name in ('ae','vlm','vision','next_vision')}
        ae, vlm, nxt = (request[name]['kernels'] if request[name] else [] for name in ('ae','vlm','next_vision'))
        row = dict(request_id=key, request=host['request'], lead=host['lead'], slot=host['slot'],
            source_id=measured.get('source_id') if measured else None,
            frontiers=measured.get('frontiers') if measured else None,
            variant=measured.get('variant') if measured else None,
            AE_x_currentVLM_ms=base.overlap_metrics(ae,vlm)['overlap_ms'],
            AE_x_nextVision_ms=base.overlap_metrics(ae,nxt)['overlap_ms'] if request['next_vision'] else None,
            ae_finish_minus_currentVLM_finish_kernel_ms=(phases['ae']['end_ns']-phases['vlm']['end_ns'])/1e6,
            exposed_ae_after_currentVLM_kernel_ms=max(0.0,(phases['ae']['end_ns']-phases['vlm']['end_ns'])/1e6),
            AE_kernel_tail_before_or_at_VLM_end=phases['ae']['end_ns']<=phases['vlm']['end_ns'],
            ae_graph_launch_after_vlm_graph_launch_ms=(request['ae']['api']['start']-request['vlm']['api']['start'])/1e6,
            ae_kernel_start_after_vlm_kernel_start_ms=(phases['ae']['start_ns']-phases['vlm']['start_ns'])/1e6,
            next_encoder_verified=bool(request['next_vision']),
            next_encoder_finish_minus_AE_finish_kernel_ms=(phases['next_vision']['end_ns']-phases['ae']['end_ns'])/1e6 if phases['next_vision'] else None,
            graph_kernels_after_host_NVTX=sum(r['end']>host['end'] for r in ae+vlm+nxt),
            note='Host NVTX marks AE submission, not the action response; all correlated GPU tails retained')
        for name, metrics in phases.items():
            if metrics:
                phase_rows.append(dict(request_id=key,request=host['request'],phase=name,**metrics))
                row[name+'_busy_ms']=metrics['busy_ms']
                row[name+'_envelope_ms']=metrics['envelope_ms']
        fences = {name:graph_events(request[name],launches,event_rows) for name in phases if request[name]}
        check = dict(request_id=key,request=host['request'],source_id=row['source_id'],
                     nsys_explicit_event_fences=fences)
        if measured:
            row['ae_finish_minus_currentVLM_finish_profile_event_ms']=measured.get('ae_finish_after_vlm_ms')
            row['exposed_ae_after_currentVLM_profile_event_ms']=measured.get('exposed_ae_after_vlm_ms')
            check['ae_finish_minus_currentVLM_finish_kernel_ms']=row['ae_finish_minus_currentVLM_finish_kernel_ms']
            check['ae_finish_minus_currentVLM_finish_profile_event_ms']=measured.get('ae_finish_after_vlm_ms')
            if measured.get('ae_finish_after_vlm_ms') is not None:
                check['ae_vlm_tail_kernel_minus_profile_event_ms']=row['ae_finish_minus_currentVLM_finish_kernel_ms']-measured['ae_finish_after_vlm_ms']
        for name, field in (('ae','ae_ms'),('vlm','vlm_ms'),('vision','vision_ms'),('next_vision','next_vision_ms')):
            if measured and phases[name]:
                check[field+'_profile_event']=measured.get(field)
                check[field+'_nsys_kernel_envelope']=phases[name]['envelope_ms']
                if fences[name]['available']:
                    check[field+'_nsys_event']=fences[name]['span_ms']
                    if measured.get(field) is not None:
                        check[field+'_event_residual']=fences[name]['span_ms']-measured[field]
        if request['next_vision']:
            profile_delta = -measured['ae_finish_minus_next_encoder_ms'] if measured and measured.get('ae_finish_minus_next_encoder_ms') is not None else None
            check['next_encoder_finish_minus_AE_finish_profile_event_ms']=profile_delta
            check['next_encoder_finish_minus_AE_finish_kernel_ms']=row['next_encoder_finish_minus_AE_finish_kernel_ms']
            if profile_delta is not None:
                check['kernel_delta_minus_profile_event_delta_ms']=row['next_encoder_finish_minus_AE_finish_kernel_ms']-profile_delta
                row['next_encoder_finish_minus_AE_finish_profile_event_ms']=profile_delta
            if fences['ae']['available'] and fences['next_vision']['available']:
                delta=(fences['next_vision']['end_ns']-fences['ae']['end_ns'])/1e6
                check['next_encoder_finish_minus_AE_finish_nsys_event_ms']=delta
                if profile_delta is not None:
                    check['event_delta_residual_ms']=delta-profile_delta
        summary_rows.append(row)
        crosschecks.append(check)
    model = [k for k in kernels if k.get('request_id') and k['graph_node'] not in (None,0)]
    resource=resource_audit(sqlite_path,kernels,profile)
    distribution_fields = sorted({k for row in summary_rows for k,v in row.items() if k.endswith('_ms') and isinstance(v,(float,int))})
    report = dict(status='analyzed_existing_trace',trace_sqlite=str(sqlite_path.resolve()),
        provenance=provenance, profile_args=(profile or {}).get('args'),
        profile_source_before=(profile or {}).get('source_before'), schema=schema,
        resource_audit=resource,
        ae_tail_before_or_at_vlm_requests=sum(row['AE_kernel_tail_before_or_at_VLM_end'] for row in summary_rows),
        attribution=dict(total_kernels=len(kernels),graph_launches=len(launches),requests=len(requests),
            profile_rows=len(profile_rows),graph_kernels_assigned_to_requests=len(model),
            unassigned_graph_launches=[dict(key=r['key'],stage=r['stage'],api_start=r['api']['start'])
                                      for r in launches if r['request_key'] is None],
            helper_kernels_by_stage={stage:dict(Counter(k['helper'] for k in kernels if k['stage']==stage and k['helper']))
                                     for stage in ('AE','VLM','Vision')},
            explicit_cuda_events=len(event_rows),events_with_device_timestamp=sum((r.get('timestamp') or 0)>0 for r in event_rows),
            kernel_tails_never_clipped=True, graph_ancestry_complete=True),
        distributions={name:stats([r.get(name) for r in summary_rows]) for name in distribution_fields},
        requests=summary_rows,phases=phase_rows,crosschecks=crosschecks,
        definitions=dict(overlap='Union of actual GPU kernel intersections on same process/device but distinct streams. Excludes memory copies, event waits and NVTX envelopes.',
            busy='Union of kernel intervals; envelope is last kernel end minus first kernel start. Their difference includes gaps/waits, not measured CPU time.',
            Vision='Whole VisionEncoder+Projector+Metadata Graph, including language embedding/masks; not pure image encoder.',
            requests='AE GraphLaunch linked by correlation to Host request NVTX. Prefer explicit submit/frame/slot/gen for VLM and Vision, checked against profile generation and /ahead. Legacy traces use strict preceding same-thread GraphLaunch order. Ambiguities rejected. First unassigned prefix can be bootstrap.',
            nextEncoder='Profile next_encoder and consecutive source_id certify that next request Vision is the current lookahead. Without matching profile JSON this is unknown, not zero.',
            finish_delta='Positive nextEncoder_end - AE_end means Encoder finishes later. Kernel endpoints exclude output copy/event tails; profile CUDA Event endpoints include them. Explicit nsys device Event timestamps are used when exported, otherwise unavailable.',
            inference='Diagnostic trace timings only; no performance speedup or quality claim. Captured mask/time/attention math is not revalidated here.'))
    save(output/'report.json',report)
    save(output/'resource_audit.json',resource)
    csv_write(output/'resource_assignments.csv',resource['assignments'])
    csv_write(output/'requests.csv',summary_rows)
    csv_write(output/'phases.csv',phase_rows)
    csv_write(output/'event_crosschecks.csv',crosschecks)
    csv_write(output/'kernels.csv',kernels)
    csv_write(output/'graph_launch_map.csv',[dict(process=r['key'][0],correlation=r['key'][1],stage=r['stage'],
        request_id=r['request_key'],mapping=r['mapping'],launch_start=r['api']['start'],launch_end=r['api']['end'],
        submit_label=r['submit']['label'] if r.get('submit') else None,
        host_NVTX=r['host']['label'] if r['host'] else None,kernel_count=len(r['kernels']),
        first_kernel=min(k['start'] for k in r['kernels']),last_kernel=max(k['end'] for k in r['kernels'])) for r in launches])
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-result',type=Path,help='Matching progressive_kv_bench profile result.json; required to certify nextEncoder/source identity')
    parser.add_argument('--no-export',action='store_true')
    parser.add_argument('--nsys',default='nsys')
    args=parser.parse_args()
    if not args.trace.is_file():
        parser.error(f'Trace does not exist yet: {args.trace}; no experiment result claimed')
    args.output.mkdir(parents=True,exist_ok=True)
    source=args.trace
    exported=False
    if source.suffix!='.sqlite':
        sqlite_path=args.output/'trace.sqlite'
        export_receipt=args.output/'export_receipt.json'
        signature=dict(path=str(source.resolve()),size=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns)
        reused=sqlite_path.exists() and export_receipt.exists() and json.loads(export_receipt.read_text()).get('source')==signature
        if not reused:
            if args.no_export:parser.error('No matching exported SQLite receipt; --no-export prohibits export')
            if sqlite_path.exists():parser.error('Output trace.sqlite exists for another/incomplete source; use a fresh output directory')
            command=[args.nsys,'export','--type','sqlite','--output',str(sqlite_path),str(source)]
            with (args.output/'export.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            save(export_receipt,dict(source=signature,command=command,cpu_only=True))
            exported=True
        source=sqlite_path
    profile_discovery='explicit' if args.profile_result else 'none'
    if args.profile_result is None:
        match=re.search(r'(?:lead|fresh)(\d+)',args.trace.stem)
        candidates=[]
        if match:
            for candidate in Path(__file__).resolve().parent.glob('**/result.json'):
                try:
                    value=json.loads(candidate.read_text())
                except (OSError,json.JSONDecodeError):
                    continue
                cfg=value.get('args',{})
                if cfg.get('profile') and cfg.get('profile_lead')==int(match[1]) and any(r.get('phase')=='profile' for r in value.get('rows',[])):
                    candidates.append(candidate)
        if len(candidates)==1:
            args.profile_result=candidates[0]
            profile_discovery='unique_completed_profile_with_matching_lead'
        elif len(candidates)>1:
            parser.error('Multiple matching profile results; pass --profile-result explicitly: '+', '.join(map(str,candidates)))
    profile=json.loads(args.profile_result.read_text()) if args.profile_result else None
    if profile is not None and not any(r.get('phase')=='profile' for r in profile.get('rows',[])):
        parser.error('Provided result.json has no completed profile rows')
    provenance=dict(trace=str(args.trace.resolve()),profile_result=str(args.profile_result.resolve()) if args.profile_result else None,
        profile_discovery=profile_discovery,
        analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_layer_trace_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
        profile_sha256=hashlib.sha256(args.profile_result.read_bytes()).hexdigest() if args.profile_result else None,
        exported_now=exported,gpu_launched=False)
    report=analyze(source,args.output,profile,provenance)
    print(json.dumps(dict(status=report['status'],attribution=report['attribution'],distributions=report['distributions']),indent=2))


if __name__=='__main__':
    main()
