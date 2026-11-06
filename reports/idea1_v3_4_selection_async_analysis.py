"""Compact incremental analysis; never changes an episode or policy."""
import json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.common import OUT,directory as original_directory,ck,write_csv


def directory(condition,case):
    if condition=='v12_p3_q4_optimized_async':
        return OUT/'optimized_closed_loop'/condition/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
    return original_directory(condition,case)


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def kernel_events(path):
    # Parse one event at a time; traces can be >100MB and full json.load creates
    # excessive temporary Python objects alongside active model services.
    active=False;chunk=None
    for line in Path(path).open():
        if '"traceEvents": [' in line:active=True;continue
        if not active:continue
        if line=='  {\n':chunk=[line]
        elif chunk is not None:
            chunk.append(line)
            if line in ('  },\n','  }\n'):
                block=''.join(chunk).rstrip().rstrip(',');chunk=None
                if '"cat": "kernel"' in block:yield json.loads(block)


def profiling():
    output={}
    for mode in ('q4_batched','q4_split_nomask'):
        path=OUT/'compression'/mode/'trace.json';by=Counter();n=0
        for e in kernel_events(path):by[e['name']]+=e['dur']/1000;n+=1
        output[mode]=dict(kernels=n,kernel_work_ms=sum(by.values()),
            top_kernels=[dict(name=name,ms=ms) for name,ms in by.most_common(12)])
    streams=defaultdict(list)
    for e in kernel_events(OUT/'overlap/overlap_trace.json'):
        streams[str(e['args']['stream'])].append((e['ts'],e['ts']+e['dur']))
    def merge(a):
        result=[]
        for x,y in sorted(a):
            if result and x<=result[-1][1]:result[-1][1]=max(result[-1][1],y)
            else:result.append([x,y])
        return result
    merged={s:merge(a) for s,a in streams.items()}
    ordered=sorted(merged,key=lambda s:sum(b-a for a,b in merged[s]),reverse=True)
    a,b=merged[ordered[0]],merged[ordered[1]];i=j=0;overlap=0.
    while i<len(a) and j<len(b):
        overlap+=max(0.,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    output['overlap']=dict(kernel_overlap_ms=overlap/1000,
        kernel_work_ms={s:sum(b-a for a,b in a)/1000 for s,a in merged.items()},
        source='Chrome trace cat=kernel only; no operator or nested record-function double counting')
    atomic(OUT/'profiling_analysis.json',output);return output


def analyze():
    m=json.loads((OUT/'closed_loop_manifest.json').read_text())
    conditions=list(m['conditions'])
    if (OUT/'optimized_manifest.json').exists():conditions.append(json.loads((OUT/'optimized_manifest.json').read_text())['condition'])
    summary=[];episodes=[];requests=[];paired=[];prefix=[];loaded={}
    for cfg in conditions:
        group=[]
        for case in m['cases']:
            d=directory(cfg['id'],case)
            if not (d/'complete.json').exists():continue
            result=json.loads((d/'result.json').read_text());rows=read_rows(d/'infer.jsonl')
            terminal=json.loads((d/'terminal.json').read_text());key=ck(case)
            loaded[(cfg['id'],key)]=(d,result,rows,terminal)
            events=[e for row in rows for e in row['selector_events']]+terminal['selector_events']
            warm_ms=1000*(result['policy_wall_s']-result['capture_s'])
            tail=rows[3:];tail_actions=sum(row['executed_length'] for row in tail)
            tail_ms=sum(row['timing']['policy_wall_ms']-row['timing']['selector_capture_ms'] for row in tail)
            tail_ms+=1000*result['timing']['finish_s']
            categories=Counter(row['route']['request_category'] for row in rows)
            rr=dict(condition=cfg['id'],case=key,success=int(result['success']),actions=result['actions'],
                requests=result['predictions'],wall_s=result['wall_s'],capture_s=result['capture_s'],
                warm_policy_ms=warm_ms,warm_ms_per_action=warm_ms/result['actions'],
                warm_tail_ms=tail_ms,warm_tail_actions=tail_actions,
                warm_tail_ms_per_action=tail_ms/tail_actions if tail_actions else None,
                fulls=result['counts']['full'],drafts=result['counts']['draft'],selectors=len(events),
                selector_wait_ms=sum(e['wait_ms'] for e in events),selector_gpu_span_ms=sum(e['gpu_ms'] for e in events),
                selector_ready_before_wait=sum(e['ready_before_wait'] for e in events),
                score_ms=sum(e['score_ms'] for e in events),
                env_acquire_s=result['timing']['environment_acquire_s'],reset_s=result['timing']['reset_s'],
                execution_s=result['timing']['step_s'],input_save_s=result['timing']['input_save_s'],
                fallbacks=sum(row['route']['fallback'] for row in rows),
                fast_full_prefixes=categories['fast_accept_5'],fast_partial_prefixes=categories['fast_accept_partial'])
            group.append(rr);episodes.append(rr)
            for row in rows:
                t=row['route'];cost=row['timing']
                requests.append(dict(condition=cfg['id'],case=key,request=row['replan'],step=row['step'],
                    path=t['path'],category=t['request_category'],executed=row['executed_length'],
                    full_index=t.get('actual_full_index'),refresh=t['mask_refresh'],
                    policy_ms=cost['policy_wall_ms'],capture_ms=cost['selector_capture_ms'],
                    warm_policy_ms=cost['policy_wall_ms']-cost['selector_capture_ms'],
                    selector_wait_ms=cost['selector_wait_ms'],execution_ms=row['execution_wall_ms'],
                    selector_collect_ms=cost.get('selector_collect_ms',0),selector_launch_ms=cost.get('selector_launch_ms',0),
                    full_path_ms=cost.get('full_path_wall_ms',0),draft_path_ms=cost.get('draft_path_wall_ms',0),
                    verify_path_ms=cost.get('verify_path_wall_ms',0),
                    full_vision_ms=cost.get('full_vision_ms',0),full_vlm_ms=cost.get('full_vlm_ms',0),
                    full_ae_ms=cost.get('full_ae_ms',0),draft_vision_ms=cost.get('draft_vision_ms',0),
                    draft_vlm_ms=cost.get('draft_vlm_ms',0),draft_ae_ms=cost.get('draft_ae_ms',0),
                    verify_ms=cost.get('verify_ae_ms',0),
                    extrapolation_ratio=t.get('draft_extrapolation_ratio'),mask_age=t['mask_age_env_steps']))
        if not group:continue
        totals={key:sum(row[key] for row in group) for key in ('actions','requests','warm_policy_ms','wall_s',
            'capture_s','fulls','drafts','selectors','selector_wait_ms','selector_gpu_span_ms','selector_ready_before_wait',
            'score_ms','fallbacks','warm_tail_ms','warm_tail_actions')}
        s=dict(condition=cfg['id'],n=len(group),target=cfg['n'],success=sum(row['success'] for row in group),
            success_rate=sum(row['success'] for row in group)/len(group),**totals)
        s.update(warm_ms_per_action=totals['warm_policy_ms']/totals['actions'],
            warm_tail_ms_per_action=totals['warm_tail_ms']/totals['warm_tail_actions'] if totals['warm_tail_actions'] else None,
            full_fraction=totals['fulls']/totals['requests'],
            episodes_per_hour=3600*len(group)/totals['wall_s'],actions_per_second=totals['actions']/totals['wall_s'],
            mean_selector_wait_ms=totals['selector_wait_ms']/max(totals['selectors'],1),
            mean_selector_score_ms=totals['score_ms']/max(totals['selectors'],1))
        summary.append(s)
    by={(r['condition'],r['case']):r for r in episodes}
    for cfg in conditions:
        if cfg['id']=='full_d10':continue
        pairs=[(row,by[('full_d10',row['case'])]) for row in episodes if row['condition']==cfg['id'] and ('full_d10',row['case']) in by]
        if not pairs:continue
        row=dict(condition=cfg['id'],n=len(pairs),baseline_success=sum(b['success'] for c,b in pairs),
            candidate_success=sum(c['success'] for c,b in pairs),
            recovered=sum(c['success'] and not b['success'] for c,b in pairs),
            new_failures=sum(b['success'] and not c['success'] for c,b in pairs),
            policy_speedup_geomean=float(np.exp(np.mean([math.log(b['warm_ms_per_action']/c['warm_ms_per_action']) for c,b in pairs]))),
            policy_speedup_pooled=(sum(b['warm_policy_ms'] for c,b in pairs)/sum(b['actions'] for c,b in pairs))/(sum(c['warm_policy_ms'] for c,b in pairs)/sum(c['actions'] for c,b in pairs)),
            tail_policy_speedup_geomean=float(np.exp(np.mean([math.log(b['warm_tail_ms_per_action']/c['warm_tail_ms_per_action']) for c,b in pairs if c['warm_tail_actions'] and b['warm_tail_actions']]))),
            tail_policy_speedup_pooled=(sum(b['warm_tail_ms'] for c,b in pairs)/sum(b['warm_tail_actions'] for c,b in pairs))/(sum(c['warm_tail_ms'] for c,b in pairs)/sum(c['warm_tail_actions'] for c,b in pairs)),
            wall_speedup_geomean=float(np.exp(np.mean([math.log(b['wall_s']/c['wall_s']) for c,b in pairs]))),
            warm_wall_speedup_geomean=float(np.exp(np.mean([math.log((b['wall_s']-b['capture_s'])/(c['wall_s']-c['capture_s'])) for c,b in pairs]))))
        paired.append(row)
    # Every condition has the same first Full and first following input.
    for (condition,key),(d,result,rows,terminal) in loaded.items():
        if condition=='full_d10' or ('full_d10',key) not in loaded:continue
        bd,br,brows,bt=loaded[('full_d10',key)]
        with np.load(d/'arrays.npz') as a,np.load(bd/'arrays.npz') as b:
            check=dict(condition=condition,case=key,
                first_input=rows[0]['input_hash']==brows[0]['input_hash'],
                first_raw=bool(np.allclose(a['raw_0'],b['raw_0'],atol=1e-5,rtol=1e-5)),
                first_actions=bool(np.allclose(a['actions_0'],b['actions_0'],atol=1e-5,rtol=1e-5)),
                second_input=len(rows)>1 and len(brows)>1 and rows[1]['input_hash']==brows[1]['input_hash'])
            check['passed']=all(check[k] for k in ('first_input','first_raw','first_actions','second_input'));prefix.append(check)
    for name,rows in (('summary',summary),('episodes',episodes),('requests',requests),('paired',paired),('prefix',prefix)):
        write_csv(OUT/f'{name}.csv',rows)
    result=dict(summary=summary,paired=paired,prefix_checks=len(prefix),prefix_passed=all(r['passed'] for r in prefix),
        completed=len(episodes),target=sum(c['n'] for c in conditions),main_target=sum(c['n'] for c in m['conditions']))
    atomic(OUT/'analysis.json',result);return result


if __name__=='__main__':
    result=analyze();print(json.dumps(dict(completed=result['completed'],prefix_passed=result['prefix_passed'],paired=result['paired']),ensure_ascii=False))
