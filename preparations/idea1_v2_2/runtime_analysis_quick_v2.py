from benchmark_light_revision import POLICY
from quick_stage_scope_v2 import INITS, load_scope, check_prediction_metadata, verify_protocol_receipts
"""Recompute paired statistics, full-horizon errors, latency and completion evidence."""
import csv, html, importlib.util, json
from pathlib import Path
import numpy as np
from common import ROOT, SUITES, atomic_json, planned, sha, digest

def read(path): return json.loads(Path(path).read_text())
def rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines()]
def table(path,data,fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    receipt=path.parent/'analysis_input_receipt.json'
    if data and receipt.exists() and fields is None: data=[dict(x,source_receipt='analysis_input_receipt.json',source_receipt_sha256=sha(receipt)) for x in data]
    fields=fields or list(data[0]) if data else fields or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(data)
def bootstrap(values):
    """Stratified task-cluster bootstrap, with every task's three main-stage init paired."""
    if set(values)!={(s,t,i) for s in SUITES for t in range(10) for i in INITS}: return None,None
    rng=np.random.default_rng(1701); draws=[]
    groups={s:[np.mean([values[(s,t,i)] for i in INITS]) for t in range(10)] for s in SUITES}
    for _ in range(2000): draws.append(np.mean([np.mean(np.asarray(groups[s])[rng.integers(0,10,10)]) for s in SUITES])*100)
    return tuple(float(x) for x in np.percentile(draws,[2.5,97.5]))
def native_control(conditions,model,mask):
    selected=[c for c in conditions if c['model']==model and c['mask']==mask]
    native=next((c for c in selected if c['schedule']['kind']=='native' and c['schedule']['num_steps']>2),None)
    return native if native is not None else next(c for c in selected if c['schedule'].get('m')==.5)
def trajectory_distance(a,b):
    left=rows(Path(a['directory'])/'telemetry.jsonl'); right=rows(Path(b['directory'])/'telemetry.jsonl'); n=min(len(left),len(right))
    delta=np.linalg.norm(np.asarray([x['eef_pos'] for x in left[:n]])-np.asarray([x['eef_pos'] for x in right[:n]]),axis=1)
    sustained=next((i for i in range(max(0,n-4)) if np.all(delta[i:i+5]>.05)),None)
    return dict(reference_key=a['key'],condition_key=b['key'],common_steps=n,first_sustained_50mm_step=sustained,same_time_eef_p95_mm=float(np.percentile(delta,95)*1000),reference_telemetry_sha256=sha(Path(a['directory'])/'telemetry.jsonl'),condition_telemetry_sha256=sha(Path(b['directory'])/'telemetry.jsonl'),interpretation='descriptive spatial and timing difference, not causal failure attribution')
def audit(root,strict=True,check_outputs=True):
    root=Path(root).resolve(); m=read(root/'condition_manifest.json'); scope=load_scope(root)
    expected={x['key']:x for x in planned(m) if x['init'] in INITS}
    valid={key:dict(scope['valid_records'][key],directory=scope['valid_records'][key]['resolved_directory']) for key in scope['main_keys']}
    errors=[]; conditions={c['id']:c for c in m['conditions']}
    from runtime_data import audited_protocol
    check_prediction=check_prediction_metadata
    try: verify_protocol_receipts(root)
    except Exception as exc: errors.append(dict(scope='freeze',error=str(exc)))
    bench={}; offline={}; parity={}
    for model in m['protocol_hashes']:
        try:
            parity[model]=not any(error.get('scope')=='freeze' for error in errors)
        except Exception: parity[model]=False
        conds=[c for c in m['conditions'] if c['model']==model]; bs=[]; off=[]
        try: protocol=audited_protocol(root,model)
        except Exception: protocol={}
        paired_noise={}
        for c in conds:
            for point in range(32):
                op=root/'offline'/model/f'point{point:02d}'/(c['id']+'.json')
                try:
                    x=read(op); source=m['observations'][model][point]
                    check_prediction(x,c,protocol,source['seed']); assert x['input_hash']==source['content_hash']
                    if point in paired_noise: assert paired_noise[point]==x['noise_hash']
                    paired_noise[point]=x['noise_hash']; off.append((c['id'],point))
                except Exception: pass
                for repeat in range(sum(POLICY[model])):
                    p=root/'benchmark'/model/'samples'/f"p{point:02d}_r{repeat:02d}_{c['id']}.json"
                    try:
                        x=read(p); assert x['condition']==c['id'] and x['point']==point and x['repeat']==repeat and x['independent_forward'] is True
                        source=m['observations'][model][point]; check_prediction(x,c,protocol,source['seed'])
                        assert x['input_hash']==source['content_hash'] and x['noise_hash']==paired_noise[point]
                        assert x['phase']==('warmup' if repeat<POLICY[model][0] else 'measurement') and x['timing']['action_generation_ms']>0 and x['timing']['server_total_ms']>0 and x['timing']['client_roundtrip_ms']>0
                        bs.append((c['id'],point,repeat))
                    except Exception: pass
        bench[model]=dict(valid=len(bs),expected=len(conds)*32*sum(POLICY[model]),complete=len(bs)==len(conds)*32*sum(POLICY[model]))
        offline[model]=dict(valid=len(off),expected=len(conds)*32,complete=len(off)==len(conds)*32)
    def coverage(selected):
        keys={x['key'] for x in expected.values() if x['condition'] in selected}
        return dict(valid=len(keys&valid.keys()),expected=len(keys),complete=keys<=valid.keys())
    extension={c['id'] for c in m['conditions'] if c['extension']}; main={c['id'] for c in m['conditions']} - extension
    e1={c['id'] for c in m['conditions'] if c['model']!='db-cogact' and c['schedule']['kind']=='flow'}
    e2={c['id'] for c in m['conditions'] if c['model']!='pi0' and (c['schedule']['kind']=='native' or c['model']=='xiaomi-robotic0' and c['schedule'].get('m')==.5)}
    result=dict(complete=False,expected=len(expected),valid=len(valid),missing=sorted(set(expected)-valid.keys()),errors=errors,main=coverage(main),E1=coverage(e1),E2=coverage(e2),db_extension=coverage(extension),parity=parity,benchmark=bench,offline=offline,manifest_sha256=sha(root/'condition_manifest.json'))
    result.update(quick_stage_complete=False,original_full_complete=scope['original_full_complete'],original_planned=10800,extras_valid=len(scope['extra_keys']),verification_scope=scope['verification_scope'],independent_C01_C08_all_passed=False)
    result['complete']=not result['missing'] and not errors and all(parity.values()) and all(x['complete'] for x in bench.values()) and all(x['complete'] for x in offline.values())
    if check_outputs:
        checks={}; required=['report.md','success_by_condition_suite.csv','paired_comparisons.csv','action_error_by_offset_phase.csv','action_error_summary.csv','latency_breakdown.csv','action_gen_speedup.csv','failure_stage_counts.csv','trajectory_index.csv','media_index.md','gallery.html']
        try: output_hashes=read(root/'analysis_sources.json')
        except Exception: output_hashes={}
        for name in required:
            try:
                path=root/name
                if sha(path)!=output_hashes.get(name): raise ValueError('Derived artifact source hash mismatch/missing')
                if name.endswith('.csv'):
                    with path.open() as f: data=list(csv.DictReader(f))
                    if not data: raise ValueError('Required result table has no data')
                    if name=='success_by_condition_suite.csv':
                        assert {(x['condition'],x['suite']) for x in data}=={(c,s) for c in conditions for s in (*SUITES,'overall')}
                        assert all(int(x['denominator'])==(120 if x['suite']=='overall' else 30) for x in data)
                        for x in data:
                            actual=[int(e['success']) for e in valid.values() if e['condition']==x['condition'] and (x['suite']=='overall' or e['suite']==x['suite'])]
                            assert int(x['successes'])==sum(actual) and len(actual)==int(x['denominator']) and np.isclose(float(x['success_rate']),np.mean(actual))
                    if name in ('action_gen_speedup.csv','latency_breakdown.csv'):
                        assert {x['condition'] for x in data}==set(conditions)
                        assert all(int(x['denominator'])==32*POLICY[conditions[x['condition']]['model']][1] for x in data)
                    if name=='latency_breakdown.csv':
                        cache={}
                        for x in data:
                            if x['condition'] not in cache:
                                c=conditions[x['condition']]; cache[x['condition']]=[read(p) for p in (root/'benchmark'/c['model']/'samples').glob('*_'+c['id']+'.json') if read(p)['phase']=='measurement']
                            values=[r['timing'][x['component']] for r in cache[x['condition']]]
                            assert np.isclose(float(x['mean_ms']),np.mean(values)) and np.isclose(float(x['p50_ms']),np.percentile(values,50)) and np.isclose(float(x['p95_ms']),np.percentile(values,95))
                    if name=='trajectory_index.csv': assert {x['episode_key'] for x in data}==set(expected)
                    if name=='action_error_by_offset_phase.csv':
                        observed={(x['condition'],int(x['point']),int(x['offset'])) for x in data}
                        assert observed=={(c['id'],p,o) for c in m['conditions'] for p in range(32) for o in range(1,read(root/'protocols'/f"{c['model']}.json")['H']+1)}
                        cache={}
                        for x in data:
                            a=root/x['source']; b=a.with_name(x['reference']+'.json')
                            for p in (a,b):
                                if str(p) not in cache: cache[str(p)]=np.asarray(read(p)['actions'])
                            offset=int(x['offset'])-1; delta=cache[str(a)][offset]-cache[str(b)][offset]
                            assert sha(a)==x['source_sha256'] and np.isclose(float(x['translation_l2']),np.linalg.norm(delta[:3])) and np.isclose(float(x['rotation_l2']),np.linalg.norm(delta[3:6]))
                else:
                    text=path.read_text()
                    if name=='media_index.md': assert all(e in text for e in expected)
                    if name=='gallery.html': assert '<a href=' in text and '.mp4' in text
                    if name=='report.md': assert all(term in text for term in ('E1','E2','DB','completion_audit.json','SHA256'))
                checks[name]=dict(valid=True,sha256=sha(path))
            except Exception as exc: checks[name]=dict(valid=False,error=str(exc))
        result['deliverables']=checks; result['complete']=result['complete'] and all(x['valid'] for x in checks.values())
    result['quick_stage_complete']=result['complete']
    atomic_json(root/'completion_audit.json',result)
    if strict and not result['complete']: raise RuntimeError(f"Incomplete: {len(valid)}/{len(expected)} valid episodes; inspect completion_audit.json")
    return result,valid

def analyze(root):
    root=Path(root).resolve(); m=read(root/'condition_manifest.json'); completion,valid=audit(root,False,False); by={c['id']:{} for c in m['conditions']}; sources={}
    inputs=[root/'condition_manifest.json',root/'episode_ledger.json']+[Path(e['directory'])/'validated.json' for e in valid.values()]
    inputs+=list((root/'offline').glob('*/*/*.json'))+list((root/'benchmark').glob('*/samples/*.json'))+list((root/'protocols').glob('*.json'))+list((root/'parity').glob('*.json'))
    atomic_json(root/'analysis_input_receipt.json',dict(analysis_code_sha256=sha(__file__),raw_inputs={str(p.relative_to(root)):sha(p) for p in inputs if p.exists()},note='Independent CPU analysis revision; raw scientific runtime and valid episodes remain unchanged'))
    for e in valid.values():
        by[e['condition']][(e['suite'],e['task'],e['init'])]=int(e['result']['success']); sources[(e['condition'],e['suite'],e['task'],e['init'])]=e
    success=[]
    for c in m['conditions']:
        for suite in (*SUITES,'overall'):
            values=[v for (s,t,i),v in by[c['id']].items() if suite=='overall' or suite==s]
            success.append(dict(condition=c['id'],suite=suite,successes=sum(values),denominator=len(values),expected=120 if suite=='overall' else 30,success_rate=np.mean(values) if values else None,source='episode_ledger.json'))
    table(root/'success_by_condition_suite.csv',success)
    comparisons=[]
    def compare(label,terms):
        keys=set.intersection(*(set(by[c]) for c,w in terms)); values={k:sum(w*by[c][k] for c,w in terms) for k in keys}; lo,hi=bootstrap(values)
        comparisons.append(dict(comparison=label,terms=json.dumps(terms),pairs=len(keys),difference_pp=float(np.mean(list(values.values()))*100) if keys else None,ci95_low_pp=lo,ci95_high_pp=hi,bootstrap_repetitions=2000,bootstrap_seed=1701,source='episode_ledger.json'))
    for c in m['conditions']:
        same=[x for x in m['conditions'] if x['model']==c['model'] and x['mask']==c['mask']]
        custom=c['schedule']['kind'] in ('flow','ddim_custom')
        control=next((x for x in same if x['schedule']['kind']==c['schedule']['kind'] and x['schedule'].get('m')==.5),None)
        if custom: compare('E1 '+c['id']+' minus '+control['id'],[(c['id'],1),(control['id'],-1)])
        if c['mask']!='full':
            full=next(x for x in m['conditions'] if x['model']==c['model'] and x['mask']=='full' and x['schedule']==c['schedule']); compare(c['id']+' minus Full',[(c['id'],1),(full['id'],-1)])
            if custom:
                fcontrol=next(x for x in m['conditions'] if x['model']==c['model'] and x['mask']=='full' and x['schedule']['kind']==c['schedule']['kind'] and x['schedule'].get('m')==.5)
                compare('E1 '+c['id']+' interaction',[(c['id'],1),(control['id'],-1),(full['id'],-1),(fcontrol['id'],1)])
    for model in ('xiaomi-robotic0','db-cogact'):
        reduced={}; controls={}
        for mask in sorted({c['mask'] for c in m['conditions'] if c['model']==model}):
            control=native_control(m['conditions'],model,mask); d2=next(c for c in m['conditions'] if c['model']==model and c['mask']==mask and c['schedule']['num_steps']==2 and (c['schedule']['kind']=='native' if model=='db-cogact' else c['schedule'].get('m')==.5))
            reduced[mask]=d2; controls[mask]=control; compare('E2 '+d2['id']+' minus native',[(d2['id'],1),(control['id'],-1)])
        variant=next(mask for mask in reduced if mask!='full')
        compare('E2 '+model+' interaction',[(reduced[variant]['id'],1),(controls[variant]['id'],-1),(reduced['full']['id'],-1),(controls['full']['id'],1)])
    table(root/'paired_comparisons.csv',comparisons)
    latency=[]; aggregate={}
    for c in m['conditions']:
        sample=[read(p) for p in (root/'benchmark'/c['model']/'samples').glob('*_'+c['id']+'.json')]; measured=[x for x in sample if x['phase']=='measurement']
        for component in sorted({k for x in measured for k in x['timing']}):
            values=[x['timing'][component] for x in measured if isinstance(x['timing'].get(component),(float,int)) and not isinstance(x['timing'].get(component),bool)]
            if not values: continue
            stat=dict(condition=c['id'],component=component,denominator=len(values),mean_ms=float(np.mean(values)),p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),source=f"benchmark/{c['model']}/samples")
            latency.append(stat); aggregate[(c['id'],component)]=stat
    table(root/'latency_breakdown.csv',latency)
    speed=[]
    for c in m['conditions']:
        current=aggregate.get((c['id'],'action_generation_ms')); control=native_control(m['conditions'],c['model'],c['mask']); full=native_control(m['conditions'],c['model'],'full')
        if not current: continue
        observed=list(by[c['id']].values())
        row=dict(condition=c['id'],model=c['model'],mask=c['mask'],schedule=json.dumps(c['schedule']),denominator=current['denominator'],same_mask_control=control['id'],full_control=full['id'],control_type='u50' if c['model']=='pi0' else 'native_D',successes=sum(observed),episode_denominator=len(observed),success_rate=float(np.mean(observed)) if observed else None,source='latency_breakdown.csv',source_sha256=sha(root/'latency_breakdown.csv'))
        for stat in ('mean_ms','p50_ms','p95_ms'): row['action_generation_'+stat]=current[stat]
        for label,ref in [('same_mask_control',control),('full_control',full)]:
            baseline=aggregate.get((ref['id'],'action_generation_ms'))
            for stat in ('mean_ms','p50_ms'):
                row[label+'_'+stat+'_speedup']=baseline[stat]/current[stat] if baseline else None
                row[label+'_'+stat+'_reduction_pct']=(1-current[stat]/baseline[stat])*100 if baseline else None
        full_same=next(x for x in m['conditions'] if x['model']==c['model'] and x['mask']=='full' and x['schedule']==c['schedule'])
        baseline=aggregate.get((full_same['id'],'action_generation_ms'))
        for stat in ('mean_ms','p50_ms'): row['same_schedule_full_'+stat+'_speedup']=baseline[stat]/current[stat] if baseline else None
        for component in ('server_total_ms','client_roundtrip_ms'):
            summary=aggregate.get((c['id'],component))
            for stat in ('mean_ms','p50_ms','p95_ms'): row[component+'_'+stat]=summary[stat] if summary else None
        speed.append(row)
    table(root/'action_gen_speedup.csv',speed)
    errors=[]
    for c in m['conditions']:
        for point in m['observations'][c['model']]:
            p=root/'offline'/c['model']/f"point{point['index']:02d}"/(c['id']+'.json')
            if not p.exists(): continue
            same=[x for x in m['conditions'] if x['model']==c['model'] and x['mask']==c['mask']]
            refs=[('same_mask_u50',next((x for x in same if x['schedule']['kind']==c['schedule']['kind'] and x['schedule'].get('m')==.5),native_control(m['conditions'],c['model'],c['mask'])))]
            if c['mask']!='full': refs.append(('same_schedule_full',next(x for x in m['conditions'] if x['model']==c['model'] and x['mask']=='full' and x['schedule']==c['schedule'])))
            a=np.asarray(read(p)['actions'])
            for label,ref in refs:
                q=p.with_name(ref['id']+'.json')
                if not q.exists(): continue
                b=np.asarray(read(q)['actions']); delta=a-b
                for offset,d in enumerate(delta,1):
                    protocol=read(root/'protocols'/f"{c['model']}.json"); r=protocol.get('R_by_suite',protocol['R']); r=r[point['suite']] if isinstance(r,dict) else r
                    row=dict(condition=c['id'],reference=ref['id'],comparison=label,point=point['index'],suite=point['suite'],phase=point.get('phase','unknown'),offset=offset,first_five=offset<=5,execution_prefix=offset<=r,H=len(delta),R=r,denominator=1,translation_l2=float(np.linalg.norm(d[:3])),rotation_l2=float(np.linalg.norm(d[3:6])),gripper_absolute=float(abs(d[6])),gripper_direction_mismatch=int(np.sign(a[offset-1,6])!=np.sign(b[offset-1,6])),units='model native action units; not physical millimeters',source=str(p.relative_to(root)),source_sha256=sha(p))
                    for axis,value in zip('xyz',d[:3]): row[axis+'_signed']=float(value); row[axis+'_absolute']=float(abs(value))
                    for axis,value in zip('xyz',d[3:6]): row['rotation_'+axis+'_signed']=float(value); row['rotation_'+axis+'_absolute']=float(abs(value))
                    errors.append(row)
    table(root/'action_error_by_offset_phase.csv',errors)
    metrics=['translation_l2','rotation_l2','gripper_direction_mismatch','gripper_absolute']+[a+s for a in ('x','y','z','rotation_x','rotation_y','rotation_z') for s in ('_signed','_absolute')]
    groups={}
    for x in errors:
        for window in ('full_H',*(['first_1_5'] if x['first_five'] else []),*(['executed_R'] if x['execution_prefix'] else [])):
            key=(x['condition'],x['reference'],x['phase'],window,'all')
            groups.setdefault(key,[]).append(x)
        groups.setdefault((x['condition'],x['reference'],x['phase'],'offset',x['offset']),[]).append(x)
    summaries=[]
    for (condition,reference,phase,window,offset),group in groups.items():
        for metric in metrics:
            values=[x[metric] for x in group]
            summaries.append(dict(condition=condition,reference=reference,phase=phase,window=window,offset=offset,metric=metric,denominator=len(values),observations=len({x['point'] for x in group}),mean=float(np.mean(values)),p05=float(np.percentile(values,5)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),units='native action units; gripper mismatch is fraction',source='action_error_by_offset_phase.csv',source_sha256=sha(root/'action_error_by_offset_phase.csv')))
    table(root/'action_error_summary.csv',summaries)
    failure=[]; media=[]; media_links=[]; trajectories=[]; plots=[]; selections=[]; deviations=[]
    if errors:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for condition,reference in sorted({(x['condition'],x['reference']) for x in errors}):
            data=[x for x in errors if x['condition']==condition and x['reference']==reference]
            fig,axes=plt.subplots(1,3,figsize=(13,3.5))
            for ax,metric in zip(axes,('translation_l2','rotation_l2','gripper_direction_mismatch')):
                offsets=sorted({x['offset'] for x in data}); values=[[x[metric] for x in data if x['offset']==o] for o in offsets]
                ax.plot(offsets,[np.mean(v) for v in values]); ax.fill_between(offsets,[np.percentile(v,5) for v in values],[np.percentile(v,95) for v in values],alpha=.2)
                ax.axvline(5,color='grey',linestyle=':'); ax.set(xlabel='Future action offset (1..H)',ylabel=metric+' (native units)' if metric!='gripper_direction_mismatch' else 'Gripper direction disagreement fraction')
            fig.suptitle(condition+' vs '+reference+'; mean and p05–p95 across 32 fixed observations'); fig.tight_layout(); output=root/'figures'/f'actions_{condition}_vs_{reference}.png'; output.parent.mkdir(exist_ok=True); fig.savefig(output,dpi=130); plt.close(fig); plots.append(str(output.relative_to(root)))
    spec=importlib.util.spec_from_file_location('old_analysis',ROOT/'preparations/idea1_v2_1/runtime_analysis.py'); old=importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0,str(ROOT/'preparations/idea1_v2_1')); spec.loader.exec_module(old)
    for c in m['conditions']:
        episodes=[x for x in valid.values() if x['condition']==c['id']]; phases={}
        for e in episodes:
            d=Path(e['directory']); telemetry=rows(d/'telemetry.jsonl'); reached={x.get('semantic_phase','unknown') for x in telemetry}
            far=e['result'].get('furthest_phase','unknown')
            for phase in reached|{far}:
                key=phase; item=phases.setdefault(key,dict(condition=c['id'],phase=phase,reached=0,reached_eventual_success=0,furthest=0,denominator=len(episodes),source='episode_ledger.json'))
                item['reached']+=int(phase in reached); item['reached_eventual_success']+=int(phase in reached and e['success']); item['furthest']+=int(far==phase)
            video=Path(e['directory'])/'video.mp4'; rel=video.relative_to(root); media.append(f"- `{e['key']}` [{c['id']} {e['suite']}/{e['task']}/{e['init']}]({rel}) SHA256 `{e['result']['artifacts']['video.mp4']}`")
            media_links.append((str(rel),f"{c['id']} {e['suite']}/{e['task']}/{e['init']} {e['key']}"))
            trajectories.append(dict(condition=c['id'],suite=e['suite'],task=e['task'],init=e['init'],episode_key=e['key'],telemetry=str((d/'telemetry.jsonl').relative_to(root)),telemetry_sha256=sha(d/'telemetry.jsonl'),video=str(rel),success=e['success']))
        failure+=list(phases.values())
        if c['mask']=='full': continue
        full=next(x for x in m['conditions'] if x['model']==c['model'] and x['mask']=='full' and x['schedule']==c['schedule']); selected=set()
        for key in sorted(by[c['id']]):
            if key not in by[full['id']]: continue
            a=sources[(full['id'],*key)]; b=sources[(c['id'],*key)]
            deviations.append(dict(reference=full['id'],condition=c['id'],suite=key[0],task=key[1],init=key[2],comparison='same_schedule_full_variant',**trajectory_distance(a,b)))
            kind='double_success' if by[c['id']][key] and by[full['id']][key] else 'new_failure' if not by[c['id']][key] and by[full['id']][key] else None
            if kind is None or kind in selected: continue
            selected.add(kind); a=sources[(full['id'],*key)]; b=sources[(c['id'],*key)]; output=root/'figures'/f"{c['id']}_{kind}.png"; output.parent.mkdir(exist_ok=True)
            deviation=old.trajectory_plot(rows(Path(a['directory'])/'telemetry.jsonl'),rows(Path(b['directory'])/'telemetry.jsonl'),output,f"{full['id']} vs {c['id']} {key}; physical simulation steps")
            plots.append(str(output.relative_to(root)))
        selections.append(dict(condition=c['id'],comparison='same_schedule_full_variant',available=sorted(selected),absent=sorted({'double_success','new_failure'}-selected)))
    for model in ('xiaomi-robotic0','db-cogact'):
        masks=sorted({c['mask'] for c in m['conditions'] if c['model']==model})
        for mask in masks:
            control=native_control(m['conditions'],model,mask); reduced=next(c for c in m['conditions'] if c['model']==model and c['mask']==mask and c['schedule']['num_steps']==2 and (c['schedule']['kind']=='native' if model=='db-cogact' else c['schedule'].get('m')==.5)); selected=set()
            for key in sorted(by[reduced['id']]):
                if key not in by[control['id']]: continue
                a=sources[(control['id'],*key)]; b=sources[(reduced['id'],*key)]
                deviations.append(dict(reference=control['id'],condition=reduced['id'],suite=key[0],task=key[1],init=key[2],comparison='same_mask_native_vs_D2',**trajectory_distance(a,b)))
                kind='double_success' if by[reduced['id']][key] and by[control['id']][key] else 'new_failure' if not by[reduced['id']][key] and by[control['id']][key] else None
                if kind is None or kind in selected: continue
                selected.add(kind); a=sources[(control['id'],*key)]; b=sources[(reduced['id'],*key)]; output=root/'figures'/f"native_vs_d2_{model}_{mask}_{kind}.png"; output.parent.mkdir(exist_ok=True)
                deviation=old.trajectory_plot(rows(Path(a['directory'])/'telemetry.jsonl'),rows(Path(b['directory'])/'telemetry.jsonl'),output,f"{control['id']} blue vs {reduced['id']} orange {key}; same mask native D vs D2")
                plots.append(str(output.relative_to(root)))
            selections.append(dict(condition=reduced['id'],comparison='same_mask_native_vs_D2',available=sorted(selected),absent=sorted({'double_success','new_failure'}-selected)))
    table(root/'failure_stage_counts.csv',failure); table(root/'trajectory_index.csv',trajectories)
    table(root/'sustained_trajectory_deviation.csv',deviations); atomic_json(root/'representative_selection.json',selections)
    (root/'media_index.md').write_text('\n'.join(media)+'\n')
    (root/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>v2.2</title>'+''.join(f'<img src="{html.escape(p)}" style="max-width:100%">' for p in plots)+''.join(f'<p><a href="{html.escape(p)}">{html.escape(label)}</a></p>' for p,label in media_links))
    text=f"# idea1 v2.2 快速验证阶段（非原10800全量完成）\n\n主批次有效唯一 episode：{len(valid)}/{completion['expected']}。主范围：{completion['main']}。DB 扩展：{completion['db_extension']}。\n\n当前完成状态：{completion['complete']}。所有缺失键与技术异常见 completion_audit.json；缺失样本不当作失败或成功。\n\n成功率和配对差异见 success_by_condition_suite.csv、paired_comparisons.csv。按 suite 分层、task 成簇保留三个 init（6/7/8），2000 次 bootstrap，seed=1701。候选最高分仅属探索结果，未经多重比较确认。\n\n动作生成独立加速比见 action_gen_speedup.csv，分项时延见 latency_breakdown.csv。测速每条件32真实观测；DB保留5预热/20测量（分母640），π/X为1预热/5测量（分母160）；共享缓存离线成本不能替代独立部署时延。\n\n同观测全 H 误差见 action_error_by_offset_phase.csv。平移、旋转按原生单位独立报告，不能将混合动作维度解释为毫米。阶段来自参考 Full 轨迹，unknown 保留；最远阶段不等于失败原因。\n\n代表 XYZ 图固定选 suite/task/init 排序首个双成功和首个新增失败；无匹配类型时不绘制。Full 蓝色、变体橙色；共同仿真步轨迹距离同时包含空间和节奏差。完整视频与哈希见 media_index.md，原始轨迹索引见 trajectory_index.csv。\n\n协议来源、代码和 checkpoint 证据见 condition_manifest.json 与 protocols/；原始 validated.json 与 ledger 保留失败及 invalid 历史。\n"
    text+='\nE1 完成状态：'+json.dumps(completion['E1'],ensure_ascii=False)+'；E2 完成状态：'+json.dumps(completion['E2'],ensure_ascii=False)+'；DB 扩展完成状态：'+json.dumps(completion['db_extension'],ensure_ascii=False)+'。\n\n缺失选图类型（未用其他类型替代）：\n'+''.join(f"- {x['condition']} {x['comparison']}: {','.join(x['absent']) or '无缺失'}\n" for x in selections)
    text+='\n数据源 SHA256：condition_manifest.json `'+sha(root/'condition_manifest.json')+'`；episode_ledger.json `'+(sha(root/'episode_ledger.json') if (root/'episode_ledger.json').exists() else 'missing')+'`。各表原始来源及哈希列、analysis_sources.json 可重算；动作偏差分位数与前1–5/实际R/完整H窗口见 action_error_summary.csv，持续偏离描述见 sustained_trajectory_deviation.csv。\n'
    text+='\n本阶段由用户授权提前停止原10800计划补量；主批次54条件×4suite×10task×3init=6480。extras只在独立supplement表列实际分母，不混主成功率。复用worker逐episode验证、已有独立媒体覆盖和测速证据；本轮未独立全量复验raw/视频，不声称旧C01–C08全过。\\n'
    (root/'report.md').write_text(text)
    atomic_json(root/'analysis_sources.json',{p.name:sha(p) for p in root.iterdir() if p.suffix in ('.csv','.md','.html')})
    audit(root,False)



