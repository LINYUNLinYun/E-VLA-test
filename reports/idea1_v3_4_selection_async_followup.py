"""Finish the finite paced comparison and offline diagnosis after free loops."""
import json,os,subprocess,time,traceback
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3_4_selection_async.common import OUT,ROOT,ck,sha,write_csv
from preparations.idea1_v3_4_selection_async import controller as services
from reports.idea1_v3_4_selection_async_analysis import analyze,profiling,read_rows,directory


def equality():
    m=json.loads((OUT/'closed_loop_manifest.json').read_text());checks=[]
    for case in m['cases']:
        if case['task'] not in (0,2):continue
        a=directory('v12_p3_q4_serial_sync',case);b=directory('v12_p3_q4_serial_async',case)
        aa=read_rows(a/'infer.jsonl');bb=read_rows(b/'infer.jsonl')
        equal=len(aa)==len(bb);max_raw=0.;max_action=0.;max_g=0.
        with np.load(a/'arrays.npz') as x,np.load(b/'arrays.npz') as y:
            for i,(ar,br) in enumerate(zip(aa,bb)):
                equal=equal and ar['input_hash']==br['input_hash'] and ar['executed_length']==br['executed_length']
                equal=equal and ar['route']['path']==br['route']['path'] and ar['route']['draft_skip']==br['route']['draft_skip']
                max_raw=max(max_raw,float(np.max(np.abs(x[f'raw_{i}']-y[f'raw_{i}']))))
                max_action=max(max_action,float(np.max(np.abs(x[f'actions_{i}']-y[f'actions_{i}']))))
                equal=equal and bool(np.allclose(x[f'raw_{i}'],y[f'raw_{i}'],atol=1e-5,rtol=1e-5))
                equal=equal and bool(np.allclose(x[f'actions_{i}'],y[f'actions_{i}'],atol=1e-5,rtol=1e-5))
            def gradients(archive):
                return {key.split('_request')[0].split('_terminal')[0]:archive[key] for key in archive.files if key.startswith('gate_gradient_job')}
            gx=gradients(x);gy=gradients(y);equal=equal and gx.keys()==gy.keys()
            for key in gx:
                max_g=max(max_g,float(np.max(np.abs(gx[key]-gy[key]))))
                equal=equal and bool(np.allclose(gx[key],gy[key],atol=1e-4,rtol=1e-3))
        checks.append(dict(case=ck(case),passed=equal,requests=len(aa),raw_max_abs=max_raw,
            action_max_abs=max_action,gradient_max_abs=max_g))
    result=dict(passed=all(c['passed'] for c in checks),checks=checks)
    atomic(OUT/'closed_loop_sync_async_equality.json',result);assert result['passed'];return result


def choose_followup(analysis):
    if (OUT/'followup_manifest.json').exists():return json.loads((OUT/'followup_manifest.json').read_text())
    m=json.loads((OUT/'closed_loop_manifest.json').read_text());by={r['condition']:r for r in analysis['summary']}
    chosen=[]
    for q in (1,2):
        choices=[c for c in m['conditions'] if c['projections']==q and c['mode']=='verified']
        choices.sort(key=lambda c:(-by[c['id']]['success'],by[c['id']]['warm_tail_ms_per_action'],c['id']))
        chosen.append(choices[0]['id'])
    optimized=json.loads((OUT/'optimized_manifest.json').read_text())['condition']
    ids=['full_d10','v12_p3_q4_batch_async']+chosen
    manifest=dict(conditions=[c for c in m['conditions'] if c['id'] in ids]+[optimized],
        cases=[c for c in m['cases'] if c['task'] in (0,2)],pace_hz=20,
        diagnostic_conditions=chosen+[optimized['id']],selection_rule='For each q=1/2: most successes on the fixed 40-case discovery set, then lowest warm-tail policy ms/action. Also retain the independently tested four-direction fused candidate.',
        scope='Paced timing confirmation on the same fixed eight cases; not independent quality holdout.',
        source_files={str(p.relative_to(ROOT)):sha(p) for p in list(Path(__file__).parent.glob('idea1_v3_4_selection_async*.py'))+list((ROOT/'reports/idea1_v3_4_selection_async_optimized').glob('*.py'))},
        closed_loop_manifest_hash=m['manifest_hash'],optimized_manifest_hash=json.loads((OUT/'optimized_manifest.json').read_text())['manifest_hash'],time=time.time())
    manifest['manifest_hash']=digest(manifest);atomic(OUT/'followup_manifest.json',manifest);return manifest


def paced_analysis(manifest):
    recovery_path=OUT/'paced_recovery_manifest.json'
    recovery=json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    cases=manifest['cases']+([] if recovery is None else recovery['cases'])
    excluded=[] if recovery is None else recovery['strict_excluded_cases']
    strict_cases=[case for case in cases if case not in excluded]
    episodes=[];summary=[];loaded={}
    for cfg in manifest['conditions']:
        group=[]
        for case in cases:
            d=OUT/'paced'/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            assert (d/'complete.json').exists()
            if case not in excluded:assert json.loads((d/'pacing_equivalence.json').read_text())['passed']
            r=json.loads((d/'result.json').read_text());rows=read_rows(d/'infer.jsonl');terminal=json.loads((d/'terminal.json').read_text())
            events=[e for row in rows for e in row['selector_events']]+terminal['selector_events']
            full_trigger=[row['timing']['policy_wall_ms']-row['timing']['selector_capture_ms'] for row in rows if row['route']['mask_refresh']]
            tail_actions=sum(row['executed_length'] for row in rows[3:])
            tail_ms=sum(row['timing']['policy_wall_ms']-row['timing']['selector_capture_ms'] for row in rows[3:])+1000*r['timing']['finish_s']
            row=dict(condition=cfg['id'],case=ck(case),success=int(r['success']),actions=r['actions'],requests=r['predictions'],
                wall_s=r['wall_s'],capture_s=r['capture_s'],warm_policy_ms=1000*(r['policy_wall_s']-r['capture_s']),
                policy_ms_per_action=1000*(r['policy_wall_s']-r['capture_s'])/r['actions'],
                tail_policy_ms_per_action=tail_ms/tail_actions,tail_ms=tail_ms,tail_actions=tail_actions,
                warm_wall_ms_per_action=1000*(r['wall_s']-r['capture_s'])/r['actions'],
                pace_sleep_s=r['timing']['pace_sleep_s'],execution_s=r['timing']['step_s'],
                nominal_execution_s=r['actions']/20,
                execution_overrun_s=max(0.,r['timing']['step_s']-r['actions']/20),
                execution_windows_overrun_5ms=sum(row['execution_wall_ms']>50*row['executed_length']+5 for row in rows),
                selectors=len(events),selector_ready=sum(e['ready_before_wait'] for e in events),
                selector_wait_ms=sum(e['wait_ms'] for e in events),
                mean_refresh_full_ms=float(np.mean(full_trigger)) if full_trigger else 0.,
                p95_refresh_full_ms=float(np.percentile(full_trigger,95)) if full_trigger else 0.,
                fulls=r['counts']['full'],drafts=r['counts']['draft'])
            group.append(row);episodes.append(row);loaded[(cfg['id'],ck(case))]=row
        totals={key:sum(r[key] for r in group) for key in ('actions','requests','wall_s','capture_s','warm_policy_ms','selectors','selector_ready','selector_wait_ms','fulls','drafts','execution_s','nominal_execution_s','execution_overrun_s','execution_windows_overrun_5ms')}
        summary.append(dict(condition=cfg['id'],n=len(group),success=sum(r['success'] for r in group),**totals,
            policy_ms_per_action=totals['warm_policy_ms']/totals['actions'],
            warm_wall_ms_per_action=1000*(totals['wall_s']-totals['capture_s'])/totals['actions'],
            episodes_per_hour=3600*len(group)/totals['wall_s'],actions_per_second=totals['actions']/totals['wall_s']))
    pairs=[]
    for cfg in manifest['conditions']:
        if cfg['id']=='full_d10':continue
        got=[(loaded[(cfg['id'],ck(case))],loaded[('full_d10',ck(case))]) for case in strict_cases]
        pairs.append(dict(condition=cfg['id'],n=len(got),
            compute_speedup_geomean=float(np.exp(np.mean([np.log(b['policy_ms_per_action']/c['policy_ms_per_action']) for c,b in got]))),
            compute_speedup_pooled=(sum(b['warm_policy_ms'] for c,b in got)/sum(b['actions'] for c,b in got))/(sum(c['warm_policy_ms'] for c,b in got)/sum(c['actions'] for c,b in got)),
            tail_compute_speedup_geomean=float(np.exp(np.mean([np.log(b['tail_policy_ms_per_action']/c['tail_policy_ms_per_action']) for c,b in got]))),
            tail_compute_speedup_pooled=(sum(b['tail_ms'] for c,b in got)/sum(b['tail_actions'] for c,b in got))/(sum(c['tail_ms'] for c,b in got)/sum(c['tail_actions'] for c,b in got)),
            warm_wall_per_action_speedup_geomean=float(np.exp(np.mean([np.log(b['warm_wall_ms_per_action']/c['warm_wall_ms_per_action']) for c,b in got]))),
            warm_episode_wall_speedup_geomean=float(np.exp(np.mean([np.log((b['wall_s']-b['capture_s'])/(c['wall_s']-c['capture_s'])) for c,b in got]))),
            selectors=sum(c['selectors'] for c,b in got),selector_ready=sum(c['selector_ready'] for c,b in got),
            mean_selector_wait_ms=sum(c['selector_wait_ms'] for c,b in got)/max(sum(c['selectors'] for c,b in got),1)))
    for name,rows in (('episodes',episodes),('summary',summary),('paired',pairs)):write_csv(OUT/f'paced_{name}.csv',rows)
    result=dict(summary=summary,paired=pairs,pacing_equivalence_passed=not bool(excluded),
        strict_pacing_equivalence_passed=True,strict_cases=strict_cases,excluded_cases=excluded,measured_cases=cases)
    atomic(OUT/'paced_analysis.json',result);return result


def main():
    atomic(OUT/'followup_process.json',dict(pid=os.getpid(),time=time.time(),
        source_files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('idea1_v3_4_selection_async*.py')}))
    while not (OUT/'closed_loop_execution_complete.json').exists():
        if (OUT/'closed_loop_failure.json').exists():raise RuntimeError('Free-loop evaluation failed; inspect before followup')
        time.sleep(10)
    while not (OUT/'optimized_execution_complete.json').exists():
        if (OUT/'optimized_failure.json').exists():raise RuntimeError('Optimized four-direction condition failed')
        time.sleep(3)
    analysis=analyze();assert analysis['completed']==analysis['target'] and analysis['prefix_passed']
    from preparations.idea1_v3_4_selection_async.audit import episode as audit
    m=json.loads((OUT/'closed_loop_manifest.json').read_text())
    for cfg in m['conditions']:
        for case in m['cases']:
            d=directory(cfg['id'],case)
            if not (d/'complete.json').exists():continue
            path=OUT/'episode_audits'/cfg['id']/f'{ck(case)}.json'
            if not path.exists():atomic(path,audit(cfg,d))
    assert len(list((OUT/'episode_audits').glob('*/*.json')))==analysis['main_target']
    assert len(list((OUT/'optimized_audits').glob('*.json')))==40
    if not (OUT/'closed_loop_sync_async_equality.json').exists():equality()
    else:assert json.loads((OUT/'closed_loop_sync_async_equality.json').read_text())['passed']
    if not (OUT/'profiling_analysis.json').exists():profiling()
    from reports.idea1_v3_4_selection_async_fusion_analysis import run as fusion_analysis
    if not (OUT/'fusion_analysis.json').exists():fusion_analysis()
    from reports.idea1_v3_4_selection_async_projections import run as projection_analysis
    projection_analysis()
    while not (OUT/'elementwise_batch_probe/released.json').exists():
        if (OUT/'sdpa_batch_probe/supervisor_failure.json').exists():raise RuntimeError('Bounded SDPA probe supervisor failed')
        time.sleep(3)
    manifest=choose_followup(analysis)
    runtime=OUT/'followup_runtime';runtime.mkdir(exist_ok=True);services.OUT=runtime
    services.MODULE='reports.idea1_v3_4_selection_async_optimized'
    (runtime/'cross_service_reference.npz').write_bytes((OUT/'cross_service_reference.npz').read_bytes())
    workers=[];diagnosis=None;diagnostic_worker=None
    try:
        while not workers:
            for gpu,uuid in services.available():
                worker=services.start(gpu,uuid)
                if worker is not None:workers.append(worker)
                if len(workers)>=(2 if (OUT/'paced_recovery_manifest.json').exists() else 4):break
            if not workers:time.sleep(30)
        # Use cards already free at this boundary; do not start a late model
        # after the eight cases have already been claimed.
        while not all(w['phase']=='done' for w in workers):
            for w in workers:
                if w['phase']=='loading':
                    if w['service'].poll() is not None:raise RuntimeError('Followup service failed')
                    try:rpc(w['port'],dict(op='health'))
                    except (OSError,ConnectionError):continue
                    with (w['wd']/'gate.log').open('a') as stream:
                        w['gate']=subprocess.Popen([str(services.MODEL),'-u','-m',services.MODULE+'.gate','--port',str(w['port']),
                            '--output',str(w['wd']/'gate')],cwd=ROOT,env=w['env'],stdout=stream,stderr=subprocess.STDOUT)
                    w['phase']='gate'
                elif w['phase']=='gate' and w['gate'].poll() is not None:
                    if w['gate'].returncode:raise RuntimeError('Followup policy gate failed')
                    services.cross_gate(w)
                    with (w['wd']/'paced.log').open('a') as stream:
                        w['evaluation']=subprocess.Popen([str(services.EVAL),'-u','-m','reports.idea1_v3_4_selection_async_paced',
                            '--port',str(w['port']),'--worker',f"gpu{w['gpu']}"],cwd=ROOT,env=w['render'],stdout=stream,stderr=subprocess.STDOUT)
                    w['phase']='evaluating'
                elif w['phase']=='evaluating' and w['evaluation'].poll() is not None:
                    if w['evaluation'].returncode:raise RuntimeError('Paced evaluation failed')
                    services.finish(w);w['phase']='done'
                atomic(runtime/f"gpu{w['gpu']}_state.json",dict(phase=w['phase'],time=time.time()))
            time.sleep(3)
        paced_analysis(manifest)
        from reports.idea1_v3_4_selection_async_timing import run as timing_analysis
        timing_analysis()
        # New process for offline truth keeps all diagnostic work outside the
        # timing run and guarantees no truth action reaches the evaluated policy.
        chosen=None
        while chosen is None:
            for gpu,uuid in services.available():
                lease=services.lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock',blocking=False)
                try:lease.__enter__()
                except BlockingIOError:continue
                chosen=(gpu,uuid,lease);break
            if chosen is None:time.sleep(30)
        gpu,uuid,lease=chosen
        try:
            env=services.environment(gpu);env['PYTHONPATH']=str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
            out=OUT/'diagnosis';out.mkdir(exist_ok=True)
            atomic(out/'launch.json',dict(gpu=gpu,uuid=uuid,time=time.time()))
            with (out/'run.log').open('a') as stream:
                diagnosis=subprocess.Popen([str(services.MODEL),'-u','-m','reports.idea1_v3_4_selection_async_diagnose',
                    '--output',str(out)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            while diagnosis.poll() is None:time.sleep(3)
            if diagnosis.returncode:raise RuntimeError('Offline diagnosis failed')
        finally:lease.__exit__(None,None,None)
        additional=json.loads((OUT/'paced_recovery_manifest.json').read_text())['cases'] if (OUT/'paced_recovery_manifest.json').exists() else []
        atomic(OUT/'followup_complete.json',dict(time=time.time(),paced_episodes=len(manifest['conditions'])*(len(manifest['cases'])+len(additional)),
            diagnostic_conditions=manifest['diagnostic_conditions']))
    finally:
        for w in workers:
            if w['phase']!='done':services.finish(w)
        if diagnosis is not None and diagnosis.poll() is None:diagnosis.terminate();diagnosis.wait(timeout=30)


if __name__=='__main__':
    try:main()
    except Exception:atomic(OUT/'followup_failure.json',dict(error=traceback.format_exc(),time=time.time()));raise
