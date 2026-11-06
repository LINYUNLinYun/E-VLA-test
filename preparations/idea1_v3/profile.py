"""P0 measurement artifacts; engineering selection requires common complete keys."""
import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np

from preparations.idea1_v3.io import append, atomic


def summarize(attempt):
    attempt=Path(attempt); directory=attempt/'profiling'; directory.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted(directory.glob('profile_worker*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    # One consolidated file for analysis, without rehashing any old artifacts.
    with (directory/'profile_raw.jsonl').open('w') as stream:
        for row in rows: stream.write(json.dumps(row)+'\n')
    episodes=[x for x in rows if x['kind']=='episode']
    manifest=json.loads((attempt/'manifest.json').read_text())
    task_index={x['key']:x for x in manifest['tasks']}
    request_rpc={}
    for row in rows:
        if row['kind']=='request': request_rpc[row['key']]=request_rpc.get(row['key'],0)+row.get('rpc_ms',0)/1000
    with (directory/'episode_wall_decomposition.csv').open('w') as stream:
        fields=['key','execution','suite','termination','wall_s','environment_create_s','reset_s',
                'environment_step_and_capture_s','rpc_roundtrip_s','video_drain_s','consolidated_save_s',
                'known_nonoverlap_s','unattributed_residual_s']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for episode in episodes:
            timing=episode['timings']
            detail={name:timing.get(name,0) for name in ('environment_create_s','reset_s','environment_step_and_capture_s','video_drain_s','consolidated_save_s')}
            detail['rpc_roundtrip_s']=request_rpc.get(episode['key'],0)
            known=sum(detail.values())
            writer.writerow(dict(key=episode['key'],execution=episode['execution'],suite=episode['suite'],
                termination=episode['termination'],wall_s=episode['wall_s'],**detail,
                known_nonoverlap_s=known,unattributed_residual_s=episode['wall_s']-known))
    groups={}
    for row in rows:
        if row['kind']=='request':
            for name,value in row.items():
                if name.endswith('_ms') and isinstance(value,(float,int)):
                    execution=task_index.get(row['key'],{}).get('execution','unknown')
                    groups.setdefault((f'{execution}/request',name),[]).append(value)
        elif row['kind']=='episode':
            key=f"{row['execution']}/{row['termination']}"
            groups.setdefault((key,'episode_wall_s'),[]).append(row['wall_s'])
            for name,value in row['timings'].items(): groups.setdefault((key,name),[]).append(value)
    fields=['group','metric','n','mean','p50','p95']
    with (directory/'profile_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for (group,name),values in sorted(groups.items()):
            writer.writerow(dict(group=group,metric=name,n=len(values),mean=float(np.mean(values)),
                p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95))))
    by={mode:{(e['suite'],e['task'],e['init']):e for e in episodes if e['execution']==mode} for mode in ('A','B')}
    common=set(by['A'])&set(by['B'])
    quality=all(by['A'][key]['success']==by['B'][key]['success'] and by['A'][key]['actions']==by['B'][key]['actions'] for key in common)
    seconds={mode:sum(by[mode][key]['wall_s'] for key in common) for mode in ('A','B')}
    startup=[]
    for path in (attempt/'workers').glob('*/events.jsonl'):
        startup.extend(json.loads(line) for line in path.read_text().splitlines() if json.loads(line).get('kind')=='startup')
    startup_s=sum(row['seconds'] for row in startup)
    # A/B share a resident service. Charge the same one-time service startup to
    # either standalone fixed workload; do not charge it only to the first arm.
    total_cost={mode:seconds[mode]+startup_s for mode in ('A','B')}
    gain=1-seconds['B']/seconds['A'] if seconds['A'] else None
    lines=['# P0 engineering selection','',f'Common complete suite/task/init keys: {len(common)} / 4.',
        f'A total episode seconds: {seconds["A"]:.3f}; B: {seconds["B"]:.3f}.',
        f'Shared service startup/load: {startup_s:.3f} s. Standalone total cost A={total_cost["A"]:.3f}, B={total_cost["B"]:.3f} s (episode includes lazy compile/warmup and final drain).',
        f'Success and executed action count agreement: {quality}.',
        'A = consolidated compressed NPZ; B = consolidated uncompressed NPZ. Scientific configuration unchanged.',
        'Both use the same serial worker, persistent model and no global ledger lock.',
        'Episode wall includes reset, simulation, RPC, telemetry, encoding, saving and drain. Server startup is in worker events.',
        'environment_step_and_capture_s contains telemetry/video enqueue; these intervals must not be summed.',
        'RPC includes model and tensor-copy time; CUDA sub-stages are nested and not additive to RPC.',
        'GPU/CPU utilization and disk-wait signals are unavailable unless separately collected; no utilization-based speed claim.',
        'No asynchronous writer: queue length is zero; synchronous save and drain remain on the main timeline.',
        'Output array equality check below is required before selecting B.']
    equality=True
    for key in sorted(common):
        a=attempt/'episodes'/by['A'][key]['experiment']/by['A'][key]['condition']/'A'/key[0]/f'task{key[1]:02d}'/f'init{key[2]:02d}'/'arrays.npz'
        b=attempt/'episodes'/by['B'][key]['experiment']/by['B'][key]['condition']/'B'/key[0]/f'task{key[1]:02d}'/f'init{key[2]:02d}'/'arrays.npz'
        with np.load(a) as x,np.load(b) as y:
            equality &= set(x.files)==set(y.files) and all(np.array_equal(x[n],y[n]) for n in x.files)
    lines.append(f'Exact A/B consolidated array equality: {equality if common else "unmeasured"}.')
    selection='A'
    # First-use prompt/kernel compilation is inside the first A episodes. A single
    # A/B order cannot fairly assign this cost to B; retain the established path.
    lines.extend([f'Selected execution: {selection}.',
        'B requires at least 10% common-workload improvement and output parity; otherwise retain A.',
        'This first A/B ordering includes first-use prompt/kernel setup in A. No reversed-order confirmation is present: A is retained and B is not declared faster.',
        'No warm-request latency conclusion until microbenchmark is complete. P0 episodes are excluded from scientific tables.'])
    (directory/'optimization_selection.md').write_text('\n\n'.join(lines)+'\n')
    atomic(directory/'selection.json',dict(execution=selection,common_episodes=len(common),gain=gain,
        output_equal=equality,service_startup_s=startup_s,total_cost_s=total_cost,
        fair_cold_start_comparison=False))
    # Standard standalone artifact; each bar is whole episode wall including drain.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axis=plt.subplots(figsize=(10,4))
    for index,row in enumerate(episodes):
        axis.barh(index,row['wall_s'],color='#337ab7' if row['execution']=='A' else '#ee8a31')
    axis.set_yticks(range(len(episodes)),[f"{e['execution']} {e['suite']} {e['termination']}" for e in episodes])
    axis.set_xlabel('Episode wall seconds (including drain)'); figure.tight_layout()
    figure.savefig(directory/'profile_timeline.png',dpi=160); plt.close(figure)
    with (attempt/'summaries.csv').open('w') as stream:
        writer=csv.writer(stream); writer.writerow(['condition','execution','suite','successes','episodes','actions','wall_s','steps_per_s','requests_per_s'])
        for condition,mode,suite in sorted({(x['condition'],x['execution'],x['suite']) for x in episodes}):
            subset=[x for x in episodes if (x['condition'],x['execution'],x['suite'])==(condition,mode,suite)]
            seconds=sum(x['wall_s'] for x in subset); actions=sum(x['actions'] for x in subset); requests=sum(x['predictions'] for x in subset)
            writer.writerow([condition,mode,suite,sum(x['success'] for x in subset),len(subset),actions,seconds,actions/seconds,requests/seconds])
    print(json.dumps(dict(episodes=len(episodes),common=len(common),selection=selection)))


def micro(attempt):
    """Replay the same twenty real observations, reset route state each repetition."""
    from preparations.idea1_v3.policy import Policy
    attempt=Path(attempt); manifest=json.loads((attempt/'manifest.json').read_text())
    files=sorted((attempt/'episodes').glob('E1/*/A/*/task00/init00/profile_inputs.pkl'))
    if len(files)!=4: raise ValueError('P0 micro requires completed A episode in each suite')
    sequences=[pickle.loads(path.read_bytes()) for path in files]
    if any(len(seq)!=5 for seq in sequences): raise ValueError('Need exactly 20 real saved points; report short phase coverage without inventing inputs')
    start=time.perf_counter(); policy=Policy(manifest['conditions'][0]['config'])
    append(attempt/'profiling'/'micro.jsonl',dict(kind='load',seconds=time.perf_counter()-start))
    measured=[]
    for repeat in range(6):
        for sequence in sequences:
            budget=attempt/'profiling'/'budget.json'
            if budget.exists() and time.monotonic()-json.loads(budget.read_text())['monotonic_start']>=3600:
                atomic(attempt/'profiling'/'micro_incomplete.json',dict(measured=len(measured),reason='shared P0 60-minute budget exhausted; no new replay segment started'))
                return
            policy.reset()
            for case in sequence:
                tick=time.perf_counter(); result=policy.predict(**case)
                policy.acknowledge(result['actions'][:result['execute_length']])
                elapsed=(time.perf_counter()-tick)*1000
                row=dict(kind='warmup' if repeat==0 else 'measurement',repeat=repeat,
                    ms=elapsed,route=result['route'],timing=result['timing'])
                # First replay initializes graph/prompt paths. It is startup, not hot latency.
                append(attempt/'profiling'/'micro.jsonl',row)
                if repeat: measured.append(elapsed)
    atomic(attempt/'profiling'/'micro_summary.json',dict(n=len(measured),mean=float(np.mean(measured)),
        p50=float(np.percentile(measured,50)),p95=float(np.percentile(measured,95)),
        label='state-reset sequence microbenchmark; not closed-loop average',
        coverage='real first five replans per suite task0/init0; phase diversity not guaranteed',
        warmup_requests=20,measurement_requests=100))
