"""Trace private real-case native probes and fixed first-prompt sensitivity."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records,pairing


def run():
    groups=records();probes=[];comparisons=[]
    for cid,rows in groups.items():
        for p in (OUT/cid).glob('native_probe.gpu*.worker*.json'):
            data=json.loads(p.read_text());worker=data['worker_binding']['worker_pid'];same=[]
            for r in rows:
                d=Path(r['_directory']);identity=json.loads((d/'identity.json').read_text());binding=identity.get('engineering_binding',{})
                if binding.get('worker_pid')==worker:same.append((d.joinpath('identity.json').stat().st_mtime,r))
            if not same:continue
            first=min(same,key=lambda x:x[0])[1];d=Path(first['_directory'])
            with np.load(d/'input0.npz') as saved:instruction=str(saved['instruction'])
            prompt_cases=[]
            for r in rows:
                with np.load(Path(r['_directory'])/'input0.npz') as saved:current=str(saved['instruction'])
                if current==instruction:prompt_cases.append(r['case'])
            probes.append(dict(condition=cid,physical_gpu=data['worker_binding']['model_gpu'],worker_pid=worker,model_server_pid=data['model_server_pid'],probe_file=str(p),probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                passed=data['passed'],cross_restart=data.get('cross_restart'),
                first_valid_case=first['case'],instruction=instruction,all_valid_cases_with_same_prompt=prompt_cases,
                scope='Private probe response associates rollout worker; first valid identity mtime reconstructs first prompt. Invalid attempts remain separate, so this does not claim an exact count of all probe calls.'))
    for cid,rows in groups.items():
        if not cid.endswith('_no_c'):continue
        source=cid.removesuffix('_no_c');own=[r for r in probes if r['condition']==cid]
        excluded={tuple(c[k] for k in ('suite','task','init')) for p in own for c in p['all_valid_cases_with_same_prompt']}
        sensitivity=[r for r in rows if tuple(r['case'][k] for k in ('suite','task','init')) not in excluded]
        reference={tuple(r['case'][k] for k in ('suite','task','init')):r for r in groups[source]};initial=[]
        for r in rows:
            key=tuple(r['case'][k] for k in ('suite','task','init'))
            if key not in reference:continue
            a,b=Path(r['_directory']),Path(reference[key]['_directory'])
            with np.load(a/'arrays.npz') as aa,np.load(b/'arrays.npz') as bb:
                initial.append(dict(case=list(key),initial_Full10_raw_exact=bool(np.array_equal(aa['raw_0'],bb['raw_0'])),initial_Full10_decoded_exact=bool(np.array_equal(aa['actions_0'],bb['actions_0'])),
                    candidate_arrays_sha256=hashlib.sha256((a/'arrays.npz').read_bytes()).hexdigest(),reference_arrays_sha256=hashlib.sha256((b/'arrays.npz').read_bytes()).hexdigest()))
        comparisons.append(dict(candidate=cid,reference=source,formal80=pairing(rows,groups[source]),
            excluded_fixed_probe_prompt_cases=[list(k) for k in sorted(excluded)],probe_prompt_sensitivity=pairing(sensitivity,groups[source]),
            initial_Full10_comparison=initial,initial_Full10_all_exact=all(c['initial_Full10_raw_exact'] and c['initial_Full10_decoded_exact'] for c in initial),
            initial_Full10_exact_count=sum(c['initial_Full10_raw_exact'] and c['initial_Full10_decoded_exact'] for c in initial),
            formal_samples_unchanged=True,causal_limit='No-c is exploratory; initialization/probe history differs. If effects concentrate on these first prompts, do not identify an independent effect of c.'))
    result=dict(time=time.time(),probes=probes,no_c_comparisons=comparisons)
    atomic(OUT/'probe_prompt_history_receipt.json',result)
    return result

