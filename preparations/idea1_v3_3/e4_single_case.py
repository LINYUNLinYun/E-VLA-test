"""Audit the sole300-case difference without attributing it to verification."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records,summary,pairing


def run():
    ids=['E4_tau0.8','E4_periodic'];tail=Path('episodes/libero_goal/task03/init06');dirs=[OUT/c/tail for c in ids]
    logs=[[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()] for d in dirs]
    tele=[[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()] for d in dirs]
    identities=[json.loads((d/'identity.json').read_text()) for d in dirs];results=[json.loads((d/'result.json').read_text()) for d in dirs]
    with np.load(dirs[0]/'input0.npz') as a,np.load(dirs[1]/'input0.npz') as b:initial={k:bool(np.array_equal(a[k],b[k])) for k in a.files}
    with np.load(dirs[0]/'arrays.npz') as a,np.load(dirs[1]/'arrays.npz') as b:
        first=next(i for i in range(min(map(len,logs))) if not np.array_equal(a[f'actions_{i}'],b[f'actions_{i}']))
        comparisons={str(i):dict(raw_exact=bool(np.array_equal(a[f'raw_{i}'],b[f'raw_{i}'])),decoded_exact=bool(np.array_equal(a[f'actions_{i}'],b[f'actions_{i}'])),raw_max_abs=float(np.max(np.abs(a[f'raw_{i}']-b[f'raw_{i}']))),decoded_max_abs=float(np.max(np.abs(a[f'actions_{i}']-b[f'actions_{i}'])))) for i in range(first+1)}
    first_verify=next(i for i,r in enumerate(logs[0]) if 'distance' in r['route']);x,y=logs[0][first],logs[1][first]
    assert first==1 and x['step']==y['step']==10 and first_verify==1
    assert all(initial.values()) and comparisons['0']['raw_exact'] and comparisons['0']['decoded_exact']
    assert tele[0][:11]==tele[1][:11] and x['input_hash']!=y['input_hash']
    assert x['seed']==y['seed'] and x['noise_hash']==y['noise_hash']
    assert identities[0]['protocol']['server_pid']==identities[1]['protocol']['server_pid']==109338
    assert all(i['engineering_binding']['model_gpu']==i['engineering_binding']['render_gpu']==0 for i in identities)
    data=dict(time=time.time(),case=dict(suite='libero_goal',task=3,init=6),conditions=ids,same_physical_model_render_gpu=0,same_model_service_pid=109338,
        first_proposal_difference_replan=first,step=10,initial_all_arrays_exact=initial,telemetry_step0_through10_exact=True,proposal_comparisons=comparisons,
        request0=[dict(input_hash=r[0]['input_hash'],seed=r[0]['seed'],noise_hash=r[0]['noise_hash'],path=r[0]['route']['path']) for r in logs],
        request1=[dict(input_hash=r['input_hash'],seed=r['seed'],noise_hash=r['noise_hash'],path=r['route']['path'],accepted_length=r['route'].get('accepted_length')) for r in (x,y)],
        first_verified_draft_is_generated_before_first_verify=True,verification_calls_before_first_divergent_draft=0,
        service_history='Same persistent confirmation GPU0 model service: tau.8 first condition with native probe; then Full300 shard, pure300 shard, periodic300 shard. Different rollout/environment processes; fixed case seeds and per-case policy configure/reset.',
        explicit_RNG_path='Xiaomi draft sampling draws torch noise inside seeded(torch,seed); verifier creates its noise from local np.random.default_rng(verification_seed), with no explicit torch random draw. This source inspection is not a snapshot of all hidden model/environment state.',
        interpretation='The current episode already has a different complete observation hash before its FIRST verification. Its first draft differs before verify and is accepted in full. This rules out the current-episode acceptance decision as the initial divergence; it does not prove universal absence of verifier/model side effects. Which unsaved observation component differs, and why, remain unknown; do not attribute this single failure to verifier.',
        results=[dict(success=r['success'],actions=r['actions'],requests=r['predictions']) for r in results],
        sources=[dict(directory=str(d),sha256={n:hashlib.sha256((d/n).read_bytes()).hexdigest() for n in ['input0.npz','identity.json','result.json','arrays.npz','infer.jsonl','telemetry.jsonl']}) for d in dirs],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),RNG_source_sha256={str(Path(__file__).resolve().parents[2]/p):hashlib.sha256((Path(__file__).resolve().parents[2]/p).read_bytes()).hexdigest() for p in ['preparations/idea1_v3_3/backends.py','preparations/idea1_v2_2/runtime_models.py']})
    atomic(OUT/'E4_single_case_difference_receipt.json',data)
    groups=records();scores={cid:summary(groups[cid]) for cid in ['E4_tau0.8','E4_periodic','xiaomi_full_d2','xiaomi_pure_d2']}
    atomic(OUT/'E4_causal_interpretation_receipt.json',dict(time=time.time(),scores=scores,tau08_vs_Full=pairing(groups['E4_tau0.8'],groups['xiaomi_full_d2']),
        interpretation='First80 action equivalence identifies no filtering contribution in that batch. Across300, tau.8 and Full both293, periodic294, pure282; tau.8 vs Full exchanges5 recoveries/5 new failures. Do not uniquely attribute the+11 vs pure to P2: Full alone reaches the same total, hypotheses were selected on first80, comparisons are unadjusted, and one cross-policy discrepancy precedes the first verify. Evidence supports no demonstrated incremental verifier quality benefit, not a universal causal theorem.',
        supersedes='Any earlier wording uniquely attributing quality gains to P2/hybrid capacity, including the interpretation in E4_tau08_periodic_exact_receipt.json; its first80 exact-action facts remain valid.',
        single_case_receipt_sha256=hashlib.sha256((OUT/'E4_single_case_difference_receipt.json').read_bytes()).hexdigest()))
    return data


if __name__=='__main__':run()
