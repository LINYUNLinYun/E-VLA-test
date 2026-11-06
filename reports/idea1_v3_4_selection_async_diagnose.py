"""Current-observation Full-D10 truth, used offline after the policy has ended."""
import argparse,json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v3_4_selection_async.runtime import setup,OUT
from preparations.idea1_v3_4_selection_async.common import write_csv,ck
from reports.idea1_v3_4_selection_async_analysis import directory
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture,prepare_patch


def normalized_error(a,b):return np.linalg.norm(a[:5,:6]-b[:5,:6],axis=-1)/np.sqrt(6.)


def load(row):
    path=Path(row['input_path'])
    from preparations.idea1_v3_4_selection_cost.common import sha
    assert sha(path)==row['input_file_sha256']
    return load_observation(dict(path=str(path),seed=row['seed'],sha256=row['input_file_sha256'],input_hash=row['input_hash']))


def run(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);backend=setup(out)
    m=json.loads((OUT/'closed_loop_manifest.json').read_text())
    m['conditions']=m['conditions']+[json.loads((OUT/'optimized_manifest.json').read_text())['condition']]
    selected=json.loads((OUT/'followup_manifest.json').read_text())['diagnostic_conditions']
    results=[];saved={};truths={};tick=time.perf_counter()
    # First-Fast observations are shared by all variants. Reuse truth only after
    # checking input/noise identity, never after trajectories have diverged.
    for case in m['cases']:
        for cfg in m['conditions']:
            if cfg['mode']=='full' or cfg['id']=='v12_p3_q4_serial_sync':continue
            d=directory(cfg['id'],case)
            if not (d/'complete.json').exists():continue
            rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
            row=rows[1] if len(rows)>1 else None
            if row is None or not row['route']['draft_attempted']:continue
            key=(row['input_hash'],row['seed'])
            if key not in truths:
                obs,prompt=load(row);c,_=backend.prepare(obs,prompt,());_,raw,_=backend.sample(c,row['seed'],10)
                gold=raw.float().cpu().numpy().copy();_,raw2,_=backend.sample(c,row['seed'],2)
                truths[key]=(gold,raw2.float().cpu().numpy().copy())
            gold,full_d2=truths[key]
            with np.load(d/'arrays.npz') as a:
                draft=a['draft_future_request1'];hats=a['reconstructed_future_request1']
            e=normalized_error(draft,gold);v=np.linalg.norm(hats[:,:,:6]-draft[None,:,:6],axis=-1)/np.sqrt(6.)
            length=int(row['route']['accepted_length']);exec_length=length if row['route']['path']=='fast_verified' else 0
            signed=(draft[:5,6]<0)!=(gold[:5,6]<0)
            results.append(dict(condition=cfg['id'],case=ck(case),request=1,scope='shared_first_fast',
                accepted_actions=exec_length,true_prefix=int(np.cumprod(e<=.10).sum()),
                max_true_error=float(e.max()),mean_true_error=float(e.mean()),
                max_verifier_error=float(v.max()),mean_d2_only_error=float(normalized_error(full_d2,gold).mean()),
                accepted_bad_actions=int(np.sum(e[:exec_length]>.10)),
                accepted_gripper_disagreements=int(np.sum(signed[:exec_length])),
                extrapolation_ratio=row['route']['draft_extrapolation_ratio']))
    # Fixed early-bank requests of selected candidates include subsequent Fulls,
    # ages and extrapolation ratios. Explicitly reconstruct the historical
    # residuals; that reconstruction must match each stored draft.
    for condition in selected:
        for case in m['cases']:
            if case['task'] not in (0,2):continue
            d=directory(condition,case);rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
            history=[]
            with np.load(d/'arrays.npz') as a:
                for row in rows[:15]:
                    if 'input_path' not in row:continue
                    obs,prompt=load(row);route=row['route'];index=row['replan']
                    if route['draft_attempted']:
                        c,_=backend.prepare(obs,prompt,());_,raw,_=backend.sample(c,row['seed'],10)
                        gold=raw.float().cpu().numpy().copy();_,d2,_=backend.sample(c,row['seed'],2)
                        d2=d2.float().cpu().numpy().copy()
                        skip=route['draft_skip'];c0,_=backend.prepare(obs,prompt,skip);_,r0,_=backend.sample(c0,row['seed'],2)
                        uncomp=r0.float().cpu().numpy().copy()
                        last=history[-1];older=history[-2] if len(history)>1 else None
                        ratio=route['draft_extrapolation_ratio']
                        cp,_=prepare_patch(backend,obs,prompt,skip,last['residuals'],
                            None if older is None else older['residuals'],ratio)
                        _,rp,_=backend.sample(cp,row['seed'],2);replay=rp.float().cpu().numpy().copy()
                        draft=a[f'draft_future_request{index}'];hats=a[f'reconstructed_future_request{index}']
                        replay_passed=bool(np.allclose(replay[:5],draft,atol=1e-5,rtol=1e-5))
                        error=normalized_error(draft,gold);length=int(route['accepted_length'])
                        accepted=length if route['path']=='fast_verified' else 0
                        signed=(draft[:5,6]<0)!=(gold[:5,6]<0)
                        record=dict(condition=condition,case=ck(case),request=index,scope='fixed_early_bank',
                            replay_passed=replay_passed,replay_max_abs=float(np.max(np.abs(replay[:5]-draft))),
                            accepted_actions=accepted,true_prefix=int(np.cumprod(error<=.10).sum()),
                            max_true_error=float(error.max()),mean_true_error=float(error.mean()),
                            max_verifier_error=float((np.linalg.norm(hats[:,:,:6]-draft[None,:,:6],axis=-1)/np.sqrt(6.)).max()),
                            mean_d2_only_error=float(normalized_error(d2,gold).mean()),
                            mean_uncompensated_error=float(normalized_error(uncomp,gold).mean()),
                            accepted_bad_actions=int(np.sum(error[:accepted]>.10)),
                            accepted_gripper_disagreements=int(np.sum(signed[:accepted])),
                            extrapolation_ratio=ratio,residual_age=route['draft_residual_age'])
                        results.append(record)
                        if not replay_passed:
                            saved[f'{condition}_{ck(case)}_{index}_saved']=draft
                            saved[f'{condition}_{ck(case)}_{index}_replay']=replay[:5]
                    if route['path']=='full':
                        c,_=prepare_full(backend,obs,prompt,trace=True)
                        residuals,_=backend.adapter.measure(lambda:capture(c.pop('hidden_trace')))
                        history=(history+[dict(step=row['step'],residuals=residuals)])[-2:]
            write_csv(out/'requests.csv',results)
    np.savez(out/'replay_mismatches.npz',**saved)
    summary=[]
    for condition in sorted({r['condition'] for r in results}):
        for scope in ('shared_first_fast','fixed_early_bank'):
            group=[r for r in results if r['condition']==condition and r['scope']==scope]
            if not group:continue
            summary.append(dict(condition=condition,scope=scope,n=len(group),
                mean_true_error=float(np.mean([r['mean_true_error'] for r in group])),
                mean_d2_only_error=float(np.mean([r['mean_d2_only_error'] for r in group])),
                mean_uncompensated_error=float(np.mean([r['mean_uncompensated_error'] for r in group])) if scope=='fixed_early_bank' else None,
                accepted_actions=sum(r['accepted_actions'] for r in group),
                accepted_bad_actions=sum(r['accepted_bad_actions'] for r in group),
                accepted_gripper_disagreements=sum(r['accepted_gripper_disagreements'] for r in group),
                replay_mismatches=sum(not r.get('replay_passed',True) for r in group)))
    write_csv(out/'summary.csv',summary);atomic(out/'complete.json',dict(requests=len(results),
        summary=summary,wall_s=time.perf_counter()-tick,time=time.time(),
        note='Offline action disagreement is diagnostic evidence, not a causal explanation of each episode failure.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
