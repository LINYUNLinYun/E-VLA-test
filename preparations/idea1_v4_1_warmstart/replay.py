"""Fixed-observation screening; reference agreement is not task success."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v3_3.rules import acceptance
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v2_1.runtime_client import rpc
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture, prepare_patch
from preparations.idea1_v4_0.rules import KalmanPrior
from .common import OUT, read, key, grid
from .schedule import mix
from .sampling import sample_initial, sample_gaussian


def observation(row):
    return load_observation(dict(path=row['input_path'],sha256=row['input_file_sha256'],
        seed=row['seed'],input_hash=row['input_hash']))


def run_case(backend, bank):
    d = Path(bank['directory'])
    out = OUT/'replay'/key(bank['case'])
    if (out/'complete.json').exists():
        return dict(completed=True, reused=True)
    out.mkdir(parents=True,exist_ok=True)
    rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
    by_step = {r['step']:r for r in rows}
    telemetry = [json.loads(s) for s in (d/'telemetry.jsonl').read_text().splitlines()]
    commands = [r for r in telemetry if r['action'] is not None]
    posterior, observed, cache, metrics, saved = KalmanPrior(), 0, {}, [], {}
    source_arrays = np.load(d/'arrays.npz')
    started = time.perf_counter()
    for request in bank['requests']:
        row = rows[request]
        step, seed = row['step'], row['seed']
        while observed < len(commands) and commands[observed]['step'] <= step:
            posterior.observe(commands[observed]['action'])
            observed += 1
        assert posterior.filter.count==step and observed==step
        history = row['route']['draft_history_full_steps']
        for hstep in history:
            if hstep not in cache:
                full_obs, prompt = observation(by_step[hstep])
                context, _ = prepare_full(backend,full_obs,prompt,trace=True)
                residual, _ = backend.adapter.measure(lambda:capture(context.pop('hidden_trace')))
                cache[hstep] = dict(context=context,residuals=residual)
        cache = {h:cache[h] for h in history}
        current_obs, prompt = observation(row)
        reference_context, _ = backend.prepare(current_obs,prompt,())
        full_actions, full_raw, _ = backend.sample(reference_context,seed,10)
        full_np = full_raw.float().cpu().numpy().copy()
        old = cache[history[-1]]
        older = cache[history[-2]] if len(history)>1 else None
        context, _ = prepare_patch(backend,current_obs,prompt,row['route']['draft_skip'],old['residuals'],
            None if older is None else older['residuals'],row['route']['draft_extrapolation_ratio'])
        coordinates = DecoderCoordinates(backend.adapter.backend.output_transform,context['transformed_state'])
        prior_commands = posterior.forecast()
        prior = coordinates.encode_commands(prior_commands,np.zeros((50,32),np.float32))
        previous = rows[request-1]
        last_gripper = float(source_arrays[f'raw_{request-1}'][previous['executed_length']-1,6])
        prefix = f'request{request}'
        saved[prefix+'_reference_raw'] = full_np[:5]
        saved[prefix+'_prior_raw'] = prior[:5]
        base = dict(case=key(bank['case']),request=request,step=step,seed=seed,
            input_hash=row['input_hash'],skip=row['route']['draft_skip'],history=history,kalman_observations=observed)

        def record(name, actions, raw, cost, cfg=None, verify=True):
            raw_np = raw.float().cpu().numpy().copy() if hasattr(raw,'float') else np.asarray(raw).copy()
            delta = np.asarray(actions)[:5]-np.asarray(full_actions)[:5]
            item = dict(base,id=name,nfe=0 if cfg is None else cfg['fast_steps'],
                mse=float(np.mean(delta**2)),spatial_rmse=float(np.sqrt(np.mean(delta[:,:6]**2))),
                xyz_rmse=float(np.sqrt(np.mean(delta[:,:3]**2))),rotation_rmse=float(np.sqrt(np.mean(delta[:,3:6]**2))),
                gripper_mismatch=float(np.mean((raw_np[:5,6]<0)!=(full_np[:5,6]<0))),
                ae_ms=cost.get('ae_ms',0.),nodes=cost.get('nodes'),config=cfg)
            if verify:
                tensor = backend.torch.as_tensor(raw_np,device='cuda',dtype=backend.torch.bfloat16)
                hats, vcost = backend.verify(old['context'],context,tensor,seed,bank['case']['suite'])
                length, details = acceptance(raw_np,hats,.10,5,last_gripper=last_gripper,protect=True)
                item.update(accepted_length=length,verify_gripper=details['verify_gripper'],
                    verify_ae_ms=vcost['verify_ae_ms'])
            metrics.append(item)
            saved[prefix+'_'+name] = raw_np[:5]

        record('raw_kalman',prior_commands,prior,{},verify=True)
        epsilon = np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
        record('raw_noise',coordinates.decode(epsilon),epsilon,{},verify=False)
        for nfe in (1,2):
            a,r,cost = backend.sample(context,seed,nfe)
            record(f'gaussian_d{nfe}',a,r,cost,dict(fast_steps=nfe))
            a,r,cost = sample_initial(backend,context,prior,nfe,1.,time_encoding='legacy')
            record(f'legacy_kf_t1_d{nfe}',a,r,cost,dict(fast_steps=nfe))
            a,r,cost = sample_gaussian(backend,context,seed,nfe)
            record(f'gaussian_time_fixed_d{nfe}',a,r,cost,dict(fast_steps=nfe))
        for candidate in grid():
            tick = time.perf_counter()
            initial = mix(prior,seed,candidate['start_time'],candidate['noise_scale'])
            mix_ms = (time.perf_counter()-tick)*1000
            a,r,cost = sample_initial(backend,context,initial,candidate['fast_steps'],candidate['start_time'])
            record(candidate['id'],a,r,cost,candidate)
            metrics[-1]['mix_ms'] = mix_ms
        print(json.dumps(dict(case=base['case'],request=request,completed_candidates=len(grid()))),flush=True)
    source_arrays.close()
    atomic(out/'metrics.json',metrics)
    np.savez(out/'prefixes.npz',**saved)
    result = dict(valid=True,case=bank['case'],samples=len(bank['requests']),candidates=len(grid()),
        elapsed_s=time.perf_counter()-started,time=time.time())
    atomic(out/'complete.json',result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--worker',required=True)
    args = parser.parse_args()
    for bank in read()['replay_bank']:
        claim = lock(OUT/'replay_claims'/f"{key(bank['case'])}.lock",blocking=False)
        try:
            claim.__enter__()
        except BlockingIOError:
            continue
        try:
            if (OUT/'replay'/key(bank['case'])/'complete.json').exists():
                continue
            atomic(OUT/'workers'/args.worker/'current.json',dict(stage='replay',case=bank['case'],time=time.time()))
            print(rpc(args.port,dict(op='replay_case',bank=bank)),flush=True)
        finally:
            claim.__exit__(None,None,None)


if __name__=='__main__':
    main()
