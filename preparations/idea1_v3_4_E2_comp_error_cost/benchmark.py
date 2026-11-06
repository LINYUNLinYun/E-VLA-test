"""Frozen, same-input AB/BA GPU cost probe; no new closed-loop experiment.

8 recorded cases (four suites x task00/01, init14), observed steps 0/15/30.
20 paired warmups, 30 paired measurements per case. Both methods do D10 Full
replay and exactly one VJP. Error construction and copying are INCLUDED in B.
GPU unavailability is recorded explicitly; CPU timings are never substituted.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, value):
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def freeze(out):
    samples = []
    for suite in SUITES:
        for task in [0, 1]:
            d = ROOT/'results/idea1_v3_4/full_d10/episodes'/suite/f'task{task:02d}'/'init14'
            rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
            sides = []
            for index in [0, 3, 6]:
                r = rows[index]
                assert r['step'] == index*5
                p = Path(r['input_path'])
                assert sha(p) == r['input_file_sha256']
                sides.append(dict(path=str(p),sha256=sha(p),seed=r['seed'],
                                  input_hash=r['input_hash'],step=r['step'],request=index))
            samples.append(dict(suite=suite,task=task,init=14,older=sides[0],
                                last=sides[1],current=sides[2]))
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT/p for p in [
        'preparations/idea1_v3_4_E2_cycle3_full10_jacobian/model.py',
        'preparations/idea1_v3_4_E2_online_time_weights/selector.py',
        'preparations/idea1_v3_4_E2_E4_single/torch_model.py',
        'preparations/idea1_v3_4_E2_cycle3_extrapolate/backend.py',
        'preparations/idea1_v3_4_E2_cycle3/selection.py',
        'preparations/idea1_v3_4_v12_compensation/backend.py',
        'preparations/idea1_v3/policy.py',
        'policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py']]
    m = dict(samples=samples,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files},
             warmup_pairs=20,measured_pairs=30,method_A='original_residual_direction',
             method_B='observed_forecast_error_direction',model_steps=10,vjp_count=1,
             weights='time [1,2,4,8,16]/31; seven dimensions +1',
             error='R30 - [R15 + (30-15)/(15-0)*(R15-R0)]',
             scope='Recorded Full-trajectory cost/functional probe; not closed-loop success',
             created_at=time.time())
    dump(out/'manifest.json',m)
    return m


def idle_gpu():
    q = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True)
    if q.returncode:
        return None, dict(status='gpu_unavailable',detail=q.stdout+q.stderr)
    a = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                        '--format=csv,noheader'],capture_output=True,text=True)
    if a.returncode:
        return None, dict(status='gpu_unavailable',detail=a.stdout+a.stderr)
    busy = {line.split(',')[0].strip() for line in a.stdout.splitlines() if line.strip()}
    for line in q.stdout.splitlines():
        index, uuid, used = [s.strip() for s in line.split(',')]
        if uuid not in busy and float(used) <= 1500:
            return int(index), dict(status='idle',gpu=int(index),uuid=uuid,
                                    memory_used_mib=float(used))
    return None, dict(status='waiting_gpu',detail='No unoccupied GPU with <=1500 MiB used')


def run(out, m, gpu, identity):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA has no usable devices')
    from preparations.idea1_v3.safe_qk import install
    install(str(out/'qk_receipt.json'))
    from preparations.idea1_v3.policy import TritonBackend
    from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
    from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture
    from preparations.idea1_v3_4_E2_cycle3.common import coefficients,choose
    from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Selector as Original,calculate
    from preparations.kv_reuse_probe.probe import load_observation
    from .model import Selector,forecast_error
    tick = time.perf_counter()
    b = TritonBackend()
    A, B = Original(b), Selector(b)
    setup_s = time.perf_counter()-tick
    coeff = coefficients()
    records, checks, arrays, arithmetic = [], [], {}, []
    for si, sample in enumerate(m['samples']):
        prepare_tick = time.perf_counter()
        banks, ctx = [], None
        prompts = []
        for side in ['older','last','current']:
            obs, prompt = load_observation(sample[side])
            prompts.append(prompt)
            ctx, _ = prepare_full(b,obs,prompt,trace=True)
            banks.append(capture(ctx.pop('hidden_trace')))
        assert len(set(prompts)) == 1
        current, last, older = banks[2], banks[1], banks[0]
        error = forecast_error(current,last,older)
        expected = torch.stack([current[i]-(last[i]+(last[i]-older[i]))
                                for i in range(1,18)])
        assert torch.equal(error,expected)
        del expected
        seed = sample['current']['seed']
        _, grad_A, _, cold_A = A.run(ctx,seed,coeff,4)
        _, grad_B, _, cold_B = B.run(ctx,seed,coeff,error,4)
        assert torch.equal(A.raw,B.raw), 'Neutral error gates changed Full output'
        eager_raw, eager_grad = calculate(B.model,B.item,B.W,B.g)
        assert torch.equal(eager_raw,B.raw)
        assert torch.allclose(eager_grad,B.grad,atol=1e-4,rtol=1e-3)
        eager_grad_max = float((eager_grad-B.grad).abs().max())
        arrays[f'raw_{si}'] = A.raw.float().cpu().numpy()
        arrays[f'original_gradient_{si}'] = grad_A
        arrays[f'error_gradient_{si}'] = grad_B
        checks.append(dict(sample=si,neutral_full_bitwise_equal=True,
                           forecast_formula_bitwise_equal=True,
                           graph_eager_raw_bitwise_equal=True,
                           graph_eager_gradient_max_abs=eager_grad_max,
                           original_capture_ms=cold_A['selector_capture_ms'],
                           error_capture_ms=cold_B['selector_capture_ms'],
                           history_preparation_and_capture_s=time.perf_counter()-prepare_tick,
                           skip4=choose(abs(grad_B),4),skip6=choose(abs(grad_B),6)))
        def old_call():
            return A.run(ctx,seed,coeff,4)
        def new_call():
            forecast_error(current,last,older,out=error)
            return B.run(ctx,seed,coeff,error,4)
        for _ in range(m['warmup_pairs']):
            old_call()
            new_call()
        for repeat in range(m['measured_pairs']):
            values = {}
            for name, fn in ([('A',old_call),('B',new_call)] if repeat%2==0
                             else [('B',new_call),('A',old_call)]):
                torch.cuda.synchronize()
                t = time.perf_counter()
                fn()
                values[name] = (time.perf_counter()-t)*1000
            records.append(dict(sample=si,suite=sample['suite'],task=sample['task'],
                                repeat=repeat,order='AB' if repeat%2==0 else 'BA',
                                original_ms=values['A'],error_ms=values['B'],
                                added_ms=values['B']-values['A']))
        # Separate arithmetic timing; its work is already included in B above.
        for repeat in range(40):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t = time.perf_counter()
            begin.record()
            forecast_error(current,last,older,out=error)
            end.record()
            end.synchronize()
            arithmetic.append(dict(sample=si,repeat=repeat,gpu_ms=begin.elapsed_time(end),
                                   wall_ms=(time.perf_counter()-t)*1000))
        print(json.dumps(dict(completed_samples=si+1,total_samples=len(m['samples']))),flush=True)
        A.clear()
        B.clear()
    for name, rows in [('paired_timings.csv',records),('error_math_timings.csv',arithmetic)]:
        with (out/name).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(out/'probe_arrays.npz',**arrays)
    summary = dict(passed=True,actual_gpu_measurement=True,gpu=identity,
                   model_setup_s=setup_s,samples=len(m['samples']),pairs=len(records),
                   original_median_ms=float(np.median([r['original_ms'] for r in records])),
                   error_median_ms=float(np.median([r['error_ms'] for r in records])),
                   paired_added_median_ms=float(np.median([r['added_ms'] for r in records])),
                   paired_added_mean_ms=float(np.mean([r['added_ms'] for r in records])),
                   error_math_gpu_median_ms=float(np.median([r['gpu_ms'] for r in arithmetic])),
                   error_math_wall_median_ms=float(np.median([r['wall_ms'] for r in arithmetic])),
                   checks=checks,scope=m['scope'])
    dump(out/'summary.json',summary)
    dump(out/'status.json',dict(status='complete',actual_gpu_measurement=True,time=time.time()))
    print(json.dumps(summary),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',default=str(ROOT/'results/idea1_v3_4_E2_comp_error_cost'))
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    m = json.loads((out/'manifest.json').read_text()) if (out/'manifest.json').exists() else freeze(out)
    assert all(sha(ROOT/name)==h for name,h in m['source_hashes'].items())
    if (out/'summary.json').exists():
        raise FileExistsError('Probe already complete; refusing to overwrite')
    gpu, state = idle_gpu()
    if gpu is None:
        dump(out/'status.json',dict(state,actual_gpu_measurement=False,time=time.time()))
        print(json.dumps(state),flush=True)
        return
    try:
        dump(out/'status.json',dict(status='running',gpu=state,time=time.time()))
        run(out,m,gpu,state)
    except Exception as e:
        dump(out/'status.json',dict(status='failed',error=repr(e),actual_gpu_measurement=False,time=time.time()))
        raise


if __name__ == '__main__':
    main()
