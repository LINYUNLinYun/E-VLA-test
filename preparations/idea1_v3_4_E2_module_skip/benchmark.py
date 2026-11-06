"""Bounded native, same-input AB/BA cost measurement and module route checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/idea1_v3_4_E2_module_skip'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(gpu, out):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4')
    import numpy as np
    import torch
    from preparations.idea1_v3.io import atomic, write_rows
    from preparations.idea1_v3_1.common import lock
    from preparations.idea1_v3.safe_qk import install
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3.common import coefficients
    from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Selector as BlockSelector
    from .backend import Backend
    from .block_backend import prepare_block
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True, exist_ok=True)
    with lock(out/'benchmark.lock', blocking=False):
        if (out/'cost_table.json').exists():
            raise FileExistsError('Measured costs already committed; use a new revision')
        uuid = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=uuid',
                                        '--format=csv,noheader'], text=True).strip()
        source = ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json'
        samples = json.loads(source.read_text())['samples']
        manifest = dict(stage='A', samples=samples, input_manifest_sha256=sha(source),
                        gpu=gpu, gpu_uuid=uuid, maximum_native_requests=5000,
                        measured_pairs_per_module=4, warmup_pairs_per_module=1,
                        common_warmups_per_shape=20, conditions=['A','F','AF'],
                        timing='native CUDA event VLM time for incremental costs; full request wall time also retained',
                        historical_step_warning='old step0/15/30 bank is for cost and route checks, not R5 forecast quality',
                        source_files={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
                        created_at=time.time())
        atomic(out/'benchmark_manifest.json', manifest)
        atomic(out/'benchmark_process.json', dict(pid=os.getpid(), gpu=gpu, gpu_uuid=uuid, started=time.time()))
        install(str(out/'benchmark_qk_receipt.json'))
        tick = time.perf_counter()
        b = Backend()
        load_s = time.perf_counter()-tick
        requests = 0
        records, gates, profiles, anchors = [], [], [], []

        def call(obs, prompt, seed, kind='none', skip=(), history=None, older=None,
                 capture=False, profile=False, reference=False):
            nonlocal requests
            if requests >= 5000:
                raise RuntimeError('Stage A native request limit reached')
            requests += 1
            torch.cuda.synchronize()
            start = time.perf_counter()
            if kind == 'block':
                c, t = prepare_block(b, obs, prompt, skip=skip, capture=capture,
                                     history=history, older=older, ratio=.8, reference=reference)
            else:
                c, t = b.prepare_modules(obs, prompt,
                    skip_attn=skip if kind in ('A', 'AF') else (),
                    skip_ffn=skip if kind in ('F', 'AF') else (), capture=capture,
                    history=history, older=older, ratio=.8, profile=profile, reference=reference)
            actions, raw, cost = b.sample(c, seed, 2)
            torch.cuda.synchronize()
            return c, np.asarray(actions).copy(), raw.clone(), dict(t, **cost,
                request_wall_ms=(time.perf_counter()-start)*1000)

        for si, sample in enumerate(samples):
            histories, block_histories = [], []
            for side in ('older', 'last', 'current'):
                obs, prompt = load_observation(sample[side])
                c, _, _, _ = call(obs, prompt, sample[side]['seed'], capture=True)
                histories.append(c['residuals'])
                bc, _, _, _ = call(obs, prompt, sample[side]['seed'], kind='block', capture=True)
                block_histories.append(bc['residuals'])
            obs, prompt = load_observation(sample['current'])
            seed = sample['current']['seed']
            history, older = histories[1], histories[0]
            block_history, block_older = block_histories[1], block_histories[0]
            # A full native path is the direct oracle for all-retained modules.
            direct, _ = b.prepare(obs, prompt)
            requests += 1
            da, dr, _ = b.sample(direct, seed, 2)
            c, a, r, t = call(obs, prompt, seed, profile=True)
            assert torch.allclose(dr, r, atol=1e-5, rtol=1e-5)
            assert np.allclose(da, a, atol=1e-5, rtol=1e-5)
            profiles.append(dict(sample=si, seq=c['seq'], profile=c['module_meta'].get('profile_ms', {})))
            # Real current residual injection must restore Full-D2 actions.
            for kind in ('A', 'F', 'AF'):
                _, a, r, _ = call(obs, prompt, seed, kind=kind, skip=(8,), history=histories[2])
                native, na, nr, _ = call(obs, prompt, seed, kind=kind, skip=(8,), history=histories[2], reference=True)
                assert torch.allclose(r, nr, atol=1e-5, rtol=1e-5), ('native reference', si, kind)
                assert np.allclose(a, na, atol=1e-5, rtol=1e-5)
                assert torch.allclose(dr, r, atol=1e-5, rtol=1e-5), ('residual injection', si, kind)
                assert np.allclose(da, a, atol=1e-5, rtol=1e-5)
            gates.append(dict(sample=si, all_retained=True, current_residual_restores_full_d2=True,
                              A_F_AF_native_reference=True, atol=1e-5, rtol=1e-5))
            # Use exactly the original whole-block Jacobian to define fixed-input anchors.
            selector = BlockSelector(b)
            skip4, grad, score, st = selector.run(c, seed, coefficients(), 4)
            from preparations.idea1_v3_4_E2_adjacency_200.common import choose
            skip4, skip6 = choose(score, 4, 'spaced'), choose(score, 6, 'spaced')
            selector.clear()
            del selector
            for _ in range(20):
                call(obs, prompt, seed)
            for budget_id, skip in (('B4', skip4), ('B6', skip6)):
                # Warm the old whole-block branch before timed AB/BA comparisons.
                call(obs, prompt, seed, kind='block', skip=skip, history=block_history, older=block_older)
                for repeat in range(4):
                    values = {}
                    for name in (('keep','skip') if repeat % 2 == 0 else ('skip','keep')):
                        kw = dict(kind='block', skip=skip, history=block_history, older=block_older) if name == 'skip' else {}
                        _, _, _, values[name] = call(obs, prompt, seed, **kw)
                    anchors.append(dict(sample=si, budget=budget_id, mask=skip, repeat=repeat,
                        saved_ms=values['keep']['vlm_ms']-values['skip']['vlm_ms'],
                        fast_ms=values['skip']['request_wall_ms'], keep=values['keep'], skip=values['skip']))
            for kind in ('F', 'A', 'AF'):
                for layer in range(1, 18):
                    # Route-specific warmup then four paired, reversed-order measurements.
                    for _ in range(1):
                        call(obs, prompt, seed)
                        call(obs, prompt, seed, kind=kind, skip=(layer,), history=history, older=older)
                    for repeat in range(4):
                        values = {}
                        for name in (('keep','skip') if repeat % 2 == 0 else ('skip','keep')):
                            kw = dict(kind=kind, skip=(layer,), history=history, older=older) if name == 'skip' else {}
                            _, _, _, values[name] = call(obs, prompt, seed, **kw)
                        records.append(dict(sample=si, seq=c['seq'], module=kind, layer=layer, repeat=repeat,
                            order='AB' if repeat % 2 == 0 else 'BA',
                            saved_ms=values['keep']['vlm_ms']-values['skip']['vlm_ms'],
                            request_saved_ms=values['keep']['request_wall_ms']-values['skip']['request_wall_ms'],
                            keep=values['keep'], skip=values['skip']))
            atomic(out/'benchmark_status.json', dict(status='running', completed_samples=si+1,
                   total_samples=len(samples), native_requests=requests, time=time.time()))
            atomic(out/f'cost_sample_{si:02d}.json', dict(sample=sample, module_pairs=records[-204:],
                                                        anchors=anchors[-8:], gates=gates[-1], profile=profiles[-1]))
            print(json.dumps(dict(completed_samples=si+1, requests=requests)), flush=True)
            del c, direct, histories, block_histories, history, older, block_history, block_older
        layer_costs = []
        for layer in range(1, 18):
            item = dict(layer=layer)
            for kind, label in (('F','ffn'),('A','attn'),('AF','af')):
                vals = [r['saved_ms'] for r in records if r['module']==kind and r['layer']==layer]
                med = float(np.median(vals))
                item[f'{label}_saved_ms'] = med
                item[f'{label}_noise_ms'] = float(np.median(np.abs(np.asarray(vals)-med)))
            item['measurement_noise_ms'] = item['ffn_noise_ms']
            layer_costs.append(item)
        budget_anchors = {}
        for budget_id in ('B4', 'B6'):
            rows = [r for r in anchors if r['budget']==budget_id]
            budget_anchors[budget_id] = dict(saved_ms=float(np.median([r['saved_ms'] for r in rows])),
                fast_ms=float(np.median([r['fast_ms'] for r in rows])),
                per_input=[dict(sample=si, saved_ms=float(np.median([r['saved_ms'] for r in rows if r['sample']==si])),
                                fast_ms=float(np.median([r['fast_ms'] for r in rows if r['sample']==si]))) for si in range(len(samples))])
        table = dict(schema_version=1, measured=True, units='ms', layer_costs=layer_costs,
            anchors=budget_anchors, provenance=dict(input_manifest_sha256=sha(source), gpu_uuid=uuid,
                gpu_index=gpu, protocol=b.protocol, native_requests=requests, maximum_native_requests=5000,
                model_load_s=load_s, cost='AB/BA native VLM elapsed time; skip includes FP32 residual extrapolation patch',
                aggregation='median across eight inputs and four paired repeats; MAD retained as measurement noise',
                source_files=manifest['source_files']), profiles=profiles, numerical_checks=gates)
        write_rows(out/'cost_pairs.jsonl', records)
        write_rows(out/'cost_anchors.jsonl', anchors)
        atomic(out/'cost_table.json', table)
        atomic(out/'benchmark_status.json', dict(status='complete', passed=True, native_requests=requests,
                                                gpu_uuid=uuid, time=time.time()))
        print(json.dumps(dict(status='complete', native_requests=requests, anchors=budget_anchors)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', default=str(OUT))
    a = p.parse_args()
    out = Path(a.output)
    try:
        run(a.gpu, out)
    except Exception:
        import traceback
        from preparations.idea1_v3.io import atomic
        atomic(out/'benchmark_failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    main()
