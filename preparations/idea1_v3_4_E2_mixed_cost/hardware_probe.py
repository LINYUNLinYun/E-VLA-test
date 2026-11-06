"""Fresh RTX 4090 native A/F/AF cost measurements; frozen sources are reused.

Default schedule is exactly 4976 complete native VLM+D2 calls, bounded at 5000.
The historical B4/B6 millisecond anchors are copied with their source hash;
all per-module costs in this table are freshly measured on the selected 4090.
Kernel profiles are diagnostics. Uninstrumented, balanced-order whole-VLM
contrasts determine usable net savings, including actual compensation kernels.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparations.idea1_v3.io import atomic

OUT = ROOT / 'results/idea1_v3_4_E2_mixed_cost/hardware_probe'
INPUT_MANIFEST = ROOT / 'results/idea1_v3_4_E2_comp_error_cost/manifest.json'
OLD_COST_TABLE = ROOT / 'results/idea1_v3_4_E2_module_skip/cost_table.json'
LAYERS = tuple(range(1, 18))
KINDS = ('A', 'F', 'AF')
ROUTES = ('keep', 'A0', 'A+', 'F0', 'F+', 'AF0', 'AF+')
REPEATS = 4
COMMON_WARMUPS = 20
ROUTE_WARMUPS = 1
MAX_REQUESTS = 5000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def planned_requests(samples=8):
    return samples * (3 + COMMON_WARMUPS + 4 + len(LAYERS) * len(ROUTES) * (ROUTE_WARMUPS + REPEATS))


def route_order(sample, layer, repeat):
    """Two forward and two reverse sweeps per fixed input/layer."""
    return ROUTES if (sample + layer + repeat) % 2 == 0 else tuple(reversed(ROUTES))


def route_kwargs(route, layers, history=None, older=None):
    if route == 'keep':
        return {}
    if route not in ROUTES:
        raise ValueError(f'Unknown route {route}')
    kind = route[:-1]
    result = dict(skip_attn=tuple(layers) if kind in ('A', 'AF') else (),
                  skip_ffn=tuple(layers) if kind in ('F', 'AF') else ())
    if route.endswith('+'):
        if history is None or older is None:
            raise ValueError('Compensated cost measurement requires both FP32 histories')
        result.update(history=history, older=older, ratio=.8)
    return result


def gpu_snapshot(gpu):
    props = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader,nounits'], text=True).strip()
    rows = list(csv.reader([props]))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError('Could not identify the selected GPU')
    name, uuid, memory, driver = [v.strip() for v in rows[0]]
    apps = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True).strip()
    snapshot = dict(index=gpu, name=name, uuid=uuid, memory_total_mib=float(memory),
                    driver_version=driver, compute_apps=apps.splitlines() if apps else [])
    if '4090' not in name:
        raise RuntimeError(f'This independent probe requires an RTX 4090, found {name}')
    if apps:
        raise RuntimeError(f'Selected GPU is occupied; refusing to start: {apps}')
    return snapshot


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        if self.path.exists() and self.path.stat().st_size:
            raise FileExistsError('A prior bounded probe already consumed requests; use a deliberate new revision')
        self.count = 0
        self.stream = self.path.open('a', buffering=1)

    def consume(self, **identity):
        if self.count >= MAX_REQUESTS:
            raise RuntimeError('Hardware probe reached its frozen 5000-request limit')
        self.count += 1
        row = dict(request=self.count, time=time.time(), **identity)
        self.stream.write(json.dumps(row) + '\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return self.count

    def close(self):
        self.stream.close()


def median_mad(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Expected nonempty finite timing observations')
    med = float(np.median(values))
    return med, float(np.median(np.abs(values - med)))


def aggregate_costs(pairs, profiles):
    """Pure aggregation; negative/noisy readings remain in the output."""
    layers = []
    for layer in LAYERS:
        row = dict(layer=layer)
        for kind, label in (('A', 'attn'), ('F', 'ffn'), ('AF', 'af')):
            items = [r for r in pairs if r['layer'] == layer and r['kind'] == kind]
            for source, target in (('net_saved_ms', 'saved_ms'), ('gross_saved_ms', 'gross_saved_ms'),
                                   ('patch_overhead_ms', 'patch_overhead_ms'),
                                   ('request_net_saved_ms', 'request_saved_ms')):
                med, mad = median_mad([r[source] for r in items])
                row[f'{label}_{target}'] = med
                row[f'{label}_{target}_mad_ms'] = mad
            row[f'{label}_noise_ms'] = row[f'{label}_saved_ms_mad_ms']
            row[f'{label}_eligible'] = row[f'{label}_saved_ms'] > max(0., row[f'{label}_noise_ms'])
        row['measurement_noise_ms'] = row['ffn_noise_ms']
        combined = [r['af_nonadditivity_ms'] for r in pairs if r['layer'] == layer and r['kind'] == 'AF']
        row['af_nonadditivity_ms'], row['af_nonadditivity_mad_ms'] = median_mad(combined)
        stage_values = defaultdict(list)
        for sample in profiles:
            def stage(route, name):
                values = [r['ms'] for r in sample['routes'][route] if r['layer'] == layer and r['stage'] == name]
                if len(values) != 1:
                    raise ValueError(f'Missing or duplicate profile {route}/{name}/L{layer}')
                return values[0]
            qkv, update, ffn = stage('keep', 'qkv'), stage('keep', 'attn'), stage('keep', 'ffn')
            kv = stage('A+', 'kv')
            patch_a, patch_f = stage('A+', 'patch_attn'), stage('F+', 'patch_ffn')
            stage_values['qkv_ms'].append(qkv)
            stage_values['kv_retained_ms'].append(kv)
            stage_values['attn_update_and_o_ms'].append(update)
            stage_values['attention_complete_ms'].append(qkv + update)
            stage_values['attention_removable_before_patch_ms'].append(qkv + update - kv)
            stage_values['ffn_complete_ms'].append(ffn)
            stage_values['attn_patch_kernel_ms'].append(patch_a)
            stage_values['ffn_patch_kernel_ms'].append(patch_f)
            stage_values['af_patch_kernel_ms'].append(stage('AF+', 'patch_attn') + stage('AF+', 'patch_ffn'))
            stage_values['af_profile_net_saved_ms'].append(qkv + update + ffn - stage('AF+', 'kv')
                - stage('AF+', 'patch_attn') - stage('AF+', 'patch_ffn'))
        row['profile'] = {name: dict(zip(('median_ms', 'mad_ms'), median_mad(values))) for name, values in stage_values.items()}
        layers.append(row)
    distribution = {}
    for name in ('attn_saved_ms', 'ffn_saved_ms', 'af_saved_ms', 'attn_patch_overhead_ms', 'ffn_patch_overhead_ms', 'af_nonadditivity_ms'):
        values = np.asarray([row[name] for row in layers])
        distribution[name] = dict(median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()),
            p25=float(np.quantile(values, .25)), p75=float(np.quantile(values, .75)), values=values.tolist())
    attention = [row['profile']['attention_complete_ms']['median_ms'] for row in layers]
    ffn = [row['profile']['ffn_complete_ms']['median_ms'] for row in layers]
    distribution['complete_attention_over_ffn_ratio'] = dict(median=float(np.median(np.asarray(attention) / ffn)),
        per_layer=(np.asarray(attention) / ffn).tolist(), definition='(native RMS/QKV/RoPE + attention + output projection) / native FFN')
    return layers, distribution


def run(gpu, output=OUT, input_manifest=INPUT_MANIFEST, old_cost_table=OLD_COST_TABLE):
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8',
                      TORCHINDUCTOR_COMPILE_THREADS='4', OMP_NUM_THREADS='4')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    from preparations.idea1_v3_1.common import lock
    with lock(output / 'profile.lock', blocking=False):
        if (output / 'manifest.json').exists():
            raise FileExistsError('Probe revision already exists; no automatic retry or extra measurement')
        hardware = gpu_snapshot(gpu)  # Under the independent lock, before CUDA allocation.
        input_manifest, old_cost_table = Path(input_manifest).resolve(), Path(old_cost_table).resolve()
        samples = json.loads(input_manifest.read_text())['samples']
        old = json.loads(old_cost_table.read_text())
        expected = {(suite, task, 14) for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10') for task in (0, 1)}
        if len(samples) != 8 or {(s['suite'], s['task'], s['init']) for s in samples} != expected:
            raise ValueError('The eight fixed real inputs do not match the declared cost bank')
        if planned_requests(len(samples)) > MAX_REQUESTS or old.get('measured') is not True:
            raise ValueError('Invalid frozen schedule or historical anchor table')
        source_names = ('preparations/idea1_v3_4_E2_module_skip/backend.py', 'preparations/idea1_v3/policy.py',
            'preparations/idea1_v3/safe_qk.py', 'preparations/idea1_v2_2/pi0_adapter.py',
            'policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py')
        sources = {name: sha(ROOT / name) for name in source_names}
        sources[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
        manifest = dict(schema_version=1, hardware=hardware, samples=samples,
            input_manifest=str(input_manifest), input_manifest_sha256=sha(input_manifest),
            historical_anchor_table=str(old_cost_table), historical_anchor_sha256=sha(old_cost_table),
            historical_anchors=old['anchors'], anchor_policy='Copy historical B4/B6 measured millisecond anchors; do not retune using new per-layer costs or outcomes.',
            source_files=sources, planned_native_requests=planned_requests(), maximum_native_requests=MAX_REQUESTS,
            routes=list(ROUTES), repeats=REPEATS, common_warmups_per_input=COMMON_WARMUPS,
            route_warmups_per_layer=ROUTE_WARMUPS, hot_order='four seven-route sweeps, two forward/two reverse per fixed input/layer',
            timing='Whole VLM CUDA events from native adapter; complete request boundary CUDA-synchronized; module profile events resolved only after encoder boundary.',
            profile_warning='Diagnostic per-stage event instrumentation is excluded from net-saving measurements.',
            compensation='Native FP32 residual read/extrapolate/add; ratio .8 from previous two real Full banks; no-comp routes omit all patch kernels.',
            numerical_gate='Same-service Full and current true A/F/AF residual reinjection; all 18 current KV, raw and decoded D2 actions at atol=rtol=1e-5.',
            numerical_reference_boundary='No comparison against a historical process raw output; native BF16 cross-process equality is not assumed.',
            reference_bank_warning='Historical step0/15/30 inputs provide shapes and residual tensors for timing, not period-five prediction-quality evidence.',
            env=dict(CUDA_VISIBLE_DEVICES=str(gpu), CUBLAS_WORKSPACE_CONFIG=':4096:8', deterministic=True, TF32=False),
            created=time.time())
        atomic(output / 'manifest.json', manifest)
        atomic(output / 'status.json', dict(status='loading', pid=os.getpid(), hardware=hardware, native_requests=0, time=time.time()))
        import torch
        from preparations.idea1_v3.safe_qk import install
        from preparations.idea1_v3_4_E2_module_skip.backend import Backend
        from preparations.kv_reuse_probe.probe import load_observation
        torch.set_num_threads(4)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        install(str(output / 'qk_receipt.json'))
        tick = time.perf_counter()
        try:
            backend = Backend()
        except Exception:
            atomic(output / 'failure.json', dict(phase='model_load', error=traceback.format_exc(), native_requests=0, time=time.time()))
            atomic(output / 'status.json', dict(status='failed', phase='model_load', native_requests=0, time=time.time()))
            raise
        load_s = time.perf_counter() - tick
        atomic(output / 'process.json', dict(pid=os.getpid(), gpu=gpu, gpu_uuid=hardware['uuid'], model_load_s=load_s, protocol=backend.protocol))
        ledger = Ledger(output / 'requests.jsonl')
        timings = (output / 'request_timings.jsonl').open('a', buffering=1)
        pairs, profiles, checks = [], [], []

        def call(obs, prompt, seed, sample, phase, route='keep', layer=None, **kwargs):
            request = ledger.consume(sample=sample, phase=phase, route=route, layer=layer, seed=int(seed))
            torch.cuda.synchronize()
            start = time.perf_counter()
            c, timing = backend.prepare_modules(obs, prompt, **kwargs)
            actions, raw, cost = backend.sample(c, seed, 2)
            actions, raw = np.asarray(actions).copy(), raw.clone()
            torch.cuda.synchronize()
            result = dict(request=request, **timing, **cost, request_wall_ms=(time.perf_counter() - start) * 1000)
            timings.write(json.dumps(dict(sample=sample, phase=phase, route=route, layer=layer, **result)) + '\n')
            return c, actions, raw, result

        try:
            for sample_index, sample in enumerate(samples):
                banks = []
                current = None
                for side in ('older', 'last', 'current'):
                    obs, prompt = load_observation(sample[side])
                    c, actions, raw, timing = call(obs, prompt, sample[side]['seed'], sample_index, 'capture', capture=True)
                    banks.append(c.pop('residuals'))
                    if side == 'current':
                        current = dict(K=c['K'], V=c['V'], raw=raw, actions=actions)
                older, history, truth = banks
                obs, prompt = load_observation(sample['current'])
                seed = sample['current']['seed']
                atomic(output / 'status.json', dict(status='warming', sample=sample_index, native_requests=ledger.count, time=time.time()))
                for _ in range(COMMON_WARMUPS):
                    call(obs, prompt, seed, sample_index, 'common_warmup')
                # Warm all actual layer/route shapes before any profiled/hot calls.
                for layer in LAYERS:
                    for _ in range(ROUTE_WARMUPS):
                        for route in ROUTES:
                            call(obs, prompt, seed, sample_index, 'route_warmup', route, layer,
                                 **route_kwargs(route, (layer,), history, older))
                sample_profile = dict(sample=sample_index, routes={})
                for route in ('keep', 'A+', 'F+', 'AF+'):
                    # Identical current banks make extrapolation exactly the true
                    # residual while using the warmed EXTRAP=True patch path.
                    c, actions, raw, timing = call(obs, prompt, seed, sample_index, 'profile_and_gate', route,
                        profile=True, **route_kwargs(route, LAYERS, truth, truth))
                    maxima = {name: float((c[name] - current[name]).abs().max().item()) for name in ('K', 'V')}
                    proof = dict(sample=sample_index, route=route,
                        K_equal=bool(torch.allclose(c['K'], current['K'], atol=1e-5, rtol=1e-5)),
                        V_equal=bool(torch.allclose(c['V'], current['V'], atol=1e-5, rtol=1e-5)),
                        raw_equal=bool(torch.allclose(raw, current['raw'], atol=1e-5, rtol=1e-5)),
                        actions_equal=bool(np.allclose(actions, current['actions'], atol=1e-5, rtol=1e-5)),
                        K_max_abs=maxima['K'], V_max_abs=maxima['V'],
                        raw_max_abs=float((raw - current['raw']).abs().max().item()),
                        actions_max_abs=float(np.max(np.abs(actions - current['actions']))))
                    checks.append(proof)
                    if not all(proof[name] for name in ('K_equal', 'V_equal', 'raw_equal', 'actions_equal')):
                        atomic(output / 'numerical_failure.json', dict(proof=proof, native_requests=ledger.count, policy='Stop; no tolerance relaxation or automatic retry'))
                        raise RuntimeError(f'Native route/reinjection gate failed: {proof}')
                    sample_profile['routes'][route] = c['module_meta']['profile_ms']
                profiles.append(sample_profile)
                atomic(output / 'status.json', dict(status='measuring', sample=sample_index, native_requests=ledger.count, time=time.time()))
                sample_pairs = []
                for layer in LAYERS:
                    for repeat in range(REPEATS):
                        order = route_order(sample_index, layer, repeat)
                        values = {}
                        for route in order:
                            _, _, _, values[route] = call(obs, prompt, seed, sample_index, 'hot', route, layer,
                                **route_kwargs(route, (layer,), history, older))
                        savings = {kind: values['keep']['vlm_ms'] - values[kind + '+']['vlm_ms'] for kind in KINDS}
                        for kind in KINDS:
                            row = dict(sample=sample_index, layer=layer, repeat=repeat, kind=kind, order=list(order),
                                net_saved_ms=savings[kind],
                                gross_saved_ms=values['keep']['vlm_ms'] - values[kind + '0']['vlm_ms'],
                                patch_overhead_ms=values[kind + '+']['vlm_ms'] - values[kind + '0']['vlm_ms'],
                                request_net_saved_ms=values['keep']['request_wall_ms'] - values[kind + '+']['request_wall_ms'],
                                keep=values['keep'], bare=values[kind + '0'], patched=values[kind + '+'])
                            if kind == 'AF':
                                row['af_nonadditivity_ms'] = savings['AF'] - savings['A'] - savings['F']
                            pairs.append(row)
                            sample_pairs.append(row)
                atomic(output / f'sample_{sample_index:02d}.json', dict(sample=sample, pairs=sample_pairs, profile=sample_profile,
                    numerical_checks=checks[-4:], native_requests=ledger.count))
                atomic(output / 'status.json', dict(status='running', completed_samples=sample_index + 1,
                    total_samples=8, native_requests=ledger.count, time=time.time()))
                print(json.dumps(dict(completed_samples=sample_index + 1, native_requests=ledger.count)), flush=True)
                del banks, older, history, truth, current, c
            if ledger.count != planned_requests():
                raise ValueError(f'Actual request schedule differs from frozen plan: {ledger.count}')
            layer_costs, distribution = aggregate_costs(pairs, profiles)
            table = dict(schema_version=2, measured=True, units='ms', n_layers=17,
                layer_costs=layer_costs, anchors=old['anchors'], hardware=hardware,
                historical_anchor_reference=dict(path=str(old_cost_table), sha256=sha(old_cost_table),
                    original_gpu_uuid=old['provenance'].get('gpu_uuid'), policy='Historical absolute B4/B6 millisecond anchors, not remeasured or retuned here'),
                provenance=dict(gpu_index=gpu, gpu_uuid=hardware['uuid'], protocol=backend.protocol,
                    input_manifest_sha256=sha(input_manifest), source_files=sources, model_load_s=load_s,
                    native_requests=ledger.count, maximum_native_requests=MAX_REQUESTS,
                    aggregation='Median and MAD over eight fixed inputs and four balanced hot sweeps per layer/module',
                    cost='Uninstrumented whole native VLM savings: keep minus actual skip plus FP32 extrapolation; no clipping of negative/noisy values',
                    historical_comparison='Any comparison against old F-E candidates must distinguish the historical cost candidate set from this newly measured 4090 candidate set'),
                layer_distribution=distribution, profiles=profiles, numerical_checks=checks)
            atomic(output / 'cost_table.json', table)
            atomic(output / 'summary.json', dict(status='complete', passed=True, native_requests=ledger.count,
                planned_native_requests=planned_requests(), numerical_checks=len(checks), all_numerical_checks_passed=True,
                hardware=hardware, layer_distribution=distribution, historical_anchor_reference=table['historical_anchor_reference']))
            atomic(output / 'status.json', dict(status='complete', passed=True, native_requests=ledger.count,
                total_samples=8, completed_samples=8, gpu_uuid=hardware['uuid'], time=time.time()))
            return table
        except Exception:
            atomic(output / 'failure.json', dict(error=traceback.format_exc(), native_requests=ledger.count,
                maximum_native_requests=MAX_REQUESTS, time=time.time()))
            atomic(output / 'status.json', dict(status='failed', native_requests=ledger.count, time=time.time()))
            raise
        finally:
            timings.close()
            ledger.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu', type=int, required=True)
    p.add_argument('--output', type=Path, default=OUT)
    p.add_argument('--input-manifest', type=Path, default=INPUT_MANIFEST)
    p.add_argument('--old-cost-table', type=Path, default=OLD_COST_TABLE)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if a.plan_only:
        print(json.dumps(dict(planned_native_requests=planned_requests(), maximum_native_requests=MAX_REQUESTS,
            samples=8, layers=17, routes=list(ROUTES), repeats=REPEATS)))
    else:
        table = run(a.gpu, a.output, a.input_manifest, a.old_cost_table)
        print(json.dumps(dict(status='complete', native_requests=table['provenance']['native_requests'],
            gpu_uuid=table['hardware']['uuid'])), flush=True)


if __name__ == '__main__':
    main()
