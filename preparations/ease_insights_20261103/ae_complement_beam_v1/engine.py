"""Reuse native full-prefix preparation and the frozen observed AE dispatcher."""
import hashlib
import numpy as np
from .common import MASKS, require, load_observation
from .metrics import errors


def noise_hash(seed):
    value = np.random.default_rng(seed).standard_normal((50, 32), dtype=np.float32)
    return hashlib.sha256(value.tobytes()).hexdigest()


def sample(backend, context, seed, skip):
    from preparations.ease_insights_20261103.module_tradeoff_v1.ae_dispatch import sample_with_dispatch
    result, receipt = sample_with_dispatch(backend.adapter, backend.ae_dispatch, context, seed, 2, skip)
    actions, raw, noise, nodes, ae_ms, decode_ms = result
    require(nodes['actual_NFE'] == 2 and hashlib.sha256(noise.tobytes()).hexdigest() == noise_hash(seed), 'Actual NFE or generated noise differs')
    require(np.asarray(actions).shape == (50, 7) and np.isfinite(actions).all(), 'Invalid decoded action array')
    raw = raw.float().cpu().numpy()
    require(raw.shape == (50, 32) and np.isfinite(raw).all(), 'Invalid raw action array')
    return np.asarray(actions, dtype=np.float32), raw, receipt, float(ae_ms), float(decode_ms)


def gate(backend, manifest, output):
    from pathlib import Path
    from .common import SUITES, atomic_json
    checks, saved = [], {}
    seeds = manifest['noise_seeds']
    for suite in SUITES:
        row = next(r for r in manifest['observations'] if r['suite'] == suite and r['task'] == 0 and r['phase'] == 'initial')
        context, _ = backend.adapter.prepare(load_observation(row), row['instruction'], ())
        native = backend.adapter.sample(context, seeds[0], dict(kind='native', m=None, num_steps=2), False)
        actual = sample(backend, context, seeds[0], ())
        for name, a, b in [('decoded', native[0], actual[0]), ('raw', native[1].float().cpu().numpy(), actual[1])]:
            a, b = np.asarray(a), np.asarray(b)
            checks.append(dict(suite=suite, kind='unmodified_adapter_D2_no_bypass_' + name,
                               max_abs=float(np.abs(a - b).max()), passed=bool(np.allclose(a, b, atol=1e-5, rtol=1e-5))))
        saved[suite + '_dense_D2_decoded'] = actual[0]
        saved[suite + '_dense_D2_raw'] = actual[1]
        for mask in MASKS:
            value = sample(backend, context, seeds[0], tuple(mask))
            receipt = value[2]
            passed = (receipt['actual_layer_calls'] == 32 and receipt['bypassed_layer_calls'] == 4 and
                      receipt['executed_layers_per_step'] == [i for i in range(1, 19) if i not in mask] and
                      receipt['bypassed_layers_per_step'] == mask and receipt['original_helper_restored'])
            checks.append(dict(suite=suite, kind='ordered_visits_finite_raw_and_decoded', skip=mask, passed=bool(passed)))
            if mask == [2, 8]:
                saved[suite + '_skip02_08_D2_decoded'] = value[0]
                saved[suite + '_skip02_08_D2_raw'] = value[1]
        reset = sample(backend, context, seeds[0], ())
        checks.append(dict(suite=suite, kind='dense_after_all_masks', passed=bool(np.array_equal(reset[0], actual[0]) and np.array_equal(reset[1], actual[1]))))
    np.savez_compressed(Path(output) / 'gate_actions.npz', **saved)
    value = dict(passed=all(c['passed'] for c in checks), checks=checks, manifest_hash=manifest['manifest_hash'],
                 atol=1e-5, rtol=1e-5, physical_episodes=0,
                 limitation='Native dense numerical parity plus actual ordered bypass visits and finiteness. Masked actions are approximations; no masked quality equivalence or rollout success claim.')
    atomic_json(Path(output) / 'gate.json', value)
    require(value['passed'], 'Joint-mask numerical or dispatch gate failed')
    return value


def replay(backend, manifest, index):
    require(type(index) is int and 0 <= index < len(manifest['observations']), 'Invalid snapshot index')
    row = manifest['observations'][index]
    context, prefix_timing = backend.adapter.prepare(load_observation(row), row['instruction'], ())
    masks = [[], *MASKS]
    actions = np.empty((3, 154, 50, 7), dtype=np.float32)
    raw = np.empty((3, 154, 50, 32), dtype=np.float32)
    ae_ms, decode_ms = np.empty((3, 154)), np.empty((3, 154))
    receipts = []
    for si, seed in enumerate(manifest['noise_seeds']):
        for mi, mask in enumerate(masks):
            a, r, receipt, ae, decode = sample(backend, context, seed, tuple(mask))
            actions[si, mi], raw[si, mi], ae_ms[si, mi], decode_ms[si, mi] = a, r, ae, decode
            if si == 0:
                receipts.append(receipt)
            else:
                require(receipts[mi] == receipt, 'Execution identities changed across repeated seeds')
    metrics = errors(actions)
    return dict(manifest_hash=manifest['manifest_hash'], index=index, sample_id=row['sample_id'],
                input_sha256=row['sha256'], noise_seeds=manifest['noise_seeds'], generated_noise_fp32_sha256=[noise_hash(s) for s in manifest['noise_seeds']],
                masks=masks, execution_receipts=receipts, prefix_timing=prefix_timing, physical_episodes=0,
                arrays=dict(actions=actions, raw=raw, ae_ms=ae_ms, decode_ms=decode_ms, **metrics))
