"""Derive module cancellation from saved residual RMS without inventing tensors."""
import argparse
import numpy as np
import pandas as pd

from scripts.experiment10.common import RESULTS, SUITES


def cancellation(a, f, b):
    a, f, b = (np.asarray(v, dtype=np.float64) for v in (a, f, b))
    if any(np.any(~np.isfinite(v)) or np.any(v < 0) for v in (a, f, b)):
        raise ValueError('Residual RMS must be finite and nonnegative')
    denominator = 2*a*f
    cosine = np.full(np.broadcast_shapes(a.shape, f.shape, b.shape), np.nan)
    np.divide(b*b-a*a-f*f, denominator, out=cosine, where=denominator > 0)
    if np.any(np.abs(cosine[np.isfinite(cosine)]) > 1.0001):
        raise RuntimeError('Residual norm identity inconsistent; inspect capture semantics')
    ratio = np.full_like(cosine, np.nan)
    np.divide(b, a+f, out=ratio, where=a+f > 0)
    return np.clip(cosine, -1, 1), ratio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    source = RESULTS / 'probe' / args.suite / 'activation_statistics.csv'
    data = pd.read_csv(source)
    keys = ['sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase',
            'split', 'layer', 'modality']
    tensors = ['H_in', 'H_attn', 'delta_attn', 'delta_ffn', 'delta_block']
    selected = data[data.tensor.isin(tensors)]
    if selected.duplicated(keys + ['tensor']).any():
        raise RuntimeError('Duplicate numerical records')
    table = selected.pivot(index=keys, columns='tensor', values='rms')
    missing = set(tensors) - set(table.columns)
    if missing:
        raise RuntimeError('Missing tensor names: ' + repr(missing))
    a, f, b = table.delta_attn, table.delta_ffn, table.delta_block
    out = table.reset_index()[keys].copy()
    # Equal-sized residual tensors: ||a+f||²=||a||²+||f||²+2<a,f>.
    # RMS squared has the common element-count factor cancelled.
    cosine, ratio = cancellation(a, f, b)
    out['attn_ffn_cosine_derived'] = cosine
    out['cancellation_ratio'] = ratio
    out['relative_attn_update'] = (a / table.H_in.where(table.H_in > 0)).to_numpy()
    out['relative_ffn_update'] = (f / table.H_attn.where(table.H_attn > 0)).to_numpy()
    out['method'] = 'FP32 residual RMS identity; undefined zero denominators'
    destination = RESULTS / 'numerical_updates' / args.suite
    destination.mkdir(parents=True, exist_ok=True)
    out.to_csv(destination / 'module_cancellation.csv', index=False)
    print(str(destination), len(out), flush=True)


if __name__ == '__main__':
    main()
