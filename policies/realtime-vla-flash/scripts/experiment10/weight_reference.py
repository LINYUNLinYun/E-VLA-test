"""CPU-only, fixed-checkpoint VLM matrix numerical reference (not sensitivity)."""
import csv
import json
import pickle

import numpy as np
import torch

from scripts.experiment10.common import CONVERTED, RESULTS, sha256, dump_json


def statistics(value):
    x = value.detach().cpu().float().numpy().reshape(-1)
    if not np.isfinite(x).all():
        raise RuntimeError('Nonfinite checkpoint weights')
    absolute = np.abs(x)
    p = np.percentile(absolute, [50, 90, 99])
    return {'shape': json.dumps(list(value.shape)), 'dtype': str(value.dtype),
            'parameters': x.size, 'mean': float(np.mean(x, dtype=np.float64)),
            'rms': float(np.sqrt(np.mean(x*x, dtype=np.float64))),
            'abs_p50': float(p[0]), 'abs_p90': float(p[1]), 'abs_p99': float(p[2]),
            'max_abs': float(absolute.max()), 'zero_fraction': float(np.mean(x == 0))}


def main():
    torch.set_num_threads(2)
    path = CONVERTED / 'base/base_weights.pkl'
    before = sha256(path)
    with path.open('rb') as stream:
        weights = pickle.load(stream)
    keys = ['encoder_attn_qkv_w', 'encoder_attn_o_w', 'encoder_ffn_gate_w',
            'encoder_ffn_up_w', 'encoder_ffn_down_w']
    rows = []
    for key in keys:
        matrix = weights[key]
        if matrix.ndim != 3 or matrix.shape[0] != 18:
            raise RuntimeError('Unexpected converted matrix layout: ' + key)
        for layer in range(18):
            rows.append({'layer': layer+1, 'matrix': key,
                         'layout': 'converted fused QKV' if 'qkv' in key else 'converted projection',
                         **statistics(matrix[layer])})
        print('Completed ' + key, flush=True)
    if sha256(path) != before:
        raise RuntimeError('Checkpoint changed during analysis')
    output = RESULTS / 'weight_reference'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'weight_reference.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dump_json(output / 'manifest.json', {'complete': True, 'checkpoint_sha256': before,
              'matrices': len(rows), 'interpretation': 'fixed weight description, not stage sensitivity',
              'qkv_mapping': 'fused matrix reported as stored; no unverified Q/K/V split'})


if __name__ == '__main__':
    main()
