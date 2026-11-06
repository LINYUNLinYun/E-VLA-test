#!/usr/bin/env python3
"""Validate complete submodule probes and capture experiment provenance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--expected-count',type=int,required=True)
    parser.add_argument('--expected-noises-per-observation',type=int,default=1)
    args=parser.parse_args()
    if len(args.inputs)!=args.expected_count:
        raise ValueError(f'Expected {args.expected_count} probes, got {len(args.inputs)}')
    records=[]
    identities=set()
    for path in args.inputs:
        with np.load(path,allow_pickle=False) as z:
            m=json.loads(z['metadata_json'].item())
            observation_identity=m['observation_sha256']
            identity=(observation_identity,m['noise_sha256'])
            if identity in identities:
                raise ValueError(f'Duplicate observation/noise pair: {path}')
            identities.add(identity)
            if not np.array_equal(z['layer_ids'],np.arange(1,18)):
                raise ValueError(f'Layer IDs invalid: {path}')
            for key in z.files:
                if key!='metadata_json' and not np.isfinite(z[key]).all():
                    raise ValueError(f'Nonfinite array {key}: {path}')
            if hashlib.sha256(z['noise'].tobytes()).hexdigest()!=m['noise_sha256']:
                raise ValueError(f'Noise hash mismatch: {path}')
            if m['repeated_baseline_max_abs_error']!=0 or m['ordinary_baseline_max_abs_error']>1e-4:
                raise ValueError(f'Baseline validation failed: {path}')
            if (m['denoise_steps'],m['vlm_layer_pattern'],m['action_layer_pattern'])!=(10,'vlm_full18','full18'):
                raise ValueError(f'Unexpected configuration: {path}')
            records.append({'path':str(path.resolve()),'sample_id':m['sample_id'],'phase':m['semantic_phase'],
                'observation_sha256':observation_identity,'noise_sha256':m['noise_sha256'],
                'ordinary_parity_max_abs':m['ordinary_baseline_max_abs_error'],
                'repeated_parity_max_abs':m['repeated_baseline_max_abs_error'],
                'hidden_shape':list(z['hidden_inputs'].shape),'bytes':path.stat().st_size})
    coverage={r['observation_sha256']:set() for r in records}
    for record in records:
        coverage[record['observation_sha256']].add(record['noise_sha256'])
    if any(len(noises)!=args.expected_noises_per_observation for noises in coverage.values()):
        raise ValueError('Incomplete noise coverage for one or more observations')
    root=Path(__file__).resolve().parents[2]
    code_files=['scripts/spec/triton/pi0_infer.py','scripts/spec/triton/triton_pi0_runtime.py',
        'scripts/experiment8/collect_submodules.py','scripts/experiment8/analyze_submodules.py']
    report={'status':'validated','probe_count':len(records),'task_count':len({r['sample_id'] for r in records}),
        'observation_count':len(coverage),'noises_per_observation':args.expected_noises_per_observation,
        'main_forward_count':35*len(records),'parity_forward_count':2*len(records),
        'code_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in code_files},'probes':records}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('code_sha256','probes')},indent=2))


if __name__=='__main__':
    main()
