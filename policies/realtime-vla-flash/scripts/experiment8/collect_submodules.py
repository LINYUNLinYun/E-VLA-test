#!/usr/bin/env python3
"""Fixed-observation VLM Attention/FFN probes, preserving original QKV."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/spec'))
from scripts.experiment6.layer_metrics import make_modality_spans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--base-triton-path', type=Path, required=True)
    parser.add_argument('--prompt-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--dry-run', action='store_true', help='Validate observations/prompt manifest without loading a model or using GPU')
    args = parser.parse_args()
    prompts = json.loads(args.prompt_manifest.read_text())['prompts']
    preflight = []
    for path in args.inputs:
        with np.load(path,allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json'].item()))
            prompt = str(data['prompt'].item())
            if prompt not in prompts:
                raise ValueError(f'Prompt not in selected manifest: {prompt}')
            if path.stem.endswith('step0001__goal_alignment'):
                raise ValueError('Excluded stale phase label; use corrected observations')
            for key in ('agent_image','wrist_image','robot_state'):
                if not np.isfinite(data[key]).all():
                    raise ValueError(f'Nonfinite {key}: {path}')
        preflight.append({'observation':str(path.resolve()),'sample_id':meta['sample_id'],
            'phase':meta.get('semantic_phase'),'observation_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    if len({p.stem for p in args.inputs}) != len(args.inputs):
        raise ValueError('Duplicate observation stems would overwrite results')
    if args.dry_run:
        print(json.dumps({'dry_run':True,'gpu_inference_started':False,'observations':preflight,
            'main_forward_count':35*len(preflight),'parity_forward_count':2*len(preflight)},indent=2))
        return
    import spec_serve_policy
    from scripts.spec.triton import triton_pi0_runtime as runtime
    weights, artifact = spec_serve_policy._resolve_triton_base_artifact(args.base_triton_path)
    config, transform, output_transform = spec_serve_policy._build_policy_components(config_name='pi0_libero', checkpoint_dir=artifact)
    pool = runtime.TritonRuntimePool(base_weights_path=weights, manifest_path=args.prompt_manifest.resolve(),
        num_views=2, chunk_size=int(config.model.action_horizon), vlm_layer_pattern='vlm_full18',
        ablation_layer_pattern='full18', denoise_steps=10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(args.inputs):
        out = args.output_dir / (path.stem + '.submodule_probe.npz')
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment: {out}')
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json'].item()))
            prompt = str(data['prompt'].item())
            transformed = transform({'observation/image':data['agent_image'],
                'observation/wrist_image':data['wrist_image'], 'observation/state':data['robot_state'], 'prompt':prompt})
        identity = hashlib.sha256(path.read_bytes()).digest()
        rng = np.random.default_rng(np.random.SeedSequence([args.seed, int.from_bytes(identity[:4], 'little')]))
        noise = rng.standard_normal((int(config.model.action_horizon),int(config.model.action_dim)),dtype=np.float32)
        images, state, n = runtime.prepare_triton_inputs_from_transformed(transformed=transformed, device='cuda',
            action_horizon=int(config.model.action_horizon),action_dim=int(config.model.action_dim),noise=noise)
        session = pool.start_session(prompt)
        prepared = session.prepare_observation(images=images,state=state)
        traces = []
        baseline = session.run_submodule_probe(prepared=prepared,noise=n,submodule_trace=traces)
        baseline_np = baseline.float().cpu().numpy()
        def policy_actions(raw):
            raw = np.asarray(raw)
            if raw.ndim == 3 and raw.shape[0] == 1:
                raw = raw[0]
            return np.asarray(output_transform({'state':np.asarray(transformed['state']).copy(),
                'actions':raw.copy()})['actions'],dtype=np.float32)
        hidden = [torch.stack([entry[j] for entry in traces]).float().cpu().numpy() for j in range(3)]
        del traces
        # Compare instrumented eager baseline with the ordinary full execution.
        ordinary = session.run_full(prepared=prepared,noise=n).detach().float().cpu().numpy().copy()
        parity_max = float(np.max(np.abs(ordinary - baseline_np)))
        if not np.allclose(ordinary,baseline_np,atol=1e-4,rtol=1e-4):
            raise RuntimeError(f'Instrumented baseline mismatch max_abs={parity_max}')
        interventions = {}
        for module in ('attn','ffn'):
            values=[]
            for layer in range(17):
                actions = session.run_submodule_probe(prepared=prepared,noise=n,
                    **{f'skip_{module}_layers':(layer,)})
                values.append(actions.float().cpu().numpy())
            interventions[f'raw_skip_{module}_actions'] = np.stack(values)
            interventions[f'skip_{module}_actions'] = np.stack([policy_actions(v) for v in values])
        # Independent repeated baseline detects buffer contamination by skips.
        repeated = session.run_submodule_probe(prepared=prepared,noise=n)
        repeat_error=float(np.max(np.abs(repeated.float().cpu().numpy()-baseline_np)))
        if repeat_error != 0:
            raise RuntimeError(f'Baseline changed after intervention: {repeat_error}')
        spans = make_modality_spans(token_count=hidden[0].shape[1],num_views=2,
            view_names=('agent_view','wrist_view'))
        meta.update(schema_version=1,modality_spans=[vars(s) for s in spans],
            denoise_steps=10,vlm_layer_pattern='vlm_full18',action_layer_pattern='full18',
            noise_seed=args.seed,noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
            observation_sha256=identity.hex(),ordinary_baseline_max_abs_error=parity_max,
            repeated_baseline_max_abs_error=repeat_error,attention_skip_semantics='keep_original_QKV_skip_attention_and_output_projection',
            hidden_dtype='float32',layer18_note='KV-only; not an Attention/FFN intervention')
        meta.update(action_space='unnormalized policy command; not measured EEF xyz',
            checkpoint_path=str(weights.resolve()),prompt_manifest_path=str(args.prompt_manifest.resolve()))
        np.savez_compressed(out,hidden_inputs=hidden[0],hidden_after_attn=hidden[1],hidden_outputs=hidden[2],
            baseline_actions=policy_actions(baseline_np),raw_baseline_actions=baseline_np,layer_ids=np.arange(1,18),noise=noise,
            metadata_json=np.asarray(json.dumps(meta,sort_keys=True)),**interventions)
        print(f'[{index+1}/{len(args.inputs)}] {out} ordinary_parity={parity_max} repeated_parity={repeat_error}',flush=True)


if __name__ == '__main__':
    main()
