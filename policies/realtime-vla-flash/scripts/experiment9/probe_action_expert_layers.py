"""Offline decoder hidden similarity and whole-layer skip sensitivity; VLM stays Full18."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
FLOW_STEPS = (0, 2, 5, 9)
CRITERIA = {"first12_nmse_task_mean_max":0.01,"first12_nmse_task_bootstrap_upper_max":0.02,
            "first12_gripper_flip_task_mean_max":0.01,"worst_phase_task_mean_nmse_max":0.05,
            "minimum_tasks_per_suite_phase":3,
            "maximum_closed_loop_candidates":2}


def action_metrics(candidate, baseline):
    result = {}
    for horizon in (12,50):
        a,b = np.asarray(candidate[:horizon,:7],dtype=np.float64),np.asarray(baseline[:horizon,:7],dtype=np.float64)
        result[f"first{horizon}_nmse"] = float(np.sum((a-b)**2)/max(float(np.sum(b*b)),1e-12))
        result[f"first{horizon}_gripper_flip"] = float(np.mean((a[:,6]>0)!=(b[:,6]>0)))
        result[f"first{horizon}_translation_rmse"] = float(np.sqrt(np.mean((a[:,:3]-b[:,:3])**2)))
        result[f"first{horizon}_rotation_rmse"] = float(np.sqrt(np.mean((a[:,3:6]-b[:,3:6])**2)))
    return result


def summarize_candidates(rows):
    summaries = []
    for layer in range(1,19):
        chosen=[r for r in rows if r['layer']==layer]
        if not chosen:
            continue
        task_keys=sorted({(r['suite'],r['task_id']) for r in chosen})
        task_means=np.asarray([np.mean([r['first12_nmse'] for r in chosen if (r['suite'],r['task_id'])==k]) for k in task_keys])
        flips=np.asarray([np.mean([r['first12_gripper_flip'] for r in chosen if (r['suite'],r['task_id'])==k]) for k in task_keys])
        rng=np.random.default_rng(20261005+layer)
        bootstrap=task_means[rng.integers(0,len(task_means),size=(2000,len(task_means)))].mean(axis=1)
        phases=sorted({r['semantic_phase'] for r in chosen})
        phase_means={p:float(np.mean([np.mean([r['first12_nmse'] for r in chosen if r['semantic_phase']==p and (r['suite'],r['task_id'])==k]) for k in task_keys if any(r['semantic_phase']==p and (r['suite'],r['task_id'])==k for r in chosen)])) for p in phases}
        entry={'layer':layer,'task_count':len(task_means),'observation_seed_count':len(chosen),'task_mean_nmse':float(task_means.mean()),'bootstrap95_low':float(np.quantile(bootstrap,.025)),'bootstrap95_high':float(np.quantile(bootstrap,.975)),'task_mean_gripper_flip':float(flips.mean()),'phase_task_mean_nmse':phase_means}
        coverage={(s,p):len({r['task_id'] for r in chosen if r['suite']==s and r['semantic_phase']==p}) for s in ('libero_object','libero_spatial','libero_goal','libero_10') for p in ('initial','contact','goal_alignment')}
        entry['minimum_suite_phase_task_count']=min(coverage.values())
        entry['eligible']=entry['task_mean_nmse']<=CRITERIA['first12_nmse_task_mean_max'] and entry['bootstrap95_high']<=CRITERIA['first12_nmse_task_bootstrap_upper_max'] and entry['task_mean_gripper_flip']<=CRITERIA['first12_gripper_flip_task_mean_max'] and max(phase_means.values())<=CRITERIA['worst_phase_task_mean_nmse_max'] and entry['minimum_suite_phase_task_count']>=CRITERIA['minimum_tasks_per_suite_phase']
        summaries.append(entry)
    return sorted(summaries,key=lambda r:r['task_mean_nmse'])


def write_csv(path, rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]) if rows else ['sample_id'])
        writer.writeheader();writer.writerows(rows)


def decoder_forward(runtime, images, state, noise, *, skip_layer=None, capture=False):
    import torch
    from scripts.experiment1.run_intermediate_action_probe import _decoder_layer
    from scripts.spec.triton import pi0_infer as kernels
    w,b=runtime.weights,runtime.buffers
    seq=int(b['encoder_x'].shape[0])
    b['observation_images_normalized'].copy_(images);b['observation_state_normalized'].copy_(state);b['diffusion_noise'].copy_(noise)
    b['encoder_x'][runtime.num_views*256:].copy_(w['language_embeds'])
    kernels.vision_encoder(w,b,runtime.num_views);kernels.transformer_encoder(w,b,seq)
    kernels.matmul_1_32_1024_bias(b['observation_state_normalized'],w['decoder_state_in_proj_w'],w['decoder_state_in_proj_b'],b['decoder_state_buf'])
    cosine,residual=[],[]
    visits=[0]*18
    for step in range(10):
        b['decoder_x'][:1].copy_(b['decoder_state_buf'])
        kernels.matmul_k_32_1024_bias_silu(b['diffusion_noise'],w['decoder_action_fused_in_proj_w'],w['decoder_action_fused_time_biases'][step],b['decoder_x_buf'])
        kernels.matmul_k_1024_1024_bias(b['decoder_x_buf'],w['decoder_action_mlp_w'],w['decoder_action_mlp_b'],b['decoder_x'][1:])
        snapshots=[b['decoder_x'][1:].clone()] if capture and step in FLOW_STEPS else None
        for layer in range(18):
            if layer != skip_layer:
                _decoder_layer(w,b,seq,layer);visits[layer]+=1
            if snapshots is not None:
                snapshots.append(b['decoder_x'][1:].clone())
        if snapshots is not None:
            flat=torch.stack(snapshots).reshape(19,-1).double()
            norms=torch.linalg.vector_norm(flat,dim=1)
            unit=flat/torch.clamp(norms[:,None],min=1e-12)
            cosine.append((unit @ unit.T).cpu().numpy().astype(np.float32))
            residual.append((torch.linalg.vector_norm(flat[1:]-flat[:-1],dim=1)/torch.clamp(norms[:-1],min=1e-12)).cpu().numpy().astype(np.float32))
        kernels.rms_matmul_k_1024_32_bias_res(b['decoder_x'][1:],w['decoder_action_fused_out_proj_w'],w['decoder_action_fused_out_proj_b'],b['diffusion_noise'],b['decoder_norm_factor_buf'])
    torch.cuda.synchronize()
    expected=[0 if i==skip_layer else 10 for i in range(18)]
    if visits!=expected:
        raise RuntimeError(f'Unexpected layer visits: {visits}')
    return b['diffusion_noise'].clone(),np.asarray(cosine),np.asarray(residual)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--converted-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--noise-seeds',nargs='+',type=int,default=[12345,23456,34567])
    parser.add_argument('--sample-limit',type=int)
    parser.add_argument('--worker',action='store_true',help='Internal seed-shard worker; disable automatic parallel dispatch')
    args=parser.parse_args()
    existing_summary=args.output_dir/'summary.json'
    if args.worker and existing_summary.is_file():
        saved=json.loads(existing_summary.read_text())
        source=json.loads(args.observations.read_text())
        if saved.get('complete') and saved.get('checkpoint_unchanged') and saved.get('noise_seeds')==args.noise_seeds and saved.get('source_manifest')==str(args.observations.resolve()) and saved.get('sample_count')==len(source['observations']) and saved.get('task_count')==40:
            print('Reusing completed seed-shard summary; merged audit will verify every sample identity.',flush=True)
            return
    if not args.worker and args.sample_limit is None and args.noise_seeds==[12345,23456,34567]:
        import subprocess
        command=[sys.executable,str(Path(__file__).with_name('run_parallel_action_probes.py')),'--observations',str(args.observations),'--converted-root',str(args.converted_root),'--output-dir',str(args.output_dir)]
        subprocess.run(command,check=True,cwd=ROOT)
        return
    import pickle
    import torch
    from scripts.experiment1.run_intermediate_action_probe import _load_prompt_embeddings, _sha256
    from scripts.spec.spec_serve_policy import _build_policy_components
    from scripts.spec.triton import pi0_infer
    from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
    torch.set_grad_enabled(False)
    source=json.loads(args.observations.read_text())
    paths=[Path(p) for p in source['observations']]
    if args.sample_limit:
        paths=paths[:args.sample_limit]
    metadata=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            metadata.append(json.loads(str(data['metadata_json'].item())))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    sample_dir=args.output_dir/'samples';sample_dir.mkdir(exist_ok=True)
    (args.output_dir/'prespecified_criteria.json').write_text(json.dumps(CRITERIA,indent=2)+'\n')
    embeddings=_load_prompt_embeddings(args.converted_root,metadata)
    _,transform,output_transform=_build_policy_components(config_name='pi0_libero',checkpoint_dir=args.converted_root/'base')
    weights_path=args.converted_root/'base/base_weights.pkl'
    weight_hash=_sha256(weights_path)
    with weights_path.open('rb') as stream:
        base=pickle.load(stream)
    groups={}
    for index,meta in enumerate(metadata):
        groups.setdefault(len(embeddings[meta['suite'],meta['task_description']]),[]).append(index)
    sensitivity,hidden=[],[]
    completed=0
    for length,indices in sorted(groups.items()):
        checkpoint=dict(base);meta=metadata[indices[0]]
        checkpoint['language_embeds']=embeddings[meta['suite'],meta['task_description']]
        runtime=pi0_infer.Pi0Inference(checkpoint,num_views=2,chunk_size=50)
        for index in indices:
            meta=metadata[index];path=paths[index];obs_hash=hashlib.sha256(path.read_bytes()).hexdigest()
            runtime.weights['language_embeds'].copy_(embeddings[meta['suite'],meta['task_description']].to('cuda'))
            with np.load(path,allow_pickle=False) as data:
                transformed=transform({'observation/image':data['agent_image'],'observation/wrist_image':data['wrist_image'],'observation/state':data['robot_state'],'prompt':str(data['prompt'].item())})
            for seed in args.noise_seeds:
                target=sample_dir/f"{meta['sample_id']}_seed{seed}.npz"
                if target.exists():
                    with np.load(target,allow_pickle=False) as existing:
                        info=json.loads(str(existing['metadata_json'].item()))
                        if info['observation_sha256']!=obs_hash or info['noise_seed']!=seed or info['checkpoint_sha256']!=weight_hash:
                            raise RuntimeError(f'Resume identity mismatch: {target}')
                else:
                    rng=np.random.default_rng(np.random.SeedSequence([seed,int(obs_hash[:8],16)]))
                    noise_np=rng.standard_normal((50,32),dtype=np.float32)
                    images,state,noise=prepare_triton_inputs_from_transformed(transformed=transformed,device='cuda',action_horizon=50,action_dim=32,noise=noise_np)
                    ordinary=runtime.forward(images,state,noise).clone()
                    baseline,cosine,residual=decoder_forward(runtime,images,state,noise,capture=True)
                    if not torch.equal(ordinary,baseline):
                        raise RuntimeError(f'Instrumented baseline not bitwise equal: {meta["sample_id"]}')
                    def commands(raw):
                        return np.asarray(output_transform({'state':np.asarray(transformed['state']).copy(),'actions':raw.float().cpu().numpy().copy()})['actions'],dtype=np.float32)
                    baseline_command=commands(baseline)
                    variants=[]
                    for layer in range(18):
                        candidate,_,_=decoder_forward(runtime,images,state,noise,skip_layer=layer)
                        variants.append(commands(candidate))
                    repeated,_,_=decoder_forward(runtime,images,state,noise)
                    if not torch.equal(repeated,baseline):
                        raise RuntimeError('Baseline changed after layer interventions')
                    info={**meta,'noise_seed':seed,'noise_sha256':hashlib.sha256(noise_np.tobytes()).hexdigest(),'observation_sha256':obs_hash,'checkpoint_sha256':weight_hash,'full_baseline_bitwise_parity':True,'repeat_baseline_bitwise_parity':True,'vlm_layer_pattern':'vlm_full18','denoise_steps':10,'hidden_tokens':'50 action tokens; state token excluded','hidden_layers':list(range(19)),'hidden_layer0':'decoder action embedding before layer1','flow_steps':FLOW_STEPS,'flow_times':[1-s/10 for s in FLOW_STEPS],'intervention':'Skip one complete Action Expert block at every denoising step; prefix VLM stays full18; no production runtime mutation'}
                    np.savez_compressed(target,baseline_actions=baseline_command,skip_actions=np.stack(variants),layer_cosine=cosine,adjacent_relative_residual=residual,noise=noise_np,metadata_json=np.asarray(json.dumps(info,sort_keys=True)))
                with np.load(target,allow_pickle=False) as data:
                    info=json.loads(str(data['metadata_json'].item()))
                    for layer in range(1,19):
                        sensitivity.append({**{k:info[k] for k in ('sample_id','suite','task_id','episode_idx','semantic_phase','noise_seed')},'layer':layer,**action_metrics(data['skip_actions'][layer-1],data['baseline_actions'])})
                        for j,step in enumerate(FLOW_STEPS):
                            hidden.append({**{k:info[k] for k in ('sample_id','suite','task_id','semantic_phase','noise_seed')},'layer':layer,'flow_step':step,'flow_t':1-step/10,'adjacent_cosine':float(data['layer_cosine'][j,layer,layer-1]),'cosine_vs_layer18':float(data['layer_cosine'][j,layer,18]),'relative_residual':float(data['adjacent_relative_residual'][j,layer-1])})
                completed+=1
                print(f'[{completed}/{len(paths)*len(args.noise_seeds)}] {target.name}: parity passed; 18 interventions saved',flush=True)
        del runtime,checkpoint;gc.collect();torch.cuda.empty_cache()
    write_csv(args.output_dir/'layer_action_sensitivity.csv',sensitivity)
    write_csv(args.output_dir/'layer_hidden_similarity.csv',hidden)
    ranking=summarize_candidates(sensitivity)
    candidates=[r['layer'] for r in ranking if r['eligible']][:CRITERIA['maximum_closed_loop_candidates']]
    unchanged=weight_hash==_sha256(weights_path)
    if not unchanged:
        raise RuntimeError('Checkpoint hash changed')
    summary={'complete':True,'sample_count':len(paths),'noise_seeds':args.noise_seeds,'task_count':len({(m['suite'],m['task_id']) for m in metadata}),'fixed_observation_inference_count':len(paths)*len(args.noise_seeds)*21,'checkpoint_sha256':weight_hash,'checkpoint_unchanged':unchanged,'gpu':torch.cuda.get_device_name(0),'criteria':CRITERIA,'ranking':ranking,'closed_loop_candidates':candidates,'source_manifest':str(args.observations.resolve()),'decision':'screen_selected_layers' if candidates else 'No credible single-layer pruning candidates under the prespecified criteria'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    if '--worker' in sys.argv:
        import fcntl
        output=Path(sys.argv[sys.argv.index('--output-dir')+1])
        output.mkdir(parents=True,exist_ok=True)
        with (output/'.worker.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            main()
    else:
        main()
