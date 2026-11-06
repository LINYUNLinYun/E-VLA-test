"""Nonuniform Euler for the actual converted Triton π0 checkpoint."""
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'policies/realtime-vla-flash'

class Pi0Adapter:
    def __init__(self,output_dir=None):
        import torch
        self.torch=torch
        self.output_dir=Path(output_dir) if output_dir else None
        sys.path.insert(0,str(PROJECT))
        from scripts.experiment10.backend import Backend
        from scripts.experiment1.run_intermediate_action_probe import SUITE_ARTIFACT
        provenance=PROJECT/'results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json'
        old=json.loads(provenance.read_text())
        checkpoint=Path(old['checkpoint']['path'])
        metadata=[]
        self.prompt_suites={}
        for suite,artifact in SUITE_ARTIFACT.items():
            manifest=json.loads((checkpoint.parents[1]/artifact/'manifest.json').read_text())
            for prompt in manifest['prompts']:
                metadata.append(dict(suite=suite,task_description=prompt))
                self.prompt_suites.setdefault(prompt,suite)
        self.backend=Backend(metadata)
        if self.backend.weight_hash!=old['checkpoint']['sha256']:
            raise RuntimeError('π0 checkpoint differs from frozen Full18/V14 source')
        source=PROJECT/'results/experiment11_layer_skip_prefix_20261010/protocol/candidate_manifest.json'
        candidate=json.loads(source.read_text())
        mask=next(r['skip_layers'] for r in candidate['candidates'] if r['variant_id']=='V14_A03')
        if mask!=[3,4,9,11]: raise RuntimeError('Historical V14 mask mismatch')
        self.protocol=dict(model_id='pi0',H=50,R=5,D=10,depth=18,CFG=None,dtype='bfloat16',
            device='cuda',attention='native Triton',backend='scripts.experiment10.backend.Backend / Triton Pi0Inference',
            checkpoint=str(checkpoint),checkpoint_sha256=self.backend.weight_hash,
            provenance=str(provenance),provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),
            source_manifest=str(source),source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            mask_variant=mask,cameras=['agentview','wrist'],state_used=True,image_rotation=180,
            skip_semantics='hidden-update bypass with original-index KV shells; action expert full18')
        from preparations.idea1_v2_2.provenance import collect_checkpoint_assets
        self.protocol['checkpoint_assets']=collect_checkpoint_assets('pi0',checkpoint)
        self.protocol.update(execution_mode='eager Triton kernel dispatch; no historical CUDA graph replay',
            native_reference_execution_mode='single eager historical Backend.record; numerical reference only',
            startup_accounting='each new runtime prompt length counts actual record_run calls: normally 3 warmups + 1 graph capture, D10 each; distinct from requested logical predictions')
        self.schedules={}
        self.startup_events=[]

    def measure(self,fn):
        a,b=self.torch.cuda.Event(enable_timing=True),self.torch.cuda.Event(enable_timing=True)
        a.record(); out=fn(); b.record(); b.synchronize()
        return out,float(a.elapsed_time(b))

    def prepare(self,obs,instruction,skip):
        from preparations.idea1_v2_1.runtime_models import native_observation
        from scripts.spec.triton import pi0_infer as k
        if instruction not in self.prompt_suites: raise ValueError('π0 instruction absent from frozen native embedding manifest')
        tick=time.perf_counter()
        if 'agent_image' in obs:
            images=[obs['agent_image'],obs['wrist_image']]; state=obs['robot_state']
        elif 'observation/image' in obs:
            images=[obs['observation/image'],obs['observation/wrist_image']]; state=obs['observation/state']
        else:
            images,state=native_observation(obs)
        observation={'observation/image':np.asarray(images[0]),'observation/wrist_image':np.asarray(images[1]),
                     'observation/state':state,'prompt':instruction}
        startup_calls=[]
        original_record=k.Pi0Inference.record_run
        def counted_record(runtime,*args,**kwargs):
            startup_calls.append(int(runtime.denoise_steps))
            return original_record(runtime,*args,**kwargs)
        k.Pi0Inference.record_run=counted_record
        new_events=[]; startup_error=None
        try:
            self.backend.prepare(observation,dict(suite=self.prompt_suites[instruction],task_description=instruction),np.zeros((50,32),dtype=np.float32))
        except BaseException as exc:
            startup_error=repr(exc)
            raise
        finally:
            k.Pi0Inference.record_run=original_record
            if startup_calls:
                event=dict(id=uuid.uuid4().hex,category='startup_warmup_and_graph_capture',
                    forward_calls=len(startup_calls),action_NFE=sum(startup_calls),steps_per_forward=startup_calls,
                    warmup_calls=min(3,len(startup_calls)),graph_capture_calls=max(0,len(startup_calls)-3),
                    prompt_length=self.backend.length,error=startup_error,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
                self.startup_events.append(event); new_events.append(event)
                if self.output_dir:
                    with (self.output_dir/'startup_forward_ledger.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')
        r=self.backend.runtime; w,b=r.weights,r.buffers
        prep=(time.perf_counter()-tick)*1000
        b['observation_images_normalized'].copy_(self.backend.inputs[0]); b['observation_state_normalized'].copy_(self.backend.inputs[1])
        b['encoder_x'][512:].copy_(w['language_embeds'])
        _,vision=self.measure(lambda:k.vision_encoder(w,b,2))
        seq=b['encoder_x'].shape[0]
        _,vlm=self.measure(lambda:k.transformer_encoder(w,b,seq,vlm_layer_indices=tuple(i for i in range(18) if i+1 not in skip)))
        # Prefix K/V snapshots are independent of reusable decoder scratch buffers.
        c=dict(runtime=r,transformed_state=np.asarray(self.backend.transformed['state']).copy(),
            state=self.backend.inputs[1].clone(),K=b['encoder_K'][:,:seq].clone(),V=b['encoder_V'][:,:seq].clone(),seq=seq,
            startup_events=new_events)
        return c,dict(input_processing_ms=prep,vision_ms=vision,vlm_ms=vlm,prefill_total_ms=vision+vlm)

    def schedule(self,r,nodes):
        torch=self.torch; w=r.weights
        key=(id(r),tuple(nodes['nodes']))
        if key not in self.schedules:
            if len(nodes['timesteps'])==10:
                # Native D10 is a byte-level frozen fused checkpoint path.
                # Re-fusing BF16 source tensors changes rounding and is not
                # numerically equivalent to these released fused tensors.
                self.schedules[key]=(w['decoder_action_fused_time_biases'][:10].clone(),
                    [w['decoder_action_fused_out_proj_w'].clone() for _ in range(10)],
                    [w['decoder_action_fused_out_proj_b'].clone() for _ in range(10)],
                    torch.tensor(nodes['timesteps'],device=w['decoder_action_fused_time_biases'].device,dtype=torch.float32))
                return self.schedules[key]
            mlp=w['decoder_action_time_mlp_in_w']; hidden=1024
            action_bias=mlp[:hidden].T @ w['decoder_action_in_proj_b']
            times=torch.tensor(nodes['timesteps'],device=w['decoder_action_fused_time_biases'].device,dtype=torch.float32)
            fraction=torch.arange(0,hidden,2,device=times.device,dtype=torch.float32)/hidden
            period=4e-3*(4./4e-3)**fraction
            # Preserve Backend._build_custom_step_weights arithmetic order.
            pos=(1.0/period*(2.0*torch.pi))[None,:]*times[:,None]
            emb=torch.cat([torch.sin(pos),torch.cos(pos)],dim=1).to(mlp.dtype)
            bias=(action_bias[None,:]+emb@mlp[hidden:]+w['decoder_action_time_mlp_in_b'][None,:]).to(torch.bfloat16)
            # Native output W/b have dt=-.1 fused; scale each update separately.
            self.schedules[key]=(bias.clone(),[(w['decoder_action_fused_out_proj_w']*(-dt*10)).clone() for dt in nodes['dt']],
                [(w['decoder_action_fused_out_proj_b']*(-dt*10)).clone() for dt in nodes['dt']],times.clone())
        return self.schedules[key]

    def sample(self,c,seed,schedule,instrumented=True):
        from preparations.idea1_v2_2.schedules import flow_schedule
        from scripts.experiment1.run_intermediate_action_probe import _decoder_layer
        from scripts.spec.triton import pi0_infer as k
        torch=self.torch; r=c['runtime']; w,b=r.weights,r.buffers; seq=c['seq']
        nodes=flow_schedule('pi0',schedule['m'],schedule['num_steps'])
        biases,out_w,out_b,times=self.schedule(r,nodes)
        noise=np.random.default_rng(int(seed)).standard_normal((50,32),dtype=np.float32)
        noise_tensor=torch.as_tensor(noise,device='cuda',dtype=torch.bfloat16)
        b['observation_state_normalized'].copy_(c['state']); b['encoder_K'][:,:seq].copy_(c['K']); b['encoder_V'][:,:seq].copy_(c['V'])
        b['diffusion_noise'].copy_(noise_tensor)
        traces=[]; diagnostics=[]
        def run():
            calls=0; applied=[]
            k.matmul_1_32_1024_bias(b['observation_state_normalized'],w['decoder_state_in_proj_w'],w['decoder_state_in_proj_b'],b['decoder_state_buf'])
            for i in range(nodes['NFE']):
                if instrumented: before=b['diffusion_noise'].clone()
                b['decoder_x'][:1].copy_(b['decoder_state_buf'])
                k.matmul_k_32_1024_bias_silu(b['diffusion_noise'],w['decoder_action_fused_in_proj_w'],biases[i],b['decoder_x_buf'])
                k.matmul_k_1024_1024_bias(b['decoder_x_buf'],w['decoder_action_mlp_w'],w['decoder_action_mlp_b'],b['decoder_x'][1:])
                for layer in range(18): _decoder_layer(w,b,seq,layer)
                k.rms_matmul_k_1024_32_bias_res(b['decoder_x'][1:],out_w[i],out_b[i],b['diffusion_noise'],b['decoder_norm_factor_buf'])
                calls+=1
                if instrumented: applied.append(biases[i].detach())
                if instrumented: traces.append((b['diffusion_noise'].float()-before.float()).clone())
            return b['diffusion_noise'],calls,applied
        (raw,calls,applied),ms=self.measure(run)
        raw=raw.clone()
        tick=time.perf_counter()
        raw_np=raw.float().cpu().numpy()
        actions=np.asarray(self.backend.output_transform({'state':c['transformed_state'].copy(),'actions':raw_np.copy()})['actions'],dtype=np.float32)
        decode=(time.perf_counter()-tick)*1000
        nodes['steps']=[dict(update_rms=float(dx.square().mean().sqrt().cpu()),prediction_rms=float((dx/dt).square().mean().sqrt().cpu())) for dx,dt in zip(traces,nodes['dt'])]
        nodes['actual_timesteps']=times.float().cpu().tolist() if instrumented else None
        nodes['actual_NFE']=calls
        nodes['actual_time_bias_sha256']=[hashlib.sha256(v.float().cpu().numpy().tobytes()).hexdigest() for v in applied]
        nodes['time_evaluation']='scalar time fused into immutable action input bias; hashes record biases actually passed to kernel'
        nodes['startup_events']=c['startup_events']
        nodes['startup_event_reuse_note']='event IDs identify construction once; repeated sampling of same prepared cache does not incur startup again'
        return actions,raw,noise,nodes,ms,decode
