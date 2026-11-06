"""Real v2.2 native samplers, isolated from upstream and v2.1 artifacts."""
from contextlib import nullcontext
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from preparations.idea1_v2_1.runtime_models import NativePolicy as V21Policy, digest, native_observation, seeded
from preparations.idea1_v2_2.schedules import normalize_schedule, flow_schedule, ddim_indices
from preparations.idea1_v2_2.provenance import collect_checkpoint_assets, collect_scientific_sources


class NativePolicy(V21Policy):
    def __init__(self,model_id,output_dir=None):
        if model_id=='pi0':
            import torch
            from preparations.idea1_v2_2.pi0_adapter import Pi0Adapter
            self.torch=torch; self.model_id=model_id; self.calls=0
            self.output_dir=Path(output_dir) if output_dir else None
            if self.output_dir: self.output_dir.mkdir(parents=True,exist_ok=True)
            self.pi0=Pi0Adapter(self.output_dir); self.protocol=self.pi0.protocol; self.layers=[]
        else:
            super().__init__(model_id,output_dir)
            verified_path=ROOT/f'preparations/idea1_v2_1/metadata/{model_id}/weights_verified.json'
            verified=json.loads(verified_path.read_text())
            checkpoint=Path(verified['checkpoint'])
            if not verified['ok']: raise RuntimeError('Checkpoint verification source did not pass')
            shards=[]
            for row in verified['shards']:
                path=Path(row['path'])
                if path.stat().st_size!=row['expected_bytes']:
                    raise RuntimeError(f'Checkpoint shard size changed: {path}')
                # Rehash actual loaded weights, rather than trusting historical metadata.
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
                if h.hexdigest()!=row['actual_sha256']: raise RuntimeError(f'Checkpoint hash changed: {path}')
                shards.append(dict(path=str(path),sha256=h.hexdigest(),bytes=path.stat().st_size))
            assets=collect_checkpoint_assets(model_id,checkpoint)
            self.protocol.update(checkpoint=str(checkpoint),checkpoint_revision=verified['revision'],
                checkpoint_sha256=digest(shards),checkpoint_shards=shards,checkpoint_assets=assets,
                checkpoint_verification_source=str(verified_path))
        self.protocol['R_by_suite'] = self.protocol['R'] if isinstance(self.protocol['R'],dict) else {
            s:self.protocol['R'] for s in ('libero_spatial','libero_object','libero_goal','libero_10')}
        self.protocol.update(sampling_version='v2.2',cache_sharing='same policy/input/mask only',
            action_generation_boundary='prepared VLM condition and state/noise projection through final output projection; excludes decode')
        self.code_hashes=collect_scientific_sources(model_id)
        self.protocol['v22_code_hashes']=self.code_hashes
        self.protocol['scientific_code_hashes']=self.code_hashes
        from preparations.idea1_v2_2.provenance import receipt
        loaded={}
        objects=([self.pi0.backend,self.pi0.backend.transform,self.pi0.backend.output_transform] if model_id=='pi0' else
                 [self.model,*self.modules.values(),getattr(self,'processor',None),getattr(self,'tokenizer',None)])
        for obj in objects:
            if obj is None: continue
            cls=obj if inspect.isfunction(obj) else type(obj)
            source=inspect.getsourcefile(cls)
            if source: loaded[f'{cls.__module__}.{cls.__name__}']=receipt(source)
        self.protocol['loaded_scientific_sources']=loaded
        self.protocol['timing_boundaries']=dict(
            action_generation_ms='GPU stage: prepared model/mask VLM features and initial noise through state/action projection, all sampling updates and final output projection; excludes VLM prefill, decode, audit reductions and persistence',
            server_total_ms='prepare wall plus sampling/decode, trace reductions and CPU tensor copies; stops before result hash/row assembly, prediction/noise NPZ compression/write and ledger persistence; shared cache includes original prepare wall for comparability',
            request_wall_ms='independent predict wall including prepare, sampling/decode/audit, NPZ compression/write and prediction ledger append; excludes RPC transport and caller logs')
        self._samplers={}
        if self.output_dir:
            (self.output_dir/'native_protocol.json').write_text(json.dumps(self.protocol,indent=2))

    def reset(self): return {'ok':True}

    def _measure(self,fn):
        torch=self.torch
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record(); result=fn(); b.record(); b.synchronize()
        return result,float(a.elapsed_time(b))

    def prepare_condition(self,obs,instruction,skip=(),reference=False,probe=False):
        skip=tuple(sorted(skip))
        depth=self.protocol['depth']
        if len(set(skip))!=len(skip) or any(isinstance(x,bool) or not isinstance(x,int) or x<1 or x>depth for x in skip):
            raise ValueError('Invalid original 1-based skip mask')
        if probe: raise ValueError('v2.2 feature probes are separate diagnostics, not sampler requests')
        start=time.perf_counter()
        if self.model_id=='pi0':
            condition,timing=self.pi0.prepare(obs,instruction,skip)
        else:
            data,_=self._prepare(obs,instruction,0)
            prep=(time.perf_counter()-start)*1000
            self._clear_redundant_class_forwards()
            events=[]; handles=[]
            def before(*args):
                e=self.torch.cuda.Event(enable_timing=True); e.record(); events.append([e,None])
            def after(*args):
                e=self.torch.cuda.Event(enable_timing=True); e.record(); events[-1][1]=e
            handles=[self.modules['vision'].register_forward_pre_hook(before),self.modules['vision'].register_forward_hook(after)]
            try:
                with self.torch.inference_mode(),self.adapter(self.model,skip,reference=reference):
                    if self.model_id=='xiaomi-robotic0':
                        inputs={k:v for k,v in data.items() if k not in ('state','action_mask','seed','num_steps')}
                        out,ms=self._measure(lambda:self.model.vlm(**inputs,use_cache=True))
                        condition=dict(vlm=out,state=data['state'],action_mask=data['action_mask'])
                    else:
                        out,ms=self._measure(lambda:self.model(input_ids=data['input_ids'],images=data['image_tensor'],use_cache=True))
                        condition=out.logits[:,-1,:].unsqueeze(1).clone()
            finally:
                for h in handles: h.remove()
                self._clear_redundant_class_forwards()
            vision=sum(a.elapsed_time(b) for a,b in events)
            timing=dict(input_processing_ms=prep,vision_ms=float(vision),vlm_ms=ms-float(vision),prefill_total_ms=ms)
        return dict(owner=id(self),condition=condition,input_hash=digest({'obs':obs,'instruction':instruction}),
                    skip=skip,timing=timing,prepare_wall_ms=(time.perf_counter()-start)*1000,
                    cache_key=digest(dict(model=self.model_id,protocol=digest(self.protocol),skip=skip,input=digest({'obs':obs,'instruction':instruction}),dtype='bfloat16')))

    def _xiaomi_sample(self,c,seed,schedule,instrumented=True):
        torch=self.torch; model=self.model
        mask,state,out=c['action_mask'],c['state'],c['vlm']
        nodes=flow_schedule(self.model_id,schedule['m'],schedule['num_steps'])
        with seeded(torch,int(seed)): noise=torch.randn_like(mask)
        def run():
            bs,length,_=mask.shape; query=length+state.shape[1]+1
            ids=torch.arange(query,device=mask.device).view(1,1,-1).repeat(3,bs,1)+out.position_ids.max(dim=-1)[0][...,None]+1
            pos=model.rotary_emb(mask,ids)
            dit_mask=torch.tril(torch.ones((bs,query,query),device=mask.device))
            cache_mask=out.attention_mask[:,None,:].expand(-1,query,-1)
            attn=torch.cat([cache_mask,dit_mask],dim=-1)[:,None].bool()
            state_embed=model.state_projector(state); x=noise.clone(); traces=[]; actual=[]; diagnostics=[]; calls=0
            for t,dt in zip(nodes['timesteps'],nodes['dt']):
                ts=torch.ones((bs,1,1),device=x.device,dtype=x.dtype)*t
                if instrumented: actual.append(ts.detach())
                v=model.dit_forward(x,ts,mask,state_embed,pos,out.past_key_values,attn)
                calls+=1
                updated=x+v*dt
                if instrumented: traces.append((v.detach(),(updated-x).detach()))
                x=updated
            return x,traces,actual,diagnostics,calls
        (raw,traces,actual,diagnostics,calls),ms=self._measure(run)
        tick=time.perf_counter()
        actions=self.processor.decode_action(raw,'libero_all')[0,:,:7].float().cpu().numpy()
        return actions,raw,noise,dict(nodes,actual_timesteps=[float(t.flatten()[0].cpu()) for t in actual] if instrumented else None,
            actual_NFE=calls,steps=self._traces(traces)),ms,(time.perf_counter()-tick)*1000

    def _traces(self,traces):
        return [dict(prediction_rms=float(v.float().square().mean().sqrt().cpu()),
                     update_rms=float(dx.float().square().mean().sqrt().cpu())) for v,dx in traces]

    def _db_sampler(self,schedule):
        key=tuple(schedule.items())
        if key not in self._samplers:
            head=self.model.model.action_head
            if schedule['kind']=='native':
                sampler=head.create_ddim(schedule['num_steps'])
            else:
                from dexbotic.model.cogact.action_model.diffusion import SpacedDiffusion, create_diffusion
                native=create_diffusion('ddim10',noise_schedule=head.noise_schedule,diffusion_steps=head.diffusion_steps,sigma_small=True,learn_sigma=False)
                base=head.diffusion
                indices=ddim_indices(max(native.timestep_map),schedule['m'])
                sampler=SpacedDiffusion(use_timesteps=set(indices),betas=base.betas,
                    model_mean_type=native.model_mean_type,model_var_type=native.model_var_type,loss_type=native.loss_type)
            self._samplers[key]=sampler
        return self._samplers[key]

    def _db_sample(self,c,seed,schedule,instrumented=True):
        torch=self.torch; head=self.model.model.action_head
        sampler=self._db_sampler(schedule)
        # Explicit assignment prevents legacy head state from leaking across schedules.
        head.ddim_diffusion=sampler
        with seeded(torch,int(seed)):
            noise=torch.randn(c.shape[0],self.model.config.chunk_size,self.model.config.action_dim,device=c.device,dtype=c.dtype)
        actual=[]; predictions=[]
        def net(x,t,**kwargs):
            actual.append(t.detach().clone())
            v=head.net.forward_with_cfg(x,t,**kwargs); predictions.append(v.detach())
            return v
        def run():
            x=torch.cat([noise.clone(),noise.clone()],0)
            uncondition=head.net.z_embedder.uncondition.unsqueeze(0).expand(c.shape[0],1,-1)
            return sampler.ddim_sample_loop(net if instrumented else head.net.forward_with_cfg,x.shape,x,clip_denoised=False,
                model_kwargs=dict(z=torch.cat([c,uncondition],0),cfg_scale=1.5),progress=False,device=c.device,eta=0.)
        raw,ms=self._measure(run)
        tick=time.perf_counter()
        actions=np.asarray(self.model._denorm(raw.chunk(2,dim=0)[0][0].float().cpu().numpy(),self.norm),dtype=np.float32)
        actions[:,-1]=np.where(actions[:,-1]<0,-1,1)
        mapping=list(reversed(sampler.timestep_map))
        audit=dict(NFE=len(mapping),actual_NFE=len(actual) if instrumented else None,
            timesteps=[int(t[0].cpu()) for t in actual] if instrumented else mapping,nodes=mapping+['clean'],
            timestep_map=list(sampler.timestep_map),alpha_cumprod=sampler.alphas_cumprod[::-1].tolist(),
            sigma=np.sqrt(1-sampler.alphas_cumprod[::-1]).tolist(),
            SNR=(sampler.alphas_cumprod[::-1]/(1-sampler.alphas_cumprod[::-1])).tolist(),
            alpha_clean=1.,eta=0.,CFG=1.5,cfg_batch_multiplier=2,
            prediction_rms=[float(v.float().square().mean().sqrt().cpu()) for v in predictions])
        if instrumented and audit['timesteps']!=mapping: raise RuntimeError('Actual DDIM mapping differs from sampler')
        return actions,raw,noise,audit,ms,(time.perf_counter()-tick)*1000

    def sample_condition(self,prepared,seed,schedule=None,category='offline',cache_path=None):
        if prepared['owner']!=id(self): raise ValueError('Condition belongs to another policy')
        schedule=normalize_schedule(self.model_id,schedule)
        instrumented=category not in ('measurement','warmup','benchmark')
        start=time.perf_counter()
        with self.torch.inference_mode():
            if self.model_id=='pi0': result=self.pi0.sample(prepared['condition'],seed,schedule,instrumented)
            elif self.model_id=='xiaomi-robotic0': result=self._xiaomi_sample(prepared['condition'],seed,schedule,instrumented)
            else: result=self._db_sample(prepared['condition'],seed,schedule,instrumented)
        actions,raw,noise,audit,ms,decode=result
        if actions.shape!=(self.protocol['H'],7) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid action output {actions.shape}')
        if audit['NFE']!=schedule['num_steps']: raise RuntimeError('Action NFE budget mismatch')
        noise=np.asarray(noise.float().cpu().numpy() if hasattr(noise,'cpu') else noise)
        raw=np.asarray(raw.float().cpu().numpy() if hasattr(raw,'cpu') else raw)
        timing=dict(prepared['timing'],action_generation_ms=ms,action_expert_ms=ms,decode_ms=decode,
                    measurement_mode='instrumented_audit' if instrumented else 'minimal_action_generation',
                    server_total_ms=prepared['prepare_wall_ms']+(time.perf_counter()-start)*1000,
                    shared_condition=True)
        self.calls+=1
        audit.update(schedule=schedule,skip=list(prepared['skip']),cache_key=prepared['cache_key'],
                     instrumentation_mode='instrumented_audit' if instrumented else 'minimal',
                     diagnostic_in_action_generation='trace capture included, no subtraction' if instrumented else
                         'no per-step trace copies/events/hooks or tensor collection; only outer stage events; NFE inferred from separately audited schedule')
        row=dict(ok=True,actions=actions.astype(np.float32),timing=timing,input_hash=prepared['input_hash'],
                 noise_hash=digest(noise),noise_shapes=[list(noise.shape)],audit=audit,
                 seed=int(seed),call=self.calls,category=category,raw_prediction_hash=digest(raw),loaded_code_hashes=self.code_hashes)
        dest=Path(cache_path) if cache_path else (self.output_dir/'predictions'/f'{self.calls:09d}_{uuid.uuid4().hex}.npz' if self.output_dir else None)
        if dest:
            dest.parent.mkdir(parents=True,exist_ok=True)
            noise_path=dest.with_name(dest.stem+'_noise.npz'); pred_path=dest.with_name(dest.stem+'_prediction.npz')
            np.savez_compressed(noise_path,initial_noise=noise)
            np.savez_compressed(pred_path,actions=row['actions'],raw_prediction=raw)
            row.update(noise_path=str(noise_path),prediction_path=str(pred_path))
        else: row.update(noise_path=None,prediction_path=None)
        if self.output_dir:
            with (self.output_dir/'prediction_ledger.jsonl').open('a') as f:
                f.write(json.dumps({k:v for k,v in row.items() if k!='actions'})+'\n')
        return row

    def predict(self,obs,instruction,seed,skip=(),schedule=None,category='formal',reference=False,probe=False,cache_path=None,**unused):
        started=time.perf_counter()
        prepared=self.prepare_condition(obs,instruction,skip,reference,probe)
        result=self.sample_condition(prepared,seed,schedule,category,cache_path)
        result['timing']['shared_condition']=False
        result['timing']['request_wall_ms']=(time.perf_counter()-started)*1000
        return result

    def parity(self,obs,instruction,seed=7,skip=None,schedules=None):
        """One real point/mask. Caller must run two points and Full/variant.

        At most 4*S+8 predictions: <=192 total for four point/mask pairs
        when S<=10. Native references and round-trip calls are charged too.
        """
        schedules=schedules or [dict(kind='flow' if self.model_id!='db-cogact' else 'ddim_custom',m=m,num_steps=2) for m in (.5,.3,.7,.1,.25,.9,.75,.6)]
        schedules=[normalize_schedule(self.model_id,s) for s in schedules]
        skip=skip or []
        initial_calls=self.calls
        startup_start=len(self.pi0.startup_events) if self.model_id=='pi0' else 0
        prepared=self.prepare_condition(obs,instruction,skip)
        rows=[]; cached_outputs={}
        def compare(label,a,b,noise=True):
            x,y=a['actions'],b['actions']
            same_noise=a.get('noise_hash')==b.get('noise_hash') if noise else None
            finite=bool(np.isfinite(x).all() and np.isfinite(y).all())
            row=dict(comparison=label,max_abs=float(np.max(np.abs(x-y))),mean_abs=float(np.mean(np.abs(x-y))),
                rtol=1e-4,atol=1e-5,same_noise=same_noise,passed=finite and bool(np.allclose(x,y,rtol=1e-4,atol=1e-5)) and (same_noise if noise else True),
                left_audit=a.get('audit'),right_audit=b.get('audit'),input_hash=prepared['input_hash'])
            rows.append(row)
        for s in schedules:
            cached=self.sample_condition(prepared,seed,s,category='parity')
            fresh=self.predict(obs,instruction,seed,skip,s,category='parity')
            cached_outputs[tuple(s.items())]=cached
            compare('cache_vs_independent',cached,fresh)
            rows[-1].update(schedule=s)
            rows[-1]['passed'] &= cached['audit']['actual_NFE']==s['num_steps'] and fresh['audit']['actual_NFE']==s['num_steps']
        native_n=5 if self.model_id=='xiaomi-robotic0' else 10
        native=normalize_schedule(self.model_id,dict(kind='native',m=None,num_steps=native_n))
        d2=normalize_schedule(self.model_id,dict(kind='native',m=None,num_steps=2))
        # Start and end use independent prefill. All custom schedules are visited
        # between them so a stale sampler or mutated cache is exposed.
        before=self.predict(obs,instruction,seed,skip,native,category='parity')
        reduced=self.predict(obs,instruction,seed,skip,d2,category='parity')
        for s in schedules:
            switched=self.sample_condition(prepared,seed,s,category='parity')
            compare('roundtrip_custom_vs_initial',switched,cached_outputs[tuple(s.items())])
            minimal=self.sample_condition(prepared,seed,s,category='measurement')
            compare('minimal_vs_instrumented',minimal,switched)
        after=self.predict(obs,instruction,seed,skip,native,category='parity')
        compare('native_roundtrip',before,after)
        for n,pred in ((native_n,before),(2,reduced)):
            reference=self.native_reference(obs,instruction,seed,skip,n,return_details=True)
            compare(f'upstream_native_D{n}',pred,reference)
            rows[-1]['passed'] &= reference['audit']['actual_NFE']==n
        if self.model_id!='db-cogact':
            uniform=normalize_schedule(self.model_id,dict(kind='flow',m=.5,num_steps=2))
            own=cached_outputs.get(tuple(uniform.items()))
            if own is None: own=self.predict(obs,instruction,seed,skip,uniform,category='parity')
            compare('u50_vs_native_D2',own,reduced)
        if skip and self.model_id!='pi0':
            reference=self.predict(obs,instruction,seed,skip,d2,category='parity',reference=True)
            compare('reference_vs_fast_hidden_bypass',reference,reduced)
        result=dict(passed=all(r['passed'] for r in rows),comparisons=rows,
            actual_predictions=self.calls-initial_calls,skip=list(skip),seed=int(seed),input_hash=prepared['input_hash'],
            coverage=dict(custom_schedules=schedules,native_reference_steps=[native_n,2],roundtrip=True),
            source_hashes=self.code_hashes)
        startup_events=self.pi0.startup_events[startup_start:] if self.model_id=='pi0' else []
        result['startup_events']=startup_events
        result['startup_forward_calls']=sum(e['forward_calls'] for e in startup_events)
        result['startup_action_NFE']=sum(e['action_NFE'] for e in startup_events)
        result['actual_network_forwards_including_startup']=result['actual_predictions']+result['startup_forward_calls']
        if self.output_dir:
            with (self.output_dir/'parity_checks.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
        return result

    def native_reference(self,obs,instruction,seed=7,skip=(),num_steps=None,return_details=False):
        """Bounded acceptance only: execute the unmodified upstream native path."""
        n=num_steps or (5 if self.model_id=='xiaomi-robotic0' else 10)
        torch=self.torch
        captured=[]; actual=[]; handles=[]
        if self.model_id=='pi0':
            self.prepare_condition(obs,instruction,skip)
            backend=self.pi0.backend
            noise=np.random.default_rng(int(seed)).standard_normal((50,32),dtype=np.float32)
            from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
            backend.inputs=prepare_triton_inputs_from_transformed(transformed=backend.transformed,
                device='cuda',action_horizon=50,action_dim=32,noise=noise)
            from scripts.experiment1 import run_intermediate_action_probe as probe_module
            original_layer=probe_module._decoder_layer
            def layer(*args,**kwargs):
                actual.append(args[-1]); return original_layer(*args,**kwargs)
            # Count one eager native record, avoiding CUDA graph warmup accounting.
            probe_module._decoder_layer=layer
            try:
                images,state,initial=backend.inputs
                b=backend.runtime.buffers
                b['observation_images_normalized'].copy_(images)
                b['observation_state_normalized'].copy_(state)
                b['diffusion_noise'].copy_(initial)
                backend.record(tuple(skip),n)
                torch.cuda.synchronize()
                raw=b['diffusion_noise'].float().cpu().numpy().copy()
                actions=np.asarray(backend.output_transform(dict(state=np.asarray(backend.transformed['state']).copy(),actions=raw))['actions'],dtype=np.float32)
            finally: probe_module._decoder_layer=original_layer
            actual_nfe=len(actual)//18
            noise_hash=digest(noise)
        else:
            data,_=self._prepare(obs,instruction,seed)
            self._clear_redundant_class_forwards()
            original_randn,original_like=torch.randn,torch.randn_like
            def capture(fn):
                def call(*args,**kwargs):
                    out=fn(*args,**kwargs); captured.append(out.detach()); return out
                return call
            target=self.model if self.model_id=='xiaomi-robotic0' else self.model.model.action_head.net
            method='dit_forward' if self.model_id=='xiaomi-robotic0' else 'forward_with_cfg'
            original=getattr(target,method)
            def counted(*args,**kwargs):
                actual.append((kwargs.get('t') if 't' in kwargs else args[1]).detach().clone())
                return original(*args,**kwargs)
            setattr(target,method,counted)
            torch.randn,torch.randn_like=capture(original_randn),capture(original_like)
            try:
                with seeded(torch,int(seed)),torch.inference_mode(),self.adapter(self.model,skip):
                    if self.model_id=='xiaomi-robotic0':
                        data['num_steps']=n
                        output=self.model(**data)
                        actions=self.processor.decode_action(output.actions,'libero_all')[0,:,:7].float().cpu().numpy()
                    else:
                        self.model.model.action_head.create_ddim(n)
                        actions=np.asarray(self.model.inference_action(**data,inference_args=dict(
                            cfg_scale=1.5,num_ddim_steps=n,action_norms=self.norm)),dtype=np.float32)
                        actions[:,-1]=np.where(actions[:,-1]<0,-1,1)
            finally:
                torch.randn,torch.randn_like=original_randn,original_like
                delattr(target,method)
            self._clear_redundant_class_forwards()
            noise=captured[0].float().cpu().numpy()
            noise_hash=digest(noise); actual_nfe=len(actual)
        self.calls+=1
        paths={}
        if self.output_dir:
            destination=self.output_dir/'native_references'/f'{self.calls:09d}_{uuid.uuid4().hex}'
            destination.parent.mkdir(parents=True,exist_ok=True)
            noise_path=destination.with_name(destination.name+'_noise.npz')
            prediction_path=destination.with_name(destination.name+'_prediction.npz')
            np.savez_compressed(noise_path,initial_noise=noise)
            np.savez_compressed(prediction_path,actions=actions)
            paths=dict(noise_path=str(noise_path),prediction_path=str(prediction_path))
            with (self.output_dir/'native_reference_ledger.jsonl').open('a') as f:
                f.write(json.dumps(dict(call=self.calls,category='diagnostic_native_reference',num_steps=n,
                    seed=int(seed),skip=list(skip),input_hash=digest(dict(obs=obs,instruction=instruction)),
                    actions_hash=digest(actions),noise_hash=noise_hash,actual_NFE=actual_nfe,source_hashes=self.code_hashes,**paths))+'\n')
        details=dict(actions=actions,noise_hash=noise_hash,audit=dict(actual_NFE=actual_nfe,
            actual_timesteps=[float(t.flatten()[0].cpu()) for t in actual] if self.model_id!='pi0' else None),**paths)
        return details if return_details else actions
