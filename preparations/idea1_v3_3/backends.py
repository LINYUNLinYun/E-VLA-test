"""Isolated native backend adaptations. No checkpoint or upstream edits."""
import os
import copy
import time
import numpy as np
from preparations.idea1_v3.policy import TritonBackend, verification_seed
from preparations.idea1_v3.history import DecoderCoordinates


class PiBackend(TritonBackend):
    def sample(self,c,seed,steps):
        import hashlib
        self.last_noise_hash=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest()
        return super().sample(c,seed,steps)

    def prepare_metadata(self, obs, instruction):
        # Native transforms/prompt runtime only: suppress all visual/model calls.
        from scripts.spec.triton import pi0_infer as k
        from preparations.idea1_v2_1.runtime_models import native_observation
        images,state=native_observation(obs)
        observation={'observation/image':np.asarray(images[0]),'observation/wrist_image':np.asarray(images[1]),'observation/state':state,'prompt':instruction}
        original=k.Pi0Inference.record_infer_graph
        k.Pi0Inference.record_infer_graph=lambda runtime:None
        try:
            self.adapter.backend.prepare(observation,dict(suite=self.adapter.prompt_suites[instruction],task_description=instruction),np.zeros((50,32),np.float32))
        finally: k.Pi0Inference.record_infer_graph=original
        r=self.adapter.backend.runtime
        return dict(runtime=r,transformed_state=np.asarray(self.adapter.backend.transformed['state']).copy(),state=self.adapter.backend.inputs[1].clone(),seq=r.buffers['encoder_x'].shape[0],startup_events=[])

    def encode_commands(self,c,commands):
        coordinates=DecoderCoordinates(self.adapter.backend.output_transform,c['transformed_state'])
        raw=coordinates.encode_commands(commands,np.zeros((50,32),np.float32))
        return self.torch.as_tensor(raw,device='cuda',dtype=self.torch.float32)


class XiaomiBackend:
    def __init__(self):
        from preparations.idea1_v2_2.runtime_models import NativePolicy
        self.native=NativePolicy('xiaomi-robotic0',output_dir=None)
        self.torch=self.native.torch
        self.protocol=dict(self.native.protocol,H=10,R=10,D=2,verification='native forward time 1-q; separate AE branches')
        self.startup_events=[]

    def prepare(self,obs,instruction,skip=()):
        entry=self.native.prepare_condition(obs,instruction,skip=skip)
        # prepare_condition returns fresh native past_key_values. Clone every
        # tensor so future scratch/cache operations cannot overwrite Full KV.
        c=entry['condition']
        def clone(value):
            if self.torch.is_tensor(value): return value.detach().clone()
            if isinstance(value,tuple): return tuple(clone(x) for x in value)
            if isinstance(value,list): return [clone(x) for x in value]
            return value
        c['vlm'].past_key_values=clone(copy.deepcopy(c['vlm'].past_key_values))
        return c,entry['timing']

    def sample(self,c,seed,steps):
        import hashlib
        with self.torch.inference_mode():
            actions,raw,noise,audit,ms,decode=self.native._xiaomi_sample(c,seed,dict(kind='native',m=None,num_steps=steps),instrumented=False)
        self.last_noise_hash=hashlib.sha256(noise.float().cpu().numpy().tobytes()).hexdigest()
        return actions,raw[0].detach().clone(),dict(ae_ms=ms,decode_ms=decode,NFE=steps)

    def verify(self,full,current,raw,seed,suite):
        torch=self.torch; model=self.native.model
        mask,state,out=current['action_mask'],current['state'],full['vlm']
        noise=torch.as_tensor(np.random.default_rng(verification_seed(seed)).standard_normal(tuple(raw.shape),dtype=np.float32),device=raw.device,dtype=raw.dtype)[None]
        def run():
            bs,length,_=mask.shape; query=length+state.shape[1]+1
            ids=torch.arange(query,device=mask.device).view(1,1,-1).repeat(3,bs,1)+out.position_ids.max(dim=-1)[0][...,None]+1
            pos=model.rotary_emb(mask,ids)
            dit_mask=torch.tril(torch.ones((bs,query,query),device=mask.device))
            cache_mask=out.attention_mask[:,None,:].expand(-1,query,-1)
            attn=torch.cat([cache_mask,dit_mask],dim=-1)[:,None].bool()
            state_embed=model.state_projector(state)
            hats=[]
            for q in (.10,.05):
                z=(1-q)*raw[None]+q*noise
                ts=torch.ones((bs,1,1),device=z.device,dtype=z.dtype)*(1-q)
                v=model.dit_forward(z,ts,mask,state_embed,pos,out.past_key_values,attn)
                hats.append((z+q*v)[0])
            return torch.stack(hats)
        with torch.inference_mode(): hats,ms=self.native._measure(run)
        return hats.float().cpu().numpy(),dict(verify_ae_ms=ms)
