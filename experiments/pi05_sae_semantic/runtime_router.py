"""B1 semantic routing around the original OpenPI sampling/AE/KV path.

Only VLM FFN forwards after L5 are switched. SAE reconstruction is never written
to the policy. The full released encoder/TopK is used before selecting features.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from .check_public_sae import load_official_loader
from .native_probe import sampling_layout
from .select_sae_concepts import concept_label


def select_mask(confidence,rules):
    active=sorted(c for c,value in confidence.items() if value>=rules['early_concepts'][c].get('theta',.8))
    if not active:return (),active,'no_confident_concept'
    eligible=[]
    for candidate in rules['masks']:
        support=candidate['supported_concepts']
        if all(c in support and support[c]['D']<=rules['delta_semantic'] for c in active):
            eligible.append((max(support[c]['D'] for c in active),tuple(candidate['mask'])))
    if not eligible:return (),active,'no_supported_mask'
    return min(eligible)[1],active,'semantic_prior'


class SemanticSkipRuntime:
    def __init__(self,probe,rules,assets,sae_device=None,capture_debug=False):
        assets=Path(assets).resolve()
        self.probe,self.backend,self.model=probe,probe.backend,probe.model
        self.rules=rules;self.capture_debug=capture_debug;self.last_debug=None
        self.sae_device=torch.device(sae_device or self.backend.device)
        self.sae=load_official_loader(assets).load_sae(assets/'saes/per_token/paligemma/sae_layer5.safetensors',device=str(self.sae_device))
        self.sae.requires_grad_(False)
        self.ids=sorted({f['feature_id'] for d in rules['early_concepts'].values() for f in d['features']})
        self.columns={f:i for i,f in enumerate(self.ids)}
        self.original_embed=self.model.embed_prefix
        self.original_mlp={i:probe.lm.layers[i].mlp.forward for i in (6,8,10,12,14,15)}
        self.context=None;self.counters={};self.last_trace=None;self.lock=threading.RLock();self.attached=False
        for entry in rules['masks']:
            m=entry['mask']
            if len(m)!=2 or len(set(m))!=2 or not set(m)<=set(self.original_mlp) or abs(m[0]-m[1])==1:
                raise ValueError('Runtime only supports two distinct nonadjacent candidate FFNs')
        self.attach()

    def attach(self):
        if self.attached:return
        def embed(images,img_masks,lang_tokens,lang_masks):
            ctx=self.context
            if ctx and ctx['needs_sae']:
                embedding=self.model.paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.position_embedding
                expected=len(images)*embedding.weight.shape[0]+lang_tokens.shape[1]
                prepared=SimpleNamespace(images=images,image_masks=img_masks,tokens=lang_tokens,token_masks=lang_masks)
                ctx['layout']=sampling_layout(prepared,ctx['instruction'],expected,self.probe.tokenizer,self.model.config.discrete_state_input)
            output=self.original_embed(images,img_masks,lang_tokens,lang_masks)
            if ctx and ctx['needs_sae'] and output[0].shape[1]!=ctx['layout']['sequence_length']:
                raise ValueError('Runtime vision token layout changed')
            return output
        self.model.embed_prefix=embed
        self.handle=self.probe.lm.layers[5].register_forward_hook(self._at_l5)
        self.attached=True

    def detach(self):
        if not self.attached:return
        self._restore();self.handle.remove();self.model.embed_prefix=self.original_embed;self.attached=False

    def _restore(self):
        for i,forward in self.original_mlp.items():self.probe.lm.layers[i].mlp.forward=forward

    def reset_episode(self,key,start_calls=0):self.counters[key]=start_calls

    def _scores(self,hidden,layout):
        sample=hidden[0,layout['selected_positions']].detach()
        if self.capture_debug:self.last_debug=dict(samples=sample.cpu().clone(),layout=layout)
        x=sample.to(self.sae_device,dtype=torch.float32)
        previous_tf32=None
        if self.sae_device.type=='cuda':
            previous_tf32=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:
            with torch.inference_mode():z=self.sae.encode(x)
            pool=self.rules.get('token_pool','all')
            indexes=[i for i,k in enumerate(layout['selected_kinds'])
                     if pool=='all' or (pool=='instruction' and k=='instruction') or (pool=='image' and k.startswith('camera_'))]
            a=z[indexes][:,self.ids].max(0).values.detach().float().cpu().numpy()
        finally:
            if previous_tf32 is not None:torch.backends.cuda.matmul.allow_tf32=previous_tf32
        scores={}
        for name,definition in self.rules['early_concepts'].items():
            features=definition['features'];values=a[[self.columns[f['feature_id']] for f in features]]
            u=(values-np.array([f['tau'] for f in features]))/np.array([f['b'] for f in features])
            scores[name]=float(np.clip(u,0,1) @ np.array([f['weight'] for f in features]))
        return scores

    def _at_l5(self,module,args,output):
        ctx=self.context
        if ctx is None:return
        ctx['seen']+=1
        if ctx['seen']!=1:raise ValueError('Unexpected multiple VLM prefixes in one policy call')
        mode=ctx['mode'];mask=();reason='full';active=[];confidence={}
        if mode in ('sae_route','sae_only') and ctx['needs_sae']:
            started=time.perf_counter();confidence=self._scores(output[0],ctx['layout'])
            ctx['sae_router_ms']=(time.perf_counter()-started)*1000
            mask,active,reason=select_mask(confidence,self.rules)
        elif mode=='prompt':
            confidence={c:float(concept_label(ctx['instruction'],*c.split('/',1))) for c in self.rules['early_concepts']}
            mask,active,reason=select_mask(confidence,self.rules)
        elif mode=='fixed':
            if (ctx['request']+self.rules.get('static_offset',0))%self.rules.get('static_period',1)==0:
                mask=tuple(self.rules['static_mask']);reason='fixed_schedule'
            else:reason='fixed_full_call'
        if mode in ('full','sae_only'):mask=();reason='full' if mode=='full' else 'readout_only'
        elif ctx['refresh']:mask=();reason='first_full_calls' if ctx['request']<2 else 'periodic_full_call'
        for i in mask:self.probe.lm.layers[i].mlp.forward=lambda x:torch.zeros_like(x)
        ctx.update(mask=list(mask),reason=reason,active_concepts=active,confidence=confidence)

    def infer_prepared(self,prepared,instruction,noise,episode_key,mode='sae_route'):
        if mode not in ('full','sae_only','sae_route','fixed','prompt'):raise ValueError(mode)
        with self.lock:
            if self.context is not None:raise RuntimeError('B1 routing context is already in use')
            request=self.counters.get(episode_key,0)
            refresh=request<self.rules.get('first_full_calls',2) or (request+1)%self.rules.get('periodic_full_every_calls',10)==0
            ctx=dict(instruction=instruction,request=request,mode=mode,refresh=refresh,seen=0,
                     needs_sae=mode=='sae_only' or (mode=='sae_route' and not refresh),sae_router_ms=0.)
            self.context=ctx;started=time.perf_counter()
            try:
                raw=self.backend.original(prepared,noise)
                decoded=self.backend.output(prepared,raw)
                if ctx['seen']!=1:raise RuntimeError('L5 hook was not visible in native action generation')
                self.counters[episode_key]=request+1
                self.last_trace={k:v for k,v in ctx.items() if k not in ('layout','instruction','needs_sae')}
                self.last_trace.update(policy_call=request+1,mode=mode,policy_ms=(time.perf_counter()-started)*1000)
                return raw,decoded,self.last_trace.copy()
            finally:
                self._restore();self.context=None
