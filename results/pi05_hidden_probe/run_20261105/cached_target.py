"""A patched-prefill KV implementation, validated against full recomputation.

Only generation changes. Candidate scoring retains the reference implementation.
Every cache is private to one generation call and contains KVs from the patched
prefill; earlier-layer placeholder KVs remain exactly as in the original method.
"""
import hashlib
from pathlib import Path
import numpy as np
from experiments.pi05_hidden_probe.hf_backend import HFBackend


class CachedHFBackend(HFBackend):
    def _cached_prefill(self,prefix,patch=None):
        t=self.torch
        self._validate_patch(prefix,patch)
        raw=t.tensor(prefix.metadata['raw_embeddings'],device=self.device,dtype=self.dtype)[None]
        scaled=t.tensor(prefix.embedded,device=self.device,dtype=self.dtype)[None]
        values=None if patch is None else t.tensor(patch.values,device=self.device,dtype=self.dtype)
        mask,positions,readout=self._layout(prefix,[])
        handles=[]
        def first(module,args):
            hidden=scaled.clone()
            if patch is not None and patch.layer==0:hidden[:,patch.positions]=values[None]
            return (hidden,*args[1:])
        def after(module,args,output):
            hidden=output[0].clone();hidden[:,patch.positions]=values[None]
            return (hidden,*output[1:])
        handles.append(self._lm.layers[0].register_forward_pre_hook(first))
        if patch is not None and patch.layer>0:
            handles.append(self._lm.layers[patch.layer-1].register_forward_hook(after))
        try:
            with t.inference_mode():
                output=self._lm(inputs_embeds=raw,attention_mask=mask,position_ids=positions,use_cache=True,return_dict=True)
                logits=self._model.lm_head(output.last_hidden_state[:,readout])
        finally:
            for handle in handles:handle.remove()
        if output.past_key_values is None:raise RuntimeError('HF did not return the patched prefill cache')
        return logits,output.past_key_values

    def _cached_advance(self,prefix,cache,answer):
        t=self.torch
        valid=t.as_tensor(np.r_[prefix.input_mask,np.ones(len(answer),bool)],device=self.device,dtype=t.bool)
        attention=t.where(valid,t.zeros((),device=self.device,dtype=self.dtype),
                          t.full((),t.finfo(self.dtype).min,device=self.device,dtype=self.dtype))[None,None,None]
        with t.inference_mode():
            output=self._lm(input_ids=t.tensor([[answer[-1]]],device=self.device),attention_mask=attention,
                position_ids=t.tensor([[int(prefix.input_mask.sum())+len(answer)]],device=self.device),
                cache_position=t.tensor([len(prefix.input_mask)+len(answer)-1],device=self.device),
                past_key_values=cache,use_cache=True,return_dict=True)
            logits=self._model.lm_head(output.last_hidden_state[:,-1])
        return logits,output.past_key_values

    def generate(self,prefix,patch=None,max_new_tokens=16):
        if max_new_tokens<1:raise ValueError('max_new_tokens must be positive')
        logits,cache=self._cached_prefill(prefix,patch)
        answer=[]
        while len(answer)<max_new_tokens:
            token=int(self.torch.argmax(logits[0]).item());answer.append(token)
            if token==self._tokenizer.eos_token_id:break
            if len(answer)<max_new_tokens:logits,cache=self._cached_advance(prefix,cache,answer)
        return {'text':self._tokenizer.decode(answer,skip_special_tokens=True),'token_ids':answer,
                'eos':bool(answer[-1]==self._tokenizer.eos_token_id)}

    def audit(self,prefix=None):
        return {**super().audit(prefix),'generation':'private patched-prefill KV cache; standard greedy causal answer decoding',
                'generation_implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'candidate_scoring':'reference full recomputation, unchanged'}
