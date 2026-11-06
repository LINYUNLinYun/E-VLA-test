"""CPU-only execution tests for Attention and FFN intervention semantics."""
import contextlib

import pytest
import torch

from scripts.spec.triton import pi0_infer as infer
from scripts.spec.triton.triton_pi0_runtime import TritonRuntimeSession


@pytest.mark.parametrize('skip_attn,skip_ffn,expected',[((),(),51),((0,),(),50),((),(0,),49),((0,),(0,),48)])
def test_independent_submodule_updates_and_kv(monkeypatch,skip_attn,skip_ffn,expected):
    calls=[]
    monkeypatch.setattr(infer,'nvtx_range',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(infer,'layer_norm_matmul_n256_1152_2048_bias',lambda *a:None)
    monkeypatch.setattr(infer,'rms_matmul_n_2048_2560_qkv_rope',lambda *a:calls.append('qkv'))
    monkeypatch.setattr(infer,'rms_matmul_n_2048_2560_kv_rope',lambda *a:calls.append('kv'))
    monkeypatch.setattr(infer,'AttnSingleKey',lambda *a:torch.zeros(1))
    monkeypatch.setattr(infer,'matmul_n_2048_2048_res',lambda a,w,out:out.add_(1))
    monkeypatch.setattr(infer,'rms_matmul_n_2048_16384_gate',lambda *a:None)
    monkeypatch.setattr(infer,'matmul_n_16384_2048_res',lambda a,w,out:out.add_(2))
    weights={name:torch.zeros(18) for name in ('encoder_attn_qkv_w','encoder_attn_o_w','encoder_ffn_gate_w','encoder_ffn_up_w','encoder_ffn_down_w')}
    weights.update({name:None for name in ('vision_final_norm_w','vision_final_norm_b','encoder_multi_modal_projector_w','encoder_multi_modal_projector_b')})
    buffers={name:torch.zeros(1,1) for name in ('vision_x','vision_x_norm','encoder_x','encoder_x_norm','encoder_Q','encoder_rope_weights','encoder_hidden')}
    buffers.update(encoder_K=torch.zeros(18,1,1),encoder_V=torch.zeros(18,1,1))
    traces=[]
    infer.transformer_encoder(weights,buffers,1,skip_attn_layers=skip_attn,skip_ffn_layers=skip_ffn,submodule_trace=traces)
    assert buffers['encoder_x'].item()==expected
    assert calls==['qkv']*18  # Retain original K/V, including when attention is skipped.
    assert len(traces)==17
    assert traces[0][0].item()==0  # Snapshots cannot alias mutated encoder buffer.
    assert traces[0][1].item()==(0 if skip_attn else 1)
    assert traces[0][2].item()==(0 if skip_attn else 1)+(0 if skip_ffn else 2)


def test_kv_only_layer_cannot_be_submodule_candidate():
    with pytest.raises(ValueError,match='KV-only'):
        infer.transformer_encoder({}, {}, 1, skip_ffn_layers=(17,))


def test_session_passes_masks_and_trace_without_full_graph():
    observed={}
    class FakeRuntime:
        def run_layer_probe(self, images, state, noise, **kwargs):
            observed.update(kwargs)
            kwargs['submodule_trace'].append('snapshot')
            return noise.clone(),None
    session=TritonRuntimeSession(prompt='test',runtime=FakeRuntime())
    prepared=session.prepare_observation(images=torch.zeros(2,1),state=torch.zeros(1))
    trace=[]
    actions=session.run_submodule_probe(prepared=prepared,noise=torch.ones(1),
        skip_attn_layers=(2,),skip_ffn_layers=(4,),submodule_trace=trace)
    assert actions.item()==1
    assert observed['skip_attn_layers']==(2,) and observed['skip_ffn_layers']==(4,)
    assert observed['capture_hidden'] is False and trace==['snapshot']
