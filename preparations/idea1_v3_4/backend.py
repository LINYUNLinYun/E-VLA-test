"""Native KV prefix truncation and explicit warm-start Euler; upstream untouched."""
import time
import numpy as np
from preparations.idea1_v3_3.backends import PiBackend


def prefix_encoder(k, w, b, seq, stop, kept):
    """Materialize only L1..L(stop-1) KV. The final hidden output is unused."""
    if not 1 <= stop <= 19:
        raise ValueError('1-based first old layer must be in [1,19]')
    k.layer_norm_matmul_n256_1152_2048_bias(
        b['vision_x'], w['vision_final_norm_w'], w['vision_final_norm_b'],
        w['encoder_multi_modal_projector_w'], w['encoder_multi_modal_projector_b'],
        b['encoder_x'], b['vision_x_norm'])
    for i in range(stop-1):
        project = k.rms_matmul_n_2048_2560_qkv_rope if i in kept else k.rms_matmul_n_2048_2560_kv_rope
        project(b['encoder_x'], w['encoder_attn_qkv_w'][i], b['encoder_rope_weights'],
                *([b['encoder_Q']] if i in kept else []),
                b['encoder_K'][i, :seq], b['encoder_V'][i, :seq], b['encoder_x_norm'])
        if i in kept and i < min(17, stop-2):
            attn = k.AttnSingleKey(b['encoder_Q'], b['encoder_K'][i,:seq], b['encoder_V'][i,:seq], 256**-.5)
            k.matmul_n_2048_2048_res(attn, w['encoder_attn_o_w'][i], b['encoder_x'])
            k.rms_matmul_n_2048_16384_gate(b['encoder_x'], w['encoder_ffn_gate_w'][i],
                w['encoder_ffn_up_w'][i], b['encoder_hidden'], b['encoder_x_norm'])
            k.matmul_n_16384_2048_res(b['encoder_hidden'], w['encoder_ffn_down_w'][i], b['encoder_x'])


class Backend(PiBackend):
    def prepare_hybrid(self, obs, instruction, full, cutoff=16):
        from scripts.spec.triton import pi0_infer as k
        from preparations.idea1_v3.policy import V14
        original = k.transformer_encoder
        def truncated(w, b, seq, **kwargs):
            prefix_encoder(k, w, b, seq, cutoff, frozenset(i for i in range(18) if i+1 not in V14))
        k.transformer_encoder = truncated
        try:
            c, timing = self.prepare(obs, instruction, V14)
        finally:
            k.transformer_encoder = original
        if c['seq'] != full['seq']:
            raise ValueError('Hybrid prompt/token shape mismatch')
        for key in ('K','V'):
            c[key][cutoff-1:].copy_(full[key][cutoff-1:])
        timing.update(current_kv_layers=cutoff-1, old_kv_layers=19-cutoff,
                      current_hidden_updates=sum(i+1 not in V14 for i in range(max(0,cutoff-2))))
        return c, timing

    def warm_sample(self, c, initial, noise_seed, q=.3, steps=2):
        """Exact specified q->0 Euler on the native AE, with BF16 input."""
        from scripts.experiment1.run_intermediate_action_probe import _decoder_layer
        from scripts.spec.triton import pi0_infer as k
        torch = self.torch
        if not 0 < q <= 1 or steps < 1:
            raise ValueError('Invalid refinement schedule')
        w, b, seq = c['runtime'].weights, c['runtime'].buffers, c['seq']
        times=[q*(1-i/steps) for i in range(steps)]
        nodes=dict(nodes=times+[0.],timesteps=times,dt=[-q/steps]*steps,NFE=steps)
        biases, out_w, out_b, _ = self.adapter.schedule(c['runtime'], nodes)
        noise=np.random.default_rng(noise_seed).standard_normal((50,32),dtype=np.float32)
        initial=torch.as_tensor(initial,device='cuda',dtype=torch.float32)
        z=((1-q)*initial+q*torch.as_tensor(noise,device='cuda')).to(torch.bfloat16)
        b['observation_state_normalized'].copy_(c['state'])
        b['encoder_K'][:,:seq].copy_(c['K']); b['encoder_V'][:,:seq].copy_(c['V'])
        b['diffusion_noise'].copy_(z)
        def run():
            k.matmul_1_32_1024_bias(b['observation_state_normalized'],w['decoder_state_in_proj_w'],w['decoder_state_in_proj_b'],b['decoder_state_buf'])
            for i in range(steps):
                b['decoder_x'][:1].copy_(b['decoder_state_buf'])
                k.matmul_k_32_1024_bias_silu(b['diffusion_noise'],w['decoder_action_fused_in_proj_w'],biases[i],b['decoder_x_buf'])
                k.matmul_k_1024_1024_bias(b['decoder_x_buf'],w['decoder_action_mlp_w'],w['decoder_action_mlp_b'],b['decoder_x'][1:])
                for layer in range(18): _decoder_layer(w,b,seq,layer)
                k.rms_matmul_k_1024_32_bias_res(b['decoder_x'][1:],out_w[i],out_b[i],b['diffusion_noise'],b['decoder_norm_factor_buf'])
            return b['diffusion_noise'].clone()
        raw,ms=self.adapter.measure(run)
        tick=time.perf_counter(); actions=self.decode(c,raw)
        return actions,raw,dict(ae_ms=ms,decode_ms=(time.perf_counter()-tick)*1000,nodes=nodes['nodes'],NFE=steps)


def shifted_tail(raw, executed):
    a=np.asarray(raw)
    if a.shape!=(50,32) or not 0 < executed < 50:
        raise ValueError('Expected H50/raw32 and actual prefix length')
    return np.concatenate([a[executed:],np.repeat(a[-1:],executed,axis=0)])
