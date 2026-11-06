"""Native eager AE kernels with a supplied initial tensor; no AE CUDA Graph."""
import time
import numpy as np


def sample_initial(backend, context, initial, steps):
    from preparations.idea1_v2_2.schedules import flow_schedule
    from scripts.experiment1.run_intermediate_action_probe import _decoder_layer
    from scripts.spec.triton import pi0_infer as kernels
    adapter = backend.adapter
    torch = adapter.torch
    runtime = context['runtime']
    w, buffers = runtime.weights, runtime.buffers
    seq = context['seq']
    initial = np.asarray(initial, dtype=np.float32)
    if initial.shape != (50, 32) or not np.isfinite(initial).all():
        raise ValueError('Finite H50 x D32 initial state required')
    nodes = flow_schedule('pi0', None, steps)
    biases, out_w, out_b, _ = adapter.schedule(runtime, nodes)
    initial_tensor = torch.as_tensor(initial, device='cuda', dtype=torch.bfloat16)
    buffers['observation_state_normalized'].copy_(context['state'])
    buffers['encoder_K'][:, :seq].copy_(context['K'])
    buffers['encoder_V'][:, :seq].copy_(context['V'])
    buffers['diffusion_noise'].copy_(initial_tensor)

    def run():
        kernels.matmul_1_32_1024_bias(buffers['observation_state_normalized'],
            w['decoder_state_in_proj_w'], w['decoder_state_in_proj_b'], buffers['decoder_state_buf'])
        for i in range(nodes['NFE']):
            buffers['decoder_x'][:1].copy_(buffers['decoder_state_buf'])
            kernels.matmul_k_32_1024_bias_silu(buffers['diffusion_noise'],
                w['decoder_action_fused_in_proj_w'], biases[i], buffers['decoder_x_buf'])
            kernels.matmul_k_1024_1024_bias(buffers['decoder_x_buf'],
                w['decoder_action_mlp_w'], w['decoder_action_mlp_b'], buffers['decoder_x'][1:])
            for layer in range(18):
                _decoder_layer(w, buffers, seq, layer)
            kernels.rms_matmul_k_1024_32_bias_res(buffers['decoder_x'][1:],
                out_w[i], out_b[i], buffers['diffusion_noise'], buffers['decoder_norm_factor_buf'])
        return buffers['diffusion_noise']

    raw, ae_ms = adapter.measure(run)
    raw = raw.clone()
    tick = time.perf_counter()
    actions = np.asarray(adapter.backend.output_transform(dict(
        state=context['transformed_state'].copy(), actions=raw.float().cpu().numpy().copy()))['actions'], np.float32)
    return actions, raw, dict(ae_ms=ae_ms, decode_ms=(time.perf_counter()-tick)*1000,
        nodes=dict(nodes, actual_NFE=steps), initial_state='kalman', native_ae_graph=False)
