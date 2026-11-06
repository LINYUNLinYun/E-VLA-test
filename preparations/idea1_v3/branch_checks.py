"""CPU branch checks: command re-anchoring/H50 and tiny native JAX KV masks."""
import argparse
import json
import numpy as np


def history_check():
    from preparations.idea1_v3.history import DecoderCoordinates,make_history_chunk,HistoryPolicy
    from preparations.idea1_v3.checks import Backend
    scale=np.array([.1,.2,.3,.4,.5,.6,.7],np.float32)
    def transform(data):
        # Same affine structure as Unnormalize -> AbsoluteActions -> first7.
        return {'actions':data['actions'][:,:7]*scale+data['state'][:7]}
    old=np.arange(32,dtype=np.float32); current=old+2
    coordinates=DecoderCoordinates(transform,old)
    raw=np.linspace(-1,1,1600,dtype=np.float32).reshape(50,32)
    commands=transform(dict(state=current,actions=raw))['actions']
    historical=transform(dict(state=old,actions=raw[:3]))['actions']
    for r in range(1,6):
        historical=transform(dict(state=old,actions=raw[:r]))['actions']
        chunk,future,actual_r=make_history_chunk(historical,raw[:r],commands,raw,coordinates)
        assert chunk.shape==(50,32) and actual_r==r
        np.testing.assert_allclose(coordinates.decode(chunk)[r:],commands[:50-r],atol=2e-5)
        np.testing.assert_allclose(coordinates.decode(chunk)[:r],historical,atol=2e-5)
        assert not np.allclose(future[:,:6],raw[:50-r,:6])
    class HistoryFake(Backend):
        def verify_history(self,full,current,raw,seed,suite,previous):
            self.last_history=previous
            hats=np.stack([raw[:5].copy(),raw[:5].copy()])
            if self.reject: hats[:,:,:6]+=1
            return hats,raw[:5],{},dict(history_length=len(previous['commands']),anchor_step=previous['step'])
    backend=HistoryFake(); policy=HistoryPolicy(dict(experiment='E3',draft_steps=2,tau=.15,period=2),backend)
    first=policy.predict(1,'test','libero_goal',123); policy.acknowledge(first['actions'][:3])
    assert len(policy.history['commands'])==3 and policy.history['step']==0
    backend.reject=True; result=policy.predict(2,'test','libero_goal',123)
    assert result['route']['full_reason']=='zero_accept' and policy.step==3
    assert len(policy.history['commands'])==3 and policy.history['condition']['state']==1
    policy.acknowledge(result['actions'][:2])
    assert len(policy.history['commands'])==2 and policy.history['condition']['state']==2 and policy.history['step']==3
    policy.reset(); assert policy.history is None
    first=policy.predict(1,'test','libero_goal',123)
    first['actions'][0,0]=1.4
    policy.acknowledge(first['actions'][:1])
    assert policy.history['commands'][0,0]==1. and policy.history['sent_commands'][0,0]>1.
    print('history coordinate roundtrip/H50/r1..5/rejected draft excluded/reset: OK')


def native_history_check():
    from preparations.idea1_v3.history import DecoderCoordinates
    from scripts.spec.spec_serve_policy import _build_policy_components
    _,_,transform=_build_policy_components(config_name='pi0_libero',
        checkpoint_dir='shared/checkpoints/realtime-vla-flash/converted/base')
    coordinates=DecoderCoordinates(transform,np.linspace(-.4,.5,32,dtype=np.float32))
    raw=np.linspace(-.8,.7,1600,dtype=np.float32).reshape(50,32)
    commands=coordinates.decode(raw)
    inverse=coordinates.encode_commands(commands,raw)
    np.testing.assert_allclose(inverse[:,:7],raw[:,:7],atol=2e-5,rtol=2e-5)
    print('actual released pi0_libero output-transform affine inverse: OK')


def pi05_check():
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from openpi.models import gemma
    from preparations.idea1_v3.pi05 import install_vlm_mask
    config=gemma.Config(width=8,depth=18,mlp_dim=16,num_heads=2,num_kv_heads=1,head_dim=4)
    model=gemma.Module(configs=[config,config],embed_dtype='float32')
    inputs=[jnp.arange(24,dtype=jnp.float32).reshape(1,3,8)*.01,None]
    positions=jnp.arange(3)[None,:]; mask=jnp.ones((1,3,3),dtype=bool)
    variables=nn.Module.init(model,jax.random.key(1),[inputs[0],jnp.ones((1,2,8),jnp.float32)],
        jnp.arange(5)[None,:],jnp.ones((1,5,5),dtype=bool),method='__call__')
    original=install_vlm_mask((3,4,9,11))
    outputs,cache=model.apply(variables,inputs,positions,mask)
    assert cache[0].shape==(18,1,3,1,4) and np.isfinite(np.asarray(outputs[0])).all()
    # First untouched layer has the exact same KV as native Full. Later layers
    # may differ because skipped hidden updates change their input.
    gemma.Module.__call__=original
    _,full_cache=model.apply(variables,inputs,positions,mask)
    np.testing.assert_array_equal(cache[0][0],full_cache[0][0])
    suffix=[None,jnp.ones((1,2,8),jnp.float32)*.02]
    suffix_positions=jnp.array([[3,4]]); suffix_mask=jnp.ones((1,2,5),dtype=bool)
    original_out,original_kv=model.apply(variables,suffix,suffix_positions,suffix_mask,kv_cache=cache)
    install_vlm_mask((3,4,9,11))
    skipped_out,skipped_kv=model.apply(variables,suffix,suffix_positions,suffix_mask,kv_cache=cache)
    np.testing.assert_allclose(original_out[1],skipped_out[1],rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(original_kv[0],skipped_kv[0],rtol=1e-5,atol=1e-5)
    # Benchmark v2 dispatch: actual original method, while the global prefix
    # override remains installed. It must trace a scan and leave the mask alone.
    current=gemma.Module.__call__
    scanned_out,scanned_kv=model.apply(variables,suffix,suffix_positions,suffix_mask,kv_cache=cache,
        method=gemma.Module._v3_original_call)
    np.testing.assert_array_equal(original_out[1],scanned_out[1])
    np.testing.assert_array_equal(original_kv[0],scanned_kv[0])
    again,again_cache=model.apply(variables,inputs,positions,mask)
    np.testing.assert_array_equal(outputs[0],again[0]); np.testing.assert_array_equal(cache[0],again_cache[0])
    assert gemma.Module.__call__ is current
    traced=jax.make_jaxpr(lambda tokens:model.apply(variables,[None,tokens],suffix_positions,suffix_mask,
        kv_cache=cache,method=gemma.Module._v3_original_call)[0][1])(suffix[1])
    assert 'scan[' in str(traced), 'v2 suffix must use original native scanned implementation'
    print('bench v2 explicit original scanned suffix: exact native AE parity / prefix mask unchanged / scan trace: OK')
    print('tiny pi05-compatible JAX:18-layer shell shape/prefix layer0/native full AE parity: OK')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--native-history',action='store_true'); parser.add_argument('--pi05',action='store_true')
    args=parser.parse_args(); history_check()
    if args.native_history: native_history_check()
    if args.pi05: pi05_check()
