"""CPU-only supplement checks; JAX_PLATFORM_NAME=cpu is required."""
import os
import numpy as np


def dispatch_check():
    import tempfile
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace
    from preparations.idea1_v3 import branches
    from preparations.idea1_v3_supplement import cli
    original_worker=branches.worker; original_popen=subprocess.Popen; commands=[]
    def fake_worker(args):
        subprocess.Popen(['python','-m','preparations.idea1_v3.branch_server','--config','{}'])
    def fake_popen(command,*a,**kw): commands.append(command)
    branches.worker=fake_worker; subprocess.Popen=fake_popen
    try:
        with tempfile.TemporaryDirectory() as directory:
            cli.worker(SimpleNamespace(attempt=Path(directory),shard=0))
            assert commands[0][2]=='preparations.idea1_v3_supplement.server'
    finally: branches.worker=original_worker; subprocess.Popen=original_popen
    print('PASS CPU dispatch: supplement child model server, no GPU/process started')


def pi05_check():
    assert os.environ.get('JAX_PLATFORM_NAME')=='cpu'
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from openpi.models import gemma
    from preparations.idea1_v3_supplement.pi05 import install,validate_skip
    for mask in ((3,4,9,11),(3,4,5,9,11)): assert validate_skip(mask)==mask
    for bad in ((1,3,4,5),(3,3,4,5),(3,4,5)):
        try: validate_skip(bad)
        except ValueError: pass
        else: raise AssertionError('invalid mask accepted')
    cfg=gemma.Config(width=8,depth=18,mlp_dim=16,num_heads=2,num_kv_heads=1,head_dim=4)
    model=gemma.Module(configs=[cfg,cfg],embed_dtype='float32')
    x=jnp.arange(24,dtype=jnp.float32).reshape(1,3,8)*.01
    variables=nn.Module.init(model,jax.random.key(1),[x,jnp.ones((1,2,8))],jnp.arange(5)[None,:],jnp.ones((1,5,5),bool),method='__call__')
    for skip in ((3,4,9,11),(3,4,5,9,11)):
        original=install(skip)
        out,cache=model.apply(variables,[x,None],jnp.arange(3)[None,:],jnp.ones((1,3,3),bool))
        assert cache[0].shape==(18,1,3,1,4)
        prefix=gemma.Module.__call__
        suffix=[None,jnp.ones((1,2,8))*.02]; positions=jnp.array([[3,4]]); mask=jnp.ones((1,2,5),bool)
        actual=model.apply(variables,suffix,positions,mask,kv_cache=cache)
        expected=model.apply(variables,suffix,positions,mask,kv_cache=cache,method=original)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)): np.testing.assert_array_equal(a,b)
        trace=jax.make_jaxpr(lambda t:model.apply(variables,[None,t],positions,mask,kv_cache=cache)[0][1])(suffix[1])
        assert 'scan[' in str(trace)
        again,againcache=model.apply(variables,[x,None],jnp.arange(3)[None,:],jnp.ones((1,3,3),bool))
        np.testing.assert_array_equal(out[0],again[0]); np.testing.assert_array_equal(cache[0],againcache[0])
        assert gemma.Module.__call__ is prefix
    print('PASS CPU tiny18: explicit4/5+contiguous masks, native scanned suffix exact parity, prefix/KV stable')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--dispatch-only',action='store_true'); args=p.parse_args()
    dispatch_check()
    if not args.dispatch_only: pi05_check()
