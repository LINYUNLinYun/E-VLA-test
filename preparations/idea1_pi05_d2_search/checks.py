"""CPU-only checks of actual native Euler scheduling, scan/KV and dispatch."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np


def scheduling():
    import jax
    import jax.numpy as jnp
    from openpi.models import pi0
    observed=[]
    class Fake:
        action_horizon=10; action_dim=2
        def embed_prefix(self,observation): return jnp.ones((1,3,2)),jnp.ones((1,3),bool),jnp.zeros((3,),bool)
        def embed_suffix(self,observation,x,time):
            jax.debug.callback(lambda t:observed.append(float(np.asarray(t)[0])),time,ordered=True)
            return jnp.broadcast_to(time[:,None,None],(1,10,2)),jnp.ones((1,10),bool),jnp.ones((10,),bool),None
        def action_out_proj(self,x): return x
    fake=Fake(); fake.PaliGemma=SimpleNamespace(llm=lambda embedded,**kwargs:(embedded,()))
    original=pi0._model.preprocess_observation; pi0._model.preprocess_observation=lambda key,obs,train:obs
    try:
        actual=pi0.Pi0.sample_actions(fake,jax.random.key(0),SimpleNamespace(state=jnp.zeros((1,2))),num_steps=2,noise=jnp.zeros((1,10,2)))
        jax.block_until_ready(actual); jax.effects_barrier()
    finally: pi0._model.preprocess_observation=original
    assert observed==[1.,.5],observed
    np.testing.assert_array_equal(actual,np.full((1,10,2),-.75,np.float32))
    print('PASS actual original Pi0.sample_actions: runtime t=[1,.5], dt=-.5, endpoint0, exactly two suffix evaluations')


def tiny():
    import jax
    import jax.numpy as jnp
    import flax.linen as nn
    from openpi.models import gemma
    from preparations.idea1_pi05_d2_search.policy import validate_skip,install
    for skip in ([],[3,9],list(range(3,19))): assert len(validate_skip(skip))==len(skip)
    for skip in ([1],[2],[3,3],[19]):
        try: validate_skip(skip)
        except ValueError: pass
        else: raise AssertionError('invalid mask accepted')
    cfg=gemma.Config(width=8,depth=18,mlp_dim=16,num_heads=2,num_kv_heads=1,head_dim=4)
    model=gemma.Module(configs=[cfg,cfg],embed_dtype='float32'); x=jnp.ones((1,3,8))*.01
    variables=nn.Module.init(model,jax.random.key(1),[x,jnp.ones((1,2,8))],jnp.arange(5)[None,:],jnp.ones((1,5,5),bool),method='__call__')
    for skip in ((),(3,6,9,11,14,17),tuple(range(3,19))):
        original=install(skip)
        prefix,cache=model.apply(variables,[x,None],jnp.arange(3)[None,:],jnp.ones((1,3,3),bool))
        assert cache[0].shape==(18,1,3,1,4) and np.isfinite(np.asarray(prefix[0])).all()
        suffix=[None,jnp.ones((1,2,8))*.02]; positions=jnp.array([[3,4]]); mask=jnp.ones((1,2,5),bool)
        actual=model.apply(variables,suffix,positions,mask,kv_cache=cache)
        expected=model.apply(variables,suffix,positions,mask,kv_cache=cache,method=original)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)): np.testing.assert_array_equal(a,b)
        trace=jax.make_jaxpr(lambda t:model.apply(variables,[None,t],positions,mask,kv_cache=cache)[0][1])(suffix[1])
        assert 'scan[' in str(trace)
    print('PASS tiny18: skip0/6/16 KV shells, full original scanned AE exact parity, first2 retained')


def dispatch():
    import subprocess
    import json
    from preparations.idea1_v3 import branches
    from preparations.idea1_pi05_d2_search import cli
    old_worker=branches.worker; old_popen=subprocess.Popen; commands=[]
    branches.worker=lambda args:subprocess.Popen(['python','-m','preparations.idea1_v3.branch_server'])
    subprocess.Popen=lambda command,*a,**kw:commands.append(command)
    try:
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'manifest.json').write_text(json.dumps(dict(revision=cli.REVISION,shards=4,tasks=[])))
            cli.worker(SimpleNamespace(attempt=root,shard=0))
            assert commands[0][2]=='preparations.idea1_pi05_d2_search.server'
            data=cli.manifest(SimpleNamespace(masks='{"FullD2":[],"n6a":[3,6,9,11,14,17]}',shards=4))
            assert len(data['tasks'])==80 and {r['owner'] for r in data['tasks']}=={0,1,2,3}
            assert all(r['config']['denoise_steps']==2 and r['checkpoint']['D']==2 for r in data['tasks'])
    finally: branches.worker=old_worker; subprocess.Popen=old_popen
    print('PASS CPU dispatch/manifest: new actual model server, D2 keys and four static owners; no process/GPU started')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--dispatch-only',action='store_true'); args=p.parse_args(); dispatch()
    if not args.dispatch_only:
        assert os.environ.get('JAX_PLATFORM_NAME')=='cpu'
        scheduling(); tiny()
